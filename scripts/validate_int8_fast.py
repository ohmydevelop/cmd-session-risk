#!/usr/bin/env python3
"""Fast full-prefix FP32-vs-INT8 validation with saturation reuse.

The original ``validate_int8.py`` calls both graphs once per prefix.  For the
million-scale corpus that means ~44M batch-1 inference calls and is not a
practical evidence step.

The exported v4/v5 tokenizer uses ``session_tail_tokens == 0``.  Once a prefix
exceeds ``max_length`` it is encoded as the first ``session_head_tokens`` tokens
plus one ``<TRUNCATED>`` marker, so every longer prefix in that session has the
same input IDs.  This script evaluates each session only through the first
truncated prefix, then reuses that score for all later prefixes while still
counting every prefix exactly once.  The old implementation is retained and can
be used on a small sample for equivalence checks.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for root in (SRC_ROOT, SCRIPTS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from validate_int8 import _commands, _iter_records, _rates  # noqa: E402


def _weighted_percentile(counts: dict[float, int], q: float) -> float:
    if not counts:
        return float("nan")
    if not 0 <= q <= 1:
        raise ValueError("percentile fraction must be between 0 and 1")
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    position = q * (total - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    lower_value = upper_value = None
    cumulative = 0
    for value in sorted(counts):
        next_cumulative = cumulative + counts[value]
        if lower_value is None and lower_index < next_cumulative:
            lower_value = value
        if upper_index < next_cumulative:
            upper_value = value
            break
        cumulative = next_cumulative
    assert lower_value is not None and upper_value is not None
    if lower_index == upper_index:
        return float(lower_value)
    weight = position - lower_index
    return float(lower_value) * (1.0 - weight) + float(upper_value) * weight


def _delta_summary(counts: dict[float, int]) -> dict[str, float | None]:
    total = sum(counts.values())
    if not total:
        return {"max": None, "mean": None, "p95": None, "p99": None}
    return {
        "max": max(counts),
        "mean": sum(value * count for value, count in counts.items()) / total,
        "p95": _weighted_percentile(counts, 0.95),
        "p99": _weighted_percentile(counts, 0.99),
    }


def _first_truncated_length(scorer: Any, commands: list[tuple[str, float | None]]) -> int | None:
    tokenizer = scorer._tokenizer
    for length in range(1, len(commands) + 1):
        encoded = tokenizer.encode([{"cmd": cmd, "dt": dt} for cmd, dt in commands[:length]])
        if encoded.truncated:
            return length
    return None


def _session_prefixes(
    commands: list[tuple[str, float | None]], stop: int
) -> list[list[tuple[str, float | None]]]:
    return [commands[:length] for length in range(1, stop + 1)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--input", required=True, help="Labelled session JSONL")
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0, help="0 = all sessions")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")

    import numpy as np
    import onnxruntime as ort

    from cmdrisk.artifact_manifest import ArtifactManifestError, verify_export_manifest
    from cmdrisk.runtime import ScoringError, SessionScorer

    artifact = Path(args.artifact_dir).expanduser().resolve()
    if not artifact.is_dir():
        raise SystemExit(f"artifact directory does not exist: {artifact}")
    try:
        verify_export_manifest(artifact)
    except ArtifactManifestError as exc:
        raise SystemExit(f"artifact verification failed: {exc}") from exc

    def _session(path: Path) -> Any:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    try:
        fp32 = SessionScorer.from_onnx(
            artifact,
            onnx_session=_session(artifact / "model.onnx"),
            verify_manifest=False,
        )
        int8 = SessionScorer.from_onnx(
            artifact,
            onnx_session=_session(artifact / "model.int8.onnx"),
            verify_manifest=False,
        )
    except ScoringError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"scorer construction failed: {exc}") from exc

    tokenizer = fp32._tokenizer
    if int(getattr(tokenizer, "session_tail_tokens", 0)) != 0:
        raise SystemExit(
            "fast validator requires session_tail_tokens=0; use validate_int8.py "
            "for artifacts with a non-zero tail budget"
        )

    threshold = float(fp32.malicious_threshold)
    records = list(_iter_records(Path(args.input).expanduser().resolve()))
    if args.limit:
        records = records[: args.limit]

    delta_counts: dict[float, int] = {}
    prefix_agree = 0
    prefix_total = 0
    prefixes_inferred = 0
    prefixes_reused = 0
    fp_decisions: list[bool] = []
    int8_decisions: list[bool] = []
    labels: list[int] = []
    index_drift: list[float] = []
    skipped = 0

    for record_index, record in enumerate(records, start=1):
        commands = _commands(record)
        label = record.get("label")
        label_value = label.get("malicious") if isinstance(label, dict) else None
        if not commands or label_value not in (0, 1):
            skipped += 1
            continue

        stop = _first_truncated_length(fp32, commands)
        saturated = stop is not None
        probe_stop = stop if saturated else len(commands)
        prefixes = _session_prefixes(commands, probe_stop)
        fp_first: int | None = None
        int8_first: int | None = None
        fp_peak = 0.0
        int8_peak = 0.0
        last_fp_prob = 0.0
        last_int8_prob = 0.0

        for offset in range(0, len(prefixes), args.batch_size):
            chunk = prefixes[offset : offset + args.batch_size]
            try:
                fp_entries = fp32.score_batch(chunk)
                int8_entries = int8.score_batch(chunk)
            except ScoringError as exc:
                raise SystemExit(f"scoring failed for {record.get('session_id')}: {exc}") from exc
            for local_index, (fp_entry, int8_entry) in enumerate(
                zip(fp_entries, int8_entries, strict=True)
            ):
                length = offset + local_index + 1
                fp_prob = float(fp_entry.malicious_probability)
                int8_prob = float(int8_entry.malicious_probability)
                last_fp_prob = fp_prob
                last_int8_prob = int8_prob
                delta = abs(fp_prob - int8_prob)
                delta_counts[delta] = delta_counts.get(delta, 0) + 1
                prefix_total += 1
                prefixes_inferred += 1
                if (fp_prob >= threshold) == (int8_prob >= threshold):
                    prefix_agree += 1
                fp_peak = max(fp_peak, fp_prob)
                int8_peak = max(int8_peak, int8_prob)
                if fp_first is None and fp_prob >= threshold:
                    fp_first = length
                if int8_first is None and int8_prob >= threshold:
                    int8_first = length

        if saturated:
            repeats = len(commands) - probe_stop
            if repeats:
                delta = abs(last_fp_prob - last_int8_prob)
                delta_counts[delta] = delta_counts.get(delta, 0) + repeats
                prefix_total += repeats
                prefixes_reused += repeats
                if (last_fp_prob >= threshold) == (last_int8_prob >= threshold):
                    prefix_agree += repeats
                if fp_first is None and last_fp_prob >= threshold:
                    fp_first = probe_stop
                if int8_first is None and last_int8_prob >= threshold:
                    int8_first = probe_stop

        fp_alert = fp_peak >= threshold
        int8_alert = int8_peak >= threshold
        fp_decisions.append(fp_alert)
        int8_decisions.append(int8_alert)
        labels.append(int(label_value))
        if fp_first is not None and int8_first is not None:
            index_drift.append(abs(fp_first - int8_first))
        elif fp_first is not None or int8_first is not None:
            index_drift.append(float(len(commands)))

        if args.progress_every and record_index % args.progress_every == 0:
            print(
                f"validated {record_index}/{len(records)} sessions, "
                f"prefixes={prefix_total:,}, inferred={prefixes_inferred:,}, "
                f"reused={prefixes_reused:,}",
                file=sys.stderr,
                flush=True,
            )

    if not labels:
        raise SystemExit("no labelled sessions were scored")

    fp_rates = _rates(fp_decisions, labels)
    int8_rates = _rates(int8_decisions, labels)
    session_agree = sum(1 for a, b in zip(fp_decisions, int8_decisions, strict=True) if a == b)

    def _delta(key: str) -> float | None:
        left = int8_rates.get(key)
        right = fp_rates.get(key)
        if left is None or right is None:
            return None
        return float(left) - float(right)

    report = {
        "artifact_dir": str(artifact),
        "quantized_graph": "model.int8.onnx",
        "quantized_graph_bytes": (artifact / "model.int8.onnx").stat().st_size,
        "input": str(Path(args.input).expanduser().resolve()),
        "threshold": threshold,
        "sessions_scored": len(labels),
        "sessions_skipped": skipped,
        "prefixes_scored": prefix_total,
        "prefixes_inferred": prefixes_inferred,
        "prefixes_reused": prefixes_reused,
        "validation_method": "full_prefix_saturation_reuse",
        "saturation_config": {
            "max_length": int(fp32.max_length),
            "session_head_tokens": int(tokenizer.session_head_tokens),
            "session_tail_tokens": int(tokenizer.session_tail_tokens),
        },
        "prefix_probability_delta": _delta_summary(delta_counts),
        "prefix_decision_agreement": (prefix_agree / prefix_total if prefix_total else None),
        "session_decision_agreement": session_agree / len(labels),
        "fp32": fp_rates,
        "int8": int8_rates,
        "delta_recall_int8_minus_fp32": _delta("recall"),
        "delta_fpr_int8_minus_fp32": _delta("fpr"),
        "first_alert_index_drift": {
            "max": max(index_drift) if index_drift else 0.0,
            "mean": float(np.mean(index_drift)) if index_drift else 0.0,
            "sessions_differing": sum(1 for value in index_drift if value > 0),
        },
    }

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else artifact / "int8_validation.json"
    )
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nwrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

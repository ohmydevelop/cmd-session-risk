#!/usr/bin/env python3
"""Measure FP32-vs-INT8 agreement for a cmd-session-risk ONNX artifact.

Dynamic INT8 quantization changes logits by far more than the FP32 parity
tolerance, so a raw logit ``atol`` is not a meaningful acceptance test for the
quantized graph.  What actually matters in deployment is whether the quantized
model makes the *same decisions* as the FP32 model on real, labelled sessions.

This script therefore reports, over a labelled JSONL slice:

* prefix-level probability deltas (max / mean / p95 / p99),
* prefix-level decision-agreement rate at the calibrated threshold,
* session-level recall / FPR for FP32 and INT8 plus their deltas,
* session-level decision-agreement and first-alert-index drift.

The output is written next to the artifact as ``int8_validation.json``.  It is
deliberately *evidence*, not a gate: nothing here flips ``promotion.passed``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if isinstance(record, dict):
                yield record


def _commands(record: dict[str, Any]) -> list[tuple[str, float | None]]:
    out: list[tuple[str, float | None]] = []
    for item in record.get("commands", ()):
        if isinstance(item, str):
            out.append((item, None))
        elif isinstance(item, dict):
            cmd = item.get("cmd")
            dt = item.get("dt")
            if isinstance(cmd, str) and cmd.strip():
                out.append((cmd, None if dt is None else float(dt)))
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _rates(decisions: list[bool], labels: list[int]) -> dict[str, Any]:
    tp = sum(1 for d, y in zip(decisions, labels, strict=True) if y == 1 and d)
    fn = sum(1 for d, y in zip(decisions, labels, strict=True) if y == 1 and not d)
    fp = sum(1 for d, y in zip(decisions, labels, strict=True) if y == 0 and d)
    tn = sum(1 for d, y in zip(decisions, labels, strict=True) if y == 0 and not d)
    recall = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": recall,
        "fpr": fpr,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--input", required=True, help="Labelled session JSONL")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 = all sessions")
    parser.add_argument(
        "--int8-filename", default="model.int8.onnx", help="Quantized graph filename"
    )
    parser.add_argument(
        "--fp32-filename", default="model.onnx", help="Full-precision graph filename"
    )
    args = parser.parse_args(argv)

    import numpy as np
    import onnxruntime as ort

    from cmdrisk.artifact_manifest import ArtifactManifestError, verify_export_manifest
    from cmdrisk.runtime import ScoringError, SessionScorer

    artifact = Path(args.artifact_dir).expanduser().resolve()
    int8_path = artifact / args.int8_filename
    if not artifact.is_dir():
        raise SystemExit(f"artifact directory does not exist: {artifact}")
    if not int8_path.is_file():
        raise SystemExit(f"quantized graph not found: {int8_path}")

    # ``SessionScorer.from_onnx`` itself prefers ``model.int8.onnx`` whenever it
    # exists, so both scorers must be pinned to an explicit ORT session: a bare
    # load would silently compare the quantized graph against itself.  Verify
    # the manifest directly instead of building a throwaway session.
    try:
        verify_export_manifest(artifact)
    except ArtifactManifestError as exc:
        raise SystemExit(f"artifact verification failed: {exc}") from exc

    fp32_path = artifact / args.fp32_filename
    if not fp32_path.is_file():
        raise SystemExit(f"fp32 graph not found: {fp32_path}")

    def _session(path: Path) -> Any:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )

    try:
        fp32 = SessionScorer.from_onnx(
            artifact, onnx_session=_session(fp32_path), verify_manifest=False
        )
        int8 = SessionScorer.from_onnx(
            artifact, onnx_session=_session(int8_path), verify_manifest=False
        )
    except ScoringError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"scorer construction failed: {exc}") from exc

    threshold = float(fp32.malicious_threshold)
    records = list(_iter_records(Path(args.input).expanduser().resolve()))
    if args.limit:
        records = records[: args.limit]

    prob_deltas: list[float] = []
    prefix_agree = 0
    prefix_total = 0
    fp_decisions: list[bool] = []
    int8_decisions: list[bool] = []
    labels: list[int] = []
    index_drift: list[float] = []
    skipped = 0

    for record in records:
        commands = _commands(record)
        label = record.get("label")
        label_value = label.get("malicious") if isinstance(label, dict) else None
        if not commands or label_value not in (0, 1):
            skipped += 1
            continue

        fp_first = None
        int8_first = None
        fp_peak = 0.0
        int8_peak = 0.0
        for length in range(1, len(commands) + 1):
            prefix = commands[:length]
            try:
                fp_entry = fp32.score_commands(prefix)
                int8_entry = int8.score_commands(prefix)
            except ScoringError as exc:
                raise SystemExit(f"scoring failed for {record.get('session_id')}: {exc}") from exc
            fp_prob = float(fp_entry.malicious_probability)
            int8_prob = float(int8_entry.malicious_probability)
            prob_deltas.append(abs(fp_prob - int8_prob))
            prefix_total += 1
            if (fp_prob >= threshold) == (int8_prob >= threshold):
                prefix_agree += 1
            fp_peak = max(fp_peak, fp_prob)
            int8_peak = max(int8_peak, int8_prob)
            if fp_first is None and fp_prob >= threshold:
                fp_first = length
            if int8_first is None and int8_prob >= threshold:
                int8_first = length

        fp_alert = fp_peak >= threshold
        int8_alert = int8_peak >= threshold
        fp_decisions.append(fp_alert)
        int8_decisions.append(int8_alert)
        labels.append(int(label_value))
        if fp_first is not None and int8_first is not None:
            index_drift.append(abs(fp_first - int8_first))
        elif fp_first is not None or int8_first is not None:
            index_drift.append(float(len(commands)))

    if not labels:
        raise SystemExit("no labelled sessions were scored")

    fp_rates = _rates(fp_decisions, labels)
    int8_rates = _rates(int8_decisions, labels)
    session_agree = sum(
        1 for a, b in zip(fp_decisions, int8_decisions, strict=True) if a == b
    )

    def _delta(key: str) -> float | None:
        left = int8_rates.get(key)
        right = fp_rates.get(key)
        if left is None or right is None:
            return None
        return float(left) - float(right)

    report = {
        "artifact_dir": str(artifact),
        "quantized_graph": str(int8_path.name),
        "quantized_graph_bytes": int8_path.stat().st_size,
        "input": str(Path(args.input).expanduser().resolve()),
        "threshold": threshold,
        "sessions_scored": len(labels),
        "sessions_skipped": skipped,
        "prefixes_scored": prefix_total,
        "prefix_probability_delta": {
            "max": max(prob_deltas) if prob_deltas else None,
            "mean": float(np.mean(prob_deltas)) if prob_deltas else None,
            "p95": _percentile(prob_deltas, 0.95),
            "p99": _percentile(prob_deltas, 0.99),
        },
        "prefix_decision_agreement": (
            prefix_agree / prefix_total if prefix_total else None
        ),
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

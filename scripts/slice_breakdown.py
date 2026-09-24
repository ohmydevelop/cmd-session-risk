#!/usr/bin/env python3
"""Per-slice operating-point breakdown of already-scored ONNX prefixes.

Motivation
----------
The v4 corpus labels every session with exactly one ``evaluation_slice``:

    real_only             <- real_malicious, real_benign
    real_plus_semi_real   <- script_benign, real_history_pool
    synthetic_stress      <- llm_variant, hard_negative
    ood_2024              <- explicit 2024 temporal holdout

Because the mapping is single-label, the slice literally named
``real_plus_semi_real`` receives *only* the two benign semi-real sources and
therefore carries zero positives.  The training pipeline correctly reports that
as ``insufficient_positives`` -- but it is a labelling artefact, not a data
gap: the real tier does have positives, and the union the slice name implies
does too.

This script quantifies both readings on the *same* scored prefixes so the
difference is visible:

  * ``recorded:<slice>``   -- the single-label mapping exactly as recorded
  * ``corrected:real_plus_semi_real`` -- real tier UNION semi-real tier

It is a read-only diagnostic.  It never fits on the test split, never rewrites
data, and never changes any promotion gate; ``promotion.passed`` stays false.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from cmdrisk.training.data import (  # noqa: E402
    _evaluation_slice,
    load_records,
    select_prefix_indices,
)
from cmdrisk.training.metrics import binary_classification_metrics  # noqa: E402

REAL_KINDS = {"real_malicious", "real_benign"}
SEMI_REAL_KINDS = {"script_benign", "real_history_pool"}
STRESS_KINDS = {"llm_variant", "hard_negative"}

RECORDED_ORDER = ("real_only", "real_plus_semi_real", "synthetic_stress", "ood_2024")


def _sigmoid(value: float) -> float:
    clipped = min(max(value, -80.0), 80.0)
    return 1.0 / (1.0 + math.exp(-clipped))


def _transform(probability: float, source_temperature: float, target_temperature: float) -> float:
    p = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    raw_logit = math.log(p / (1.0 - p)) * source_temperature
    return _sigmoid(raw_logit / target_temperature)


def _load_selected_scores(
    scores_path: Path,
    data_path: Path,
    source_temperature: float,
    target_temperature: float,
) -> tuple[dict[str, float], dict[str, int], dict[str, str]]:
    """Return the family-aware cap-32 session score, label and kind per session."""
    selected: dict[str, set[int]] = {}
    for record in load_records(str(data_path)):
        commands = record.get("commands", ())
        selected[str(record["session_id"])] = set(
            select_prefix_indices(len(commands), max_prefixes=32)
        )

    labels: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for record in load_records(str(data_path)):
        session_id = str(record["session_id"])
        labels[session_id] = int(record.get("label", {}).get("malicious", 0))
        kinds[session_id] = str(record.get("source", {}).get("kind") or "unknown")

    session_scores: dict[str, float] = {}
    with scores_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            session_id = str(payload["session_id"])
            wanted = selected.get(session_id)
            if wanted is None:
                raise SystemExit(f"{scores_path}: session {session_id} is not in {data_path}")
            if int(payload["command_index"]) not in wanted:
                continue
            probability = _transform(
                float(payload["malicious_probability"]), source_temperature, target_temperature
            )
            if probability > session_scores.get(session_id, float("-inf")):
                session_scores[session_id] = probability

    missing = sorted(set(labels) - set(session_scores))
    if missing:
        raise SystemExit(
            f"{scores_path}: no selected prefix scored for {len(missing)} session(s), "
            f"first={missing[:3]}"
        )
    return session_scores, labels, kinds


def _metrics(
    session_ids: list[str], scores: dict[str, float], labels: dict[str, int], threshold: float
) -> dict:
    targets = np.asarray([labels[sid] for sid in session_ids], dtype=np.int64)
    probabilities = np.asarray([scores[sid] for sid in session_ids], dtype=np.float64)
    return binary_classification_metrics(targets, probabilities, threshold=threshold)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Labelled session JSONL")
    parser.add_argument(
        "--scores", type=Path, required=True, help="Flat ONNX score JSONL for --data"
    )
    parser.add_argument("--source-temperature", type=float, required=True)
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    scores, labels, kinds = _load_selected_scores(
        args.scores, args.data, args.source_temperature, args.target_temperature
    )

    # Re-use the *exact* derivation the training evaluator uses, so the
    # "recorded" groups reproduce the frozen metrics.json slice counts.
    recorded: dict[str, list[str]] = defaultdict(list)
    for record in load_records(str(args.data)):
        session_id = str(record["session_id"])
        recorded[_evaluation_slice(record, kinds.get(session_id, "unknown"))].append(session_id)

    groups: dict[str, list[str]] = {}
    for name, session_ids in recorded.items():
        groups[f"recorded:{name}"] = session_ids

    real_union_semi = [
        session_id
        for session_id, kind in kinds.items()
        if kind in REAL_KINDS or kind in SEMI_REAL_KINDS
    ]
    groups["corrected:real_plus_semi_real"] = sorted(real_union_semi)
    groups["tier:real"] = sorted(sid for sid, kind in kinds.items() if kind in REAL_KINDS)
    groups["tier:semi_real"] = sorted(sid for sid, kind in kinds.items() if kind in SEMI_REAL_KINDS)

    report = {
        "schema": "cmdrisk.slice_breakdown",
        "version": 1,
        "split": args.split_name,
        "data": str(args.data),
        "scores": str(args.scores),
        "methodology": {
            "session_decision": "max probability over the family-aware cap-32 prefixes >= threshold",  # noqa: E501
            "threshold": args.threshold,
            "source_temperature": args.source_temperature,
            "target_temperature": args.target_temperature,
            "test_used_for_fitting": False,
            "note": (
                "recorded:* uses the single-label evaluation_slice field; "
                "corrected:real_plus_semi_real unions the real and semi-real tiers, "
                "which is what the slice name implies."
            ),
        },
        "sessions_scored": len(scores),
        "groups": {},
    }

    for name, session_ids in sorted(groups.items()):
        if not session_ids:
            continue
        metrics = _metrics(session_ids, scores, labels, args.threshold)
        entry = {
            "sessions": len(session_ids),
            "positives": int(sum(labels[sid] for sid in session_ids)),
            "negatives": int(sum(1 - labels[sid] for sid in session_ids)),
            **metrics,
        }
        report["groups"][name] = entry
        print(
            f"{name:34s} n={entry['sessions']:6d} pos={entry['positives']:6d} neg={entry['negatives']:6d} "  # noqa: E501
            f"recall={metrics['recall']} fpr={metrics['fpr']}",
            file=sys.stderr,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

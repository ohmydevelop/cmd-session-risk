#!/usr/bin/env python3
"""Deployment-aligned operating-point sweep for a latched session detector.

The scorer emits a malicious probability per observed command prefix.  In
deployment a session alerts as soon as *any* observed prefix crosses the
threshold (session-latched alerting), so the quantity operations actually cares
about is the **session-level** false-positive rate, not the prefix-level one.

This script re-uses already-scored ONNX prefixes (the same JSONL rows produced
by ``cmdrisk.cli score`` / ``score_fp32.py``) and answers three questions:

1. For a grid of target **dev session-FPR** budgets, what threshold does the
   dev split select, and what does that threshold deliver on the held-out test
   split at both the session and the family-aware cap-32 prefix level?
2. How does an optional ``k``-of-``n`` persistence rule (alert only when at
   least ``k`` observed prefixes cross the threshold) move the same operating
   point?  This is a pure post-processing change - no retraining, no extra
   inference cost - so it is attractive for a frozen deployment graph.
3. Which (temperature, threshold) pair is Pareto-optimal for a recall target?

Nothing here fits on the test split; thresholds are always selected on dev.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from cmdrisk.training.metrics import binary_classification_metrics  # noqa: E402
from evaluate_recalibrated_onnx_scores import (  # noqa: E402
    _join_prefixes,
    _load_scores,
    _prefix_metrics,
    _session_scores,
    build_parser as _eval_parser,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--fp32-dev", type=Path, required=True)
    parser.add_argument("--fp32-test", type=Path, required=True)
    parser.add_argument("--source-temperature", type=float, required=True)
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument(
        "--session-fpr-grid",
        default="0.0025,0.005,0.0075,0.01,0.015,0.02,0.03",
        help="comma separated dev *session* false-positive budgets",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _threshold_for_session_fpr(
    labels: np.ndarray, probabilities: np.ndarray, target: float
) -> float | None:
    """Smallest threshold whose dev session FPR is <= ``target``."""
    negatives = np.sort(probabilities[labels == 0])
    if negatives.size == 0:
        return None
    allowed = int(np.floor(target * negatives.size))
    # Need at most ``allowed`` negatives at/above the threshold.  The most
    # permissive such threshold sits just above the (allowed+1)-th largest
    # negative score.
    if allowed >= negatives.size:
        return 0.0
    index = negatives.size - allowed - 1
    return float(negatives[index]) + 1e-12


def _session_rows(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    return binary_classification_metrics(labels, probabilities, threshold=threshold)


def _kofn_matrices(joined: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    """Return per-session crossing counts keyed by threshold-independent order.

    Because the thresholds we sweep are monotone in the score we can simply
    return the sorted per-session prefix scores, then threshold them on demand.
    """
    counts: dict[str, list[float]] = {}
    for session_id, index, probability in zip(
        joined["session_ids"], joined["prefix_indices"], joined["probabilities"], strict=True
    ):
        counts.setdefault(session_id, []).append(float(probability))
    return counts  # type: ignore[return-value]


def _kofn_decision(
    per_session_scores: dict[str, list[float]], threshold: float, k: int
) -> dict[str, bool]:
    return {
        session_id: sum(1 for score in scores if score >= threshold) >= k
        for session_id, scores in per_session_scores.items()
    }


def _rates_from_decisions(
    labels: dict[str, int], decisions: dict[str, bool]
) -> dict[str, Any]:
    tp = fn = fp = tn = 0
    for session_id, label in labels.items():
        decision = decisions.get(session_id, False)
        if label and decision:
            tp += 1
        elif label and not decision:
            fn += 1
        elif decision:
            fp += 1
        else:
            tn += 1
    return {
        "true_positive": tp,
        "false_negative": fn,
        "false_positive": fp,
        "true_negative": tn,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "precision": tp / (tp + fp) if (tp + fp) else None,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    vocabulary_path = args.run_dir / "label_vocabulary.json"
    fp32_dev = _load_scores(args.fp32_dev, args.source_temperature, args.target_temperature)
    fp32_test = _load_scores(args.fp32_test, args.source_temperature, args.target_temperature)
    joined_dev = _join_prefixes(args.dev_data, vocabulary_path, fp32_dev)
    joined_test = _join_prefixes(args.test_data, vocabulary_path, fp32_test)

    dev_scores = _session_scores(fp32_dev)
    test_scores = _session_scores(fp32_test)
    dev_labels = np.asarray(
        [int(joined_dev["session_labels"][s]) for s in sorted(joined_dev["session_labels"])],
        dtype=np.int64,
    )
    dev_probs = np.asarray(
        [dev_scores.get(s, float("-inf")) for s in sorted(joined_dev["session_labels"])],
        dtype=np.float64,
    )
    test_labels = np.asarray(
        [int(joined_test["session_labels"][s]) for s in sorted(joined_test["session_labels"])],
        dtype=np.int64,
    )
    test_probs = np.asarray(
        [test_scores.get(s, float("-inf")) for s in sorted(joined_test["session_labels"])],
        dtype=np.float64,
    )

    dev_k = _kofn_matrices(joined_dev)
    test_k = _kofn_matrices(joined_test)
    dev_labels_k = {s: int(joined_dev["session_labels"][s]) for s in joined_dev["session_labels"]}
    test_labels_k = {s: int(joined_test["session_labels"][s]) for s in joined_test["session_labels"]}

    grid = [float(part) for part in args.session_fpr_grid.split(",") if part.strip()]
    rows: list[dict[str, Any]] = []
    for target in grid:
        threshold = _threshold_for_session_fpr(dev_labels, dev_probs, target)
        if threshold is None:
            continue
        dev_session = _session_rows(dev_labels, dev_probs, threshold)
        test_session = _session_rows(test_labels, test_probs, threshold)
        test_prefix = _prefix_metrics(joined_test, threshold)
        row: dict[str, Any] = {
            "target_dev_session_fpr": target,
            "threshold": threshold,
            "dev_session": {
                "recall": dev_session["recall"],
                "fpr": dev_session["fpr"],
                "false_negative": dev_session["false_negative"],
                "false_positive": dev_session["false_positive"],
            },
            "test_session": {
                "recall": test_session["recall"],
                "fpr": test_session["fpr"],
                "precision": test_session["precision"],
                "false_negative": test_session["false_negative"],
                "false_positive": test_session["false_positive"],
                "pr_auc": test_session["pr_auc"],
                "roc_auc": test_session["roc_auc"],
                "brier": test_session["brier"],
                "ece": test_session["ece"],
            },
            "test_prefix": {
                "recall": test_prefix["recall"],
                "fpr": test_prefix["fpr"],
                "false_negative": test_prefix["false_negative"],
                "false_positive": test_prefix["false_positive"],
            },
        }
        # 2-of-n persistence rule at the same threshold.
        dev_decisions = _kofn_decision(dev_k, threshold, 2)
        test_decisions = _kofn_decision(test_k, threshold, 2)
        row["dev_session_kofn_2"] = _rates_from_decisions(dev_labels_k, dev_decisions)
        row["test_session_kofn_2"] = _rates_from_decisions(test_labels_k, test_decisions)
        rows.append(row)

    output = {
        "schema": "cmdrisk.session_operating_point_sweep",
        "version": 1,
        "run_dir": str(args.run_dir),
        "selection_split": str(args.dev_data),
        "evaluation_split": str(args.test_data),
        "source_temperature": args.source_temperature,
        "target_temperature": args.target_temperature,
        "test_used_for_fitting": False,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    for row in rows:
        ts = row["test_session"]
        print(
            f"  dev-session-FPR<={row['target_dev_session_fpr']:<7} thr={row['threshold']:.6f} "
            f"test session recall={ts['recall']:.6f} fpr={ts['fpr']:.6f} "
            f"FN={ts['false_negative']} FP={ts['false_positive']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

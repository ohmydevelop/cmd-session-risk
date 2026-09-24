#!/usr/bin/env python3
"""Evaluate already-scored ONNX prefixes after a temperature/threshold change.

The scorer records malicious probabilities as ``sigmoid(raw_logit / T_source)``.
Changing only the runtime temperature to ``T_target`` is therefore reversible
without another expensive ONNX pass:

    conservative_logit = logit(p_source) * T_source
    p_target = sigmoid(conservative_logit / T_target)

This script joins those transformed scores with the exact family-aware,
cap-32 prefix selection used by the training evaluator.  It is intentionally
independent of model training and never fits on the test split.

It reports both prefix and session operating points because deployment alerting
is session-latched: a session alerts when any observed prefix crosses the
threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from cmdrisk.training.data import load_records, select_prefix_indices  # noqa: E402
from cmdrisk.training.labels import LabelVocabulary  # noqa: E402
from cmdrisk.training.metrics import (  # noqa: E402
    binary_classification_metrics,
    detection_delay_metrics,
)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _transform(probability: float, source_temperature: float, target_temperature: float) -> float:
    p = min(max(float(probability), 1e-15), 1.0 - 1e-15)
    raw_logit = math.log(p / (1.0 - p)) * source_temperature
    return float(_sigmoid(np.asarray([raw_logit / target_temperature]))[0])


def _load_scores(
    path: Path, source_temperature: float, target_temperature: float
) -> dict[str, dict[int, float]]:
    scores: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                session_id = str(payload["session_id"])
                command_index = int(payload["command_index"])
                probability = float(payload["malicious_probability"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(f"{path}:{line_number}: malformed score row: {exc}") from exc
            scores[session_id][command_index] = _transform(
                probability, source_temperature, target_temperature
            )
    return dict(scores)


def _session_scores(scores: dict[str, dict[int, float]]) -> dict[str, float]:
    return {session_id: max(values.values()) for session_id, values in scores.items() if values}


def _session_rates(
    labels: dict[str, int],
    scores: dict[str, float],
    *,
    threshold: float,
) -> dict[str, Any]:
    items = [(label, scores.get(session_id, float("-inf"))) for session_id, label in labels.items()]
    targets = np.asarray([label for label, _ in items], dtype=np.int64)
    probabilities = np.asarray([score for _, score in items], dtype=np.float64)
    return binary_classification_metrics(targets, probabilities, threshold=threshold)


def _join_prefixes(
    data_path: Path,
    vocabulary_path: Path,
    scores: dict[str, dict[int, float]],
) -> dict[str, Any]:
    vocabulary = LabelVocabulary.load(vocabulary_path)
    session_labels: dict[str, int] = {}
    targets: list[float] = []
    probabilities: list[float] = []
    session_ids: list[str] = []
    prefix_indices: list[int] = []
    missing: list[str] = []

    for record in load_records(str(data_path)):
        session_id = str(record["session_id"])
        label = record.get("label", {})
        session_label = int(label.get("malicious", 0))
        session_labels[session_id] = session_label
        commands = record.get("commands", ())
        first_evidence = label.get("first_evidence_index")
        technique_evidence = label.get("technique_evidence", {}) or {}
        evidence_indices = [
            int(index)
            for values in technique_evidence.values()
            for index in ([values] if isinstance(values, (int, float)) else values)
        ]
        selected = select_prefix_indices(
            len(commands),
            max_prefixes=32,
            first_evidence_index=None if first_evidence is None else int(first_evidence),
            evidence_indices=evidence_indices,
        )
        for index in selected:
            probability = scores.get(session_id, {}).get(index)
            if probability is None:
                missing.append(f"{session_id}:{index}")
                continue
            target = vocabulary.encode(label, index).malicious
            targets.append(float(target))
            probabilities.append(float(probability))
            session_ids.append(session_id)
            prefix_indices.append(index)

    if missing:
        raise SystemExit(
            f"{len(missing)} selected prefixes have no score; first missing: {missing[:3]}"
        )
    return {
        "targets": np.asarray(targets, dtype=np.float64),
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "session_ids": session_ids,
        "prefix_indices": prefix_indices,
        "session_labels": session_labels,
    }


def _prefix_metrics(joined: dict[str, Any], threshold: float) -> dict[str, Any]:
    metrics = binary_classification_metrics(
        joined["targets"], joined["probabilities"], threshold=threshold
    )
    metrics["threshold"] = threshold
    metrics["detection_delay"] = detection_delay_metrics(
        joined["targets"],
        joined["probabilities"],
        session_ids=joined["session_ids"],
        prefix_indices=joined["prefix_indices"],
        threshold=threshold,
    )
    return metrics


def _int8_comparison(
    joined_fp32: dict[str, Any],
    joined_int8: dict[str, Any],
    fp32_all: dict[str, dict[int, float]],
    int8_all: dict[str, dict[int, float]],
    *,
    threshold: float,
) -> dict[str, Any]:
    _fp_targets = np.asarray(joined_fp32["targets"])
    fp_probs = np.asarray(joined_fp32["probabilities"])
    int8_probs = np.asarray(joined_int8["probabilities"])
    fp_decisions = fp_probs >= threshold
    int8_decisions = int8_probs >= threshold
    deltas = np.abs(fp_probs - int8_probs)

    fp32_sessions = _session_scores(fp32_all)
    int8_sessions = _session_scores(int8_all)
    session_ids = sorted(set(fp32_sessions) & set(int8_sessions))
    fp_session_decisions = [fp32_sessions[session_id] >= threshold for session_id in session_ids]
    int8_session_decisions = [int8_sessions[session_id] >= threshold for session_id in session_ids]
    labels = [int(joined_fp32["session_labels"][session_id]) for session_id in session_ids]

    def _rates(decisions: list[bool]) -> dict[str, Any]:
        tp = sum(1 for decision, label in zip(decisions, labels, strict=True) if decision and label)
        fn = sum(
            1 for decision, label in zip(decisions, labels, strict=True) if not decision and label
        )
        fp = sum(
            1 for decision, label in zip(decisions, labels, strict=True) if decision and not label
        )
        tn = sum(
            1
            for decision, label in zip(decisions, labels, strict=True)
            if not decision and not label
        )
        return {
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "recall": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
        }

    fp_rates = _rates(fp_session_decisions)
    int8_rates = _rates(int8_session_decisions)

    def _delta(key: str) -> float | None:
        if fp_rates[key] is None or int8_rates[key] is None:
            return None
        return float(int8_rates[key] - fp_rates[key])

    drift: list[float] = []
    for session_id in session_ids:
        fp_commands = fp32_all.get(session_id, {})
        int8_commands = int8_all.get(session_id, {})
        fp_first = next(
            (index for index in sorted(fp_commands) if fp_commands[index] >= threshold), None
        )
        int8_first = next(
            (index for index in sorted(int8_commands) if int8_commands[index] >= threshold), None
        )
        if fp_first is None and int8_first is None:
            drift.append(0.0)
        elif fp_first is not None and int8_first is not None:
            drift.append(float(abs(fp_first - int8_first)))
        else:
            max_index = max([*fp_commands.keys(), *int8_commands.keys(), 0])
            drift.append(float(max_index + 1))

    return {
        "threshold": threshold,
        "sessions_scored": len(labels),
        "prefixes_scored": int(len(fp_probs)),
        "prefix_probability_delta": {
            "max": float(np.max(deltas)) if deltas.size else None,
            "mean": float(np.mean(deltas)) if deltas.size else None,
            "p95": float(np.percentile(deltas, 95)) if deltas.size else None,
            "p99": float(np.percentile(deltas, 99)) if deltas.size else None,
        },
        "prefix_decision_agreement": float(np.mean(fp_decisions == int8_decisions)),
        "session_decision_agreement": sum(
            left == right
            for left, right in zip(fp_session_decisions, int8_session_decisions, strict=True)
        )
        / len(session_ids),
        "fp32": fp_rates,
        "int8": int8_rates,
        "delta_recall_int8_minus_fp32": _delta("recall"),
        "delta_fpr_int8_minus_fp32": _delta("fpr"),
        "first_alert_index_drift": {
            "max": max(drift) if drift else 0.0,
            "mean": float(np.mean(drift)) if drift else 0.0,
            "sessions_differing": sum(value > 0 for value in drift),
        },
    }


def _summary(metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    delay = metrics.get("detection_delay", {})
    return {
        "threshold": threshold,
        "recall": metrics["recall"],
        "fpr": metrics["fpr"],
        "precision": metrics["precision"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "true_positive": metrics["true_positive"],
        "false_negative": metrics["false_negative"],
        "false_positive": metrics["false_positive"],
        "true_negative": metrics["true_negative"],
        "positive_count": metrics["positive_count"],
        "negative_count": metrics["negative_count"],
        "detection_rate": delay.get("detection_rate"),
        "missed_sessions": delay.get("missed_sessions"),
        "false_positive_sessions": delay.get("false_positive_sessions"),
        "median_detection_delay_commands": delay.get("median_detection_delay_commands"),
    }


def _session_summary(metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    return {
        "threshold": threshold,
        "recall": metrics["recall"],
        "fpr": metrics["fpr"],
        "precision": metrics["precision"],
        "pr_auc": metrics["pr_auc"],
        "roc_auc": metrics["roc_auc"],
        "brier": metrics["brier"],
        "ece": metrics["ece"],
        "true_positive": metrics["true_positive"],
        "false_negative": metrics["false_negative"],
        "false_positive": metrics["false_positive"],
        "true_negative": metrics["true_negative"],
        "positive_count": metrics["positive_count"],
        "negative_count": metrics["negative_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dev-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--fp32-dev", type=Path, required=True)
    parser.add_argument("--fp32-test", type=Path, required=True)
    parser.add_argument("--int8-dev", type=Path, required=True)
    parser.add_argument("--int8-test", type=Path, required=True)
    parser.add_argument("--source-temperature", type=float, required=True)
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vocabulary_path = args.run_dir / "label_vocabulary.json"
    if not vocabulary_path.is_file():
        raise SystemExit(f"label vocabulary not found: {vocabulary_path}")

    fp32_dev = _load_scores(args.fp32_dev, args.source_temperature, args.target_temperature)
    fp32_test = _load_scores(args.fp32_test, args.source_temperature, args.target_temperature)
    _int8_dev = _load_scores(args.int8_dev, args.source_temperature, args.target_temperature)
    int8_test = _load_scores(args.int8_test, args.source_temperature, args.target_temperature)

    joined_dev = _join_prefixes(args.dev_data, vocabulary_path, fp32_dev)
    joined_test = _join_prefixes(args.test_data, vocabulary_path, fp32_test)
    joined_int8_test = _join_prefixes(args.test_data, vocabulary_path, int8_test)

    dev_prefix = _prefix_metrics(joined_dev, args.threshold)
    test_prefix = _prefix_metrics(joined_test, args.threshold)

    dev_sessions = _session_rates(
        joined_dev["session_labels"], _session_scores(fp32_dev), threshold=args.threshold
    )
    test_sessions = _session_rates(
        joined_test["session_labels"], _session_scores(fp32_test), threshold=args.threshold
    )
    int8_test_sessions = _session_rates(
        joined_test["session_labels"], _session_scores(int8_test), threshold=args.threshold
    )

    int8_rates_prefix = binary_classification_metrics(
        joined_int8_test["targets"], joined_int8_test["probabilities"], threshold=args.threshold
    )

    output = {
        "schema": "cmdrisk.onnx.recalibrated_evaluation",
        "version": 1,
        "methodology": {
            "selection_split": str(args.dev_data),
            "evaluation_split": str(args.test_data),
            "prefix_selection": "family-aware cap32 used by training evaluator",
            "session_decision": "any selected prefix >= threshold",
            "source_temperature": args.source_temperature,
            "target_temperature": args.target_temperature,
            "threshold": args.threshold,
            "test_used_for_fitting": False,
        },
        "dev": {
            "prefix": _summary(dev_prefix, args.threshold),
            "session": _session_summary(dev_sessions, args.threshold),
        },
        "test": {
            "prefix": _summary(test_prefix, args.threshold),
            "session": _session_summary(test_sessions, args.threshold),
            "prefix_full": test_prefix,
            "session_full": test_sessions,
        },
        "int8_test": {
            "prefix": _summary(int8_rates_prefix, args.threshold),
            "session": _session_summary(int8_test_sessions, args.threshold),
        },
        "int8_comparison": _int8_comparison(
            joined_test,
            joined_int8_test,
            fp32_test,
            int8_test,
            threshold=args.threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"dev": output["dev"], "test": output["test"], "int8": output["int8_comparison"]},
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

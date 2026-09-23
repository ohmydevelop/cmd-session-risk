#!/usr/bin/env python3
"""Evaluate cmdrisk ONNX artifacts on the frozen common_eval_v2 suite.

The evaluator consumes the *full-prefix* flat JSONL emitted by
``cmdrisk score --flat`` and can derive both the full-prefix view and the
training-aligned ``select_prefix_indices(max_prefixes=32)`` view.

All thresholds come from the artifact's own ``calibration.json``.  The public
clean set is never used to tune thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.training.data import select_prefix_indices  # noqa: E402
from cmdrisk.training.labels import LabelVocabulary  # noqa: E402
from cmdrisk.training.metrics import (  # noqa: E402
    binary_classification_metrics,
    detection_delay_metrics,
    multilabel_metrics,
    wilson_interval,
)

SCHEMA = "cmdrisk.common_eval_v2"
VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _evidence_indices(label: Mapping[str, Any]) -> list[int]:
    out: list[int] = []
    for values in _mapping(label.get("technique_evidence", {})).values():
        if isinstance(values, (int, float)) and not isinstance(values, bool):
            out.append(int(values))
        elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            out.extend(int(value) for value in values)
    return out


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        session_id = record.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError(f"record without session_id in {path}")
        if session_id in records:
            raise ValueError(f"duplicate session_id {session_id!r} in {path}")
        commands = record.get("commands")
        if not isinstance(commands, list) or not commands:
            raise ValueError(f"session {session_id!r} has no commands")
        records[session_id] = record
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def load_scores(paths: Sequence[Path]) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for path in paths:
        for record in _iter_jsonl(path):
            session_id = record.get("session_id")
            command_index = record.get("command_index")
            probability = record.get("malicious_probability")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"score without session_id in {path}")
            if not isinstance(command_index, int) or isinstance(command_index, bool):
                raise ValueError(f"score without integer command_index in {path}")
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise ValueError(f"score without malicious_probability in {path}")
            key = (session_id, command_index)
            if key in rows:
                raise ValueError(f"duplicate score key {key!r} in {path}")
            rows[key] = record
    if not rows:
        raise ValueError("no scores loaded")
    return rows


def selected_indices(record: Mapping[str, Any], mode: str) -> tuple[int, ...]:
    commands = record["commands"]
    if mode == "full":
        return tuple(range(len(commands)))
    if mode != "cap32":
        raise ValueError(f"unsupported prefix mode: {mode}")
    label = _mapping(record.get("label", {}))
    first_evidence = label.get("first_evidence_index")
    return select_prefix_indices(
        len(commands),
        max_prefixes=32,
        first_evidence_index=(
            None if first_evidence is None else int(first_evidence)
        ),
        evidence_indices=_evidence_indices(label),
    )


def _prediction_ids(score: Mapping[str, Any], field: str) -> set[str]:
    values = score.get(field, ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    return {
        str(item.get("id"))
        for item in values
        if isinstance(item, Mapping) and item.get("id")
    }


def _session_metrics(
    records: Mapping[str, Mapping[str, Any]],
    rows: Sequence[tuple[str, int, Mapping[str, Any]]],
    *,
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for session_id, command_index, score in rows:
        grouped[session_id].append((command_index, float(score["malicious_probability"])))

    tp = fn = fp = tn = 0
    for session_id, record in records.items():
        label = _mapping(record.get("label", {}))
        malicious = bool(int(label.get("malicious", 0)))
        observations = grouped.get(session_id, ())
        if malicious:
            onset = int(label.get("first_evidence_index") or 0)
            detected = any(
                command_index >= onset and probability >= threshold
                for command_index, probability in observations
            )
            tp += int(detected)
            fn += int(not detected)
        else:
            false_alarm = any(probability >= threshold for _, probability in observations)
            fp += int(false_alarm)
            tn += int(not false_alarm)

    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    recall_ci = wilson_interval(tp, tp + fn)
    fpr_ci = wilson_interval(fp, fp + tn)
    return {
        "true_positive_sessions": tp,
        "false_negative_sessions": fn,
        "false_positive_sessions": fp,
        "true_negative_sessions": tn,
        "malicious_sessions": tp + fn,
        "benign_sessions": fp + tn,
        "recall": recall,
        "recall_ci_low": recall_ci[0],
        "recall_ci_high": recall_ci[1],
        "fpr": fpr,
        "fpr_ci_low": fpr_ci[0],
        "fpr_ci_high": fpr_ci[1],
    }


def _taxonomy_metrics(
    records: Mapping[str, Mapping[str, Any]],
    rows: Sequence[tuple[str, int, Mapping[str, Any]]],
    *,
    vocabulary: LabelVocabulary,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    tactic_targets: list[list[float]] = []
    tactic_scores: list[list[float]] = []
    technique_targets: list[list[float]] = []
    technique_scores: list[list[float]] = []
    for session_id, command_index, score in rows:
        label = _mapping(records[session_id].get("label", {}))
        target = vocabulary.encode(label, command_index)
        tactic_targets.append(list(target.tactics))
        technique_targets.append(list(target.techniques))
        predicted_tactics = _prediction_ids(score, "tactics")
        predicted_techniques = _prediction_ids(score, "techniques")
        # Missing labels are filtered as negatives by the runtime.  Use -1 as a
        # sentinel so this remains correct even if a calibration threshold is 0.
        tactic_scores.append(
            [
                float(next((item["score"] for item in score.get("tactics", ()) if item.get("id") == label_id), -1.0))
                if label_id in predicted_tactics
                else -1.0
                for label_id in vocabulary.tactics
            ]
        )
        technique_scores.append(
            [
                float(next((item["score"] for item in score.get("techniques", ()) if item.get("id") == label_id), -1.0))
                if label_id in predicted_techniques
                else -1.0
                for label_id in vocabulary.techniques
            ]
        )

    tactic_thresholds = calibration.get("tactic_thresholds") or {}
    technique_thresholds = calibration.get("technique_thresholds") or {}
    tactic_threshold_values = [
        float(tactic_thresholds.get(label_id, 0.5)) for label_id in vocabulary.tactics
    ]
    technique_threshold_values = [
        float(technique_thresholds.get(label_id, 0.5))
        for label_id in vocabulary.techniques
    ]
    return {
        "tactics": multilabel_metrics(
            np.asarray(tactic_targets, dtype=np.float64),
            np.asarray(tactic_scores, dtype=np.float64),
            thresholds=tactic_threshold_values,
        ),
        "techniques": multilabel_metrics(
            np.asarray(technique_targets, dtype=np.float64),
            np.asarray(technique_scores, dtype=np.float64),
            thresholds=technique_threshold_values,
        ),
    }


def evaluate_view(
    records: Mapping[str, Mapping[str, Any]],
    all_scores: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    vocabulary: LabelVocabulary,
    calibration: Mapping[str, Any],
    prefix_mode: str,
) -> dict[str, Any]:
    threshold = float(calibration["malicious_threshold"])
    selected: dict[str, tuple[int, ...]] = {
        session_id: selected_indices(record, prefix_mode)
        for session_id, record in records.items()
    }
    row_keys = [
        (session_id, command_index)
        for session_id, indices in selected.items()
        for command_index in indices
    ]
    missing = [key for key in row_keys if key not in all_scores]
    if missing:
        preview = missing[:10]
        raise ValueError(
            f"{len(missing)} required prefix scores are missing; first keys: {preview}"
        )
    rows = [(session_id, index, all_scores[(session_id, index)]) for session_id, index in row_keys]

    targets = np.asarray(
        [
            1.0
            if int(_mapping(records[session_id].get("label", {})).get("malicious", 0))
            and command_index
            >= int(
                _mapping(records[session_id].get("label", {})).get(
                    "first_evidence_index", 0
                )
                or 0
            )
            else 0.0
            for session_id, command_index, _ in rows
        ],
        dtype=np.float64,
    )
    probabilities = np.asarray(
        [float(score["malicious_probability"]) for _, _, score in rows],
        dtype=np.float64,
    )
    session_ids = [session_id for session_id, _, _ in rows]
    prefix_indices = [command_index for _, command_index, _ in rows]

    binary = binary_classification_metrics(
        targets, probabilities, threshold=threshold
    )
    session = _session_metrics(records, rows, threshold=threshold)
    delay = detection_delay_metrics(
        targets,
        probabilities,
        session_ids=session_ids,
        prefix_indices=prefix_indices,
        threshold=threshold,
    )
    taxonomy = _taxonomy_metrics(
        records,
        rows,
        vocabulary=vocabulary,
        calibration=calibration,
    )

    source_counts: Counter[str] = Counter()
    for record in records.values():
        source = _mapping(record.get("source", {}))
        source_counts[str(source.get("name", "unknown"))] += 1
    return {
        "prefix_mode": prefix_mode,
        "threshold": threshold,
        "sessions": len(records),
        "prefixes": len(rows),
        "selected_prefixes_per_session": {
            "min": min(len(value) for value in selected.values()),
            "max": max(len(value) for value in selected.values()),
            "mean": sum(len(value) for value in selected.values()) / len(selected),
        },
        "sources": dict(sorted(source_counts.items())),
        "malicious_prefix": binary,
        "session": session,
        "detection_delay": delay,
        "taxonomy": taxonomy,
    }


def evaluate(
    *,
    clean_path: Path,
    ood_path: Path,
    scores_paths: Sequence[Path],
    artifact_dir: Path,
) -> dict[str, Any]:
    calibration = _read_json(artifact_dir / "calibration.json")
    vocabulary = LabelVocabulary.load(artifact_dir / "labels.json")
    clean = load_records(clean_path)
    ood = load_records(ood_path)
    all_scores = load_scores(scores_paths)

    expected_keys = {
        (session_id, command_index)
        for records, mode in ((clean, "full"), (clean, "cap32"), (ood, "full"), (ood, "cap32"))
        for session_id, record in records.items()
        for command_index in selected_indices(record, mode)
    }
    missing = sorted(expected_keys - set(all_scores))
    if missing:
        raise ValueError(
            f"score file is incomplete: {len(missing)} expected keys missing; first={missing[:5]}"
        )

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "artifact_dir": str(artifact_dir),
        "score_files": [str(path) for path in scores_paths],
        "calibration": {
            "malicious_threshold": calibration.get("malicious_threshold"),
            "target_fpr": calibration.get("target_fpr"),
        },
        "clean": {
            "full": evaluate_view(
                clean,
                all_scores,
                vocabulary=vocabulary,
                calibration=calibration,
                prefix_mode="full",
            ),
            "cap32": evaluate_view(
                clean,
                all_scores,
                vocabulary=vocabulary,
                calibration=calibration,
                prefix_mode="cap32",
            ),
        },
        "ood_2024": {
            "full": evaluate_view(
                ood,
                all_scores,
                vocabulary=vocabulary,
                calibration=calibration,
                prefix_mode="full",
            ),
            "cap32": evaluate_view(
                ood,
                all_scores,
                vocabulary=vocabulary,
                calibration=calibration,
                prefix_mode="cap32",
            ),
        },
        "coverage": {
            "score_rows": len(all_scores),
            "unique_required_keys": len(expected_keys),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, action="append", type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--clean", required=True, type=Path)
    parser.add_argument("--ood", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = evaluate(
        clean_path=args.clean,
        ood_path=args.ood,
        scores_paths=args.scores,
        artifact_dir=args.artifact_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "clean_full": {
            "session_recall": report["clean"]["full"]["session"]["recall"],
            "session_fpr": report["clean"]["full"]["session"]["fpr"],
            "prefix_pr_auc": report["clean"]["full"]["malicious_prefix"]["pr_auc"],
            "prefix_roc_auc": report["clean"]["full"]["malicious_prefix"]["roc_auc"],
        },
        "clean_cap32": {
            "session_recall": report["clean"]["cap32"]["session"]["recall"],
            "session_fpr": report["clean"]["cap32"]["session"]["fpr"],
            "prefix_pr_auc": report["clean"]["cap32"]["malicious_prefix"]["pr_auc"],
            "prefix_roc_auc": report["clean"]["cap32"]["malicious_prefix"]["roc_auc"],
        },
        "ood_full_recall": report["ood_2024"]["full"]["session"]["recall"],
        "ood_cap32_recall": report["ood_2024"]["cap32"]["session"]["recall"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

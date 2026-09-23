#!/usr/bin/env python3
"""Recompute held-out metrics from a flat ``cmdrisk score`` run.

The training pipeline records metrics from the calibrated PyTorch model.  This
script produces an *independent* measurement: it consumes the JSONL emitted by
``cmdrisk score --flat`` (which is produced by the exported ONNX graph through
ONNX Runtime) and re-derives the malicious-head, detection-delay and coverage
statistics against the labelled sessions.

Usage::

    python scripts/evaluate_validation.py \
        --scores artifacts/<run>/validation_scores.jsonl \
        --test data/processed/v2/test.jsonl \
        --artifact-dir artifacts/<run>/onnx \
        --output reports/validation_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.training.metrics import (  # noqa: E402
    binary_classification_metrics,
    detection_delay_metrics,
)

SCHEMA = "cmdrisk.validation_metrics"
VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    """Return ``session_id -> label metadata`` for the held-out sessions."""

    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            session_id = record.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"session without an id in {path}")
            label = record.get("label") or {}
            source = record.get("source") or {}
            labels[session_id] = {
                "malicious": int(bool(label.get("malicious", 0))),
                "first_evidence_index": int(label.get("first_evidence_index") or 0),
                "command_count": len(record.get("commands") or []),
                "source_kind": source.get("kind"),
                "source_name": source.get("name"),
            }
    if not labels:
        raise ValueError(f"no labelled sessions found in {path}")
    return labels


def load_scores(path: Path) -> list[dict[str, Any]]:
    """Return the flat prefix predictions emitted by ``cmdrisk score --flat``."""

    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            session_id = record.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(f"prediction without a session id in {path}")
            probability = record.get("malicious_probability")
            if not isinstance(probability, (int, float)) or isinstance(probability, bool):
                raise ValueError(f"prediction without a probability in {path}")
            rows.append(
                {
                    "session_id": session_id,
                    "prefix_index": int(record.get("command_index", 0)),
                    "probability": float(probability),
                    "alert": bool(record.get("alert", False)),
                    "model_version": record.get("model_version"),
                }
            )
    if not rows:
        raise ValueError(f"no predictions found in {path}")
    return rows


def prefix_labels(
    labels: dict[str, dict[str, Any]], session_id: str, prefix_index: int
) -> float:
    """Return the prefix-level target for one observation."""

    entry = labels[session_id]
    if not entry["malicious"]:
        return 0.0
    return 1.0 if prefix_index >= entry["first_evidence_index"] else 0.0


def evaluate(
    labels: dict[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    expected_line_count: int | None = None,
) -> dict[str, Any]:
    """Compute coverage, malicious-head and detection-delay evidence."""

    unseen = sorted({row["session_id"] for row in rows} - set(labels))
    if unseen:
        raise ValueError(f"predictions reference {len(unseen)} unknown sessions")

    scored_sessions = {row["session_id"] for row in rows}
    missing = sorted(set(labels) - scored_sessions)

    targets = np.asarray(
        [
            prefix_labels(labels, row["session_id"], row["prefix_index"])
            for row in rows
        ],
        dtype=np.float64,
    )
    probabilities = np.asarray([row["probability"] for row in rows], dtype=np.float64)
    session_ids = [row["session_id"] for row in rows]
    prefix_indices = [row["prefix_index"] for row in rows]

    malicious = binary_classification_metrics(
        targets, probabilities, threshold=threshold
    )
    detection_delay = detection_delay_metrics(
        targets,
        probabilities,
        session_ids=session_ids,
        prefix_indices=prefix_indices,
        threshold=threshold,
    )

    observed_max = (
        max((row["prefix_index"] for row in rows), default=-1) + 1
    )
    per_session = Counter(row["session_id"] for row in rows)
    prefix_mismatch = {
        session_id: {"expected": entry["command_count"], "scored": per_session[session_id]}
        for session_id, entry in labels.items()
        if session_id in per_session and per_session[session_id] != entry["command_count"]
    }

    versions = sorted({str(row.get("model_version")) for row in rows})
    source_kinds = Counter(
        str(labels[session_id].get("source_kind")) for session_id in scored_sessions
    )

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "threshold": float(threshold),
        "model_versions": versions,
        "line_count": len(rows),
        "expected_line_count": expected_line_count,
        "coverage": {
            "labelled_sessions": len(labels),
            "scored_sessions": len(scored_sessions),
            "missing_sessions": missing,
            "unknown_sessions": unseen,
            "prefix_count_mismatch": prefix_mismatch,
            "max_prefix_index_observed": observed_max,
            "source_kinds": dict(sorted(source_kinds.items())),
        },
        "malicious": malicious,
        "detection_delay": detection_delay,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="ONNX artifact directory; supplies the calibrated threshold.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--expected-line-count",
        type=int,
        default=None,
        help="Fail when the score file does not contain exactly this many rows.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    threshold = args.threshold
    if threshold is None:
        if args.artifact_dir is None:
            print(
                json.dumps({"error": "provide --threshold or --artifact-dir"}),
                file=sys.stderr,
            )
            return 2
        calibration = _read_json(args.artifact_dir / "calibration.json")
        raw = calibration.get("malicious_threshold")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            print(
                json.dumps({"error": "calibration.json lacks malicious_threshold"}),
                file=sys.stderr,
            )
            return 2
        threshold = float(raw)

    labels = load_labels(args.test)
    rows = load_scores(args.scores)
    if args.expected_line_count is not None and len(rows) != args.expected_line_count:
        print(
            json.dumps(
                {
                    "error": "unexpected prediction count",
                    "expected": args.expected_line_count,
                    "actual": len(rows),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    report = evaluate(
        labels,
        rows,
        threshold=threshold,
        expected_line_count=args.expected_line_count,
    )
    rendered = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    coverage = report["coverage"]
    summary = {
        "lines": report["line_count"],
        "sessions": coverage["scored_sessions"],
        "recall": report["malicious"]["recall"],
        "fpr": report["malicious"]["fpr"],
        "detection_rate": report["detection_delay"]["detection_rate"],
    }
    print(
        json.dumps(summary, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
    )
    return 0 if not coverage["missing_sessions"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

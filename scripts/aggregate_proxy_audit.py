#!/usr/bin/env python3
"""Aggregate recorded blind reviewer votes into a proxy-audit report.

`scripts/run_proxy_audit.py` expects OpenAI-compatible endpoints.  When the
reviewers are external agents/files instead, this script replays their recorded
votes through the exact same adjudication code (`ProxyAuditor`) so the report
schema, Fleiss' kappa, and the promotion gate stay identical.  Additional
tie-break reviewer vote directories can be supplied with
``--tiebreak-reviewer-id`` and are consulted only for otherwise unresolved
cases.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.teacher import AuditCase, ProxyAuditor  # noqa: E402
from cmdrisk.teacher.prompts import reviewer_payload  # noqa: E402
from cmdrisk.teacher.responses import ProxyReviewResponse  # noqa: E402

REVIEWER_IDS = ("R1", "R2", "R3")


class ReplayReviewer:
    """Serve recorded votes from disk behind the normal reviewer interface."""

    def __init__(self, reviewer_id: str, votes: dict[str, dict[str, Any]]) -> None:
        self.reviewer_id = reviewer_id
        self.votes = votes

    def review(self, review_context: dict[str, Any]) -> ProxyReviewResponse:
        session_id = review_context["session_id"]
        if session_id not in self.votes:
            raise KeyError(f"{self.reviewer_id}: no recorded vote for {session_id}")
        return ProxyReviewResponse.model_validate(self.votes[session_id])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--votes-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-out", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--tiebreak-reviewer-id",
        action="append",
        default=[],
        metavar="ID",
        help=(
            "optional recorded reviewer directory consulted only for unresolved "
            "cases; repeatable and loaded relative to --votes-dir"
        ),
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path("reports/proxy_audit/REVIEWER_CONTRACT.md"),
    )
    parser.add_argument(
        "--reviewer-label",
        action="append",
        default=[],
        metavar="ID=LABEL",
        help="reviewer identity for the provenance record (repeatable)",
    )
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def _load_votes(
    votes_dir: Path, reviewer_id: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    directory = votes_dir / reviewer_id
    files = sorted(directory.glob("chunk_*.jsonl"))
    if not files:
        raise ValueError(f"no vote chunks found in {directory}")
    digest = hashlib.sha256()
    votes: dict[str, dict[str, Any]] = {}
    decisions: Counter[str] = Counter()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
        for row in _read_jsonl(path):
            session_id = row.get("session_id")
            if not session_id:
                raise ValueError(f"{path}: vote missing session_id")
            if session_id in votes:
                raise ValueError(f"{path}: duplicate vote for {session_id}")
            payload = {key: value for key, value in row.items() if key != "session_id"}
            votes[session_id] = payload
            decisions[str(row.get("decision"))] += 1
    provenance = {
        "reviewer": reviewer_id,
        "chunks": [path.name for path in files],
        "votes": len(votes),
        "decisions": dict(sorted(decisions.items())),
        "votes_sha256": digest.hexdigest(),
    }
    return votes, provenance


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        session_id = row.get("session_id")
        if not session_id:
            raise ValueError(f"{path}: prediction missing session_id")
        predictions[session_id] = {
            key: value for key, value in row.items() if key != "session_id"
        }
    return predictions


def _parse_labels(values: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--reviewer-label expects ID=LABEL, got {item!r}")
        key, _, value = item.partition("=")
        labels[key.strip()] = value.strip()
    return labels


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_lines = args.cases.read_text(encoding="utf-8").splitlines()
    cases = [AuditCase.model_validate_json(line) for line in raw_lines if line.strip()]
    if not cases:
        raise SystemExit(f"no cases in {args.cases}")

    predictions = _load_predictions(args.predictions) if args.predictions else {}
    if predictions:
        missing = [case.session_id for case in cases if case.session_id not in predictions]
        if missing:
            raise ValueError(f"{len(missing)} cases have no prediction, e.g. {missing[0]}")
        cases = [
            case.model_copy(update={"prediction": predictions[case.session_id]})
            for case in cases
        ]

    reviewer_ids = REVIEWER_IDS + tuple(args.tiebreak_reviewer_id)
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise ValueError("reviewer IDs must be unique, including tie-break reviewers")

    reviewers = []
    provenance: list[dict[str, Any]] = []
    for position, reviewer_id in enumerate(reviewer_ids):
        votes, record = _load_votes(args.votes_dir, reviewer_id)
        record["stage"] = "tiebreak" if position >= len(REVIEWER_IDS) else "primary"
        reviewers.append(ReplayReviewer(reviewer_id, votes))
        provenance.append(record)

    covered = {case.session_id for case in cases}
    for record, reviewer in zip(provenance, reviewers, strict=True):
        record["cases_without_vote"] = len(covered - set(reviewer.votes))

    auditor = ProxyAuditor(
        tuple(reviewers[: len(REVIEWER_IDS)]),
        tuple(reviewers[len(REVIEWER_IDS) :]),
    )
    report = auditor.audit(cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    if args.provenance_out is not None:
        contract = args.contract
        labels = _parse_labels(args.reviewer_label)
        document = {
            "contract": str(contract),
            "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest()
            if contract.exists()
            else None,
            "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "reviewers": [
                {**record, "label": labels.get(record["reviewer"])}
                for record in provenance
            ],
            "blind_payload_fields": sorted(reviewer_payload({"session_id": "", "commands": []})),
            "prediction_source": str(args.predictions) if args.predictions else None,
            "reviewer_kind": "external agent reviewers replayed through ProxyAuditor",
        }
        args.provenance_out.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(
        json.dumps(
            {
                "total_cases": report.total_cases,
                "resolved_cases": report.resolved_cases,
                "majority_resolved": report.majority_resolved,
                "tiebreak_resolved": report.tiebreak_resolved,
                "reviewers_total": report.reviewers_total,
                "unresolved": report.unresolved,
                "high_consensus": report.high_consensus,
                "medium_consensus": report.medium_consensus,
                "fleiss_kappa": report.fleiss_kappa,
                "eligible_for_promotion": report.eligible_for_promotion,
                "promotion_blockers": report.promotion_blockers,
                "prediction_metrics": report.prediction_metrics.model_dump(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

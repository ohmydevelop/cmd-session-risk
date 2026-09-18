"""Blind three-reviewer proxy audit with majority adjudication and Fleiss kappa."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cmdrisk.schema import CommandEvent
from cmdrisk.taxonomy import normalize_tactic, normalize_technique, tactics_for_techniques
from cmdrisk.teacher.client import DryRunError
from cmdrisk.teacher.prompts import reviewer_payload
from cmdrisk.teacher.responses import ProxyReviewResponse

Consensus = Literal["high", "medium", "unresolved"]


class AuditCase(BaseModel):
    """Input case; gold and prediction are deliberately excluded from prompts."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    commands: list[CommandEvent] = Field(min_length=1)
    gold: dict[str, Any] = Field(default_factory=dict)
    prediction: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


class AuditVote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reviewer: str
    decision: Literal["malicious", "benign", "uncertain", "error"]
    first_evidence_index: int | None = None
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    error: str | None = None


class ProxyAuditResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    decision: Literal["malicious", "benign", "unresolved"]
    consensus: Consensus
    first_evidence_index: int | None = None
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    votes: list[AuditVote]
    training_eligible: bool


class ProxyAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int
    high_consensus: int
    medium_consensus: int
    unresolved: int
    fleiss_kappa: float | None
    fleiss_cases: int
    results: list[ProxyAuditResult]


class ReviewModel(Protocol):
    def review(self, review_context: dict[str, Any]) -> ProxyReviewResponse: ...


class ProxyAuditor:
    """Run three isolated reviewers and adjudicate their outputs."""

    def __init__(self, reviewers: tuple[ReviewModel, ReviewModel, ReviewModel]) -> None:
        if len(reviewers) != 3:
            raise ValueError("proxy audit requires exactly three reviewers")
        self.reviewers = reviewers

    def audit_case(self, case: AuditCase) -> ProxyAuditResult:
        context = reviewer_payload(case.model_dump(mode="json"))
        votes: list[AuditVote] = []
        for index, reviewer in enumerate(self.reviewers, start=1):
            try:
                response = reviewer.review(context)
                votes.append(
                    AuditVote(
                        reviewer=f"reviewer_{index}",
                        decision=response.decision,
                        first_evidence_index=response.first_evidence_index,
                        technique_evidence=response.technique_evidence,
                        tactics=response.tactics,
                        confidence=response.confidence,
                    )
                )
            except DryRunError:
                raise
            except Exception as exc:  # reviewer failure is data for an unresolved record
                votes.append(
                    AuditVote(
                        reviewer=f"reviewer_{index}",
                        decision="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return _adjudicate(case.session_id, votes, len(case.commands))

    def audit(self, cases: Iterable[AuditCase]) -> ProxyAuditReport:
        results = [self.audit_case(case) for case in cases]
        high = sum(result.consensus == "high" for result in results)
        medium = sum(result.consensus == "medium" for result in results)
        unresolved = sum(result.consensus == "unresolved" for result in results)
        kappa, kappa_cases = _fleiss_kappa(results)
        return ProxyAuditReport(
            total_cases=len(results),
            high_consensus=high,
            medium_consensus=medium,
            unresolved=unresolved,
            fleiss_kappa=kappa,
            fleiss_cases=kappa_cases,
            results=results,
        )


def _adjudicate(
    session_id: str,
    votes: list[AuditVote],
    command_count: int,
) -> ProxyAuditResult:
    counts = Counter(vote.decision for vote in votes)
    decision_candidates = [
        (count, decision)
        for decision, count in counts.items()
        if decision in {"malicious", "benign", "uncertain"}
    ]
    if not decision_candidates:
        return ProxyAuditResult(
            session_id=session_id,
            decision="unresolved",
            consensus="unresolved",
            votes=votes,
            training_eligible=False,
        )

    max_count = max(count for count, _ in decision_candidates)
    winners = [decision for count, decision in decision_candidates if count == max_count]
    if len(winners) != 1 or winners[0] == "uncertain":
        return ProxyAuditResult(
            session_id=session_id,
            decision="unresolved",
            consensus="unresolved",
            votes=votes,
            training_eligible=False,
        )

    decision = winners[0]
    consensus: Consensus = "high" if max_count == 3 else "medium"
    if decision == "benign":
        return ProxyAuditResult(
            session_id=session_id,
            decision="benign",
            consensus=consensus,
            confidence=_mean_confidence(votes),
            votes=votes,
            training_eligible=consensus == "high",
        )

    malicious_votes = [vote for vote in votes if vote.decision == "malicious"]
    evidence = [
        vote.first_evidence_index
        for vote in malicious_votes
        if vote.first_evidence_index is not None
        and vote.first_evidence_index < command_count
    ]
    if not evidence:
        return ProxyAuditResult(
            session_id=session_id,
            decision="unresolved",
            consensus="unresolved",
            votes=votes,
            training_eligible=False,
        )
    first_evidence = min(evidence)
    technique_evidence = _technique_evidence(malicious_votes, first_evidence, command_count)
    techniques = list(technique_evidence)
    tactics = _tactics(malicious_votes, techniques)
    return ProxyAuditResult(
        session_id=session_id,
        decision="malicious",
        consensus=consensus,
        first_evidence_index=first_evidence,
        technique_evidence=technique_evidence,
        tactics=tactics,
        techniques=techniques,
        confidence=_mean_confidence(votes),
        votes=votes,
        training_eligible=consensus == "high",
    )


def _technique_evidence(
    votes: list[AuditVote],
    first_evidence: int,
    command_count: int,
) -> dict[str, list[int]]:
    by_technique: dict[str, dict[str, set[int]]] = {}
    for vote in votes:
        for raw_technique, raw_indices in vote.technique_evidence.items():
            technique = normalize_technique(raw_technique)
            if technique is None:
                continue
            valid_indices = {
                index
                for index in raw_indices
                if first_evidence <= index < command_count
            }
            if valid_indices:
                by_technique.setdefault(technique, {})[vote.reviewer] = valid_indices
    return {
        technique: sorted(set().union(*role_indices.values()))
        for technique, role_indices in sorted(by_technique.items())
        if len(role_indices) >= 2
    }


def _tactics(votes: list[AuditVote], techniques: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for vote in votes:
        normalized = {
            tactic for raw in vote.tactics if (tactic := normalize_tactic(raw)) is not None
        }
        counts.update(normalized)
    explicit = {tactic for tactic, count in counts.items() if count >= 2}
    return sorted(explicit | set(tactics_for_techniques(techniques)))


def _mean_confidence(votes: list[AuditVote]) -> float | None:
    values = [vote.confidence for vote in votes if vote.confidence is not None]
    return sum(values) / len(values) if values else None


def _fleiss_kappa(results: list[ProxyAuditResult]) -> tuple[float | None, int]:
    eligible = [
        result
        for result in results
        if all(vote.decision != "error" for vote in result.votes)
    ]
    if not eligible:
        return None, 0
    categories = sorted(
        {vote.decision for result in eligible for vote in result.votes}
    )
    if len(categories) < 2:
        return None, len(eligible)
    category_index = {category: index for index, category in enumerate(categories)}
    matrix = [
        [0] * len(categories)
        for _ in eligible
    ]
    for row, result in enumerate(eligible):
        for vote in result.votes:
            matrix[row][category_index[vote.decision]] += 1
    n = len(eligible[0].votes)
    if n < 2:
        return None, len(eligible)
    p_bar = sum(
        (sum(count * count for count in row) - n) / (n * (n - 1))
        for row in matrix
    ) / len(matrix)
    column_totals = [
        sum(row[index] for row in matrix)
        for index in range(len(categories))
    ]
    p_e = sum((total / (len(matrix) * n)) ** 2 for total in column_totals)
    if abs(1.0 - p_e) < 1e-12:
        return None, len(eligible)
    return (p_bar - p_e) / (1.0 - p_e), len(eligible)

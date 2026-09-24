"""Blind proxy audit with majority adjudication and Fleiss' kappa.

The primary panel requires at least three reviewers. A malicious or benign
decision needs a strict majority of the binary votes, and an ``uncertain``
count at least as large as that decision leaves the case unresolved. If the
primary panel cannot resolve a case, optional tie-break reviewers are consulted
one at a time until a strict majority exists; every tie-break call is retained
in the result's votes, and a case that remains unresolved after a tie-break
attempt is marked with ``adjudication='tiebreak'``.

``high`` consensus means near-unanimity: the winning count is at least
``ceil(2/3 * valid_votes)`` (for example, 3/4 and 4/4 are high, while 2/4 is
not). The original three-reviewer contract is preserved for compatibility:
2/3 remains ``medium`` and 3/3 is ``high``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cmdrisk.schema import CommandEvent
from cmdrisk.taxonomy import normalize_tactic, normalize_technique, tactics_for_techniques
from cmdrisk.teacher.client import DryRunError
from cmdrisk.teacher.prompts import reviewer_payload
from cmdrisk.teacher.responses import ProxyReviewResponse

Consensus = Literal["high", "medium", "unresolved"]
Adjudication = Literal["majority", "tiebreak", "unresolved"]


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
    adjudication: Adjudication
    reviewers_total: int = Field(ge=1)
    first_evidence_index: int | None = None
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    votes: list[AuditVote]
    training_eligible: bool


class ProxyAuditMetrics(BaseModel):
    """Prediction quality against adjudicated, non-unresolved audit cases."""

    model_config = ConfigDict(extra="forbid")

    resolved_cases: int
    scored_cases: int
    unpredicted_cases: int
    prediction_coverage: float
    threshold: float = Field(default=0.5, ge=0, le=1)
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    accuracy: float | None
    precision: float | None
    recall: float | None
    false_positive_rate: float | None
    fpr: float | None
    brier_score: float | None
    prediction_agreement: float | None
    malicious_prevalence: float | None


class ProxyAuditReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int
    resolved_cases: int
    majority_resolved: int
    tiebreak_resolved: int
    reviewers_total: int
    high_consensus: int
    medium_consensus: int
    unresolved: int
    fleiss_kappa: float | None
    fleiss_cases: int
    minimum_resolved_sessions: int = 500
    eligible_for_promotion: bool
    promotion_blockers: list[str] = Field(default_factory=list)
    prediction_metrics: ProxyAuditMetrics
    results: list[ProxyAuditResult]


class ReviewModel(Protocol):
    def review(self, review_context: dict[str, Any]) -> ProxyReviewResponse: ...


class ProxyAuditor:
    """Run at least three primary reviewers plus optional staged tie-breakers."""

    def __init__(
        self,
        reviewers: Sequence[ReviewModel],
        tiebreak_reviewers: Sequence[ReviewModel] = (),
    ) -> None:
        if len(reviewers) < 3:
            raise ValueError("proxy audit requires at least three reviewers")
        self.reviewers = tuple(reviewers)
        self.tiebreak_reviewers = tuple(tiebreak_reviewers)

    def audit_case(self, case: AuditCase) -> ProxyAuditResult:
        context = reviewer_payload(case.model_dump(mode="json"))
        votes = [
            _review_once(reviewer, index, context)
            for index, reviewer in enumerate(self.reviewers, start=1)
        ]
        result = _adjudicate(
            case.session_id,
            votes,
            len(case.commands),
            adjudication="majority",
        )
        if result.decision != "unresolved" or not self.tiebreak_reviewers:
            return result

        next_index = len(self.reviewers) + 1
        for reviewer in self.tiebreak_reviewers:
            votes.append(_review_once(reviewer, next_index, context))
            result = _adjudicate(
                case.session_id,
                votes,
                len(case.commands),
                adjudication="tiebreak",
            )
            if result.decision != "unresolved":
                return result
            next_index += 1
        return result

    def audit(self, cases: Iterable[AuditCase]) -> ProxyAuditReport:
        case_list = list(cases)
        results = [self.audit_case(case) for case in case_list]
        majority_resolved = sum(
            result.decision != "unresolved" and result.adjudication == "majority"
            for result in results
        )
        tiebreak_resolved = sum(
            result.decision != "unresolved" and result.adjudication == "tiebreak"
            for result in results
        )
        high = sum(result.consensus == "high" for result in results)
        medium = sum(result.consensus == "medium" for result in results)
        unresolved = sum(result.consensus == "unresolved" for result in results)
        resolved = len(results) - unresolved
        reviewers_total = sum(result.reviewers_total for result in results)
        kappa, kappa_cases = _fleiss_kappa(results)
        prediction_metrics = _prediction_metrics(
            list(zip(case_list, results, strict=True)),
        )
        blockers: list[str] = []
        minimum_resolved_sessions = 500
        if resolved < minimum_resolved_sessions:
            blockers.append(
                f"resolved_sessions_below_minimum:{resolved}<{minimum_resolved_sessions}"
            )
        if unresolved:
            blockers.append(f"unresolved_sessions_present:{unresolved}")
        if prediction_metrics.scored_cases == 0:
            blockers.append("prediction_labels_missing")
        elif prediction_metrics.scored_cases < resolved:
            blockers.append(
                f"prediction_labels_incomplete:{prediction_metrics.scored_cases}<{resolved}"
            )
        return ProxyAuditReport(
            total_cases=len(results),
            resolved_cases=resolved,
            majority_resolved=majority_resolved,
            tiebreak_resolved=tiebreak_resolved,
            reviewers_total=reviewers_total,
            high_consensus=high,
            medium_consensus=medium,
            unresolved=unresolved,
            fleiss_kappa=kappa,
            fleiss_cases=kappa_cases,
            minimum_resolved_sessions=minimum_resolved_sessions,
            eligible_for_promotion=not blockers,
            promotion_blockers=blockers,
            prediction_metrics=prediction_metrics,
            results=results,
        )


def _review_once(
    reviewer: ReviewModel,
    index: int,
    context: dict[str, Any],
) -> AuditVote:
    try:
        response = reviewer.review(context)
        return AuditVote(
            reviewer=f"reviewer_{index}",
            decision=response.decision,
            first_evidence_index=response.first_evidence_index,
            technique_evidence=response.technique_evidence,
            tactics=response.tactics,
            confidence=response.confidence,
        )
    except DryRunError:
        raise
    except Exception as exc:  # reviewer failure is data for an unresolved record
        return AuditVote(
            reviewer=f"reviewer_{index}",
            decision="error",
            error=f"{type(exc).__name__}: {exc}",
        )


def _adjudicate(
    session_id: str,
    votes: list[AuditVote],
    command_count: int,
    *,
    adjudication: Literal["majority", "tiebreak"] = "majority",
) -> ProxyAuditResult:
    counts = Counter(vote.decision for vote in votes)
    malicious = counts["malicious"]
    benign = counts["benign"]
    uncertain = counts["uncertain"]
    valid_votes = malicious + benign + uncertain
    unresolved_adjudication: Adjudication = (
        "unresolved" if adjudication == "majority" else "tiebreak"
    )

    # An uncertain plurality or tie means the binary reviewers did not agree.
    if uncertain >= max(malicious, benign):
        return _unresolved_result(
            session_id,
            votes,
            adjudication=unresolved_adjudication,
        )

    binary_votes = malicious + benign
    if malicious * 2 > binary_votes:
        decision = "malicious"
        max_count = malicious
    elif benign * 2 > binary_votes:
        decision = "benign"
        max_count = benign
    else:
        return _unresolved_result(
            session_id,
            votes,
            adjudication=unresolved_adjudication,
        )

    consensus = _consensus(max_count, valid_votes, len(votes))
    if decision == "benign":
        return ProxyAuditResult(
            session_id=session_id,
            decision="benign",
            consensus=consensus,
            adjudication=adjudication,
            reviewers_total=len(votes),
            confidence=_mean_confidence(votes),
            votes=votes,
            training_eligible=consensus == "high",
        )

    malicious_votes = [vote for vote in votes if vote.decision == "malicious"]
    evidence = [
        vote.first_evidence_index
        for vote in malicious_votes
        if vote.first_evidence_index is not None and vote.first_evidence_index < command_count
    ]
    if not evidence:
        return _unresolved_result(
            session_id,
            votes,
            adjudication=unresolved_adjudication,
        )
    first_evidence = min(evidence)
    technique_evidence = _technique_evidence(malicious_votes, first_evidence, command_count)
    techniques = list(technique_evidence)
    tactics = _tactics(malicious_votes, techniques)
    return ProxyAuditResult(
        session_id=session_id,
        decision="malicious",
        consensus=consensus,
        adjudication=adjudication,
        reviewers_total=len(votes),
        first_evidence_index=first_evidence,
        technique_evidence=technique_evidence,
        tactics=tactics,
        techniques=techniques,
        confidence=_mean_confidence(votes),
        votes=votes,
        training_eligible=consensus == "high",
    )


def _unresolved_result(
    session_id: str,
    votes: list[AuditVote],
    *,
    adjudication: Adjudication = "unresolved",
) -> ProxyAuditResult:
    return ProxyAuditResult(
        session_id=session_id,
        decision="unresolved",
        consensus="unresolved",
        adjudication=adjudication,
        reviewers_total=len(votes),
        votes=votes,
        training_eligible=False,
    )


def _consensus(max_count: int, valid_votes: int, reviewers_total: int) -> Consensus:
    # Keep the pre-tie-break three-reviewer scoring contract byte-compatible.
    if reviewers_total == 3:
        return "high" if max_count == 3 else "medium"
    high_threshold = (2 * valid_votes + 2) // 3
    return "high" if max_count >= high_threshold else "medium"


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
                index for index in raw_indices if first_evidence <= index < command_count
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


def _prediction_metrics(
    pairs: list[tuple[AuditCase, ProxyAuditResult]],
    *,
    threshold: float = 0.5,
) -> ProxyAuditMetrics:
    """Score model predictions only where reviewers resolved the case.

    Unresolved cases are deliberately excluded from agreement metrics: treating
    a failed adjudication as a gold label would hide exactly the uncertainty the
    audit is supposed to surface.
    """

    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")
    resolved = [(case, result) for case, result in pairs if result.consensus != "unresolved"]
    scored: list[tuple[int, int]] = []
    brier_values: list[float] = []
    unpredicted = 0
    for case, result in resolved:
        predicted = _prediction_label(case.prediction, threshold=threshold)
        if predicted is None:
            unpredicted += 1
            continue
        adjudicated = 1 if result.decision == "malicious" else 0
        scored.append((adjudicated, predicted))
        probability = _prediction_probability(case.prediction)
        if probability is not None:
            brier_values.append((probability - adjudicated) ** 2)

    true_positive = sum(actual == 1 and predicted == 1 for actual, predicted in scored)
    true_negative = sum(actual == 0 and predicted == 0 for actual, predicted in scored)
    false_positive = sum(actual == 0 and predicted == 1 for actual, predicted in scored)
    false_negative = sum(actual == 1 and predicted == 0 for actual, predicted in scored)
    positives = true_positive + false_negative
    negatives = true_negative + false_positive
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    )
    recall = true_positive / positives if positives else None
    false_positive_rate = false_positive / negatives if negatives else None
    accuracy = (true_positive + true_negative) / len(scored) if scored else None
    return ProxyAuditMetrics(
        resolved_cases=len(resolved),
        scored_cases=len(scored),
        unpredicted_cases=unpredicted,
        prediction_coverage=len(scored) / len(resolved) if resolved else 0.0,
        threshold=threshold,
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        false_positive_rate=false_positive_rate,
        fpr=false_positive_rate,
        brier_score=sum(brier_values) / len(brier_values) if brier_values else None,
        prediction_agreement=accuracy,
        malicious_prevalence=(positives / len(scored)) if scored else None,
    )


def _prediction_label(prediction: dict[str, Any], *, threshold: float = 0.5) -> int | None:
    probability = _prediction_probability(prediction)
    if probability is not None:
        return int(probability >= threshold)
    for key in ("malicious", "label", "decision"):
        if key not in prediction:
            continue
        value = prediction[key]
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in (0, 1):
            return int(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"malicious", "attack", "1", "true", "positive"}:
                return 1
            if normalized in {"benign", "normal", "0", "false", "negative"}:
                return 0
    return None


def _prediction_probability(prediction: dict[str, Any]) -> float | None:
    value = prediction.get("malicious_probability", prediction.get("probability"))
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and 0 <= float(value) <= 1:
        return float(value)
    return None


def _fleiss_kappa(results: list[ProxyAuditResult]) -> tuple[float | None, int]:
    eligible = [
        result
        for result in results
        if result.consensus != "unresolved"
        and all(vote.decision != "error" for vote in result.votes)
    ]
    if not eligible:
        return None, 0
    categories = sorted({vote.decision for result in eligible for vote in result.votes})
    if len(categories) < 2:
        return None, len(eligible)
    category_index = {category: index for index, category in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in eligible]
    for row, result in enumerate(eligible):
        for vote in result.votes:
            matrix[row][category_index[vote.decision]] += 1

    reviewer_counts = [len(result.votes) for result in eligible]
    if any(count < 2 for count in reviewer_counts):
        return None, len(eligible)
    p_bar = sum(
        (sum(count * count for count in row) - reviewer_count)
        / (reviewer_count * (reviewer_count - 1))
        for row, reviewer_count in zip(matrix, reviewer_counts, strict=True)
    ) / len(matrix)
    total_ratings = sum(reviewer_counts)
    column_totals = [sum(row[index] for row in matrix) for index in range(len(categories))]
    p_e = sum((total / total_ratings) ** 2 for total in column_totals)
    if abs(1.0 - p_e) < 1e-12:
        return None, len(eligible)
    return (p_bar - p_e) / (1.0 - p_e), len(eligible)

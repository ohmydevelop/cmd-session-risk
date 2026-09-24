from __future__ import annotations

import pytest

from cmdrisk.teacher.audit import AuditCase, ProxyAuditor
from cmdrisk.teacher.prompts import reviewer_payload
from cmdrisk.teacher.responses import ProxyReviewResponse


class FakeReviewer:
    def __init__(self, response: ProxyReviewResponse | Exception) -> None:
        self.response = response
        self.contexts: list[dict[str, object]] = []

    def review(self, review_context: dict[str, object]) -> ProxyReviewResponse:
        self.contexts.append(review_context)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _review(
    decision: str,
    evidence: int | None = None,
    *,
    technique: str | None = None,
) -> ProxyReviewResponse:
    techniques = {technique: [evidence]} if technique is not None and evidence is not None else {}
    return ProxyReviewResponse(
        decision=decision,
        first_evidence_index=evidence,
        technique_evidence=techniques,
        tactics=["TA0011"] if decision == "malicious" else [],
        confidence=0.9,
    )


def _case(session_id: str = "case-1") -> AuditCase:
    return AuditCase(
        session_id=session_id,
        commands=[
            {"cmd": "whoami"},
            {"cmd": "curl -o /tmp/a URL"},
            {"cmd": "chmod +x /tmp/a"},
            {"cmd": "/tmp/a"},
        ],
        gold={"malicious": 1, "first_evidence_index": 3},
        prediction={"malicious_probability": 0.97},
    )


class RoutedReviewer:
    def __init__(self, responses: dict[str, ProxyReviewResponse]) -> None:
        self.responses = responses

    def review(self, review_context: dict[str, object]) -> ProxyReviewResponse:
        session_id = str(review_context["session_id"])
        return self.responses[session_id]


def _prediction_case(session_id: str, probability: float) -> AuditCase:
    return _case(session_id).model_copy(
        update={"prediction": {"malicious_probability": probability}}
    )


def test_reviewer_context_hides_gold_prediction_and_meta() -> None:
    context = reviewer_payload(_case().model_dump(mode="json"))
    assert set(context) == {"session_id", "commands"}
    serialized = str(context)
    assert "gold" not in serialized
    assert "prediction" not in serialized
    assert "0.97" not in serialized


def test_three_reviewer_majority_builds_high_consensus() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 3, technique="T1222")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())
    assert result.decision == "malicious"
    assert result.consensus == "high"
    assert result.adjudication == "majority"
    assert result.reviewers_total == 3
    assert result.first_evidence_index == 2
    assert result.techniques == ["T1105"]
    assert result.training_eligible is True


def test_two_to_one_review_is_medium_not_training_eligible() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())
    assert result.decision == "malicious"
    assert result.consensus == "medium"
    assert result.adjudication == "majority"
    assert result.reviewers_total == 3
    assert result.training_eligible is False


def test_split_vote_is_unresolved() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("uncertain")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())
    assert result.decision == "unresolved"
    assert result.consensus == "unresolved"
    assert result.adjudication == "unresolved"
    assert result.reviewers_total == 3
    assert result.training_eligible is False


def test_fewer_than_three_reviewers_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least three"):
        ProxyAuditor(
            (
                FakeReviewer(_review("benign")),
                FakeReviewer(_review("benign")),
            )
        )


def test_tiebreak_reviewer_resolves_three_way_split() -> None:
    primary = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("uncertain")),
    )
    tiebreak = FakeReviewer(_review("malicious", 3, technique="T1105"))
    result = ProxyAuditor(primary, (tiebreak,)).audit_case(_case())

    assert result.decision == "malicious"
    assert result.adjudication == "tiebreak"
    assert result.reviewers_total == 4
    assert [vote.reviewer for vote in result.votes] == [
        "reviewer_1",
        "reviewer_2",
        "reviewer_3",
        "reviewer_4",
    ]
    assert result.votes[-1].decision == "malicious"
    assert result.consensus == "medium"
    assert result.training_eligible is False


def test_report_surfaces_tiebreak_counts_and_total_reviewers() -> None:
    primary = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("uncertain")),
    )
    tiebreak = FakeReviewer(_review("malicious", 3, technique="T1105"))
    report = ProxyAuditor(primary, (tiebreak,)).audit([_case()])

    assert report.resolved_cases == 1
    assert report.majority_resolved == 0
    assert report.tiebreak_resolved == 1
    assert report.reviewers_total == 4
    assert report.unresolved == 0


def test_unsuccessful_tiebreak_is_recorded_as_tiebreak_unresolved() -> None:
    primary = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("uncertain")),
    )
    tiebreak = FakeReviewer(_review("uncertain"))
    result = ProxyAuditor(primary, (tiebreak,)).audit_case(_case())

    assert result.decision == "unresolved"
    assert result.adjudication == "tiebreak"
    assert result.reviewers_total == 4
    assert result.votes[-1].reviewer == "reviewer_4"
    assert result.votes[-1].decision == "uncertain"


def test_tiebreak_reviewers_are_not_called_after_primary_resolution() -> None:
    tiebreak = FakeReviewer(RuntimeError("should not be called"))
    primary = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
    )
    result = ProxyAuditor(primary, (tiebreak,)).audit_case(_case())

    assert result.decision == "malicious"
    assert result.adjudication == "majority"
    assert result.reviewers_total == 3
    assert tiebreak.contexts == []


def test_four_primary_reviewers_can_tie_unresolved() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("benign")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())

    assert result.decision == "unresolved"
    assert result.consensus == "unresolved"
    assert result.adjudication == "unresolved"
    assert result.reviewers_total == 4


def test_four_primary_reviewers_three_to_one_is_high_consensus() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())

    assert result.decision == "malicious"
    assert result.consensus == "high"
    assert result.adjudication == "majority"
    assert result.reviewers_total == 4
    assert result.training_eligible is True


def test_four_primary_reviewers_unanimous_is_high_consensus() -> None:
    reviewers = tuple(FakeReviewer(_review("malicious", 2, technique="T1105")) for _ in range(4))
    result = ProxyAuditor(reviewers).audit_case(_case())

    assert result.decision == "malicious"
    assert result.consensus == "high"
    assert result.adjudication == "majority"
    assert result.reviewers_total == 4
    assert result.training_eligible is True


def test_two_of_four_valid_votes_is_not_high_consensus() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("uncertain")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())

    assert result.decision == "malicious"
    assert result.consensus == "medium"
    assert result.adjudication == "majority"
    assert result.training_eligible is False


def test_reviewer_error_is_preserved_as_unresolved_record() -> None:
    reviewers = (
        FakeReviewer(RuntimeError("provider unavailable")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("benign")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())
    assert result.decision == "benign"
    assert result.consensus == "medium"
    assert result.votes[0].decision == "error"
    assert "provider unavailable" in (result.votes[0].error or "")


def test_fleiss_kappa_is_reported_for_complete_reviews() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
    )
    report = ProxyAuditor(reviewers).audit([_case("case-1")])
    assert report.fleiss_kappa is None
    assert report.fleiss_cases == 1

    unanimous = (
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
        FakeReviewer(_review("malicious", 2, technique="T1105")),
    )
    # Reuse three per-case auditor fixtures because reviewer instances are stateful mocks.
    first = ProxyAuditor(unanimous).audit_case(_case("case-1"))
    benign_reviewers = (
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("benign")),
        FakeReviewer(_review("benign")),
    )
    second = ProxyAuditor(benign_reviewers).audit_case(_case("case-2"))
    from cmdrisk.teacher.audit import _fleiss_kappa

    kappa, count = _fleiss_kappa([first, second])
    assert count == 2
    assert kappa == pytest.approx(1.0)


def test_malicious_majority_without_in_range_evidence_is_unresolved() -> None:
    reviewers = (
        FakeReviewer(_review("malicious", 99, technique="T1105")),
        FakeReviewer(_review("malicious", 99, technique="T1105")),
        FakeReviewer(_review("benign")),
    )
    result = ProxyAuditor(reviewers).audit_case(_case())
    assert result.decision == "unresolved"
    assert result.consensus == "unresolved"
    assert result.first_evidence_index is None
    assert result.training_eligible is False


def test_prediction_metrics_exclude_unresolved_cases_and_expose_gate() -> None:
    cases = [
        _prediction_case("tp", 0.9),
        _prediction_case("fn", 0.1),
        _prediction_case("fp", 0.8),
        _prediction_case("tn", 0.2),
        _prediction_case("unresolved", 0.99),
    ]
    first = {case.session_id: _review("malicious", 2, technique="T1105") for case in cases}
    second = {
        case.session_id: _review("malicious", 2, technique="T1105")
        if case.session_id in {"tp", "fn"}
        else _review("benign")
        for case in cases
    }
    third = {
        case.session_id: _review("malicious", 2, technique="T1105")
        if case.session_id in {"tp", "fn"}
        else _review("benign")
        for case in cases
    }
    # Only the unresolved row changes third reviewer to uncertain.
    third["unresolved"] = _review("uncertain")
    # The unresolved row is excluded even though the model predicted attack.
    report = ProxyAuditor(
        (
            RoutedReviewer(first),
            RoutedReviewer(second),
            RoutedReviewer(third),
        )
    ).audit(cases)
    metrics = report.prediction_metrics
    assert report.resolved_cases == 4
    assert report.unresolved == 1
    assert metrics.scored_cases == 4
    assert metrics.unpredicted_cases == 0
    assert metrics.true_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.false_positive_rate == pytest.approx(0.5)
    assert metrics.fpr == pytest.approx(0.5)
    assert metrics.brier_score is not None
    assert report.minimum_resolved_sessions == 500
    assert report.eligible_for_promotion is False
    assert any("resolved_sessions_below_minimum" in item for item in report.promotion_blockers)
    assert any("unresolved_sessions_present" in item for item in report.promotion_blockers)


def test_proxy_audit_gate_accepts_500_resolved_complete_predictions() -> None:
    cases = [_prediction_case(f"case-{index}", 0.01) for index in range(500)]
    benign = {case.session_id: _review("benign") for case in cases}
    report = ProxyAuditor(
        (
            RoutedReviewer(benign),
            RoutedReviewer(benign),
            RoutedReviewer(benign),
        )
    ).audit(cases)
    assert report.resolved_cases == 500
    assert report.unresolved == 0
    assert report.prediction_metrics.scored_cases == 500
    assert report.eligible_for_promotion is True
    assert report.promotion_blockers == []

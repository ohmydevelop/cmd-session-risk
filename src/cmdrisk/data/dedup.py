"""Deterministic exact deduplication and near-duplicate family assignment."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable

from cmdrisk.schema import SessionRecord

from .normalization import session_skeleton


def _geometry_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    product = 1.0
    for value in values:
        product *= max(value, 1e-12)
    return product ** (1.0 / len(values))


def jaccard(left: set[str] | tuple[str, ...], right: set[str] | tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def session_jaccard(left: SessionRecord, right: SessionRecord) -> float:
    return jaccard(set(session_skeleton(left)), set(session_skeleton(right)))


def exact_deduplicate(records: Iterable[SessionRecord]) -> list[SessionRecord]:
    """Keep the first record for each exact normalized command sequence.

    Exact deduplication deliberately uses literal cleaned commands, not the
    masked skeleton. Two sessions that differ only in URLs, IPs, paths, or
    other high-cardinality values are distinct raw observations and are grouped
    later by :func:`assign_families` after skeletonization.
    """

    kept: list[SessionRecord] = []
    seen: dict[tuple[str, ...], str] = {}
    for record in records:
        sequence = tuple(event.cmd for event in record.commands)
        original = seen.get(sequence)
        if original is not None:
            continue
        seen[sequence] = record.session_id
        kept.append(record)
    return kept


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _shingles(commands: tuple[str, ...], width: int = 3) -> set[str]:
    if len(commands) < width:
        return {f"c:{command}" for command in commands}
    return {
        "s:" + "\x1f".join(commands[index : index + width])
        for index in range(len(commands) - width + 1)
    }


def assign_families(
    records: Iterable[SessionRecord], *, threshold: float = 0.9
) -> list[SessionRecord]:
    """Cluster near-duplicate sessions using command-set Jaccard similarity."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    values = list(records)
    if not values:
        return []
    skeletons = [session_skeleton(record) for record in values]
    command_sets = [set(skeleton) for skeleton in skeletons]
    union_find = _UnionFind(len(values))

    inverted: dict[str, list[int]] = defaultdict(list)
    for index, skeleton in enumerate(skeletons):
        for token in _shingles(skeleton):
            inverted[token].append(index)
        if len(skeleton) < 3:
            for command in skeleton:
                inverted[f"c:{command}"].append(index)

    compared: set[tuple[int, int]] = set()
    for members in inverted.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        for offset, left in enumerate(members):
            for right in members[offset + 1 :]:
                pair = (left, right)
                if pair in compared:
                    continue
                compared.add(pair)
                left_size = len(command_sets[left])
                right_size = len(command_sets[right])
                if min(left_size, right_size) / max(left_size, right_size, 1) < threshold:
                    continue
                similarity = jaccard(command_sets[left], command_sets[right])
                if similarity >= threshold:
                    union_find.union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(values)):
        clusters[union_find.find(index)].append(index)

    family_for_root: dict[int, str] = {}
    for root, indices in clusters.items():
        member_ids = sorted(values[index].session_id for index in indices)
        digest = hashlib.sha256("\x1f".join(member_ids).encode("utf-8")).hexdigest()[:24]
        family_for_root[root] = f"family:{digest}"

    return [
        record.model_copy(update={"family_id": family_for_root[union_find.find(index)]})
        for index, record in enumerate(values)
    ]


def _stable_rank(record: SessionRecord, seed: int, family_id: str) -> str:
    value = f"{seed}\x1f{family_id}\x1f{record.session_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


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
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must be in (0, 1]")
    cap = max(minimum, round(len(values) * max_fraction))
    grouped: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in values:
        grouped[record.family_id].append(record)
    kept: list[SessionRecord] = []
    for family_id, members in grouped.items():
        ranked = sorted(members, key=lambda item: _stable_rank(item, seed, family_id))
        kept.extend(ranked[:cap])
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

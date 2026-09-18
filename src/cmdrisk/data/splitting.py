"""Family-aware deterministic train/dev/test splitting."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping

from cmdrisk.schema import SessionRecord, SplitName

DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "dev": 0.1, "test": 0.1}


def _family_rank(family_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{family_id}".encode("utf-8")).hexdigest()


def family_aware_split(
    records: Iterable[SessionRecord],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
) -> list[SessionRecord]:
    """Assign complete families to splits using deterministic capacity balancing."""

    split_ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    if set(split_ratios) != {"train", "dev", "test"}:
        raise ValueError("split ratios must contain train, dev, and test")
    if any(value < 0 for value in split_ratios.values()) or not abs(
        sum(split_ratios.values()) - 1.0
    ) < 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1.0")

    values = list(records)
    if not values:
        return []
    families: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in values:
        families[record.family_id].append(record)

    total = len(values)
    targets = {name: total * ratio for name, ratio in split_ratios.items()}
    assigned_counts = {name: 0 for name in split_ratios}
    assignments: dict[str, str] = {}
    ordered_families = sorted(
        families, key=lambda family_id: _family_rank(family_id, seed), reverse=True
    )
    for family_id in ordered_families:
        size = len(families[family_id])
        priorities = {
            name: (targets[name] - assigned_counts[name]) / max(targets[name], 1.0)
            for name in split_ratios
        }
        split = max(split_ratios, key=lambda name: (priorities[name], -list(split_ratios).index(name)))
        assignments[family_id] = split
        assigned_counts[split] += size

    output: list[SessionRecord] = []
    for record in values:
        split = SplitName(assignments[record.family_id])
        output.append(record.model_copy(update={"split": split}))
    return output

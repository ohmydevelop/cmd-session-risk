"""Class-aware deterministic sampling for the first training corpus."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from cmdrisk.schema import SessionRecord, SourceKind

DEFAULT_QUOTA: dict[str, float] = {
    "real_malicious": 0.40,
    "benign": 0.40,
    "llm_variant": 0.15,
    "hard_negative": 0.05,
}


def source_bucket(record: SessionRecord) -> str:
    kind = str(record.source.kind)
    if kind == str(SourceKind.REAL_MALICIOUS):
        return "real_malicious"
    if kind in {str(SourceKind.REAL_BENIGN), str(SourceKind.SCRIPT_BENIGN)}:
        return "benign"
    if kind == str(SourceKind.LLM_VARIANT):
        return "llm_variant"
    if kind == str(SourceKind.HARD_NEGATIVE):
        return "hard_negative"
    return "benign"


def _rank(record: SessionRecord, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{record.session_id}".encode("utf-8")).hexdigest()


def sample_training_quota(
    records: Iterable[SessionRecord],
    *,
    total: int = 60_000,
    quota: Mapping[str, float] | None = None,
    seed: int = 42,
) -> list[SessionRecord]:
    """Sample requested buckets and backfill shortages from other buckets."""

    if total < 1:
        return []
    ratios = dict(quota or DEFAULT_QUOTA)
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("quota values must sum to 1.0")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("quota values must be non-negative")

    grouped: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in records:
        grouped[source_bucket(record)].append(record)
    for bucket in ratios:
        grouped[bucket].sort(key=lambda item: _rank(item, seed))

    allocations = {
        bucket: min(len(grouped[bucket]), int(round(total * ratio)))
        for bucket, ratio in ratios.items()
    }
    selected: dict[str, list[SessionRecord]] = defaultdict(list)
    selected.update(
        {bucket: grouped[bucket][:count] for bucket, count in allocations.items()}
    )

    remaining = total - sum(len(items) for items in selected.values())
    if remaining > 0:
        candidates = [record for bucket in ratios for record in grouped[bucket][allocations[bucket] :]]
        candidates.sort(key=lambda item: _rank(item, seed + 1))
        for record in candidates:
            if remaining == 0:
                break
            selected[source_bucket(record)].append(record)
            remaining -= 1

    result = [record for bucket in ratios for record in selected[bucket]]
    return sorted(result, key=lambda item: _rank(item, seed))

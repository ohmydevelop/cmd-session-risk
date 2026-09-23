"""Class-aware deterministic sampling for the first training corpus."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping

from cmdrisk.schema import SessionRecord, SourceKind

# The requested target keeps real and script benign in one 40% bucket.  The
# finer source kinds are still available to custom quotas for source audits.
DEFAULT_QUOTA: dict[str, float] = {
    "real_malicious": 0.40,
    "benign": 0.40,
    "llm_variant": 0.15,
    "hard_negative": 0.05,
}

_SOURCE_BUCKETS = {
    str(SourceKind.REAL_MALICIOUS): "real_malicious",
    str(SourceKind.REAL_BENIGN): "real_benign",
    str(SourceKind.SCRIPT_BENIGN): "script_benign",
    str(SourceKind.REAL_HISTORY_POOL): "real_history_pool",
    str(SourceKind.LLM_VARIANT): "llm_variant",
    str(SourceKind.HARD_NEGATIVE): "hard_negative",
}
_BENIGN_BUCKETS = {"real_benign", "script_benign", "real_history_pool"}
_ATOMIC_SOURCE_NAMES = {"atomic_red_team", "atomic-red-team"}
_ATOMIC_META_KEYS = {"atomic_source", "atomic_test_name", "atomic_seed", "technique_seed"}
_ATOMIC_ROLES = {"atomic_seed", "technique_seed"}


def is_atomic_seed(record: SessionRecord) -> bool:
    """Return whether a real-malicious record is an Atomic expansion seed."""

    if record.source.kind != SourceKind.REAL_MALICIOUS:
        return False
    if record.source.name.casefold() in _ATOMIC_SOURCE_NAMES:
        return True
    if any(key in record.meta for key in _ATOMIC_META_KEYS):
        return True
    role = str(record.meta.get("dataset_role", "")).casefold()
    return role in _ATOMIC_ROLES


def source_bucket(record: SessionRecord) -> str:
    """Map a record to the legacy/default primary training bucket."""

    kind = str(record.source.kind)
    try:
        bucket = _SOURCE_BUCKETS[kind]
    except KeyError as error:
        raise ValueError(f"unsupported source kind for quota sampling: {kind!r}") from error
    if bucket in _BENIGN_BUCKETS:
        return "benign"
    return bucket


def _record_bucket(record: SessionRecord, quota: Mapping[str, float]) -> str:
    kind = str(record.source.kind)
    try:
        bucket = _SOURCE_BUCKETS[kind]
    except KeyError as error:
        raise ValueError(f"unsupported source kind for quota sampling: {kind!r}") from error
    if "benign" in quota and bucket in _BENIGN_BUCKETS:
        return "benign"
    return bucket


def _rank(record: SessionRecord, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{record.session_id}".encode()).hexdigest()


def _allocate(total: int, ratios: Mapping[str, float]) -> dict[str, int]:
    """Allocate integer counts with deterministic largest-remainder rounding."""

    raw = {bucket: total * ratio for bucket, ratio in ratios.items()}
    allocations = {bucket: math.floor(value) for bucket, value in raw.items()}
    remainder = total - sum(allocations.values())
    order = {bucket: index for index, bucket in enumerate(ratios)}
    ranked = sorted(
        ratios,
        key=lambda bucket: (-(raw[bucket] - allocations[bucket]), order[bucket]),
    )
    for bucket in ranked[:remainder]:
        allocations[bucket] += 1
    return allocations


def allocate_quota(
    total: int,
    quota: Mapping[str, float] | None = None,
) -> dict[str, int]:
    """Return deterministic integer bucket counts for ``total`` and ``quota``.

    The allocation uses the same largest-remainder rounding as
    :func:`sample_training_quota`, so callers can pre-allocate exactly as many
    synthetic rows as the sampler will later demand.
    """

    if total < 1:
        return {bucket: 0 for bucket in (quota or DEFAULT_QUOTA)}
    ratios = dict(quota or DEFAULT_QUOTA)
    if not ratios:
        raise ValueError("quota must contain at least one bucket")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("quota values must sum to 1.0")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("quota values must be non-negative")
    return _allocate(total, ratios)


def sample_training_quota(
    records: Iterable[SessionRecord],
    *,
    total: int = 60_000,
    quota: Mapping[str, float] | None = None,
    seed: int = 42,
    strict: bool = True,
    allow_backfill: bool = False,
    drift_tolerance: float = 0.0,
) -> list[SessionRecord]:
    """Select a deterministic class-aware training sample.

    Strict mode is the default and fails when any requested bucket is short or
    when the realized composition drifts. ``allow_backfill=True`` is an
    explicit opt-in for smoke tests and legacy callers; it fills the remaining
    rows from underrepresented buckets and intentionally permits drift.
    """

    if total < 1:
        return []
    ratios = dict(quota or DEFAULT_QUOTA)
    if not ratios:
        raise ValueError("quota must contain at least one bucket")
    if any(not bucket for bucket in ratios):
        raise ValueError("quota bucket names must be non-empty")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("quota values must sum to 1.0")
    if any(value < 0 for value in ratios.values()):
        raise ValueError("quota values must be non-negative")
    if drift_tolerance < 0:
        raise ValueError("drift_tolerance must be non-negative")
    if not strict and not allow_backfill:
        allow_backfill = True

    grouped: dict[str, list[SessionRecord]] = defaultdict(list)
    atomic_count = 0
    for record in records:
        if is_atomic_seed(record):
            atomic_count += 1
            continue
        grouped[_record_bucket(record, ratios)].append(record)
    for bucket in ratios:
        grouped[bucket].sort(key=lambda item: _rank(item, seed))

    allocations = _allocate(total, ratios)
    shortages = {
        bucket: {"requested": allocations[bucket], "available": len(grouped[bucket])}
        for bucket in ratios
        if len(grouped[bucket]) < allocations[bucket]
    }
    if shortages and not allow_backfill:
        detail = ", ".join(
            f"{bucket}={item['available']}/{item['requested']}"
            for bucket, item in sorted(shortages.items())
        )
        atomic_note = f"; excluded {atomic_count} atomic seed(s)" if atomic_count else ""
        raise ValueError(f"training quota cannot be satisfied: {detail}{atomic_note}")

    selected: dict[str, list[SessionRecord]] = {
        bucket: grouped[bucket][: allocations[bucket]] for bucket in ratios
    }
    remaining = total - sum(len(items) for items in selected.values())
    if remaining > 0:
        if not allow_backfill:
            raise ValueError(f"training quota selected {total - remaining}/{total} sessions")
        candidates = [
            record for bucket in ratios for record in grouped[bucket][allocations[bucket] :]
        ]
        candidates.sort(key=lambda item: _rank(item, seed + 1))
        for record in candidates:
            if remaining == 0:
                break
            selected[_record_bucket(record, ratios)].append(record)
            remaining -= 1

    result = [record for bucket in ratios for record in selected[bucket]]
    result.sort(key=lambda item: _rank(item, seed))
    if not allow_backfill and drift_tolerance > 0:
        realized = {bucket: len(selected[bucket]) for bucket in ratios}
        drift = {bucket: abs(realized[bucket] - allocations[bucket]) / total for bucket in ratios}
        if any(value > drift_tolerance for value in drift.values()):
            detail = ", ".join(
                f"{bucket}={drift[bucket]:.4f}/{drift_tolerance:.4f}" for bucket in ratios
            )
            raise ValueError(f"training quota drift exceeds tolerance: {detail}")
    return result

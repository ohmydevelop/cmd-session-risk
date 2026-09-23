"""Composable data preparation pipeline and provenance summary."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from cmdrisk.schema import SessionRecord, SplitName

from .augmentation import generate_deterministic_stress
from .dedup import exact_deduplicate, family_aware_deduplicate
from .holdout import HOLDOUT_PERIOD, OOD_SLICE_NAME, is_temporal_holdout
from .normalization import normalize_record, session_skeleton
from .sampling import DEFAULT_QUOTA, allocate_quota, sample_training_quota, source_bucket
from .splitting import DEFAULT_SPLIT_RATIOS, audit_split_leakage, family_aware_split


def prepare_sessions_with_stats(
    records: Iterable[SessionRecord],
    *,
    total: int = 60_000,
    quota: Mapping[str, float] | None = None,
    split_ratios: Mapping[str, float] | None = None,
    jaccard_threshold: float = 0.9,
    max_family_fraction: float = 0.0025,
    minimum_family_cap: int = 50,
    seed: int = 42,
    strict_quota: bool = True,
    allow_quota_backfill: bool = False,
    quota_drift_tolerance: float = 0.0,
    balance_tolerance: float = 0.10,
    enforce_balance: bool = True,
) -> tuple[list[SessionRecord], dict[str, int]]:
    raw = list(records)
    normalized = [normalize_record(record) for record in raw]
    exact = exact_deduplicate(normalized)
    deduplicated = family_aware_deduplicate(
        exact,
        threshold=jaccard_threshold,
        max_fraction=max_family_fraction,
        minimum_family_cap=minimum_family_cap,
        seed=seed,
    )
    sampled = sample_training_quota(
        deduplicated,
        total=total,
        quota=quota or DEFAULT_QUOTA,
        seed=seed,
        strict=strict_quota,
        allow_backfill=allow_quota_backfill,
        drift_tolerance=quota_drift_tolerance,
    )
    prepared = family_aware_split(
        sampled,
        ratios=split_ratios or DEFAULT_SPLIT_RATIOS,
        seed=seed,
        balance_tolerance=balance_tolerance,
        enforce_balance=enforce_balance,
    )
    return prepared, {
        "input_count": len(raw),
        "normalized_count": len(normalized),
        "exact_count": len(exact),
        "dedup_count": len(deduplicated),
        "sampled_count": len(sampled),
        "output_count": len(prepared),
    }


def prepare_sessions(
    records: Iterable[SessionRecord],
    *,
    total: int = 60_000,
    quota: Mapping[str, float] | None = None,
    split_ratios: Mapping[str, float] | None = None,
    jaccard_threshold: float = 0.9,
    max_family_fraction: float = 0.0025,
    minimum_family_cap: int = 50,
    seed: int = 42,
    strict_quota: bool = True,
    allow_quota_backfill: bool = False,
    quota_drift_tolerance: float = 0.0,
    balance_tolerance: float = 0.10,
    enforce_balance: bool = True,
) -> list[SessionRecord]:
    prepared, _ = prepare_sessions_with_stats(
        records,
        total=total,
        quota=quota,
        split_ratios=split_ratios,
        jaccard_threshold=jaccard_threshold,
        max_family_fraction=max_family_fraction,
        minimum_family_cap=minimum_family_cap,
        seed=seed,
        strict_quota=strict_quota,
        allow_quota_backfill=allow_quota_backfill,
        quota_drift_tolerance=quota_drift_tolerance,
        balance_tolerance=balance_tolerance,
        enforce_balance=enforce_balance,
    )
    return prepared


def prepare_sessions_v2_with_stats(
    records: Iterable[SessionRecord],
    *,
    total: int = 5_000,
    quota: Mapping[str, float] | None = None,
    split_ratios: Mapping[str, float] | None = None,
    jaccard_threshold: float = 0.9,
    max_family_fraction: float = 0.0025,
    minimum_family_cap: int = 50,
    seed: int = 42,
    holdout_period: str = HOLDOUT_PERIOD,
    balance_tolerance: float = 0.10,
    enforce_balance: bool = False,
) -> tuple[list[SessionRecord], dict[str, object]]:
    """Build the v2 corpus with a temporal holdout and deterministic stress rows.

    Pipeline order is fixed and fail-closed:

    1. normalize
    2. partition the temporal holdout (never enters ``train``/``dev``)
    3. exact + family-aware near-duplicate dedup on each partition
    4. deterministic attack variants and hard negatives, generated **only**
       from non-holdout rows
    5. strict ``40/40/15/5`` quota sampling (shortages raise)
    6. family-aware ``train``/``dev``/``test`` split of the sampled corpus
    7. append the holdout to ``test`` and audit cross-split lineage
    """

    ratios = dict(quota or DEFAULT_QUOTA)
    if not ratios:
        raise ValueError("quota must contain at least one bucket")
    allocations = allocate_quota(total, ratios)
    for bucket in ("llm_variant", "hard_negative"):
        if bucket not in ratios:
            allocations.setdefault(bucket, 0)

    raw = list(records)
    normalized = [normalize_record(record) for record in raw]
    holdout_flags = [is_temporal_holdout(record, period=holdout_period) for record in normalized]
    base_raw = [
        record for record, flag in zip(normalized, holdout_flags, strict=True) if not flag
    ]
    holdout_raw = [
        record for record, flag in zip(normalized, holdout_flags, strict=True) if flag
    ]

    def _dedup(values: list[SessionRecord]) -> tuple[list[SessionRecord], list[SessionRecord]]:
        exact = exact_deduplicate(values)
        deduplicated = family_aware_deduplicate(
            exact,
            threshold=jaccard_threshold,
            max_fraction=max_family_fraction,
            minimum_family_cap=minimum_family_cap,
            seed=seed,
        )
        return exact, deduplicated

    base_exact, base_dedup = _dedup(base_raw)
    holdout_exact, holdout_dedup = _dedup(holdout_raw)

    stress = generate_deterministic_stress(
        base_dedup,
        malicious_count=int(allocations.get("llm_variant", 0)),
        hard_negative_count=int(allocations.get("hard_negative", 0)),
        seed=seed,
    )
    pool = [*base_dedup, *stress]
    sampled = sample_training_quota(
        pool,
        total=total,
        quota=ratios,
        seed=seed,
        strict=True,
        allow_backfill=False,
        drift_tolerance=0.0,
    )
    prepared = family_aware_split(
        sampled,
        ratios=split_ratios or DEFAULT_SPLIT_RATIOS,
        seed=seed,
        balance_tolerance=balance_tolerance,
        enforce_balance=enforce_balance,
    )

    base_skeletons = {session_skeleton(record) for record in prepared}
    holdout_records: list[SessionRecord] = []
    holdout_skeleton_overlap = 0
    for record in holdout_dedup:
        if session_skeleton(record) in base_skeletons:
            holdout_skeleton_overlap += 1
        meta = dict(record.meta)
        meta["evaluation_slice"] = OOD_SLICE_NAME
        holdout_records.append(record.model_copy(update={"split": SplitName.TEST, "meta": meta}))

    output = [*prepared, *holdout_records]
    audit_split_leakage(output, check_skeletons=False)

    stats: dict[str, object] = {
        "input_count": len(raw),
        "normalized_count": len(normalized),
        "base_input_count": len(base_raw),
        "holdout_input_count": len(holdout_raw),
        "exact_count": len(base_exact),
        "dedup_count": len(base_dedup),
        "holdout_exact_count": len(holdout_exact),
        "holdout_dedup_count": len(holdout_dedup),
        "stress_generated": len(stress),
        "stress_attack_variants": int(allocations.get("llm_variant", 0)),
        "stress_hard_negatives": int(allocations.get("hard_negative", 0)),
        "sampled_count": len(sampled),
        "holdout_count": len(holdout_records),
        "holdout_skeleton_overlap": holdout_skeleton_overlap,
        "quota_allocations": allocations,
        "output_count": len(output),
    }
    return output, stats


def prepare_sessions_v2(records: Iterable[SessionRecord], **kwargs: object) -> list[SessionRecord]:
    """Return only the prepared v2 sessions for ``records``."""

    prepared, _ = prepare_sessions_v2_with_stats(records, **kwargs)  # type: ignore[arg-type]
    return prepared


def _ratio_map(counter: Mapping[str, int], total: int) -> dict[str, float]:
    if total <= 0:
        return {key: 0.0 for key in sorted(counter)}
    return {key: round(value / total, 6) for key, value in sorted(counter.items())}


def _split_checksum(records: Iterable[SessionRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.session_id):
        payload = {
            "session_id": record.session_id,
            "family_id": record.family_id,
            "commands": [event.model_dump(mode="json") for event in record.commands],
            "source": record.source.model_dump(mode="json"),
            "label": record.label.model_dump(mode="json"),
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def provenance_report(
    records: Iterable[SessionRecord],
    *,
    requested_quota: Mapping[str, float] | None = None,
    input_count: int | None = None,
    exact_count: int | None = None,
    dedup_count: int | None = None,
    split_hashes: Mapping[str, str] | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    values = list(records)
    by_kind = Counter(str(record.source.kind) for record in values)
    by_source = Counter(record.source.name for record in values)
    by_license = Counter(record.source.license for record in values)
    by_split = Counter(
        str(record.split) if record.split is not None else "unassigned" for record in values
    )
    by_label = Counter("malicious" if record.label.malicious else "benign" for record in values)
    by_family = Counter(record.family_id for record in values)
    by_family_size = Counter(by_family.values())

    cross_tab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_tab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    label_tab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in values:
        split = str(record.split) if record.split is not None else "unassigned"
        kind = str(record.source.kind)
        label = "malicious" if record.label.malicious else "benign"
        cross_tab[split][kind] += 1
        source_tab[split][record.source.name] += 1
        label_tab[split][label] += 1

    lengths = np.asarray([len(record.commands) for record in values], dtype=np.int64)
    command_lengths: dict[str, float | int | None] = {"min": None, "max": None}
    if lengths.size:
        command_lengths = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 3),
            "p50": float(np.quantile(lengths, 0.5)),
            "p95": float(np.quantile(lengths, 0.95)),
        }

    total_commands = sum(len(record.commands) for record in values)
    missing_dt = sum(
        event.dt is None for record in values for event in record.commands
    )
    quota_requested = dict(requested_quota or DEFAULT_QUOTA)
    quota_realized_counts = Counter(source_bucket(record) for record in values)
    quota_realized = _ratio_map(quota_realized_counts, len(values))
    quota_requested_counts = {
        bucket: round(ratio * len(values)) for bucket, ratio in quota_requested.items()
    }
    quota_drift = {
        bucket: round(quota_realized.get(bucket, 0.0) - float(ratio), 6)
        for bucket, ratio in quota_requested.items()
    }

    if split_hashes is None:
        computed_hashes = {
            split: _split_checksum(
                record
                for record in values
                if record.split is not None and str(record.split) == split
            )
            for split in sorted(by_split)
        }
    else:
        computed_hashes = {str(key): str(value) for key, value in split_hashes.items()}

    raw_count = len(values) if input_count is None else input_count
    exact_total = raw_count if exact_count is None else exact_count
    dedup_total = exact_total if dedup_count is None else dedup_count
    deduplication = {
        "input_count": raw_count,
        "exact_count": exact_total,
        "dedup_count": dedup_total,
        "output_count": len(values),
        "exact_removed": max(raw_count - exact_total, 0),
        "family_removed": max(exact_total - dedup_total, 0),
        "quota_or_filter_removed": max(dedup_total - len(values), 0),
        "exact_retention": round(exact_total / raw_count, 6) if raw_count else 0.0,
        "family_retention": round(dedup_total / exact_total, 6) if exact_total else 0.0,
        "final_retention": round(len(values) / raw_count, 6) if raw_count else 0.0,
    }

    slices: dict[str, int] = defaultdict(int)
    for record in values:
        declared = record.meta.get("evaluation_slice")
        if declared:
            slices[str(declared)] += 1

    report = {
        "schema_version": "1.0",
        "total_sessions": len(values),
        "total_commands": int(lengths.sum()) if lengths.size else 0,
        "by_kind": dict(sorted(by_kind.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_license": dict(sorted(by_license.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_label": dict(sorted(by_label.items())),
        "split_by_kind": {
            key: dict(sorted(value.items())) for key, value in sorted(cross_tab.items())
        },
        "split_by_source": {
            key: dict(sorted(value.items())) for key, value in sorted(source_tab.items())
        },
        "split_by_label": {
            key: dict(sorted(value.items())) for key, value in sorted(label_tab.items())
        },
        "split_checksums": computed_hashes,
        "quota": {
            "requested": quota_requested,
            "requested_counts": quota_requested_counts,
            "realized": quota_realized,
            "realized_counts": dict(sorted(quota_realized_counts.items())),
            "drift": quota_drift,
        },
        "deduplication": deduplication,
        "missing_dt": missing_dt,
        "missing_dt_rate": round(missing_dt / total_commands, 6) if total_commands else 0.0,
        "family_count": len(by_family),
        "largest_family_size": max(by_family.values(), default=0),
        "family_size_distribution": {
            str(size): count for size, count in sorted(by_family_size.items())
        },
        "class_distribution": _ratio_map(by_label, len(values)),
        "source_distribution": _ratio_map(by_source, len(values)),
        "kind_distribution": _ratio_map(by_kind, len(values)),
        "command_length": command_lengths,
        "evaluation_slices": dict(sorted(slices.items())),
    }
    if extra:
        report.update({str(key): value for key, value in extra.items()})
    return report

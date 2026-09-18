"""Composable data preparation pipeline and provenance summary."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from cmdrisk.schema import SessionRecord

from .dedup import exact_deduplicate, family_aware_deduplicate
from .normalization import normalize_record
from .sampling import DEFAULT_QUOTA, sample_training_quota
from .splitting import DEFAULT_SPLIT_RATIOS, family_aware_split


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
) -> list[SessionRecord]:
    normalized = [normalize_record(record) for record in records]
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
    )
    return family_aware_split(
        sampled,
        ratios=split_ratios or DEFAULT_SPLIT_RATIOS,
        seed=seed,
    )


def provenance_report(records: Iterable[SessionRecord]) -> dict[str, object]:
    values = list(records)
    by_kind = Counter(str(record.source.kind) for record in values)
    by_source = Counter(record.source.name for record in values)
    by_split = Counter(
        str(record.split) if record.split is not None else "unassigned" for record in values
    )
    by_label = Counter("malicious" if record.label.malicious else "benign" for record in values)
    by_family = Counter(record.family_id for record in values)

    cross_tab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in values:
        split = str(record.split) if record.split is not None else "unassigned"
        cross_tab[split][str(record.source.kind)] += 1

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

    return {
        "total_sessions": len(values),
        "total_commands": int(lengths.sum()) if lengths.size else 0,
        "by_kind": dict(sorted(by_kind.items())),
        "by_source": dict(sorted(by_source.items())),
        "by_split": dict(sorted(by_split.items())),
        "by_label": dict(sorted(by_label.items())),
        "split_by_kind": {
            key: dict(sorted(value.items())) for key, value in sorted(cross_tab.items())
        },
        "family_count": len(by_family),
        "largest_family_size": max(by_family.values(), default=0),
        "command_length": command_lengths,
    }

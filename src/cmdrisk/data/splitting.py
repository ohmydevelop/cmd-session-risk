"""Family-aware deterministic train/dev/test splitting."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from cmdrisk.schema import SessionRecord, SplitName

from .normalization import session_skeleton

DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.8, "dev": 0.1, "test": 0.1}
_SPLITS = ("train", "dev", "test")
_CAMPAIGN_KEYS = {"campaign", "campaign_id", "campaign_name", "attack_campaign"}
_NEAR_DUPLICATE_KEYS = {
    "near_duplicate_cluster",
    "near_duplicate_cluster_id",
    "duplicate_cluster",
    "duplicate_cluster_id",
}


class SplitLeakageError(ValueError):
    """Raised when related records cross train/dev/test boundaries."""


class SplitBalanceError(ValueError):
    """Raised when a split materially unbalances protected data dimensions."""


def _family_rank(family_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{family_id}".encode()).hexdigest()


def _metadata_values(meta: Mapping[str, Any], keys: set[str]) -> set[str]:
    values: set[str] = set()
    stack: list[Mapping[str, Any]] = [meta]
    while stack:
        current = stack.pop()
        for key, value in current.items():
            normalized = str(key).casefold()
            if normalized in keys and value is not None:
                if isinstance(value, (list, tuple, set)):
                    values.update(str(item) for item in value if item is not None)
                else:
                    values.add(str(value))
            if isinstance(value, Mapping):
                stack.append(value)
    return values


def _split_name(record: SessionRecord) -> str:
    if record.split is None:
        raise SplitLeakageError(f"record {record.session_id!r} has no split assignment")
    split = str(record.split)
    if split not in _SPLITS:
        raise SplitLeakageError(f"record {record.session_id!r} has unsupported split {split!r}")
    return split


def _assert_single_owner(
    *,
    owner_splits: Mapping[str, set[str]],
    label: str,
    error_type: type[ValueError] = SplitLeakageError,
) -> None:
    leaked = {key: sorted(splits) for key, splits in owner_splits.items() if len(splits) > 1}
    if leaked:
        detail = "; ".join(f"{key}: {','.join(splits)}" for key, splits in sorted(leaked.items()))
        raise error_type(f"{label} crosses splits: {detail}")


def audit_split_leakage(
    records: Iterable[SessionRecord],
    *,
    check_skeletons: bool = True,
) -> dict[str, int]:
    """Fail if any known relationship crosses train/dev/test boundaries.

    ``check_skeletons=False`` keeps family, lineage, campaign, and near-duplicate
    cluster gates active while allowing intentionally disjoint evaluation
    periods (for example a temporal OOD holdout) to reuse command shapes that
    also occur in training.  The overlap is reported by the caller instead.
    """

    values = list(records)
    if not values:
        return {
            "families": 0,
            "parent_links": 0,
            "campaigns": 0,
            "near_duplicate_clusters": 0,
            "ordered_skeletons": 0,
        }

    family_splits: dict[str, set[str]] = defaultdict(set)
    parent_splits: dict[str, set[str]] = defaultdict(set)
    campaign_splits: dict[str, set[str]] = defaultdict(set)
    cluster_splits: dict[str, set[str]] = defaultdict(set)
    skeleton_splits: dict[str, set[str]] = defaultdict(set)
    session_splits = {record.session_id: _split_name(record) for record in values}
    parent_links = 0
    campaign_count = 0
    cluster_count = 0
    skeleton_count = 0

    for record in values:
        split = session_splits[record.session_id]
        family_splits[record.family_id].add(split)
        skeleton = "\x1f".join(session_skeleton(record))
        skeleton_splits[skeleton].add(split)
        skeleton_count += 1

        parent_id = record.source.parent_session_id
        if parent_id is not None:
            parent_links += 1
            parent_splits[parent_id].add(split)
            if parent_id in session_splits:
                parent_splits[parent_id].add(session_splits[parent_id])

        campaigns = _metadata_values(record.meta, _CAMPAIGN_KEYS)
        for campaign in campaigns:
            campaign_splits[campaign].add(split)
        campaign_count += len(campaigns)

        clusters = _metadata_values(record.meta, _NEAR_DUPLICATE_KEYS)
        for cluster in clusters:
            cluster_splits[cluster].add(split)
        cluster_count += len(clusters)

    _assert_single_owner(owner_splits=family_splits, label="family")
    _assert_single_owner(owner_splits=parent_splits, label="parent/child lineage")
    _assert_single_owner(owner_splits=campaign_splits, label="campaign")
    _assert_single_owner(owner_splits=cluster_splits, label="near-duplicate output cluster")
    if check_skeletons:
        _assert_single_owner(owner_splits=skeleton_splits, label="normalized session")
    return {
        "families": len(family_splits),
        "parent_links": parent_links,
        "campaigns": campaign_count,
        "near_duplicate_clusters": cluster_count,
        "ordered_skeletons": skeleton_count,
    }


def _conditional_distribution_drift(
    values: list[SessionRecord],
    *,
    dimension: str,
    tolerance: float,
) -> tuple[float, dict[str, dict[str, float]]]:
    if dimension == "label":

        def key(record: SessionRecord) -> str:
            return str(record.label.malicious)
    elif dimension == "source":

        def key(record: SessionRecord) -> str:
            return str(record.source.kind)
    else:
        raise ValueError(f"unsupported balance dimension: {dimension}")

    global_counts = Counter(key(record) for record in values)
    total = len(values)
    by_split: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in values:
        by_split[_split_name(record)].append(record)

    report: dict[str, dict[str, float]] = {}
    maximum = 0.0
    for split in _SPLITS:
        split_values = by_split[split]
        split_total = len(split_values)
        split_counts = Counter(key(record) for record in split_values)
        drift = {
            category: abs(
                (split_counts[category] / split_total if split_total else 0.0) - count / total
            )
            for category, count in global_counts.items()
        }
        report[split] = drift
        maximum = max(maximum, max(drift.values(), default=0.0))
    if maximum > tolerance:
        detail = ", ".join(
            f"{split}:{category}={value:.4f}"
            for split, values_for_split in report.items()
            for category, value in sorted(values_for_split.items())
            if value > tolerance
        )
        raise SplitBalanceError(f"{dimension} balance drift exceeds {tolerance:.4f}: {detail}")
    return maximum, report


def audit_split_balance(
    records: Iterable[SessionRecord],
    *,
    ratios: Mapping[str, float] | None = None,
    tolerance: float = 0.10,
) -> dict[str, object]:
    """Validate split size, label, source, and family distribution drift."""

    if tolerance < 0 or tolerance > 1:
        raise ValueError("balance tolerance must be in [0, 1]")
    split_ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    values = list(records)
    if not values:
        return {"maximum_drift": 0.0, "label": {}, "source": {}, "family": {}}

    by_split: dict[str, list[SessionRecord]] = defaultdict(list)
    for record in values:
        by_split[_split_name(record)].append(record)

    size_drift = {
        split: abs(len(by_split[split]) / len(values) - split_ratios[split]) for split in _SPLITS
    }
    if any(value > tolerance for value in size_drift.values()):
        detail = ", ".join(
            f"{split}={value:.4f}" for split, value in size_drift.items() if value > tolerance
        )
        raise SplitBalanceError(f"split-size drift exceeds {tolerance:.4f}: {detail}")

    label_drift, label_report = _conditional_distribution_drift(
        values, dimension="label", tolerance=tolerance
    )
    source_drift, source_report = _conditional_distribution_drift(
        values, dimension="source", tolerance=tolerance
    )

    global_families = {record.family_id for record in values}
    family_count_drift = {
        split: abs(
            len({record.family_id for record in by_split[split]}) / len(global_families)
            - split_ratios[split]
        )
        for split in _SPLITS
    }
    family_size_counts = Counter(Counter(record.family_id for record in values).values())
    family_size_total = sum(family_size_counts.values())
    family_size_report: dict[str, dict[str, float]] = {}
    family_size_drift = 0.0
    for split in _SPLITS:
        split_sizes = Counter(Counter(record.family_id for record in by_split[split]).values())
        split_total = sum(split_sizes.values())
        drift = {
            str(size): abs(
                (split_sizes[size] / split_total if split_total else 0.0)
                - count / family_size_total
            )
            for size, count in family_size_counts.items()
        }
        family_size_report[split] = drift
        family_size_drift = max(family_size_drift, max(drift.values(), default=0.0))
    maximum_family_drift = max(max(family_count_drift.values(), default=0.0), family_size_drift)
    if maximum_family_drift > tolerance:
        details = ", ".join(f"{split}={family_count_drift[split]:.4f}" for split in _SPLITS)
        raise SplitBalanceError(
            "family balance drift exceeds "
            f"{tolerance:.4f}: family-count {details}; family-size={family_size_drift:.4f}"
        )

    return {
        "maximum_drift": max(
            max(size_drift.values(), default=0.0),
            label_drift,
            source_drift,
            maximum_family_drift,
        ),
        "split_size": size_drift,
        "label": label_report,
        "source": source_report,
        "family": {
            "count": family_count_drift,
            "size": family_size_report,
        },
    }


def _skeleton_key(record: SessionRecord) -> str:
    return "\x1f".join(session_skeleton(record))


def union_identical_skeleton_families(
    records: Iterable[SessionRecord],
    *,
    seed: int = 42,
) -> tuple[list[SessionRecord], int]:
    """Merge families that contain an identical normalized-skeleton session.

    ``family_aware_split`` assigns whole families to one split and then audits
    that no normalized skeleton crosses a split boundary.  Deduplication
    already unions byte-identical shingle sets, but this pass makes the audit
    guarantee true *by construction* for the exact criterion the audit checks:
    any two sessions whose full masked skeleton is identical are placed in the
    same family before a split is chosen.

    Returns the (possibly re-labelled) records plus the number of merges.
    """

    values = list(records)
    if not values:
        return [], 0

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        root = node
        while parent.get(root, root) != root:
            root = parent[root]
        while node != root:
            nxt = parent.get(node, root)
            parent[node] = root
            node = nxt
        return root

    def union(left: str, right: str) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        # Deterministic, order-independent choice of the surviving id.
        keep, drop = sorted((left_root, right_root))
        parent[drop] = keep
        return True

    by_skeleton: dict[str, set[str]] = defaultdict(set)
    for record in values:
        by_skeleton[_skeleton_key(record)].add(record.family_id)

    merges = 0
    for families in by_skeleton.values():
        if len(families) < 2:
            continue
        ordered = sorted(families)
        for family_id in ordered[1:]:
            if union(ordered[0], family_id):
                merges += 1

    if not merges:
        return values, 0
    return [
        record.model_copy(update={"family_id": find(record.family_id)}) for record in values
    ], merges


def family_aware_split(
    records: Iterable[SessionRecord],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
    balance_tolerance: float = 0.10,
    enforce_balance: bool = True,
) -> list[SessionRecord]:
    """Assign complete families, then enforce leakage and balance gates."""

    split_ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    if set(split_ratios) != {"train", "dev", "test"}:
        raise ValueError("split ratios must contain train, dev, and test")
    if (
        any(value < 0 for value in split_ratios.values())
        or not abs(sum(split_ratios.values()) - 1.0) < 1e-9
    ):
        raise ValueError("split ratios must be non-negative and sum to 1.0")

    values, _merged = union_identical_skeleton_families(records, seed=seed)
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
        split = max(
            split_ratios, key=lambda name: (priorities[name], -list(split_ratios).index(name))
        )
        assignments[family_id] = split
        assigned_counts[split] += size

    output: list[SessionRecord] = []
    for record in values:
        split = SplitName(assignments[record.family_id])
        output.append(record.model_copy(update={"split": split}))
    audit_split_leakage(output)
    if enforce_balance:
        audit_split_balance(output, ratios=split_ratios, tolerance=balance_tolerance)
    return output

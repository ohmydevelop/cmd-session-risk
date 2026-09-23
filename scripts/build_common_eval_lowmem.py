#!/usr/bin/env python3
"""Rebuild a family-disjoint common evaluation pool with bounded memory.

This is the low-memory counterpart of :mod:`scripts.build_common_eval`.  It
keeps only the fields needed for scoring and family assignment, streams all
input records, and uses a sparse exact Jaccard scan for near-duplicate
clustering.  The latter is important for command corpora where a few very
common shingles occur in thousands of sessions: the old inverted-index pair
set can grow to tens of gigabytes even though the final number of families is
small.

The output contract is intentionally the same as ``build_common_eval.py``:

* ``common_clean.jsonl`` contains family-disjoint base sessions;
* ``slices/ood_2024.jsonl`` contains the temporal holdout;
* neither output shares a family or session with any supplied v2/v3
  train/dev split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.sparse import csr_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data import adapter_for_kind  # noqa: E402
from cmdrisk.data.dedup import cap_families, exact_deduplicate  # noqa: E402
from cmdrisk.data.holdout import HOLDOUT_PERIOD, OOD_SLICE_NAME, is_temporal_holdout  # noqa: E402
from cmdrisk.data.io import write_jsonl  # noqa: E402
from cmdrisk.data.manifest import (  # noqa: E402
    load_data_sources,
    sha256_file,
    write_json_atomic,
    write_sha256_manifest,
)
from cmdrisk.data.normalization import normalize_record, session_skeleton  # noqa: E402
from cmdrisk.schema import SessionRecord, SplitName  # noqa: E402


def _parse_source_arguments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--source must be ID=PATH, got: {value}")
        source_id, raw_path = value.split("=", 1)
        if not source_id or not raw_path:
            raise SystemExit(f"--source must be ID=PATH, got: {value}")
        result[source_id] = Path(raw_path).expanduser().resolve()
    return result


def _parse_source_limits(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--source-limit must be ID=N, got: {value}")
        source_id, raw_limit = value.split("=", 1)
        try:
            limit = int(raw_limit)
        except ValueError as exc:
            raise SystemExit(f"--source-limit must use an integer, got: {value}") from exc
        if limit < 1:
            raise SystemExit(f"--source-limit must be positive, got: {value}")
        if source_id in result:
            raise SystemExit(f"duplicate --source-limit for: {source_id}")
        result[source_id] = limit
    return result


def _input_fingerprint(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: str(item)):
        entries = sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        for entry in entries:
            relative = entry.relative_to(path).as_posix() if path.is_dir() else path.name
            digest.update(f"{relative}\0".encode())
            with entry.open("rb") as handle:
                while chunk := handle.read(4 * 1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _load_used_lines(exclude_dirs: list[Path]) -> tuple[set[str], set[str], dict[str, Any]]:
    """Stream exclude split files and collect both family and session IDs.

    Family IDs are context-dependent because near-duplicate connected components
    can change when a corpus is rebuilt.  Excluding only family IDs is therefore
    insufficient: a session reused at train/dev time can receive a different
    family ID in an independently clustered evaluation pool.  The caller must
    exclude the exact session ID as well.
    """

    used_families: set[str] = set()
    used_sessions: set[str] = set()
    audit: dict[str, Any] = {}
    for directory in exclude_dirs:
        split_audit: dict[str, Any] = {}
        for split in ("train", "dev"):
            path = directory / f"{split}.jsonl"
            if not path.is_file():
                continue
            sessions = 0
            families: set[str] = set()
            split_sessions: set[str] = set()
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    family_id = payload.get("family_id")
                    session_id = payload.get("session_id")
                    if not isinstance(family_id, str) or not family_id:
                        raise RuntimeError(f"family_id missing in {path}")
                    if not isinstance(session_id, str) or not session_id:
                        raise RuntimeError(f"session_id missing in {path}")
                    families.add(family_id)
                    split_sessions.add(session_id)
                    sessions += 1
            used_families.update(families)
            used_sessions.update(split_sessions)
            split_audit[split] = {
                "path": str(path),
                "sessions": sessions,
                "unique_sessions": len(split_sessions),
                "families": len(families),
                "sha256": sha256_file(path),
            }
        audit[str(directory)] = split_audit
    return used_families, used_sessions, audit


def _compact(record: SessionRecord) -> SessionRecord:
    """Drop raw commands and redundant provenance after holdout detection.

    ``normalize_record`` deliberately preserves the original commands for
    audit.  The common-evaluation pool only needs normalized commands, labels,
    source lineage, and a small set of slice metadata, so retaining that full
    history would unnecessarily multiply memory by several times.
    """

    normalized = normalize_record(record)
    return normalized.model_copy(update={"meta": {}})


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
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


def _assign_families_sparse(
    records: list[SessionRecord],
    *,
    threshold: float = 0.9,
    chunk_size: int = 96,
) -> list[SessionRecord]:
    """Exact Jaccard connected components without a quadratic pair set.

    Sparse matrix multiplication computes the exact shingle intersections for
    each row block.  Each unordered pair is visited once (``column > row``),
    then filtered with the same size-ratio and Jaccard rules as
    ``cmdrisk.data.dedup.assign_families``.  This changes neither the edge
    predicate nor the resulting connected components, but keeps memory bounded
    when a shingle occurs in thousands of sessions.
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if not records:
        return []
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    values = list(records)
    n = len(values)
    union_find = _UnionFind(n)

    # Explicit lineage and parent/child relationships have the same union
    # semantics as the reference implementation.
    explicit_families: dict[str, list[int]] = defaultdict(list)
    parent_groups: dict[str, list[int]] = defaultdict(list)
    sessions: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(values):
        family_id = record.family_id
        if family_id != "pending" and not family_id.startswith("pending:"):
            explicit_families[family_id].append(index)
        if record.source.parent_session_id is not None:
            parent_groups[record.source.parent_session_id].append(index)
        sessions[record.session_id].append(index)
    for indices in explicit_families.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)
    for parent_id, children in parent_groups.items():
        for child in children[1:]:
            union_find.union(children[0], child)
        for parent in sessions.get(parent_id, []):
            union_find.union(parent, children[0])

    shingle_sets = [frozenset(_shingles(session_skeleton(record))) for record in values]
    signature_groups: dict[frozenset[str], list[int]] = defaultdict(list)
    for index, shingles in enumerate(shingle_sets):
        signature_groups[shingles].append(index)
    signatures = list(signature_groups)
    signature_sizes = np.asarray([len(item) for item in signatures], dtype=np.int32)

    # Identical skeleton shingle sets are the same family and can be collapsed
    # before the pairwise scan.
    for indices in signature_groups.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)

    token_ids: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for row, shingles in enumerate(signatures):
        for token in shingles:
            column = token_ids.get(token)
            if column is None:
                column = len(token_ids)
                token_ids[token] = column
            rows.append(row)
            cols.append(column)
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(signatures), len(token_ids)),
        dtype=np.int32,
    )

    for start in range(0, len(signatures), chunk_size):
        stop = min(start + chunk_size, len(signatures))
        products = matrix[start:stop].dot(matrix.T).tocoo()
        for row_offset, column, intersection in zip(
            products.row, products.col, products.data, strict=True
        ):
            left = int(row_offset) + start
            right = int(column)
            if right <= left:
                continue
            left_size = int(signature_sizes[left])
            right_size = int(signature_sizes[right])
            smaller = min(left_size, right_size)
            larger = max(left_size, right_size)
            if larger == 0 or smaller / larger < threshold:
                continue
            union = left_size + right_size - int(intersection)
            if union and int(intersection) / union >= threshold:
                union_find.union(left, right)

    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(len(signatures)):
        clusters[union_find.find(index)].append(index)
    family_for_root: dict[int, str] = {}
    for root, signature_indices in clusters.items():
        explicit_ids = sorted(
            {
                values[index].family_id
                for signature_index in signature_indices
                for index in signature_groups[signatures[signature_index]]
                if values[index].family_id != "pending"
                and not values[index].family_id.startswith("pending:")
            }
        )
        if len(explicit_ids) > 1:
            raise ValueError(
                "near-duplicate or lineage cluster contains conflicting explicit "
                f"family IDs: {', '.join(explicit_ids)}"
            )
        member_ids: list[str] = []
        for signature_index in signature_indices:
            member_ids.extend(values[index].session_id for index in signature_groups[signatures[signature_index]])
        digest = hashlib.sha256("\x1f".join(sorted(member_ids)).encode("utf-8")).hexdigest()[:24]
        family_for_root[root] = explicit_ids[0] if explicit_ids else f"family:{digest}"

    return [
        record.model_copy(update={"family_id": family_for_root[union_find.find(_signature_index_for_record(index, signature_groups, shingle_sets))]})
        for index, record in enumerate(values)
    ]


def _signature_index_for_record(
    record_index: int, signature_groups: dict[frozenset[str], list[int]], shingle_sets: list[frozenset[str]]
) -> int:
    # This helper is replaced below by an inverse map in production; keeping it
    # isolated makes the main assignment code easier to audit.
    raise AssertionError("internal helper must be overridden")


def _shingles(commands: tuple[str, ...], width: int = 3) -> set[str]:
    if len(commands) < width:
        return {f"c:{command}" for command in commands}
    return {
        "s:" + "\x1f".join(commands[index : index + width])
        for index in range(len(commands) - width + 1)
    }


def _deduplicate(
    records: list[SessionRecord],
    *,
    threshold: float,
    max_family_fraction: float,
    minimum_family_cap: int,
    seed: int,
) -> list[SessionRecord]:
    exact = exact_deduplicate(records)
    # Build a direct record -> signature map so family IDs are assigned only
    # once and without reconstructing sets from the returned records.
    signature_groups: dict[frozenset[str], list[int]] = defaultdict(list)
    shingle_sets: list[frozenset[str]] = []
    for index, record in enumerate(exact):
        shingles = frozenset(_shingles(session_skeleton(record)))
        shingle_sets.append(shingles)
        signature_groups[shingles].append(index)
    # The sparse helper above intentionally keeps this implementation local;
    # call a compact copy that receives the precomputed signature data.
    assigned = _assign_families_from_shingles(
        exact,
        shingle_sets,
        signature_groups,
        threshold=threshold,
    )
    return cap_families(
        assigned,
        max_fraction=max_family_fraction,
        minimum=minimum_family_cap,
        seed=seed,
    )


def _assign_families_from_shingles(
    values: list[SessionRecord],
    shingle_sets: list[frozenset[str]],
    signature_groups: dict[frozenset[str], list[int]],
    *,
    threshold: float,
    chunk_size: int = 96,
) -> list[SessionRecord]:
    n = len(values)
    union_find = _UnionFind(n)
    explicit_families: dict[str, list[int]] = defaultdict(list)
    parent_groups: dict[str, list[int]] = defaultdict(list)
    sessions: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(values):
        family_id = record.family_id
        if family_id != "pending" and not family_id.startswith("pending:"):
            explicit_families[family_id].append(index)
        if record.source.parent_session_id is not None:
            parent_groups[record.source.parent_session_id].append(index)
        sessions[record.session_id].append(index)
    for indices in explicit_families.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)
    for parent_id, children in parent_groups.items():
        for child in children[1:]:
            union_find.union(children[0], child)
        for parent in sessions.get(parent_id, []):
            union_find.union(parent, children[0])

    signatures = list(signature_groups)
    signature_sizes = np.asarray([len(item) for item in signatures], dtype=np.int32)
    for indices in signature_groups.values():
        for index in indices[1:]:
            union_find.union(indices[0], index)

    token_ids: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for row, shingles in enumerate(signatures):
        for token in shingles:
            column = token_ids.get(token)
            if column is None:
                column = len(token_ids)
                token_ids[token] = column
            rows.append(row)
            cols.append(column)
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(signatures), len(token_ids)),
        dtype=np.int32,
    )
    for start in range(0, len(signatures), chunk_size):
        stop = min(start + chunk_size, len(signatures))
        products = matrix[start:stop].dot(matrix.T).tocoo()
        for row_offset, column, intersection in zip(
            products.row, products.col, products.data, strict=True
        ):
            left = int(row_offset) + start
            right = int(column)
            if right <= left:
                continue
            left_size = int(signature_sizes[left])
            right_size = int(signature_sizes[right])
            smaller = min(left_size, right_size)
            larger = max(left_size, right_size)
            if larger == 0 or smaller / larger < threshold:
                continue
            union = left_size + right_size - int(intersection)
            if union and int(intersection) / union >= threshold:
                union_find.union(left, right)

    # Inverse map: record index -> signature row.
    signature_index: dict[int, int] = {}
    for row, shingles in enumerate(signatures):
        for index in signature_groups[shingles]:
            signature_index[index] = row
    clusters: dict[int, list[int]] = defaultdict(list)
    for row in range(len(signatures)):
        clusters[union_find.find(row)].append(row)
    family_for_root: dict[int, str] = {}
    for root, signature_indices in clusters.items():
        explicit_ids = sorted(
            {
                values[index].family_id
                for signature_row in signature_indices
                for index in signature_groups[signatures[signature_row]]
                if values[index].family_id != "pending"
                and not values[index].family_id.startswith("pending:")
            }
        )
        if len(explicit_ids) > 1:
            raise ValueError(
                "near-duplicate or lineage cluster contains conflicting explicit "
                f"family IDs: {', '.join(explicit_ids)}"
            )
        member_ids = sorted(
            values[index].session_id
            for signature_row in signature_indices
            for index in signature_groups[signatures[signature_row]]
        )
        digest = hashlib.sha256("\x1f".join(member_ids).encode("utf-8")).hexdigest()[:24]
        family_for_root[root] = explicit_ids[0] if explicit_ids else f"family:{digest}"
    return [
        record.model_copy(update={"family_id": family_for_root[union_find.find(signature_index[index])]})
        for index, record in enumerate(values)
    ]


def _summarize(records: list[SessionRecord]) -> dict[str, Any]:
    commands = [len(record.commands) for record in records]
    lengths = sorted(commands)
    return {
        "sessions": len(records),
        "families": len({record.family_id for record in records}),
        "labels": dict(sorted(Counter(int(record.label.malicious) for record in records).items())),
        "source_kinds": dict(sorted(Counter(str(record.source.kind) for record in records).items())),
        "sources": dict(sorted(Counter(record.source.name for record in records).items())),
        "command_count": {
            "min": min(commands, default=0),
            "max": max(commands, default=0),
            "mean": (sum(commands) / len(commands)) if commands else 0.0,
            "p50": float(lengths[len(lengths) // 2]) if lengths else 0.0,
            "p95": float(lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))]) if lengths else 0.0,
        },
    }


def _resolve_sources(
    args: argparse.Namespace,
    configured: dict[str, dict[str, Any]],
) -> tuple[dict[str, Path], dict[str, int], int | None]:
    paths = _parse_source_arguments(args.source)
    limits = _parse_source_limits(args.source_limit)
    default_limit: int | None = None
    if args.prepare_state is not None:
        state = json.loads(args.prepare_state.read_text(encoding="utf-8"))
        options = state.get("preparation_options", {})
        for item in options.get("sources", []):
            source_id = str(item["id"])
            if source_id not in paths:
                paths[source_id] = Path(str(item["path"])).expanduser().resolve()
        default_limit = options.get("max_records_per_source")
        for source_id, limit in (options.get("source_limits") or {}).items():
            limits.setdefault(str(source_id), int(limit))
        if default_limit is not None:
            default_limit = int(default_limit)
    aliases = {
        "microsoft": "microsoft_iot",
        "microsoft_iot": "microsoft_iot",
        "shell": "shell_attack_evolution",
        "shell_evolution": "shell_attack_evolution",
        "zenodo": "zenodo",
        "atomic": "atomic_red_team",
        "scripts": "script_repo",
    }
    adapter_sources: dict[str, dict[str, Any]] = {}
    for source in configured.values():
        adapter_sources.setdefault(str(source["adapter"]), source)
    for requested in list(paths):
        if requested not in configured and requested in aliases:
            configured[requested] = adapter_sources.get(aliases[requested], {})
    unknown = set(paths) - set(configured)
    if unknown:
        raise SystemExit(f"unknown source id(s): {', '.join(sorted(unknown))}")
    unknown_limits = set(limits) - set(paths)
    if unknown_limits:
        raise SystemExit(f"--source-limit references unknown source(s): {', '.join(sorted(unknown_limits))}")
    for source_id, path in paths.items():
        if not path.exists():
            raise SystemExit(f"source {source_id} does not exist: {path}")
    return paths, limits, default_limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data_sources.yaml")
    parser.add_argument("--prepare-state", type=Path)
    parser.add_argument("--source", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--source-limit", action="append", default=[], metavar="ID=N")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "common_eval")
    parser.add_argument("--exclude-dir", action="append", type=Path, default=[])
    parser.add_argument("--jaccard-threshold", type=float, default=0.9)
    parser.add_argument("--max-family-fraction", type=float, default=0.0025)
    parser.add_argument("--minimum-family-cap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-period", default=HOLDOUT_PERIOD)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exclude_dirs = list(args.exclude_dir) or [
        PROJECT_ROOT / "data" / "processed" / "v2",
        PROJECT_ROOT / "data" / "processed" / "v3",
    ]
    if args.dry_run:
        print(json.dumps({"output_dir": str(args.output_dir), "exclude_dirs": [str(p) for p in exclude_dirs]}, indent=2))
        return 0

    configured = {str(item["id"]): item for item in load_data_sources(args.config)}
    source_paths, source_limits, default_limit = _resolve_sources(args, configured)
    if not source_paths:
        raise SystemExit("at least one --source ID=PATH or --prepare-state with sources is required")
    used_families, used_sessions, exclusion_audit = _load_used_lines(exclude_dirs)

    base_raw: list[SessionRecord] = []
    holdout_raw: list[SessionRecord] = []
    per_source: dict[str, int] = {}
    for source_id, path in sorted(source_paths.items()):
        source = configured[source_id]
        adapter = adapter_for_kind(str(source["adapter"]))
        kwargs: dict[str, Any] = {}
        if source.get("password"):
            kwargs["password"] = str(source["password"])
        limit = source_limits.get(source_id, default_limit)
        count = 0
        for record in adapter.iter_records(path, **kwargs):
            if limit is not None and count >= int(limit):
                break
            count += 1
            if is_temporal_holdout(record, period=args.holdout_period):
                holdout_raw.append(_compact(record))
            else:
                base_raw.append(_compact(record))
        per_source[source_id] = count

    base_exact = exact_deduplicate(base_raw)
    holdout_exact = exact_deduplicate(holdout_raw)
    base_dedup = _deduplicate(
        base_exact,
        threshold=args.jaccard_threshold,
        max_family_fraction=args.max_family_fraction,
        minimum_family_cap=args.minimum_family_cap,
        seed=args.seed,
    )
    holdout_dedup = _deduplicate(
        holdout_exact,
        threshold=args.jaccard_threshold,
        max_family_fraction=args.max_family_fraction,
        minimum_family_cap=args.minimum_family_cap,
        seed=args.seed,
    )

    clean = [
        record
        for record in base_dedup
        if record.family_id not in used_families and record.session_id not in used_sessions
    ]
    ood = [
        record
        for record in holdout_dedup
        if record.family_id not in used_families and record.session_id not in used_sessions
    ]
    clean_families = {record.family_id for record in clean}
    ood = [record for record in ood if record.family_id not in clean_families]

    if {record.session_id for record in clean} & {record.session_id for record in ood}:
        raise RuntimeError("common clean and OOD pools contain overlapping session ids")
    if {record.family_id for record in clean} & {record.family_id for record in ood}:
        raise RuntimeError("common clean and OOD pools contain overlapping families")

    clean = [
        record.model_copy(
            update={
                "split": SplitName.TEST,
                "meta": {**record.meta, "evaluation_slice": "common_clean"},
            }
        )
        for record in clean
    ]
    ood = [
        record.model_copy(
            update={
                "split": SplitName.TEST,
                "meta": {**record.meta, "evaluation_slice": OOD_SLICE_NAME},
            }
        )
        for record in ood
    ]
    clean.sort(key=lambda record: record.session_id)
    ood.sort(key=lambda record: record.session_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = args.output_dir / "common_clean.jsonl"
    ood_path = args.output_dir / "slices" / "ood_2024.jsonl"
    clean_tmp = clean_path.with_suffix(".jsonl.tmp")
    ood_tmp = ood_path.with_suffix(".jsonl.tmp")
    write_jsonl(clean, clean_tmp)
    write_jsonl(ood, ood_tmp)
    os.replace(clean_tmp, clean_path)
    os.replace(ood_tmp, ood_path)

    provenance = {
        "schema": "cmdrisk.common_eval",
        "version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "description": "Family- and session-disjoint evaluation pool; no listed train/dev family or session was reused.",
        "output": {
            "common_clean": str(clean_path.resolve().relative_to(PROJECT_ROOT)),
            "ood_2024": str(ood_path.resolve().relative_to(PROJECT_ROOT)),
        },
        "input": {
            "sessions": len(base_raw) + len(holdout_raw),
            "by_source": per_source,
            "fingerprint": _input_fingerprint(list(source_paths.values())),
        },
        "source_limits": source_limits,
        "default_source_limit": default_limit,
        "exclusions": exclusion_audit,
        "excluded_families": len(used_families),
        "deduplication": {
            "base_exact": len(base_exact),
            "base_family_dedup": len(base_dedup),
            "holdout_exact": len(holdout_exact),
            "holdout_family_dedup": len(holdout_dedup),
            "jaccard_threshold": args.jaccard_threshold,
            "max_family_fraction": args.max_family_fraction,
            "minimum_family_cap": args.minimum_family_cap,
            "seed": args.seed,
            "clusterer": "sparse_exact_jaccard",
        },
        "common_clean": _summarize(clean),
        "ood_2024": _summarize(ood),
        "leakage_audit": {
            "excluded_family_overlap": sorted(
                {record.family_id for record in [*clean, *ood]} & used_families
            ),
            "excluded_session_overlap": sorted(
                {record.session_id for record in [*clean, *ood]} & used_sessions
            ),
            "clean_ood_family_overlap": 0,
            "clean_ood_session_overlap": 0,
        },
        "limitations": [
            "The clean common pool contains only family-disjoint leftovers and is malicious-heavy.",
            "OOD contains malicious positives only, so ROC-AUC, PR-AUC, and FPR are undefined for that slice; use recall and detection delay only.",
            "The clean benign count is small; report FPR with Wilson confidence intervals and treat it as auxiliary evidence.",
        ],
    }
    reports = args.output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json_atomic(reports / "provenance.json", provenance)
    write_sha256_manifest([clean_path, ood_path], reports / "SHA256SUMS", root=args.output_dir)
    print(json.dumps({
        "common_clean": provenance["common_clean"],
        "ood_2024": provenance["ood_2024"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

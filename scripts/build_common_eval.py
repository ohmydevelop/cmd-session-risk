#!/usr/bin/env python3
"""Build a family-disjoint common evaluation pool for model comparisons.

The pool is reconstructed from the same pinned raw sources and deduplication
rules used by ``prepare_data.py``.  Any family already used by the train or dev
split of an existing corpus is excluded.  The remaining base sessions and the
temporal 2024 holdout are written to separate files so single-class OOD data
can never silently distort binary FPR/ROC metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data import adapter_for_kind  # noqa: E402
from cmdrisk.data.dedup import exact_deduplicate, family_aware_deduplicate  # noqa: E402
from cmdrisk.data.holdout import HOLDOUT_PERIOD, OOD_SLICE_NAME, is_temporal_holdout  # noqa: E402
from cmdrisk.data.io import write_jsonl  # noqa: E402
from cmdrisk.data.manifest import (  # noqa: E402
    load_data_sources,
    sha256_file,
    write_json_atomic,
    write_sha256_manifest,
)
from cmdrisk.data.normalization import normalize_record  # noqa: E402
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
        if not source_id or not raw_limit:
            raise SystemExit(f"--source-limit must be ID=N, got: {value}")
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


def _input_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        entries = (
            sorted(item for item in path.rglob("*") if item.is_file()) if path.is_dir() else [path]
        )
        for entry in entries:
            relative = entry.relative_to(path).as_posix() if path.is_dir() else path.name
            digest.update(f"{relative}\0".encode())
            with entry.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _read_records(path: Path) -> list[SessionRecord]:
    return [
        SessionRecord.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_used_families(exclude_dirs: list[Path]) -> tuple[set[str], dict[str, dict[str, Any]]]:
    used: set[str] = set()
    audit: dict[str, dict[str, Any]] = {}
    for directory in exclude_dirs:
        split_audit: dict[str, Any] = {}
        for split in ("train", "dev"):
            path = directory / f"{split}.jsonl"
            if not path.is_file():
                continue
            records = _read_records(path)
            families = {record.family_id for record in records}
            used.update(families)
            split_audit[split] = {
                "path": str(path),
                "sessions": len(records),
                "families": len(families),
                "sha256": sha256_file(path),
            }
        audit[str(directory)] = split_audit
    return used, audit


def _summarize(records: list[SessionRecord]) -> dict[str, Any]:
    commands = [len(record.commands) for record in records]
    return {
        "sessions": len(records),
        "families": len({record.family_id for record in records}),
        "labels": dict(sorted(Counter(int(record.label.malicious) for record in records).items())),
        "source_kinds": dict(
            sorted(Counter(str(record.source.kind) for record in records).items())
        ),
        "sources": dict(sorted(Counter(record.source.name for record in records).items())),
        "command_count": {
            "min": min(commands, default=0),
            "max": max(commands, default=0),
            "mean": (sum(commands) / len(commands)) if commands else 0.0,
            "p50": float(sorted(commands)[len(commands) // 2]) if commands else 0,
            "p95": float(sorted(commands)[min(len(commands) - 1, int(len(commands) * 0.95))])
            if commands
            else 0,
        },
    }


def _deduplicate(
    records: list[SessionRecord], *, args: argparse.Namespace
) -> tuple[list[SessionRecord], list[SessionRecord]]:
    exact = exact_deduplicate(records)
    deduplicated = family_aware_deduplicate(
        exact,
        threshold=args.jaccard_threshold,
        max_fraction=args.max_family_fraction,
        minimum_family_cap=args.minimum_family_cap,
        seed=args.seed,
    )
    return exact, deduplicated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "configs" / "data_sources.yaml"
    )
    parser.add_argument("--source", action="append", default=[], metavar="ID=PATH")
    parser.add_argument("--source-limit", action="append", default=[], metavar="ID=N")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "common_eval"
    )
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
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "exclude_dirs": [str(p) for p in exclude_dirs],
                },
                indent=2,
            )
        )
        return 0

    configured = {str(item["id"]): item for item in load_data_sources(args.config)}
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
    source_paths = _parse_source_arguments(args.source)
    source_limits = _parse_source_limits(args.source_limit)
    for requested in source_paths:
        if requested not in configured and requested in aliases:
            configured[requested] = adapter_sources.get(aliases[requested], {})
    unknown = set(source_paths) - set(configured)
    if unknown:
        raise SystemExit(f"unknown source id(s): {', '.join(sorted(unknown))}")
    if not source_paths:
        raise SystemExit("at least one --source ID=PATH is required")
    unknown_limits = set(source_limits) - set(source_paths)
    if unknown_limits:
        raise SystemExit(
            f"--source-limit references unknown source(s): {', '.join(sorted(unknown_limits))}"
        )
    for source_id, path in source_paths.items():
        if not path.exists():
            raise SystemExit(f"source {source_id} does not exist: {path}")
        if not configured[source_id]:
            raise SystemExit(f"source {source_id} could not be resolved from the source manifest")

    raw: list[SessionRecord] = []
    per_source: dict[str, int] = {}
    for source_id, path in sorted(source_paths.items()):
        source = configured[source_id]
        adapter = adapter_for_kind(str(source["adapter"]))
        kwargs: dict[str, Any] = {}
        if source.get("password"):
            kwargs["password"] = str(source["password"])
        count = 0
        for record in adapter.iter_records(path, **kwargs):
            raw.append(record)
            count += 1
            if source_id in source_limits and count >= source_limits[source_id]:
                break
        per_source[source_id] = count

    normalized = [normalize_record(record) for record in raw]
    holdout_flags = [
        is_temporal_holdout(record, period=args.holdout_period) for record in normalized
    ]
    base_raw = [record for record, flag in zip(normalized, holdout_flags, strict=True) if not flag]
    holdout_raw = [record for record, flag in zip(normalized, holdout_flags, strict=True) if flag]
    base_exact, base_dedup = _deduplicate(base_raw, args=args)
    holdout_exact, holdout_dedup = _deduplicate(holdout_raw, args=args)

    used_families, exclusion_audit = _load_used_families(exclude_dirs)
    clean = [record for record in base_dedup if record.family_id not in used_families]
    ood = [record for record in holdout_dedup if record.family_id not in used_families]
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
    write_jsonl(clean, clean_path)
    write_jsonl(ood, ood_path)

    provenance = {
        "schema": "cmdrisk.common_eval",
        "version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "description": "Family-disjoint evaluation pool; no listed train/dev family was reused.",
        "output": {
            "common_clean": str(clean_path.relative_to(PROJECT_ROOT)),
            "ood_2024": str(ood_path.relative_to(PROJECT_ROOT)),
        },
        "input": {
            "sessions": len(raw),
            "by_source": per_source,
            "fingerprint": _input_fingerprint(list(source_paths.values())),
        },
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
        },
        "common_clean": _summarize(clean),
        "ood_2024": _summarize(ood),
        "leakage_audit": {
            "excluded_family_overlap": sorted(
                {record.family_id for record in [*clean, *ood]} & used_families
            ),
            "clean_ood_family_overlap": 0,
            "clean_ood_session_overlap": 0,
        },
        "limitations": [
            "The clean common pool contains only family-disjoint leftovers and is malicious-heavy (3760/3886 in the frozen v2/v3 run).",  # noqa: E501
            "OOD contains malicious positives only, so ROC-AUC, PR-AUC, and FPR are undefined for that slice; use recall and detection delay only.",  # noqa: E501
            "The 126 clean benign sessions make FPR confidence intervals wide; report them as auxiliary evidence rather than a production false-positive estimate.",  # noqa: E501
        ],
    }
    reports = args.output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_json_atomic(reports / "provenance.json", provenance)
    write_sha256_manifest([clean_path, ood_path], reports / "SHA256SUMS", root=args.output_dir)

    print(
        json.dumps(
            {
                "common_clean": provenance["common_clean"],
                "ood_2024": provenance["ood_2024"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

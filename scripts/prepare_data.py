#!/usr/bin/env python3
"""Prepare normalized, deduplicated 60K sessions from local raw sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data import (  # noqa: E402
    HOLDOUT_PERIOD,
    adapter_for_kind,
    provenance_report,
)
from cmdrisk.data.io import write_jsonl  # noqa: E402
from cmdrisk.data.manifest import (  # noqa: E402
    load_data_sources,
    sha256_file,
    write_json_atomic,
    write_sha256_manifest,
)
from cmdrisk.data.pipeline import (  # noqa: E402
    prepare_sessions_v2_with_stats,
    prepare_sessions_with_stats,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data_sources.yaml",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="local raw path for a manifest source; repeat as needed",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--total", type=int, default=60_000)
    parser.add_argument("--jaccard-threshold", type=float, default=0.9)
    parser.add_argument("--max-family-fraction", type=float, default=0.0025)
    parser.add_argument("--minimum-family-cap", type=int, default=50)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--max-records-per-source",
        type=int,
        default=None,
        help="bound memory for large archives by limiting each source independently",
    )
    parser.add_argument(
        "--source-limit",
        action="append",
        default=[],
        metavar="ID=N",
        help="per-source input cap; repeat as needed and override the global fallback",
    )
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="BUCKET=RATIO",
        help="training bucket ratio; repeat as needed (values must sum to 1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-quota-backfill",
        action="store_true",
        help="allow smoke tests to backfill quota shortages from other buckets",
    )
    parser.add_argument("--quota-drift-tolerance", type=float, default=0.0)
    parser.add_argument("--balance-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--enforce-split-balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="fail when label/source/family drift exceeds --balance-tolerance",
    )
    parser.add_argument(
        "--pipeline-version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 writes sessions.jsonl plus splits; v2 adds a temporal holdout, "
        "deterministic stress rows, strict 40/40/15/5 quota, and evaluation slices",
    )
    parser.add_argument(
        "--holdout-period",
        default=HOLDOUT_PERIOD,
        help="temporal period reserved for the out-of-distribution test slice (v2 only)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


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
    limits: dict[str, int] = {}
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
        if source_id in limits:
            raise SystemExit(f"duplicate --source-limit for: {source_id}")
        limits[source_id] = limit
    return limits


def _parse_quota(values: list[str]) -> dict[str, float] | None:
    if not values:
        return None
    quota: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--quota must be BUCKET=RATIO, got: {value}")
        bucket, raw_ratio = value.split("=", 1)
        if not bucket or not raw_ratio:
            raise SystemExit(f"--quota must be BUCKET=RATIO, got: {value}")
        try:
            ratio = float(raw_ratio)
        except ValueError as exc:
            raise SystemExit(f"--quota ratio must be numeric, got: {value}") from exc
        if ratio < 0:
            raise SystemExit(f"--quota ratio must be non-negative, got: {value}")
        if bucket in quota:
            raise SystemExit(f"duplicate --quota for: {bucket}")
        quota[bucket] = ratio
    if abs(sum(quota.values()) - 1.0) > 1e-9:
        raise SystemExit("--quota values must sum to 1.0")
    return quota


def _input_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        if path.is_dir():
            entries = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            entries = [path]
        for entry in entries:
            if path.is_dir():
                relative = entry.relative_to(path).as_posix()
            else:
                relative = path.name
            digest.update(f"{relative}\0".encode())
            with entry.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _normalized_preparation_options(
    args: argparse.Namespace,
    *,
    source_paths: dict[str, Path],
    configured: dict[str, dict[str, Any]],
    source_limits: dict[str, int],
    quota: dict[str, float] | None,
) -> dict[str, object]:
    sources: list[dict[str, object]] = []
    for source_id, path in sorted(source_paths.items()):
        source = configured[source_id]
        identity = {key: value for key, value in sorted(source.items()) if key not in {"password"}}
        if source.get("password"):
            identity["password_sha256"] = hashlib.sha256(
                str(source["password"]).encode()
            ).hexdigest()
        sources.append(
            {
                "id": source_id,
                "path": str(path),
                "config": identity,
            }
        )
    return {
        "version": 1,
        "pipeline_version": getattr(args, "pipeline_version", "v1"),
        "holdout_period": getattr(args, "holdout_period", HOLDOUT_PERIOD),
        "sources": sources,
        "total": args.total,
        "max_records": args.max_records,
        "max_records_per_source": args.max_records_per_source,
        "source_limits": source_limits,
        "quota": quota,
        "seed": args.seed,
        "jaccard_threshold": args.jaccard_threshold,
        "max_family_fraction": args.max_family_fraction,
        "minimum_family_cap": args.minimum_family_cap,
        "allow_quota_backfill": args.allow_quota_backfill,
        "quota_drift_tolerance": args.quota_drift_tolerance,
        "enforce_split_balance": args.enforce_split_balance,
        "balance_tolerance": args.balance_tolerance,
    }


def main() -> int:
    args = _parse_args()
    source_paths = _parse_source_arguments(args.source)
    source_limits = _parse_source_limits(args.source_limit)
    quota = _parse_quota(args.quota)
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
    for requested in source_paths:
        if requested in configured:
            continue
        canonical = aliases.get(requested, requested)
        if canonical in adapter_sources:
            configured[requested] = adapter_sources[canonical]
    unknown = set(source_paths) - set(configured)
    if unknown:
        raise SystemExit(f"unknown source id(s): {', '.join(sorted(unknown))}")
    unknown_limits = set(source_limits) - set(source_paths)
    if unknown_limits:
        raise SystemExit(
            f"--source-limit references source(s) not supplied with --source: "
            f"{', '.join(sorted(unknown_limits))}"
        )
    if not source_paths and not args.dry_run:
        raise SystemExit("at least one --source ID=PATH is required")

    if args.dry_run:
        print(
            json.dumps(
                {
                    "sources": {key: str(value) for key, value in source_paths.items()},
                    "output_dir": str(args.output_dir),
                    "total": args.total,
                    "seed": args.seed,
                },
                sort_keys=True,
            )
        )
        return 0

    for source_id, path in source_paths.items():
        if not path.exists():
            raise SystemExit(f"source {source_id} does not exist: {path}")

    is_v2 = args.pipeline_version == "v2"
    if is_v2:
        artifact_names = [
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "reports/provenance.json",
            "reports/prepare_state.json",
        ]
    else:
        artifact_names = [
            "sessions.jsonl",
            "train.jsonl",
            "dev.jsonl",
            "test.jsonl",
            "provenance.json",
            "prepare_state.json",
        ]
    reports_dir = args.output_dir / "reports" if is_v2 else args.output_dir
    state_path = reports_dir / "prepare_state.json"
    fingerprint = _input_fingerprint(list(source_paths.values()))
    preparation_options = _normalized_preparation_options(
        args,
        source_paths=source_paths,
        configured=configured,
        source_limits=source_limits,
        quota=quota,
    )
    options_fingerprint = _canonical_digest(preparation_options)
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if (
            state.get("input_fingerprint") == fingerprint
            and state.get("options_fingerprint") == options_fingerprint
            and all((args.output_dir / name).exists() for name in artifact_names)
        ):
            print(f"resume: outputs are current in {args.output_dir}")
            return 0

    records = []
    per_source: dict[str, int] = {}
    for source_id, path in sorted(source_paths.items()):
        source = configured[source_id]
        adapter = adapter_for_kind(str(source["adapter"]))
        kwargs: dict[str, Any] = {}
        if source.get("password"):
            kwargs["password"] = str(source["password"])
        count = 0
        source_limit = source_limits.get(source_id, args.max_records_per_source)
        for record in adapter.iter_records(path, **kwargs):
            records.append(record)
            count += 1
            if args.max_records is not None and len(records) >= args.max_records:
                break
            if source_limit is not None and count >= source_limit:
                break
        per_source[source_id] = count
        if args.max_records is not None and len(records) >= args.max_records:
            break

    if is_v2:
        if args.allow_quota_backfill:
            raise SystemExit("v2 does not support --allow-quota-backfill; quota must be exact")
        prepared, preparation_stats = prepare_sessions_v2_with_stats(
            records,
            total=args.total,
            quota=quota,
            jaccard_threshold=args.jaccard_threshold,
            max_family_fraction=args.max_family_fraction,
            minimum_family_cap=args.minimum_family_cap,
            seed=args.seed,
            holdout_period=args.holdout_period,
            balance_tolerance=args.balance_tolerance,
            # The temporal holdout is appended to test after the split, so
            # whole-corpus balance gates cannot hold by construction.
            enforce_balance=False,
        )
    else:
        prepared, preparation_stats = prepare_sessions_with_stats(
            records,
            total=args.total,
            quota=quota,
            jaccard_threshold=args.jaccard_threshold,
            max_family_fraction=args.max_family_fraction,
            minimum_family_cap=args.minimum_family_cap,
            seed=args.seed,
            allow_quota_backfill=args.allow_quota_backfill,
            quota_drift_tolerance=args.quota_drift_tolerance,
            balance_tolerance=args.balance_tolerance,
            enforce_balance=args.enforce_split_balance,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not is_v2:
        write_jsonl(prepared, args.output_dir / "sessions.jsonl")
    for split in ("train", "dev", "test"):
        write_jsonl(
            (record for record in prepared if str(record.split) == split),
            args.output_dir / f"{split}.jsonl",
        )
    split_hashes = {
        split: sha256_file(args.output_dir / f"{split}.jsonl")
        for split in ("train", "dev", "test")
        if (args.output_dir / f"{split}.jsonl").exists()
    }
    report = provenance_report(
        prepared,
        requested_quota=quota,
        input_count=int(preparation_stats["input_count"]),
        exact_count=int(preparation_stats["exact_count"]),
        dedup_count=int(preparation_stats["dedup_count"]),
        split_hashes=split_hashes,
        extra=(
            {
                "pipeline_version": "v2",
                "holdout_period": args.holdout_period,
                "v2_stats": preparation_stats,
            }
            if is_v2
            else {"pipeline_version": "v1"}
        ),
    )
    report["input_sessions"] = len(records)
    report["input_sessions_by_source"] = per_source
    report["input_fingerprint"] = fingerprint
    report["sampled_sessions"] = preparation_stats["sampled_count"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(reports_dir / "provenance.json", report)

    slice_files: list[Path] = []
    if is_v2:
        slices_dir = args.output_dir / "slices"
        slices_dir.mkdir(parents=True, exist_ok=True)
        slice_names = sorted(
            {
                str(record.meta.get("evaluation_slice"))
                for record in prepared
                if record.meta.get("evaluation_slice")
            }
        )
        for slice_name in slice_names:
            members = [
                record
                for record in prepared
                if str(record.meta.get("evaluation_slice")) == slice_name
            ]
            path = slices_dir / f"{slice_name}.jsonl"
            write_jsonl(members, path)
            slice_files.append(path)
        missing = {"ood_2024", "synthetic_stress"} - set(slice_names)
        if missing:
            raise SystemExit(
                "v2 corpus is missing required evaluation slice(s): " + ", ".join(sorted(missing))
            )
    write_json_atomic(
        state_path,
        {
            "pipeline_version": args.pipeline_version,
            "input_fingerprint": fingerprint,
            "options_fingerprint": options_fingerprint,
            "preparation_options": preparation_options,
            "input_sessions": len(records),
            "prepared_sessions": len(prepared),
            "preparation_stats": preparation_stats,
            "seed": args.seed,
            "total": args.total,
        },
    )

    output_files = [
        args.output_dir / name for name in artifact_names if (args.output_dir / name).exists()
    ]
    output_files.extend(slice_files)
    manifest_path = reports_dir / "SHA256SUMS"
    write_sha256_manifest(output_files, manifest_path, root=args.output_dir)
    print(
        f"prepared {len(prepared)} sessions ({args.pipeline_version}); "
        f"report: {reports_dir / 'provenance.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

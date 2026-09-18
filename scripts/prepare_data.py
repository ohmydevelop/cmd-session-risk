#!/usr/bin/env python3
"""Prepare normalized, deduplicated 60K sessions from local raw sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data import adapter_for_kind, prepare_sessions, provenance_report  # noqa: E402
from cmdrisk.data.io import write_jsonl  # noqa: E402
from cmdrisk.data.manifest import (  # noqa: E402
    load_data_sources,
    sha256_file,
    write_json_atomic,
    write_sha256_manifest,
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
    parser.add_argument("--seed", type=int, default=42)
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


def _input_fingerprint(paths: list[Path]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        if path.is_dir():
            entries = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            entries = [path]
        for entry in entries:
            stat = entry.stat()
            digest.update(
                f"{entry.resolve()}\x1f{stat.st_size}\x1f{stat.st_mtime_ns}\n".encode("utf-8")
            )
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    source_paths = _parse_source_arguments(args.source)
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

    artifact_names = [
        "sessions.jsonl",
        "train.jsonl",
        "dev.jsonl",
        "test.jsonl",
        "provenance.json",
        "prepare_state.json",
    ]
    state_path = args.output_dir / "prepare_state.json"
    fingerprint = _input_fingerprint(list(source_paths.values()))
    if args.resume and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("input_fingerprint") == fingerprint and all(
            (args.output_dir / name).exists() for name in artifact_names
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
        for record in adapter.iter_records(path, **kwargs):
            records.append(record)
            count += 1
            if args.max_records is not None and len(records) >= args.max_records:
                break
        per_source[source_id] = count
        if args.max_records is not None and len(records) >= args.max_records:
            break

    prepared = prepare_sessions(
        records,
        total=args.total,
        jaccard_threshold=args.jaccard_threshold,
        max_family_fraction=args.max_family_fraction,
        minimum_family_cap=args.minimum_family_cap,
        seed=args.seed,
    )
    report = provenance_report(prepared)
    report["input_sessions"] = len(records)
    report["input_sessions_by_source"] = per_source
    report["input_fingerprint"] = fingerprint

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(prepared, args.output_dir / "sessions.jsonl")
    for split in ("train", "dev", "test"):
        write_jsonl(
            (record for record in prepared if str(record.split) == split),
            args.output_dir / f"{split}.jsonl",
        )
    write_json_atomic(args.output_dir / "provenance.json", report)
    write_json_atomic(
        state_path,
        {
            "input_fingerprint": fingerprint,
            "input_sessions": len(records),
            "prepared_sessions": len(prepared),
            "seed": args.seed,
            "total": args.total,
        },
    )

    output_files = [args.output_dir / name for name in artifact_names if (args.output_dir / name).exists()]
    manifest_path = args.output_dir / "SHA256SUMS"
    write_sha256_manifest(output_files, manifest_path, root=args.output_dir)
    print(f"prepared {len(prepared)} sessions; report: {args.output_dir / 'provenance.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

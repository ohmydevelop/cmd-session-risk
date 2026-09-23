#!/usr/bin/env python3
"""Bounded-memory builder for the ``cmd-session-risk`` v4 corpus.

Design goals
------------
* Reuse the reviewed ``prepare_sessions_v2_with_stats`` pipeline (temporal
  holdout, exact + family-aware dedup, deterministic stress, strict quota,
  family-aware split) so v4 stays method-comparable with v3 and every
  fail-closed guarantee of the v2 path still holds.
* Resolve every pinned v4 source from the on-disk raw tree automatically, and
  record exactly what was read into the provenance report.
* Never touch the v3 artefact tree: every output path is v4-scoped.
* Emit the extra evaluation slices the v4 corpus actually has
  (``real_only`` / ``real_plus_semi_real``) next to ``ood_2024`` and
  ``synthetic_stress`` so the training evaluator can surface the gaps.

This is an additive script.  ``scripts/prepare_data.py`` remains the v1/v2/v3
entry point and is deliberately left untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data.adapters import adapter_for_kind  # noqa: E402
from cmdrisk.data.augmentation import (  # noqa: E402
    _ATTACK_VARIANT_NAMES,
    _HARD_NEGATIVE_TEMPLATES,
)
from cmdrisk.data.holdout import (  # noqa: E402
    HOLDOUT_PERIOD,
    OOD_SLICE_NAME,
    is_temporal_holdout,
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
    provenance_report,
)
from cmdrisk.data.sampling import _record_bucket, is_atomic_seed  # noqa: E402
from cmdrisk.training.data import select_prefix_indices  # noqa: E402

# Source id -> path relative to --raw-dir.  Pinned by configs/data_sources_v4.yaml.
SOURCE_PATHS: dict[str, str] = {
    "microsoft_iot_honeypot": "microsoft_iot_honeypot.zip",
    "shell_attack_evolution": "shell_attack_evolution",
    "kypo_zenodo": "kypo_zenodo.bin",
    "atomic_red_team": "atomic_red_team",
    "actions_starter_workflows": "actions_starter_workflows",
    "docker_awesome_compose": "docker_awesome_compose",
    "ansible_examples": "ansible_examples",
    "kubernetes_examples": "kubernetes_examples",
    "puppetlabs_apache": "puppetlabs_apache",
    "asdf": "v4_scripts/asdf",
    "bash_it": "v4_scripts/bash-it",
    "docker_bench_security": "v4_scripts/docker-bench-security",
    "nvm": "v4_scripts/nvm",
    "ohmyzsh": "v4_scripts/ohmyzsh",
    "pyenv": "v4_scripts/pyenv",
    "devops_bash_tools": "v4_histories/DevOps-Bash-tools",
    "mathiasbynens_dotfiles": "v4_histories/mathiasbynens-dotfiles",
    "hf_bash_history": "hf_bash_history/bash.txt",
    "ohmybash": "v4_scripts/ohmybash",
    "zsh_autosuggestions": "v4_scripts/zsh_autosuggestions",
    "zsh_syntax_highlighting": "v4_scripts/zsh_syntax_highlighting",
    "fzf": "v4_scripts/fzf",
}

# Sources that are inventories / technique seeds rather than session corpora.
SKIP_ADAPTERS = {"command_pool"}

DEFAULT_V4_QUOTA: dict[str, float] = {
    "real_malicious": 0.45,
    "benign": 0.35,
    "llm_variant": 0.15,
    "hard_negative": 0.05,
}

SLICE_BY_KIND = {
    "llm_variant": "synthetic_stress",
    "hard_negative": "synthetic_stress",
    "real_malicious": "real_only",
    "real_benign": "real_only",
    "script_benign": "real_plus_semi_real",
    "real_history_pool": "real_plus_semi_real",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "data_sources_v4.yaml")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "v4")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="override a source path, given as ID=PATH (repeatable)",
    )
    parser.add_argument("--total", type=int, default=58_000, help="target base session count before holdout")
    parser.add_argument("--quota", default=None, help="bucket=ratio,... overriding the v4 default quota")
    parser.add_argument("--jaccard-threshold", type=float, default=0.9)
    parser.add_argument("--max-family-fraction", type=float, default=0.0025)
    parser.add_argument("--minimum-family-cap", type=int, default=50)
    parser.add_argument("--holdout-period", default=HOLDOUT_PERIOD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument(
        "--quota-retry-step",
        type=float,
        default=0.03,
        help="fraction to shrink --total by when a strict quota cannot be met (0 disables)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parse_quota(value: str | None) -> dict[str, float]:
    if not value:
        return dict(DEFAULT_V4_QUOTA)
    quota: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"quota entry must be bucket=ratio: {item!r}")
        bucket, raw = item.split("=", 1)
        quota[bucket.strip()] = float(raw)
    if not quota:
        raise SystemExit("quota must contain at least one bucket")
    return quota


def _resolve_paths(args: argparse.Namespace, configured: dict[str, dict[str, Any]]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for item in args.source:
        if "=" not in item:
            raise SystemExit(f"--source must be given as ID=PATH: {item!r}")
        source_id, raw_path = item.split("=", 1)
        overrides[source_id.strip()] = Path(raw_path.strip())

    resolved: dict[str, Path] = {}
    for source_id, source in configured.items():
        if str(source["adapter"]) in SKIP_ADAPTERS:
            continue
        if source_id in overrides:
            resolved[source_id] = overrides[source_id]
            continue
        relative = SOURCE_PATHS.get(source_id)
        if relative is None:
            raise SystemExit(f"no raw path is declared for source {source_id!r}; pass --source {source_id}=PATH")
        candidate = args.raw_dir / relative
        if not candidate.exists():
            raise SystemExit(f"raw path for {source_id!r} does not exist: {candidate}")
        resolved[source_id] = candidate
    return resolved


def _ingest(
    configured: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    *,
    max_records_per_source: int | None,
) -> tuple[list[Any], dict[str, int], dict[str, int]]:
    records: list[Any] = []
    per_source: dict[str, int] = {}
    per_source_commands: dict[str, int] = {}
    for source_id in sorted(paths):
        source = configured[source_id]
        adapter = adapter_for_kind(str(source["adapter"]))
        kwargs: dict[str, Any] = {}
        if source.get("password"):
            kwargs["password"] = str(source["password"])
        print(f"  ... ingesting {source_id} from {paths[source_id]}", flush=True)
        count = 0
        commands = 0
        for record in adapter.iter_records(paths[source_id], **kwargs):
            records.append(record)
            count += 1
            commands += len(record.commands)
            if max_records_per_source is not None and count >= max_records_per_source:
                break
        per_source[source_id] = count
        per_source_commands[source_id] = commands
        print(f"  ingested {source_id:28s} sessions={count:7d} commands={commands:9d}", flush=True)
    return records, per_source, per_source_commands


def _cap32_counts(prepared: list[Any]) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {}
    for record in prepared:
        split = str(record.split)
        label = record.label
        evidence = [
            int(index)
            for values in label.technique_evidence.values()
            for index in values
        ]
        selected = select_prefix_indices(
            len(record.commands),
            max_prefixes=32,
            first_evidence_index=label.first_evidence_index,
            evidence_indices=evidence,
        )
        bucket = totals.setdefault(split, {"sessions": 0, "cap32_prefixes": 0, "malicious_prefixes": 0})
        bucket["sessions"] += 1
        bucket["cap32_prefixes"] += len(selected)
        bucket["malicious_prefixes"] += sum(
            1
            for index in selected
            if label.malicious == 1
            and label.first_evidence_index is not None
            and index >= label.first_evidence_index
        )
    return totals


def main() -> int:
    args = _parse_args()
    configured = {str(item["id"]): item for item in load_data_sources(args.config)}
    paths = _resolve_paths(args, configured)
    quota = _parse_quota(args.quota)

    print(f"v4 sources: {len(paths)} session corpora (skipping {sorted(SKIP_ADAPTERS)})")
    if args.dry_run:
        print(json.dumps({"sources": {k: str(v) for k, v in paths.items()}, "total": args.total,
                          "quota": quota, "output_dir": str(args.output_dir)}, indent=2))
        return 0

    print("ingesting source records ...")
    records, per_source, per_source_commands = _ingest(
        configured, paths, max_records_per_source=args.max_records_per_source
    )
    print(f"ingested {len(records)} raw sessions")

    # Fail closed *before* the expensive dedup/split work: a corpus with no
    # temporal holdout cannot produce the required ``ood_2024`` slice.
    holdout_preview = sum(
        1 for record in records if is_temporal_holdout(record, period=args.holdout_period)
    )
    print(f"  temporal holdout ({args.holdout_period!r}) sessions: {holdout_preview}", flush=True)
    if holdout_preview == 0:
        raise SystemExit(
            "no sessions matched the temporal holdout period "
            f"{args.holdout_period!r}; refusing to build a corpus that cannot "
            "produce the required ood_2024 evaluation slice (check --max-records-per-source "
            "or the shell_attack_evolution ingestion)"
        )

    # The strict sampler refuses to backfill, so shrink the target up front to
    # the largest volume the available real buckets can actually support.
    available = Counter(
        _record_bucket(record, quota) for record in records if not is_atomic_seed(record)
    )
    # ``llm_variant``/``hard_negative`` rows are *generated*, not ingested, so
    # their ceiling comes from their real bases times the number of deterministic
    # styles/templates, never from a (always-empty) ingested bucket count.
    malicious_bases = sum(
        1
        for record in records
        if str(record.source.kind) == "real_malicious" and not is_atomic_seed(record)
    )
    strict_benign_bases = sum(
        1 for record in records if str(record.source.kind) in {"real_benign", "script_benign"}
    )
    generated_caps = {
        "llm_variant": malicious_bases * len(_ATTACK_VARIANT_NAMES),
        "hard_negative": strict_benign_bases * len(_HARD_NEGATIVE_TEMPLATES),
    }
    capacity: dict[str, int] = {}
    for bucket, ratio in quota.items():
        if ratio <= 0:
            continue
        cap = generated_caps.get(bucket, available.get(bucket, 0))
        capacity[bucket] = int(cap * 0.95 / ratio)
    feasible = min(capacity.values()) if capacity else args.total
    total = min(args.total, max(feasible, 1_000))
    print(
        "  per-bucket session ceiling: "
        f"{dict(sorted(capacity.items()))} -> feasible total={feasible}",
        flush=True,
    )
    if total != args.total:
        print(
            f"  target {args.total} exceeds available real data; capping at {total} "
            f"(ingested buckets: {dict(sorted(available.items()))}, "
            f"generated bases: malicious={malicious_bases} strict_benign={strict_benign_bases})",
            flush=True,
        )
    attempts = 0
    while True:
        attempts += 1
        try:
            prepared, stats = prepare_sessions_v2_with_stats(
                records,
                total=total,
                quota=quota,
                jaccard_threshold=args.jaccard_threshold,
                max_family_fraction=args.max_family_fraction,
                minimum_family_cap=args.minimum_family_cap,
                seed=args.seed,
                holdout_period=args.holdout_period,
                enforce_balance=False,
            )
            break
        except ValueError as exc:
            if "training quota cannot be satisfied" not in str(exc) or args.quota_retry_step <= 0:
                raise
            new_total = int(total * (1.0 - args.quota_retry_step))
            if new_total < 1000 or attempts > 40:
                raise
            print(f"  strict quota unmet at total={total} ({exc}); retrying at total={new_total}", flush=True)
            total = new_total
    print(f"prepared {len(prepared)} sessions (base target={total}, attempts={attempts})")

    output_dir = args.output_dir
    (output_dir / "reports").mkdir(parents=True, exist_ok=True)
    (output_dir / "slices").mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        write_jsonl((r for r in prepared if str(r.split) == split), output_dir / f"{split}.jsonl")

    split_hashes = {
        split: sha256_file(output_dir / f"{split}.jsonl")
        for split in ("train", "dev", "test")
        if (output_dir / f"{split}.jsonl").exists()
    }

    slice_files: list[Path] = []
    slice_names = sorted(
        {
            str(record.meta.get("evaluation_slice"))
            for record in prepared
            if record.meta.get("evaluation_slice")
        }
        | {SLICE_BY_KIND[str(r.source.kind)] for r in prepared if str(r.source.kind) in SLICE_BY_KIND}
    )
    for slice_name in slice_names:
        members = [
            record
            for record in prepared
            if str(record.meta.get("evaluation_slice") or SLICE_BY_KIND.get(str(record.source.kind))) == slice_name
        ]
        if not members:
            continue
        path = output_dir / "slices" / f"{slice_name}.jsonl"
        write_jsonl(members, path)
        slice_files.append(path)
    missing = {OOD_SLICE_NAME, "synthetic_stress"} - {p.stem for p in slice_files}
    if missing:
        raise SystemExit("v4 corpus is missing required evaluation slice(s): " + ", ".join(sorted(missing)))

    cap32 = _cap32_counts(prepared)
    report = provenance_report(
        prepared,
        requested_quota=quota,
        input_count=int(stats["input_count"]),
        exact_count=int(stats["exact_count"]),
        dedup_count=int(stats["dedup_count"]),
        split_hashes=split_hashes,
        extra={
            "pipeline_version": "v4",
            "holdout_period": args.holdout_period,
            "v4_stats": stats,
            "cap32_prefix_counts": cap32,
            "source_paths": {k: str(v) for k, v in sorted(paths.items())},
            "skipped_adapters": sorted(SKIP_ADAPTERS),
            "seed": args.seed,
            "requested_total": args.total,
            "realized_total": total,
        },
    )
    report["input_sessions"] = len(records)
    report["input_sessions_by_source"] = per_source
    report["input_commands_by_source"] = per_source_commands
    report["sampled_sessions"] = stats["sampled_count"]
    report["slice_counts"] = {p.stem: sum(1 for _ in p.open()) for p in slice_files}
    write_json_atomic(output_dir / "reports" / "provenance.json", report)
    write_json_atomic(
        output_dir / "reports" / "prepare_state.json",
        {
            "pipeline_version": "v4",
            "input_sessions": len(records),
            "prepared_sessions": len(prepared),
            "requested_total": args.total,
            "realized_total": total,
            "quota": quota,
            "seed": args.seed,
            "cap32_prefix_counts": cap32,
        },
    )

    output_files = [output_dir / f"{s}.jsonl" for s in ("train", "dev", "test")]
    output_files += slice_files
    output_files += [output_dir / "reports" / "provenance.json", output_dir / "reports" / "prepare_state.json"]
    write_sha256_manifest(output_files, output_dir / "reports" / "SHA256SUMS", root=output_dir)

    by_kind = Counter(str(r.source.kind) for r in prepared)
    print(f"wrote {len(prepared)} sessions to {output_dir}")
    print(f"  by kind: {dict(sorted(by_kind.items()))}")
    print(f"  cap32 prefixes: { {k: v['cap32_prefixes'] for k, v in sorted(cap32.items())} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

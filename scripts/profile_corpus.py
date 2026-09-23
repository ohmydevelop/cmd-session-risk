#!/usr/bin/env python3
"""Profile a processed cmd-session-risk corpus and emit a JSON report.

The report is the factual basis for the "how much data do we have and how is
it laid out" section of the production-readiness documents.  Everything is
derived from the split files themselves plus the exact prefix-selection
function used by the training evaluator, so the numbers cannot drift from what
training actually sees.

Usage::

    scripts/profile_corpus.py --corpus data/processed/v4 \
        --output data/processed/v4/reports/dataset_profile.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.training.data import _evaluation_slice, select_prefix_indices  # noqa: E402

SPLITS = ("train", "dev", "test")
TIER_BY_KIND = {
    "real_malicious": "real",
    "real_benign": "real",
    "real_history_pool": "real",
    "script_benign": "semi_real",
    "llm_variant": "semi_real",
    "hard_negative": "hard_negative",
}


def _percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    idx = min(len(values) - 1, int(len(values) * q))
    return values[idx]


def profile_split(path: Path) -> dict[str, object]:
    sessions = malicious = commands_total = prefixes = prefixes_mal = 0
    families: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    slices: dict[str, list[int]] = {}
    licenses: Counter[str] = Counter()
    lengths: list[int] = []
    for raw in path.open(encoding="utf-8"):
        raw = raw.strip()
        if not raw:
            continue
        record = json.loads(raw)
        sessions += 1
        label = record.get("label") or {}
        is_mal = 1 if label.get("malicious") else 0
        malicious += is_mal
        families[str(record.get("family_id"))] += 1
        source = record.get("source") or {}
        kind = str(source.get("kind", "?"))
        kinds[kind] += 1
        tiers[TIER_BY_KIND.get(kind, "other")] += 1
        licenses[str(source.get("license", "?"))] += 1
        command_count = len(record.get("commands") or [])
        commands_total += command_count
        lengths.append(command_count)
        selected = select_prefix_indices(command_count, max_prefixes=32)
        prefixes += len(selected)
        prefixes_mal += is_mal * len(selected)
        slice_name = _evaluation_slice(record, kind)
        bucket = slices.setdefault(slice_name, [0, 0])
        bucket[0] += is_mal
        bucket[1] += 1
    return {
        "file": path.as_posix(),
        "sessions": sessions,
        "malicious_sessions": malicious,
        "benign_sessions": sessions - malicious,
        "malicious_rate": round(malicious / sessions, 6) if sessions else 0.0,
        "families": len(families),
        "largest_family": max(families.values()) if families else 0,
        "commands_total": commands_total,
        "command_length": {
            "mean": round(commands_total / sessions, 3) if sessions else 0.0,
            "p50": _percentile(lengths, 0.50),
            "p95": _percentile(lengths, 0.95),
            "max": max(lengths) if lengths else 0,
        },
        "cap32_prefixes": prefixes,
        "cap32_prefixes_malicious": prefixes_mal,
        "cap32_prefixes_benign": prefixes - prefixes_mal,
        "source_kinds": dict(kinds.most_common()),
        "source_tiers": dict(tiers.most_common()),
        "evaluation_slices": {
            name: {"malicious": mal, "sessions": total, "benign": total - mal}
            for name, (mal, total) in sorted(slices.items())
        },
        "licenses": dict(licenses.most_common()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = {
        "schema": "cmdrisk.dataset_profile",
        "version": 1,
        "corpus": args.corpus.as_posix(),
        "splits": {},
    }
    total_prefixes = 0
    for split in SPLITS:
        path = args.corpus / f"{split}.jsonl"
        if not path.is_file():
            continue
        payload = profile_split(path)
        profile["splits"][split] = payload
        total_prefixes += int(payload["cap32_prefixes"])
        print(
            f"{split:5s} sessions={payload['sessions']:7d} "
            f"mal={payload['malicious_sessions']:7d} ben={payload['benign_sessions']:7d} "
            f"families={payload['families']:6d} cap32_prefixes={payload['cap32_prefixes']:8d}"
        )
    profile["cap32_prefixes_total"] = total_prefixes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

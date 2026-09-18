#!/usr/bin/env python3
"""Generate evidence-aware 1..N prefixes from prepared sessions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data import build_prefix_records  # noqa: E402
from cmdrisk.data.io import read_sessions, write_jsonl  # noqa: E402
from cmdrisk.data.manifest import write_json_atomic, write_sha256_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "processed" / "prefixes.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.dry_run:
        print(json.dumps({"inputs": [str(path) for path in args.input], "output": str(args.output)}))
        return 0
    missing = [str(path) for path in args.input if not path.exists()]
    if missing:
        raise SystemExit(f"input file(s) do not exist: {', '.join(missing)}")

    state_path = args.output.with_suffix(args.output.suffix + ".state.json")
    if args.resume and args.output.exists() and state_path.exists():
        print(f"resume: {args.output} already exists")
        return 0

    sessions = [session for path in args.input for session in read_sessions(path)]
    prefixes = build_prefix_records(sessions)
    write_jsonl(prefixes, args.output)
    write_json_atomic(
        state_path,
        {
            "inputs": [str(path.resolve()) for path in args.input],
            "session_count": len(sessions),
            "prefix_count": len(prefixes),
        },
    )
    write_sha256_manifest([args.output, state_path], args.output.parent / "prefixes.SHA256SUMS")
    print(f"built {len(prefixes)} prefixes from {len(sessions)} sessions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

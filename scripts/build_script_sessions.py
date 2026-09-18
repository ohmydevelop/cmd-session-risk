#!/usr/bin/env python3
"""Extract benign command sessions from local DevOps/script repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data import ScriptRepoAdapter, family_aware_deduplicate, normalize_record  # noqa: E402
from cmdrisk.data.io import write_jsonl  # noqa: E402
from cmdrisk.data.manifest import write_json_atomic, write_sha256_manifest  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "interim" / "script_sessions.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_repos(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--repo must be NAME=PATH, got: {value}")
        name, path = value.split("=", 1)
        result[name] = Path(path).expanduser().resolve()
    return result


def _repo_fingerprint(repos: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, root in sorted(repos.items()):
        digest.update(f"{name}\x1f{root}\n".encode("utf-8"))
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            stat = path.stat()
            digest.update(f"{path.relative_to(root)}\x1f{stat.st_size}\x1f{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    repos = _parse_repos(args.repo)
    if args.dry_run:
        print(json.dumps({"repos": {key: str(value) for key, value in repos.items()}, "output": str(args.output)}))
        return 0
    missing = [str(path) for path in repos.values() if not path.exists()]
    if missing:
        raise SystemExit(f"repository path(s) do not exist: {', '.join(missing)}")

    state_path = args.output.with_suffix(args.output.suffix + ".state.json")
    fingerprint = _repo_fingerprint(repos)
    if args.resume and args.output.exists() and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("repo_fingerprint") == fingerprint:
            print(f"resume: {args.output} is current")
            return 0

    records = []
    for _, root in sorted(repos.items()):
        records.extend(ScriptRepoAdapter().iter_records(root))
    records = [normalize_record(record) for record in records]
    records = family_aware_deduplicate(records, seed=args.seed)
    write_jsonl(records, args.output)
    write_json_atomic(
        state_path,
        {"repo_fingerprint": fingerprint, "session_count": len(records), "seed": args.seed},
    )
    write_sha256_manifest([args.output, state_path], args.output.parent / "script_sessions.SHA256SUMS")
    print(f"built {len(records)} script sessions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

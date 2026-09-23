#!/usr/bin/env python3
"""Extract benign command sessions from local DevOps/script repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cmdrisk.data import ScriptRepoAdapter, family_aware_deduplicate, normalize_record  # noqa: E402
from cmdrisk.data.io import write_jsonl  # noqa: E402
from cmdrisk.data.manifest import (  # noqa: E402
    sha256_file,
    write_json_atomic,
    write_sha256_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "data" / "interim" / "script_sessions.jsonl"
    )
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


def _git_revision(root: Path) -> str | None:
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip() if result.returncode == 0 else ""
    return revision or None


def _repo_metadata(repos: dict[str, Path]) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for name, root in sorted(repos.items()):
        files = sorted(item for item in root.rglob("*") if item.is_file())
        metadata.append(
            {
                "name": name,
                "path": str(root),
                "revision": _git_revision(root),
                "file_count": len(files),
                "content_sha256": hashlib.sha256(
                    "".join(
                        f"{path.relative_to(root)}\x1f{sha256_file(path)}\n" for path in files
                    ).encode()
                ).hexdigest(),
            }
        )
    return metadata


def _repo_fingerprint(repos: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for entry in _repo_metadata(repos):
        digest.update(
            (
                f"{entry['name']}\x1f{entry['path']}\x1f{entry['revision']}\x1f"
                f"{entry['content_sha256']}\n"
            ).encode()
        )
    return digest.hexdigest()


def main() -> int:
    args = _parse_args()
    repos = _parse_repos(args.repo)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "repos": {key: str(value) for key, value in repos.items()},
                    "output": str(args.output),
                }
            )
        )
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
        {
            "repo_fingerprint": fingerprint,
            "repositories": _repo_metadata(repos),
            "session_count": len(records),
            "seed": args.seed,
        },
    )
    write_sha256_manifest(
        [args.output, state_path], args.output.parent / "script_sessions.SHA256SUMS"
    )
    print(f"built {len(records)} script sessions: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Download and verify pinned public data sources."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--source", action="append", default=[], help="source id to download")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def _http_download(source: dict[str, Any], destination: Path, *, timeout: float) -> None:
    url = str(source["url"])
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "cmd-session-risk-data/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"download failed for {url}: {exc}") from exc
    with response:
        status = getattr(response, "status", 200)
        mode = "ab" if offset and status == 206 else "wb"
        if mode == "wb":
            offset = 0
        with partial.open(mode) as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    if offset and partial.stat().st_size <= offset:
        raise RuntimeError(f"server did not provide resumed bytes for {url}")
    partial.replace(destination)


def _git_download(source: dict[str, Any], destination: Path, *, resume: bool) -> None:
    url = str(source["url"])
    revision = source.get("revision")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (destination / ".git").exists():
        if resume:
            return
        subprocess.run(["git", "-C", str(destination), "fetch", "--all", "--tags"], check=True)
    elif destination.exists():
        raise RuntimeError(f"destination exists and is not a Git repository: {destination}")
    else:
        subprocess.run(["git", "clone", url, str(destination)], check=True)
    if revision:
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", str(revision)], check=True
        )


def main() -> int:
    args = _parse_args()
    sources = load_data_sources(args.config)
    selected = set(args.source)
    if selected:
        known = {str(source["id"]) for source in sources}
        unknown = selected - known
        if unknown:
            raise SystemExit(f"unknown source id(s): {', '.join(sorted(unknown))}")
        sources = [source for source in sources if str(source["id"]) in selected]

    if args.dry_run:
        for source in sources:
            kind = "git" if str(source["url"]).endswith(".git") else "http"
            destination = args.output_dir / str(source["id"])
            print(json.dumps({"id": source["id"], "kind": kind, "destination": str(destination)}))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for source in sources:
        source_id = str(source["id"])
        url = str(source["url"])
        destination = args.output_dir / source_id
        if url.endswith(".git"):
            if args.resume and destination.exists():
                print(f"skip {source_id}: {destination}")
                outputs.append(destination)
                continue
            _git_download(source, destination, resume=args.resume)
            outputs.append(destination)
        else:
            suffix = Path(urllib.parse.urlparse(url).path).suffix or ".bin"
            destination = destination.with_suffix(suffix)
            if args.resume and destination.exists():
                print(f"skip {source_id}: {destination}")
                outputs.append(destination)
                continue
            _http_download(source, destination, timeout=args.timeout)
            outputs.append(destination)

        expected = source.get("sha256")
        if expected and not destination.is_dir():
            actual = sha256_file(destination)
            if actual.casefold() != str(expected).casefold():
                raise RuntimeError(f"SHA256 mismatch for {source_id}: expected {expected}, got {actual}")

    manifest_path = args.output_dir / "SHA256SUMS"
    files = [item for item in outputs if item.is_file()]
    if files:
        write_sha256_manifest(files, manifest_path, root=args.output_dir)
    else:
        manifest_path.write_text("", encoding="utf-8")
    write_json_atomic(args.output_dir / "sources.json", sources)
    print(f"downloaded {len(outputs)} source(s); manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

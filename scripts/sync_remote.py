#!/usr/bin/env python3
"""Synchronise prepared data and training artifacts with a Lightning host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS: Mapping[str, str] = {
    "checkpoints": "artifacts/checkpoints",
    "onnx": "artifacts/onnx",
    "metrics": "artifacts/metrics",
}


class SyncError(RuntimeError):
    """Raised for invalid paths, transfer failures, or checksum mismatches."""


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise SyncError(f"unsafe manifest path: {value!r}")
    return path


def create_sha256_manifest(
    root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    relative_files: Iterable[str | Path] | None = None,
) -> Path:
    """Create a deterministic ``sha256sum``-compatible manifest."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise SyncError(f"manifest root is not a directory: {root_path}")
    destination = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else root_path / "SHA256SUMS"
    )
    if relative_files is None:
        files = sorted(
            (path for path in root_path.rglob("*") if path.is_file() and path != destination),
            key=lambda path: path.relative_to(root_path).as_posix(),
        )
    else:
        files = []
        for raw in relative_files:
            relative = _safe_relative(str(raw))
            candidate = root_path.joinpath(*relative.parts)
            if not candidate.is_file():
                raise SyncError(f"manifest input is not a file: {candidate}")
            files.append(candidate)
        files.sort(key=lambda path: path.relative_to(root_path).as_posix())

    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root_path).as_posix()}"
        for path in files
    ]
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def parse_sha256_manifest(path: str | Path) -> list[tuple[str, str]]:
    manifest = Path(path)
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SyncError(f"cannot read checksum manifest {manifest}: {exc}") from exc

    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise SyncError(f"{manifest}:{line_number}: malformed checksum line")
        digest, raw_relative = fields
        raw_relative = raw_relative.lstrip("*")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise SyncError(f"{manifest}:{line_number}: invalid SHA256 digest")
        relative = _safe_relative(raw_relative).as_posix()
        if relative in seen:
            raise SyncError(f"{manifest}:{line_number}: duplicate path {relative}")
        seen.add(relative)
        entries.append((digest.lower(), relative))
    if not entries:
        raise SyncError(f"checksum manifest is empty: {manifest}")
    return entries


def verify_sha256_manifest(
    root: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Verify every file and return a compact report."""

    root_path = Path(root).expanduser().resolve()
    entries = parse_sha256_manifest(manifest_path)
    failures: list[dict[str, str]] = []
    for expected, relative in entries:
        target = root_path.joinpath(*PurePosixPath(relative).parts)
        try:
            actual = sha256_file(target)
        except OSError as exc:
            failures.append({"path": relative, "error": str(exc)})
            continue
        if actual != expected:
            failures.append(
                {
                    "path": relative,
                    "expected": expected,
                    "actual": actual,
                    "error": "checksum mismatch",
                }
            )
    report = {
        "manifest": str(Path(manifest_path)),
        "root": str(root_path),
        "checked": len(entries),
        "ok": not failures,
        "failures": failures,
    }
    if failures:
        raise SyncError(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


def _rsync_command(
    source: str,
    destination: str,
    *,
    ssh_target: str,
    ssh_port: int | None,
    ssh_key: str | None,
    dry_run: bool,
) -> list[str]:
    command = ["rsync", "-az", "--partial"]
    if dry_run:
        command.append("--dry-run")
    if ssh_target:
        ssh_command = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if ssh_port is not None:
            ssh_command.extend(["-p", str(ssh_port)])
        if ssh_key:
            ssh_command.extend(["-i", str(Path(ssh_key).expanduser())])
        command.extend(["-e", shlex.join(ssh_command)])
    command.extend([source, destination])
    return command


def run_rsync(
    source: str,
    destination: str,
    *,
    ssh_target: str,
    ssh_port: int | None = None,
    ssh_key: str | None = None,
    dry_run: bool = False,
    runner: Any = subprocess.run,
) -> None:
    command = _rsync_command(
        source,
        destination,
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )
    try:
        completed = runner(command, check=False, text=True, capture_output=True)
    except (FileNotFoundError, OSError) as exc:
        raise SyncError(f"cannot execute rsync: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise SyncError(f"rsync failed with exit code {completed.returncode}: {detail}")


def _remote_path(remote_root: str, relative: str) -> str:
    return f"{remote_root.rstrip('/')}/{relative.lstrip('/')}"


def upload_processed(
    *,
    local_root: Path,
    remote_root: str,
    ssh_target: str,
    ssh_port: int | None = None,
    ssh_key: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    source = local_root / "data" / "processed"
    if not source.is_dir():
        raise SyncError(f"processed data directory does not exist: {source}")
    manifest = create_sha256_manifest(source)
    verify_sha256_manifest(source, manifest)
    run_rsync(
        f"{source}/",
        f"{ssh_target}:{_remote_path(remote_root, 'data/processed')}/",
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )
    return {
        "direction": "upload",
        "component": "processed",
        "source": str(source),
        "destination": f"{ssh_target}:{_remote_path(remote_root, 'data/processed')}",
        "manifest": str(manifest),
        "dry_run": dry_run,
    }


def download_artifacts(
    *,
    local_root: Path,
    remote_root: str,
    ssh_target: str,
    components: Sequence[str],
    ssh_port: int | None = None,
    ssh_key: str | None = None,
    dry_run: bool = False,
    require_manifests: bool = True,
) -> dict[str, Any]:
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise SyncError(f"unknown components: {', '.join(unknown)}")
    if not components:
        raise SyncError("at least one download component is required")

    manifest_root = local_root / "manifests"
    reports: dict[str, Any] = {}
    for component in components:
        local_relative = COMPONENTS[component]
        destination = local_root / local_relative
        destination.mkdir(parents=True, exist_ok=True)
        run_rsync(
            f"{ssh_target}:{_remote_path(remote_root, local_relative)}/",
            f"{destination}/",
            ssh_target=ssh_target,
            ssh_port=ssh_port,
            ssh_key=ssh_key,
            dry_run=dry_run,
        )
        reports[component] = {
            "destination": str(destination),
            "manifest": str(manifest_root / f"{component}.sha256"),
        }

    manifest_root.mkdir(parents=True, exist_ok=True)
    run_rsync(
        f"{ssh_target}:{_remote_path(remote_root, 'manifests')}/",
        f"{manifest_root}/",
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        ssh_key=ssh_key,
        dry_run=dry_run,
    )

    if not dry_run:
        for component in components:
            manifest = manifest_root / f"{component}.sha256"
            if not manifest.is_file():
                if require_manifests:
                    raise SyncError(f"remote checksum manifest is missing: {manifest}")
                continue
            reports[component]["verification"] = verify_sha256_manifest(
                local_root / COMPONENTS[component], manifest
            )
    return {
        "direction": "download",
        "components": list(components),
        "ssh_target": ssh_target,
        "remote_root": remote_root,
        "dry_run": dry_run,
        "reports": reports,
    }


def _local_root(value: str | None) -> Path:
    return Path(value).expanduser().resolve() if value else ROOT


def _add_connection_options(parser: argparse.ArgumentParser, *, require_host: bool) -> None:
    parser.add_argument(
        "--ssh-target",
        default=os.environ.get("LIGHTNING_SSH_TARGET"),
        required=require_host and not os.environ.get("LIGHTNING_SSH_TARGET"),
        help="SSH destination such as user@ssh.lightning.ai (or LIGHTNING_SSH_TARGET).",
    )
    parser.add_argument("--remote-root", default="~/cmd-session-risk")
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("--ssh-key", default=os.environ.get("LIGHTNING_SSH_KEY"))
    parser.add_argument("--dry-run", action="store_true")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    upload = subparsers.add_parser("upload", help="Upload data/processed with a local manifest.")
    _add_connection_options(upload, require_host=True)
    upload.add_argument("--local-root", default=None)

    download = subparsers.add_parser(
        "download", help="Download checkpoints, ONNX artifacts, and metrics with verification."
    )
    _add_connection_options(download, require_host=True)
    download.add_argument("--local-root", default=None)
    download.add_argument(
        "--components",
        nargs="+",
        choices=tuple(COMPONENTS),
        default=list(COMPONENTS),
    )
    download.add_argument(
        "--allow-missing-manifests",
        action="store_true",
        help="Download despite missing remote manifests (unsafe; not recommended).",
    )

    verify = subparsers.add_parser("verify", help="Verify a local SHA256 manifest.")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    manifest = subparsers.add_parser("manifest", help="Create a SHA256 manifest.")
    manifest.add_argument("--root", type=Path, required=True)
    manifest.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "upload":
            report = upload_processed(
                local_root=_local_root(args.local_root),
                remote_root=args.remote_root,
                ssh_target=args.ssh_target,
                ssh_port=args.ssh_port,
                ssh_key=args.ssh_key,
                dry_run=args.dry_run,
            )
        elif args.command == "download":
            report = download_artifacts(
                local_root=_local_root(args.local_root),
                remote_root=args.remote_root,
                ssh_target=args.ssh_target,
                components=args.components,
                ssh_port=args.ssh_port,
                ssh_key=args.ssh_key,
                dry_run=args.dry_run,
                require_manifests=not args.allow_missing_manifests,
            )
        elif args.command == "verify":
            report = verify_sha256_manifest(args.root, args.manifest)
        else:
            manifest_path = create_sha256_manifest(args.root, args.manifest)
            report = {"manifest": str(manifest_path), "ok": True}
    except SyncError as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebuild an ONNX artifact manifest's integrity block in place.

Used when files are added to an already-exported artifact directory - most
commonly a post-hoc ``model.int8.onnx`` produced by ONNX Runtime dynamic
quantization.  The script preserves every other manifest field (parity,
metrics, latency, promotion) and only rewrites the ``artifacts`` /
``checksums`` / ``sizes`` blocks so the bundle stays fail-closed verifiable.

The top-level ``model_version`` is re-derived from ``runtime_config.json`` -
the single source of truth written by the exporter - so a refreshed bundle
keeps the same shape as a freshly exported one.  ``--model-version`` pins it
explicitly, and a disagreement between the two is a hard error.

Example::

    python scripts/refresh_artifact_manifest.py \
        --artifact-dir artifacts/run/onnx --int8-model model.int8.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cmdrisk.artifact_manifest import (  # noqa: E402
    EXPORT_MANIFEST_FILENAME,
    EXPORT_MANIFEST_SCHEMA,
    EXPORT_MANIFEST_VERSION,
    build_artifact_entries,
    verify_export_manifest,
    write_manifest_atomic,
)


def _collect_files(root: Path) -> list[str]:
    manifest = root / EXPORT_MANIFEST_FILENAME
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    )


def _runtime_model_version(root: Path) -> str | None:
    """Return ``model_version`` from ``runtime_config.json`` when available."""

    path = root / "runtime_config.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"runtime_config.json is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("runtime_config.json must contain a JSON object")
    value = payload.get("model_version")
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--fp32-model", default="model.onnx")
    parser.add_argument("--int8-model", default=None)
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create a manifest when one does not exist yet",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Pin the manifest model_version. Defaults to the value recorded in "
            "runtime_config.json when that file is present."
        ),
    )
    args = parser.parse_args(argv)

    root = Path(args.artifact_dir).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"artifact directory does not exist: {root}")
    manifest_path = root / EXPORT_MANIFEST_FILENAME

    if manifest_path.is_file():
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif args.create:
        manifest = {
            "schema": EXPORT_MANIFEST_SCHEMA,
            "version": EXPORT_MANIFEST_VERSION,
            "format": "onnx",
        }
    else:
        raise SystemExit(f"no manifest at {manifest_path}; pass --create to build a minimal one")

    if not (root / args.fp32_model).is_file():
        raise SystemExit(f"fp32 model not found: {root / args.fp32_model}")
    manifest["fp32_model"] = args.fp32_model
    if args.int8_model is not None:
        if not (root / args.int8_model).is_file():
            raise SystemExit(f"int8 model not found: {root / args.int8_model}")
        manifest["int8_model"] = args.int8_model
    elif not (root / str(manifest.get("int8_model", ""))).is_file():
        manifest["int8_model"] = None

    runtime_version = _runtime_model_version(root)
    pinned = args.model_version.strip() if isinstance(args.model_version, str) else None
    if pinned == "":
        pinned = None
    if pinned is not None and runtime_version is not None and pinned != runtime_version:
        raise SystemExit(
            "model version mismatch: --model-version "
            f"{pinned!r} != runtime_config.json {runtime_version!r}"
        )
    resolved_version = pinned or runtime_version or manifest.get("model_version")
    recorded = manifest.get("model_version")
    if isinstance(recorded, str) and runtime_version is not None and recorded != runtime_version:
        raise SystemExit(
            "manifest model_version "
            f"{recorded!r} disagrees with runtime_config.json {runtime_version!r}"
        )
    if resolved_version is not None:
        manifest["model_version"] = resolved_version

    entries = build_artifact_entries(root, _collect_files(root))
    manifest["artifacts"] = entries
    manifest["checksums"] = {name: entry["sha256"] for name, entry in entries.items()}
    manifest["sizes"] = {name: entry["size"] for name, entry in entries.items()}

    write_manifest_atomic(manifest_path, manifest)
    verified = verify_export_manifest(root, manifest_path)
    print(
        json.dumps(
            {
                "manifest": verified["manifest"],
                "declared_files": verified["checked"],
                "fp32_model": manifest["fp32_model"],
                "int8_model": manifest.get("int8_model"),
                "model_version": manifest.get("model_version"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

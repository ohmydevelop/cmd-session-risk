"""Integrity helpers for self-contained model artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

EXPORT_MANIFEST_SCHEMA = "cmdrisk.onnx.artifact"
EXPORT_MANIFEST_VERSION = 1
EXPORT_MANIFEST_FILENAME = "export_manifest.json"


class ArtifactManifestError(RuntimeError):
    """Raised when an artifact manifest is missing, unsafe, or inconsistent."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA256 digest for ``path``."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_relative(value: Any) -> str:
    """Validate and normalise a manifest-relative POSIX path."""

    if not isinstance(value, str) or not value:
        raise ArtifactManifestError("artifact path must be a non-empty string")
    # Backslashes are not path separators on POSIX, but accepting them makes
    # manifests non-portable and can hide traversal attempts on other systems.
    if "\\" in value:
        raise ArtifactManifestError(f"unsafe artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ArtifactManifestError(f"unsafe artifact path: {value!r}")
    if path.parts[0] == EXPORT_MANIFEST_FILENAME:
        raise ArtifactManifestError("the export manifest cannot contain itself")
    return path.as_posix()


def resolve_artifact_path(root: str | Path, relative: Any) -> Path:
    """Resolve a declared artifact path without allowing an escape from root."""

    root_path = Path(root).expanduser().resolve()
    normalised = safe_artifact_relative(relative)
    candidate = root_path.joinpath(*PurePosixPath(normalised).parts)
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root_path)
    except (OSError, ValueError) as exc:
        raise ArtifactManifestError(
            f"artifact path escapes the artifact directory: {normalised!r}"
        ) from exc
    return candidate


def artifact_entry(root: str | Path, relative: Any) -> dict[str, Any]:
    """Build the size and SHA256 entry stored in an export manifest."""

    normalised = safe_artifact_relative(relative)
    path = resolve_artifact_path(root, normalised)
    try:
        size = path.stat().st_size
        digest = sha256_file(path)
    except OSError as exc:
        raise ArtifactManifestError(f"cannot hash artifact {normalised}: {exc}") from exc
    if not path.is_file():
        raise ArtifactManifestError(f"declared artifact is not a file: {normalised}")
    return {"path": normalised, "size": int(size), "sha256": digest}


def build_artifact_entries(
    root: str | Path,
    relative_files: Sequence[str],
    *,
    extra_files: Sequence[str] = (),
) -> dict[str, dict[str, Any]]:
    """Return a deterministic path-to-integrity mapping."""

    names = list(dict.fromkeys([*relative_files, *extra_files]))
    return {name: artifact_entry(root, name) for name in sorted(names)}


def _manifest_payload(
    root: Path,
    manifest_path: str | Path | None,
    payload: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else (root / EXPORT_MANIFEST_FILENAME)
    )
    if payload is not None:
        return path, dict(payload)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArtifactManifestError(
            f"required artifact manifest not found: {path.name}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactManifestError(f"cannot load artifact manifest {path.name}: {exc}") from exc
    if not isinstance(loaded, Mapping):
        raise ArtifactManifestError("export manifest must contain a JSON object")
    return path, dict(loaded)


def verify_export_manifest(
    root: str | Path,
    manifest_path: str | Path | None = None,
    *,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify schema, declared files, sizes, and SHA256 digests.

    Verification is intentionally fail-closed: missing entries, malformed
    metadata, unsupported versions, path traversal, modified files, and
    symlinks that resolve outside the artifact directory are rejected.
    """

    root_path = Path(root).expanduser().resolve()
    if not root_path.is_dir():
        raise ArtifactManifestError(f"artifact directory does not exist: {root_path}")
    path, manifest = _manifest_payload(root_path, manifest_path, payload)

    if manifest.get("schema") != EXPORT_MANIFEST_SCHEMA:
        raise ArtifactManifestError(
            f"unsupported export manifest schema: {manifest.get('schema')!r}"
        )
    if manifest.get("version") != EXPORT_MANIFEST_VERSION:
        raise ArtifactManifestError(
            f"unsupported export manifest version: {manifest.get('version')!r}"
        )

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or not raw_artifacts:
        raise ArtifactManifestError("export manifest artifacts must be a non-empty object")

    checked: list[dict[str, Any]] = []
    for raw_relative, raw_entry in sorted(raw_artifacts.items(), key=lambda item: str(item[0])):
        normalised = safe_artifact_relative(raw_relative)
        if not isinstance(raw_entry, Mapping):
            raise ArtifactManifestError(
                f"artifact entry must be an object with size and sha256: {normalised}"
            )
        expected_digest = raw_entry.get("sha256")
        expected_size = raw_entry.get("size")
        if not isinstance(expected_digest, str) or len(expected_digest) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in expected_digest
        ):
            raise ArtifactManifestError(f"invalid SHA256 for artifact: {normalised}")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ArtifactManifestError(f"invalid size for artifact: {normalised}")

        candidate = resolve_artifact_path(root_path, normalised)
        try:
            if not candidate.is_file():
                raise ArtifactManifestError(f"declared artifact is missing: {normalised}")
            actual_size = candidate.stat().st_size
        except OSError as exc:
            raise ArtifactManifestError(f"cannot stat artifact {normalised}: {exc}") from exc
        if actual_size != expected_size:
            raise ArtifactManifestError(
                f"artifact size mismatch for {normalised}: "
                f"expected {expected_size}, got {actual_size}"
            )
        try:
            actual_digest = sha256_file(candidate)
        except OSError as exc:
            raise ArtifactManifestError(f"cannot hash artifact {normalised}: {exc}") from exc
        if actual_digest.lower() != expected_digest.lower():
            raise ArtifactManifestError(
                f"artifact SHA256 mismatch for {normalised}: "
                f"expected {expected_digest.lower()}, got {actual_digest.lower()}"
            )
        checked.append(
            {
                "path": normalised,
                "size": actual_size,
                "sha256": actual_digest.lower(),
            }
        )

    fp32_model = manifest.get("fp32_model", "model.onnx")
    safe_artifact_relative(fp32_model)
    if fp32_model not in raw_artifacts:
        raise ArtifactManifestError("fp32_model is not listed in artifacts")
    int8_model = manifest.get("int8_model")
    if int8_model is not None:
        safe_artifact_relative(int8_model)
        if int8_model not in raw_artifacts:
            raise ArtifactManifestError("int8_model is not listed in artifacts")

    return {
        "manifest": str(path),
        "root": str(root_path),
        "schema": EXPORT_MANIFEST_SCHEMA,
        "version": EXPORT_MANIFEST_VERSION,
        "checked": len(checked),
        "artifacts": checked,
        "ok": True,
    }


def write_manifest_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a deterministic JSON manifest."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise

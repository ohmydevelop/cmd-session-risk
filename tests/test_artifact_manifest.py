from __future__ import annotations

import json
from pathlib import Path

import pytest

from cmdrisk.artifact_manifest import (
    EXPORT_MANIFEST_FILENAME,
    EXPORT_MANIFEST_SCHEMA,
    EXPORT_MANIFEST_VERSION,
    ArtifactManifestError,
    artifact_entry,
    build_artifact_entries,
    resolve_artifact_path,
    safe_artifact_relative,
    sha256_file,
    verify_export_manifest,
    write_manifest_atomic,
)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _manifest_for(root: Path, names: list[str], **extra: object) -> dict[str, object]:
    artifacts = build_artifact_entries(root, names)
    manifest: dict[str, object] = {
        "schema": EXPORT_MANIFEST_SCHEMA,
        "version": EXPORT_MANIFEST_VERSION,
        "fp32_model": "model.onnx",
        "int8_model": "model.int8.onnx" if "model.int8.onnx" in names else None,
        "artifacts": artifacts,
    }
    manifest.update(extra)
    return manifest


def test_safe_artifact_relative_rejects_unsafe_paths() -> None:
    assert safe_artifact_relative("model.onnx") == "model.onnx"
    assert safe_artifact_relative("sub/model.onnx") == "sub/model.onnx"
    for bad in ("", "/etc/passwd", "../x", "a/../../b", "a\\b", "."):
        with pytest.raises(ArtifactManifestError):
            safe_artifact_relative(bad)
    # Interior "." segments are harmless and normalised away.
    assert safe_artifact_relative("./x") == "x"
    assert safe_artifact_relative("x/./y") == "x/y"
    with pytest.raises(ArtifactManifestError):
        safe_artifact_relative(EXPORT_MANIFEST_FILENAME)


def test_artifact_entry_and_digest(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"payload")
    entry = artifact_entry(tmp_path, "model.onnx")
    assert entry["size"] == len(b"payload")
    assert entry["sha256"] == sha256_file(model)


def test_write_and_verify_round_trip(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"fp32")
    (tmp_path / "model.int8.onnx").write_bytes(b"int8")
    manifest = _manifest_for(tmp_path, ["model.onnx", "model.int8.onnx"])
    write_manifest_atomic(tmp_path / EXPORT_MANIFEST_FILENAME, manifest)
    result = verify_export_manifest(tmp_path)
    assert result["ok"] is True
    assert result["checked"] == 2
    assert result["schema"] == EXPORT_MANIFEST_SCHEMA
    assert result["version"] == EXPORT_MANIFEST_VERSION


def test_verify_rejects_missing_manifest(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    with pytest.raises(ArtifactManifestError, match="not found"):
        verify_export_manifest(tmp_path)


def test_verify_rejects_bad_schema_and_version(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    write_manifest_atomic(
        tmp_path / EXPORT_MANIFEST_FILENAME,
        _manifest_for(tmp_path, ["model.onnx"], schema="evil", version=2),
    )
    with pytest.raises(ArtifactManifestError):
        verify_export_manifest(tmp_path)


def test_verify_rejects_tampered_file(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"original")
    write_manifest_atomic(
        tmp_path / EXPORT_MANIFEST_FILENAME, _manifest_for(tmp_path, ["model.onnx"])
    )
    (tmp_path / "model.onnx").write_bytes(b"tampered!!")
    with pytest.raises(ArtifactManifestError, match="size mismatch|SHA256 mismatch"):
        verify_export_manifest(tmp_path)


def test_verify_rejects_missing_declared_file(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    (tmp_path / "model.int8.onnx").write_bytes(b"y")
    manifest = _manifest_for(tmp_path, ["model.onnx", "model.int8.onnx"])
    (tmp_path / "model.int8.onnx").unlink()
    write_manifest_atomic(tmp_path / EXPORT_MANIFEST_FILENAME, manifest)
    with pytest.raises(ArtifactManifestError, match="missing"):
        verify_export_manifest(tmp_path)


def test_verify_rejects_traversal_entry(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    manifest = _manifest_for(tmp_path, ["model.onnx"])
    manifest["artifacts"]["../escape"] = {"size": 1, "sha256": "0" * 64}
    write_manifest_atomic(tmp_path / EXPORT_MANIFEST_FILENAME, manifest)
    with pytest.raises(ArtifactManifestError):
        verify_export_manifest(tmp_path)


def test_verify_rejects_string_entry(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"x")
    digest = sha256_file(tmp_path / "model.onnx")
    manifest = _manifest_for(tmp_path, ["model.onnx"])
    manifest["artifacts"]["model.onnx"] = digest
    write_manifest_atomic(tmp_path / EXPORT_MANIFEST_FILENAME, manifest)
    with pytest.raises(ArtifactManifestError):
        verify_export_manifest(tmp_path)


def test_verify_rejects_undeclared_fp32_model(tmp_path: Path) -> None:
    (tmp_path / "other.onnx").write_bytes(b"x")
    manifest = _manifest_for(tmp_path, ["other.onnx"])
    manifest["fp32_model"] = "model.onnx"
    write_manifest_atomic(tmp_path / EXPORT_MANIFEST_FILENAME, manifest)
    with pytest.raises(ArtifactManifestError, match="fp32_model"):
        verify_export_manifest(tmp_path)


def test_resolve_artifact_path_stays_in_root(tmp_path: Path) -> None:
    resolved = resolve_artifact_path(tmp_path, "sub/model.onnx")
    assert resolved == tmp_path / "sub" / "model.onnx"
    with pytest.raises(ArtifactManifestError):
        resolve_artifact_path(tmp_path, "../escape")

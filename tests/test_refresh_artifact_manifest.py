from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "refresh_artifact_manifest.py"
    spec = importlib.util.spec_from_file_location("refresh_artifact_manifest", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


refresh = _load_module()


def _artifact(tmp_path: Path, *, runtime_version: str | None = "run-42") -> Path:
    root = tmp_path / "onnx"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"stub-model")
    if runtime_version is not None:
        (root / "runtime_config.json").write_text(
            json.dumps({"model_version": runtime_version, "extra": "keep"}),
            encoding="utf-8",
        )
    (root / "export_manifest.json").write_text(
        json.dumps(
            {
                "schema": "cmdrisk.onnx.artifact",
                "version": 1,
                "format": "onnx",
                "fp32_model": "model.onnx",
                "int8_model": None,
                "parity_max_abs_error": 1e-06,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_refresh_derives_model_version_from_runtime_config(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    assert refresh.main(["--artifact-dir", str(root)]) == 0
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_version"] == "run-42"
    assert manifest["parity_max_abs_error"] == 1e-06
    assert set(manifest["artifacts"]) == {"model.onnx", "runtime_config.json"}
    assert manifest["checksums"]["model.onnx"] == manifest["artifacts"]["model.onnx"]["sha256"]


def test_refresh_honours_an_explicit_model_version(tmp_path: Path) -> None:
    root = _artifact(tmp_path, runtime_version=None)
    assert refresh.main(
        ["--artifact-dir", str(root), "--model-version", "pinned-1.0"]
    ) == 0
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_version"] == "pinned-1.0"


def test_refresh_rejects_conflicting_model_version(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    with pytest.raises(SystemExit) as excinfo:
        refresh.main(["--artifact-dir", str(root), "--model-version", "other"])
    assert "model version mismatch" in str(excinfo.value)


def test_refresh_rejects_manifest_runtime_disagreement(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    manifest_path = root / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_version"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit) as excinfo:
        refresh.main(["--artifact-dir", str(root)])
    assert "disagrees with runtime_config.json" in str(excinfo.value)


def test_refresh_keeps_int8_name_when_the_file_is_missing(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    manifest_path = root / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["int8_model"] = "model.int8.onnx"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert refresh.main(["--artifact-dir", str(root)]) == 0
    updated = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated["int8_model"] is None


def test_refresh_is_idempotent(tmp_path: Path) -> None:
    root = _artifact(tmp_path)
    assert refresh.main(["--artifact-dir", str(root)]) == 0
    first = (root / "export_manifest.json").read_bytes()
    assert refresh.main(["--artifact-dir", str(root)]) == 0
    assert (root / "export_manifest.json").read_bytes() == first


def test_refresh_create_builds_a_minimal_manifest(tmp_path: Path) -> None:
    root = _artifact(tmp_path, runtime_version=None)
    (root / "export_manifest.json").unlink()
    assert refresh.main(["--artifact-dir", str(root), "--create"]) == 0
    manifest = json.loads((root / "export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == refresh.EXPORT_MANIFEST_SCHEMA
    assert set(manifest["artifacts"]) == {"model.onnx"}

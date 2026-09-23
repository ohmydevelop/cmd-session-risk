from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "latency_bench.py"
    spec = importlib.util.spec_from_file_location("latency_bench", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_module()


def _passing_profile() -> dict:
    return {
        "platform": "Linux-test",
        "os": "Linux",
        "architecture": "x86_64",
        "processor_model": "Test CPU",
        "logical_cpus": 16,
        "affinity_cpus": [0, 1, 2, 3, 4, 5, 6, 7],
        "affinity_count": 8,
        "avx2": True,
        "onnxruntime_version": "1.20.0",
        "providers": ["CPUExecutionProvider"],
    }


def test_parse_cpu_affinity() -> None:
    assert bench.parse_cpu_affinity(None) is None
    assert bench.parse_cpu_affinity("0-3") == [0, 1, 2, 3]
    assert bench.parse_cpu_affinity("0,2,4") == [0, 2, 4]
    assert bench.parse_cpu_affinity("0-1,4") == [0, 1, 4]
    with pytest.raises(ValueError):
        bench.parse_cpu_affinity("3-1")
    with pytest.raises(ValueError):
        bench.parse_cpu_affinity("-1")


def test_percentile() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bench.percentile(values, 0.5) == pytest.approx(3.0)
    assert bench.percentile(values, 0.0) == pytest.approx(1.0)
    assert bench.percentile(values, 1.0) == pytest.approx(5.0)
    assert bench.percentile([7.0], 0.99) == pytest.approx(7.0)


def test_prefix_for_tokens_rejects_request_beyond_artifact_max_length() -> None:
    class Scorer:
        max_length = 256

    with pytest.raises(ValueError, match="exceeds artifact max_length 256"):
        bench._prefix_for_tokens(Scorer(), 512)


def test_evaluate_profile_accepts_documented_profile() -> None:
    passed, failures = bench.evaluate_profile(_passing_profile())
    assert passed is True
    assert failures == []


def test_evaluate_profile_rejects_weak_hardware() -> None:
    profile = _passing_profile()
    profile["architecture"] = "aarch64"
    profile["logical_cpus"] = 4
    profile["avx2"] = False
    passed, failures = bench.evaluate_profile(profile)
    assert passed is False
    assert "architecture_not_x86_64" in failures
    assert "insufficient_logical_cpus<8" in failures
    assert "avx2_missing" in failures


def test_evaluate_profile_treats_unknown_avx2_as_failure() -> None:
    profile = _passing_profile()
    profile["avx2"] = None
    passed, failures = bench.evaluate_profile(profile)
    assert passed is False
    assert "avx2_unknown" in failures


def _results(p95: float, p99: float) -> list[dict]:
    return [
        {
            "requested_tokens": 2048,
            "actual_tokens": 2048,
            "p50_ms": 1.0,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": 1.0,
        },
        {
            "requested_tokens": 8192,
            "actual_tokens": 8192,
            "p50_ms": 2.0,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": 2.0,
        },
    ]


def test_evaluate_latency_within_tolerance_passes() -> None:
    outcome = bench.evaluate_latency(
        _results(55.0, 75.0), p95_threshold_ms=50.0, p99_threshold_ms=70.0, tolerance_ms=10.0
    )
    assert outcome["passed"] is True
    assert outcome["failures"] == []
    assert outcome["p95_ms"] == pytest.approx(55.0)
    assert outcome["p99_ms"] == pytest.approx(75.0)


def test_evaluate_latency_beyond_tolerance_fails() -> None:
    outcome = bench.evaluate_latency(
        _results(61.0, 90.0), p95_threshold_ms=50.0, p99_threshold_ms=70.0, tolerance_ms=10.0
    )
    assert outcome["passed"] is False
    assert "p95_exceeds_tolerance" in outcome["failures"]
    assert "p99_exceeds_tolerance" in outcome["failures"]


def test_build_report_shape_is_export_compatible() -> None:
    report = bench.build_report(
        artifact_dir="/tmp/artifact",
        profile=_passing_profile(),
        results=_results(40.0, 60.0),
        cpu_affinity_cpus=[0, 1, 2, 3],
        p95_threshold_ms=50.0,
        p99_threshold_ms=70.0,
        tolerance_ms=10.0,
    )
    assert report["schema"] == bench.LATENCY_REPORT_SCHEMA
    assert report["profile_passed"] is True
    assert report["passed"] is True
    assert report["p95_threshold_ms"] == 50.0
    assert report["p99_threshold_ms"] == 70.0
    assert report["tolerance_ms"] == 10.0
    assert report["p95_ms"] == 40.0
    assert report["p99_ms"] == 60.0
    assert report["profile"]["passed"] is True
    # Fields consumed by cmdrisk.training.export._latency_report_status.
    for key in (
        "profile",
        "profile_passed",
        "p95_ms",
        "p99_ms",
        "p95_threshold_ms",
        "p99_threshold_ms",
        "tolerance_ms",
        "passed",
    ):
        assert key in report


def test_build_report_fails_closed_on_slow_run() -> None:
    report = bench.build_report(
        artifact_dir="/tmp/artifact",
        profile=_passing_profile(),
        results=_results(500.0, 900.0),
        cpu_affinity_cpus=None,
        p95_threshold_ms=50.0,
        p99_threshold_ms=70.0,
        tolerance_ms=10.0,
    )
    assert report["passed"] is False
    assert "p95_exceeds_tolerance" in report["failures"]


def test_deployment_profile_is_cpu_for_the_documented_provider() -> None:
    assert bench.deployment_profile(["CPUExecutionProvider"]) == "cpu"
    assert bench.deployment_profile([]) == "accelerated"
    assert (
        bench.deployment_profile(["CUDAExecutionProvider", "CPUExecutionProvider"])
        == "accelerated"
    )


def test_build_report_rejects_accelerated_deployment_profile() -> None:
    profile = _passing_profile()
    profile["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    report = bench.build_report(
        artifact_dir="/tmp/artifact",
        profile=profile,
        results=_results(10.0, 15.0),
        cpu_affinity_cpus=None,
        p95_threshold_ms=50.0,
        p99_threshold_ms=70.0,
        tolerance_ms=10.0,
    )
    assert report["deployment_profile"] == "accelerated"
    assert report["passed"] is False
    assert "latency_not_cpu_deployment_profile" in report["failures"]


def test_build_report_records_cpu_deployment_profile() -> None:
    report = bench.build_report(
        artifact_dir="/tmp/artifact",
        profile=_passing_profile(),
        results=_results(10.0, 15.0),
        cpu_affinity_cpus=None,
        p95_threshold_ms=50.0,
        p99_threshold_ms=70.0,
        tolerance_ms=10.0,
    )
    assert report["deployment_profile"] == "cpu"
    assert report["passed"] is True


class _FakeOnnxSession:
    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


class _FakeScorer:
    def __init__(self) -> None:
        self._onnx_session = _FakeOnnxSession()

    def _encode(self, commands):
        import numpy as np

        return {"input_ids": np.zeros((1, max(1, len(commands))), dtype=np.int64)}

    def score_commands(self, commands) -> None:
        return None


def test_main_requires_output_for_update_manifest(capsys) -> None:
    code = bench.main(["--artifact-dir", "/tmp/x", "--update-manifest"])
    assert code == 2
    captured = capsys.readouterr()
    assert "--update-manifest requires --output" in captured.err


def test_main_writes_report_and_update_manifest(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        bench.SessionScorer, "from_onnx", classmethod(lambda cls, *a, **k: _FakeScorer())
    )
    monkeypatch.setattr(bench, "hardware_profile", lambda **kwargs: _passing_profile())

    calls: dict[str, object] = {}

    def fake_update(*, artifact_dir, latency_report_path, require_promotion=False):
        calls["artifact_dir"] = str(artifact_dir)
        calls["latency_report_path"] = str(latency_report_path)
        calls["require_promotion"] = require_promotion
        return {}

    import cmdrisk.training.export as export_module

    monkeypatch.setattr(export_module, "update_export_manifest_latency", fake_update)

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    output = tmp_path / "latency.json"
    code = bench.main(
        [
            "--artifact-dir",
            str(artifact),
            "--lengths",
            "8",
            "--warmup",
            "0",
            "--iterations",
            "1",
            "--cpu-affinity",
            "0-7",
            "--output",
            str(output),
            "--update-manifest",
            "--require-promotion",
            "--format",
            "json",
        ]
    )
    capsys.readouterr()
    assert code == 0, calls
    assert output.is_file()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert calls["latency_report_path"] == str(output)
    assert calls["require_promotion"] is True


def test_main_fails_closed_on_weak_profile(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        bench.SessionScorer, "from_onnx", classmethod(lambda cls, *a, **k: _FakeScorer())
    )

    weak = _passing_profile()
    weak["avx2"] = False
    monkeypatch.setattr(bench, "hardware_profile", lambda **kwargs: weak)

    code = bench.main(
        ["--artifact-dir", str(tmp_path), "--lengths", "8", "--iterations", "1"]
    )
    capsys.readouterr()
    assert code == 1

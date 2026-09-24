from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_validation.py"


def _session(session_id: str, commands: int, *, malicious: int, onset: int = 0) -> dict:
    return {
        "session_id": session_id,
        "source": {"kind": "real_malicious" if malicious else "real_benign", "name": "fixture"},
        "commands": [{"cmd": "true", "dt": None} for _ in range(commands)],
        "label": {
            "malicious": malicious,
            "first_evidence_index": onset,
            "tactics": [],
            "techniques": [],
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )


def test_evaluate_validation_reports_recall_and_coverage(tmp_path: Path) -> None:
    test_path = tmp_path / "test.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    output_path = tmp_path / "metrics.json"

    _write_jsonl(
        test_path,
        [
            _session("s-mal", 3, malicious=1, onset=1),
            _session("s-benign", 2, malicious=0),
        ],
    )
    _write_jsonl(
        scores_path,
        [
            {
                "session_id": "s-mal",
                "command_index": 0,
                "malicious_probability": 0.10,
                "alert": False,
                "model_version": "run-1",
            },
            {
                "session_id": "s-mal",
                "command_index": 1,
                "malicious_probability": 0.90,
                "alert": True,
                "model_version": "run-1",
            },
            {
                "session_id": "s-mal",
                "command_index": 2,
                "malicious_probability": 0.95,
                "alert": True,
                "model_version": "run-1",
            },
            {
                "session_id": "s-benign",
                "command_index": 0,
                "malicious_probability": 0.05,
                "alert": False,
                "model_version": "run-1",
            },
            {
                "session_id": "s-benign",
                "command_index": 1,
                "malicious_probability": 0.20,
                "alert": False,
                "model_version": "run-1",
            },
        ],
    )

    result = _run(
        "--scores",
        str(scores_path),
        "--test",
        str(test_path),
        "--threshold",
        "0.5",
        "--output",
        str(output_path),
        "--expected-line-count",
        "5",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["schema"] == "cmdrisk.validation_metrics"
    assert report["line_count"] == 5
    assert report["model_versions"] == ["run-1"]
    coverage = report["coverage"]
    assert coverage["scored_sessions"] == 2
    assert coverage["missing_sessions"] == []
    assert coverage["prefix_count_mismatch"] == {}

    malicious = report["malicious"]
    assert malicious["true_positive"] == 2
    assert malicious["false_negative"] == 0
    assert malicious["true_negative"] == 3
    assert malicious["false_positive"] == 0
    assert malicious["recall"] == 1.0
    assert malicious["fpr"] == 0.0

    delay = report["detection_delay"]
    assert delay["malicious_sessions"] == 1
    assert delay["detected_sessions"] == 1
    assert delay["median_detection_delay_commands"] == 0.0
    assert delay["false_positive_sessions"] == 0


def test_evaluate_validation_flags_missing_sessions(tmp_path: Path) -> None:
    test_path = tmp_path / "test.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(
        test_path,
        [_session("s-a", 1, malicious=1), _session("s-b", 1, malicious=0)],
    )
    _write_jsonl(
        scores_path,
        [
            {
                "session_id": "s-a",
                "command_index": 0,
                "malicious_probability": 0.9,
                "alert": True,
            }
        ],
    )

    result = _run("--scores", str(scores_path), "--test", str(test_path), "--threshold", "0.5")
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["coverage"]["missing_sessions"] == ["s-b"]


def test_evaluate_validation_rejects_unknown_sessions(tmp_path: Path) -> None:
    test_path = tmp_path / "test.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    _write_jsonl(test_path, [_session("s-a", 1, malicious=1)])
    _write_jsonl(
        scores_path,
        [
            {
                "session_id": "s-other",
                "command_index": 0,
                "malicious_probability": 0.9,
                "alert": True,
            }
        ],
    )

    result = _run("--scores", str(scores_path), "--test", str(test_path), "--threshold", "0.5")
    assert result.returncode != 0
    assert "unknown sessions" in result.stderr


def test_evaluate_validation_uses_artifact_calibration(tmp_path: Path) -> None:
    test_path = tmp_path / "test.jsonl"
    scores_path = tmp_path / "scores.jsonl"
    artifact = tmp_path / "onnx"
    artifact.mkdir()
    (artifact / "calibration.json").write_text(
        json.dumps({"malicious_threshold": 0.5, "version": 1}), encoding="utf-8"
    )
    _write_jsonl(test_path, [_session("s-a", 1, malicious=1)])
    _write_jsonl(
        scores_path,
        [
            {
                "session_id": "s-a",
                "command_index": 0,
                "malicious_probability": 0.6,
                "alert": True,
            }
        ],
    )

    result = _run(
        "--scores", str(scores_path), "--test", str(test_path), "--artifact-dir", str(artifact)
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["threshold"] == 0.5

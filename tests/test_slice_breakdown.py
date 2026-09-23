from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "slice_breakdown.py"

SESSIONS = [
    ("session-real-malicious", "real_malicious", 1, 0.99),
    ("session-real-benign", "real_benign", 0, 0.01),
    ("session-script-benign", "script_benign", 0, 0.02),
    ("session-llm-variant", "llm_variant", 1, 0.97),
]


def _write_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "sessions.jsonl"
    scores = tmp_path / "scores.jsonl"
    with data.open("w", encoding="utf-8") as data_handle, scores.open("w", encoding="utf-8") as score_handle:
        for session_id, kind, label, probability in SESSIONS:
            record = {
                "schema_version": "1.0",
                "session_id": session_id,
                "source": {"kind": kind, "name": "fixture"},
                "family_id": f"family:{session_id}",
                "commands": [
                    {"cmd": "whoami", "dt": None},
                    {"cmd": "id", "dt": 0.5},
                ],
                "label": {"malicious": label, "tactics": [], "techniques": []},
                "split": "test",
            }
            data_handle.write(json.dumps(record) + "\n")
            for index in range(len(record["commands"])):
                score_handle.write(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "command_index": index,
                            "malicious_probability": probability,
                        }
                    )
                    + "\n"
                )
    return data, scores


def test_slice_breakdown_separates_recorded_mapping_from_corrected_union(tmp_path: Path) -> None:
    data, scores = _write_fixtures(tmp_path)
    output = tmp_path / "slice_breakdown.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data",
            str(data),
            "--scores",
            str(scores),
            "--source-temperature",
            "1.0",
            "--target-temperature",
            "1.0",
            "--threshold",
            "0.5",
            "--split-name",
            "fixture",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    groups = report["groups"]

    # The single-label mapping puts zero positives in the slice whose name
    # promises real + semi-real -- this is the frozen blocker.
    recorded = groups["recorded:real_plus_semi_real"]
    assert recorded["positives"] == 0
    assert recorded["sessions"] == 1

    # The union the name implies has both classes and a defined recall.
    corrected = groups["corrected:real_plus_semi_real"]
    assert corrected["sessions"] == 3
    assert corrected["positives"] == 1
    assert corrected["negatives"] == 2
    assert corrected["recall"] == 1.0
    assert corrected["fpr"] == 0.0

    assert groups["recorded:real_only"]["positives"] == 1
    assert groups["tier:semi_real"]["positives"] == 0


def test_slice_breakdown_rejects_scores_for_unknown_sessions(tmp_path: Path) -> None:
    data, scores = _write_fixtures(tmp_path)
    with scores.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"session_id": "session-does-not-exist", "command_index": 0, "malicious_probability": 0.5}
            )
            + "\n"
        )
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data",
            str(data),
            "--scores",
            str(scores),
            "--source-temperature",
            "1.0",
            "--target-temperature",
            "1.0",
            "--threshold",
            "0.5",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "is not in" in completed.stderr

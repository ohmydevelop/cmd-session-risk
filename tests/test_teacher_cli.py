from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_teacher_cli_without_cost_cap_is_dry_run(tmp_path: Path) -> None:
    accepted = tmp_path / "accepted.jsonl"
    isolated = tmp_path / "isolated.jsonl"
    result = _run(
        "scripts/run_teacher.py",
        "label",
        "--input",
        "tests/fixtures/teacher/sessions.jsonl",
        "--accepted-out",
        str(accepted),
        "--isolated-out",
        str(isolated),
    )
    assert result.returncode == 0
    assert '"dry_run": true' in result.stderr
    assert not accepted.exists()
    assert not isolated.exists()


def test_proxy_audit_cli_without_cost_cap_is_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = _run(
        "scripts/run_proxy_audit.py",
        "--input",
        "tests/fixtures/teacher/audit_cases.jsonl",
        "--output",
        str(output),
    )
    assert result.returncode == 0
    assert '"dry_run": true' in result.stderr
    assert not output.exists()

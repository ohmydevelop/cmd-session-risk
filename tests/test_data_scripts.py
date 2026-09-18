from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cmdrisk.data.manifest import load_data_sources, load_sha256_manifest

ROOT = Path(__file__).resolve().parents[1]
DATA_FIXTURES = ROOT / "tests" / "fixtures" / "data"
PREFIX_FIXTURE = ROOT / "tests" / "fixtures" / "prefixes" / "sessions.jsonl"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_source_manifest_records_required_provenance_fields() -> None:
    sources = load_data_sources(ROOT / "configs" / "data_sources.yaml")
    ids = {source["id"] for source in sources}

    assert {
        "microsoft_iot_honeypot",
        "shell_attack_evolution",
        "kypo_zenodo",
        "atomic_red_team",
    }.issubset(ids)
    assert all({"url", "license", "sha256", "password", "redistribution"} <= set(source) for source in sources)
    microsoft = next(source for source in sources if source["id"] == "microsoft_iot_honeypot")
    assert microsoft["password"] == "infected"
    assert "unconfirmed" in microsoft["redistribution"]


def test_all_scripts_support_dry_run() -> None:
    download = _run("scripts/download_sources.py", "--dry-run", "--source", "microsoft_iot_honeypot")
    prepare = _run(
        "scripts/prepare_data.py",
        "--dry-run",
        "--source",
        f"microsoft={DATA_FIXTURES / 'microsoft_iot.json'}",
    )
    scripts = _run(
        "scripts/build_script_sessions.py",
        "--dry-run",
        "--repo",
        f"fixture={DATA_FIXTURES / 'scripts' / 'repo'}",
    )
    prefixes = _run(
        "scripts/build_prefixes.py", "--dry-run", "--input", str(PREFIX_FIXTURE)
    )

    assert "microsoft_iot_honeypot" in download.stdout
    assert "microsoft_iot.json" in prepare.stdout
    assert "fixture" in scripts.stdout
    assert "sessions.jsonl" in prefixes.stdout


def test_prepare_script_writes_outputs_manifest_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    arguments = [
        "scripts/prepare_data.py",
        "--source",
        f"microsoft={DATA_FIXTURES / 'microsoft_iot.json'}",
        "--source",
        f"shell={DATA_FIXTURES / 'shell_attack_evolution.json'}",
        "--source",
        f"zenodo={DATA_FIXTURES / 'zenodo.jsonl'}",
        "--source",
        f"atomic={DATA_FIXTURES / 'atomic.yaml'}",
        "--source",
        f"scripts={DATA_FIXTURES / 'scripts' / 'repo'}",
        "--total",
        "10",
        "--minimum-family-cap",
        "2",
        "--output-dir",
        str(output),
    ]
    _run(*arguments)
    resumed = _run(*arguments, "--resume")

    assert (output / "sessions.jsonl").exists()
    assert (output / "train.jsonl").exists()
    assert (output / "provenance.json").exists()
    assert len(load_sha256_manifest(output / "SHA256SUMS")) == 6
    assert "resume" in resumed.stdout
    assert json.loads((output / "provenance.json").read_text())["total_sessions"] == 10


def test_script_and_prefix_builders_write_resumable_artifacts(tmp_path: Path) -> None:
    script_output = tmp_path / "script_sessions.jsonl"
    _run(
        "scripts/build_script_sessions.py",
        "--repo",
        f"fixture={DATA_FIXTURES / 'scripts' / 'repo'}",
        "--output",
        str(script_output),
    )
    prefix_output = tmp_path / "prefixes.jsonl"
    _run(
        "scripts/build_prefixes.py",
        "--input",
        str(PREFIX_FIXTURE),
        "--output",
        str(prefix_output),
    )
    resumed = _run(
        "scripts/build_prefixes.py",
        "--input",
        str(PREFIX_FIXTURE),
        "--output",
        str(prefix_output),
        "--resume",
    )

    assert script_output.exists()
    assert sum(1 for _ in script_output.open()) == 5
    assert prefix_output.exists()
    assert sum(1 for _ in prefix_output.open()) == 20
    assert "resume" in resumed.stdout

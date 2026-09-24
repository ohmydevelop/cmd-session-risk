from __future__ import annotations

import json
import os
import shutil
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


def _run_checked_failure(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
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
    assert all(
        {"url", "license", "sha256", "password", "redistribution"} <= set(source)
        for source in sources
    )
    microsoft = next(source for source in sources if source["id"] == "microsoft_iot_honeypot")
    assert microsoft["password"] == "infected"
    assert "unconfirmed" in microsoft["redistribution"]


def test_all_scripts_support_dry_run() -> None:
    download = _run(
        "scripts/download_sources.py", "--dry-run", "--source", "microsoft_iot_honeypot"
    )
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
    prefixes = _run("scripts/build_prefixes.py", "--dry-run", "--input", str(PREFIX_FIXTURE))

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
        "--allow-quota-backfill",
        "--no-enforce-split-balance",
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


def test_prepare_resume_invalidates_for_content_and_option_changes(tmp_path: Path) -> None:
    source = tmp_path / "microsoft.json"
    shutil.copyfile(DATA_FIXTURES / "microsoft_iot.json", source)
    output = tmp_path / "prepared"
    arguments = [
        "scripts/prepare_data.py",
        "--source",
        f"microsoft={source}",
        "--total",
        "2",
        "--allow-quota-backfill",
        "--no-enforce-split-balance",
        "--output-dir",
        str(output),
    ]
    _run(*arguments)
    first_state = json.loads((output / "prepare_state.json").read_text())
    original_stat = source.stat()

    payload = json.loads(source.read_text())
    payload[0]["Commands"][0] = "echo changed-content"
    source.write_text(json.dumps(payload))
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    changed = _run(*arguments, "--resume")

    second_state = json.loads((output / "prepare_state.json").read_text())
    assert "resume:" not in changed.stdout
    assert second_state["input_fingerprint"] != first_state["input_fingerprint"]

    resumed = _run(*arguments, "--resume")
    assert "resume:" in resumed.stdout

    changed_options = _run(*arguments, "--resume", "--total", "1")
    option_state = json.loads((output / "prepare_state.json").read_text())
    assert "resume:" not in changed_options.stdout
    assert option_state["preparation_options"]["total"] == 1
    assert option_state["options_fingerprint"] != second_state["options_fingerprint"]


def test_prepare_source_limits_override_global_and_invalidate_resume(
    tmp_path: Path,
) -> None:
    output = tmp_path / "prepared"
    base_arguments = [
        "scripts/prepare_data.py",
        "--source",
        f"microsoft={DATA_FIXTURES / 'microsoft_iot.json'}",
        "--source",
        f"shell={DATA_FIXTURES / 'shell_attack_evolution.json'}",
        "--max-records-per-source",
        "1",
        "--total",
        "3",
        "--allow-quota-backfill",
        "--no-enforce-split-balance",
        "--output-dir",
        str(output),
    ]
    arguments = [*base_arguments, "--source-limit", "microsoft=2"]
    _run(*arguments)

    provenance = json.loads((output / "provenance.json").read_text())
    state = json.loads((output / "prepare_state.json").read_text())
    assert provenance["input_sessions_by_source"] == {"microsoft": 2, "shell": 1}
    assert state["preparation_options"]["source_limits"] == {"microsoft": 2}

    resumed = _run(*arguments, "--resume")
    assert "resume:" in resumed.stdout

    changed = _run(
        *base_arguments,
        "--source-limit",
        "microsoft=1",
        "--resume",
    )
    changed_state = json.loads((output / "prepare_state.json").read_text())
    assert "resume:" not in changed.stdout
    assert changed_state["preparation_options"]["source_limits"] == {"microsoft": 1}
    assert changed_state["options_fingerprint"] != state["options_fingerprint"]


def test_prepare_rejects_unknown_source_limit(tmp_path: Path) -> None:
    result = _run_checked_failure(
        "scripts/prepare_data.py",
        "--source",
        f"microsoft={DATA_FIXTURES / 'microsoft_iot.json'}",
        "--source-limit",
        "missing=1",
        "--output-dir",
        str(tmp_path / "prepared"),
    )

    assert result.returncode != 0
    assert "references source(s) not supplied" in result.stderr


def test_prepare_quota_parsing_validation_and_resume_fingerprint(tmp_path: Path) -> None:
    output = tmp_path / "prepared"
    base_arguments = [
        "scripts/prepare_data.py",
        "--source",
        f"microsoft={DATA_FIXTURES / 'microsoft_iot.json'}",
        "--source",
        f"zenodo={DATA_FIXTURES / 'zenodo.jsonl'}",
    ]
    arguments = [
        *base_arguments,
        "--quota",
        "real_malicious=0.5",
        "--quota",
        "benign=0.5",
        "--total",
        "2",
        "--allow-quota-backfill",
        "--no-enforce-split-balance",
        "--output-dir",
        str(output),
    ]
    _run(*arguments)
    state = json.loads((output / "prepare_state.json").read_text())
    assert state["preparation_options"]["quota"] == {
        "real_malicious": 0.5,
        "benign": 0.5,
    }

    resumed = _run(*arguments, "--resume")
    assert "resume:" in resumed.stdout

    changed = _run(
        *base_arguments,
        "--quota",
        "real_malicious=0.6",
        "--quota",
        "benign=0.4",
        "--total",
        "2",
        "--allow-quota-backfill",
        "--no-enforce-split-balance",
        "--output-dir",
        str(output),
        "--resume",
    )
    changed_state = json.loads((output / "prepare_state.json").read_text())
    assert "resume:" not in changed.stdout
    assert changed_state["preparation_options"]["quota"] == {
        "real_malicious": 0.6,
        "benign": 0.4,
    }
    assert changed_state["options_fingerprint"] != state["options_fingerprint"]

    invalid = _run_checked_failure(
        *base_arguments,
        "--quota",
        "real_malicious=0.6",
        "--total",
        "2",
        "--allow-quota-backfill",
        "--no-enforce-split-balance",
        "--output-dir",
        str(tmp_path / "invalid"),
    )
    assert invalid.returncode != 0
    assert "must sum to 1.0" in invalid.stderr


def test_script_resume_invalidates_content_change_with_preserved_stat(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(DATA_FIXTURES / "scripts" / "repo", repo)
    output = tmp_path / "script_sessions.jsonl"
    arguments = [
        "scripts/build_script_sessions.py",
        "--repo",
        f"fixture={repo}",
        "--output",
        str(output),
    ]
    _run(*arguments)
    source = next(path for path in repo.rglob("*") if path.is_file())
    original = source.stat()
    payload = bytearray(source.read_bytes())
    if not payload:
        payload.extend(b"x")
    else:
        payload[0] ^= 1
    source.write_bytes(bytes(payload))
    os.utime(source, ns=(original.st_atime_ns, original.st_mtime_ns))

    resumed = _run(*arguments, "--resume")

    assert "resume:" not in resumed.stdout
    state = json.loads((output.with_suffix(output.suffix + ".state.json")).read_text())
    assert state["repositories"]


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


def _write_v2_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Write a diverse shell + benign pair large enough for a strict v2 quota."""

    shell = tmp_path / "shell_v2.jsonl"
    rows = []
    for index in range(16):
        period = "2024" if index >= 12 else "2021_2022"
        rows.append(
            {
                "session_id": f"sae-v2-{index}",
                "period": period,
                "src_ip": f"203.0.113.{index + 1}",
                "commands": [
                    f"fetch_stage{index} --id {index}",
                    f"persist_agent{index}",
                    f"connect_back{index} --port {9000 + index}",
                ],
                "command_count": 3,
                "session_pattern": f"discovery|download|execution|{index}",
                "attack_techniques": ["T1105", "T1059"],
            }
        )
    shell.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    zenodo = tmp_path / "benign_v2.jsonl"
    actions = []
    for index in range(24):
        for step in range(2):
            actions.append(
                {
                    "cmd_type": "bash-command",
                    "sandbox_id": f"box-{index}",
                    "timestamp_str": f"2019-08-14T15:{index % 60:02d}:{step:02d}Z",
                    "hostname": "attacker",
                    "cmd": f"benign_editor{index}_step{step} --shard {index}",
                    "ip": "10.0.0.1",
                    "username": "root",
                    "wd": "/root",
                }
            )
    zenodo.write_text("\n".join(json.dumps(action) for action in actions) + "\n", encoding="utf-8")
    return shell, zenodo


def test_prepare_v2_writes_split_only_layout_with_ood_and_stress(tmp_path: Path) -> None:
    shell, zenodo = _write_v2_sources(tmp_path)
    output = tmp_path / "v2"
    _run(
        "scripts/prepare_data.py",
        "--pipeline-version",
        "v2",
        "--source",
        f"shell={shell}",
        "--source",
        f"zenodo={zenodo}",
        "--total",
        "20",
        "--minimum-family-cap",
        "2",
        "--output-dir",
        str(output),
    )

    # Split-only layout: pointing the trainer at the directory must not
    # double-load a combined sessions file.
    assert not (output / "sessions.jsonl").exists()
    assert not (output / "provenance.json").exists()
    loaded = sorted(
        child.name for child in output.iterdir() if child.suffix in {".json", ".jsonl", ".parquet"}
    )
    assert loaded == ["dev.jsonl", "test.jsonl", "train.jsonl"]

    ood = [
        json.loads(line)
        for line in (output / "slices" / "ood_2024.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stress = [
        json.loads(line)
        for line in (output / "slices" / "synthetic_stress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert ood and all(row["split"] == "test" for row in ood)
    assert stress and all(row["meta"]["augmentation_method"] == "deterministic" for row in stress)

    train = [
        json.loads(line)
        for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dev = [
        json.loads(line)
        for line in (output / "dev.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(str(row["meta"].get("period")) != "2024" for row in train)
    assert all(str(row["meta"].get("period")) != "2024" for row in dev)

    report = json.loads((output / "reports" / "provenance.json").read_text(encoding="utf-8"))
    counts = {
        "real_malicious": 8,
        "real_benign": 8,
        "llm_variant": 3,
        "hard_negative": 1,
    }
    test_rows = [
        json.loads(line)
        for line in (output / "test.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    non_holdout_rows = (
        train + dev + [row for row in test_rows if str(row["meta"].get("period")) != "2024"]
    )
    realized = {
        bucket: sum(1 for row in non_holdout_rows if row["source"]["kind"] == kind)
        for bucket, kind in {
            "real_malicious": "real_malicious",
            "llm_variant": "llm_variant",
            "hard_negative": "hard_negative",
        }.items()
    }
    assert realized == {k: counts[k] for k in realized}
    assert report["pipeline_version"] == "v2"
    assert report["v2_stats"]["holdout_count"] == len(ood)


def test_prepare_v2_rejects_quota_backfill(tmp_path: Path) -> None:
    shell, zenodo = _write_v2_sources(tmp_path)
    result = _run_checked_failure(
        "scripts/prepare_data.py",
        "--pipeline-version",
        "v2",
        "--source",
        f"shell={shell}",
        "--source",
        f"zenodo={zenodo}",
        "--total",
        "20",
        "--allow-quota-backfill",
        "--output-dir",
        str(tmp_path / "v2"),
    )
    assert result.returncode != 0
    assert "backfill" in (result.stderr + result.stdout)

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remote_preflight_shape(tmp_path: Path) -> None:
    module = _load_script("remote_preflight.py")
    report = module.collect_preflight(tmp_path)
    assert set(report) == {"hostname", "platform", "python", "gpu", "cpu", "ram", "disk"}
    assert isinstance(report["gpu"]["count"], int)
    assert isinstance(report["gpu"]["devices"], list)
    assert isinstance(report["cpu"]["logical_count"], int)
    assert report["disk"]["path"] == str(tmp_path.resolve())
    assert report["python"]["version"]


def test_sha256_manifest_round_trip_and_tamper(tmp_path: Path) -> None:
    module = _load_script("sync_remote.py")
    root = tmp_path / "processed"
    root.mkdir()
    (root / "part.parquet").write_bytes(b"payload")
    manifest = module.create_sha256_manifest(root)
    verified = module.verify_sha256_manifest(root, manifest)
    assert verified["checked"] == 1
    assert verified["ok"] is True

    (root / "part.parquet").write_bytes(b"tampered")
    try:
        module.verify_sha256_manifest(root, manifest)
    except module.SyncError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("tampered file was accepted")


def test_rsync_command_uses_ssh_and_dry_run(tmp_path: Path) -> None:
    module = _load_script("sync_remote.py")
    captured: list[list[str]] = []

    def fake_runner(command, **kwargs):
        captured.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    module.run_rsync(
        str(tmp_path / "source/"),
        "user@example:~/cmd-session-risk/data/processed/",
        ssh_target="user@example",
        ssh_port=2222,
        ssh_key="~/.ssh/lightning_rsa",
        dry_run=True,
        runner=fake_runner,
    )
    command = captured[0]
    assert command[0:2] == ["rsync", "-az"]
    assert "--dry-run" in command
    assert "-e" in command
    ssh_command = command[command.index("-e") + 1]
    assert "BatchMode=yes" in ssh_command
    assert "-p 2222" in ssh_command
    assert "lightning_rsa" in ssh_command


def test_bootstrap_skips_with_existing_key_and_host(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir(parents=True)
    key = ssh_dir / "lightning_rsa"
    key.write_text("fake-key", encoding="utf-8")
    key.chmod(stat.S_IRUSR | stat.S_IWUSR)
    (ssh_dir / "config").write_text(
        "Host lightning.ai\n  HostName ssh.lightning.ai\n", encoding="utf-8"
    )

    env = dict(os.environ)
    env["HOME"] = str(home)
    env.pop("LIGHTNING_SETUP_URL", None)
    completed = subprocess.run(
        [str(ROOT / "scripts" / "bootstrap_lightning.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0
    assert "skipping bootstrap" in completed.stdout
    assert "LIGHTNING_SETUP_URL" not in completed.stderr


def test_bootstrap_does_not_print_setup_url(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    tools = tmp_path / "bin"
    tools.mkdir()
    secret_url = "https://example.invalid/setup?token=do-not-print"
    curl = tools / "curl"
    curl.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' 'mkdir -p \"$HOME/.ssh\"'\n"
        "printf '%s\\n' 'printf key >\"$HOME/.ssh/lightning_rsa\"'\n"
        "printf '%s\\n' "
        '\'printf "Host lightning.ai\\\\n  HostName ssh.lightning.ai\\\\n" '
        '>"$HOME/.ssh/config"\'\n',
        encoding="utf-8",
    )
    curl.chmod(0o755)
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["PATH"] = f"{tools}:{env['PATH']}"
    env["LIGHTNING_SETUP_URL"] = secret_url
    completed = subprocess.run(
        [str(ROOT / "scripts" / "bootstrap_lightning.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert secret_url not in completed.stdout
    assert secret_url not in completed.stderr
    assert (home / ".ssh" / "lightning_rsa").is_file()
    assert "Host lightning.ai" in (home / ".ssh" / "config").read_text(encoding="utf-8")


def _report(**overrides):
    report = {
        "gpu": {"count": 1, "cuda_version": "12.4", "devices": [{"memory_total_mib": 40960}]},
        "cpu": {"logical_cpus": 16, "affinity_count": 8},
        "disk": {"free_bytes": 100 * 1024**3},
    }
    report.update(overrides)
    return report


def test_validate_preflight_accepts_capable_host() -> None:
    module = _load_script("remote_preflight.py")
    gate = module.validate_preflight(
        _report(),
        require_cuda=True,
        min_vram_gb=16,
        min_free_disk_gb=50,
        min_cpu_cores=8,
        min_cuda_version="12.0",
    )
    assert gate["passed"] is True
    assert gate["failures"] == []


def test_validate_preflight_fails_closed_on_weak_host() -> None:
    module = _load_script("remote_preflight.py")
    gate = module.validate_preflight(
        {
            "gpu": {"count": 0, "cuda_version": "11.0", "devices": []},
            "cpu": {"logical_cpus": 4},
            "disk": {"free_bytes": 1},
        },
        require_cuda=True,
        min_vram_gb=16,
        min_free_disk_gb=50,
        min_cpu_cores=8,
        min_cuda_version="12.0",
    )
    assert gate["passed"] is False
    for failure in (
        "no_cuda_gpu",
        "insufficient_vram<16GiB",
        "insufficient_free_disk<50GiB",
        "insufficient_cpu_cores<8",
        "cuda_version<12.0",
    ):
        assert failure in gate["failures"]


def test_preflight_cli_exit_codes() -> None:
    script = ROOT / "scripts" / "remote_preflight.py"
    passing = subprocess.run(
        [str(script), "--min-cpu-cores", "1", "--compact"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert passing.returncode == 0, passing.stderr
    assert '"gates"' in passing.stdout

    failing = subprocess.run(
        [str(script), "--min-cpu-cores", "100000", "--compact"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert failing.returncode == 1
    assert "insufficient_cpu_cores" in failing.stderr


class _Completed:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_sync_preflight_remote_with_fake_runner() -> None:
    module = _load_script("sync_remote.py")
    seen: list[str] = []

    def fake_runner(command, **kwargs):
        remote_command = command[-1]
        seen.append(remote_command)
        if remote_command == "true":
            return _Completed(0)
        if "df -Pk" in remote_command:
            return _Completed(0, stdout="2097152\n")
        if "nvidia-smi" in remote_command:
            return _Completed(0, stdout="A100-SXM4-40GB, 40960\n")
        return _Completed(0)

    report = module.preflight_remote(
        "user@example",
        "/tmp/data",
        min_free_disk_gb=1.0,
        require_cuda=True,
        min_vram_gb=16.0,
        runner=fake_runner,
    )
    assert report["passed"] is True
    assert report["free_bytes"] == 2097152 * 1024
    assert report["gpu"]["count"] == 1
    assert any(command == "true" for command in seen)


def test_sync_preflight_remote_fails_without_cuda() -> None:
    module = _load_script("sync_remote.py")

    def fake_runner(command, **kwargs):
        remote_command = command[-1]
        if "df -Pk" in remote_command:
            return _Completed(0, stdout="1048576\n")
        return _Completed(0)

    try:
        module.preflight_remote("user@example", "/tmp/data", require_cuda=True, runner=fake_runner)
    except module.SyncError as exc:
        assert "no_cuda_gpu" in str(exc)
    else:
        raise AssertionError("preflight accepted a host without CUDA")


def test_sync_check_connectivity_failure_is_fatal() -> None:
    module = _load_script("sync_remote.py")

    def fake_runner(command, **kwargs):
        return _Completed(255, stderr="connection refused")

    try:
        module.check_connectivity("user@example", runner=fake_runner)
    except module.SyncError as exc:
        assert "cannot reach remote host" in str(exc)
    else:
        raise AssertionError("unreachable host was accepted")


def test_train_remote_never_leaks_setup_token_and_excludes_env_report(tmp_path: Path) -> None:
    tools = tmp_path / "bin"
    tools.mkdir()
    log = tmp_path / "calls.log"
    secret_url = "https://example.invalid/setup?token=super-secret-token"

    def _fake(name: str, body: str) -> None:
        script = tools / name
        script.write_text(
            f'#!/usr/bin/env bash\nprintf "%s\\n" "{name} $*" >> "$FAKE_LOG"\n' + body,
            encoding="utf-8",
        )
        script.chmod(0o755)

    _fake("ssh", "exit 0\n")
    _fake("rsync", "exit 0\n")
    # The fake curl must not log its arguments: the URL is a one-time secret
    # that only the curl process is allowed to see.
    (tools / "curl").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (tools / "curl").chmod(0o755)

    data = tmp_path / "data" / "processed"
    data.mkdir(parents=True)
    (data / "part.parquet").write_bytes(b"payload")
    output = tmp_path / "out"

    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{tools}:{env['PATH']}",
            "FAKE_LOG": str(log),
            "LIGHTNING_SETUP_URL": secret_url,
            "LIGHTNING_SSH_HOST": "user@example",
        }
    )
    completed = subprocess.run(
        [
            str(ROOT / "scripts" / "train_remote.sh"),
            "--data-dir",
            str(tmp_path / "data"),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert secret_url not in completed.stdout
    assert secret_url not in completed.stderr
    calls = log.read_text(encoding="utf-8")
    assert secret_url not in calls
    assert "remote_environment.json" in calls

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
        "'printf \"Host lightning.ai\\\\n  HostName ssh.lightning.ai\\\\n\" "
        ">\"$HOME/.ssh/config\"'\n",
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

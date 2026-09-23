from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_checkpoints.py"


def _run(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        check=check,
        text=True,
        capture_output=True,
    )


def _make_checkpoint(directory: Path, scale: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "weight": torch.full((2, 3), float(scale)),
            "bias": torch.tensor([float(scale), -float(scale)]),
            "counter": torch.tensor([int(scale)]),
        },
        str(directory / "model.safetensors"),
        metadata={"format": "pt"},
    )
    (directory / "config.json").write_text('{"hidden_size": 4}\n', encoding="utf-8")


def test_merge_checkpoints_averages_float_tensors_and_keeps_integer_buffers(tmp_path: Path) -> None:
    first = tmp_path / "checkpoint-10"
    second = tmp_path / "checkpoint-20"
    _make_checkpoint(first, 1)
    _make_checkpoint(second, 3)
    output = tmp_path / "soup"

    _run("--checkpoint", str(first), "--checkpoint", str(second), "--output-dir", str(output))

    merged = load_file(str(output / "model.safetensors"))
    assert torch.allclose(merged["weight"], torch.full((2, 3), 2.0))
    assert torch.allclose(merged["bias"], torch.tensor([2.0, -2.0]))
    # Integer buffers are copied from the first checkpoint, never averaged.
    assert merged["counter"].tolist() == [1]
    # Sidecar files travel with the weights so the soup is a drop-in checkpoint.
    assert json.loads((output / "config.json").read_text(encoding="utf-8")) == {"hidden_size": 4}

    provenance = json.loads((output / "soup_provenance.json").read_text(encoding="utf-8"))
    assert [source["weight"] for source in provenance["sources"]] == [0.5, 0.5]
    assert provenance["output"]["tensors"] == 3
    assert len(provenance["output"]["sha256"]) == 64


def test_merge_checkpoints_honours_explicit_weights(tmp_path: Path) -> None:
    first = tmp_path / "checkpoint-10"
    second = tmp_path / "checkpoint-20"
    _make_checkpoint(first, 1)
    _make_checkpoint(second, 3)
    output = tmp_path / "weighted"

    _run(
        "--checkpoint",
        str(first),
        "--checkpoint",
        str(second),
        "--weights",
        "0.25,0.75",
        "--output-dir",
        str(output),
    )

    merged = load_file(str(output / "model.safetensors"))
    assert torch.allclose(merged["weight"], torch.full((2, 3), 2.5))


def test_merge_checkpoints_last_n_selects_numerically_latest_and_refuses_overwrite(tmp_path: Path) -> None:
    for index, scale in enumerate((1, 2, 3), start=1):
        _make_checkpoint(tmp_path / f"checkpoint-{index * 10}", scale)

    output = tmp_path / "soup"
    _run("--glob", str(tmp_path / "checkpoint-*"), "--last", "2", "--output-dir", str(output))

    # checkpoint-20 and checkpoint-30 -> (2 + 3) / 2
    merged = load_file(str(output / "model.safetensors"))
    assert torch.allclose(merged["weight"], torch.full((2, 3), 2.5))

    blocked = _run("--glob", str(tmp_path / "checkpoint-*"), "--last", "2", "--output-dir", str(output), check=False)
    assert blocked.returncode != 0
    assert "force" in blocked.stderr

    allowed = _run(
        "--glob",
        str(tmp_path / "checkpoint-*"),
        "--last",
        "2",
        "--output-dir",
        str(output),
        "--force",
    )
    assert allowed.returncode == 0

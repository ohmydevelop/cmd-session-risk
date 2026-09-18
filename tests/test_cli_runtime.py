from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

import cmdrisk.cli as cli
from cmdrisk.runtime import SessionScorer

runner = CliRunner()


@dataclass(frozen=True)
class FakeLabel:
    id: str
    name: str


class FakeTokenizer:
    def __call__(self, text: str, **kwargs: object) -> dict[str, np.ndarray]:
        del kwargs
        ids = [10, *(100 + ord(character) % 50 for character in text), 11]
        return {"input_ids": np.asarray([ids], dtype=np.int64)}


class FakeOnnxSession:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_ids")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="malicious_logits"),
            SimpleNamespace(name="tactics_logits"),
            SimpleNamespace(name="techniques_logits"),
        ]

    def run(self, output_names: object, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        del output_names, input_feed
        self.calls += 1
        malicious = np.asarray([[12.0 if self.calls > 1 else -20.0]])
        tactics = np.asarray([[20.0 if self.calls > 1 else -20.0]])
        techniques = np.asarray([[20.0 if self.calls > 1 else -20.0]])
        return [malicious, tactics, techniques]


def _scorer() -> SessionScorer:
    return SessionScorer(
        FakeOnnxSession(),
        FakeTokenizer(),
        tactics=[FakeLabel("TA0002", "Execution")],
        techniques=[FakeLabel("T1105", "Ingress Tool Transfer")],
        malicious_threshold=0.9,
        max_length=128,
        input_names=("input_ids",),
        output_names={
            "malicious": "malicious_logits",
            "tactics": "tactics_logits",
            "techniques": "techniques_logits",
        },
        model_version="cli-test",
    )


def test_score_compact_jsonl(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_load_scorer", lambda _: _scorer())
    source = tmp_path / "sessions.jsonl"
    source.write_text(
        '{"session_id":"s1","commands":[{"cmd":"whoami","dt":0},'
        '{"cmd":"curl x","dt":1}]}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        cli.app,
        [
            "score",
            "--artifact-dir",
            str(tmp_path),
            "--input",
            str(source),
            "--output",
            str(tmp_path / "scores.jsonl"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads((tmp_path / "scores.jsonl").read_text(encoding="utf-8"))
    assert payload["session_id"] == "s1"
    assert payload["alerted"] is True
    assert len(payload["entries"]) == 2
    assert payload["entries"][1]["alert"] is True


def test_score_flat_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_load_scorer", lambda _: _scorer())
    result = runner.invoke(
        cli.app,
        ["score", "--artifact-dir", str(tmp_path), "--input", "-", "--flat"],
        input='{"session_id":"flat","commands":["whoami","curl x"]}\n',
    )
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert [item["command_index"] for item in lines] == [0, 1]
    assert lines[1]["session_id"] == "flat"
    assert lines[1]["alert"] is True


def test_stream_empty_stdin_is_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_load_scorer", lambda _: _scorer())
    result = runner.invoke(
        cli.app,
        ["stream", "--artifact-dir", str(tmp_path), "--session-id", "empty"],
        input="",
    )
    assert result.exit_code == 0
    assert result.output == ""


def test_stream_scores_each_command(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_load_scorer", lambda _: _scorer())
    result = runner.invoke(
        cli.app,
        ["stream", "--artifact-dir", str(tmp_path), "--session-id", "live"],
        input='{"cmd":"whoami","dt":0}\n{"cmd":"curl x","dt":1}\n',
    )
    assert result.exit_code == 0, result.output
    lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["alert"] is False
    assert lines[1]["alert"] is True
    assert lines[1]["session_id"] == "live"


def test_invalid_input_uses_stable_error_code(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_load_scorer", lambda _: _scorer())
    result = runner.invoke(
        cli.app,
        ["score", "--artifact-dir", str(tmp_path), "--input", "-"],
        input="{not-json}\n",
    )
    assert result.exit_code == 2
    assert '"code":"invalid_input"' in result.output

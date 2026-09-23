from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

import cmdrisk.cli as cli
from cmdrisk.runtime import InferenceError, SessionScorer

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


class BatchOnnxSession:
    """Batch-aware fake whose logits depend only on unmasked tokens."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="malicious_logits"),
            SimpleNamespace(name="tactics_logits"),
            SimpleNamespace(name="techniques_logits"),
        ]

    def run(self, output_names: object, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        del output_names
        ids = input_feed["input_ids"].astype(np.float64)
        mask = input_feed["attention_mask"].astype(np.float64)
        rows = int(ids.shape[0])
        self.batch_sizes.append(rows)
        totals = (ids * mask).sum(axis=1)
        malicious = np.where(totals > 400.0, 8.0, -8.0).reshape(rows, 1)
        tactics = np.where(totals > 400.0, 8.0, -8.0).reshape(rows, 1)
        techniques = np.where(totals % 3 == 0, 8.0, -8.0).reshape(rows, 1)
        return [malicious, tactics, techniques]


def _batch_scorer() -> SessionScorer:
    return SessionScorer(
        BatchOnnxSession(),
        FakeTokenizer(),
        tactics=[FakeLabel("TA0002", "Execution")],
        techniques=[FakeLabel("T1105", "Ingress Tool Transfer")],
        malicious_threshold=0.5,
        max_length=128,
        input_names=("input_ids", "attention_mask"),
        output_names={
            "malicious": "malicious_logits",
            "tactics": "tactics_logits",
            "techniques": "techniques_logits",
        },
        model_version="cli-test",
    )


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
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: _scorer())
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
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: _scorer())
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
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: _scorer())
    result = runner.invoke(
        cli.app,
        ["stream", "--artifact-dir", str(tmp_path), "--session-id", "empty"],
        input="",
    )
    assert result.exit_code == 0
    assert result.output == ""


def test_stream_scores_each_command(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: _scorer())
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
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: _scorer())
    result = runner.invoke(
        cli.app,
        ["score", "--artifact-dir", str(tmp_path), "--input", "-"],
        input="{not-json}\n",
    )
    assert result.exit_code == 2
    assert '"code":"invalid_input"' in result.output


def test_stream_inference_failure_uses_exit_code_four(monkeypatch, tmp_path: Path) -> None:
    class FailingScorer:
        model_version = "failing"

        def start_session(self, _session_id: str) -> object:
            class FailingState:
                peak_probability = 0.0

                def update(self, _cmd: str, _dt: float | None) -> object:
                    raise InferenceError("onnx failed")

            return FailingState()

    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: FailingScorer())
    result = runner.invoke(
        cli.app,
        ["stream", "--artifact-dir", str(tmp_path), "--session-id", "live"],
        input='{"cmd":"whoami","dt":0}\n',
    )
    assert result.exit_code == 4
    assert '"code":"inference_error"' in result.output

def test_score_batch_size_matches_streaming(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: _batch_scorer())
    text = (
        '{"session_id":"s1","commands":["whoami","curl x","echo hi"]}\n'
        '{"session_id":"s2","commands":["ls"]}\n'
    )
    sequential = runner.invoke(
        cli.app, ["score", "--artifact-dir", str(tmp_path), "--input", "-"], input=text
    )
    batched = runner.invoke(
        cli.app,
        ["score", "--artifact-dir", str(tmp_path), "--input", "-", "--batch-size", "2"],
        input=text,
    )
    assert sequential.exit_code == 0, sequential.output
    assert batched.exit_code == 0, batched.output
    assert batched.output == sequential.output

    flat = runner.invoke(
        cli.app,
        [
            "score",
            "--artifact-dir",
            str(tmp_path),
            "--input",
            "-",
            "--flat",
            "--batch-size",
            "2",
        ],
        input=text,
    )
    assert flat.exit_code == 0, flat.output
    lines = [json.loads(line) for line in flat.output.splitlines() if line.strip()]
    assert [item["command_index"] for item in lines] == [0, 1, 2, 0]
    assert [item["session_id"] for item in lines] == ["s1", "s1", "s1", "s2"]

def test_score_batch_honours_token_budget(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, BatchOnnxSession] = {}

    def loader(*_args: object, **_kwargs: object) -> SessionScorer:
        scorer = _batch_scorer()
        captured["session"] = scorer._onnx_session
        return scorer

    monkeypatch.setattr(cli, "_load_scorer", loader)
    commands = ",".join(f'{{"cmd":"c{index}","dt":0}}' for index in range(1, 9))
    text = f'{{"session_id":"long","commands":[{commands}]}}\n'
    result = runner.invoke(
        cli.app,
        [
            "score",
            "--artifact-dir",
            str(tmp_path),
            "--input",
            "-",
            "--batch-size",
            "8",
            "--max-batch-tokens",
            "60",
        ],
        input=text,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.strip())
    assert len(payload["entries"]) == 8
    assert len(captured["session"].batch_sizes) > 1
    assert all(size <= 8 for size in captured["session"].batch_sizes)



def test_score_batch_splits_on_runtime_failure(monkeypatch, tmp_path: Path) -> None:
    """A batch the runtime cannot fit must be retried in smaller pieces."""

    attempts: list[int] = []

    class OomOnWideBatch(BatchOnnxSession):
        def run(
            self, output_names: object, input_feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            rows = int(input_feed["input_ids"].shape[0])
            attempts.append(rows)
            if rows > 2:
                raise RuntimeError("Failed to allocate memory for requested buffer")
            return super().run(output_names, input_feed)

    scorer = _batch_scorer()
    scorer._onnx_session = OomOnWideBatch()
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: scorer)

    commands = ",".join(f'{{"cmd":"c{index}","dt":0}}' for index in range(1, 7))
    text = f'{{"session_id":"wide","commands":[{commands}]}}\n'
    result = runner.invoke(
        cli.app,
        [
            "score",
            "--artifact-dir",
            str(tmp_path),
            "--input",
            "-",
            "--flat",
            "--batch-size",
            "6",
        ],
        input=text,
    )

    assert result.exit_code == 0, result.output
    assert max(attempts) == 6
    assert min(attempts) == 1
    lines = [json.loads(line) for line in result.output.splitlines() if line.strip()]
    assert [item["command_index"] for item in lines] == [0, 1, 2, 3, 4, 5]


def test_score_batch_failure_on_single_row_still_fails(monkeypatch, tmp_path: Path) -> None:
    class AlwaysFailing(BatchOnnxSession):
        def run(
            self, output_names: object, input_feed: dict[str, np.ndarray]
        ) -> list[np.ndarray]:
            del output_names, input_feed
            raise RuntimeError("Failed to allocate memory for requested buffer")

    scorer = _batch_scorer()
    scorer._onnx_session = AlwaysFailing()
    monkeypatch.setattr(cli, "_load_scorer", lambda *_args, **_kwargs: scorer)

    text = '{"session_id":"s1","commands":["whoami"]}\n'
    result = runner.invoke(
        cli.app,
        ["score", "--artifact-dir", str(tmp_path), "--input", "-", "--batch-size", "2"],
        input=text,
    )
    assert result.exit_code == 4
    assert '"code":"inference_error"' in result.output

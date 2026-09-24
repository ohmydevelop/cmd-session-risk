from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cmdrisk.artifact_manifest import (
    EXPORT_MANIFEST_SCHEMA,
    EXPORT_MANIFEST_VERSION,
    build_artifact_entries,
    write_manifest_atomic,
)
from cmdrisk.runtime import SessionScorer, dt_bucket


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, np.ndarray]:
        self.calls.append((text, kwargs))
        ids = [10]
        for character in text:
            ids.append(100 + ord(character) % 50)
        ids.append(11)
        return {
            "input_ids": np.asarray([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }


class FakeOnnxSession:
    def __init__(self) -> None:
        self.calls = 0

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
        assert input_feed["input_ids"].ndim == 2
        self.calls += 1
        if self.calls == 1:
            malicious = np.asarray([[-20.0]])
            tactics = np.asarray([[20.0, -20.0]])
        elif self.calls == 2:
            malicious = np.asarray([[12.0]])
            tactics = np.asarray([[-20.0, 20.0]])
        else:
            malicious = np.asarray([[-12.0]])
            tactics = np.asarray([[-20.0, -20.0]])
        techniques = np.asarray([[20.0 if self.calls == 2 else -20.0]])
        return [malicious, tactics, techniques]


def _artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"fake")
    (artifact / "label_vocab.json").write_text(
        json.dumps(
            {
                "model_version": "test-v1",
                "max_length": 128,
                "tactics": [
                    {"id": "TA0007", "name": "Discovery"},
                    {"id": "TA0002", "name": "Execution"},
                ],
                "techniques": [{"id": "T1105", "name": "Ingress Tool Transfer"}],
            }
        ),
        encoding="utf-8",
    )
    (artifact / "calibration.json").write_text(
        json.dumps(
            {
                "malicious_threshold": 0.9,
                "temperature": 1.0,
                "tactics": {"default": 0.5},
                "techniques": {"default": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (artifact / "runtime_config.json").write_text(
        json.dumps(
            {
                "onnx_outputs": {
                    "malicious": "malicious_logits",
                    "tactics": "tactics_logits",
                    "techniques": "techniques_logits",
                }
            }
        ),
        encoding="utf-8",
    )
    _write_export_manifest(artifact, ["model.onnx"])
    return artifact


def _write_export_manifest(artifact: Path, extra: list[str]) -> None:
    names = [
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name != "export_manifest.json"
    ]
    names = sorted(dict.fromkeys([*names, *extra]))
    artifacts = build_artifact_entries(artifact, names)
    write_manifest_atomic(
        artifact / "export_manifest.json",
        {
            "schema": EXPORT_MANIFEST_SCHEMA,
            "version": EXPORT_MANIFEST_VERSION,
            "fp32_model": "model.onnx",
            "int8_model": None,
            "artifacts": artifacts,
            "checksums": {name: entry["sha256"] for name, entry in artifacts.items()},
            "sizes": {name: entry["size"] for name, entry in artifacts.items()},
        },
    )


def test_from_onnx_and_online_prefix_latching(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer()
    scorer = SessionScorer.from_onnx(
        _artifact(tmp_path), onnx_session=FakeOnnxSession(), tokenizer=tokenizer
    )
    state = scorer.start_session("session-1")

    first = state.update("whoami", dt=0)
    assert first.command_index == 0
    assert first.alert is False
    assert first.attack_stage == "Discovery"
    assert first.risk_score == pytest.approx(first.malicious_probability * 0.6)
    assert first.techniques == []

    second = state.update("curl -o /tmp/a https://example.invalid/a", dt=2.1)
    assert second.command_index == 1
    assert second.alert is True
    assert second.attack_stage == "Execution"
    assert second.tactics[0].id == "TA0002"
    assert second.techniques[0].id == "T1105"
    assert second.risk_score == pytest.approx(second.malicious_probability * 0.8)

    third = state.update("echo done", dt=0.4)
    assert third.alert is True
    assert third.risk_score == 0.0

    result = state.result()
    assert result.session_id == "session-1"
    assert result.alerted is True
    assert len(result.entries) == 3
    assert result.peak_probability == pytest.approx(second.malicious_probability)
    assert result.model_version == "test-v1"

    marker = scorer._tokenizer
    first_ids = scorer._encode([("whoami", 0)])["input_ids"][0].tolist()
    assert first_ids[0] == marker.special_token_id("<SESSION>")
    assert first_ids[1] == marker.special_token_id("<CMD_START>")
    assert marker.special_token_id("<DT_0S>") in first_ids
    assert marker.special_token_id("</SESSION>") in first_ids

    second_ids = scorer._encode(
        [
            ("whoami", 0),
            ("curl -o /tmp/a https://example.invalid/a", 2.1),
        ]
    )["input_ids"][0].tolist()
    assert marker.special_token_id("<DT_LE_10S>") in second_ids

    third_ids = scorer._encode(
        [
            ("whoami", 0),
            ("curl -o /tmp/a https://example.invalid/a", 2.1),
            ("echo done", 0.4),
        ]
    )["input_ids"][0].tolist()
    assert marker.special_token_id("<DT_LE_1S>") in third_ids


class BatchOnnxSession:
    """Batch-aware fake session whose logits ignore padded positions."""

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
        malicious = np.where(totals > 500.0, 9.0, -9.0).reshape(rows, 1)
        tactics = np.stack(
            [
                np.where(totals > 500.0, 9.0, -9.0),
                np.where(totals > 900.0, 9.0, -9.0),
            ],
            axis=1,
        )
        techniques = np.where(totals % 2 == 0, 9.0, -9.0).reshape(rows, 1)
        return [malicious, tactics, techniques]


def test_score_batch_matches_sequential(tmp_path: Path) -> None:
    session = BatchOnnxSession()
    scorer = SessionScorer.from_onnx(
        _artifact(tmp_path), onnx_session=session, tokenizer=FakeTokenizer()
    )
    prefixes = [
        [("whoami", 0)],
        [("whoami", 0), ("curl -o /tmp/a https://example.invalid/a", 2.1)],
        [("whoami", 0), ("uname -a", 1.0), ("chmod +x /tmp/a", 0.4)],
    ]

    batched = scorer.score_batch(prefixes)
    assert session.batch_sizes == [3]
    sequential = [scorer.score_commands(prefix) for prefix in prefixes]
    assert batched == sequential
    assert [entry.command_index for entry in batched] == [0, 1, 2]

    # Batching must not change the alert-latching contract of a session result.
    state = scorer.start_session("batched")
    state.absorb(batched)
    result = state.result()
    assert result.session_id == "batched"
    assert result.peak_probability == max(entry.malicious_probability for entry in batched)
    assert result.alerted is any(entry.alert for entry in batched)

    # Padding is inert: scoring the same prefix alone yields identical logits.
    single = scorer.score_batch([prefixes[0]])[0]
    assert single.malicious_probability == batched[0].malicious_probability


def test_token_count_matches_encode_width(tmp_path: Path) -> None:
    scorer = SessionScorer.from_onnx(
        _artifact(tmp_path), onnx_session=BatchOnnxSession(), tokenizer=FakeTokenizer()
    )
    prefix = [("whoami", 0), ("curl -o /tmp/a https://example.invalid/a", 2.1)]
    assert scorer.token_count(prefix) == scorer._encode(prefix)["input_ids"].shape[1]
    assert scorer.token_count(prefix) > scorer.token_count(prefix[:1])


def test_score_batch_handles_mixed_lengths_and_empty_input(tmp_path: Path) -> None:
    session = BatchOnnxSession()
    scorer = SessionScorer.from_onnx(
        _artifact(tmp_path), onnx_session=session, tokenizer=FakeTokenizer()
    )
    assert scorer.score_batch([]) == []
    long_prefix = [("cmd-" + "x" * 40, 0.0)]
    short_prefix = [("ls", None)]
    entries = scorer.score_batch([short_prefix, long_prefix])
    assert session.batch_sizes == [2]
    assert entries[0].command_index == 0
    assert entries[1].command_index == 0
    assert entries[0].malicious_probability < entries[1].malicious_probability


def test_empty_session_result_is_stable(tmp_path: Path) -> None:
    scorer = SessionScorer.from_onnx(
        _artifact(tmp_path), onnx_session=FakeOnnxSession(), tokenizer=FakeTokenizer()
    )
    result = scorer.start_session("empty").result()
    assert result.entries == []
    assert result.alerted is False
    assert result.peak_probability == 0.0


def test_long_prefix_keeps_head_and_truncated_tail(tmp_path: Path) -> None:
    class LongTokenizer(FakeTokenizer):
        def __call__(self, text: str, **kwargs: object) -> dict[str, np.ndarray]:
            return {"input_ids": np.arange(100, dtype=np.int64).reshape(1, -1)}

    scorer = SessionScorer(
        FakeOnnxSession(),
        LongTokenizer(),
        tactics=[],
        techniques=[],
        max_length=10,
        session_head_tokens=4,
        session_tail_tokens=5,
        input_names=("input_ids", "attention_mask"),
        output_names={"malicious": "malicious_logits"},
    )
    feed = scorer._encode([("whoami", None)])
    assert feed["input_ids"].shape == (1, 10)

    marker = scorer._tokenizer
    ids = feed["input_ids"][0].tolist()
    assert ids[:4] == [
        marker.special_token_id("<SESSION>"),
        marker.special_token_id("<CMD_START>"),
        0,
        1,
    ]
    assert ids[4] == marker.special_token_id("<TRUNCATED>")
    assert ids[5:] == [
        98,
        99,
        marker.special_token_id("<CMD_END>"),
        marker.special_token_id("<DT_UNKNOWN>"),
        marker.special_token_id("</SESSION>"),
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "<DT_UNKNOWN>"),
        (0, "<DT_0S>"),
        (1, "<DT_LE_1S>"),
        (10, "<DT_LE_10S>"),
        (60, "<DT_LE_1M>"),
        (600, "<DT_LE_10M>"),
        (600.1, "<DT_GT_10M>"),
    ],
)
def test_dt_bucket(value: float | None, expected: str) -> None:
    assert dt_bucket(value) == expected


def test_dt_bucket_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        dt_bucket(-0.1)
    with pytest.raises(ValueError):
        dt_bucket(float("nan"))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_from_onnx_prefers_int8_model(tmp_path: Path, monkeypatch) -> None:
    artifact = _artifact(tmp_path)
    (artifact / "model.int8.onnx").write_bytes(b"int8")
    seen: dict[str, str] = {}
    session = FakeOnnxSession()

    class SessionOptions:
        intra_op_num_threads = 1
        inter_op_num_threads = 1

    def make_session(path: str, sess_options: object = None, providers: object = None):
        seen["path"] = path
        return session

    fake_ort = SimpleNamespace(SessionOptions=SessionOptions, InferenceSession=make_session)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    scorer = SessionScorer.from_onnx(artifact, tokenizer=FakeTokenizer())

    assert Path(seen["path"]).name == "model.int8.onnx"
    assert scorer.model_version == "test-v1"


def test_training_format_calibration_fields(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"fake")
    _write_json(
        artifact / "labels.json",
        {
            "model_version": "train-v2",
            "tactics": [{"id": "TA0007", "name": "Discovery"}],
            "techniques": [{"id": "T1105", "name": "Ingress Tool Transfer"}],
        },
    )
    _write_json(
        artifact / "calibration.json",
        {
            "malicious_temperature": 2.5,
            "malicious_threshold": 0.72,
            "tactic_temperature": 1.5,
            "technique_temperature": 1.25,
            "tactic_thresholds": {"TA0007": 0.31},
            "technique_thresholds": {"T1105": 0.62},
        },
    )
    scorer = SessionScorer.from_onnx(
        artifact,
        onnx_session=FakeOnnxSession(),
        tokenizer=FakeTokenizer(),
        verify_manifest=False,
    )
    assert scorer.malicious_temperature == pytest.approx(2.5)
    assert scorer.malicious_threshold == pytest.approx(0.72)
    assert scorer.tactic_temperature == pytest.approx(1.5)
    assert scorer.technique_temperature == pytest.approx(1.25)
    assert scorer._tactic_thresholds == {"TA0007": pytest.approx(0.31)}
    assert scorer._technique_thresholds == {"T1105": pytest.approx(0.62)}


def test_labels_json_string_vocab_resolves_taxonomy_names(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.onnx").write_bytes(b"fake")
    _write_json(
        artifact / "labels.json",
        {"version": 1, "tactics": ["TA0007", "TA0002"], "techniques": ["T1105"]},
    )
    scorer = SessionScorer.from_onnx(
        artifact,
        onnx_session=FakeOnnxSession(),
        tokenizer=FakeTokenizer(),
        verify_manifest=False,
    )
    assert [label[0] for label in scorer.tactic_labels] == ["TA0007", "TA0002"]
    assert scorer.tactic_labels[0][1] == "Discovery"
    assert scorer.technique_labels == (("T1105", "Ingress Tool Transfer"),)


def test_singular_tactic_output_name_alias(tmp_path: Path) -> None:
    class SingularOnnx(FakeOnnxSession):
        def get_outputs(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(name="malicious_logits"),
                SimpleNamespace(name="tactic_logits"),
                SimpleNamespace(name="technique_logits"),
            ]

        def run(self, output_names: object, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
            del output_names, input_feed
            self.calls += 1
            return [
                np.asarray([[20.0]]),
                np.asarray([[20.0]]),
                np.asarray([[20.0]]),
            ]

    scorer = SessionScorer(
        SingularOnnx(),
        FakeTokenizer(),
        tactics=[SimpleNamespace(id="TA0002", name="Execution")],
        techniques=[SimpleNamespace(id="T1105", name="Ingress Tool Transfer")],
        max_length=128,
        input_names=("input_ids",),
        output_names={
            "malicious": "malicious_logits",
            "tactics": "tactic_logits",
            "techniques": "technique_logits",
        },
    )
    entry = scorer.score_commands([("whoami", 0)])
    assert [item.id for item in entry.tactics] == ["TA0002"]
    assert entry.tactics[0].name == "Execution"
    assert entry.attack_stage == "Execution"


def test_from_onnx_requires_manifest_by_default(tmp_path: Path) -> None:
    from cmdrisk.runtime.scoring import ArtifactLoadError

    artifact = _artifact(tmp_path)
    (artifact / "export_manifest.json").unlink()
    with pytest.raises(ArtifactLoadError, match="manifest verification failed"):
        SessionScorer.from_onnx(artifact, onnx_session=FakeOnnxSession(), tokenizer=FakeTokenizer())
    # The escape hatch keeps legacy bundles loadable during migration.
    scorer = SessionScorer.from_onnx(
        artifact,
        onnx_session=FakeOnnxSession(),
        tokenizer=FakeTokenizer(),
        verify_manifest=False,
    )
    assert scorer.model_version == "test-v1"


def test_from_onnx_rejects_tampered_artifact(tmp_path: Path) -> None:
    from cmdrisk.runtime.scoring import ArtifactLoadError

    artifact = _artifact(tmp_path)
    (artifact / "calibration.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArtifactLoadError, match="manifest verification failed"):
        SessionScorer.from_onnx(artifact, onnx_session=FakeOnnxSession(), tokenizer=FakeTokenizer())


def test_from_onnx_preloads_cuda_libraries_when_requested(tmp_path: Path, monkeypatch) -> None:
    artifact = _artifact(tmp_path)
    calls: list[object] = []
    session = FakeOnnxSession()

    class SessionOptions:
        intra_op_num_threads = 1
        inter_op_num_threads = 1

    def make_session(path: str, sess_options: object = None, providers: object = None):
        del path, sess_options
        calls.append(("session", list(providers or ())))
        return session

    fake_ort = SimpleNamespace(
        SessionOptions=SessionOptions,
        InferenceSession=make_session,
        preload_dlls=lambda *args, **kwargs: calls.append("preload"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    SessionScorer.from_onnx(
        artifact,
        tokenizer=FakeTokenizer(),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert calls[0] == "preload"
    assert calls[1] == ("session", ["CUDAExecutionProvider", "CPUExecutionProvider"])

    # CPU-only loads must not touch the CUDA libraries at all.
    cpu_calls: list[object] = []

    def make_cpu_session(*args: object, **kwargs: object) -> object:
        cpu_calls.append(("session", kwargs.get("providers")))
        return session

    fake_ort_cpu = SimpleNamespace(
        SessionOptions=SessionOptions,
        InferenceSession=make_cpu_session,
        preload_dlls=lambda *args, **kwargs: cpu_calls.append("preload"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort_cpu)
    SessionScorer.from_onnx(artifact, tokenizer=FakeTokenizer())
    assert cpu_calls == [("session", ["CPUExecutionProvider"])]


def test_from_onnx_tolerates_missing_preload_helper(tmp_path: Path, monkeypatch) -> None:
    artifact = _artifact(tmp_path)
    session = FakeOnnxSession()

    class SessionOptions:
        intra_op_num_threads = 1
        inter_op_num_threads = 1

    fake_ort = SimpleNamespace(
        SessionOptions=SessionOptions,
        InferenceSession=lambda *args, **kwargs: session,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    scorer = SessionScorer.from_onnx(
        artifact,
        tokenizer=FakeTokenizer(),
        providers=["CUDAExecutionProvider"],
    )
    assert scorer.model_version == "test-v1"

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from cmdrisk.model import ModernBertForCommandRisk, ModernBertRiskConfig  # noqa: E402
from cmdrisk.runtime.scoring import SessionScorer  # noqa: E402
from cmdrisk.tokenization import SPECIAL_TOKENS, CommandTokenizer  # noqa: E402
from cmdrisk.training.export import (  # noqa: E402
    export_onnx,
    update_export_manifest_latency,
)
from cmdrisk.training.labels import LabelVocabulary  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "model"

WORDS = (
    "whoami",
    "uname",
    "-a",
    "id",
    "cat",
    "/etc/passwd",
    "curl",
    "wget",
    "-o",
    "/tmp/a",
    "https://example.invalid/a",
    "chmod",
    "+x",
    "base64",
    "-d",
    "|",
    "bash",
    "tar",
    "xf",
    "python",
    "-c",
)


def _tiny_config(vocabulary: LabelVocabulary) -> ModernBertRiskConfig:
    values = json.loads(
        (FIXTURES / "tiny_modernbert_config.json").read_text(encoding="utf-8")
    )
    return ModernBertRiskConfig(
        **values,
        num_tactics=len(vocabulary.tactics),
        num_techniques=len(vocabulary.techniques),
        risk_dropout=0.0,
        label_vocabulary=vocabulary.to_dict(),
    )


def _tiny_tokenizer(max_length: int = 256) -> CommandTokenizer:
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *SPECIAL_TOKENS, *WORDS]
    tokens: list[str] = []
    for token in vocabulary:
        if token not in tokens:
            tokens.append(token)
    backend = Tokenizer(
        models.WordLevel(
            vocab={token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    return CommandTokenizer(fast, max_length=max_length, command_max_tokens=32)


def _sessions() -> list[list[object]]:
    return [
        [{"cmd": "whoami", "dt": 0}],
        [{"cmd": "whoami", "dt": 0}, {"cmd": "uname -a", "dt": 2.1}, {"cmd": "id", "dt": 0.4}],
        [
            {"cmd": "curl -o /tmp/a https://example.invalid/a", "dt": 0.2},
            {"cmd": "chmod +x /tmp/a", "dt": 6.0},
            {"cmd": "bash", "dt": 0.1},
        ],
    ]


def test_export_onnx_round_trips_through_runtime(tmp_path: Path) -> None:
    vocabulary = LabelVocabulary.canonical(["T1082", "T1105"])
    tokenizer = _tiny_tokenizer()
    model = ModernBertForCommandRisk(_tiny_config(vocabulary), label_vocabulary=vocabulary)

    checkpoint = tmp_path / "checkpoint"
    tokenizer.save_pretrained(checkpoint)
    model.save_pretrained(checkpoint)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "malicious_temperature": 1.0,
                "malicious_threshold": 0.5,
                "tactic_temperature": 1.0,
                "technique_temperature": 1.0,
                "tactic_thresholds": {},
                "technique_thresholds": {},
            }
        ),
        encoding="utf-8",
    )

    artifact = tmp_path / "onnx"
    manifest = export_onnx(
        checkpoint=checkpoint,
        output_dir=artifact,
        calibration_path=calibration,
    )

    assert manifest["fp32_model"] == "model.onnx"
    assert manifest["int8_model"] == "model.int8.onnx"
    assert manifest["parity_max_abs_error"] <= manifest["parity_atol"]
    assert manifest["int8_parity_max_abs_error"] <= manifest["int8_parity_atol"]
    assert (artifact / "model.onnx").is_file()
    assert (artifact / "model.int8.onnx").is_file()
    tokenization = json.loads(
        (artifact / "command_tokenization.json").read_text(encoding="utf-8")
    )
    assert tokenization["max_length"] == 256
    assert tokenization["command_max_tokens"] == 32
    assert json.loads((artifact / "labels.json").read_text(encoding="utf-8"))["techniques"] == [
        "T1082",
        "T1105",
    ]

    scorer = SessionScorer.from_onnx(artifact)
    assert scorer.technique_labels[0][0] == "T1082"
    for commands in _sessions():
        entry = scorer.score_commands(commands)
        assert 0.0 <= entry.malicious_probability <= 1.0
        assert np.isfinite(entry.malicious_probability)
    assert scorer.malicious_threshold == pytest.approx(0.5)


def test_export_stamps_model_version_into_the_artifact(tmp_path: Path) -> None:
    """Exported bundles must identify themselves, not report ``unknown``."""

    checkpoint, calibration = _build_checkpoint(tmp_path)
    run_dir = tmp_path / "cmd-session-risk-l4-20260919-0020"
    artifact = run_dir / "onnx"
    manifest = export_onnx(
        checkpoint=checkpoint, output_dir=artifact, calibration_path=calibration
    )

    # Defaults to the parent directory name, which is the training run id.
    runtime_config = json.loads(
        (artifact / "runtime_config.json").read_text(encoding="utf-8")
    )
    assert runtime_config["model_version"] == run_dir.name
    assert manifest["model_version"] == run_dir.name
    assert manifest["runtime_config"] == "runtime_config.json"
    assert "runtime_config.json" in manifest["artifacts"]

    scorer = SessionScorer.from_onnx(artifact)
    assert scorer.model_version == run_dir.name
    entry = scorer.score_commands([{"cmd": "whoami", "dt": 0.0}])
    assert 0.0 <= entry.malicious_probability <= 1.0


def test_export_honours_an_explicit_model_version(tmp_path: Path) -> None:
    from cmdrisk.training.export import write_runtime_config

    checkpoint, calibration = _build_checkpoint(tmp_path)
    artifact = tmp_path / "onnx"
    manifest = export_onnx(
        checkpoint=checkpoint,
        output_dir=artifact,
        calibration_path=calibration,
        model_version="cmd-session-risk-v1.2.0",
    )

    assert manifest["model_version"] == "cmd-session-risk-v1.2.0"
    assert SessionScorer.from_onnx(artifact).model_version == "cmd-session-risk-v1.2.0"

    # Rewriting must preserve unrelated runtime keys already in the file.
    write_runtime_config(
        artifact, model_version=None, extra={"provider": "CPUExecutionProvider"}
    )
    payload = json.loads((artifact / "runtime_config.json").read_text(encoding="utf-8"))
    assert payload["model_version"] == artifact.parent.name
    assert payload["provider"] == "CPUExecutionProvider"


def test_exported_graph_handles_longer_sessions_than_the_trace_sample(tmp_path: Path) -> None:
    vocabulary = LabelVocabulary.canonical([])
    tokenizer = _tiny_tokenizer(max_length=512)
    model = ModernBertForCommandRisk(_tiny_config(vocabulary), label_vocabulary=vocabulary)
    checkpoint = tmp_path / "checkpoint"
    tokenizer.save_pretrained(checkpoint)
    model.save_pretrained(checkpoint)

    artifact = tmp_path / "onnx"
    export_onnx(checkpoint=checkpoint, output_dir=artifact, skip_int8=True)

    import onnxruntime as ort

    short = tokenizer.encode(["whoami"])
    long = tokenizer.encode(
        [
            {"cmd": command, "dt": index}
            for index, command in enumerate(
                [
                    "whoami",
                    "uname -a",
                    "id",
                    "cat /etc/passwd",
                    "curl -o /tmp/a https://example.invalid/a",
                    "chmod +x /tmp/a",
                    "base64 -d | bash",
                    "python -c",
                ]
            )
        ]
    )
    assert len(long.input_ids) > len(short.input_ids) * 3

    session = ort.InferenceSession(str(artifact / "model.onnx"), providers=["CPUExecutionProvider"])
    for encoded in (short, long):
        ids = torch.tensor([list(encoded.input_ids)], dtype=torch.long)
        mask = torch.ones_like(ids)
        with torch.inference_mode():
            reference = model(ids, mask)
        raw = session.run(None, {"input_ids": ids.numpy(), "attention_mask": mask.numpy()})
        expected = [
            reference.malicious_logits.numpy(),
            reference.tactic_logits.numpy(),
            reference.technique_logits.numpy(),
        ]
        for left, right in zip(expected, raw, strict=True):
            # ``initial`` keeps the comparison valid for a zero-width head.
            assert float(np.max(np.abs(left - right), initial=0.0)) < 1e-4


def _build_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    vocabulary = LabelVocabulary.canonical(["T1082", "T1105"])
    tokenizer = _tiny_tokenizer()
    model = ModernBertForCommandRisk(_tiny_config(vocabulary), label_vocabulary=vocabulary)
    checkpoint = tmp_path / "checkpoint"
    tokenizer.save_pretrained(checkpoint)
    model.save_pretrained(checkpoint)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "malicious_temperature": 1.0,
                "malicious_threshold": 0.5,
                "tactic_temperature": 1.0,
                "technique_temperature": 1.0,
                "tactic_thresholds": {},
                "technique_thresholds": {},
            }
        ),
        encoding="utf-8",
    )
    return checkpoint, calibration


def _passing_metrics() -> dict[str, object]:
    return {
        "promotion": {
            "passed": True,
            "missing_metrics": [],
            "violations": [],
            "checks": {
                "evaluation_slices_present": True,
                "evaluation_slices_passed": True,
                "proxy_audit": True,
            },
        }
    }


def _passing_latency() -> dict[str, object]:
    return {
        "profile": {"name": "8-core x86 AVX2", "passed": True},
        "profile_passed": True,
        "p95_ms": 40.0,
        "p99_ms": 55.0,
        "p95_threshold_ms": 50.0,
        "p99_threshold_ms": 70.0,
        "tolerance_ms": 10.0,
        "passed": True,
    }


def test_export_manifest_lists_every_artifact(tmp_path: Path) -> None:
    from cmdrisk.artifact_manifest import (
        EXPORT_MANIFEST_FILENAME,
        EXPORT_MANIFEST_SCHEMA,
        EXPORT_MANIFEST_VERSION,
        verify_export_manifest,
    )

    checkpoint, calibration = _build_checkpoint(tmp_path)
    artifact = tmp_path / "onnx"
    manifest = export_onnx(
        checkpoint=checkpoint, output_dir=artifact, calibration_path=calibration
    )

    assert manifest["schema"] == EXPORT_MANIFEST_SCHEMA
    assert manifest["version"] == EXPORT_MANIFEST_VERSION
    manifest_path = artifact / EXPORT_MANIFEST_FILENAME
    assert manifest_path.is_file()
    on_disk = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() and path.name != EXPORT_MANIFEST_FILENAME
    }
    assert on_disk == set(manifest["artifacts"])
    for name, entry in manifest["artifacts"].items():
        assert entry["size"] == (artifact / name).stat().st_size
        assert manifest["checksums"][name] == entry["sha256"]
    verify_export_manifest(artifact)

    # Export without latency evidence must not be promotable.
    assert manifest["promotion"]["passed"] is False
    assert "latency_report_missing" in manifest["promotion"]["failures"]


def test_export_require_promotion_fails_without_evidence(tmp_path: Path) -> None:
    checkpoint, calibration = _build_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="not eligible for promotion"):
        export_onnx(
            checkpoint=checkpoint,
            output_dir=tmp_path / "onnx",
            calibration_path=calibration,
            require_promotion=True,
        )


def test_update_export_manifest_latency_enables_promotion(tmp_path: Path) -> None:
    from cmdrisk.artifact_manifest import verify_export_manifest

    checkpoint, calibration = _build_checkpoint(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_passing_metrics()), encoding="utf-8")
    artifact = tmp_path / "onnx"
    export_onnx(
        checkpoint=checkpoint,
        output_dir=artifact,
        calibration_path=calibration,
        metrics_path=metrics_path,
    )

    latency_path = tmp_path / "latency.json"
    latency_path.write_text(json.dumps(_passing_latency()), encoding="utf-8")
    manifest = update_export_manifest_latency(
        artifact_dir=artifact,
        latency_report_path=latency_path,
        require_promotion=True,
    )
    assert manifest["latency"]["passed"] is True
    assert manifest["promotion"]["passed"] is True
    assert (artifact / "latency_report.json").is_file()
    assert "latency_report.json" in manifest["artifacts"]
    verify_export_manifest(artifact)


def test_update_export_manifest_latency_fails_closed(tmp_path: Path) -> None:
    checkpoint, calibration = _build_checkpoint(tmp_path)
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(_passing_metrics()), encoding="utf-8")
    artifact = tmp_path / "onnx"
    export_onnx(
        checkpoint=checkpoint,
        output_dir=artifact,
        calibration_path=calibration,
        metrics_path=metrics_path,
    )

    slow = _passing_latency()
    slow["p95_ms"] = 500.0
    slow["passed"] = False
    latency_path = tmp_path / "latency.json"
    latency_path.write_text(json.dumps(slow), encoding="utf-8")
    with pytest.raises(ValueError, match="not eligible for promotion"):
        update_export_manifest_latency(
            artifact_dir=artifact,
            latency_report_path=latency_path,
            require_promotion=True,
        )


def test_tampered_artifact_rejected_by_runtime_loader(tmp_path: Path) -> None:
    from cmdrisk.runtime.scoring import ArtifactLoadError

    checkpoint, calibration = _build_checkpoint(tmp_path)
    artifact = tmp_path / "onnx"
    export_onnx(
        checkpoint=checkpoint,
        output_dir=artifact,
        calibration_path=calibration,
        skip_int8=True,
    )
    with (artifact / "model.onnx").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ArtifactLoadError, match="manifest verification failed"):
        SessionScorer.from_onnx(artifact)

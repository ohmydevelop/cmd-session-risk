from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cmdrisk.model import ModernBertForCommandRisk, ModernBertRiskConfig  # noqa: E402
from cmdrisk.training.labels import LabelVocabulary  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "model"


def _tiny_model() -> ModernBertForCommandRisk:
    values = json.loads(
        (FIXTURES / "tiny_modernbert_config.json").read_text(encoding="utf-8")
    )
    vocabulary = LabelVocabulary.canonical(["T1082", "T1105"])
    config = ModernBertRiskConfig(
        **values,
        num_tactics=len(vocabulary.tactics),
        num_techniques=len(vocabulary.techniques),
        risk_dropout=0.0,
        label_vocabulary=vocabulary.to_dict(),
    )
    return ModernBertForCommandRisk(config)


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


def test_model_saves_and_loads_dynamic_label_vocabulary(tmp_path: Path) -> None:
    model = _tiny_model()
    model.save_pretrained(tmp_path)
    loaded = ModernBertForCommandRisk.from_pretrained(tmp_path)
    assert loaded.label_vocabulary == model.label_vocabulary
    assert loaded.config.num_techniques == 2

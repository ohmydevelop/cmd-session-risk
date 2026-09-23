"""Factory helpers for interchangeable command-risk backbones."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transformers import AutoConfig

from cmdrisk.training.labels import LabelVocabulary

SMALL_BERT_MODEL_TYPE = "small-bert-risk"
MODERN_BERT_MODEL_TYPE = "modernbert-risk"


def _config_model_type(path: str | Path) -> str:
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        return ""
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return str(payload.get("model_type") or "")


def load_risk_backbone(
    model_name_or_path: str | Path,
    *,
    label_vocabulary: LabelVocabulary,
    dropout: float = 0.1,
    pos_weight_cap: float = 10.0,
    malicious_pos_weight: float = 1.0,
    tactic_pos_weight: list[float] | None = None,
    technique_pos_weight: list[float] | None = None,
    attn_implementation: str | None = "sdpa",
    **kwargs: Any,
) -> Any:
    """Load a pretrained backbone and attach the shared risk heads."""

    backbone_config = AutoConfig.from_pretrained(model_name_or_path)
    output = backbone_config.to_dict().get("model_type")
    output = str(output or getattr(backbone_config, "model_type", ""))
    common: Mapping[str, Any] = {
        "label_vocabulary": label_vocabulary,
        "dropout": dropout,
        "pos_weight_cap": pos_weight_cap,
        "malicious_pos_weight": malicious_pos_weight,
        "tactic_pos_weight": tactic_pos_weight,
        "technique_pos_weight": technique_pos_weight,
    }
    if output == "bert":
        from cmdrisk.small_model import SmallBertForCommandRisk

        return SmallBertForCommandRisk.from_pretrained_backbone(
            model_name_or_path, **common, **kwargs
        )

    from cmdrisk.model import ModernBertForCommandRisk

    return ModernBertForCommandRisk.from_pretrained_backbone(
        model_name_or_path,
        **common,
        attn_implementation=attn_implementation,
        **kwargs,
    )


def load_risk_model(path: str | Path) -> Any:
    """Load a saved command-risk checkpoint based on its serialized model type."""

    model_type = _config_model_type(path)
    if model_type == SMALL_BERT_MODEL_TYPE:
        from cmdrisk.small_model import SmallBertForCommandRisk

        return SmallBertForCommandRisk.from_pretrained(path)
    if model_type == MODERN_BERT_MODEL_TYPE:
        from cmdrisk.model import ModernBertForCommandRisk

        return ModernBertForCommandRisk.from_pretrained(path)
    raise ValueError(
        f"unsupported command-risk model_type {model_type!r} in {Path(path) / 'config.json'}"
    )

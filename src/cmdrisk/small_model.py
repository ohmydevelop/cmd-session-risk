"""CPU-oriented BERT sequence classifier for command-session risk."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel, BertPreTrainedModel

from cmdrisk.model import CommandRiskOutput
from cmdrisk.taxonomy import TACTICS
from cmdrisk.training.labels import LabelVocabulary

DEFAULT_BACKBONE = "google/bert_uncased_L-4_H-256_A-4"


class SmallBertRiskConfig(BertConfig):
    """Configuration for the compact, CPU-deployable risk classifier."""

    model_type = "small-bert-risk"

    def __init__(
        self,
        *,
        num_tactics: int = len(TACTICS),
        num_techniques: int = 0,
        risk_dropout: float = 0.1,
        pos_weight_cap: float = 10.0,
        malicious_pos_weight: float = 1.0,
        tactic_pos_weight: list[float] | None = None,
        technique_pos_weight: list[float] | None = None,
        label_vocabulary: Mapping[str, Any] | None = None,
        backbone_model_name_or_path: str = DEFAULT_BACKBONE,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_tactics = int(num_tactics)
        self.num_techniques = int(num_techniques)
        self.risk_dropout = float(risk_dropout)
        self.pos_weight_cap = float(pos_weight_cap)
        self.malicious_pos_weight = float(malicious_pos_weight)
        self.tactic_pos_weight = list(tactic_pos_weight or [1.0] * self.num_tactics)
        self.technique_pos_weight = list(technique_pos_weight or [1.0] * self.num_techniques)
        self.label_vocabulary = dict(label_vocabulary or LabelVocabulary.canonical().to_dict())
        self.backbone_model_name_or_path = str(backbone_model_name_or_path)


class SmallBertForCommandRisk(BertPreTrainedModel):
    """CLS-pooled BERT with malicious, tactic, and technique heads."""

    config_class = SmallBertRiskConfig
    base_model_prefix = "bert"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: SmallBertRiskConfig,
        label_vocabulary: LabelVocabulary | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        vocabulary = (
            LabelVocabulary.from_dict(label_vocabulary)
            if isinstance(label_vocabulary, Mapping)
            else label_vocabulary
        )
        if vocabulary is None:
            vocabulary = LabelVocabulary.from_dict(config.label_vocabulary)

        expected_tactics = tuple(item.id for item in TACTICS)
        if vocabulary.tactics != expected_tactics:
            raise ValueError("label vocabulary must contain the canonical 14 tactics in order")
        if len(vocabulary.techniques) != int(config.num_techniques):
            config.num_techniques = len(vocabulary.techniques)
        if len(vocabulary.tactics) != int(config.num_tactics):
            config.num_tactics = len(vocabulary.tactics)
        config.label_vocabulary = vocabulary.to_dict()
        config.tactic_pos_weight = list(config.tactic_pos_weight)[: config.num_tactics]
        config.technique_pos_weight = list(config.technique_pos_weight)[: config.num_techniques]
        if len(config.tactic_pos_weight) != config.num_tactics:
            config.tactic_pos_weight = [1.0] * config.num_tactics
        if len(config.technique_pos_weight) != config.num_techniques:
            config.technique_pos_weight = [1.0] * config.num_techniques

        self.label_vocabulary = vocabulary
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.risk_dropout)
        self.malicious_head = nn.Linear(config.hidden_size, 1)
        self.tactic_head = nn.Linear(config.hidden_size, config.num_tactics)
        self.technique_head = nn.Linear(config.hidden_size, config.num_techniques)
        cap = float(config.pos_weight_cap)
        self.register_buffer(
            "malicious_pos_weight",
            torch.tensor([min(config.malicious_pos_weight, cap)], dtype=torch.float32),
        )
        self.register_buffer(
            "tactic_pos_weight",
            torch.tensor(
                [min(float(value), cap) for value in config.tactic_pos_weight],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "technique_pos_weight",
            torch.tensor(
                [min(float(value), cap) for value in config.technique_pos_weight],
                dtype=torch.float32,
            ),
        )
        self.post_init()

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_name_or_path: str | Path = DEFAULT_BACKBONE,
        *,
        label_vocabulary: LabelVocabulary,
        dropout: float = 0.1,
        pos_weight_cap: float = 10.0,
        malicious_pos_weight: float = 1.0,
        tactic_pos_weight: list[float] | None = None,
        technique_pos_weight: list[float] | None = None,
        attn_implementation: str | None = "sdpa",
        **kwargs: Any,
    ) -> SmallBertForCommandRisk:
        backbone_config = BertConfig.from_pretrained(model_name_or_path)
        config_values = backbone_config.to_dict()
        config_values.pop("model_type", None)
        config = cls.config_class(
            **config_values,
            num_tactics=len(label_vocabulary.tactics),
            num_techniques=len(label_vocabulary.techniques),
            risk_dropout=dropout,
            pos_weight_cap=pos_weight_cap,
            malicious_pos_weight=malicious_pos_weight,
            tactic_pos_weight=tactic_pos_weight,
            technique_pos_weight=technique_pos_weight,
            label_vocabulary=label_vocabulary.to_dict(),
            backbone_model_name_or_path=str(model_name_or_path),
        )
        model = cls(config)

        candidates = [attn_implementation] if attn_implementation else [None]
        candidates += ["eager"]
        backbone = None
        for candidate in candidates:
            try:
                extra = {} if candidate is None else {"attn_implementation": candidate}
                backbone = BertModel.from_pretrained(
                    model_name_or_path,
                    config=backbone_config,
                    **extra,
                    **kwargs,
                )
            except (ValueError, ImportError, RuntimeError):
                continue
            else:
                used = candidate or getattr(backbone.config, "_attn_implementation", None)
                if used:
                    config._attn_implementation = used
                break
        if backbone is None:  # pragma: no cover - eager always remains available
            raise RuntimeError("unable to load a small BERT backbone")
        model.bert.load_state_dict(backbone.state_dict(), strict=True)
        return model

    def get_input_embeddings(self) -> nn.Embedding:
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.bert.embeddings.word_embeddings = value

    def _compute_loss(
        self,
        malicious_logits: torch.Tensor,
        tactic_logits: torch.Tensor,
        technique_logits: torch.Tensor,
        labels: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        malicious_targets = labels["malicious"].to(
            device=malicious_logits.device, dtype=malicious_logits.dtype
        )
        tactic_targets = labels["tactics"].to(
            device=tactic_logits.device, dtype=tactic_logits.dtype
        )
        technique_targets = labels["techniques"].to(
            device=technique_logits.device, dtype=technique_logits.dtype
        )
        malicious_loss = F.binary_cross_entropy_with_logits(
            malicious_logits.view(-1),
            malicious_targets.view(-1),
            pos_weight=self.malicious_pos_weight.to(malicious_logits.device),
        )
        tactic_loss = F.binary_cross_entropy_with_logits(
            tactic_logits,
            tactic_targets,
            pos_weight=self.tactic_pos_weight.to(tactic_logits.device),
        )
        if technique_logits.shape[-1] == 0:
            technique_loss = technique_logits.sum() * 0.0
        else:
            technique_loss = F.binary_cross_entropy_with_logits(
                technique_logits,
                technique_targets,
                pos_weight=self.technique_pos_weight.to(technique_logits.device),
            )
        return malicious_loss + tactic_loss + technique_loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: Mapping[str, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> CommandRiskOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        backbone_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled = self.dropout(backbone_outputs.last_hidden_state[:, 0])
        malicious_logits = self.malicious_head(pooled).squeeze(-1)
        tactic_logits = self.tactic_head(pooled)
        technique_logits = self.technique_head(pooled)

        loss = None
        if labels is not None:
            loss = self._compute_loss(
                malicious_logits,
                tactic_logits,
                technique_logits,
                labels,
            )
        return CommandRiskOutput(
            loss=loss,
            malicious_logits=malicious_logits,
            tactic_logits=tactic_logits,
            technique_logits=technique_logits,
        )

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        super().save_pretrained(destination, **kwargs)
        self.label_vocabulary.save(destination / "label_vocabulary.json")
        (destination / "model_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> SmallBertForCommandRisk:
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        label_path = Path(pretrained_model_name_or_path) / "label_vocabulary.json"
        if label_path.is_file():
            model.label_vocabulary = LabelVocabulary.load(label_path)
            model.config.label_vocabulary = model.label_vocabulary.to_dict()
        return model

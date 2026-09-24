"""Hugging Face Trainer integration for command-prefix classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from cmdrisk.model import ModernBertForCommandRisk
from cmdrisk.training.config import BatchConfig, TrainingRunConfig


class CommandRiskTrainer(Trainer):
    """Trainer that preserves the model's nested label dictionary."""

    def compute_loss(
        self,
        model: ModernBertForCommandRisk,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        if loss is None:  # pragma: no cover - defensive guard
            raise RuntimeError("model did not return a loss")
        return (loss, outputs) if return_outputs else loss


def _precision_flags() -> dict[str, bool]:
    if not torch.cuda.is_available():
        return {"fp16": False, "bf16": False}
    if torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    return {"fp16": True, "bf16": False}


def build_training_arguments(
    *,
    config: TrainingRunConfig,
    batch: BatchConfig,
    phase: str,
    output_dir: str | Path,
    max_length: int,
    learning_rate: float,
) -> TrainingArguments:
    precision = _precision_flags()
    use_long_phase = phase == "long"
    phase_overrides = dict(config.phase_overrides.get(phase, {}))
    phase_overrides.pop("resume_from_checkpoint", None)
    arguments: dict[str, Any] = {
        "output_dir": str(output_dir),
        "do_train": True,
        "do_eval": True,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 10,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "num_train_epochs": config.max_epochs,
        "per_device_train_batch_size": batch.micro_batch_size,
        "per_device_eval_batch_size": max(1, batch.micro_batch_size),
        "gradient_accumulation_steps": batch.gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": config.weight_decay,
        # Transformers 5 treats a positive float below 1 as a warmup ratio.
        "warmup_steps": config.warmup_ratio,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_torch",
        "max_grad_norm": 1.0,
        "seed": config.seed,
        "data_seed": config.seed,
        "dataloader_num_workers": config.dataloader_num_workers,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "gradient_checkpointing": use_long_phase,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if use_long_phase else None,
        "report_to": [] if config.report_to == "none" else [config.report_to],
        "ddp_find_unused_parameters": False,
        "dataloader_pin_memory": torch.cuda.is_available(),
    }
    arguments.update(precision)
    # ``phase_overrides`` lets the smoke test cap the run with ``max_steps``
    # and adjust other Trainer knobs without touching pipeline code.
    arguments.update(phase_overrides)
    return TrainingArguments(**arguments)


def train_phase(
    *,
    model: ModernBertForCommandRisk,
    train_dataset: Any,
    eval_dataset: Any,
    collator: Any,
    config: TrainingRunConfig,
    batch: BatchConfig,
    phase: str,
    output_dir: str | Path,
    max_length: int,
    learning_rate: float,
) -> CommandRiskTrainer:
    resume_from_checkpoint = config.phase_overrides.get(phase, {}).get("resume_from_checkpoint")
    arguments = build_training_arguments(
        config=config,
        batch=batch,
        phase=phase,
        output_dir=output_dir,
        max_length=max_length,
        learning_rate=learning_rate,
    )
    trainer = CommandRiskTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir))
    return trainer

"""Training configuration and VRAM-aware batch planning."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Phase = Literal["short", "long"]


@dataclass(frozen=True)
class BatchConfig:
    phase: Phase
    micro_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    effective_batch_size: int


def choose_micro_batch(*, vram_gb: float, phase: Phase) -> int:
    """Choose a conservative sequence-length-aware micro batch."""

    if vram_gb < 0:
        raise ValueError("vram_gb must be non-negative")
    if phase == "short":
        if vram_gb >= 40:
            return 32
        if vram_gb >= 24:
            return 16
        return 8
    if phase == "long":
        if vram_gb >= 40:
            return 4
        if vram_gb >= 24:
            return 2
        return 1
    raise ValueError(f"unknown phase: {phase}")


def target_effective_batch(phase: Phase) -> int:
    if phase == "short":
        return 64
    if phase == "long":
        return 16
    raise ValueError(f"unknown phase: {phase}")


def choose_batch_config(
    *,
    vram_gb: float,
    phase: Phase,
    world_size: int = 1,
) -> BatchConfig:
    if world_size < 1:
        raise ValueError("world_size must be positive")
    micro = choose_micro_batch(vram_gb=vram_gb, phase=phase)
    target = target_effective_batch(phase)
    accumulation = max(1, math.ceil(target / (micro * world_size)))
    effective = micro * accumulation * world_size
    return BatchConfig(phase, micro, accumulation, world_size, effective)


@dataclass(frozen=True)
class TrainingRunConfig:
    data_path: str
    output_dir: str
    model_name: str = "answerdotai/ModernBERT-base"
    seed: int = 42
    max_epochs: int = 2
    short_max_length: int = 2048
    long_max_length: int = 8192
    command_max_tokens: int = 512
    max_prefixes: int = 32
    learning_rate_short: float = 2e-5
    learning_rate_long: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    dropout: float = 0.1
    pos_weight_cap: float = 10.0
    min_technique_sessions: int = 50
    min_technique_families: int = 3
    calibration_fpr: float = 0.01
    proxy_audit_path: str | None = None
    early_stopping_patience: int = 1
    dataloader_num_workers: int = 2
    vram_gb: float | None = None
    world_size: int | None = None
    skip_long_phase: bool = False
    report_to: str = "none"
    phase_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def output_path(self) -> Path:
        return Path(self.output_dir)

    def __post_init__(self) -> None:
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.short_max_length < 1 or self.long_max_length < 1:
            raise ValueError("sequence lengths must be positive")
        if not 0 <= self.calibration_fpr <= 1:
            raise ValueError("calibration_fpr must be in [0, 1]")
        if self.pos_weight_cap <= 0:
            raise ValueError("pos_weight_cap must be positive")

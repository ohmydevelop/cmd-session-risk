"""Public data contracts for training, scoring, and subagent review."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"


class SourceKind(StrEnum):
    REAL_MALICIOUS = "real_malicious"
    REAL_BENIGN = "real_benign"
    SCRIPT_BENIGN = "script_benign"
    LLM_VARIANT = "llm_variant"
    HARD_NEGATIVE = "hard_negative"


class SplitName(StrEnum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
    BLIND = "blind"


class LabelQuality(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    REJECTED = "rejected"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=True)


class SourceInfo(StrictModel):
    kind: SourceKind
    name: str = Field(min_length=1)
    parent_session_id: str | None = None
    license: str = Field(min_length=1)


class CommandEvent(StrictModel):
    cmd: str = Field(min_length=1)
    dt: float | None = Field(default=None, ge=0)

    @field_validator("cmd")
    @classmethod
    def command_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cmd must contain non-whitespace characters")
        return value


class Label(StrictModel):
    malicious: Literal[0, 1]
    first_evidence_index: int | None = Field(default=None, ge=0)
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    quality: LabelQuality
    teacher_votes: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_evidence_consistency(self) -> Label:
        if self.malicious == 0 and self.first_evidence_index is not None:
            raise ValueError("benign labels cannot have first_evidence_index")
        if self.malicious == 1 and self.first_evidence_index is None:
            raise ValueError("malicious labels require first_evidence_index")
        indices = [idx for values in self.technique_evidence.values() for idx in values]
        if self.first_evidence_index is not None and indices:
            if min(indices) < self.first_evidence_index:
                raise ValueError("technique evidence cannot precede first_evidence_index")
        return self


class SessionRecord(StrictModel):
    schema_version: str = SCHEMA_VERSION
    session_id: str = Field(min_length=1)
    source: SourceInfo
    family_id: str = Field(min_length=1)
    commands: list[CommandEvent] = Field(min_length=1)
    meta: dict[str, Any] = Field(default_factory=dict)
    label: Label
    split: SplitName | None = None

    @model_validator(mode="after")
    def check_indices(self) -> SessionRecord:
        if self.label.first_evidence_index is not None:
            if self.label.first_evidence_index >= len(self.commands):
                raise ValueError("first_evidence_index is outside commands")
        for technique, indices in self.label.technique_evidence.items():
            if not technique.startswith("T") or not technique[1:].replace(".", "").isdigit():
                raise ValueError(f"invalid ATT&CK technique id: {technique}")
            if any(index < 0 or index >= len(self.commands) for index in indices):
                raise ValueError(f"technique evidence index out of range for {technique}")
        return self


class TacticScore(StrictModel):
    id: str
    name: str
    score: float = Field(ge=0, le=1)


class TechniqueScore(StrictModel):
    id: str
    name: str
    score: float = Field(ge=0, le=1)


class ScoreEntry(StrictModel):
    command_index: int = Field(ge=0)
    malicious_probability: float = Field(ge=0, le=1)
    risk_score: float = Field(ge=0, le=1)
    attack_stage: str
    tactics: list[TacticScore]
    techniques: list[TechniqueScore]
    alert: bool


class SequenceScore(StrictModel):
    session_id: str
    entries: list[ScoreEntry]
    peak_probability: float = Field(ge=0, le=1)
    alerted: bool
    model_version: str
    schema_version: str = SCHEMA_VERSION


class PrefixRecord(StrictModel):
    session_id: str
    command_count: int = Field(ge=1)
    split: SplitName
    source: SourceInfo
    family_id: str
    commands: list[CommandEvent]
    label: Label

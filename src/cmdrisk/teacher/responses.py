"""Strict response contracts returned by Teacher and proxy-audit models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictResponse(BaseModel):
    """Reject provider-specific extras so model output cannot silently drift."""

    model_config = ConfigDict(extra="forbid")


class TeacherLabelResponse(StrictResponse):
    malicious: bool
    first_evidence_index: int | None = Field(default=None, ge=0)
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_evidence_consistency(self) -> TeacherLabelResponse:
        if self.malicious and self.first_evidence_index is None:
            raise ValueError("malicious labels require first_evidence_index")
        if not self.malicious and self.first_evidence_index is not None:
            raise ValueError("benign labels cannot have first_evidence_index")
        if not self.malicious and (self.technique_evidence or self.tactics):
            raise ValueError("benign labels cannot include attack evidence or tactics")
        for technique, indices in self.technique_evidence.items():
            if not technique or not indices:
                raise ValueError("technique_evidence entries require IDs and indices")
            if any(index < 0 for index in indices):
                raise ValueError("technique evidence indices must be non-negative")
            if self.first_evidence_index is not None and any(
                index < self.first_evidence_index for index in indices
            ):
                raise ValueError("technique evidence cannot precede first_evidence_index")
        return self


class RewriteVariant(StrictResponse):
    commands: list[str] = Field(min_length=1)
    first_evidence_index: int = Field(ge=0)
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_evidence_index(self) -> RewriteVariant:
        if self.first_evidence_index >= len(self.commands):
            raise ValueError("first_evidence_index is outside commands")
        for technique, indices in self.technique_evidence.items():
            if not technique.strip() or not indices:
                raise ValueError("technique_evidence entries require IDs and indices")
            if any(index < self.first_evidence_index for index in indices):
                raise ValueError("technique evidence cannot precede first_evidence_index")
            if any(index >= len(self.commands) for index in indices):
                raise ValueError("technique evidence index is outside commands")
        return self


class RewriteBatchResponse(StrictResponse):
    variants: list[RewriteVariant]


class HardNegativeBatchResponse(StrictResponse):
    variants: list[list[str]]

    @model_validator(mode="after")
    def validate_variants(self) -> HardNegativeBatchResponse:
        if any(not variant for variant in self.variants):
            raise ValueError("hard-negative variants must contain at least one command")
        return self


class MaliciousValidationResponse(StrictResponse):
    malicious: bool
    parent_behavior_preserved: bool
    first_evidence_index: int | None = Field(default=None, ge=0)
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_evidence_consistency(self) -> MaliciousValidationResponse:
        if self.malicious and self.first_evidence_index is None:
            raise ValueError("malicious validation requires first_evidence_index")
        if not self.malicious and self.first_evidence_index is not None:
            raise ValueError("benign validation cannot include first_evidence_index")
        if not self.malicious and (self.technique_evidence or self.techniques or self.tactics):
            raise ValueError("benign validation cannot include attack evidence or tactics")
        for technique, indices in self.technique_evidence.items():
            if not technique.strip() or not indices:
                raise ValueError("technique_evidence entries require IDs and indices")
            if any(index < 0 for index in indices):
                raise ValueError("technique evidence indices must be non-negative")
            if self.first_evidence_index is not None and any(
                index < self.first_evidence_index for index in indices
            ):
                raise ValueError("technique evidence cannot precede first_evidence_index")
        return self


class HardNegativeValidationResponse(StrictResponse):
    malicious: bool
    confidence: float = Field(ge=0, le=1)


class ProxyReviewResponse(StrictResponse):
    decision: Literal["malicious", "benign", "uncertain"]
    first_evidence_index: int | None = Field(default=None, ge=0)
    technique_evidence: dict[str, list[int]] = Field(default_factory=dict)
    tactics: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_decision(self) -> ProxyReviewResponse:
        if self.decision == "malicious" and self.first_evidence_index is None:
            raise ValueError("malicious review requires first_evidence_index")
        if self.decision != "malicious" and self.first_evidence_index is not None:
            raise ValueError("non-malicious reviews cannot include first_evidence_index")
        if self.decision != "malicious" and (self.technique_evidence or self.tactics):
            raise ValueError("non-malicious reviews cannot include attack evidence or tactics")
        if self.first_evidence_index is not None:
            for technique, indices in self.technique_evidence.items():
                if not technique.strip() or not indices:
                    raise ValueError("technique_evidence entries require IDs and indices")
                if any(index < 0 for index in indices):
                    raise ValueError("technique evidence indices must be non-negative")
                if any(index < self.first_evidence_index for index in indices):
                    raise ValueError("technique evidence cannot precede first_evidence_index")
        return self

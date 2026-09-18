"""Validated malicious augmentation and hard-negative generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

from cmdrisk.schema import (
    CommandEvent,
    Label,
    LabelQuality,
    SessionRecord,
    SourceInfo,
    SourceKind,
    SplitName,
)
from cmdrisk.taxonomy import normalize_tactic, normalize_technique, tactics_for_techniques
from cmdrisk.teacher.responses import (
    HardNegativeBatchResponse,
    HardNegativeValidationResponse,
    MaliciousValidationResponse,
    RewriteBatchResponse,
)


class AugmentationError(ValueError):
    """Raised when a generation batch cannot satisfy its validation contract."""


class InsufficientVariantsError(AugmentationError):
    def __init__(self, *, accepted: int, required: int, rejected: int) -> None:
        super().__init__(
            f"only {accepted} valid variants were produced; "
            f"required={required}, rejected={rejected}"
        )
        self.accepted = accepted
        self.required = required
        self.rejected = rejected


class MaliciousModel(Protocol):
    def rewrite(self, session: SessionRecord, count: int) -> RewriteBatchResponse: ...

    def validate_malicious_variant(
        self,
        parent: SessionRecord,
        commands: list[str],
        first_evidence_index: int,
    ) -> MaliciousValidationResponse: ...


class HardNegativeModel(Protocol):
    def hard_negative(self, seed: dict[str, object], count: int) -> HardNegativeBatchResponse: ...

    def validate_hard_negative(
        self,
        commands: list[str],
    ) -> HardNegativeValidationResponse: ...


@dataclass(frozen=True)
class AugmentationResult:
    parent_session_id: str
    generated: int
    rejected: int
    records: tuple[SessionRecord, ...]


@dataclass(frozen=True)
class HardNegativeResult:
    generated: int
    rejected: int
    records: tuple[SessionRecord, ...]


class MaliciousAugmenter:
    """Expand one malicious parent into 8-12 semantic-preserving variants."""

    def __init__(
        self,
        generator: MaliciousModel,
        validators: tuple[MaliciousModel, MaliciousModel],
        *,
        confidence_floor: float = 0.75,
        evidence_tolerance: int = 1,
        max_rounds: int = 3,
    ) -> None:
        if len(validators) != 2:
            raise ValueError("malicious augmentation requires exactly two validators")
        if confidence_floor < 0 or confidence_floor > 1:
            raise ValueError("confidence_floor must be between 0 and 1")
        if evidence_tolerance < 0:
            raise ValueError("evidence_tolerance must be non-negative")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.generator = generator
        self.validators = validators
        self.confidence_floor = confidence_floor
        self.evidence_tolerance = evidence_tolerance
        self.max_rounds = max_rounds

    def augment(self, parent: SessionRecord, count: int = 10) -> AugmentationResult:
        if parent.label.malicious != 1 or parent.label.first_evidence_index is None:
            raise AugmentationError("augmentation parent must have a malicious consensus label")
        if count < 8 or count > 12:
            raise AugmentationError("count must be between 8 and 12")

        accepted: list[SessionRecord] = []
        seen: set[tuple[str, ...]] = set()
        rejected = 0
        generated = 0
        for _ in range(self.max_rounds):
            remaining = count - len(accepted)
            if remaining <= 0:
                break
            batch = self.generator.rewrite(parent, remaining)
            if not batch.variants:
                continue
            for variant in batch.variants:
                generated += 1
                command_key = tuple(variant.commands)
                if command_key in seen:
                    rejected += 1
                    continue
                seen.add(command_key)
                record = self._validate_variant(parent, variant)
                if record is None:
                    rejected += 1
                    continue
                accepted.append(record)
                if len(accepted) >= count:
                    break
        if len(accepted) < count:
            raise InsufficientVariantsError(
                accepted=len(accepted),
                required=count,
                rejected=rejected,
            )
        return AugmentationResult(
            parent_session_id=parent.session_id,
            generated=generated,
            rejected=rejected,
            records=tuple(accepted[:count]),
        )

    def _validate_variant(self, parent: SessionRecord, variant: object) -> SessionRecord | None:
        from cmdrisk.teacher.responses import RewriteVariant

        if not isinstance(variant, RewriteVariant):
            return None
        parent_evidence = parent.label.first_evidence_index
        assert parent_evidence is not None
        if abs(variant.first_evidence_index - parent_evidence) > self.evidence_tolerance:
            return None
        commands = variant.commands
        first_validator = self.validators[0].validate_malicious_variant(
            parent, commands, variant.first_evidence_index
        )
        second_validator = self.validators[1].validate_malicious_variant(
            parent, commands, variant.first_evidence_index
        )
        validators = (first_validator, second_validator)
        if not all(
            _validator_accepts(
                item,
                parent_evidence,
                self.evidence_tolerance,
                self.confidence_floor,
            )
            for item in validators
        ):
            return None
        validator_indices = [item.first_evidence_index for item in validators]
        if None in validator_indices:
            return None
        assert validator_indices[0] is not None and validator_indices[1] is not None
        all_evidence_indices = [variant.first_evidence_index, *validator_indices]
        if max(all_evidence_indices) - min(all_evidence_indices) > self.evidence_tolerance:
            return None
        first_evidence = min(all_evidence_indices)
        techniques = _consensus_techniques(
            [
                variant.techniques,
                first_validator.techniques,
                second_validator.techniques,
            ]
        )
        tactics = _consensus_tactics(
            [variant.tactics, first_validator.tactics, second_validator.tactics],
            techniques,
        )
        label = Label(
            malicious=1,
            first_evidence_index=first_evidence,
            technique_evidence={technique: [first_evidence] for technique in techniques},
            tactics=tactics,
            techniques=techniques,
            quality=LabelQuality.HIGH,
            teacher_votes=[
                {
                    "role": "generator",
                    "first_evidence_index": variant.first_evidence_index,
                    "confidence": variant.confidence,
                },
                {
                    "role": "validator_1",
                    "first_evidence_index": first_validator.first_evidence_index,
                    "confidence": first_validator.confidence,
                },
                {
                    "role": "validator_2",
                    "first_evidence_index": second_validator.first_evidence_index,
                    "confidence": second_validator.confidence,
                },
            ],
        )
        source = SourceInfo(
            kind=SourceKind.LLM_VARIANT,
            name="teacher",
            parent_session_id=parent.session_id,
            license=parent.source.license,
        )
        return SessionRecord(
            session_id=_variant_id(parent.session_id, commands, first_evidence),
            source=source,
            family_id=parent.family_id,
            commands=[CommandEvent(cmd=command) for command in commands],
            meta={
                **parent.meta,
                "parent_first_evidence_index": parent_evidence,
                "generator_variant_index": variant.first_evidence_index,
            },
            label=label,
            split=parent.split,
        )


class HardNegativeGenerator:
    """Generate and independently validate benign attack-lookalike sessions."""

    def __init__(
        self,
        generator: HardNegativeModel,
        validators: tuple[HardNegativeModel, HardNegativeModel],
        *,
        confidence_floor: float = 0.75,
        max_rounds: int = 3,
    ) -> None:
        if len(validators) != 2:
            raise ValueError("hard-negative generation requires exactly two validators")
        if confidence_floor < 0 or confidence_floor > 1:
            raise ValueError("confidence_floor must be between 0 and 1")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        self.generator = generator
        self.validators = validators
        self.confidence_floor = confidence_floor
        self.max_rounds = max_rounds

    def generate(
        self,
        seed: dict[str, object],
        *,
        source: SourceInfo,
        family_id: str,
        count: int = 8,
        split: SplitName | None = None,
    ) -> HardNegativeResult:
        if count < 1 or count > 32:
            raise AugmentationError("count must be between 1 and 32")
        accepted: list[SessionRecord] = []
        seen: set[tuple[str, ...]] = set()
        rejected = 0
        generated = 0
        for _ in range(self.max_rounds):
            remaining = count - len(accepted)
            if remaining <= 0:
                break
            batch = self.generator.hard_negative(seed, remaining)
            for commands in batch.variants:
                generated += 1
                key = tuple(commands)
                if key in seen:
                    rejected += 1
                    continue
                seen.add(key)
                first = self.validators[0].validate_hard_negative(commands)
                second = self.validators[1].validate_hard_negative(commands)
                if (
                    first.malicious
                    or second.malicious
                    or first.confidence < self.confidence_floor
                    or second.confidence < self.confidence_floor
                ):
                    rejected += 1
                    continue
                accepted.append(
                    _hard_negative_record(
                        commands,
                        source=source,
                        family_id=family_id,
                        split=split,
                        validators=(first, second),
                    )
                )
                if len(accepted) >= count:
                    break
        if len(accepted) < count:
            raise InsufficientVariantsError(
                accepted=len(accepted),
                required=count,
                rejected=rejected,
            )
        return HardNegativeResult(
            generated=generated,
            rejected=rejected,
            records=tuple(accepted[:count]),
        )


def _validator_accepts(
    response: MaliciousValidationResponse,
    parent_evidence: int,
    tolerance: int,
    confidence_floor: float,
) -> bool:
    return (
        response.malicious
        and response.parent_behavior_preserved
        and response.first_evidence_index is not None
        and abs(response.first_evidence_index - parent_evidence) <= tolerance
        and response.confidence >= confidence_floor
    )


def _consensus_techniques(groups: list[list[str]]) -> list[str]:
    counts: Counter[str] = Counter()
    for group in groups:
        counts.update(
            {
                technique
                for raw in group
                if (technique := normalize_technique(raw)) is not None
            }
        )
    return sorted(technique for technique, count in counts.items() if count >= 2)


def _consensus_tactics(groups: list[list[str]], techniques: list[str]) -> list[str]:
    counts: Counter[str] = Counter()
    for group in groups:
        counts.update(
            {tactic for raw in group if (tactic := normalize_tactic(raw)) is not None}
        )
    explicit = {tactic for tactic, count in counts.items() if count >= 2}
    return sorted(explicit | set(tactics_for_techniques(techniques)))


def _variant_id(parent_id: str, commands: list[str], first_evidence: int) -> str:
    payload = json.dumps(
        {
            "parent": parent_id,
            "commands": commands,
            "first_evidence_index": first_evidence,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _hard_negative_record(
    commands: list[str],
    *,
    source: SourceInfo,
    family_id: str,
    split: SplitName | None,
    validators: tuple[HardNegativeValidationResponse, HardNegativeValidationResponse],
) -> SessionRecord:
    payload = json.dumps(commands, separators=(",", ":"), ensure_ascii=False)
    session_id = f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"
    return SessionRecord(
        session_id=session_id,
        source=source.model_copy(update={"kind": SourceKind.HARD_NEGATIVE}),
        family_id=family_id,
        commands=[CommandEvent(cmd=command) for command in commands],
        label=Label(
            malicious=0,
            first_evidence_index=None,
            technique_evidence={},
            tactics=[],
            techniques=[],
            quality=LabelQuality.HIGH,
            teacher_votes=[
                {"role": "validator_1", "confidence": validators[0].confidence},
                {"role": "validator_2", "confidence": validators[1].confidence},
            ],
        ),
        split=split,
    )

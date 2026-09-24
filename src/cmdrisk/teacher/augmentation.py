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
from cmdrisk.taxonomy import normalize_technique, tactics_for_techniques
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
    family_id: str | None = None


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
        reviewer_configs: object | None = None,
        require_reviewer_revisions: bool = True,
    ) -> None:
        if len(validators) != 2:
            raise ValueError("malicious augmentation requires exactly two validators")
        if confidence_floor < 0 or confidence_floor > 1:
            raise ValueError("confidence_floor must be between 0 and 1")
        if evidence_tolerance < 0:
            raise ValueError("evidence_tolerance must be non-negative")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if reviewer_configs is not None:
            from cmdrisk.teacher.client import validate_distinct_reviewer_configs

            validate_distinct_reviewer_configs(
                reviewer_configs,  # type: ignore[arg-type]
                roles=("generator", "validator_1", "validator_2"),
                require_revision=require_reviewer_revisions,
            )
        self.generator = generator
        self.validators = validators
        self.confidence_floor = confidence_floor
        self.evidence_tolerance = evidence_tolerance
        self.max_rounds = max_rounds

    def augment(self, parent: SessionRecord, count: int = 10) -> AugmentationResult:
        if parent.label.malicious != 1 or parent.label.first_evidence_index is None:
            raise AugmentationError("augmentation parent must have a malicious consensus label")
        if not parent.label.technique_evidence:
            raise AugmentationError("augmentation parent must include per-technique evidence")
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
        source_evidence = [
            _source_technique_evidence(
                variant.technique_evidence,
                variant.techniques,
                variant.first_evidence_index,
                len(commands),
            ),
            _source_technique_evidence(
                first_validator.technique_evidence,
                first_validator.techniques,
                first_validator.first_evidence_index,
                len(commands),
            ),
            _source_technique_evidence(
                second_validator.technique_evidence,
                second_validator.techniques,
                second_validator.first_evidence_index,
                len(commands),
            ),
        ]
        technique_evidence = _consensus_technique_evidence(
            source_evidence,
            first_evidence,
            len(commands),
        )
        if not technique_evidence:
            return None
        tactics = _tactics_for_evidence(technique_evidence)
        label = Label(
            malicious=1,
            first_evidence_index=first_evidence,
            technique_evidence=technique_evidence,
            tactics=tactics,
            techniques=list(technique_evidence),
            quality=LabelQuality.HIGH,
            teacher_votes=[
                _malicious_vote_payload(
                    "generator",
                    variant.first_evidence_index,
                    variant.technique_evidence,
                    variant.techniques,
                    variant.confidence,
                ),
                _malicious_vote_payload(
                    "validator_1",
                    first_validator.first_evidence_index,
                    first_validator.technique_evidence,
                    first_validator.techniques,
                    first_validator.confidence,
                ),
                _malicious_vote_payload(
                    "validator_2",
                    second_validator.first_evidence_index,
                    second_validator.technique_evidence,
                    second_validator.techniques,
                    second_validator.confidence,
                ),
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
                "source_generator": "teacher",
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
        reviewer_configs: object | None = None,
        require_reviewer_revisions: bool = True,
    ) -> None:
        if len(validators) != 2:
            raise ValueError("hard-negative generation requires exactly two validators")
        if confidence_floor < 0 or confidence_floor > 1:
            raise ValueError("confidence_floor must be between 0 and 1")
        if max_rounds < 1:
            raise ValueError("max_rounds must be positive")
        if reviewer_configs is not None:
            from cmdrisk.teacher.client import validate_distinct_reviewer_configs

            validate_distinct_reviewer_configs(
                reviewer_configs,  # type: ignore[arg-type]
                roles=("generator", "validator_1", "validator_2"),
                require_revision=require_reviewer_revisions,
            )
        self.generator = generator
        self.validators = validators
        self.confidence_floor = confidence_floor
        self.max_rounds = max_rounds

    def generate(
        self,
        seed: dict[str, object],
        *,
        source: SourceInfo,
        family_id: str | None = None,
        count: int = 8,
        split: SplitName | None = None,
    ) -> HardNegativeResult:
        """Generate hard negatives from a seed/context.

        ``family_id`` is accepted for backwards compatibility but deliberately
        ignored.  Family identity is derived from the canonical seed and source
        so an operator cannot accidentally merge unrelated families.
        """

        if count < 1 or count > 32:
            raise AugmentationError("count must be between 1 and 32")
        derived_family_id = derive_hard_negative_family_id(seed, source)
        seed_sha256 = _seed_sha256(seed, source)
        seed_parent_session_id = _seed_parent_session_id(seed)
        effective_source = source
        if seed_parent_session_id and not source.parent_session_id:
            effective_source = source.model_copy(
                update={"parent_session_id": seed_parent_session_id}
            )
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
                        source=effective_source,
                        family_id=derived_family_id,
                        split=split,
                        validators=(first, second),
                        seed=seed,
                        seed_sha256=seed_sha256,
                        seed_parent_session_id=seed_parent_session_id,
                        operator_family_id=family_id,
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
            family_id=derived_family_id,
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
        and bool(response.technique_evidence)
    )


def _source_technique_evidence(
    evidence: dict[str, list[int]],
    techniques: list[str],
    first_evidence: int | None,
    command_count: int,
) -> dict[str, list[int]]:
    """Normalize one response's evidence without inventing indices."""

    if first_evidence is None:
        return {}
    result: dict[str, list[int]] = {}
    # Evidence keys are authoritative.  The advertised technique list is kept
    # separately in teacher_votes and never used to synthesize an index.
    for raw_technique, raw_indices in evidence.items():
        technique = normalize_technique(raw_technique)
        if technique is None:
            continue
        valid = sorted(
            {
                int(index)
                for index in raw_indices
                if isinstance(index, int) and first_evidence <= index < command_count
            }
        )
        if valid:
            result[technique] = sorted(set(result.get(technique, [])) | set(valid))
    return result


def _consensus_technique_evidence(
    groups: list[dict[str, list[int]]],
    first_evidence: int,
    command_count: int,
) -> dict[str, list[int]]:
    counts: Counter[str] = Counter()
    indices: dict[str, set[int]] = {}
    for group in groups:
        for technique, values in group.items():
            valid = {index for index in values if first_evidence <= index < command_count}
            if not valid:
                continue
            counts[technique] += 1
            indices.setdefault(technique, set()).update(valid)
    return {
        technique: sorted(indices[technique])
        for technique in sorted(indices)
        if counts[technique] >= 2
    }


def _tactics_for_evidence(technique_evidence: dict[str, list[int]]) -> list[str]:
    return tactics_for_techniques(list(technique_evidence))


def _malicious_vote_payload(
    role: str,
    first_evidence_index: int | None,
    technique_evidence: dict[str, list[int]],
    techniques: list[str],
    confidence: float,
) -> dict[str, object]:
    return {
        "role": role,
        "provenance": "raw_model_vote",
        "first_evidence_index": first_evidence_index,
        "technique_evidence": {
            technique: list(indices) for technique, indices in technique_evidence.items()
        },
        "techniques": list(techniques),
        "confidence": confidence,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def derive_hard_negative_family_id(seed: dict[str, object], source: SourceInfo) -> str:
    """Derive a stable family ID from the seed and source provenance."""

    payload = {
        "version": 1,
        "seed": seed,
        "source": source.model_dump(mode="json", exclude_none=True),
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"hard-negative:{digest}"


def _seed_sha256(seed: dict[str, object], source: SourceInfo) -> str:
    payload = {
        "seed": seed,
        "source": source.model_dump(mode="json", exclude_none=True),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _seed_parent_session_id(seed: dict[str, object]) -> str | None:
    for key in ("parent_session_id", "seed_parent_session_id", "session_id"):
        value = seed.get(key)
        if isinstance(value, str) and value:
            return value
    nested = seed.get("source")
    if isinstance(nested, dict):
        for key in ("parent_session_id", "session_id"):
            value = nested.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _variant_id(parent_id: str, commands: list[str], first_evidence: int) -> str:
    payload = _canonical_json(
        {
            "parent": parent_id,
            "commands": commands,
            "first_evidence_index": first_evidence,
        }
    )
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def _hard_negative_record(
    commands: list[str],
    *,
    source: SourceInfo,
    family_id: str,
    split: SplitName | None,
    validators: tuple[HardNegativeValidationResponse, HardNegativeValidationResponse],
    seed: dict[str, object],
    seed_sha256: str,
    seed_parent_session_id: str | None,
    operator_family_id: str | None,
) -> SessionRecord:
    session_payload = {
        "family_id": family_id,
        "commands": commands,
        "seed_sha256": seed_sha256,
    }
    session_id = f"sha256:{hashlib.sha256(_canonical_json(session_payload).encode()).hexdigest()}"
    meta: dict[str, object] = {
        "hard_negative_seed_sha256": seed_sha256,
        "hard_negative_seed": seed,
        "hard_negative_family_id": family_id,
        "seed_parent_session_id": seed_parent_session_id,
        "seed_source": source.model_dump(mode="json", exclude_none=True),
    }
    if operator_family_id is not None and operator_family_id != family_id:
        meta["operator_family_id_ignored"] = operator_family_id
    return SessionRecord(
        session_id=session_id,
        source=source.model_copy(update={"kind": SourceKind.HARD_NEGATIVE}),
        family_id=family_id,
        commands=[CommandEvent(cmd=command) for command in commands],
        meta=meta,
        label=Label(
            malicious=0,
            first_evidence_index=None,
            technique_evidence={},
            tactics=[],
            techniques=[],
            quality=LabelQuality.HIGH,
            teacher_votes=[
                {
                    "role": "validator_1",
                    "provenance": "raw_model_vote",
                    "confidence": validators[0].confidence,
                },
                {
                    "role": "validator_2",
                    "provenance": "raw_model_vote",
                    "confidence": validators[1].confidence,
                },
            ],
        ),
        split=split,
    )

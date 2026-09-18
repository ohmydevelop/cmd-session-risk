from __future__ import annotations

import pytest

from cmdrisk.schema import CommandEvent, Label, LabelQuality, SessionRecord, SourceInfo
from cmdrisk.teacher.augmentation import (
    HardNegativeGenerator,
    InsufficientVariantsError,
    MaliciousAugmenter,
)
from cmdrisk.teacher.responses import (
    HardNegativeBatchResponse,
    HardNegativeValidationResponse,
    MaliciousValidationResponse,
    RewriteBatchResponse,
    RewriteVariant,
)


class FakeMaliciousModel:
    def __init__(self, *, count: int, offset: int = 0, validator: bool = False) -> None:
        self.count = count
        self.offset = offset
        self.validator = validator

    def rewrite(self, session: SessionRecord, count: int) -> RewriteBatchResponse:
        base = session.label.first_evidence_index or 0
        return RewriteBatchResponse(
            variants=[
                RewriteVariant(
                    commands=[
                        f"curl -o /tmp/a{index} URL",
                        f"chmod +x /tmp/a{index}",
                        f"/tmp/a{index}",
                        "true",
                    ],
                    first_evidence_index=max(0, base + self.offset),
                    tactics=["TA0011", "TA0005"],
                    techniques=["T1105", "T1222"],
                    confidence=0.95,
                )
                for index in range(min(count, self.count))
            ]
        )

    def validate_malicious_variant(
        self,
        parent: SessionRecord,
        commands: list[str],
        first_evidence_index: int,
    ) -> MaliciousValidationResponse:
        return MaliciousValidationResponse(
            malicious=not self.validator,
            parent_behavior_preserved=not self.validator,
            first_evidence_index=first_evidence_index if not self.validator else None,
            tactics=["TA0011", "TA0005"] if not self.validator else [],
            techniques=["T1105", "T1222"] if not self.validator else [],
            confidence=0.95,
        )


class FakeHardNegativeModel:
    def __init__(
        self,
        *,
        confidence: float = 0.95,
        count: int = 2,
        malicious: bool = False,
    ) -> None:
        self.confidence = confidence
        self.count = count
        self.malicious = malicious

    def hard_negative(self, seed: dict[str, object], count: int) -> HardNegativeBatchResponse:
        return HardNegativeBatchResponse(
            variants=[
                [
                    f"git clone https://example.invalid/project{index}.git",
                    "cd project",
                    "./configure",
                    "make",
                    "make install",
                ]
                for index in range(min(count, self.count))
            ]
        )

    def validate_hard_negative(self, commands: list[str]) -> HardNegativeValidationResponse:
        return HardNegativeValidationResponse(
            malicious=self.malicious,
            confidence=self.confidence,
        )


def _parent() -> SessionRecord:
    return SessionRecord(
        session_id="parent-1",
        source=SourceInfo(kind="real_malicious", name="fixture", license="CC0-1.0"),
        family_id="attack-family",
        commands=[
            CommandEvent(cmd="curl -o /tmp/a URL"),
            CommandEvent(cmd="chmod +x /tmp/a"),
            CommandEvent(cmd="/tmp/a"),
            CommandEvent(cmd="true"),
        ],
        label=Label(
            malicious=1,
            first_evidence_index=1,
            technique_evidence={"T1105": [1], "T1222": [1]},
            tactics=["TA0011", "TA0005"],
            techniques=["T1105", "T1222"],
            quality=LabelQuality.HIGH,
        ),
        split="train",
    )


def test_malicious_augmentation_requires_two_accepting_validators() -> None:
    generator = FakeMaliciousModel(count=8)
    result = MaliciousAugmenter(
        generator,
        (FakeMaliciousModel(count=8), FakeMaliciousModel(count=8)),
        max_rounds=1,
    ).augment(_parent(), count=8)
    assert len(result.records) == 8
    for record in result.records:
        assert record.source.kind == "llm_variant"
        assert record.source.parent_session_id == "parent-1"
        assert record.family_id == "attack-family"
        assert record.split == "train"
        assert record.label.first_evidence_index == 1
        assert record.label.techniques == ["T1105", "T1222"]


def test_malicious_augmentation_rejects_validator_disagreement() -> None:
    generator = FakeMaliciousModel(count=8)
    with pytest.raises(InsufficientVariantsError):
        MaliciousAugmenter(
            generator,
            (
                FakeMaliciousModel(count=8),
                FakeMaliciousModel(count=8, validator=True),
            ),
            max_rounds=1,
        ).augment(_parent(), count=8)


def test_malicious_augmentation_rejects_evidence_drift() -> None:
    generator = FakeMaliciousModel(count=8, offset=2)
    with pytest.raises(InsufficientVariantsError):
        MaliciousAugmenter(
            generator,
            (FakeMaliciousModel(count=8), FakeMaliciousModel(count=8)),
            max_rounds=1,
        ).augment(_parent(), count=8)


def test_hard_negative_requires_two_benign_validators() -> None:
    generator = FakeHardNegativeModel(confidence=0.95, count=2)
    result = HardNegativeGenerator(
        generator,
        (
            FakeHardNegativeModel(confidence=0.95),
            FakeHardNegativeModel(confidence=0.96),
        ),
        max_rounds=1,
    ).generate(
        {"context": "development build"},
        source=SourceInfo(kind="script_benign", name="repo", license="MIT"),
        family_id="benign-family",
        count=2,
        split="dev",
    )
    assert len(result.records) == 2
    for record in result.records:
        assert record.source.kind == "hard_negative"
        assert record.label.malicious == 0
        assert record.label.quality == LabelQuality.HIGH
        assert record.split == "dev"


def test_hard_negative_malicious_validator_rejects_candidate() -> None:
    generator = FakeHardNegativeModel(confidence=0.95, count=2)
    with pytest.raises(InsufficientVariantsError):
        HardNegativeGenerator(
            generator,
            (
                FakeHardNegativeModel(confidence=0.95),
                FakeHardNegativeModel(confidence=0.95, malicious=True),
            ),
            max_rounds=1,
        ).generate(
            {"context": "development build"},
            source=SourceInfo(kind="script_benign", name="repo", license="MIT"),
            family_id="benign-family",
            count=2,
        )


def test_hard_negative_low_confidence_is_rejected() -> None:
    generator = FakeHardNegativeModel(confidence=0.95, count=2)
    with pytest.raises(InsufficientVariantsError):
        HardNegativeGenerator(
            generator,
            (
                FakeHardNegativeModel(confidence=0.95),
                FakeHardNegativeModel(confidence=0.10),
            ),
            max_rounds=1,
        ).generate(
            {"context": "development build"},
            source=SourceInfo(kind="script_benign", name="repo", license="MIT"),
            family_id="benign-family",
            count=2,
        )

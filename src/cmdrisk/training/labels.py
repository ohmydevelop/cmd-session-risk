"""Data-derived label vocabularies and target encoding."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cmdrisk.taxonomy import TACTICS, tactics_for_techniques

_TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"expected a mapping-like record, got {type(value)!r}")


@dataclass(frozen=True)
class LabelTarget:
    malicious: float
    tactics: tuple[float, ...]
    techniques: tuple[float, ...]


@dataclass(frozen=True)
class LabelVocabulary:
    """Fixed tactic labels plus a data-derived, frequency-filtered technique set."""

    tactics: tuple[str, ...]
    techniques: tuple[str, ...]

    def __post_init__(self) -> None:
        valid_tactics = {item.id for item in TACTICS}
        if len(set(self.tactics)) != len(self.tactics):
            raise ValueError("tactic vocabulary contains duplicates")
        unknown = sorted(set(self.tactics) - valid_tactics)
        if unknown:
            raise ValueError(f"unknown tactic IDs: {unknown}")
        if len(set(self.techniques)) != len(self.techniques):
            raise ValueError("technique vocabulary contains duplicates")
        invalid = [item for item in self.techniques if not _TECHNIQUE_RE.fullmatch(item)]
        if invalid:
            raise ValueError(f"invalid technique IDs: {invalid}")

    @classmethod
    def canonical(cls, techniques: Iterable[str] = ()) -> LabelVocabulary:
        return cls(tuple(item.id for item in TACTICS), tuple(sorted(set(techniques))))

    @classmethod
    def build(
        cls,
        records: Iterable[Any],
        *,
        min_technique_sessions: int = 50,
        min_technique_families: int = 3,
    ) -> LabelVocabulary:
        if min_technique_sessions < 1 or min_technique_families < 1:
            raise ValueError("minimum technique thresholds must be positive")

        session_counts: Counter[str] = Counter()
        family_counts: dict[str, set[str]] = {}
        for record in records:
            item = _as_mapping(record)
            label = _as_mapping(item.get("label", {}))
            family_id = str(item.get("family_id", item.get("session_id", "")))
            techniques = {
                str(value).upper()
                for value in label.get("techniques", ())
                if _TECHNIQUE_RE.fullmatch(str(value).upper())
            }
            for technique in techniques:
                session_counts[technique] += 1
                family_counts.setdefault(technique, set()).add(family_id)

        retained = sorted(
            technique
            for technique, count in session_counts.items()
            if count >= min_technique_sessions
            and len(family_counts[technique]) >= min_technique_families
        )
        return cls.canonical(retained)

    @property
    def tactic_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.tactics)}

    @property
    def technique_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.techniques)}

    def encode(
        self,
        label: Mapping[str, Any] | Any,
        command_index: int,
        *,
        malicious: bool | None = None,
    ) -> LabelTarget:
        """Encode a prefix target at ``command_index``.

        A malicious prefix is positive starting at ``first_evidence_index``.
        Techniques/tactics only become positive once their own evidence index
        has been observed.
        """

        item = _as_mapping(label)
        is_malicious = (
            bool(malicious) if malicious is not None else bool(int(item.get("malicious", 0)))
        )
        if not is_malicious:
            return LabelTarget(0.0, (0.0,) * len(self.tactics), (0.0,) * len(self.techniques))

        first_evidence = item.get("first_evidence_index")
        if first_evidence is None:
            raise ValueError("malicious labels require first_evidence_index")
        if command_index < int(first_evidence):
            return LabelTarget(0.0, (0.0,) * len(self.tactics), (0.0,) * len(self.techniques))

        declared_techniques = {str(value).upper() for value in item.get("techniques", ())}
        evidence_raw = item.get("technique_evidence", {}) or {}
        evidence: dict[str, list[int]] = {}
        for technique, indices in _as_mapping(evidence_raw).items():
            if isinstance(indices, (int, float)):
                evidence[str(technique).upper()] = [int(indices)]
            else:
                values = [int(value) for value in indices]
                if values:
                    evidence[str(technique).upper()] = values

        active_techniques: set[str] = set()
        for technique in declared_techniques:
            technique_evidence = evidence.get(technique)
            if not technique_evidence:
                # A missing per-technique index means the session-level first
                # evidence is the only available conservative lower bound.
                active_techniques.add(technique)
            elif min(technique_evidence) <= command_index:
                active_techniques.add(technique)

        active_tactics = set(tactics_for_techniques(sorted(active_techniques)))
        declared_tactics = {str(value).upper() for value in item.get("tactics", ())}
        if not active_techniques:
            active_tactics.update(declared_tactics)
        else:
            # Keep a declared parent tactic only when no finer mapping exists.
            if not active_tactics:
                active_tactics.update(declared_tactics)

        tactic_values = [1.0 if item in active_tactics else 0.0 for item in self.tactics]
        technique_values = [1.0 if item in active_techniques else 0.0 for item in self.techniques]
        return LabelTarget(
            1.0,
            tuple(tactic_values),
            tuple(technique_values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "tactics": list(self.tactics),
            "techniques": list(self.techniques),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> LabelVocabulary:
        return cls(
            tuple(str(item) for item in value.get("tactics", ())),
            tuple(str(item) for item in value.get("techniques", ())),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> LabelVocabulary:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def compute_pos_weight(
    labels: Iterable[Iterable[float]],
    *,
    cap: float = 10.0,
) -> list[float]:
    """Compute BCE positive weights with the required hard cap."""

    rows = [[float(value) for value in row] for row in labels]
    if not rows:
        return []
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all label rows must have equal width")
    if cap <= 0:
        raise ValueError("cap must be positive")

    weights: list[float] = []
    for index in range(width):
        positives = sum(row[index] > 0 for row in rows)
        negatives = len(rows) - positives
        weight = negatives / positives if positives and negatives else 1.0
        weights.append(min(float(cap), weight))
    return weights

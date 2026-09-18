"""A/B/C consensus labeling and taxonomy-normalised evidence voting."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from cmdrisk.schema import Label, LabelQuality, SessionRecord
from cmdrisk.taxonomy import normalize_tactic, normalize_technique, tactics_for_techniques
from cmdrisk.teacher.responses import TeacherLabelResponse


class LabelModel(Protocol):
    def label(self, session: SessionRecord) -> TeacherLabelResponse: ...


class ConsensusDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    label: Label
    consensus: Literal["high", "medium"]
    accepted_for_training: bool
    tiebreak_used: bool
    vote_count: int


class ConsensusLabeler:
    """Label all sessions with A/B, escalating disagreements to C.

    A 2/2 agreement is high consensus. A C tie-break is accepted only when all
    three models agree on maliciousness and their first-evidence indices span at
    most one command. Any 2/1 outcome is medium consensus and is isolated.
    """

    def __init__(
        self,
        models: Mapping[str, LabelModel],
        *,
        confidence_floor: float = 0.75,
        first_evidence_tolerance: int = 1,
    ) -> None:
        missing = {"A", "B"} - set(models)
        if missing:
            raise ValueError(f"missing required label models: {sorted(missing)}")
        if confidence_floor < 0 or confidence_floor > 1:
            raise ValueError("confidence_floor must be between 0 and 1")
        if first_evidence_tolerance < 0:
            raise ValueError("first_evidence_tolerance must be non-negative")
        self.models = dict(models)
        self.confidence_floor = confidence_floor
        self.first_evidence_tolerance = first_evidence_tolerance

    def label(self, session: SessionRecord) -> ConsensusDecision:
        single_whoami = _is_single_whoami(session)
        votes: dict[str, TeacherLabelResponse] = {
            role: self.models[role].label(session) for role in ("A", "B")
        }
        if single_whoami:
            votes = {
                role: _as_benign_whoami_vote(vote) for role, vote in votes.items()
            }
        needs_tiebreak = False if single_whoami else self._needs_tiebreak(votes["A"], votes["B"])
        if needs_tiebreak:
            if "C" not in self.models:
                raise ValueError("A/B conflict requires a configured C model")
            votes["C"] = self.models["C"].label(session)

        high_consensus = self._is_high_consensus(votes)
        malicious = sum(vote.malicious for vote in votes.values()) > len(votes) / 2
        quality = LabelQuality.HIGH if high_consensus else LabelQuality.MEDIUM
        label = self._build_label(session, votes, malicious=malicious, quality=quality)
        return ConsensusDecision(
            session_id=session.session_id,
            label=label,
            consensus="high" if high_consensus else "medium",
            accepted_for_training=high_consensus,
            tiebreak_used=needs_tiebreak,
            vote_count=len(votes),
        )

    def _needs_tiebreak(
        self,
        first: TeacherLabelResponse,
        second: TeacherLabelResponse,
    ) -> bool:
        if first.malicious != second.malicious:
            return True
        if first.confidence < self.confidence_floor or second.confidence < self.confidence_floor:
            return True
        if first.malicious:
            if first.first_evidence_index is None or second.first_evidence_index is None:
                return True
            evidence_delta = abs(first.first_evidence_index - second.first_evidence_index)
            if evidence_delta > self.first_evidence_tolerance:
                return True
        return False

    def _is_high_consensus(
        self,
        votes: Mapping[str, TeacherLabelResponse],
    ) -> bool:
        labels = [vote.malicious for vote in votes.values()]
        if len(labels) == 2:
            return labels[0] == labels[1]
        if not all(label == labels[0] for label in labels):
            return False
        if any(vote.confidence < self.confidence_floor for vote in votes.values()):
            return False
        malicious_indices = [
            vote.first_evidence_index
            for vote in votes.values()
            if vote.malicious and vote.first_evidence_index is not None
        ]
        evidence_span = (
            max(malicious_indices) - min(malicious_indices)
            if malicious_indices
            else 0
        )
        if evidence_span > self.first_evidence_tolerance:
            return False
        return True

    def _build_label(
        self,
        session: SessionRecord,
        votes: Mapping[str, TeacherLabelResponse],
        *,
        malicious: bool,
        quality: LabelQuality,
    ) -> Label:
        first_evidence_index: int | None = None
        technique_evidence: dict[str, list[int]] = {}
        tactics: list[str] = []

        if malicious:
            evidence_indices = [
                vote.first_evidence_index
                for vote in votes.values()
                if vote.first_evidence_index is not None
            ]
            if not evidence_indices:
                raise ValueError("malicious consensus requires a first evidence index")
            first_evidence_index = min(evidence_indices)
            technique_evidence = self._technique_evidence(session, votes, first_evidence_index)
            tactics = self._tactics(votes, technique_evidence)

        return Label(
            malicious=1 if malicious else 0,
            first_evidence_index=first_evidence_index,
            technique_evidence=technique_evidence,
            tactics=tactics,
            techniques=list(technique_evidence),
            quality=quality,
            teacher_votes=[_vote_payload(role, vote) for role, vote in votes.items()],
        )

    def _technique_evidence(
        self,
        session: SessionRecord,
        votes: Mapping[str, TeacherLabelResponse],
        first_evidence_index: int,
    ) -> dict[str, list[int]]:
        by_technique: dict[str, dict[str, set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for role, vote in votes.items():
            for raw_technique, raw_indices in vote.technique_evidence.items():
                technique = normalize_technique(raw_technique)
                if technique is None:
                    continue
                valid_indices = {
                    index
                    for index in raw_indices
                    if first_evidence_index <= index < len(session.commands)
                }
                if valid_indices:
                    by_technique[technique][role].update(valid_indices)

        return {
            technique: sorted(
                set().union(*(indices for indices in role_indices.values()))
            )
            for technique, role_indices in sorted(by_technique.items())
            if len(role_indices) >= 2
        }

    @staticmethod
    def _tactics(
        votes: Mapping[str, TeacherLabelResponse],
        technique_evidence: Mapping[str, list[int]],
    ) -> list[str]:
        counts: dict[str, int] = defaultdict(int)
        for vote in votes.values():
            seen = {
                tactic
                for raw_tactic in vote.tactics
                if (tactic := normalize_tactic(raw_tactic)) is not None
            }
            for tactic in seen:
                counts[tactic] += 1
        explicit = {tactic for tactic, count in counts.items() if count >= 2}
        derived = set(tactics_for_techniques(list(technique_evidence)))
        return sorted(explicit | derived, key=_tactic_order)


def _is_single_whoami(session: SessionRecord) -> bool:
    return len(session.commands) == 1 and session.commands[0].cmd.strip() == "whoami"


def _as_benign_whoami_vote(vote: TeacherLabelResponse) -> TeacherLabelResponse:
    if not vote.malicious:
        return vote
    return TeacherLabelResponse(
        malicious=False,
        first_evidence_index=None,
        technique_evidence={},
        tactics=[],
        confidence=vote.confidence,
    )


def _vote_payload(role: str, vote: TeacherLabelResponse) -> dict[str, object]:
    return {
        "role": role,
        "malicious": vote.malicious,
        "first_evidence_index": vote.first_evidence_index,
        "technique_evidence": vote.technique_evidence,
        "tactics": vote.tactics,
        "confidence": vote.confidence,
    }


def _tactic_order(tactic_id: str) -> int:
    from cmdrisk.taxonomy import TACTICS

    order = {tactic.id: index for index, tactic in enumerate(TACTICS)}
    return order.get(tactic_id, len(order))

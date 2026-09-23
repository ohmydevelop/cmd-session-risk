"""Strict prefix expansion with evidence-aware labels."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from cmdrisk.schema import Label, PrefixRecord, SessionRecord


def build_prefixes(record: SessionRecord) -> Iterator[PrefixRecord]:
    """Yield every prefix from command 1 through the complete session."""

    if record.split is None:
        raise ValueError("prefix generation requires a session split")
    full_length = len(record.commands)
    for length in range(1, full_length + 1):
        last_index = length - 1
        label = _prefix_label(record, last_index)
        yield PrefixRecord(
            session_id=record.session_id,
            command_count=full_length,
            split=record.split,
            source=record.source,
            family_id=record.family_id,
            commands=record.commands[:length],
            label=label,
        )


def _prefix_label(record: SessionRecord, last_index: int) -> Label:
    source = record.label
    if source.malicious == 0 or source.first_evidence_index is None:
        return Label(
            malicious=0,
            quality=source.quality,
            teacher_votes=source.teacher_votes,
        )
    if last_index < source.first_evidence_index:
        return Label(
            malicious=0,
            quality=source.quality,
            teacher_votes=source.teacher_votes,
        )

    evidence = {
        technique: sorted(
            index for index in indices if source.first_evidence_index <= index <= last_index
        )
        for technique, indices in source.technique_evidence.items()
    }
    evidence = {technique: indices for technique, indices in evidence.items() if indices}
    techniques = list(evidence)
    return Label(
        malicious=1,
        first_evidence_index=source.first_evidence_index,
        technique_evidence=evidence,
        tactics=source.tactics,
        techniques=techniques,
        quality=source.quality,
        teacher_votes=source.teacher_votes,
    )


def build_prefix_records(records: Iterable[SessionRecord]) -> list[PrefixRecord]:
    return [prefix for record in records for prefix in build_prefixes(record)]

"""Temporal holdout detection for out-of-distribution evaluation slices.

The Shell Attack Evolution corpus ships a ``period`` label per session.  The
2024 slice is deliberately reserved as a temporal out-of-distribution (OOD)
evaluation period and must never enter ``train`` or ``dev``.  Detection is
kept explicit: only well-known provenance keys are inspected so that a stray
``2024`` inside a command string can never move a benign session into OOD.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cmdrisk.schema import SessionRecord

HOLDOUT_PERIOD = "2024"
OOD_SLICE_NAME = "ood_2024"

# Only these provenance keys are trusted to carry a temporal period.
_PERIOD_KEYS = (
    "period",
    "year",
    "holdout_period",
    "temporal_holdout",
    "temporal_period",
)
_EVALUATION_KEYS = ("evaluation_slice", "dataset_slice")


def _as_period(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    return text or None


def _iter_sources(meta: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the record meta plus any nested ``meta`` mappings."""

    sources: list[Mapping[str, Any]] = [meta]
    nested = meta.get("meta")
    if isinstance(nested, Mapping):
        sources.append(nested)
    return sources


def record_period(record: SessionRecord) -> str | None:
    """Return the declared temporal period of ``record``, if any."""

    for source in _iter_sources(record.meta):
        for key in _PERIOD_KEYS:
            period = _as_period(source.get(key))
            if period is not None:
                return period
    return None


def is_temporal_holdout(
    record: SessionRecord,
    *,
    period: str = HOLDOUT_PERIOD,
) -> bool:
    """Return whether ``record`` belongs to the temporal holdout period."""

    target = str(period)
    for source in _iter_sources(record.meta):
        for key in _EVALUATION_KEYS:
            declared = _as_period(source.get(key))
            if declared is not None and declared.casefold() == OOD_SLICE_NAME:
                return True
        declared_period = None
        for key in _PERIOD_KEYS:
            declared_period = _as_period(source.get(key))
            if declared_period is not None:
                break
        if declared_period is None:
            continue
        if declared_period == target:
            return True
        # ``2024-01`` style values and ``2024`` both mark the holdout period.
        if declared_period.split("-", 1)[0].strip() == target:
            return True
    return False


def annotate_holdout_slice(records: Any) -> list[SessionRecord]:
    """Return ``records`` with holdout members tagged for the OOD slice."""

    annotated: list[SessionRecord] = []
    for record in records:
        if not is_temporal_holdout(record):
            annotated.append(record)
            continue
        meta = dict(record.meta)
        meta["evaluation_slice"] = OOD_SLICE_NAME
        annotated.append(record.model_copy(update={"meta": meta}))
    return annotated

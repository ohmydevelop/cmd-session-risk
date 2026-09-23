"""Session loading and prefix expansion for the training pipeline."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cmdrisk.training.labels import LabelVocabulary


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    raise TypeError(f"expected a mapping-like record, got {type(value)!r}")


def _evaluation_slice(item: Mapping[str, Any], source_kind: str) -> str:
    meta = item.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    explicit = (
        item.get("evaluation_slice")
        or item.get("dataset_slice")
        or meta.get("evaluation_slice")
        or meta.get("dataset_slice")
    )
    if explicit:
        return str(explicit)
    timestamp = str(item.get("timestamp", ""))
    period = str(meta.get("period", "") or meta.get("year", "") or item.get("year", ""))
    if timestamp.startswith("2024") or period.split("-", 1)[0].strip() == "2024":
        return "ood_2024"
    if source_kind in {"llm_variant", "hard_negative"}:
        return "synthetic_stress"
    if source_kind in {"real_malicious", "real_benign"}:
        return "real_only"
    if source_kind in {"script_benign", "real_history_pool"}:
        return "real_plus_semi_real"
    return "unspecified"


@dataclass(frozen=True)
class PrefixExample:
    session_id: str
    family_id: str
    source_kind: str
    split: str
    command_count: int
    prefix_index: int
    commands: tuple[str, ...]
    dts: tuple[float | None, ...]
    malicious: float
    tactics: tuple[float, ...]
    techniques: tuple[float, ...]
    evaluation_slice: str = "unspecified"

    @property
    def prefix_length(self) -> int:
        return self.prefix_index + 1


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL/JSON/Parquet session records without Torch."""

    source = Path(path)
    if source.is_dir():
        records: list[dict[str, Any]] = []
        for child in sorted(source.iterdir()):
            if child.suffix.lower() in {".jsonl", ".json", ".parquet"}:
                records.extend(load_records(child))
        return records
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [dict(item) for item in value]
        if isinstance(value, Mapping) and "records" in value:
            return [dict(item) for item in value["records"]]
        return [dict(value)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on optional train stack
            raise RuntimeError("pandas is required to read Parquet session data") from exc
        return pd.read_parquet(source).to_dict(orient="records")
    raise ValueError(f"unsupported record format: {source}")


def select_prefix_indices(
    command_count: int,
    *,
    max_prefixes: int = 32,
    first_evidence_index: int | None = None,
    evidence_indices: Iterable[int] = (),
) -> tuple[int, ...]:
    """Select deterministic prefixes, retaining evidence transitions and ends."""

    if command_count < 1:
        raise ValueError("command_count must be positive")
    if max_prefixes < 1:
        raise ValueError("max_prefixes must be positive")
    if command_count <= max_prefixes:
        return tuple(range(command_count))

    mandatory = {0, command_count - 1}
    if first_evidence_index is not None:
        mandatory.update(
            index
            for index in range(first_evidence_index - 2, first_evidence_index + 2)
            if 0 <= index < command_count
        )
    mandatory.update(
        index for index in evidence_indices if 0 <= int(index) < command_count
    )

    if len(mandatory) > max_prefixes:
        # Preserve chronology rather than allowing a label with an unusually
        # large technique list to exceed the configured sample budget.
        ordered = sorted(mandatory)
        keep = {ordered[0], ordered[-1]}
        if first_evidence_index is not None and 0 <= first_evidence_index < command_count:
            keep.add(first_evidence_index)
        remaining = [index for index in ordered if index not in keep]
        while len(keep) < max_prefixes and remaining:
            target = len(keep)
            population = len(remaining)
            position = round(target * (population - 1) / max(max_prefixes - 1, 1))
            keep.add(remaining.pop(min(max(position, 0), len(remaining) - 1)))
        return tuple(sorted(keep))

    selected = set(mandatory)
    # Fill from a uniformly spaced span until the budget is full.
    for step in range(max_prefixes):
        if len(selected) >= max_prefixes:
            break
        selected.add(round(step * (command_count - 1) / max(max_prefixes - 1, 1)))
    if len(selected) < max_prefixes:
        for index in range(command_count):
            selected.add(index)
            if len(selected) >= max_prefixes:
                break
    return tuple(sorted(selected))


def build_prefix_examples(
    record: Any,
    vocabulary: LabelVocabulary,
    *,
    max_prefixes: int = 32,
) -> list[PrefixExample]:
    item = _as_mapping(record)
    commands_raw: Sequence[Any] = item.get("commands", ())
    if not commands_raw:
        raise ValueError("session record must contain at least one command")
    commands: list[str] = []
    dts: list[float | None] = []
    for command in commands_raw:
        command_item = _as_mapping(command)
        commands.append(str(command_item["cmd"]))
        dt = command_item.get("dt")
        dts.append(None if dt is None else float(dt))

    label = _as_mapping(item.get("label", {}))
    first_evidence = label.get("first_evidence_index")
    technique_evidence = _as_mapping(label.get("technique_evidence", {}) or {})
    evidence_indices = [
        int(index)
        for values in technique_evidence.values()
        for index in ([values] if isinstance(values, (int, float)) else values)
    ]
    selected = select_prefix_indices(
        len(commands),
        max_prefixes=max_prefixes,
        first_evidence_index=None if first_evidence is None else int(first_evidence),
        evidence_indices=evidence_indices,
    )

    source = _as_mapping(item.get("source", {}))
    source_kind = str(source.get("kind", "unknown"))
    evaluation_slice = _evaluation_slice(item, source_kind)
    examples: list[PrefixExample] = []
    for prefix_index in selected:
        target = vocabulary.encode(label, prefix_index)
        examples.append(
            PrefixExample(
                session_id=str(item["session_id"]),
                family_id=str(item["family_id"]),
                source_kind=source_kind,
                split=str(item.get("split") or "train"),
                command_count=len(commands),
                prefix_index=prefix_index,
                commands=tuple(commands[: prefix_index + 1]),
                dts=tuple(dts[: prefix_index + 1]),
                malicious=target.malicious,
                tactics=target.tactics,
                techniques=target.techniques,
                evaluation_slice=evaluation_slice,
            )
        )
    return examples


def expand_records(
    records: Iterable[Any],
    vocabulary: LabelVocabulary,
    *,
    split: str | None = None,
    max_prefixes: int = 32,
) -> list[PrefixExample]:
    examples: list[PrefixExample] = []
    for record in records:
        item = _as_mapping(record)
        record_split = str(item.get("split") or "train")
        if split is not None and record_split != split:
            continue
        examples.extend(
            build_prefix_examples(record, vocabulary, max_prefixes=max_prefixes)
        )
    return examples

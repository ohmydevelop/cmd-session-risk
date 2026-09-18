"""JSONL I/O helpers used by data preparation scripts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from cmdrisk.schema import PrefixRecord, SessionRecord

ModelT = TypeVar("ModelT", bound=BaseModel)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object")
            yield value


def read_jsonl(
    path: str | Path, model: type[ModelT] = SessionRecord
) -> list[ModelT]:
    return [model.model_validate(value) for value in iter_jsonl(path)]


def read_sessions(path: str | Path) -> list[SessionRecord]:
    return read_jsonl(path, SessionRecord)


def read_prefixes(path: str | Path) -> list[PrefixRecord]:
    return read_jsonl(path, PrefixRecord)


def write_jsonl(
    records: Iterable[BaseModel | dict[str, Any]], path: str | Path, *, resume: bool = False
) -> Path:
    destination = Path(path)
    if resume and destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            value = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
            handle.write(json.dumps(value, separators=(",", ":"), ensure_ascii=True))
            handle.write("\n")
    temporary.replace(destination)
    return destination

"""Source manifests and reproducible file hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .yaml_utils import load_yaml_file

REQUIRED_SOURCE_FIELDS = {
    "id",
    "adapter",
    "url",
    "license",
    "sha256",
    "password",
    "redistribution",
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_data_sources(path: str | Path) -> list[dict[str, Any]]:
    document = load_yaml_file(path)
    if not isinstance(document, dict) or not isinstance(document.get("sources"), list):
        raise ValueError("data source manifest must contain a 'sources' list")
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(document["sources"]):
        if not isinstance(source, dict):
            raise ValueError(f"source #{index} must be a mapping")
        missing = REQUIRED_SOURCE_FIELDS.difference(source)
        if missing:
            raise ValueError(f"source #{index} missing fields: {sorted(missing)}")
        source_id = str(source["id"]).strip()
        if not source_id or source_id in seen:
            raise ValueError(f"source #{index} has an empty or duplicate id")
        seen.add(source_id)
        sources.append(source)
    return sources


def write_sha256_manifest(
    paths: Iterable[str | Path], output: str | Path, *, root: str | Path | None = None
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(root) if root is not None else output_path.parent
    lines: list[str] = []
    for value in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        try:
            display = value.resolve().relative_to(base.resolve())
        except ValueError:
            display = value
        lines.append(f"{sha256_file(value)}  {display.as_posix()}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def load_sha256_manifest(path: str | Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            digest, name = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid SHA256 manifest line {line_number}") from exc
        manifest[name] = digest
    return manifest


def write_json_atomic(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination

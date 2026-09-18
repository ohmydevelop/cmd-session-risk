"""Small YAML compatibility layer.

PyYAML is used when available.  The project's own configuration file is also
valid JSON, so a deployment without the optional parser can still load its
manifest.  Atomic Red Team inputs require the real YAML parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml_text(text: str, *, source: str = "<string>", require_yaml: bool = False) -> Any:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        if require_yaml:
            raise RuntimeError("PyYAML is required to parse Atomic Red Team YAML") from None
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is not installed and the document is not JSON-compatible YAML"
            ) from exc

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {source}: {exc}") from exc


def load_yaml_file(path: str | Path, *, require_yaml: bool = False) -> Any:
    source = Path(path)
    return load_yaml_text(
        source.read_text(encoding="utf-8"),
        source=str(source),
        require_yaml=require_yaml,
    )

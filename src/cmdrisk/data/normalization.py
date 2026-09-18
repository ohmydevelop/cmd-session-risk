"""Command cleaning, raw-command preservation, and skeleton generation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable

from cmdrisk.schema import CommandEvent, SessionRecord

ANSI_ESCAPE_RE = re.compile(
    r"(?:\x1B[@-_][0-?]*[ -/]*[@-~])|(?:\x1B\][^\x07]*(?:\x07|\x1B\\))"
)
PROMPT_RE = re.compile(
    r"^\s*(?:(?:\([^)]{0,64}\)\s*)?"
    r"(?:\[[^\]]{0,64}\]\s*)?"
    r"(?:[A-Za-z0-9_.-]+@[A-Za-z0-9_.-]+(?::[^#$>]{0,128})?|"
    r"[A-Za-z0-9_.-]+:[^#$>]{1,128})[#$>]\s+)"
)
URL_RE = re.compile(r"(?i)\b(?:https?|ftp)://[^\s'\"<>]+")
IPV4_RE = re.compile(r"(?<![A-Za-z0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![A-Za-z0-9.])")
IPV6_RE = re.compile(r"(?<![A-Za-z0-9:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![A-Za-z0-9:])")
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
HEX_RE = re.compile(r"(?<![A-Za-z0-9])[0-9a-fA-F]{16,}(?![A-Za-z0-9])")
# Require at least one base64-only marker (+, /, or padding) so that plain
# hexadecimal secrets are classified by HEX_RE rather than as base64.
BASE64_RE = re.compile(
    r"(?<![A-Za-z0-9+/])"
    r"(?=[A-Za-z0-9+/=]{24,}(?![A-Za-z0-9+/=]))(?=[A-Za-z0-9+/=]*[+/=])"
    r"[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])"
)
PATH_RE = re.compile(
    r"(?<![A-Za-z0-9._-])(?:~?/|\.\.?/)[A-Za-z0-9._~%+@=-]+(?:/[A-Za-z0-9._~%+@=-]+)*"
)


def strip_ansi(value: str) -> str:
    return ANSI_ESCAPE_RE.sub("", value)


def clean_command(value: str) -> str:
    """Remove terminal control bytes and a leading shell prompt from one command."""

    cleaned = strip_ansi(value).replace("\r\n", "\n").replace("\r", "\n")
    prompt_stripped = PROMPT_RE.sub("", cleaned, count=1)
    return " ".join(prompt_stripped.strip().split())


def skeletonize_command(value: str) -> str:
    """Mask high-cardinality strings while retaining executable/argument structure."""

    skeleton = clean_command(value)
    skeleton = URL_RE.sub("<URL>", skeleton)
    skeleton = UUID_RE.sub("<UUID>", skeleton)
    skeleton = IPV4_RE.sub("<IP>", skeleton)
    skeleton = IPV6_RE.sub("<IP>", skeleton)
    skeleton = HEX_RE.sub("<HEX>", skeleton)
    skeleton = BASE64_RE.sub("<B64>", skeleton)
    skeleton = PATH_RE.sub("<PATH>", skeleton)
    return " ".join(skeleton.split())


def session_skeleton(record: SessionRecord) -> tuple[str, ...]:
    return tuple(skeletonize_command(event.cmd) for event in record.commands)


def stable_session_id(
    source_name: str, commands: Iterable[str | CommandEvent], *, external_id: str | None = None
) -> str:
    payload: list[dict[str, object]] = []
    for event in commands:
        if isinstance(event, CommandEvent):
            payload.append({"cmd": event.cmd, "dt": event.dt})
        else:
            payload.append({"cmd": event})
    canonical = json.dumps(
        {"source": source_name, "external_id": external_id, "commands": payload},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_record(record: SessionRecord) -> SessionRecord:
    """Return a copy with cleaned commands and an immutable raw representation."""

    raw_commands = list(record.meta.get("raw_commands", [event.cmd for event in record.commands]))
    normalized_commands = [
        CommandEvent(cmd=clean_command(event.cmd), dt=event.dt) for event in record.commands
    ]
    meta = dict(record.meta)
    meta["raw_commands"] = raw_commands
    meta["skeletons"] = [skeletonize_command(event.cmd) for event in record.commands]
    return record.model_copy(update={"commands": normalized_commands, "meta": meta})

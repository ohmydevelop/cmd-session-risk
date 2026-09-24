"""Source-specific adapters for public command-session data."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import re
import zipfile
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from cmdrisk.schema import (
    CommandEvent,
    Label,
    LabelQuality,
    SessionRecord,
    SourceInfo,
    SourceKind,
)
from cmdrisk.taxonomy import normalize_technique, tactics_for_techniques

from .normalization import clean_command, stable_session_id
from .streaming import (
    iter_stream_values,
    iter_zip_member_texts,
    open_json_text,
)
from .yaml_utils import load_yaml_file

PLACEHOLDER_RE = re.compile(r"#\{([A-Za-z0-9_.-]+)\}")
TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)


class Adapter(Protocol):
    def iter_records(self, path: str | Path, **kwargs: Any) -> Iterator[SessionRecord]: ...


class AdapterError(RuntimeError):
    """Raised when an input does not match the expected source format."""


def _as_commands(values: Iterable[Any]) -> list[str]:
    commands: list[str] = []
    for value in values:
        if isinstance(value, str):
            command = value
        elif isinstance(value, dict):
            command = value.get("cmd") or value.get("command") or value.get("Command") or ""
        else:
            command = str(value)
        command = clean_command(command)
        if command:
            commands.append(command)
    return commands


def _label(
    *,
    malicious: bool,
    commands: list[str],
    techniques: list[str] | None = None,
    quality: LabelQuality = LabelQuality.HIGH,
) -> Label:
    techniques = [item for item in dict.fromkeys(techniques or []) if item]
    if not malicious:
        return Label(malicious=0, quality=quality)
    evidence = {technique: [0] for technique in techniques}
    return Label(
        malicious=1,
        first_evidence_index=0,
        technique_evidence=evidence,
        tactics=tactics_for_techniques(techniques),
        techniques=techniques,
        quality=quality,
    )


def _make_record(
    *,
    source_name: str,
    source_kind: SourceKind,
    license_name: str,
    external_id: str,
    commands: list[str],
    malicious: bool,
    techniques: list[str] | None = None,
    quality: LabelQuality = LabelQuality.HIGH,
    meta: dict[str, Any] | None = None,
) -> SessionRecord:
    session_id = stable_session_id(source_name, commands, external_id=external_id)
    meta = dict(meta or {})
    meta.setdefault("raw_commands", list(commands))
    return SessionRecord(
        session_id=session_id,
        source=SourceInfo(kind=source_kind, name=source_name, license=license_name),
        family_id=f"pending:{session_id}",
        commands=[CommandEvent(cmd=command) for command in commands],
        meta=meta,
        label=_label(
            malicious=malicious, commands=commands, techniques=techniques, quality=quality
        ),
    )


def _read_json_document(path: Path, password: str | None = None) -> Any:
    if path.suffix.casefold() != ".zip":
        return json.loads(path.read_text(encoding="utf-8"))
    with zipfile.ZipFile(path) as archive:
        json_names = [name for name in archive.namelist() if name.casefold().endswith(".json")]
        if not json_names:
            raise AdapterError(f"ZIP archive has no JSON document: {path}")
        name = json_names[0]
        password_bytes = password.encode("utf-8") if password else None
        try:
            content = archive.open(name, pwd=password_bytes).read()
        except RuntimeError:
            if password_bytes is None:
                raise AdapterError(f"encrypted ZIP requires a password: {path}") from None
            try:
                content = archive.open(name, pwd=password_bytes).read()
            except RuntimeError as exc:
                raise AdapterError(f"could not decrypt {name} in {path}") from exc
        return json.loads(content.decode("utf-8-sig"))


def _candidate_mappings(document: Any) -> Iterator[dict[str, Any]]:
    if isinstance(document, list):
        for value in document:
            if isinstance(value, dict):
                yield value
    elif isinstance(document, dict):
        for key in ("value", "values", "data", "records", "sessions"):
            value = document.get(key)
            if isinstance(value, list):
                yield from (item for item in value if isinstance(item, dict))
                return
        yield document


class MicrosoftIoTAdapter:
    """Read Microsoft IoT honeypot ZIP/JSON command sequences."""

    source_name = "microsoft_iot"
    license_name = "Microsoft repository sample data; redistribution unconfirmed"

    def iter_records(
        self, path: str | Path, *, password: str | None = None, **_: Any
    ) -> Iterator[SessionRecord]:
        source = Path(path)
        paths = (
            sorted(
                (
                    *source.rglob("*.json"),
                    *source.rglob("*.jsonl"),
                    *source.rglob("*.zip"),
                )
            )
            if source.is_dir()
            else [source]
        )
        for document_path in paths:
            # Streamed: the encrypted archive expands to ~878 MB and must
            # never be materialised (v3 loaded it whole, which capped the
            # corpus at 5,000 sessions per source).
            with open_json_text(document_path, password=password) as text:
                for value in iter_stream_values(text):
                    for item in _candidate_mappings(value):
                        record = self._record_from_item(item)
                        if record is not None:
                            yield record

    def _record_from_item(self, item: dict[str, Any]) -> SessionRecord | None:
        raw_commands = item.get("Commands") or item.get("commands") or []
        if isinstance(raw_commands, str):
            raw_commands = raw_commands.splitlines()
        commands = _as_commands(raw_commands)
        if not commands:
            return None
        external_id = str(item.get("ID") or item.get("id") or stable_session_id("raw", commands))
        meta = {
            key: item[key]
            for key in ("Protocol", "TimesSeen", "FirstSeen", "LastSeen")
            if key in item
        }
        meta["raw_commands"] = list(raw_commands)
        return _make_record(
            source_name=self.source_name,
            source_kind=SourceKind.REAL_MALICIOUS,
            license_name=self.license_name,
            external_id=external_id,
            commands=commands,
            malicious=True,
            meta=meta,
        )


class ShellAttackEvolutionAdapter:
    """Read Shell Attack Evolution JSON/JSONL records and optional patterns."""

    source_name = "shell_attack_evolution"
    license_name = "CC-BY-4.0"

    def iter_records(self, path: str | Path, **_: Any) -> Iterator[SessionRecord]:
        source = Path(path)
        if source.is_dir():
            paths = sorted(
                item
                for item in source.rglob("*")
                if item.is_file() and item.suffix.casefold() in {".json", ".jsonl", ".zip"}
            )
            for document_path in paths:
                yield from self.iter_records(document_path)
            return
        if source.suffix.casefold() == ".zip" or zipfile.is_zipfile(source):
            for name, text in iter_zip_member_texts(source):
                yield from self._iter_stream(iter_stream_values(text), external_prefix=name)
            return
        with source.open("r", encoding="utf-8-sig") as text:
            yield from self._iter_stream(iter_stream_values(text), external_prefix=source.name)

    def _iter_content(self, content: str, *, external_prefix: str) -> Iterator[SessionRecord]:
        """Compatibility wrapper that streams from an in-memory document."""

        yield from self._iter_stream(
            iter_stream_values(io.StringIO(content)), external_prefix=external_prefix
        )

    def _iter_stream(
        self, values: Iterable[Any], *, external_prefix: str
    ) -> Iterator[SessionRecord]:
        stream = iter(values)
        try:
            first = next(stream)
        except StopIteration:
            return
        chained = itertools.chain([first], stream)
        if _looks_like_cowrie_event(first):
            # Cowrie raw event logs: one JSON object per event; sessions are
            # reconstructed from the connection key plus an idle gap.
            yield from self._iter_cowrie_events(chained, external_prefix=external_prefix)
            return

        for index, item in enumerate(_expand_sessions(chained)):
            if not isinstance(item, dict):
                continue
            raw_commands = item.get("commands") or item.get("command_sequence") or []
            commands = _as_commands(raw_commands)
            if not commands:
                continue
            techniques = _extract_techniques(item.get("attack_techniques"))
            external_id = str(
                item.get("session_id") or item.get("id") or f"{external_prefix}:{index}"
            )
            meta = {
                key: item[key]
                for key in ("period", "src_ip", "command_count", "session_pattern")
                if key in item
            }
            meta["raw_commands"] = (
                list(raw_commands)
                if not isinstance(raw_commands, str)
                else _as_commands([raw_commands])
            )
            yield _make_record(
                source_name=self.source_name,
                source_kind=SourceKind.REAL_MALICIOUS,
                license_name=self.license_name,
                external_id=external_id,
                commands=commands,
                malicious=True,
                techniques=techniques,
                meta=meta,
            )

    def _iter_cowrie_events(
        self, values: Iterable[Any], *, external_prefix: str
    ) -> Iterator[SessionRecord]:
        current_key: tuple[Any, ...] | None = None
        current: list[dict[str, Any]] = []
        previous: datetime | None = None
        closed: set[tuple[Any, ...]] = set()
        index = 0

        def build(events: list[dict[str, Any]]) -> SessionRecord | None:
            commands = []
            for event in events:
                command = _command_from_cowrie_event(event)
                if command:
                    commands.append(command)
            if not commands:
                return None
            source_ip = events[0].get("src_host")
            external_id = f"{external_prefix}:{source_ip}:{index}"
            return _make_record(
                source_name=self.source_name,
                source_kind=SourceKind.REAL_MALICIOUS,
                license_name=self.license_name,
                external_id=external_id,
                commands=commands,
                malicious=True,
                meta={
                    "period": "2024" if "2024" in external_prefix else "unknown",
                    "src_ip": source_ip,
                    "raw_commands": commands,
                    "cowrie_event_count": len(events),
                },
            )

        for item in values:
            if not isinstance(item, dict):
                continue
            key = _cowrie_session_key(item)
            if item.get("info") == "session.closed":
                closed.add(key)
                continue
            if item.get("info") != "command.input":
                continue
            timestamp = _parse_timestamp(str(item.get("timestamp", "")).replace(" ", "T"))
            gap = (
                previous is not None
                and timestamp is not None
                and (timestamp - previous).total_seconds() > 3600
            )
            if current and (key != current_key or gap or key in closed):
                record = build(current)
                if record is not None:
                    yield record
                index += 1
                current = []
            closed.discard(key)
            current_key = key
            current.append(item)
            previous = timestamp
        if current:
            record = build(current)
            if record is not None:
                yield record


def _looks_like_cowrie_event(value: Any) -> bool:
    """Return whether a streamed object is a raw Cowrie event record."""

    if not isinstance(value, dict):
        return False
    if "commands" in value or "command_sequence" in value:
        return False
    return "info" in value or "tshark" in value or "session.closed" in str(value.get("info", ""))


def _expand_sessions(values: Iterable[Any]) -> Iterator[dict[str, Any]]:
    """Lazily unwrap common session containers without materialising them."""

    wrapper_keys = ("sessions", "records", "values", "value", "data")
    for value in values:
        if isinstance(value, dict):
            expanded = False
            for key in wrapper_keys:
                nested = value.get(key)
                if isinstance(nested, list):
                    for item in nested:
                        if isinstance(item, dict):
                            yield item
                    expanded = True
                    break
            if not expanded:
                yield value
        elif isinstance(value, list):
            yield from _expand_sessions(value)


def _cowrie_session_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Return a connection/session boundary without merging reused client ports."""

    explicit: Any = None
    for field in ("session", "session_id", "sessionid", "uuid", "login"):
        candidate = item.get(field)
        if candidate not in (None, "", {}, []):
            explicit = str(candidate)
            break
    connection = (
        item.get("src_host"),
        item.get("src_port"),
        item.get("dst_host"),
        item.get("dst_port"),
        item.get("protocol"),
        item.get("pot_id"),
    )
    return (*connection, explicit)


def _extract_techniques(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, dict)):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    techniques: list[str] = []
    for item in values:
        if isinstance(item, dict):
            candidate = item.get("technique_id") or item.get("id") or item.get("technique") or ""
        else:
            candidate = str(item)
        candidate = candidate.strip().upper()
        match = re.search(r"\bT\d{4}(?:\.\d{3})?\b", candidate)
        if match:
            techniques.append(match.group(0))
        else:
            normalized = normalize_technique(candidate)
            if normalized:
                techniques.append(normalized)
    return list(dict.fromkeys(techniques))


class ZenodoAdapter:
    """Read KYPO/Zenodo user-action JSONL and reconstruct terminal sessions."""

    source_name = "kypo"
    license_name = "CC-BY-4.0"

    def __init__(self, *, gap_seconds: float = 900.0) -> None:
        self.gap_seconds = gap_seconds

    def iter_records(self, path: str | Path, **_: Any) -> Iterator[SessionRecord]:
        source = Path(path)
        if source.is_dir():
            paths = sorted(item for item in source.rglob("*.json") if item.is_file())
            for document_path in paths:
                yield from self.iter_records(document_path)
            return
        if source.suffix.casefold() == ".zip" or zipfile.is_zipfile(source):
            for name, text in iter_zip_member_texts(source):
                yield from self._iter_action_values(iter_stream_values(text), source_name=name)
            return
        with source.open("r", encoding="utf-8-sig") as text:
            yield from self._iter_action_values(iter_stream_values(text), source_name=source.name)

    def _iter_actions(self, content: str, *, source_name: str) -> Iterator[SessionRecord]:
        """Compatibility wrapper for callers that already hold a string."""

        yield from self._iter_action_values(
            iter_stream_values(io.StringIO(content)), source_name=source_name
        )

    def _iter_action_values(
        self, values: Iterable[Any], *, source_name: str
    ) -> Iterator[SessionRecord]:
        """Reconstruct KYPO sessions from any streamed JSON/JSONL shape."""

        current: list[dict[str, Any]] = []
        previous_timestamp: datetime | None = None
        previous_context: tuple[Any, ...] | None = None
        chunk = 0

        def emit(actions: list[dict[str, Any]], index: int) -> Iterator[SessionRecord]:
            commands: list[str] = []
            dts: list[float | None] = []
            local_previous: datetime | None = None
            for action in actions:
                command = clean_command(str(action.get("cmd") or ""))
                if not command:
                    continue
                timestamp = _parse_timestamp(action.get("timestamp_str"))
                dt = None
                if local_previous is not None and timestamp is not None:
                    dt = max(0.0, (timestamp - local_previous).total_seconds())
                commands.append(command)
                dts.append(dt)
                if timestamp is not None:
                    local_previous = timestamp
            if not commands:
                return
            sandbox = actions[0].get("sandbox_id", "unknown")
            external_id = f"{source_name}:{sandbox}:{index}"
            record = _make_record(
                source_name=self.source_name,
                source_kind=SourceKind.REAL_BENIGN,
                license_name=self.license_name,
                external_id=external_id,
                commands=commands,
                malicious=False,
                quality=LabelQuality.MEDIUM,
                meta={
                    "sandbox_id": sandbox,
                    "requires_teacher_review": True,
                    "raw_events": actions,
                    "raw_commands": [str(action.get("cmd") or "") for action in actions],
                },
            )
            record.commands = [
                CommandEvent(cmd=command, dt=dt) for command, dt in zip(commands, dts, strict=True)
            ]
            yield record

        for action in values:
            if not isinstance(action, dict) or not action.get("cmd"):
                continue
            timestamp = _parse_timestamp(action.get("timestamp_str"))
            context = (
                action.get("sandbox_id"),
                action.get("hostname"),
                action.get("username"),
                action.get("wd"),
            )
            gap = (
                previous_timestamp is not None
                and timestamp is not None
                and (timestamp - previous_timestamp).total_seconds() > self.gap_seconds
            )
            if current and (gap or context != previous_context):
                yield from emit(current, chunk)
                chunk += 1
                current = []
            current.append(action)
            previous_timestamp = timestamp
            previous_context = context
        if current:
            yield from emit(current, chunk)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _command_from_cowrie_event(event: dict[str, Any]) -> str | None:
    for key in ("request", "input", "command"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return clean_command(value)
    tshark = str(event.get("tshark") or "")
    patterns = (
        r"command\s+found\s*\[\s*b?['\"](.*?)['\"]\s*\]",
        r"command\s+['\"](.*?)['\"]",
    )
    for pattern in patterns:
        match = re.search(pattern, tshark, flags=re.IGNORECASE)
        if match:
            return clean_command(match.group(1))
    return None


class AtomicRedTeamAdapter:
    """Expand Linux Atomic Red Team YAML test templates into sessions."""

    source_name = "atomic_red_team"
    license_name = "MIT"
    _linux_platforms = {"linux", "unix", "bash", "sh"}

    def iter_records(self, path: str | Path, **_: Any) -> Iterator[SessionRecord]:
        source = Path(path)
        paths = [source]
        if source.is_dir():
            paths = sorted((*source.rglob("*.yaml"), *source.rglob("*.yml")))
        for yaml_path in paths:
            document = load_yaml_file(yaml_path, require_yaml=True)
            if not isinstance(document, dict):
                continue
            yield from self._iter_document(document, yaml_path)

    def _iter_document(self, document: dict[str, Any], path: Path) -> Iterator[SessionRecord]:
        attack_technique = str(
            document.get("attack_technique") or document.get("external_id") or ""
        ).upper()
        techniques = [attack_technique] if TECHNIQUE_RE.fullmatch(attack_technique) else []
        for index, test in enumerate(document.get("atomic_tests") or []):
            if not isinstance(test, dict):
                continue
            platforms = test.get("supported_platforms") or document.get("supported_platforms") or []
            if platforms and not {str(item).casefold() for item in platforms}.intersection(
                self._linux_platforms
            ):
                continue
            executor = test.get("executor") or {}
            if isinstance(executor, list):
                executors = executor
            else:
                executors = [executor]
            arguments = test.get("input_arguments") or {}
            rendered: list[str] = []
            for executor_item in executors:
                if not isinstance(executor_item, dict):
                    continue
                command = executor_item.get("command") or executor_item.get("cmd")
                if not command:
                    continue
                rendered.extend(
                    _render_atomic_command(str(command), arguments, test.get("name", path.stem))
                )
            commands = _as_commands(rendered)
            if not commands:
                continue
            test_name = str(test.get("name") or f"atomic-{index}")
            external_id = f"{path}:{index}:{test_name}"
            yield _make_record(
                source_name=self.source_name,
                source_kind=SourceKind.REAL_MALICIOUS,
                license_name=self.license_name,
                external_id=external_id,
                commands=commands,
                malicious=True,
                techniques=techniques,
                meta={
                    "atomic_test_name": test_name,
                    "executor": (executor[0] if executors else {}).get("name")
                    if isinstance(executor, list)
                    else executor.get("name"),
                    "atomic_source": str(path),
                    "raw_commands": list(rendered),
                },
            )


def _render_atomic_command(command: str, arguments: Any, fallback: str) -> list[str]:
    values: dict[str, str] = {}
    if isinstance(arguments, dict):
        for name, definition in arguments.items():
            if isinstance(definition, dict):
                value = definition.get("default")
            else:
                value = definition
            values[str(name)] = fallback if value in (None, "") else str(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        return values.get(name, f"ATOMIC_{name.upper()}")

    rendered = PLACEHOLDER_RE.sub(replace, command)
    return _split_shell_lines(rendered)


def _split_shell_lines(value: str) -> list[str]:
    lines: list[str] = []
    pending = ""
    for raw_line in value.splitlines():
        line = raw_line.rstrip()
        continuation = line.endswith("\\")
        line = line[:-1] if continuation else line
        pending = f"{pending} {line.strip()}".strip()
        if continuation:
            continue
        if pending and not pending.startswith("#"):
            lines.append(pending)
        pending = ""
    if pending:
        lines.append(pending)
    return lines


class ScriptRepoAdapter:
    """Extract benign command sessions from DevOps and infrastructure files."""

    source_name = "script_repo"
    license_name = "permissive repository license"
    suffixes = {".sh", ".bash", ".zsh", ".ksh"}
    ignored_parts = {".git", ".venv", "venv", "node_modules", "vendor", "dist", "build"}

    def iter_records(self, path: str | Path, **_: Any) -> Iterator[SessionRecord]:
        source = Path(path)
        paths = (
            [source]
            if source.is_file()
            else sorted(item for item in source.rglob("*") if item.is_file())
        )
        for file_path in paths:
            if any(part in self.ignored_parts for part in file_path.parts):
                continue
            category: str | None = None
            commands: list[str] = []
            if file_path.name.casefold().startswith("dockerfile"):
                category = "dockerfile"
                commands = self.extract_dockerfile(file_path)
            elif file_path.suffix.casefold() in self.suffixes:
                category = "shell"
                commands = self.extract_shell(file_path)
            elif file_path.suffix.casefold() in {".yaml", ".yml"}:
                try:
                    category, commands = self._extract_yaml(file_path)
                except (RuntimeError, UnicodeError, ValueError):
                    # Repository YAML often contains templating or multiple
                    # documents. One unsupported file must not abort the
                    # entire benign source.
                    continue
            if not commands:
                continue
            external_id = str(file_path.resolve())
            yield _make_record(
                source_name=self.source_name,
                source_kind=SourceKind.SCRIPT_BENIGN,
                license_name=self.license_name,
                external_id=external_id,
                commands=_as_commands(commands),
                malicious=False,
                meta={"script_path": str(file_path), "script_category": category},
            )

    @staticmethod
    def extract_dockerfile(path: str | Path) -> list[str]:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        commands: list[str] = []
        pending = ""
        for raw_line in lines:
            line = raw_line.strip()
            if not pending and not re.match(r"(?i)^(RUN|CMD|ENTRYPOINT)\b", line):
                continue
            instruction = (
                re.sub(r"(?i)^(RUN|CMD|ENTRYPOINT)\s*", "", line, count=1) if not pending else line
            )
            continuation = instruction.rstrip().endswith("\\")
            instruction = instruction.rstrip()
            if continuation:
                instruction = instruction[:-1]
            pending = f"{pending} {instruction}".strip()
            if continuation:
                continue
            command = re.sub(r"^\[(.*)\]$", r"\1", pending).replace('"', "")
            command = re.sub(r"\s*,\s*", " ", command)
            if command:
                commands.append(command)
            pending = ""
        if pending:
            commands.append(pending)
        return commands

    @staticmethod
    def extract_shell(path: str | Path) -> list[str]:
        commands: list[str] = []
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            commands.extend(_split_shell_lines(stripped))
        return commands

    def _extract_yaml(self, path: Path) -> tuple[str | None, list[str]]:
        document = load_yaml_file(path)
        if isinstance(document, list):
            ansible_commands = _walk_strings_for_keys(
                document,
                {
                    "shell",
                    "command",
                    "script",
                    "raw",
                    "ansible.builtin.shell",
                    "ansible.builtin.command",
                },
            )
            if ansible_commands or "ansible" in {part.casefold() for part in path.parts}:
                return "ansible", ansible_commands
            return None, []
        if not isinstance(document, dict):
            return None, []
        lowered_parts = {part.casefold() for part in path.parts}
        if ".github" in lowered_parts and "workflows" in lowered_parts:
            return "github_actions", _walk_strings_for_keys(document, {"run"})
        if document.get("apiVersion") and document.get("kind"):
            return "kubernetes", _walk_kubernetes_commands(document)
        ansible_commands = _walk_strings_for_keys(
            document,
            {
                "shell",
                "command",
                "script",
                "raw",
                "ansible.builtin.shell",
                "ansible.builtin.command",
            },
        )
        if ansible_commands or document.get("hosts") or document.get("tasks"):
            return "ansible", ansible_commands
        return None, []


def _walk_strings_for_keys(value: Any, keys: set[str]) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).casefold() in keys:
                if isinstance(child, str):
                    found.extend(_split_shell_lines(child))
                elif isinstance(child, list):
                    found.extend(str(item) for item in child if isinstance(item, (str, int, float)))
            found.extend(_walk_strings_for_keys(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_strings_for_keys(child, keys))
    return found


def _walk_kubernetes_commands(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        command = value.get("command")
        arguments = value.get("args")
        if command is not None or arguments is not None:
            command_parts = [command] if isinstance(command, str) else list(command or [])
            argument_parts = [arguments] if isinstance(arguments, str) else list(arguments or [])
            rendered = clean_command(" ".join(str(item) for item in command_parts + argument_parts))
            if rendered:
                found.append(rendered)
            return found
        for child in value.values():
            found.extend(_walk_kubernetes_commands(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_walk_kubernetes_commands(child))
    return found


class RealHistoryPoolAdapter:
    """Reconstruct sessions from a flat pool of observed real shell commands.

    ``data/raw/hf_bash_history/bash.txt`` is a newline-delimited concatenation
    of real user shell histories.  It is *not* a session corpus: the delimiter
    between sessions is only recoverable from the literal ``exit``/``logout``
    lines that some contributors kept in their history file.  The adapter is
    therefore explicit about the provenance of every reconstructed session:

    * a fragment that ends at an observed ``exit``/``logout``/``quit`` line is
      stamped ``session_boundary="real_exit_delimiter"`` (the boundary is
      present in the source);
    * anything else is chopped into ``window``-sized fragments and stamped
      ``session_boundary="synthetic_window"`` (the boundary is ours).

    In both cases ``parent_session_id`` points at the reconstructed fragment so
    a downstream family cap can never sever the provenance link, and
    ``meta.window_index`` records the fragment's position in that history block.

    The pool is real *commands*, but the session grouping is synthetic, so the
    kind is :attr:`SourceKind.REAL_HISTORY_POOL` and the label quality defaults
    to ``medium``; such records must never enter the temporal holdout or the
    blind test split.
    """

    source_name = "bash_history_pool"
    license_name = "unverified; upstream redistribution terms not documented"
    _TERMINATORS = frozenset({"exit", "logout", "quit", "\u0004"})

    def __init__(
        self,
        *,
        window: int = 12,
        max_window: int = 32,
        min_commands: int = 1,
    ) -> None:
        if window < 1:
            raise AdapterError("window must be positive")
        if max_window < window:
            raise AdapterError("max_window must be >= window")
        if min_commands < 1:
            raise AdapterError("min_commands must be positive")
        self.window = window
        self.max_window = max_window
        self.min_commands = min_commands

    def iter_records(
        self, path: str | Path, *, source_name: str | None = None, **_: Any
    ) -> Iterator[SessionRecord]:
        source = Path(path)
        if source.is_dir():
            paths = sorted(item for item in source.rglob("*") if item.is_file())
        else:
            paths = [source]
        name = source_name or self.source_name
        for document_path in paths:
            yield from self._iter_document(document_path, name)

    def _iter_document(self, path: Path, source_name: str) -> Iterator[SessionRecord]:
        block: list[tuple[int, str]] = []
        _block_start = 1
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                command = clean_command(raw_line)
                if not command:
                    continue
                if command.casefold() in self._TERMINATORS:
                    if block:
                        yield from self._emit_block(block, path, source_name, observed=True)
                    block = []
                    _block_start = line_number + 1
                    continue
                if not block:
                    _block_start = line_number  # unused; carried in emitted tuples
                block.append((line_number, command))
            if block:
                yield from self._emit_block(block, path, source_name, observed=False)

    def _emit_block(
        self,
        block: list[tuple[int, str]],
        path: Path,
        source_name: str,
        *,
        observed: bool,
    ) -> Iterator[SessionRecord]:
        if not block:
            return
        if observed and len(block) <= self.max_window:
            chunks = [block]
        else:
            chunks = [
                block[offset : offset + self.window] for offset in range(0, len(block), self.window)
            ]
        block_first = block[0][0]
        block_last = block[-1][0]
        # The block id is positional (it identifies *where* in the pool the
        # fragment came from) and is deliberately distinct from the session id
        # so lineage never collapses into a self-loop.
        block_id = (
            "fragment:"
            + hashlib.sha256(f"{path.name}:{block_first}-{block_last}".encode()).hexdigest()[:32]
        )
        for offset, chunk in enumerate(chunks):
            commands = [command for _, command in chunk]
            if len(commands) < self.min_commands:
                continue
            line_numbers = [line_number for line_number, _ in chunk]
            boundary = (
                "real_exit_delimiter" if observed and len(chunks) == 1 else "synthetic_window"
            )
            record = _make_record(
                source_name=source_name,
                source_kind=SourceKind.REAL_HISTORY_POOL,
                license_name=self.license_name,
                external_id=f"{path.name}:{line_numbers[0]}-{line_numbers[-1]}",
                commands=_as_commands(commands),
                malicious=False,
                quality=LabelQuality.MEDIUM,
                meta={
                    "session_boundary": boundary,
                    "pool_path": str(path),
                    "pool_line_start": line_numbers[0],
                    "pool_line_end": line_numbers[-1],
                    "history_block_id": block_id,
                    "window_index": offset,
                    "window_size": len(commands),
                },
            )
            yield record.model_copy(
                update={"source": record.source.model_copy(update={"parent_session_id": block_id})}
            )


_ADAPTERS: dict[str, type[Any]] = {
    "microsoft_iot": MicrosoftIoTAdapter,
    "microsoft": MicrosoftIoTAdapter,
    "shell_attack_evolution": ShellAttackEvolutionAdapter,
    "shell_evolution": ShellAttackEvolutionAdapter,
    "kypo": ZenodoAdapter,
    "zenodo": ZenodoAdapter,
    "atomic_red_team": AtomicRedTeamAdapter,
    "atomic": AtomicRedTeamAdapter,
    "script_repo": ScriptRepoAdapter,
    "scripts": ScriptRepoAdapter,
    "real_history_pool": RealHistoryPoolAdapter,
    "bash_history": RealHistoryPoolAdapter,
    "history_pool": RealHistoryPoolAdapter,
}


def adapter_for_kind(kind: str) -> Any:
    try:
        return _ADAPTERS[kind.casefold()]()
    except KeyError:
        raise ValueError(f"unsupported data adapter: {kind}") from None

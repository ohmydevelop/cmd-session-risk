"""Command-line interface for batch and online command-session scoring."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

import typer

from cmdrisk.runtime import (
    ArtifactLoadError,
    InferenceError,
    ScoringError,
    SessionScorer,
)
from cmdrisk.schema import SequenceScore, SessionRecord

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_ARTIFACT = 3
EXIT_INFERENCE = 4

app = typer.Typer(
    name="cmdrisk",
    help="Score Unix command prefixes with a calibrated ONNX artifact.",
    no_args_is_help=True,
)


def _json_line(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _error(message: str, *, code: str, exit_code: int) -> None:
    typer.echo(
        _json_line({"error": {"code": code, "message": message}}),
        err=True,
    )
    raise typer.Exit(exit_code)


def _load_scorer(artifact_dir: str | Path) -> SessionScorer:
    try:
        return SessionScorer.from_onnx(artifact_dir)
    except ArtifactLoadError as exc:
        _error(str(exc), code="artifact_load_error", exit_code=EXIT_ARTIFACT)
    except Exception as exc:
        _error(f"cannot load scorer: {exc}", code="artifact_load_error", exit_code=EXIT_ARTIFACT)
    raise AssertionError("unreachable")


def _parse_command(value: Any, *, line_number: int) -> tuple[str, float | None]:
    if isinstance(value, str):
        cmd, dt = value, None
    elif isinstance(value, Mapping):
        cmd, dt = value.get("cmd"), value.get("dt")
    else:
        raise ValueError(f"line {line_number}: command must be a string or object")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError(f"line {line_number}: command must contain a non-empty cmd")
    if dt is not None and (
        isinstance(dt, bool) or not isinstance(dt, (int, float)) or dt < 0
    ):
        raise ValueError(f"line {line_number}: dt must be a non-negative number or null")
    return cmd, None if dt is None else float(dt)


def _parse_session(value: Any, *, line_number: int) -> tuple[str, list[tuple[str, float | None]]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"line {line_number}: each JSONL record must be an object")

    if {"schema_version", "source", "family_id", "commands", "label"}.issubset(value):
        try:
            record = SessionRecord.model_validate(value)
        except Exception as exc:
            raise ValueError(f"line {line_number}: invalid SessionRecord: {exc}") from exc
        commands = [(item.cmd, item.dt) for item in record.commands]
        return record.session_id, commands

    session_id = value.get("session_id", f"line-{line_number}")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError(f"line {line_number}: session_id must be a non-empty string")
    raw_commands = value.get("commands")
    if not isinstance(raw_commands, list) or not raw_commands:
        raise ValueError(f"line {line_number}: commands must be a non-empty list")
    return session_id, [_parse_command(item, line_number=line_number) for item in raw_commands]


def _open_input(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdin, False
    try:
        return open(path, encoding="utf-8"), True
    except OSError as exc:
        raise ValueError(f"cannot open input {path}: {exc}") from exc


def _open_output(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdout, False
    try:
        return open(path, "w", encoding="utf-8"), True
    except OSError as exc:
        raise ValueError(f"cannot open output {path}: {exc}") from exc


@app.command("score")
def score(
    artifact_dir: str = typer.Option(..., "--artifact-dir", help="ONNX artifact directory."),
    input_path: str = typer.Option("-", "--input", help="Input JSONL path, or - for stdin."),
    output_path: str = typer.Option("-", "--output", help="Output JSONL path, or - for stdout."),
    flat: bool = typer.Option(
        False,
        "--flat",
        help="Emit one prefix score per line instead of one session result.",
    ),
) -> None:
    """Score every prefix from a batch of JSONL sessions."""

    try:
        scorer = _load_scorer(artifact_dir)
    except typer.Exit:
        raise

    try:
        source, close_source = _open_input(input_path)
        sink, close_sink = _open_output(output_path)
    except ValueError as exc:
        _error(str(exc), code="input_error", exit_code=EXIT_INPUT)

    try:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                session_id, commands = _parse_session(record, line_number=line_number)
            except (json.JSONDecodeError, ValueError) as exc:
                _error(str(exc), code="invalid_input", exit_code=EXIT_INPUT)

            state = scorer.start_session(session_id)
            try:
                for cmd, dt in commands:
                    state.update(cmd, dt)
                result = state.result()
            except (ValueError, InferenceError, ScoringError) as exc:
                _error(str(exc), code="inference_error", exit_code=EXIT_INFERENCE)

            if flat:
                for entry in result.entries:
                    sink.write(
                        _json_line(
                            {
                                **entry.model_dump(mode="json"),
                                "session_id": result.session_id,
                                "peak_probability": result.peak_probability,
                                "model_version": result.model_version,
                            }
                        )
                    )
                    sink.write("\n")
            else:
                sink.write(_json_line(result))
                sink.write("\n")
            sink.flush()
    finally:
        if close_source:
            source.close()
        if close_sink:
            sink.close()


def _parse_stream_line(raw_line: str, *, line_number: int) -> tuple[str, float | None]:
    try:
        value = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
    return _parse_command(value, line_number=line_number)


def run_stream(
    scorer: SessionScorer,
    *,
    session_id: str,
    source: Iterable[str],
    sink: TextIO,
) -> SequenceScore:
    """Run the online scoring loop against arbitrary text streams."""

    state = scorer.start_session(session_id)
    for line_number, raw_line in enumerate(source, start=1):
        if not raw_line.strip():
            continue
        try:
            cmd, dt = _parse_stream_line(raw_line, line_number=line_number)
            entry = state.update(cmd, dt)
        except (ValueError, InferenceError, ScoringError) as exc:
            raise ValueError(str(exc)) from exc
        payload = {
            **entry.model_dump(mode="json"),
            "session_id": session_id,
            "peak_probability": state.peak_probability,
            "model_version": scorer.model_version,
        }
        sink.write(_json_line(payload))
        sink.write("\n")
        sink.flush()
    return state.result()


@app.command("stream")
def stream(
    session_id: str = typer.Option(..., "--session-id", help="Identifier for this session."),
    artifact_dir: str = typer.Option(..., "--artifact-dir", help="ONNX artifact directory."),
) -> None:
    """Read ``{"cmd": ..., "dt": ...}`` JSON objects from stdin and alert online."""

    try:
        scorer = _load_scorer(artifact_dir)
        run_stream(scorer, session_id=session_id, source=sys.stdin, sink=sys.stdout)
    except typer.Exit:
        raise
    except ArtifactLoadError as exc:
        _error(str(exc), code="artifact_load_error", exit_code=EXIT_ARTIFACT)
    except ValueError as exc:
        _error(str(exc), code="invalid_input", exit_code=EXIT_INPUT)
    except (InferenceError, ScoringError) as exc:
        _error(str(exc), code="inference_error", exit_code=EXIT_INFERENCE)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

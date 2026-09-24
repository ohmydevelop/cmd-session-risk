"""Command-line interface for batch and online command-session scoring."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, TextIO

import typer

from cmdrisk.runtime import (
    ArtifactLoadError,
    InferenceError,
    ScoringError,
    SessionScorer,
)
from cmdrisk.schema import ScoreEntry, SequenceScore, SessionRecord

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


def _load_scorer(
    artifact_dir: str | Path,
    *,
    providers: Sequence[str] | None = None,
    intra_op_num_threads: int | None = None,
) -> SessionScorer:
    """Load an artifact, optionally pinning ONNX Runtime execution providers.

    ``providers`` is an ordered priority list, e.g. ``["CUDAExecutionProvider",
    "CPUExecutionProvider"]`` to prefer a GPU and fall back to CPU.
    """

    kwargs: dict[str, Any] = {}
    if providers:
        kwargs["providers"] = list(providers)
    if intra_op_num_threads is not None:
        kwargs["intra_op_num_threads"] = int(intra_op_num_threads)
    try:
        return SessionScorer.from_onnx(artifact_dir, **kwargs)
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
    if dt is not None and (isinstance(dt, bool) or not isinstance(dt, (int, float)) or dt < 0):
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


MAX_BATCH_SPLIT_DEPTH = 8


def _score_batches(
    scorer: SessionScorer,
    batches: list[list[tuple[int, int, list[tuple[str, float | None]], int]]],
    entries: dict[tuple[int, int], ScoreEntry],
    *,
    depth: int = 0,
) -> None:
    """Score pre-built batches, halving any batch the runtime cannot fit.

    Long prefixes make padded batches expensive, and the memory a batch needs
    depends on the shape of the widest row rather than on a fixed token count.
    Rather than pinning the runner to one hardware profile, a batch that fails
    to execute (typically an ONNX Runtime allocation error) is split and
    retried until it fits.  A single row that still fails propagates, so real
    inference errors are never silently swallowed.
    """

    for batch in batches:
        try:
            scored = scorer.score_batch([item[2] for item in batch])
        except (ValueError, InferenceError, ScoringError):
            if len(batch) <= 1 or depth >= MAX_BATCH_SPLIT_DEPTH:
                raise
            middle = len(batch) // 2
            _score_batches(scorer, [batch[:middle], batch[middle:]], entries, depth=depth + 1)
            continue
        for item, entry in zip(batch, scored, strict=True):
            entries[(item[0], item[1])] = entry


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
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        min=1,
        help="Score this many prefixes per ONNX Runtime call (dynamic batching).",
    ),
    provider: Annotated[
        list[str] | None,
        typer.Option(
            "--provider",
            help="ONNX Runtime execution provider; repeat to build a priority list.",
        ),
    ] = None,
    max_batch_tokens: int = typer.Option(
        32768,
        "--max-batch-tokens",
        min=1,
        help="Upper bound on padded tokens (rows x width) per batched ONNX call.",
    ),
    intra_op_threads: Annotated[
        int | None,
        typer.Option(
            "--intra-op-threads",
            min=1,
            help=(
                "Pin the ONNX Runtime intra-op thread pool.  Default (unset) lets "
                "ORT size it from the physical core count, which oversubscribes "
                "the machine when several scoring jobs run concurrently."
            ),
        ),
    ] = None,
) -> None:
    """Score every prefix from a batch of JSONL sessions."""

    try:
        scorer = _load_scorer(
            artifact_dir, providers=provider, intra_op_num_threads=intra_op_threads
        )
    except typer.Exit:
        raise

    try:
        source, close_source = _open_input(input_path)
        sink, close_sink = _open_output(output_path)
    except ValueError as exc:
        _error(str(exc), code="input_error", exit_code=EXIT_INPUT)

    def emit(result: SequenceScore) -> None:
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

    def score_streaming(session_id: str, commands: list[tuple[str, float | None]]) -> None:
        state = scorer.start_session(session_id)
        for cmd, dt in commands:
            state.update(cmd, dt)
        emit(state.result())

    def score_batched(pending: list[tuple[str, list[tuple[str, float | None]]]]) -> None:
        # Prefixes of one session are nested, so their token widths differ by
        # orders of magnitude.  Sorting by width before batching keeps padding
        # waste (and therefore GPU memory) low while results stay in input order.
        items: list[tuple[int, int, list[tuple[str, float | None]], int]] = []
        for session_index, (_session_id, commands) in enumerate(pending):
            for command_index in range(len(commands)):
                prefix = commands[: command_index + 1]
                items.append((session_index, command_index, prefix, scorer.token_count(prefix)))
        items.sort(key=lambda item: -item[3])

        batches: list[list[tuple[int, int, list[tuple[str, float | None]], int]]] = []
        current: list[tuple[int, int, list[tuple[str, float | None]], int]] = []
        width = 0
        for item in items:
            item_width = item[3]
            next_width = max(width, item_width)
            if current and (
                len(current) >= batch_size or next_width * (len(current) + 1) > max_batch_tokens
            ):
                batches.append(current)
                current, width = [], 0
                next_width = item_width
            current.append(item)
            width = next_width
        if current:
            batches.append(current)

        entries: dict[tuple[int, int], ScoreEntry] = {}
        try:
            _score_batches(scorer, batches, entries)
        except (ValueError, InferenceError, ScoringError) as exc:
            _error(str(exc), code="inference_error", exit_code=EXIT_INFERENCE)

        for session_index, (session_id, commands) in enumerate(pending):
            emit(
                scorer.sequence_score(
                    session_id,
                    [entries[(session_index, index)] for index in range(len(commands))],
                )
            )

    pending: list[tuple[str, list[tuple[str, float | None]]]] = []
    try:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                session_id, commands = _parse_session(record, line_number=line_number)
            except (json.JSONDecodeError, ValueError) as exc:
                _error(str(exc), code="invalid_input", exit_code=EXIT_INPUT)

            if batch_size > 1:
                pending.append((session_id, commands))
                if len(pending) >= batch_size:
                    score_batched(pending)
                    pending = []
                continue

            try:
                score_streaming(session_id, commands)
            except (ValueError, InferenceError, ScoringError) as exc:
                _error(str(exc), code="inference_error", exit_code=EXIT_INFERENCE)

        if batch_size > 1 and pending:
            score_batched(pending)
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
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        try:
            entry = state.update(cmd, dt)
        except (InferenceError, ScoringError):
            raise
        except ValueError as exc:
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

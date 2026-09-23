#!/usr/bin/env python3
"""Score JSONL sessions with the FP32 ONNX graph of an artifact directory.

The shipped runtime (``cmdrisk score`` / :class:`SessionScorer`) prefers
``model.int8.onnx`` when it is present.  Deployment calibration work needs the
*full precision* graph instead, so this driver injects an explicit FP32 ORT
session while keeping the artifact manifest verification intact.

Output rows are byte-compatible with ``cmdrisk score --flat`` so downstream
tools (``scripts/evaluate_recalibrated_onnx_scores.py``,
``scripts/recalibrate_sweep.py`` evidence, ``scripts/evaluate_common_v2.py``)
can consume FP32 and INT8 streams interchangeably.

Usage::

    scripts/score_fp32.py \
        --artifact-dir artifacts/<run>/onnx \
        --input data/processed/v4/dev.jsonl \
        --output /tmp/recal_v4/dev_fp32.jsonl \
        --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import onnxruntime as ort  # noqa: E402

from cmdrisk.cli import _error, _parse_session, _score_batches  # noqa: E402
from cmdrisk.runtime.scoring import SessionScorer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--input", required=True, help="Labelled session JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument(
        "--fp32-filename", default="model.onnx", help="FP32 graph filename"
    )
    parser.add_argument(
        "--chunk-sessions",
        type=int,
        default=16,
        help="Sessions scored per dynamic-batching window",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=None,
        help=(
            "Pin the ONNX Runtime intra-op thread pool.  Default (unset) lets ORT "
            "size it from the physical core count, which oversubscribes the host "
            "when several scoring jobs run concurrently."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.artifact_dir)
    sess_options = ort.SessionOptions()
    if args.intra_op_threads is not None:
        sess_options.intra_op_num_threads = int(args.intra_op_threads)
    session = ort.InferenceSession(
        str(root / args.fp32_filename),
        providers=["CPUExecutionProvider"],
        sess_options=sess_options,
    )
    scorer = SessionScorer.from_onnx(root, onnx_session=session)

    def score_batched(pending: list[tuple[str, list[Any]]]) -> dict[tuple[int, int], Any]:
        items = []
        for session_index, (_sid, commands) in enumerate(pending):
            for command_index in range(len(commands)):
                prefix = commands[: command_index + 1]
                items.append(
                    (session_index, command_index, prefix, scorer.token_count(prefix))
                )
        items.sort(key=lambda item: -item[3])
        batches: list[list[Any]] = []
        current: list[Any] = []
        width = 0
        for item in items:
            item_width = item[3]
            next_width = max(width, item_width)
            if current and (
                len(current) >= args.batch_size
                or next_width * (len(current) + 1) > args.max_batch_tokens
            ):
                batches.append(current)
                current, width = [], 0
                next_width = item_width
            current.append(item)
            width = next_width
        if current:
            batches.append(current)
        entries: dict[tuple[int, int], Any] = {}
        try:
            _score_batches(scorer, batches, entries)
        except Exception as exc:  # noqa: BLE001
            _error(str(exc), code="inference_error", exit_code=2)
        return entries

    def flush(sink: Any, pending: list[tuple[str, list[Any]]]) -> None:
        if not pending:
            return
        entries = score_batched(pending)
        for session_index, (session_id, commands) in enumerate(pending):
            result = scorer.sequence_score(
                session_id,
                [entries[(session_index, i)] for i in range(len(commands))],
            )
            for entry in result.entries:
                sink.write(
                    json.dumps(
                        {
                            **entry.model_dump(mode="json"),
                            "session_id": result.session_id,
                            "peak_probability": result.peak_probability,
                            "model_version": result.model_version,
                        }
                    )
                    + "\n"
                )

    with open(args.input, encoding="utf-8") as source, open(
        args.output, "w", encoding="utf-8"
    ) as sink:
        pending: list[tuple[str, list[Any]]] = []
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
                session_id, commands = _parse_session(record, line_number=line_number)
            except (json.JSONDecodeError, ValueError) as exc:
                _error(str(exc), code="invalid_input", exit_code=3)
            pending.append((session_id, commands))
            if len(pending) >= args.chunk_sessions:
                flush(sink, pending)
                pending = []
        flush(sink, pending)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

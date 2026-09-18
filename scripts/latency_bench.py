#!/usr/bin/env python3
"""Measure end-to-end ONNX Runtime prefix-scoring latency."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.runtime import SessionScorer  # noqa: E402


def parse_cpu_affinity(value: str | None) -> list[int] | None:
    """Parse ``0,1,2`` or ``0-7`` into a sorted affinity list."""

    if value is None or not value.strip():
        return None
    cpus: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            raise ValueError("empty CPU affinity item")
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid CPU range: {token}")
            cpus.update(range(start, end + 1))
        else:
            cpu = int(token)
            if cpu < 0:
                raise ValueError(f"invalid CPU id: {token}")
            cpus.add(cpu)
    if not cpus:
        raise ValueError("CPU affinity cannot be empty")
    return sorted(cpus)


@contextlib.contextmanager
def cpu_affinity(cpus: Sequence[int] | None) -> Iterator[None]:
    """Temporarily set the process CPU affinity when the platform supports it."""

    if not cpus:
        yield
        return
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("CPU affinity is not supported on this platform")
    previous = os.sched_getaffinity(0)
    try:
        os.sched_setaffinity(0, set(cpus))
        yield
    finally:
        os.sched_setaffinity(0, previous)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile compatible with NumPy's default method."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between 0 and 1")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _prefix_for_tokens(scorer: SessionScorer, target_tokens: int) -> list[tuple[str, None]]:
    if target_tokens < 1:
        raise ValueError("target token count must be positive")

    probe_count = min(64, max(4, target_tokens // 32))
    probe = [("true", None) for _ in range(probe_count)]
    probe_tokens = scorer._encode(probe)["input_ids"].shape[1]
    per_command = max(1.0, probe_tokens / probe_count)
    command_count = max(1, math.ceil(target_tokens / per_command))
    commands: list[tuple[str, None]] = [("true", None) for _ in range(command_count)]

    # Correct the estimate because tokenizers can merge adjacent markers.
    while True:
        actual = scorer._encode(commands)["input_ids"].shape[1]
        if actual >= target_tokens or len(commands) >= 100_000:
            return commands
        deficit = target_tokens - actual
        command_count += max(1, math.ceil(deficit / per_command))
        commands = [("true", None) for _ in range(command_count)]


def run_benchmark(
    scorer: SessionScorer,
    *,
    lengths: Sequence[int],
    warmup: int,
    iterations: int,
) -> list[dict[str, Any]]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    results: list[dict[str, Any]] = []
    for target in lengths:
        commands = _prefix_for_tokens(scorer, target)
        actual_tokens = int(scorer._encode(commands)["input_ids"].shape[1])
        for _ in range(warmup):
            scorer.score_commands(commands)
        latencies: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            scorer.score_commands(commands)
            latencies.append((time.perf_counter() - started) * 1000.0)
        results.append(
            {
                "requested_tokens": target,
                "actual_tokens": actual_tokens,
                "commands": len(commands),
                "warmup": warmup,
                "iterations": iterations,
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "mean_ms": statistics.fmean(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            }
        )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[2048, 8192],
        help="Token lengths to benchmark (default: 2048 8192).",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--cpu-affinity",
        default=None,
        help="CPU list or range, for example 0-7 or 0,2,4,6.",
    )
    parser.add_argument("--intra-op-threads", type=int, default=None)
    parser.add_argument("--inter-op-threads", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--format", choices=("json", "table"), default="table"
    )
    return parser


def _format_table(results: Sequence[dict[str, Any]]) -> str:
    lines = [
        "tokens  commands  p50_ms  p95_ms  mean_ms",
        "------  --------  ------  ------  -------",
    ]
    for item in results:
        lines.append(
            f"{item['actual_tokens']:6d}  {item['commands']:8d}  "
            f"{item['p50_ms']:6.2f}  {item['p95_ms']:6.2f}  {item['mean_ms']:7.2f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        cpus = parse_cpu_affinity(args.cpu_affinity)
        with cpu_affinity(cpus):
            scorer = SessionScorer.from_onnx(
                args.artifact_dir,
                intra_op_num_threads=args.intra_op_threads,
                inter_op_num_threads=args.inter_op_threads,
            )
            results = run_benchmark(
                scorer,
                lengths=args.lengths,
                warmup=args.warmup,
                iterations=args.iterations,
            )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    payload = {
        "artifact_dir": str(args.artifact_dir),
        "cpu_affinity": cpus,
        "results": results,
    }
    rendered = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
        if args.format == "json"
        else _format_table(results)
    )
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

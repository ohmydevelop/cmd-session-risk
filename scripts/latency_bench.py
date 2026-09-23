#!/usr/bin/env python3
"""Measure end-to-end ONNX Runtime prefix-scoring latency.

The deployment profile is deliberately concrete: an 8-core (or larger) x86-64
CPU with AVX2 running the ONNX Runtime CPU execution provider.  Benchmarks are
only considered promotable when they run on that profile and stay within the
documented thresholds plus tolerance.  See ``docs/QUALITY_GATES.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.runtime import SessionScorer  # noqa: E402

LATENCY_REPORT_SCHEMA = "cmdrisk.latency_report"
LATENCY_REPORT_VERSION = 1

DEFAULT_P95_THRESHOLD_MS = 50.0
DEFAULT_P99_THRESHOLD_MS = 70.0
DEFAULT_TOLERANCE_MS = 10.0

REQUIRED_ARCHITECTURE = "x86_64"
REQUIRED_MIN_LOGICAL_CPUS = 8
REQUIRED_AVX2 = True
REQUIRED_PROVIDERS = ("CPUExecutionProvider",)
CPU_DEPLOYMENT_PROFILE = "cpu"
ACCELERATED_DEPLOYMENT_PROFILE = "accelerated"


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

    # ``_encode`` truncates to the artifact's deployment max_length.  Without
    # this guard a request for a larger length can never be satisfied, so the
    # correction loop below would keep growing the command list until its
    # 100k-command safety valve.  Fail fast instead of producing a misleading
    # benchmark after minutes of tokenizer work.
    max_length = int(getattr(scorer, "max_length", 0) or 0)
    if max_length and target_tokens > max_length:
        raise ValueError(
            f"requested {target_tokens} tokens exceeds artifact max_length "
            f"{max_length}; measure at <= max_length or train/export a larger "
            "deployment context"
        )

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
                "p99_ms": percentile(latencies, 0.99),
                "mean_ms": statistics.fmean(latencies),
                "min_ms": min(latencies),
                "max_ms": max(latencies),
            }
        )
    return results


def detect_avx2() -> bool | None:
    """Return whether the running CPU advertises AVX2, or ``None`` if unknown."""

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        cpuinfo = ""
    if cpuinfo:
        for line in cpuinfo.splitlines():
            if line.lower().startswith("flags") or line.lower().startswith("features"):
                _, _, raw = line.partition(":")
                flags = raw.split()
                return "avx2" in {flag.lower() for flag in flags}
    return None


def detect_cpu_model() -> str | None:
    model = platform.processor() or None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        cpuinfo = ""
    for line in cpuinfo.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip() in {"model name", "Hardware"} and value.strip():
            model = value.strip()
            break
    return model


def hardware_profile(
    *,
    affinity_cpus: Sequence[int] | None,
    providers: Sequence[str],
    onnxruntime_version: str | None,
) -> dict[str, Any]:
    """Collect the hardware/runtime fingerprint recorded with the benchmark."""

    return {
        "platform": platform.platform(),
        "os": platform.system(),
        "architecture": platform.machine(),
        "processor_model": detect_cpu_model(),
        "logical_cpus": os.cpu_count(),
        "affinity_cpus": list(affinity_cpus) if affinity_cpus else None,
        "affinity_count": len(affinity_cpus) if affinity_cpus else None,
        "avx2": detect_avx2(),
        "onnxruntime_version": onnxruntime_version,
        "providers": list(providers),
    }


def evaluate_profile(
    profile: Mapping[str, Any],
    *,
    required_architecture: str = REQUIRED_ARCHITECTURE,
    min_logical_cpus: int = REQUIRED_MIN_LOGICAL_CPUS,
    require_avx2: bool = REQUIRED_AVX2,
) -> tuple[bool, list[str]]:
    """Return whether the profile satisfies the documented minimum and why not."""

    failures: list[str] = []
    architecture = str(profile.get("architecture") or "")
    if architecture not in {required_architecture, "amd64"}:
        failures.append(f"architecture_not_{required_architecture}")

    logical = profile.get("logical_cpus")
    if not isinstance(logical, int) or logical < min_logical_cpus:
        failures.append(f"insufficient_logical_cpus<{min_logical_cpus}")

    if require_avx2:
        avx2 = profile.get("avx2")
        if avx2 is None:
            failures.append("avx2_unknown")
        elif avx2 is not True:
            failures.append("avx2_missing")

    return (not failures), failures


def deployment_profile(providers: Sequence[str]) -> str:
    """Return ``cpu`` when the benchmark ran on the documented CPU profile.

    The promotion gate is defined for the CPU deployment profile only.  Runs on
    an accelerated provider are still measured and reported, but they are
    labelled ``accelerated`` and never satisfy the gate.
    """

    if tuple(providers) == REQUIRED_PROVIDERS:
        return CPU_DEPLOYMENT_PROFILE
    return ACCELERATED_DEPLOYMENT_PROFILE


def evaluate_latency(
    results: Sequence[Mapping[str, Any]],
    *,
    p95_threshold_ms: float,
    p99_threshold_ms: float,
    tolerance_ms: float,
) -> dict[str, Any]:
    """Aggregate the worst percentile across lengths and apply the gate."""

    if not results:
        raise ValueError("latency results cannot be empty")
    p95 = max(float(item["p95_ms"]) for item in results)
    p99 = max(float(item["p99_ms"]) for item in results)
    failures: list[str] = []
    if p95 > p95_threshold_ms + tolerance_ms:
        failures.append("p95_exceeds_tolerance")
    if p99 > p99_threshold_ms + tolerance_ms:
        failures.append("p99_exceeds_tolerance")
    return {
        "p95_ms": p95,
        "p99_ms": p99,
        "p95_threshold_ms": float(p95_threshold_ms),
        "p99_threshold_ms": float(p99_threshold_ms),
        "threshold_ms": float(p95_threshold_ms),
        "tolerance_ms": float(tolerance_ms),
        "passed": not failures,
        "failures": failures,
    }


def build_report(
    *,
    artifact_dir: str | Path,
    profile: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
    cpu_affinity_cpus: Sequence[int] | None,
    p95_threshold_ms: float,
    p99_threshold_ms: float,
    tolerance_ms: float,
    require_avx2: bool = REQUIRED_AVX2,
    min_logical_cpus: int = REQUIRED_MIN_LOGICAL_CPUS,
) -> dict[str, Any]:
    profile_dict = dict(profile)
    profile_passed, profile_failures = evaluate_profile(
        profile_dict, min_logical_cpus=min_logical_cpus, require_avx2=require_avx2
    )
    profile_dict["passed"] = profile_passed
    profile_dict["failures"] = profile_failures

    latency = evaluate_latency(
        results,
        p95_threshold_ms=p95_threshold_ms,
        p99_threshold_ms=p99_threshold_ms,
        tolerance_ms=tolerance_ms,
    )
    measured_profile = deployment_profile(profile_dict.get("providers") or [])
    failures = [*profile_failures, *latency["failures"]]
    if measured_profile != CPU_DEPLOYMENT_PROFILE:
        failures.append("latency_not_cpu_deployment_profile")
    return {
        "schema": LATENCY_REPORT_SCHEMA,
        "deployment_profile": measured_profile,
        "version": LATENCY_REPORT_VERSION,
        "artifact_dir": str(artifact_dir),
        "cpu_affinity": list(cpu_affinity_cpus) if cpu_affinity_cpus else None,
        "profile": profile_dict,
        "profile_passed": profile_passed,
        "provider": (profile_dict.get("providers") or [None])[0],
        "onnxruntime_version": profile_dict.get("onnxruntime_version"),
        "p95_threshold_ms": latency["p95_threshold_ms"],
        "p99_threshold_ms": latency["p99_threshold_ms"],
        "threshold_ms": latency["threshold_ms"],
        "tolerance_ms": latency["tolerance_ms"],
        "p95_ms": latency["p95_ms"],
        "p99_ms": latency["p99_ms"],
        "results": [dict(item) for item in results],
        "passed": not failures,
        "failures": failures,
    }


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
    parser.add_argument(
        "--provider",
        dest="providers",
        action="append",
        default=None,
        help=(
            "ONNX Runtime execution provider to request; repeatable. "
            "Defaults to CPUExecutionProvider (the documented deployment profile)."
        ),
    )
    parser.add_argument("--intra-op-threads", type=int, default=None)
    parser.add_argument("--inter-op-threads", type=int, default=None)
    parser.add_argument("--p95-threshold-ms", type=float, default=DEFAULT_P95_THRESHOLD_MS)
    parser.add_argument("--p99-threshold-ms", type=float, default=DEFAULT_P99_THRESHOLD_MS)
    parser.add_argument("--tolerance-ms", type=float, default=DEFAULT_TOLERANCE_MS)
    parser.add_argument("--min-logical-cpus", type=int, default=REQUIRED_MIN_LOGICAL_CPUS)
    parser.add_argument(
        "--allow-non-avx2",
        action="store_true",
        help="Do not require AVX2 (report profile_passed=false is still reported).",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Attach the report to the artifact export manifest (requires --output).",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Fail closed when the artifact is not eligible for promotion.",
    )
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Exit zero even when the latency/profile gate fails (diagnostics only).",
    )
    parser.add_argument("--format", choices=("json", "table"), default="table")
    return parser


def _format_table(report: Mapping[str, Any]) -> str:
    lines = [
        "tokens  commands  p50_ms  p95_ms  p99_ms  mean_ms",
        "------  --------  ------  ------  ------  -------",
    ]
    for item in report["results"]:
        lines.append(
            f"{item['actual_tokens']:6d}  {item['commands']:8d}  "
            f"{item['p50_ms']:6.2f}  {item['p95_ms']:6.2f}  "
            f"{item['p99_ms']:6.2f}  {item['mean_ms']:7.2f}"
        )
    lines.append("")
    lines.append(
        f"deployment_profile={report['deployment_profile']} "
        f"profile_passed={report['profile_passed']} passed={report['passed']} "
        f"p95={report['p95_ms']:.2f}<={report['p95_threshold_ms'] + report['tolerance_ms']:.2f} "
        f"p99={report['p99_ms']:.2f}<={report['p99_threshold_ms'] + report['tolerance_ms']:.2f}"
    )
    if report.get("failures"):
        lines.append("failures: " + ", ".join(str(item) for item in report["failures"]))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.update_manifest and args.output is None:
        print(
            json.dumps({"error": "--update-manifest requires --output"}),
            file=sys.stderr,
        )
        return 2

    cpus: list[int] | None = None
    try:
        cpus = parse_cpu_affinity(args.cpu_affinity)
        with cpu_affinity(cpus):
            scorer = SessionScorer.from_onnx(
                args.artifact_dir,
                providers=args.providers or list(REQUIRED_PROVIDERS),
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

    providers: list[str] = []
    try:
        providers = list(scorer._onnx_session.get_providers())
    except (AttributeError, TypeError):
        providers = []
    ort_version: str | None = None
    try:
        import onnxruntime as ort

        ort_version = getattr(ort, "__version__", None)
    except ImportError:
        ort_version = None

    profile = hardware_profile(
        affinity_cpus=cpus, providers=providers, onnxruntime_version=ort_version
    )
    report = build_report(
        artifact_dir=args.artifact_dir,
        profile=profile,
        results=results,
        cpu_affinity_cpus=cpus,
        p95_threshold_ms=args.p95_threshold_ms,
        p99_threshold_ms=args.p99_threshold_ms,
        tolerance_ms=args.tolerance_ms,
        require_avx2=not args.allow_non_avx2,
        min_logical_cpus=args.min_logical_cpus,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.update_manifest:
        from cmdrisk.training.export import update_export_manifest_latency

        try:
            update_export_manifest_latency(
                artifact_dir=args.artifact_dir,
                latency_report_path=args.output,
                require_promotion=args.require_promotion,
            )
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
            return 1
    elif args.require_promotion and not report["passed"]:
        print(
            json.dumps(
                {"error": "latency gate failed", "failures": report["failures"]},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    if not report["passed"]:
        print(
            json.dumps(
                {"gate": "latency", "failures": report["failures"]},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )

    rendered = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
        if args.format == "json"
        else _format_table(report)
    )
    print(rendered)
    if report["passed"] or args.allow_failure:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

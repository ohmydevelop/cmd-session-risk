#!/usr/bin/env python3
"""Collect a machine-readable environment report for remote training hosts."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _run(
    command: Sequence[str], *, timeout: float = 5.0
) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _nvidia_preflight() -> dict[str, Any]:
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: list[dict[str, Any]] = []
    driver_version: str | None = None
    if query is not None and query.returncode == 0:
        for line in query.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 4:
                continue
            name, memory, driver, compute_cap = fields[:4]
            memory_mib = int(memory) if memory.isdigit() else None
            gpus.append(
                {
                    "name": name,
                    "memory_total_mib": memory_mib,
                    "memory_total_bytes": memory_mib * 1024 * 1024 if memory_mib else None,
                    "compute_capability": compute_cap,
                }
            )
            driver_version = driver_version or driver

    cuda_version: str | None = None
    banner = _run(["nvidia-smi"])
    if banner is not None and banner.returncode == 0:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", banner.stdout)
        if match:
            cuda_version = match.group(1)

    if not gpus:
        try:
            import torch  # type: ignore[import-not-found]
        except (ImportError, OSError):
            pass
        else:
            if torch.cuda.is_available():
                cuda_version = cuda_version or torch.version.cuda
                for index in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(index)
                    memory = int(props.total_memory)
                    gpus.append(
                        {
                            "name": props.name,
                            "memory_total_mib": memory // (1024 * 1024),
                            "memory_total_bytes": memory,
                            "compute_capability": f"{props.major}.{props.minor}",
                        }
                    )
                if gpus and driver_version is None:
                    driver_version = getattr(torch.version, "cuda", None)

    return {
        "count": len(gpus),
        "cuda_version": cuda_version,
        "driver_version": driver_version,
        "devices": gpus,
    }


def _cpu_preflight() -> dict[str, Any]:
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
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except OSError:
            affinity = None
    return {
        "architecture": platform.machine(),
        "model": model,
        "logical_count": os.cpu_count(),
        "affinity_count": len(affinity) if affinity is not None else None,
        "affinity": affinity,
    }


def _ram_preflight() -> dict[str, int | None]:
    total: int | None = None
    available: int | None = None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        meminfo = ""
    for line in meminfo.splitlines():
        key, _, raw_value = line.partition(":")
        value = raw_value.strip().split()
        if not value or not value[0].isdigit():
            continue
        kib = int(value[0])
        if key == "MemTotal":
            total = kib * 1024
        elif key == "MemAvailable":
            available = kib * 1024
    if total is None and hasattr(os, "sysconf"):
        try:
            total = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError):
            total = None
    return {"total_bytes": total, "available_bytes": available}


def _disk_preflight(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _python_preflight() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "compiler": platform.python_compiler(),
    }


def collect_preflight(path: Path | None = None) -> dict[str, Any]:
    """Collect machine information without requiring privileged commands."""

    target = (path or Path.cwd()).expanduser().resolve()
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": _python_preflight(),
        "gpu": _nvidia_preflight(),
        "cpu": _cpu_preflight(),
        "ram": _ram_preflight(),
        "disk": {
            "path": str(target),
            **_disk_preflight(target),
        },
    }


def _version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    parts: list[int] = []
    for token in re.split(r"[.\-]", value):
        if not token.isdigit():
            break
        parts.append(int(token))
    return tuple(parts) or None


def validate_preflight(
    report: Mapping[str, Any],
    *,
    require_cuda: bool = False,
    min_vram_gb: float = 0.0,
    min_free_disk_gb: float = 0.0,
    min_cpu_cores: int = 0,
    min_cuda_version: str | None = None,
) -> dict[str, Any]:
    """Apply fail-closed minimums to a collected environment report."""

    failures: list[str] = []
    checks: dict[str, Any] = {}

    gpu = report.get("gpu") if isinstance(report.get("gpu"), Mapping) else {}
    devices = gpu.get("devices") if isinstance(gpu.get("devices"), list) else []
    gpu_count = int(gpu.get("count") or len(devices) or 0)
    checks["gpu_count"] = gpu_count
    if require_cuda and gpu_count < 1:
        failures.append("no_cuda_gpu")

    max_vram_mib = 0
    for device in devices:
        if isinstance(device, Mapping):
            memory = device.get("memory_total_mib")
            if isinstance(memory, (int, float)) and memory > max_vram_mib:
                max_vram_mib = int(memory)
    checks["max_vram_mib"] = max_vram_mib
    if min_vram_gb > 0 and max_vram_mib < min_vram_gb * 1024:
        failures.append(f"insufficient_vram<{min_vram_gb:g}GiB")

    if min_cuda_version:
        required = _version_tuple(min_cuda_version)
        actual = _version_tuple(str(gpu.get("cuda_version") or ""))
        checks["cuda_version"] = gpu.get("cuda_version")
        if actual is None or (required is not None and actual < required):
            failures.append(f"cuda_version<{min_cuda_version}")

    disk = report.get("disk") if isinstance(report.get("disk"), Mapping) else {}
    free_bytes = disk.get("free_bytes")
    checks["free_bytes"] = free_bytes
    if min_free_disk_gb > 0:
        minimum = min_free_disk_gb * 1024**3
        if not isinstance(free_bytes, (int, float)) or free_bytes < minimum:
            failures.append(f"insufficient_free_disk<{min_free_disk_gb:g}GiB")

    cpu = report.get("cpu") if isinstance(report.get("cpu"), Mapping) else {}
    logical = cpu.get("logical_cpus")
    affinity = cpu.get("affinity_count")
    cores = affinity if isinstance(affinity, int) and affinity > 0 else logical
    checks["cpu_cores"] = cores
    if min_cpu_cores > 0 and (not isinstance(cores, int) or cores < min_cpu_cores):
        failures.append(f"insufficient_cpu_cores<{min_cpu_cores}")

    return {
        "passed": not failures,
        "failures": failures,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to this path instead of stdout.")
    parser.add_argument(
        "--path", type=Path, default=None, help="Path used for disk reporting (default: cwd)."
    )
    parser.add_argument("--compact", action="store_true", help="Emit single-line JSON.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--min-vram-gb", type=float, default=0.0)
    parser.add_argument("--min-free-disk-gb", type=float, default=0.0)
    parser.add_argument("--min-cpu-cores", type=int, default=0)
    parser.add_argument("--min-cuda-version", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = collect_preflight(args.path)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1

    gate = validate_preflight(
        report,
        require_cuda=args.require_cuda,
        min_vram_gb=args.min_vram_gb,
        min_free_disk_gb=args.min_free_disk_gb,
        min_cpu_cores=args.min_cpu_cores,
        min_cuda_version=args.min_cuda_version,
    )
    report = {**report, "gates": gate}
    text = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=None if args.compact else 2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    if not gate["passed"]:
        print(
            json.dumps({"gate": "remote_preflight", "failures": gate["failures"]}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

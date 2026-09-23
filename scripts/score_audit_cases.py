#!/usr/bin/env python
"""Score the proxy-audit cases with a trained ONNX artifact.

The proxy audit needs one session-level ``malicious_probability`` per case.
This helper keeps that step reproducible:

1. read ``--cases`` (audit ``cases.jsonl``) and collect its ``session_id``s;
2. stream ``--test`` (the processed split) and write the matching sessions to a
   subset JSONL, in the audit's case order;
3. call ``cmdrisk score`` on the subset to get one result per session;
4. map ``peak_probability`` (the max over prefix scores, i.e. "does this session
   ever look malicious") to ``malicious_probability`` and write one prediction
   line per audit case.

Usage:
    scripts/score_audit_cases.py \
        --cases reports/proxy_audit/cases.jsonl \
        --test data/processed/v2/test.jsonl \
        --artifact-dir artifacts/<run>/onnx \
        --out reports/proxy_audit/predictions.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=None)
    return parser


def _session_ids(path: Path) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        session_id = record.get("session_id")
        if not session_id:
            raise ValueError(f"{path}:{line_number}: missing session_id")
        ids.append(session_id)
    return ids


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_ids = _session_ids(args.cases)
    wanted = set(case_ids)
    if len(wanted) != len(case_ids):
        raise SystemExit("duplicate session_id in --cases")

    work_dir = args.work_dir or args.out.parent / "_score_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    subset_path = work_dir / "audit_subset.jsonl"
    raw_scores = work_dir / "audit_scores.jsonl"

    kept: dict[str, str] = {}
    for line in args.test.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        session_id = record.get("session_id")
        if session_id in wanted:
            kept[session_id] = line
    missing = [session_id for session_id in case_ids if session_id not in kept]
    if missing:
        raise SystemExit(f"{len(missing)} audit cases absent from --test, e.g. {missing[0]}")

    with subset_path.open("w", encoding="utf-8") as handle:
        for session_id in case_ids:
            handle.write(kept[session_id])
            handle.write("\n")

    command = [
        sys.executable,
        "-m",
        "cmdrisk.cli",
        "score",
        "--artifact-dir",
        str(args.artifact_dir),
        "--input",
        str(subset_path),
        "--output",
        str(raw_scores),
    ]
    print("running:", " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)

    scored: dict[str, dict[str, object]] = {}
    for line in raw_scores.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scored[str(record["session_id"])] = record

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for session_id in case_ids:
            record = scored[session_id]
            entries = record.get("entries") or []
            payload = {
                "session_id": session_id,
                "malicious_probability": float(record["peak_probability"]),
                "peak_probability": float(record["peak_probability"]),
                "final_probability": (
                    float(entries[-1]["malicious_probability"]) if entries else None
                ),
                "prefix_count": len(entries),
                "model_version": record.get("model_version"),
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(f"wrote {written} predictions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

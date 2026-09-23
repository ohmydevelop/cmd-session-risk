#!/usr/bin/env python3
"""Build a blind three-reviewer proxy-audit pool from held-out sessions.

The pool is deliberately built from the *test* split only, so an audit case can
never leak into training or calibration.  Reviewers receive the blind payload
(`blind/chunk_*.jsonl`), which contains nothing but `session_id` and the
ordered command list: no gold labels, no model predictions, no source metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.teacher import AuditCase  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="held-out sessions JSONL")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-malicious", type=int, default=400)
    parser.add_argument("--num-benign", type=int, default=400)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser


def _rank(seed: int, session_id: str) -> bytes:
    return hashlib.sha256(f"{seed}:{session_id}".encode()).digest()


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def _select(records: list[dict], *, malicious: bool, limit: int, seed: int) -> list[dict]:
    pool = [item for item in records if bool(item["label"]["malicious"]) is malicious]
    pool.sort(key=lambda item: _rank(seed, item["session_id"]))
    return pool[:limit]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = _read_records(args.input)
    if not records:
        raise SystemExit(f"no sessions in {args.input}")

    chosen = _select(
        records, malicious=True, limit=args.num_malicious, seed=args.seed
    ) + _select(records, malicious=False, limit=args.num_benign, seed=args.seed)
    # Interleave so every chunk holds a comparable mix; ordering is deterministic.
    chosen.sort(key=lambda item: _rank(args.seed, item["session_id"]))

    out_dir = args.output_dir
    blind_dir = out_dir / "blind"
    blind_dir.mkdir(parents=True, exist_ok=True)

    cases: list[AuditCase] = []
    chunks: list[list[dict]] = [
        chosen[index : index + args.chunk_size]
        for index in range(0, len(chosen), args.chunk_size)
    ]
    for index, chunk in enumerate(chunks):
        payload_lines: list[str] = []
        for record in chunk:
            case = AuditCase(
                session_id=record["session_id"],
                commands=record["commands"],
                gold={
                    "malicious": int(record["label"]["malicious"]),
                    "first_evidence_index": record["label"].get("first_evidence_index"),
                    "tactics": record["label"].get("tactics", []),
                    "techniques": record["label"].get("techniques", []),
                },
                prediction={},
                meta={
                    "source_kind": record["source"]["kind"],
                    "source_name": record["source"]["name"],
                    "family_id": record["family_id"],
                    "split": record.get("split"),
                    "command_count": len(record["commands"]),
                },
            )
            cases.append(case)
            payload_lines.append(
                json.dumps(
                    {
                        "session_id": record["session_id"],
                        "commands": record["commands"],
                    },
                    ensure_ascii=False,
                    sort_keys=False,
                )
            )
        (blind_dir / f"chunk_{index:03d}.jsonl").write_text(
            "\n".join(payload_lines) + "\n", encoding="utf-8"
        )

    with (out_dir / "cases.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(case.model_dump_json() + "\n")

    sources = Counter(case.meta["source_kind"] for case in cases)
    manifest = {
        "input": str(args.input),
        "input_sessions": len(records),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "chunks": len(chunks),
        "cases": len(cases),
        "malicious_sampled": sum(int(case.gold["malicious"]) for case in cases),
        "benign_sampled": sum(1 - int(case.gold["malicious"]) for case in cases),
        "source_kinds": dict(sorted(sources.items())),
        "selection": (
            "hash-ranked deterministic sample by sha256(seed:session_id); "
            "reviewers see only session_id and commands"
        ),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build *untrained* architecture variants of SmallBertForCommandRisk.

Purpose: measure how much bigger the served model can get before the INT8 ONNX
latency budget (p95 < 50 ms @ 512 tokens) is at risk.  Inference latency depends
on the architecture and dtype, not on the learned weights, so a randomly
initialised checkpoint exported through the exact same
``cmdrisk.training.export`` path yields a faithful latency probe.

CPU ISOLATION: this probe and the ONNX latency benchmark that follows it must
only run on an otherwise idle machine.  Running them next to a training job
inflated the first probe's latency by ~2.4x (see /tmp/sizing/NOTES.md) and
slowed training from 2.1 s/step to 8.5 s/step.

The generated checkpoints are probes only.  They must never be promoted,
benchmarked as accuracy evidence, or shipped: this script writes a
``PROBE_ONLY.json`` marker next to every checkpoint it emits.

Usage::

    python scripts/model_ladder_latency_probe.py \
        --base-checkpoint artifacts/cmd-session-risk-v3-l6-h256-20260920/short_phase \
        --output-dir /tmp/sizing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# (name, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size)
CANDIDATES = [
    ("L4-H256-A4", 256, 4, 4, 1024),
    ("L6-H256-A4", 256, 6, 4, 1024),
    ("L8-H256-A4", 256, 8, 4, 1024),
    ("L12-H256-A4", 256, 12, 4, 1024),
    ("L6-H384-A6", 384, 6, 6, 1536),
    ("L8-H384-A6", 384, 8, 6, 1536),
    ("L6-H512-A8", 512, 6, 8, 2048),
]

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "command_tokenization.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from cmdrisk.small_model import SmallBertForCommandRisk, SmallBertRiskConfig

    base_config = SmallBertRiskConfig.from_pretrained(args.base_checkpoint)
    marker = {
        "schema": "cmdrisk.model_ladder_probe",
        "probe_only": True,
        "note": (
            "Randomly initialised architecture probe. Not a trained model: do not "
            "promote, do not report accuracy from it, do not ship it."
        ),
        "base_checkpoint": str(args.base_checkpoint),
    }

    rows = []
    for name, hidden, layers, heads, intermediate in CANDIDATES:
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        config = SmallBertRiskConfig.from_dict(base_config.to_dict())
        config.hidden_size = hidden
        config.num_hidden_layers = layers
        config.num_attention_heads = heads
        config.intermediate_size = intermediate
        config.backbone_model_name_or_path = f"probe://{name}"
        # keep the seeded init deterministic per candidate
        model = SmallBertForCommandRisk(config)
        model.save_pretrained(out)
        for fname in TOKENIZER_FILES:
            src = args.base_checkpoint / fname
            if src.exists():
                shutil.copy2(src, out / fname)
        (out / "PROBE_ONLY.json").write_text(json.dumps({**marker, "variant": name}, indent=2) + "\n", encoding="utf-8")
        params = sum(p.numel() for p in model.parameters())
        rows.append(
            {
                "variant": name,
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "intermediate_size": intermediate,
                "parameters": params,
                "checkpoint": str(out),
            }
        )
        print(f"{name:12s} params={params:,} -> {out}")

    (args.output_dir / "ladder.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

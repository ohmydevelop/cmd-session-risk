#!/usr/bin/env python3
"""Average (soup) a set of training checkpoints into a new checkpoint directory.

Why
---
Checkpoint averaging is the cheapest "modern" optimisation available to this
run: it costs *zero* extra training steps and no inference-time FLOPs, yet it
usually buys a small but real robustness gain over the last checkpoint because
the tail of an epoch is dominated by gradient noise around a flat minimum.

This script is deliberately additive.  It never writes into, renames or deletes
any existing checkpoint, run directory or artefact; it materialises a brand-new
directory and refuses to touch an existing one unless ``--force`` is given.

What it averages
----------------
``model.safetensors`` tensors.  Every tensor in the BERT risk model is float32
(verified in run: 112/112 tensors), so a plain elementwise mean is safe.  The
script asserts dtype/shape compatibility across inputs and aborts on mismatch
rather than silently averaging mismatched weights.  Non-weight files
(config.json, tokenizer*, label_vocabulary.json, command_tokenization.json,
model_config.json) are copied verbatim from the *first* checkpoint, because
averaging happens on weights only.

Usage
-----
    # last three checkpoints of a run, uniform weights
    python scripts/merge_checkpoints.py \
        --glob 'artifacts/<run>/short_phase/checkpoint-*' --last 3 \
        --output-dir artifacts/<run>-soup-last3

    # explicit list with non-uniform weights
    python scripts/merge_checkpoints.py \
        --checkpoint a --checkpoint b --weights 0.5,0.5 \
        --output-dir /tmp/soup
"""

from __future__ import annotations

import argparse
import glob as glob_module
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

# Files copied verbatim next to the averaged weights so the result is a
# drop-in replacement for a training checkpoint.
SIDECAR_FILES = (
    "config.json",
    "model_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "label_vocabulary.json",
    "command_tokenization.json",
)

_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def _natural_key(path: Path) -> tuple:
    """Sort checkpoint-2 < checkpoint-10 (not lexicographic)."""
    numbers = [int(part) for part in re.findall(r"\d+", path.name)]
    return (numbers, path.name)


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    checkpoints: list[Path] = [Path(item) for item in args.checkpoint]
    if args.glob:
        if checkpoints:
            raise SystemExit("use either --checkpoint or --glob, not both")
        # ``Path.glob`` rejects absolute patterns, so use the stdlib glob which
        # accepts both relative (to cwd) and absolute patterns.
        matches = [Path(match) for match in glob_module.glob(args.glob) if Path(match).is_dir()]
        if not matches:
            raise SystemExit(f"--glob {args.glob!r} matched no directories")
        matches.sort(key=_natural_key)
        if args.last and args.last > 0:
            matches = matches[-args.last :]
        checkpoints = matches
    if len(checkpoints) < 2:
        raise SystemExit(f"need at least 2 checkpoints, got {len(checkpoints)}")
    return checkpoints


def _weight_path(checkpoint: Path) -> Path:
    for name in _WEIGHT_FILES:
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    raise SystemExit(f"{checkpoint}: no weights file ({', '.join(_WEIGHT_FILES)})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[], help="checkpoint dir (repeatable)")
    parser.add_argument("--glob", default=None, help="shell-style glob of checkpoint dirs")
    parser.add_argument("--last", type=int, default=0, help="keep only the last N glob matches")
    parser.add_argument("--weights", default=None, help="comma-separated weights, default uniform")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true", help="allow writing into an existing dir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    import torch
    from safetensors.torch import load_file, save_file

    checkpoints = _resolve_checkpoints(args)
    sources = [_weight_path(item) for item in checkpoints]

    if args.weights:
        weights = [float(part) for part in args.weights.split(",") if part.strip()]
        if len(weights) != len(checkpoints):
            raise SystemExit(f"--weights has {len(weights)} values for {len(checkpoints)} checkpoints")
    else:
        weights = [1.0] * len(checkpoints)
    total = sum(weights)
    if total <= 0:
        raise SystemExit("weights must sum to a positive number")
    weights = [value / total for value in weights]

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"{output_dir} exists and is not empty; pass --force to overwrite")

    print(f"merging {len(checkpoints)} checkpoints (weights={[round(w, 6) for w in weights]})")
    for checkpoint, weight in zip(checkpoints, weights, strict=True):
        print(f"  {weight:.4f}  {checkpoint}")

    state_dicts = [load_file(str(path)) for path in sources]
    reference = state_dicts[0]
    for index, (state, source) in enumerate(zip(state_dicts[1:], sources[1:], strict=True), start=2):
        if set(state) != set(reference):
            missing = sorted(set(reference) - set(state))[:5]
            extra = sorted(set(state) - set(reference))[:5]
            raise SystemExit(f"{source}: tensor set differs (missing={missing} extra={extra})")
        for key in reference:
            if state[key].shape != reference[key].shape:
                raise SystemExit(f"{source}: {key} shape {tuple(state[key].shape)} != {tuple(reference[key].shape)}")
            if state[key].dtype != reference[key].dtype:
                raise SystemExit(f"{source}: {key} dtype {state[key].dtype} != {reference[key].dtype}")
        print(f"  tensor sets match: {source}")

    averaged: dict[str, "torch.Tensor"] = {}
    for key, tensor in reference.items():
        if not tensor.is_floating_point():
            # Integer/bool buffers cannot be averaged; take the first checkpoint's
            # value verbatim and say so, rather than silently rounding.
            averaged[key] = tensor.clone()
            continue
        acc = torch.zeros_like(tensor, dtype=torch.float64)
        for state, weight in zip(state_dicts, weights, strict=True):
            acc.add_(state[key].to(torch.float64), alpha=weight)
        averaged[key] = acc.to(tensor.dtype)

    total_params = sum(int(tensor.numel()) for tensor in averaged.values())
    print(f"averaged {len(averaged)} tensors / {total_params:,} parameters")

    if args.dry_run:
        print("dry-run: nothing written")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "model.safetensors"
    save_file(averaged, str(target), metadata={"format": "pt"})
    copied: list[str] = []
    for name in SIDECAR_FILES:
        source = checkpoints[0] / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
            copied.append(name)

    provenance = {
        "schema": "cmdrisk.checkpoint_soup",
        "version": 1,
        "method": "weighted elementwise mean of model.safetensors (float64 accumulator)",
        "sources": [
            {"path": str(path), "weight": weight, "sha256": _sha256(source)}
            for path, source, weight in zip(checkpoints, sources, weights, strict=True)
        ],
        "output": {"path": str(target), "sha256": _sha256(target), "tensors": len(averaged), "parameters": total_params},
        "sidecar_files_copied_from": str(checkpoints[0]),
        "sidecar_files": copied,
    }
    (output_dir / "soup_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {target} and {output_dir / 'soup_provenance.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

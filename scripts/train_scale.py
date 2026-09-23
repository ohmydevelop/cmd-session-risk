#!/usr/bin/env python3
"""Additive training entry point for the large-scale (v4) corpus.

This is a thin wrapper around :func:`cmdrisk.training.pipeline.run_training`
that exposes the extra Hugging Face ``TrainingArguments`` knobs the
million-example runs need (frequent checkpoints, explicit micro batch,
optional ``torch.compile``) without touching ``kaggle/train.py`` or the v3
pipeline.  The v3 entry point and artefacts stay byte-for-byte untouched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cmdrisk.training.config import TrainingRunConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="google/bert_uncased_L-6_H-256_A-4")
    parser.add_argument("--base-checkpoint", default=None,
                        help="Continue training from an existing cmd-risk checkpoint dir.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--short-max-length", type=int, default=256)
    parser.add_argument("--long-max-length", type=int, default=8192)
    parser.add_argument("--command-max-tokens", type=int, default=128)
    parser.add_argument("--max-prefixes", type=int, default=32)
    parser.add_argument("--learning-rate-short", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--calibration-fpr", type=float, default=0.01)
    parser.add_argument("--proxy-audit", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--vram-gb", type=float, default=40.0)
    parser.add_argument("--skip-long-phase", action="store_true")
    parser.add_argument("--report-to", default="none")
    # extra knobs (phase overrides)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra phase override as key=value (value parsed as JSON when possible)",
    )
    return parser


def _phase_overrides(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    overrides: dict[str, object] = {}
    if args.micro_batch_size is not None:
        overrides["per_device_train_batch_size"] = int(args.micro_batch_size)
        overrides["per_device_eval_batch_size"] = int(args.micro_batch_size)
    if args.gradient_accumulation_steps is not None:
        overrides["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.save_steps is not None:
        overrides["save_steps"] = int(args.save_steps)
        overrides["save_strategy"] = "steps"
        # keep epoch checkpoints as a safety net as well
        overrides["save_total_limit"] = int(args.save_total_limit or 4)
    if args.save_total_limit is not None:
        overrides["save_total_limit"] = int(args.save_total_limit)
    if args.logging_steps is not None:
        overrides["logging_steps"] = int(args.logging_steps)
    if args.max_steps is not None:
        overrides["max_steps"] = int(args.max_steps)
    if args.torch_compile:
        overrides["torch_compile"] = True
    if args.gradient_checkpointing:
        overrides["gradient_checkpointing"] = True
        overrides["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    for raw in args.override:
        key, _, value = raw.partition("=")
        try:
            parsed: object = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        overrides[key] = parsed
    return {"short": overrides} if overrides else {}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    config = TrainingRunConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        base_checkpoint=args.base_checkpoint,
        seed=args.seed,
        max_epochs=args.max_epochs,
        short_max_length=args.short_max_length,
        long_max_length=args.long_max_length,
        command_max_tokens=args.command_max_tokens,
        max_prefixes=args.max_prefixes,
        learning_rate_short=args.learning_rate_short,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        pos_weight_cap=args.pos_weight_cap,
        calibration_fpr=args.calibration_fpr,
        proxy_audit_path=args.proxy_audit,
        early_stopping_patience=args.early_stopping_patience,
        dataloader_num_workers=args.dataloader_num_workers,
        vram_gb=args.vram_gb,
        skip_long_phase=args.skip_long_phase,
        report_to=args.report_to,
        phase_overrides=_phase_overrides(args),
    )
    from cmdrisk.training.pipeline import run_training

    final_checkpoint = run_training(config)
    print(f"Training complete; final checkpoint: {final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

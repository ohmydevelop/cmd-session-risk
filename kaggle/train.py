#!/usr/bin/env python3
"""Two-phase ModernBERT training entry point for Kaggle and Lightning."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--data", required=True, help="JSONL/JSON/Parquet session records")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=2)
    parser.add_argument("--short-max-length", type=int, default=2048)
    parser.add_argument("--long-max-length", type=int, default=8192)
    parser.add_argument("--command-max-tokens", type=int, default=512)
    parser.add_argument("--max-prefixes", type=int, default=32)
    parser.add_argument("--learning-rate-short", type=float, default=2e-5)
    parser.add_argument("--learning-rate-long", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--min-technique-sessions", type=int, default=50)
    parser.add_argument("--min-technique-families", type=int, default=3)
    parser.add_argument("--calibration-fpr", type=float, default=0.01)
    parser.add_argument(
        "--proxy-audit",
        default=None,
        help="ProxyAuditReport JSON used by the promotion gate",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--vram-gb", type=float, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--skip-long-phase", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingRunConfig:
    return TrainingRunConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        seed=args.seed,
        max_epochs=args.max_epochs,
        short_max_length=args.short_max_length,
        long_max_length=args.long_max_length,
        command_max_tokens=args.command_max_tokens,
        max_prefixes=args.max_prefixes,
        learning_rate_short=args.learning_rate_short,
        learning_rate_long=args.learning_rate_long,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        pos_weight_cap=args.pos_weight_cap,
        min_technique_sessions=args.min_technique_sessions,
        min_technique_families=args.min_technique_families,
        calibration_fpr=args.calibration_fpr,
        proxy_audit_path=args.proxy_audit,
        early_stopping_patience=args.early_stopping_patience,
        dataloader_num_workers=args.dataloader_num_workers,
        vram_gb=args.vram_gb,
        world_size=args.world_size,
        skip_long_phase=args.skip_long_phase,
        report_to=args.report_to,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from cmdrisk.training.pipeline import run_training
    except ImportError as exc:
        raise SystemExit(
            "Training dependencies are missing. Install with `uv sync --extra train`."
        ) from exc
    final_checkpoint = run_training(config_from_args(args))
    print(f"Training complete; final checkpoint: {final_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

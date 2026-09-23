#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX FP32 and dynamic INT8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-source", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--parity-atol", type=float, default=1e-4)
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument(
        "--latency-report",
        default=None,
        help="Latency benchmark JSON to embed as promotion evidence.",
    )
    parser.add_argument(
        "--require-promotion",
        action="store_true",
        help="Fail closed unless metrics and latency evidence satisfy promotion gates.",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help=(
            "Artifact revision stamped into runtime_config.json and reported by "
            "the runtime on every score. Defaults to the output directory's "
            "parent name (the training run id)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from cmdrisk.training.export import export_onnx
    except ImportError as exc:
        raise SystemExit(
            "Export dependencies are missing. Install with `uv sync --extra train`."
        ) from exc
    manifest = export_onnx(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        tokenizer_source=args.tokenizer_source,
        calibration_path=args.calibration,
        metrics_path=args.metrics,
        opset_version=args.opset,
        parity_atol=args.parity_atol,
        skip_int8=args.skip_int8,
        latency_report_path=args.latency_report,
        require_promotion=args.require_promotion,
        model_version=args.model_version,
    )
    print(f"ONNX export complete: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

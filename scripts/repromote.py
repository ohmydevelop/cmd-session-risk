#!/usr/bin/env python3
"""Recompute calibration + metrics for a finished run with a proxy-audit report.

Training runs launched without ``--proxy-audit`` record
``promotion.passed = false`` and a ``proxy_audit_missing`` blocker even when every
point-estimate gate passes.  Re-running the whole two-phase training just to
attach the audit report is wasteful, so this script replays only the cheap part:
it reloads the final checkpoint, rebuilds the dev/test prefix examples exactly as
``run_training`` does, re-fits calibration and re-evaluates test metrics with the
audit report injected.

Outputs are written to a separate directory by default so the original run
artifact stays byte-for-byte intact.

Usage:
    python scripts/repromote.py \
        --run-dir artifacts/<run> \
        --data data/processed/v2/train.jsonl \
        --proxy-audit reports/proxy_audit/report.json \
        --output-dir artifacts/<run>/repromoted
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--proxy-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint dir")
    parser.add_argument(
        "--training-config",
        type=Path,
        default=None,
        help="training_config.json (default: <run-dir>/training_config.json)",
    )
    parser.add_argument(
        "--phase",
        choices=("short", "long"),
        default=None,
        help="Batch-plan phase for inference (default: long when long_phase/ exists)",
    )
    for flag in (
        "max-prefixes",
        "seed",
        "calibration-fpr",
        "min-technique-sessions",
        "min-technique-families",
    ):
        parser.add_argument(
            f"--{flag}",
            type=float if flag == "calibration-fpr" else int,
            default=None,
            help=f"Override {flag.replace('-', '_')} from the training config",
        )
    return parser


def _load_config(run_dir: Path, config_path: Path | None, overrides: dict[str, object]):
    """Build the training config from the recorded JSON plus CLI overrides.

    ``training_config.json`` is written at the very end of ``run_training``, so a
    short-phase checkpoint can be promoted before the record exists.  Missing
    files fall back to the dataclass defaults, which must then be pinned with the
    explicit ``--max-prefixes`` / ``--seed`` / ... overrides.
    """

    from cmdrisk.training.config import TrainingRunConfig

    names = {field.name for field in dataclasses.fields(TrainingRunConfig)}
    raw: dict[str, object] = {}
    path = config_path or (run_dir / "training_config.json")
    if path.is_file():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        raw = {key: value for key, value in recorded.items() if key in names}
    else:
        print(f"warning: {path} missing; using defaults + overrides", file=sys.stderr)
    for key, value in overrides.items():
        if value is not None:
            raw[key] = value
    return TrainingRunConfig(**raw)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from cmdrisk.model_factory import load_risk_model
    from cmdrisk.tokenization import CommandTokenizer
    from cmdrisk.training.config import choose_batch_config
    from cmdrisk.training.data import expand_records, load_records
    from cmdrisk.training.labels import LabelVocabulary
    from cmdrisk.training.pipeline import (
        _calibrate_and_evaluate,
        _stable_family_partition,
    )

    run_dir = args.run_dir
    output_dir = args.output_dir or (run_dir / "repromoted")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(
        run_dir,
        args.training_config,
        {
            "max_prefixes": args.max_prefixes,
            "seed": args.seed,
            "calibration_fpr": args.calibration_fpr,
            "min_technique_sessions": args.min_technique_sessions,
            "min_technique_families": args.min_technique_families,
        },
    )
    proxy_audit_report = json.loads(args.proxy_audit.read_text(encoding="utf-8"))

    records = load_records(str(args.data))
    dev_records = [item for item in records if item.get("split") == "dev"]
    test_records = [item for item in records if item.get("split") == "test"]
    if not dev_records or not test_records:
        raise SystemExit("data must contain dev and test splits")

    vocabulary = LabelVocabulary.load(run_dir / "label_vocabulary.json")
    dev_examples = expand_records(
        dev_records, vocabulary, max_prefixes=config.max_prefixes
    )
    test_examples = expand_records(
        test_records, vocabulary, max_prefixes=config.max_prefixes
    )
    _selection, calibration_examples = _stable_family_partition(
        dev_examples, fraction=0.5, seed=config.seed
    )

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    if checkpoint is None:
        long_dir = run_dir / "long_phase"
        checkpoint = long_dir if long_dir.is_dir() else run_dir / "short_phase"

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    tokenizer = CommandTokenizer.from_pretrained(checkpoint)
    model = load_risk_model(checkpoint)
    model.to(device)
    model.eval()

    metadata_path = args.training_config or (run_dir / "training_config.json")
    metadata: dict[str, object] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    vram_gb = float(metadata.get("vram_gb") or 0.0)
    phase = args.phase or ("long" if (run_dir / "long_phase").is_dir() else "short")
    batch = choose_batch_config(vram_gb=vram_gb, phase=phase, world_size=1)
    batch_size = args.batch_size or batch.micro_batch_size

    print(
        f"repromote: checkpoint={checkpoint} device={device_name} "
        f"batch={batch_size} calibration_examples={len(calibration_examples)} "
        f"test_examples={len(test_examples)}",
        file=sys.stderr,
    )

    calibration, metrics = _calibrate_and_evaluate(
        model=model,
        tokenizer=tokenizer,
        calibration_examples=calibration_examples,
        test_examples=test_examples,
        batch_size=batch_size,
        device=device,
        vocabulary=vocabulary,
        max_fpr=config.calibration_fpr,
        proxy_audit_report=proxy_audit_report,
    )

    (output_dir / "calibration.json").write_text(
        json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    promotion = metrics.get("promotion", {})
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "data": str(args.data),
        "proxy_audit": str(args.proxy_audit),
        "device": device_name,
        "batch_size": batch_size,
        "promotion_passed": promotion.get("passed"),
        "promotion_violations": promotion.get("violations"),
        "malicious": metrics.get("malicious"),
        "detection_delay": metrics.get("detection_delay"),
    }
    (output_dir / "repromote_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

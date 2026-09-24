#!/usr/bin/env python3
"""Offline recalibration sweep for a finished run artifact (no retraining).

Re-fits temperature scaling + FPR-constrained thresholds on the *dev* split for
several false-positive budgets and re-evaluates the untouched *test* split with
the exact functions used by :mod:`cmdrisk.training.pipeline`.  Model logits are
collected once and cached as ``.npz`` so the sweep is cheap to repeat.

Nothing on the test split ever feeds a fit: temperatures and thresholds come
from the family-disjoint dev calibration half only.

Usage:
    python scripts/recalibrate_sweep.py \
        --run-dir artifacts/<run> \
        --dev data/processed/v3/dev.jsonl \
        --test data/processed/v3/test.jsonl \
        --proxy-audit reports/proxy_audit/report.json \
        --output-dir /tmp/recal/sweep
"""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--proxy-audit", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/recal/logits"))
    parser.add_argument(
        "--fpr-grid",
        default="0.005,0.01,0.015,0.02,0.025,0.03",
        help="comma separated false-positive budgets used for threshold search",
    )
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of dev families reserved for calibration.  The training "
            "pipeline uses 0.5 so the other half can drive early stopping; a "
            "frozen checkpoint can widen this to 1.0 for a sharper FPR quantile."
        ),
    )
    parser.add_argument("--max-prefixes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--phase",
        choices=("short", "long"),
        default=None,
        help="batch-plan phase for inference (default: long when long_phase/ exists)",
    )
    return parser


def _cache_key(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode())
    return digest.hexdigest()[:16]


def _collect_cached(collect, cache_path: Path, **kwargs):
    import numpy as np

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    payload = collect(**kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def _summarize(metrics: dict) -> dict:
    malicious = metrics["malicious"]
    delay = metrics["detection_delay"]
    promotion = metrics["promotion"]
    return {
        "recall": malicious["recall"],
        "fpr": malicious["fpr"],
        "precision": malicious["precision"],
        "pr_auc": malicious["pr_auc"],
        "roc_auc": malicious["roc_auc"],
        "brier": malicious["brier"],
        "ece": malicious["ece"],
        "false_negative": malicious["false_negative"],
        "false_positive": malicious["false_positive"],
        "tactic_macro_f1_ge50": metrics["tactic"]["macro_f1_support_ge_50"],
        "technique_macro_f1_ge50": metrics["technique"]["macro_f1_support_ge_50"],
        "detection_rate": delay["detection_rate"],
        "median_detection_delay_commands": delay["median_detection_delay_commands"],
        "false_positive_sessions": delay["false_positive_sessions"],
        "missed_sessions": delay["missed_sessions"],
        "promotion_passed": promotion["passed"],
        "promotion_violations": promotion["violations"],
        "slice_checks": {
            name: {"passed": item.get("passed"), "violations": item.get("violations")}
            for name, item in (promotion.get("slice_checks") or {}).items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from cmdrisk.model_factory import load_risk_model
    from cmdrisk.tokenization import CommandTokenizer
    from cmdrisk.training.calibration import fit_calibration
    from cmdrisk.training.config import choose_batch_config
    from cmdrisk.training.data import expand_records, load_records
    from cmdrisk.training.labels import LabelVocabulary
    from cmdrisk.training.pipeline import (
        _collect_logits,
        _metrics,
        _stable_family_partition,
    )

    run_dir = args.run_dir
    config_path = run_dir / "training_config.json"
    recorded: dict = {}
    if config_path.is_file():
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
    max_prefixes = args.max_prefixes or recorded.get("max_prefixes") or 32
    seed = args.seed if args.seed is not None else recorded.get("seed", 42)
    vram_gb = float(recorded.get("vram_gb") or 0.0)
    phase = args.phase or ("long" if (run_dir / "long_phase").is_dir() else "short")
    batch = choose_batch_config(vram_gb=vram_gb, phase=phase, world_size=1)
    batch_size = args.batch_size or batch.micro_batch_size

    checkpoint = args.checkpoint
    if checkpoint is None:
        long_dir = run_dir / "long_phase"
        checkpoint = long_dir if long_dir.is_dir() else run_dir / "short_phase"

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    tokenizer = CommandTokenizer.from_pretrained(checkpoint)
    model = load_risk_model(checkpoint)
    model.to(device)
    model.eval()

    vocabulary = LabelVocabulary.load(run_dir / "label_vocabulary.json")
    dev_records = load_records(str(args.dev))
    test_records = load_records(str(args.test))
    dev_examples = expand_records(dev_records, vocabulary, max_prefixes=max_prefixes)
    test_examples = expand_records(test_records, vocabulary, max_prefixes=max_prefixes)
    _selection, calibration_examples = _stable_family_partition(
        dev_examples, fraction=args.calibration_fraction, seed=seed
    )

    proxy_audit_report = (
        json.loads(args.proxy_audit.read_text(encoding="utf-8")) if args.proxy_audit else None
    )

    print(
        f"recalibrate: checkpoint={checkpoint} device={device_name} batch={batch_size} "
        f"dev_families={len({e.family_id for e in dev_examples})} "
        f"calibration_examples={len(calibration_examples)} test_examples={len(test_examples)}",
        file=sys.stderr,
    )

    # The test cache is deliberately independent of the calibration fraction so
    # that widening the calibration set never invalidates expensive test logits.
    test_tag = _cache_key("test", str(checkpoint.resolve()), args.test.resolve(), max_prefixes)
    calibration_tag = _cache_key(
        "cal",
        str(checkpoint.resolve()),
        args.dev.resolve(),
        max_prefixes,
        seed,
        args.calibration_fraction,
        len(calibration_examples),
    )
    calibration_predictions = _collect_cached(
        _collect_logits,
        args.cache_dir / f"{calibration_tag}_calibration.npz",
        model=model,
        examples=calibration_examples,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    test_predictions = _collect_cached(
        _collect_logits,
        args.cache_dir / f"{test_tag}_test.npz",
        model=model,
        examples=test_examples,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )

    grid = [float(item) for item in args.fpr_grid.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for max_fpr in grid:
        calibration = fit_calibration(
            malicious_logits=calibration_predictions["malicious_logits"],
            malicious_labels=calibration_predictions["malicious_labels"],
            tactic_logits=calibration_predictions["tactic_logits"],
            tactic_labels=calibration_predictions["tactic_labels"],
            tactic_ids=vocabulary.tactics,
            technique_logits=calibration_predictions["technique_logits"],
            technique_labels=calibration_predictions["technique_labels"],
            technique_ids=vocabulary.techniques,
            max_fpr=max_fpr,
        )
        metrics = _metrics(
            test_predictions,
            calibration,
            tactic_ids=vocabulary.tactics,
            technique_ids=vocabulary.techniques,
            proxy_audit_report=proxy_audit_report,
        )
        cell = args.output_dir / f"fpr_{max_fpr:g}"
        cell.mkdir(parents=True, exist_ok=True)
        (cell / "calibration.json").write_text(
            json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (cell / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        row = {
            "max_fpr": max_fpr,
            "malicious_temperature": calibration.malicious_temperature,
            "malicious_threshold": calibration.malicious_threshold,
            **_summarize(metrics),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "max_fpr": max_fpr,
                    "threshold": calibration.malicious_threshold,
                    "recall": row["recall"],
                    "fpr": row["fpr"],
                    "false_negative": row["false_negative"],
                    "false_positive": row["false_positive"],
                    "promotion_passed": row["promotion_passed"],
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "dev": str(args.dev),
        "test": str(args.test),
        "proxy_audit": str(args.proxy_audit) if args.proxy_audit else None,
        "device": device_name,
        "batch_size": batch_size,
        "max_prefixes": max_prefixes,
        "seed": seed,
        "calibration_fraction": args.calibration_fraction,
        "calibration_examples": len(calibration_examples),
        "calibration_sessions": len({e.session_id for e in calibration_examples}),
        "test_examples": len(test_examples),
        "test_sessions": len({e.session_id for e in test_examples}),
        "note": (
            "thresholds fitted on the family-disjoint dev calibration half; the "
            "test split is evaluation-only"
        ),
        "rows": rows,
    }
    (args.output_dir / "sweep.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cells": len(rows), "output": str(args.output_dir)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

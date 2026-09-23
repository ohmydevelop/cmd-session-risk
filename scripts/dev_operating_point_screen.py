#!/usr/bin/env python3
"""Dev-only operating-point screen for *candidate* checkpoints.

Why this exists
---------------
``recalibrate_sweep.py`` fits temperature + FPR-constrained thresholds on the
dev split and reports the *test* split.  That is exactly right for a frozen
release candidate, but it makes model *selection* impossible without reading
the holdout: comparing two candidates would mean picking the one that wins on
test, which leaks the holdout into the choice.

This script never touches the test split.  For every candidate checkpoint it

1. expands the dev sessions with the frozen family-aware cap-32 prefix policy,
2. splits dev families into a calibration half and an evaluation half with the
   same deterministic hash the training pipeline uses
   (:func:`cmdrisk.training.pipeline._stable_family_partition`),
3. fits temperature scaling + FPR-constrained thresholds on the calibration
   half only,
4. reports prefix- and session-level metrics on the *other* half, and
5. caches raw logits per (checkpoint, dev, max_prefixes) so re-runs, extra FPR
   budgets and extra calibration fractions are free.

Model selection therefore happens entirely on dev; the standard
``recalibrate_sweep.py`` (dev-fit, test-report) is then run once, on the winner,
and ``--baseline`` produces an explicit "no performance loss" verdict against
the incumbent checkpoint at every budget.

Usage
-----
    python scripts/dev_operating_point_screen.py \
        --run-dir artifacts/<run> \
        --dev data/processed/v4/dev.jsonl \
        --candidate final=artifacts/<run>/short_phase/checkpoint-16372 \
        --candidate soup3=artifacts/<run>-soup-last3 \
        --baseline final \
        --output work/recal_soup/screen.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="candidate checkpoint directory (repeatable, first should be the incumbent)",
    )
    parser.add_argument("--baseline", default=None, help="candidate label to diff against")
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=None,
        help=(
            "directory holding the tokenizer sidecars when a candidate "
            "checkpoint directory does not contain them (e.g. a raw HF trainer "
            "checkpoint-* directory)"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("work/dev_screen/logits"))
    parser.add_argument(
        "--fpr-grid",
        default="0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03",
    )
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-prefixes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--phase", choices=("short", "long"), default=None)
    parser.add_argument(
        "--recall-tolerance",
        type=float,
        default=0.0,
        help=(
            "absolute dev recall the candidate may lose before it is called a "
            "regression; 0 requires an exact-or-better recall"
        ),
    )
    return parser


def _cache_key(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode())
    return digest.hexdigest()[:16]


def parse_candidates(raw: list[str]) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in raw:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--candidate expects LABEL=PATH, got {item!r}")
        if label in seen:
            raise SystemExit(f"duplicate candidate label {label!r}")
        seen.add(label)
        candidates.append((label, Path(path)))
    if not candidates:
        raise SystemExit("at least one --candidate LABEL=PATH is required")
    return candidates


def family_split_masks(
    family_ids: list[str], *, fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Return (calibration_indices, evaluation_indices) using the training hash.

    The hash is byte-for-byte the ranking used by
    ``cmdrisk.training.pipeline._stable_family_partition`` so the split is the
    same one training used for its early-stopping half.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    unique = sorted(set(family_ids))
    target = max(1, int(round(len(unique) * fraction))) if unique else 0
    ranked = sorted(
        unique,
        key=lambda family_id: hashlib.sha256(f"{seed}:{family_id}".encode()).digest(),
    )
    calibration_families = set(ranked[:target])
    calibration_indices: list[int] = []
    evaluation_indices: list[int] = []
    for index, family_id in enumerate(family_ids):
        if family_id in calibration_families:
            calibration_indices.append(index)
        else:
            evaluation_indices.append(index)
    return calibration_indices, evaluation_indices


def _subset(payload: dict, indices: list[int]) -> dict:
    return {key: value[indices] for key, value in payload.items()}


def _summarize(metrics: dict, *, sessions: int) -> dict:
    malicious = metrics["malicious"]
    delay = metrics["detection_delay"]
    benign_sessions = sessions - int(delay["malicious_sessions"])
    return {
        "prefix_recall": malicious["recall"],
        "prefix_fpr": malicious["fpr"],
        "prefix_precision": malicious["precision"],
        "pr_auc": malicious["pr_auc"],
        "roc_auc": malicious["roc_auc"],
        "brier": malicious["brier"],
        "ece": malicious["ece"],
        "prefix_tp": malicious["true_positive"],
        "prefix_fn": malicious["false_negative"],
        "prefix_fp": malicious["false_positive"],
        "prefix_tn": malicious["true_negative"],
        "sessions": sessions,
        "malicious_sessions": int(delay["malicious_sessions"]),
        "benign_sessions": benign_sessions,
        "session_recall": delay["detection_rate"],
        "session_fpr": (
            delay["false_positive_sessions"] / benign_sessions if benign_sessions else None
        ),
        "session_false_positive": int(delay["false_positive_sessions"]),
        "session_missed": int(delay["missed_sessions"]),
        "median_detection_delay_commands": delay["median_detection_delay_commands"],
        "tactic_macro_f1_ge50": metrics["tactic"]["macro_f1_support_ge_50"],
        "technique_macro_f1_ge50": metrics["technique"]["macro_f1_support_ge_50"],
    }


def _collect_dev_logits(
    *,
    model,
    examples,
    tokenizer,
    batch_size: int,
    device,
    cache_path: Path,
):
    import numpy as np

    from cmdrisk.training.pipeline import _collect_logits

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    payload = _collect_logits(
        model=model,
        examples=examples,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    candidates = parse_candidates(args.candidate)
    labels = [label for label, _ in candidates]
    if args.baseline is not None and args.baseline not in labels:
        raise SystemExit(f"--baseline {args.baseline!r} is not one of {labels}")

    import numpy as np
    import torch

    from cmdrisk.model_factory import load_risk_model
    from cmdrisk.tokenization import CommandTokenizer
    from cmdrisk.training.calibration import fit_calibration
    from cmdrisk.training.config import choose_batch_config
    from cmdrisk.training.data import expand_records, load_records
    from cmdrisk.training.labels import LabelVocabulary
    from cmdrisk.training.pipeline import _metrics

    run_dir = args.run_dir
    recorded: dict = {}
    config_path = run_dir / "training_config.json"
    if config_path.is_file():
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
    max_prefixes = args.max_prefixes or recorded.get("max_prefixes") or 32
    seed = args.seed if args.seed is not None else recorded.get("seed", 42)
    vram_gb = float(recorded.get("vram_gb") or 0.0)
    phase = args.phase or ("long" if (run_dir / "long_phase").is_dir() else "short")
    batch = choose_batch_config(vram_gb=vram_gb, phase=phase, world_size=1)
    batch_size = args.batch_size or batch.micro_batch_size

    vocabulary = LabelVocabulary.load(run_dir / "label_vocabulary.json")
    dev_records = load_records(str(args.dev))
    dev_examples = expand_records(dev_records, vocabulary, max_prefixes=max_prefixes)
    family_ids = [example.family_id for example in dev_examples]
    calibration_indices, evaluation_indices = family_split_masks(
        family_ids, fraction=args.calibration_fraction, seed=seed
    )
    calibration_families = {family_ids[index] for index in calibration_indices}
    evaluation_families = {family_ids[index] for index in evaluation_indices}
    overlap = calibration_families & evaluation_families
    if overlap:
        raise SystemExit(f"family overlap between halves: {sorted(overlap)[:5]}")
    if not calibration_indices or not evaluation_indices:
        raise SystemExit("dev split produced an empty calibration or evaluation half")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    grid = [float(item) for item in args.fpr_grid.split(",") if item.strip()]

    print(
        json.dumps(
            {
                "dev_examples": len(dev_examples),
                "dev_sessions": len({example.session_id for example in dev_examples}),
                "calibration_examples": len(calibration_indices),
                "evaluation_examples": len(evaluation_indices),
                "calibration_fraction": args.calibration_fraction,
                "seed": seed,
                "batch_size": batch_size,
                "device": device_name,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
    )

    results: dict[str, dict] = {}
    for label, checkpoint in candidates:
        resolved = str(checkpoint.resolve())
        tokenizer_source = (
            args.tokenizer_source.resolve() if args.tokenizer_source else checkpoint
        )
        tokenizer = CommandTokenizer.from_pretrained(tokenizer_source)
        model = load_risk_model(checkpoint)
        model.to(device)
        model.eval()
        tag = _cache_key("devscreen", resolved, str(args.dev.resolve()), max_prefixes)
        payload = _collect_dev_logits(
            model=model,
            examples=dev_examples,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            cache_path=args.cache_dir / f"{tag}_dev.npz",
        )
        del model
        calibration_predictions = _subset(payload, calibration_indices)
        evaluation_predictions = _subset(payload, evaluation_indices)
        evaluation_sessions = len(
            {str(item) for item in evaluation_predictions["session_ids"]}
        )
        calibration_sessions = len(
            {str(item) for item in calibration_predictions["session_ids"]}
        )
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
                evaluation_predictions,
                calibration,
                tactic_ids=vocabulary.tactics,
                technique_ids=vocabulary.techniques,
            )
            rows.append(
                {
                    "max_fpr": max_fpr,
                    "malicious_temperature": calibration.malicious_temperature,
                    "malicious_threshold": calibration.malicious_threshold,
                    **_summarize(metrics, sessions=evaluation_sessions),
                }
            )
            print(
                json.dumps(
                    {
                        "candidate": label,
                        "max_fpr": max_fpr,
                        "threshold": calibration.malicious_threshold,
                        "prefix_recall": rows[-1]["prefix_recall"],
                        "prefix_fpr": rows[-1]["prefix_fpr"],
                        "session_recall": rows[-1]["session_recall"],
                        "session_fpr": rows[-1]["session_fpr"],
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
        results[label] = {
            "checkpoint": resolved,
            "checkpoint_sha256": _sha256_of_weights(checkpoint),
            "calibration_sessions": calibration_sessions,
            "evaluation_sessions": evaluation_sessions,
            "rows": rows,
        }

    verdicts = _verdicts(
        results,
        baseline=args.baseline,
        recall_tolerance=args.recall_tolerance,
    )
    payload_out = {
        "schema": "cmdrisk.dev_operating_point_screen",
        "version": 1,
        "run_dir": str(run_dir),
        "dev": str(args.dev),
        "max_prefixes": max_prefixes,
        "seed": seed,
        "calibration_fraction": args.calibration_fraction,
        "fpr_grid": grid,
        "recall_tolerance": args.recall_tolerance,
        "baseline": args.baseline,
        "note": (
            "dev-only: thresholds are fitted on the dev calibration half and "
            "reported on the disjoint dev evaluation half; the test split is "
            "never read by this script"
        ),
        "candidates": results,
        "verdicts": verdicts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"candidates": labels, "output": str(args.output)}))
    return 0


def _sha256_of_weights(checkpoint: Path) -> str | None:
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        return None
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verdicts(
    results: dict[str, dict], *, baseline: str | None, recall_tolerance: float
) -> dict:
    if baseline is None:
        return {}
    base_rows = {row["max_fpr"]: row for row in results[baseline]["rows"]}
    verdicts: dict[str, dict] = {}
    for label, payload in results.items():
        if label == baseline:
            continue
        per_budget = []
        wins = 0
        losses = 0
        for row in payload["rows"]:
            base = base_rows.get(row["max_fpr"])
            if base is None:
                continue
            recall_delta = row["prefix_recall"] - base["prefix_recall"]
            session_recall_delta = row["session_recall"] - base["session_recall"]
            fpr_delta = row["prefix_fpr"] - base["prefix_fpr"]
            session_fpr_delta = row["session_fpr"] - base["session_fpr"]
            no_loss = (
                recall_delta >= -recall_tolerance
                and session_recall_delta >= -recall_tolerance
                and fpr_delta <= 0.0
                and session_fpr_delta <= 0.0
            )
            wins += int(no_loss)
            losses += int(not no_loss)
            per_budget.append(
                {
                    "max_fpr": row["max_fpr"],
                    "prefix_recall_delta": recall_delta,
                    "prefix_fpr_delta": fpr_delta,
                    "session_recall_delta": session_recall_delta,
                    "session_fpr_delta": session_fpr_delta,
                    "dominates_baseline": bool(no_loss),
                }
            )
        verdicts[label] = {
            "baseline": baseline,
            "budgets": len(per_budget),
            "dominates_all_budgets": bool(per_budget) and losses == 0,
            "dominates_budgets": wins,
            "regresses_budgets": losses,
            "per_budget": per_budget,
        }
    return verdicts


if __name__ == "__main__":
    raise SystemExit(main())

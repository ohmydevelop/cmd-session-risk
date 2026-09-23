#!/usr/bin/env python3
"""Assemble a post-hoc *recalibrated* artifact from an existing trained run.

The recalibration only changes scalar temperatures/thresholds (documented in
``calibration.json``) plus the FPR budget; model weights, the ONNX graphs and
the tokenizer are byte-identical to the source run.  This script therefore
materialises a **new** run directory by hard-linking large immutable files and
copying the small, mutable JSON documents, then overwrites:

* ``calibration.json`` (run root + onnx/) - from the sweep cell,
* ``metrics.json``       (run root + onnx/) - from the sweep cell,
* ``onnx/runtime_config.json``             - new ``model_version``,
* ``onnx/README.md``                       - regenerated model card,
* ``onnx/export_manifest.json``            - ``model_version`` + promotion block.

It never mutates the source run.  After it returns you must run, in order::

    python scripts/validate_int8.py   --artifact-dir <out>/onnx --input <test.jsonl>
    python scripts/latency_bench.py   --artifact-dir <out>/onnx --iterations 100
    python scripts/refresh_artifact_manifest.py --artifact-dir <out>/onnx \
        --int8-model model.int8.onnx

Usage::

    python scripts/build_recal_artifact.py \
        --source-run artifacts/<trained-run> \
        --cell /tmp/recal/sweep/fpr_0.015 \
        --output-run artifacts/<trained-run>-recal-20260921 \
        --model-version cmd-session-risk-v3-l6-h256-recal-20260921
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

HARDLINK_MIN_BYTES = 5 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--cell", type=Path, required=True)
    parser.add_argument("--output-run", type=Path, required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=None,
        help="Path recorded in the README (defaults to the source run short_phase)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow writing into an existing output directory (never the source)",
    )
    return parser


def _copy_tree(source: Path, output: Path, *, force: bool) -> None:
    if output.exists():
        if not force:
            raise SystemExit(f"output run already exists: {output} (pass --force)")
    output.mkdir(parents=True, exist_ok=True)

    def _copy(src: str, dst: str) -> str:
        src_path = Path(src)
        dst_path = Path(dst)
        if src_path.stat().st_size >= HARDLINK_MIN_BYTES:
            try:
                if dst_path.exists():
                    dst_path.unlink()
                os.link(src_path, dst_path)
                return dst
            except OSError:
                pass
        shutil.copy2(src_path, dst_path)
        return dst

    shutil.copytree(source, output, copy_function=_copy, dirs_exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    if path.exists():
        path.unlink()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_promotion(metrics: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    promotion = metrics.get("promotion") or {}
    checks = promotion.get("checks") or {}
    metrics_passed = all(
        value for key, value in checks.items() if key != "evaluation_slices_passed"
    )
    proxy = promotion.get("proxy_audit") or {}
    failures: list[str] = []
    if not promotion.get("passed"):
        failures.append("promotion_gate_failed")
    if promotion.get("violations"):
        failures.append("promotion_violations_present")
    if "evaluation_slices_passed" in checks and not checks["evaluation_slices_passed"]:
        failures.append("evaluation_slices_passed")
    if proxy and not proxy.get("eligible_for_promotion", False):
        failures.append("proxy_audit")
    return {
        "eligible_for_promotion": bool(proxy.get("eligible_for_promotion", False)),
        "failures": failures,
        "latency_passed": bool(latency.get("passed")),
        "metrics_passed": bool(metrics_passed),
        "passed": bool(promotion.get("passed")) and bool(latency.get("passed")),
        "required": bool(promotion.get("passed")),
    }


def _render_readme(
    *,
    model_version: str,
    base_checkpoint: str,
    calibration: dict[str, Any],
    metrics: dict[str, Any],
) -> str:
    malicious = metrics.get("malicious", {})
    promotion = metrics.get("promotion", {})
    lines = [
        "# Command Session Risk ONNX model (post-hoc recalibrated)",
        "",
        f"- Model version: `{model_version}`",
        f"- Base checkpoint: `{base_checkpoint}` (Apache-2.0)",
        "- Backbone: `google/bert_uncased_L-6_H-256_A-4`",
        "- Outputs: malicious logit, 14 ATT&CK tactic logits, dynamic technique logits",
        "- Intended use: Unix/Linux command-prefix risk scoring and alerting",
        "- Not intended for automatic blocking without independent validation",
        "",
        "## Recalibration",
        "",
        "Weights are byte-identical to the base checkpoint. Only the scalar",
        "temperatures/thresholds below were post-hoc refit on the full,",
        "family-disjoint dev split (including examples previously used for epoch",
        "selection); the test split was never used for fitting.",
        "",
        "```json",
        json.dumps(
            {
                "malicious_temperature": calibration.get("malicious_temperature"),
                "malicious_threshold": calibration.get("malicious_threshold"),
                "tactic_temperature": calibration.get("tactic_temperature"),
                "technique_temperature": calibration.get("technique_temperature"),
                "target_fpr": calibration.get("target_fpr"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Headline metrics (held-out test split)",
        "",
        "```json",
        json.dumps(
            {
                key: malicious.get(key)
                for key in (
                    "recall",
                    "fpr",
                    "precision",
                    "pr_auc",
                    "roc_auc",
                    "brier",
                    "ece",
                    "threshold",
                    "positive_count",
                    "negative_count",
                )
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Promotion",
        "",
        "```json",
        json.dumps(
            {
                "passed": promotion.get("passed"),
                "violations": promotion.get("violations"),
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source_run.resolve()
    cell = args.cell.resolve()
    output = args.output_run.resolve()

    if not source.is_dir():
        raise SystemExit(f"source run not found: {source}")
    if source == output or source in output.parents:
        raise SystemExit("refusing to write inside the source run directory")
    for name in ("calibration.json", "metrics.json"):
        if not (cell / name).is_file():
            raise SystemExit(f"sweep cell missing {name}: {cell / name}")

    calibration = _read_json(cell / "calibration.json")
    metrics = _read_json(cell / "metrics.json")
    latency = _read_json(source / "onnx" / "latency_report.json")

    _copy_tree(source, output, force=args.force)

    base_checkpoint = args.base_checkpoint or str(source / "short_phase")
    _write_json(output / "calibration.json", calibration)
    _write_json(output / "metrics.json", metrics)
    _write_json(output / "onnx" / "calibration.json", calibration)
    _write_json(output / "onnx" / "metrics.json", metrics)
    _write_json(
        output / "onnx" / "runtime_config.json",
        {"model_version": args.model_version, "version": 1},
    )
    (output / "onnx" / "README.md").write_text(
        _render_readme(
            model_version=args.model_version,
            base_checkpoint=base_checkpoint,
            calibration=calibration,
            metrics=metrics,
        ),
        encoding="utf-8",
    )

    manifest_path = output / "onnx" / "export_manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        manifest["model_version"] = args.model_version
        manifest["promotion"] = _manifest_promotion(metrics, latency)
        _write_json(manifest_path, manifest)

    provenance = {
        "schema": "cmdrisk.recalibration",
        "version": 1,
        "source_run": str(source),
        "sweep_cell": str(cell),
        "model_version": args.model_version,
        "base_checkpoint": base_checkpoint,
        "weights_unchanged": True,
        "target_fpr": calibration.get("target_fpr"),
        "malicious_temperature": calibration.get("malicious_temperature"),
        "malicious_threshold": calibration.get("malicious_threshold"),
    }
    _write_json(output / "recalibration.json", provenance)

    print(
        json.dumps(
            {
                "output_run": str(output),
                "model_version": args.model_version,
                "target_fpr": calibration.get("target_fpr"),
                "malicious_threshold": calibration.get("malicious_threshold"),
                "promotion_passed": (metrics.get("promotion") or {}).get("passed"),
                "promotion_violations": (metrics.get("promotion") or {}).get("violations"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

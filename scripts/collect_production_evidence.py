#!/usr/bin/env python
"""Collect fail-closed production evidence from one or more cmd-session-risk artifacts.

Read-only. Never writes into artifact directories.

Usage:
    .venv/bin/python scripts/collect_production_evidence.py \
        --artifact v3=artifacts/cmd-session-risk-v3-l6-h256-recal-prod-20260921 \
        --artifact v4=artifacts/cmd-session-risk-v4-l6-h256-million-20260921-recal-prod-20260921 \
        --output-json /tmp/recal/evidence.json --output-md /tmp/recal/evidence.md

Every field is optional: missing files/keys are reported as null plus a note, so a
half-finished pipeline never silently produces a "clean" table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

SCHEMA = "cmdrisk.production_evidence_collection"
VERSION = 1


def _load(path: str) -> tuple[Any, str | None]:
    if not os.path.isfile(path):
        return None, f"missing:{os.path.basename(path)}"
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"unreadable:{os.path.basename(path)}:{type(exc).__name__}"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int(value: Any) -> int | None:
    v = _num(value)
    return None if v is None else int(v)


def _pick(d: Any, *keys: str) -> Any:
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _size(path: str) -> int | None:
    return os.path.getsize(path) if os.path.isfile(path) else None


def collect(label: str, root: str) -> dict:
    root = os.path.abspath(root)
    notes: list[str] = []

    recal, err = _load(os.path.join(root, "recalibration.json"))
    if err:
        notes.append(err)
    evalj, err = _load(os.path.join(root, "recalibrated_onnx_evaluation.json"))
    if err:
        notes.append(err)
    int8j, err = _load(os.path.join(root, "onnx", "int8_validation.json"))
    if err:
        notes.append(err)
    latj, err = _load(os.path.join(root, "onnx", "latency_report.json"))
    if err:
        notes.append(err)
    metj, err = _load(os.path.join(root, "metrics.json"))
    if err:
        notes.append(err)

    # INT8 evidence: prefer the fail-closed per-artifact validation, fall back to
    # the evaluation-side comparison block (explicitly flagged so the report can
    # refuse to treat the fallback as factory evidence).
    int8_source = "onnx/int8_validation.json"
    if int8j is None:
        int8j = evalj.get("int8_comparison") if isinstance(evalj, dict) else None
        if int8j is not None:
            int8_source = "recalibrated_onnx_evaluation.json#int8_comparison(fallback)"
            notes.append("int8_validation_missing_using_eval_fallback")

    test_session = _pick(evalj, "test", "session") or {}
    test_prefix = _pick(evalj, "test", "prefix") or {}
    dev_session = _pick(evalj, "dev", "session") or {}
    dev_prefix = _pick(evalj, "dev", "prefix") or {}
    promotion = _pick(metj, "promotion") or {}

    latency: dict[str, Any] | None = None
    if isinstance(latj, dict):
        results = latj.get("results") or []
        row = results[0] if results and isinstance(results[0], dict) else {}
        latency = {
            "passed": latj.get("passed"),
            "provider": latj.get("provider"),
            "profile_passed": latj.get("profile_passed"),
            "requested_tokens": row.get("requested_tokens"),
            "p50_ms": _num(row.get("p50_ms")),
            "p95_ms": _num(row.get("p95_ms")),
            "p99_ms": _num(row.get("p99_ms")),
            "max_ms": _num(row.get("max_ms")),
            "mean_ms": _num(row.get("mean_ms")),
            "iterations": row.get("iterations"),
        }

    return {
        "label": label,
        "artifact_dir": root,
        "model_version": _pick(recal, "model_version"),
        "threshold": _pick(recal, "malicious_threshold"),
        "temperature": _pick(recal, "malicious_temperature"),
        "target_fpr": _pick(recal, "target_fpr"),
        "sweep_cell": _pick(recal, "sweep_cell"),
        "weights_unchanged": _pick(recal, "weights_unchanged"),
        "test_session": {
            "recall": _num(test_session.get("recall")),
            "fpr": _num(test_session.get("fpr")),
            "precision": _num(test_session.get("precision")),
            "pr_auc": _num(test_session.get("pr_auc")),
            "roc_auc": _num(test_session.get("roc_auc")),
            "brier": _num(test_session.get("brier")),
            "ece": _num(test_session.get("ece")),
            "true_positive": _int(test_session.get("true_positive")),
            "false_positive": _int(test_session.get("false_positive")),
            "true_negative": _int(test_session.get("true_negative")),
            "false_negative": _int(test_session.get("false_negative")),
            "positive_count": _int(test_session.get("positive_count")),
            "negative_count": _int(test_session.get("negative_count")),
        },
        "test_prefix": {
            "detection_rate": _num(test_prefix.get("detection_rate")),
            "fpr": _num(test_prefix.get("fpr")),
            "true_positive": _int(test_prefix.get("true_positive")),
            "false_positive": _int(test_prefix.get("false_positive")),
            "false_negative": _int(test_prefix.get("false_negative")),
            "false_positive_sessions": _int(test_prefix.get("false_positive_sessions")),
            "missed_sessions": _int(test_prefix.get("missed_sessions")),
            "positive_count": _int(test_prefix.get("positive_count")),
            "negative_count": _int(test_prefix.get("negative_count")),
        },
        "dev_session": {
            "recall": _num(dev_session.get("recall")),
            "fpr": _num(dev_session.get("fpr")),
            "positive_count": _int(dev_session.get("positive_count")),
            "negative_count": _int(dev_session.get("negative_count")),
        },
        "dev_prefix": {
            "detection_rate": _num(dev_prefix.get("detection_rate")),
            "fpr": _num(dev_prefix.get("fpr")),
        },
        "int8": {
            "source": int8_source if int8j is not None else None,
            "threshold": _num(_pick(int8j, "threshold")),
            "prefixes_scored": _int(_pick(int8j, "prefixes_scored")),
            "sessions_scored": _int(_pick(int8j, "sessions_scored")),
            "session_decision_agreement": _num(_pick(int8j, "session_decision_agreement")),
            "prefix_decision_agreement": _num(_pick(int8j, "prefix_decision_agreement")),
            "delta_recall_int8_minus_fp32": _num(_pick(int8j, "delta_recall_int8_minus_fp32")),
            "delta_fpr_int8_minus_fp32": _num(_pick(int8j, "delta_fpr_int8_minus_fp32")),
            "max_prob_delta": _num(_pick(int8j, "prefix_probability_delta", "max")),
            "p99_prob_delta": _num(_pick(int8j, "prefix_probability_delta", "p99")),
            "mean_prob_delta": _num(_pick(int8j, "prefix_probability_delta", "mean")),
            "first_alert_index_drift_max": _num(_pick(int8j, "first_alert_index_drift", "max")),
            "first_alert_index_drift_sessions": _int(
                _pick(int8j, "first_alert_index_drift", "sessions_differing")
            ),
            "quantized_graph_bytes": _int(_pick(int8j, "quantized_graph_bytes")),
        },
        "latency": latency,
        "promotion": {
            "passed": promotion.get("passed"),
            "violations": promotion.get("violations"),
            "missing_metrics": promotion.get("missing_metrics"),
            "missing_evaluation_slices": promotion.get("missing_evaluation_slices"),
            "tactic_heads_support_ge_50": _int(promotion.get("tactic_heads_support_ge_50")),
            "technique_heads_support_ge_50": _int(promotion.get("technique_heads_support_ge_50")),
            "tactic_macro_f1_support_ge_50": _num(promotion.get("tactic_macro_f1_support_ge_50")),
            "technique_macro_f1_support_ge_50": _num(
                promotion.get("technique_macro_f1_support_ge_50")
            ),
            "proxy_audit_eligible": _pick(promotion, "proxy_audit", "eligible_for_promotion"),
        },
        "slices": _slices(metj, promotion),
        "file_bytes": {
            "model.onnx": _size(os.path.join(root, "onnx", "model.onnx")),
            "model.onnx.data": _size(os.path.join(root, "onnx", "model.onnx.data")),
            "model.int8.onnx": _size(os.path.join(root, "onnx", "model.int8.onnx")),
        },
        "notes": notes,
    }


def _slices(metj: Any, promotion: Any) -> dict[str, Any]:
    """Slice metrics live under slices.<name>.malicious with detection_delay beside it.

    Gate status lives in promotion.slice_checks.<name>{passed, class_balance.violations}.
    """
    out: dict[str, Any] = {}
    slices = _pick(metj, "slices")
    checks = _pick(promotion, "slice_checks") or {}
    if not isinstance(slices, dict):
        return out
    for name, block in slices.items():
        if not isinstance(block, dict):
            continue
        mal = block.get("malicious") if isinstance(block.get("malicious"), dict) else {}
        delay = (
            block.get("detection_delay") if isinstance(block.get("detection_delay"), dict) else {}
        )
        chk = (
            checks.get(name)
            if isinstance(checks, dict) and isinstance(checks.get(name), dict)
            else {}
        )
        balance = chk.get("class_balance") if isinstance(chk.get("class_balance"), dict) else {}
        out[name] = {
            "positive_count": _int(mal.get("positive_count")),
            "negative_count": _int(mal.get("negative_count")),
            "recall": _num(mal.get("recall")),
            "detection_rate": _num(delay.get("detection_rate")),
            "fpr": _num(mal.get("fpr")),
            "passed": chk.get("passed"),
            "violations": balance.get("violations"),
            "missing_metrics": chk.get("missing_metrics"),
        }
    return out


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_md(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# cmd-session-risk 生产证据汇总（自动采集，只读）")
    lines.append("")
    lines.append(f"schema `{SCHEMA}` v{VERSION}")
    lines.append("")
    lines.append("## 1. 会话级（运维口径）test 指标")
    lines.append("")
    lines.append("| artifact | threshold | recall | FPR | precision | PR-AUC | TP | FP | TN | FN |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["test_session"]
        lines.append(
            "| {label} | {thr} | {rec} | {fpr} | {prec} | {prauc} | {tp} | {fp} | {tn} | {fn} |".format(  # noqa: E501
                label=r["label"],
                thr=_fmt(r["threshold"], 8),
                rec=_fmt(s["recall"]),
                fpr=_fmt(s["fpr"]),
                prec=_fmt(s["precision"]),
                prauc=_fmt(s["pr_auc"]),
                tp=_fmt(s["true_positive"]),
                fp=_fmt(s["false_positive"]),
                tn=_fmt(s["true_negative"]),
                fn=_fmt(s["false_negative"]),
            )
        )
    lines.append("")
    lines.append("## 2. cap32 prefix 级 test 指标")
    lines.append("")
    lines.append(
        "| artifact | detection_rate | FPR | TP | FP | FN | FP sessions | missed sessions |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        p = r["test_prefix"]
        lines.append(
            "| {label} | {dr} | {fpr} | {tp} | {fp} | {fn} | {fps} | {ms} |".format(
                label=r["label"],
                dr=_fmt(p["detection_rate"]),
                fpr=_fmt(p["fpr"]),
                tp=_fmt(p["true_positive"]),
                fp=_fmt(p["false_positive"]),
                fn=_fmt(p["false_negative"]),
                fps=_fmt(p["false_positive_sessions"]),
                ms=_fmt(p["missed_sessions"]),
            )
        )
    lines.append("")
    lines.append("## 3. INT8 ↔ FP32 一致性（工厂证据）")
    lines.append("")
    lines.append(
        "| artifact | source | prefixes | sessions | session agree | prefix agree | Δrecall | ΔFPR | max Δp | p99 Δp | drift max |"  # noqa: E501
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        i = r["int8"]
        lines.append(
            "| {label} | {src} | {px} | {ss} | {sa} | {pa} | {dr} | {df} | {mx} | {p99} | {drift} |".format(  # noqa: E501
                label=r["label"],
                src=i["source"] or "—",
                px=_fmt(i["prefixes_scored"]),
                ss=_fmt(i["sessions_scored"]),
                sa=_fmt(i["session_decision_agreement"]),
                pa=_fmt(i["prefix_decision_agreement"]),
                dr=_fmt(i["delta_recall_int8_minus_fp32"]),
                df=_fmt(i["delta_fpr_int8_minus_fp32"]),
                mx=_fmt(i["max_prob_delta"]),
                p99=_fmt(i["p99_prob_delta"]),
                drift=_fmt(i["first_alert_index_drift_max"]),
            )
        )
    lines.append("")
    lines.append("## 4. 端侧延迟与体积")
    lines.append("")
    lines.append(
        "| artifact | tokens | p50 ms | p95 ms | p99 ms | max ms | latency pass | INT8 bytes | FP32 bytes |"  # noqa: E501
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lat = r["latency"] or {}
        fp32_bytes = None
        b = r["file_bytes"]
        if b.get("model.onnx") is not None:
            fp32_bytes = b["model.onnx"] + (b.get("model.onnx.data") or 0)
        lines.append(
            "| {label} | {tok} | {p50} | {p95} | {p99} | {mx} | {ok} | {i8} | {f32} |".format(
                label=r["label"],
                tok=_fmt(lat.get("requested_tokens")),
                p50=_fmt(lat.get("p50_ms"), 3),
                p95=_fmt(lat.get("p95_ms"), 3),
                p99=_fmt(lat.get("p99_ms"), 3),
                mx=_fmt(lat.get("max_ms"), 3),
                ok=_fmt(lat.get("passed")),
                i8=_fmt(b.get("model.int8.onnx")),
                f32=_fmt(fp32_bytes),
            )
        )
    lines.append("")
    lines.append("## 5. 门禁状态（fail-closed）")
    lines.append("")
    lines.append(
        "| artifact | promotion.passed | violations | missing metrics | missing slices | tactic heads ≥50 | technique heads ≥50 | proxy audit eligible |"  # noqa: E501
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        p = r["promotion"]
        lines.append(
            "| {label} | {ok} | {v} | {m} | {s} | {th} | {te} | {pa} |".format(
                label=r["label"],
                ok=_fmt(p["passed"]),
                v=len(p["violations"] or []) if p["violations"] is not None else "—",
                m=len(p["missing_metrics"] or []) if p["missing_metrics"] is not None else "—",
                s=len(p["missing_evaluation_slices"] or [])
                if p["missing_evaluation_slices"] is not None
                else "—",
                th=_fmt(p["tactic_heads_support_ge_50"]),
                te=_fmt(p["technique_heads_support_ge_50"]),
                pa=_fmt(p["proxy_audit_eligible"]),
            )
        )
    lines.append("")
    lines.append("## 6. 评估切片状态")
    lines.append("")
    lines.append(
        "| artifact | slice | positives | negatives | recall | detection_rate | FPR | gate passed | violations |"  # noqa: E501
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if not r["slices"]:
            lines.append(f"| {r['label']} | — | — | — | — | — | — | — | no slice data |")
            continue
        for name, s in r["slices"].items():
            violations = s.get("violations")
            vtxt = ", ".join(str(v) for v in violations) if violations else ""
            lines.append(
                "| {label} | {name} | {p} | {n} | {rec} | {dr} | {fpr} | {ok} | {v} |".format(
                    label=r["label"],
                    name=name,
                    p=_fmt(s.get("positive_count")),
                    n=_fmt(s.get("negative_count")),
                    rec=_fmt(s.get("recall")),
                    dr=_fmt(s.get("detection_rate")),
                    fpr=_fmt(s.get("fpr")),
                    ok=_fmt(s.get("passed")),
                    v=vtxt,
                )
            )
    lines.append("")
    lines.append("## 7. 采集备注")
    lines.append("")
    for r in rows:
        note = "; ".join(r["notes"]) if r["notes"] else "clean"
        lines.append(f"- **{r['label']}**: {note}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="artifact label and directory; repeatable",
    )
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    args = ap.parse_args(argv)

    if not args.artifact:
        ap.error("at least one --artifact LABEL=DIR is required")

    rows = []
    for item in args.artifact:
        if "=" not in item:
            ap.error(f"--artifact expects LABEL=DIR, got {item!r}")
        label, _, root = item.partition("=")
        rows.append(collect(label.strip(), root.strip()))

    payload = {"schema": SCHEMA, "version": VERSION, "artifacts": rows}
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)

    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)

    if args.output_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as fh:
            fh.write(render_md(rows))

    return 0


if __name__ == "__main__":
    sys.exit(main())

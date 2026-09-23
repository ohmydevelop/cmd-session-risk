#!/usr/bin/env python3
"""External SOTA-style baseline: GLiClass zero-shot on the v6 obf corpus.

GLiClass (gliclass 0.1.20, knowledgator/gliclass-small-v1) is an independent,
publicly available lightweight zero-shot classifier.  We score the exact same
eval slices as the v6 in-house gates so the comparison is on identical inputs.

Metrics are threshold-free where possible:
  - ROC-AUC of the malicious-label score vs ground truth
  - recall at the threshold that keeps benign_twins FPR <= 0.02
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    auc = 0.0
    for p in pos:
        auc += float(np.mean(p > neg)) + 0.5 * float(np.mean(p == neg))
    return auc / max(1, len(pos))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(ROOT / "data/processed/v6_obf_evals"))
    ap.add_argument("--out", default=str(ROOT / "work/v6_baseline/gliclass_small_results.json"))
    ap.add_argument("--model", default="knowledgator/gliclass-instruct-base-v1.0")
    args = ap.parse_args()

    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer

    model = GLiClassModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pipe = ZeroShotClassificationPipeline(
        model, tokenizer, classification_type="multi-label", device="cpu")
    classes = ["malicious command that will damage the system",
               "benign routine command"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = {"model": args.model, "slices": {}}
    threshold = None

    def records(name: str):
        return [json.loads(l) for l in
                (Path(args.eval_dir) / name).open() if l.strip()]

    # benign twins first: pick score threshold with FPR <= 0.02
    ben = records("benign_twins_eval.jsonl")
    ben_scores = []
    for r in ben:
        text = r["commands"][0]["cmd"]
        res = pipe(text, classes, threshold=0.0)[0]
        vals = {rr["label"]: rr["score"] for rr in res}
        ben_scores.append(float(vals.get(classes[0], 0.0)))
    ben_scores = np.asarray(ben_scores)
    if len(ben_scores):
        t_vals = np.quantile(ben_scores, 1.0 - 0.02)
        threshold = float(t_vals)
    results["threshold_by_benign_fpr_0.02"] = threshold

    for evname, label in (("critical_singles_eval.jsonl", "critical"),
                          ("obfuscation_eval.jsonl", "obfuscation"),
                          ("benign_twins_eval.jsonl", "benign")):
        rows = records(evname)
        y, s = [], []
        for r in rows:
            text = r["commands"][0]["cmd"]
            res = pipe(text, classes, threshold=0.0)[0]
            vals = {rr["label"]: rr["score"] for rr in res}
            score = float(vals.get(classes[0], 0.0))
            y.append(0 if label == "benign" else 1)
            s.append(score)
        y = np.asarray(y); s = np.asarray(s)
        slice_res = {
            "n": int(len(y)),
            "roc_auc": roc_auc(y, s) if label != "benign" else None,
            "recall_at_fpr_0.02_threshold": None,
        }
        if label != "benign" and threshold is not None:
            slice_res["recall_at_fpr_0.02_threshold"] = float(np.mean(s >= threshold))
        results["slices"][label] = slice_res
        print(f"{label:10s} n={int(len(y)):5d} score_mean={s.mean():.4f} "
              f"score_p50={np.median(s):.4f}", end="")
        if slice_res["roc_auc"] is not None:
            print(f" roc_auc={slice_res['roc_auc']:.4f}", end="")
        if slice_res["recall_at_fpr_0.02_threshold"] is not None:
            print(f" recall@FPR2%={slice_res['recall_at_fpr_0.02_threshold']:.4f}", end="")
        print()

    # pooled AUC: all catastrophic positives as y=1, benign twins as y=0
    y_all, s_all = [], []
    for evname, label in (("critical_singles_eval.jsonl", 1),
                          ("obfuscation_eval.jsonl", 1),
                          ("benign_twins_eval.jsonl", 0)):
        for r in records(evname):
            text = r["commands"][0]["cmd"]
            res = pipe(text, classes, threshold=0.0)[0]
            vals = {rr["label"]: rr["score"] for rr in res}
            y_all.append(label)
            s_all.append(float(vals.get(classes[0], 0.0)))
    y_all = np.asarray(y_all); s_all = np.asarray(s_all)
    results["pooled_roc_auc"] = roc_auc(y_all, s_all)
    print(f"pooled ROC-AUC (pos vs benign twins) = {results['pooled_roc_auc']:.4f}")

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

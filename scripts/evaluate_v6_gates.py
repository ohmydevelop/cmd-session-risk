#!/usr/bin/env python3
"""v6 promotion gates on the obfuscation corpus (deterministic, fail-closed).

Gates:
  critical_single_command_recall >= 0.995  (canonical catastrophic singles)
  obfuscation_set_recall        >= 0.95    (disjoint obfuscation families)
  benign_twin_fpr               <= 0.02    (minimal-pair false alerts)

Usage:
  scripts/evaluate_v6_gates.py --artifact-dir artifacts/<run>/onnx \
      --threshold 1.179935351014905e-04 --temperature 0.7969617216679871
      --out artifacts/<run>/v6_gates.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.tokenization import CommandTokenizer  # noqa: E402

GATES = {
    "critical_single_command_recall": 0.995,
    "obfuscation_set_recall": 0.95,
    "benign_twin_fpr": 0.02,
}
FILES = {
    "critical_single_command_recall": "critical_singles_eval.jsonl",
    "obfuscation_set_recall": "obfuscation_eval.jsonl",
    "benign_twin_fpr": "benign_twins_eval.jsonl",
}


def score_file(sess, tokenizer, path: Path, temperature: float, threshold: float) -> dict:
    rows = [json.loads(row_line) for row_line in path.open() if row_line.strip()]
    decisions, probs = [], []
    for _, row in enumerate(rows, 1):
        cmds = [(c["cmd"], c.get("dt")) for c in row["commands"]]
        enc = tokenizer.encode([{"cmd": cmd, "dt": dt} for cmd, dt in cmds])
        ids = list(enc.input_ids)
        if len(ids) > 256:
            ids = ids[:256]
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
        }
        logit = float(sess.run(None, feed)[0][0])
        prob = 1.0 / (1.0 + math.exp(-logit / temperature))
        decisions.append(prob >= threshold)
        probs.append(prob)
    n = len(decisions)
    return {
        "rows": n,
        "alerts": int(sum(decisions)),
        "min_prob": float(min(probs)) if probs else None,
        "max_prob": float(max(probs)) if probs else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--eval-dir", default=str(ROOT / "data/processed/v6_obf_evals"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import onnxruntime as ort

    tok = CommandTokenizer.from_pretrained(
        args.artifact_dir, max_length=256, command_max_tokens=128
    )
    sess = ort.InferenceSession(
        f"{args.artifact_dir}/model.onnx", providers=["CPUExecutionProvider"]
    )

    results, decisions = {}, {}
    for gate_name, fname in FILES.items():
        r = score_file(sess, tok, Path(args.eval_dir) / fname, args.temperature, args.threshold)
        if r["rows"] == 0:
            decisions[gate_name] = False
            r["pass"] = False
            r["blocker"] = "empty_eval_slice"
            results[gate_name] = r
            continue
        target = GATES[gate_name]
        if gate_name == "benign_twin_fpr":
            value = r["alerts"] / r["rows"]
            passed = value <= target
        else:
            value = r["alerts"] / r["rows"]
            passed = value >= target
        r["target"] = target
        r["value"] = value
        r["pass"] = passed
        decisions[gate_name] = passed
        results[gate_name] = r

    # pooled threshold-free ranking: all catastrophic positives vs benign twins
    ss: list = []

    def _auc(y, s):
        pos = [v for v, lab in zip(s, y, strict=True) if lab == 1]
        neg = [v for v, lab in zip(s, y, strict=True) if lab == 0]
        if not pos or not neg:
            return None
        return float(
            sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
        )

    ys, ss = [], []
    for gate_name in ("critical_single_command_recall", "obfuscation_set_recall"):
        rows = [
            json.loads(row_line)
            for row_line in (Path(args.eval_dir) / FILES[gate_name]).open()
            if row_line.strip()
        ]
        for row in rows:
            cmds = [(c["cmd"], c.get("dt")) for c in row["commands"]]
            enc = tok.encode([{"cmd": cmd, "dt": dt} for cmd, dt in cmds])
            ids = list(enc.input_ids)[:256]
            feed = {
                "input_ids": np.array([ids], np.int64),
                "attention_mask": np.ones((1, len(ids)), np.int64),
            }
            logit = float(sess.run(None, feed)[0][0])
            ys.append(1)
            ss.append(1.0 / (1.0 + math.exp(-logit / args.temperature)))
    for row in [
        json.loads(row_line)
        for row_line in (Path(args.eval_dir) / FILES["benign_twin_fpr"]).open()
        if row_line.strip()
    ]:
        cmds = [(c["cmd"], c.get("dt")) for c in row["commands"]]
        enc = tok.encode([{"cmd": cmd, "dt": dt} for cmd, dt in cmds])
        ids = list(enc.input_ids)[:256]
        feed = {
            "input_ids": np.array([ids], np.int64),
            "attention_mask": np.ones((1, len(ids)), np.int64),
        }
        logit = float(sess.run(None, feed)[0][0])
        ys.append(0)
        ss.append(1.0 / (1.0 + math.exp(-logit / args.temperature)))
    pooled_auc = _auc(ys, ss)

    overall = all(decisions.values())
    out = {
        "gate_targets": GATES,
        "gate_results": results,
        "passed": bool(overall),
        "threshold": args.threshold,
        "temperature": args.temperature,
        "artifact_dir": args.artifact_dir,
        "pooled_roc_auc_pos_vs_benign_twins": pooled_auc,
    }
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    crit_res = results["critical_single_command_recall"]
    print(
        f"critical_single_command_recall = {crit_res['value']:.6f} "
        f"(target>={GATES['critical_single_command_recall']}) "
        f"pass={decisions['critical_single_command_recall']}"
    )
    obf_res = results["obfuscation_set_recall"]
    print(
        f"obfuscation_set_recall        = {obf_res['value']:.6f} "
        f"(target>={GATES['obfuscation_set_recall']}) "
        f"pass={decisions['obfuscation_set_recall']}"
    )
    print(
        f"benign_twin_fpr               = {results['benign_twin_fpr']['value']:.6f} "
        f"(target<={GATES['benign_twin_fpr']}) pass={decisions['benign_twin_fpr']}"
    )
    print(f"overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

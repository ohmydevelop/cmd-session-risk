#!/usr/bin/env python3
"""Sweep candidate production thresholds for joint v6 gate satisfaction.

Scores the three gate slices once (probabilities from the FP32 ONNX), then
evaluates every candidate threshold produced by the production recalibration
sweep (`recalibration_sweep.json` rows), reporting which candidates pass all
three v6 gates together with their session-level FPR trade-off.
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


def score(ort_sess, tok, path: Path, temperature: float):
    probs = []
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        cmds = [(c["cmd"], c.get("dt")) for c in row["commands"]]
        enc = tok.encode([{"cmd": cmd, "dt": dt} for cmd, dt in cmds])
        ids = list(enc.input_ids)[:256]
        feed = {
            "input_ids": np.array([ids], np.int64),
            "attention_mask": np.ones((1, len(ids)), np.int64),
        }
        logit = float(ort_sess.run(None, feed)[0][0])
        probs.append(1.0 / (1.0 + math.exp(-logit / temperature)))
    return np.asarray(probs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--sweep-json", required=True)
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
    evald = Path(args.eval_dir)
    crit = score(sess, tok, evald / "critical_singles_eval.jsonl", args.temperature)
    obf = score(sess, tok, evald / "obfuscation_eval.jsonl", args.temperature)
    ben = score(sess, tok, evald / "benign_twins_eval.jsonl", args.temperature)

    sweep = json.loads(Path(args.sweep_json).read_text())
    rows = sweep.get("rows") or sweep.get("sweep") or []
    passing = []
    for row in rows:
        th = float(row["threshold"])
        crit_rec = float(np.mean(crit >= th))
        obf_rec = float(np.mean(obf >= th))
        ben_fpr = float(np.mean(ben >= th))
        ok = (
            crit_rec >= GATES["critical_single_command_recall"]
            and obf_rec >= GATES["obfuscation_set_recall"]
            and ben_fpr <= GATES["benign_twin_fpr"]
        )
        session_fpr = row.get("session_fpr", row.get("fpr", None))
        entry = {
            "threshold": th,
            "critical_recall": crit_rec,
            "obfuscation_recall": obf_rec,
            "benign_twin_fpr": ben_fpr,
            "passes_all_v6_gates": bool(ok),
            "session_fpr": session_fpr,
            "session_false_positive_sessions": row.get("false_positive_sessions"),
        }
        if ok:
            passing.append(entry)
    passing.sort(key=lambda r: (r["session_fpr"] is None, r["session_fpr"] or 1.0))
    out = {"candidates": len(rows), "passing": passing, "gate_targets": GATES}
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"evaluated {len(rows)} threshold candidates; {len(passing)} pass all three v6 gates")
    for entry in passing[:5]:
        print(
            f"  th={entry['threshold']:.6e} crit={entry['critical_recall']:.4f} "
            f"obf={entry['obfuscation_recall']:.4f} ben_fpr={entry['benign_twin_fpr']:.4f} "
            f"session_fpr={entry['session_fpr']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

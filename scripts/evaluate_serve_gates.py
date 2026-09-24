#!/usr/bin/env python3
"""v6 promotion gates exercised through the `cmdrisk serve` HTTP daemon.

Positive (恶意) sets                       -> recall
  critical_singles_eval.jsonl (400 rows)      gate: >= 0.995
  obfuscation_eval.jsonl     (1600 rows)      gate: >= 0.95
Negative (正常) set                         -> false-positive rate
  benign_twins_eval.jsonl    (800 rows)       gate: <= 0.02

Every row is a single-command session labeled `split: test` and disjoint from
training data.  Decisions come from the live daemon (same inference path as
the batch CLI), so this measures exactly what production would see.

Usage:
  scripts/evaluate_serve_gates.py --base-url http://127.0.0.1:8080
      [--eval-dir data/processed/v6_obf_evals]
      [--cli-binary build/cmdrisk] [--parity-sample 50]

Exit code 0 only when all three gates pass.  The optional --cli-binary bit-
parity check samples rows and asserts the daemon probability equals the
batch-CLI probability exactly (same FP32 weights + temperature).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GATES = {
    "critical_single_command_recall": 0.995,
    "obfuscation_set_recall": 0.95,
    "benign_twin_fpr": 0.020,
}
FILES = {
    "critical_single_command_recall": "critical_singles_eval.jsonl",
    "obfuscation_set_recall": "obfuscation_eval.jsonl",
    "benign_twin_fpr": "benign_twins_eval.jsonl",
}


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def post_score(base_url: str, session_id: str, commands: list) -> dict:
    payload = {"session_id": session_id, "commands": commands}
    req = urllib_request(base_url + "/v1/score", payload)
    return json.loads(req)


def urllib_request(url: str, payload: dict | None) -> bytes:
    import urllib.request

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def evaluate(base_url: str, eval_dir: Path) -> dict:
    results = {}
    for gate, filename in FILES.items():
        path = eval_dir / filename
        rows = load_rows(path)
        assert rows and all(len(r["commands"]) == 1 for r in rows), path
        alerts = 0
        rule_alerts = 0
        probs = []
        t0 = time.time()
        for row in rows:
            cmds = [
                {"cmd": c["cmd"], "dt": c.get("dt")}
                for c in row["commands"]
            ]
            resp = post_score(base_url, row["session_id"], cmds)
            if resp.get("alerted"):
                alerts += 1
                answer = resp["answers"][0]
                if answer.get("intent_classes", "-") != "-":
                    rule_alerts += 1
            probs.append(resp.get("peak_probability") or 0.0)
        dt = time.time() - t0
        n = len(rows)
        results[gate] = {
            "rows": n,
            "alerts": alerts,
            "rate": alerts / n,
            "rule_alerts": rule_alerts,
            "min_prob": min(probs),
            "max_prob": max(probs),
            "gate": GATES[gate],
            "threshold_expr": f">= {GATES[gate]:.4f}"
            if "recall" in gate
            else f"<= {GATES[gate]:.4f}",
            "passed": (alerts / n >= GATES[gate])
            if "recall" in gate
            else (alerts / n <= GATES[gate]),
            "seconds": round(dt, 1),
        }
    return results


def cli_parity_check(base_url: str, eval_dir: Path, cli_binary: str, sample: int) -> None:
    """Sample rows, score them via batch CLI, assert identical probabilities."""
    import urllib.request

    tdir = Path("/tmp/cmdrisk_parity")
    tdir.mkdir(exist_ok=True)
    rows = []
    for filename in FILES.values():
        for row in load_rows(eval_dir / filename)[: max(1, sample // 3)]:
            rows.append(row)
    rng = random.Random(20260924)
    rng.shuffle(rows)
    rows = rows[:sample]

    lines = []
    for row in rows:
        parts = []
        for c in row["commands"]:
            text = c["cmd"]
            if c.get("dt") is not None:
                text += f"\t{c['dt']:.17g}"
            parts.append(text)
        lines.append("\x1f".join(parts))

    inp = tdir / "parity.txt"
    inp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = subprocess.run(
        [cli_binary, str(inp)],
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if out.returncode != 0:
        raise SystemExit(f"CLI parity run failed: {out.stderr.strip()}")
    cli_probs = []
    for line in out.stdout.splitlines():
        fields = line.split("\t")
        assert len(fields) >= 3, line
        cli_probs.append(float(fields[2]))

    checked = 0
    for idx, row in enumerate(rows):
        cmds = [{"cmd": c["cmd"], "dt": c.get("dt")} for c in row["commands"]]
        resp = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    base_url + "/v1/score",
                    data=json.dumps({"session_id": row["session_id"], "commands": cmds}).encode(),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                ),
                timeout=30,
            ).read()
        )
        http_prob = resp["answers"][-1]["malicious_probability"]
        diff = abs(http_prob - cli_probs[idx])
        if diff > 1e-12:
            raise SystemExit(
                f"parity mismatch row {idx}: HTTP {http_prob!r} vs CLI {cli_probs[idx]!r}"
            )
        checked += 1
    print(f"  bit-parity vs batch CLI: {checked}/{checked} identical (tolerance 1e-12)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--eval-dir", default=str(ROOT / "data/processed/v6_obf_evals"))
    ap.add_argument("--cli-binary", default=str(ROOT / "build/cmdrisk"))
    ap.add_argument("--parity-sample", type=int, default=50)
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    for filename in FILES.values():
        if not (eval_dir / filename).is_file():
            print(f"missing eval file: {eval_dir / filename}", file=sys.stderr)
            print(
                "regenerate with: scripts/build_v6_obfuscation_data.py --out "
                + str(eval_dir),
                file=sys.stderr,
            )
            return 2

    results = evaluate(args.base_url, eval_dir)
    print("set                        rows  alerts  rate     gate    result")
    all_ok = True
    for gate, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        all_ok &= r["passed"]
        print(
            f"{gate:26s} {r['rows']:5d} {r['alerts']:6d} "
            f"{r['rate']:.5f}  {r['threshold_expr']:8s}  {status}"
        )
        print(
            f"    prob range [{r['min_prob']:.6g}, {r['max_prob']:.6g}]  "
            f"intent-layer alerts {r['rule_alerts']}  {r['seconds']}s"
        )
    if args.cli_binary and Path(args.cli_binary).is_file():
        cli_parity_check(args.base_url, eval_dir, args.cli_binary, args.parity_sample)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

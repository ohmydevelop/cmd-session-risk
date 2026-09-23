#!/usr/bin/env bash
# Zero-training-cost checkpoint-soup experiment for the v4 million-scale run.
#
# Why
# ---
# The v4 run finished exactly at the end of its single epoch, so the tail
# checkpoints (12000/13500/15000/16372 for L6-H256) are samples around a flat
# minimum.  Averaging their weights ("model soup", Wortsman et al. 2022) costs
# no extra training steps and no extra inference FLOPs - the deployed graph is
# byte-for-byte the same shape - yet it usually buys a small robustness gain.
#
# This script is strictly additive and fail-closed:
#   * it never writes into, renames or deletes the v4 run, its checkpoints, its
#     ONNX graphs or any earlier artifact,
#   * it materialises new `*-soup-lastN` directories next to the v4 run,
#   * model selection happens on the *dev* split only
#     (`dev_operating_point_screen.py` never reads test),
#   * the held-out test split is read exactly once, for the single dev-selected
#     winner, through the standard `recalibrate_sweep.py` procedure.
#
# Stage 1 (default) = soup + dev screen.  Stage 2 = test confirmation of the
# dev-selected winner.  Both stages are idempotent.
#
# Usage:
#   WORK=work/recal_soup bash scripts/run_v4_soup_experiment.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN="${RUN:-artifacts/cmd-session-risk-v4-l6-h256-million-20260921}"
DATA="${DATA:-data/processed/v4}"
WORK="${WORK:-work/recal_soup}"
FPR_GRID="${FPR_GRID:-0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03}"
PROD_BUDGET="${PROD_BUDGET:-0.01}"
RECALL_TOLERANCE="${RECALL_TOLERANCE:-0.0}"
VARIANTS="${VARIANTS:-soup-last3 soup-last4}"
PY="${PY:-.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK" logs
LOG="$WORK/soup.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [[ ! -f "$RUN/short_phase/model.safetensors" ]]; then
  say "ABORT: v4 run weights missing at $RUN/short_phase"
  exit 2
fi

# ------------------------------------------------------- 0) chain ordering
# This experiment is additive and CPU-bound, so it must not compete with the
# primary v4 evidence / v5 training chain.  systemd already orders the unit
# after `cmdrisk-v4v5-persistent.service`; this guard keeps the same promise
# when the script is launched by hand or after a reboot.
if [[ "${WAIT_FOR_CHAIN:-1}" == "1" ]]; then
  while pgrep -f 'run_remaining_v4_v5_persistent\.sh|run_v5_l8_million_chain\.sh|train_scale\.py' >/dev/null 2>&1; do
    say "waiting for the v4/v5 chain to finish before touching the soup screen"
    sleep 300
  done
fi

# ---------------------------------------------------------------- 1) soups
for variant in $VARIANTS; do
  out="$RUN-$variant"
  if [[ -s "$out/model.safetensors" ]]; then
    say "soup $variant exists; skipping merge"
  else
    last="${variant##*-last}"
    say "merge $variant (last $last checkpoints)"
    "$PY" scripts/merge_checkpoints.py \
      --glob "$RUN/short_phase/checkpoint-*" --last "$last" \
      --output-dir "$out" >>"$LOG" 2>&1
    rc=$?
    if [[ "$rc" -ne 0 || ! -s "$out/model.safetensors" ]]; then
      say "ABORT: merge failed for $variant rc=$rc"
      exit 3
    fi
  fi
  # The trainer only writes tokenizer files at the run root, so the merged
  # directory needs them copied before it can be loaded or exported.
  for f in tokenizer.json tokenizer_config.json command_tokenization.json; do
    if [[ -f "$RUN/short_phase/$f" && ! -f "$out/$f" ]]; then
      cp -p "$RUN/short_phase/$f" "$out/$f"
      say "sidecar $f -> $out"
    fi
  done
done

# ------------------------------------------------------------ 2) dev screen
CANDIDATE_ARGS=()
CANDIDATE_ARGS+=(--candidate "final=$RUN/short_phase/checkpoint-16372")
for variant in $VARIANTS; do
  CANDIDATE_ARGS+=(--candidate "$variant=$RUN-$variant")
done

if [[ -s "$WORK/screen.json" ]]; then
  say "dev screen exists; skipping"
else
  say "dev-only operating-point screen (no test access)"
  "$PY" scripts/dev_operating_point_screen.py \
    --run-dir "$RUN" --dev "$DATA/dev.jsonl" \
    --tokenizer-source "$RUN/short_phase" \
    "${CANDIDATE_ARGS[@]}" \
    --baseline final --fpr-grid "$FPR_GRID" \
    --recall-tolerance "$RECALL_TOLERANCE" \
    --cache-dir "$WORK/logits" --output "$WORK/screen.json" \
    --batch-size 16 >>"$LOG" 2>&1
  rc=$?
  if [[ "$rc" -ne 0 || ! -s "$WORK/screen.json" ]]; then
    say "ABORT: dev screen failed rc=$rc"
    exit 4
  fi
fi

# ------------------------------------------------------- 3) choose a winner
say "selecting winner at budget fpr_$PROD_BUDGET on the dev evaluation half"
"$PY" - "$WORK/screen.json" "$PROD_BUDGET" >"$WORK/selection.json" <<'SELECT'
import json, sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
budget = float(sys.argv[2])
baseline = payload.get("baseline")
candidates = payload.get("candidates", {})
verdicts = payload.get("verdicts", {})
base_rows = {row["max_fpr"]: row for row in candidates.get(baseline, {}).get("rows", [])}
base = base_rows.get(budget)
if base is None:
    raise SystemExit(f"baseline {baseline!r} has no row for budget {budget}")

eligible = []
detail = {}
for label, verdict in verdicts.items():
    entry = next(
        (item for item in verdict["per_budget"] if item["max_fpr"] == budget), None
    )
    if entry is None:
        continue
    rows = {row["max_fpr"]: row for row in candidates[label]["rows"]}
    row = rows[budget]
    detail[label] = {
        "prefix_recall": row["prefix_recall"],
        "prefix_fpr": row["prefix_fpr"],
        "session_recall": row["session_recall"],
        "session_fpr": row["session_fpr"],
        "prefix_recall_delta": entry["prefix_recall_delta"],
        "prefix_fpr_delta": entry["prefix_fpr_delta"],
        "session_recall_delta": entry["session_recall_delta"],
        "session_fpr_delta": entry["session_fpr_delta"],
        "dominates_baseline": entry["dominates_baseline"],
    }
    if entry["dominates_baseline"]:
        eligible.append((row["session_fpr"], -row["session_recall"], label))

eligible.sort()
winner = eligible[0][2] if eligible else None
print(
    json.dumps(
        {
            "budget": budget,
            "baseline": baseline,
            "baseline_metrics": base,
            "candidates": detail,
            "winner": winner,
            "reason": (
                "no dev-regressing candidate"
                if winner is None
                else f"{winner} is no worse than the incumbent on the dev "
                "evaluation half and has the best own-session-FPR tie-break"
            ),
        },
        indent=2,
        sort_keys=True,
    )
)
SELECT
say "selection written to $WORK/selection.json"

WINNER="$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['winner'] or '')" "$WORK/selection.json")"
if [[ -z "$WINNER" ]]; then
  say "NO WINNER: checkpoint soup does not beat the v4 final checkpoint on dev"
  say "SOUP_EXPERIMENT_DONE"
  exit 0
fi
say "dev winner: $WINNER"

# ------------------------------------------------- 4) held-out confirmation
CONFIRM="$WORK/$WINNER/test_sweep"
if [[ -s "$CONFIRM/sweep.json" ]]; then
  say "test confirmation exists for $WINNER; skipping"
else
  say "held-out confirmation for $WINNER (single read of test, standard procedure)"
  "$PY" scripts/recalibrate_sweep.py \
    --run-dir "$RUN" --checkpoint "$RUN-$WINNER" \
    --dev "$DATA/dev.jsonl" --test "$DATA/test.jsonl" \
    --output-dir "$CONFIRM" --cache-dir "$WORK/$WINNER/logits" \
    --calibration-fraction 1.0 --fpr-grid "$FPR_GRID" \
    --batch-size 16 >>"$LOG" 2>&1
  rc=$?
  if [[ "$rc" -ne 0 || ! -s "$CONFIRM/sweep.json" ]]; then
    say "ABORT: test confirmation failed for $WINNER rc=$rc"
    exit 5
  fi
fi
say "SOUP_EXPERIMENT_DONE winner=$WINNER"

#!/usr/bin/env bash
# Persistent, idempotent remaining chain for v4 evidence + v5 L8-H256.
#
# Work products live under the repository (not /tmp) because this host has
# rebooted during long CPU jobs.  Every stage skips completed outputs and the
# flat-score row checks discard truncated files before re-scoring.
set -uo pipefail

ROOT=/home/ubuntu/Works/llm/cmd-session-risk
cd "$ROOT"

WORK_V4="${WORK_V4:-$ROOT/work/recal_v4}"
WORK_V5="${WORK_V5:-$ROOT/work/recal_v5}"
LOG="$ROOT/logs/remaining_v4_v5_chain.log"
mkdir -p "$WORK_V4" "$WORK_V5" "$ROOT/logs"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export TOKENIZERS_PARALLELISM=false
export LATENCY_LENGTHS="${LATENCY_LENGTHS:-256}"
export INT8_VALIDATOR="${INT8_VALIDATOR:-scripts/validate_int8_fast.py}"

say "persistent chain start (pid=$$)"

if grep -q 'STAGE_B_EVIDENCE_DONE' "$WORK_V4/posttrain_b.log" 2>/dev/null; then
  say "v4 Stage B evidence already complete; skipping v4"
else
  if [[ ! -s "$WORK_V4/sweep/sweep.json" ]]; then
    say "v4 Stage A start"
    WORK="$WORK_V4" STAGE=a bash scripts/run_v4_posttrain.sh
    rc=$?
    if [[ "$rc" -ne 0 || ! -s "$WORK_V4/sweep/sweep.json" ]]; then
      say "ABORT: v4 Stage A failed rc=$rc or sweep missing"
      exit 10
    fi
    say "v4 Stage A verified"
  else
    say "v4 Stage A already complete; skipping"
  fi

  say "v4 Stage B evidence start"
  WORK="$WORK_V4" bash scripts/run_v4_stage_b_evidence.sh
  rc=$?
  if [[ "$rc" -ne 0 ]] || ! grep -q 'STAGE_B_EVIDENCE_DONE' "$WORK_V4/posttrain_b.log" 2>/dev/null; then
    say "ABORT: v4 Stage B failed rc=$rc or marker missing"
    exit 20
  fi
  say "v4 Stage B evidence verified"
fi

if grep -q 'V5_CHAIN_DONE' "$WORK_V5/chain.log" 2>/dev/null; then
  say "v5 chain already complete; nothing to do"
  exit 0
fi

say "v5 L8-H256 chain start"
WORK="$WORK_V5" bash scripts/run_v5_l8_million_chain.sh
rc=$?
if [[ "$rc" -ne 0 ]] || ! grep -q 'V5_CHAIN_DONE' "$WORK_V5/chain.log" 2>/dev/null; then
  say "ABORT: v5 chain failed rc=$rc or marker missing"
  exit 30
fi

say "all remaining work complete"

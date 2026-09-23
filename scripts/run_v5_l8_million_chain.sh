#!/usr/bin/env bash
# v5 model-ladder chain: L8-H256 on the *identical* v4 million-scale corpus.
#
# Why this exists
# ---------------
# v4 (L6-H256) is the production candidate.  The one open capacity question is
# whether one extra transformer layer buys anything at the same operating
# point.  The ladder was sized ahead of time from exact architecture-only
# params/FLOPs (see reports/PRODUCTION_READINESS_V5_zh.md):
#
#   variant          params      vs L6    FLOPs vs L6
#   L4-H256-A4      11,180,571   0.88x    0.67x
#   L6-H256-A4      12,760,091   1.00x    1.00x   <- v4
#   L8-H256-A4      14,339,611   1.12x    1.33x   <- this run
#   L12-H256-A4     17,498,651   1.37x    2.00x
#   L6-H384-A6      22,728,219   1.78x    2.25x
#
# L8 is the largest variant that still leaves ~2x head-room under the 50ms p95
# deployment gate; L12/L6-H384 push p95 to ~37-42ms, i.e. against the gate,
# with no guaranteed accuracy return.
#
# The corpus is deliberately NOT rebuilt: train/dev/test are the same
# `data/processed/v4/*.jsonl` files (same family-disjoint splits, same cap32
# prefix selection), so v4 vs v5 isolates *capacity* and nothing else.
#
# Everything is additive.  Nothing here mutates, overwrites or deletes the v3
# or v4 artefacts.  Each stage is idempotent and re-runnable.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN="${RUN:-artifacts/cmd-session-risk-v5-l8-h256-million-20260921}"
DATA="${DATA:-data/processed/v4}"
WORK="${WORK:-/tmp/recal_v5}"
MODEL_NAME="${MODEL_NAME:-}"
LATENCY_LENGTHS="${LATENCY_LENGTHS:-256}"
MODEL_VERSION="${MODEL_VERSION:-cmd-session-risk-v5-l8-h256-million-20260921}"
CELL="${CELL:-$WORK/sweep/fpr_0.01}"
PROD="${PROD:-$RUN-recal-prod-20260921}"
LOG="${LOG:-logs/train_v5_l8h256_million.log}"
PY="${PY:-.venv/bin/python}"

mkdir -p logs "$WORK"
export LATENCY_LENGTHS
export INT8_VALIDATOR="${INT8_VALIDATOR:-scripts/validate_int8_fast.py}"
if [[ -z "$MODEL_NAME" ]]; then
  if [[ -f "data/raw/pretrained/bert_uncased_L-8_H-256_A-4/pytorch_model.bin" ]]; then
    MODEL_NAME="$PWD/data/raw/pretrained/bert_uncased_L-8_H-256_A-4"
  else
    MODEL_NAME="google/bert_uncased_L-8_H-256_A-4"
  fi
fi
say() { echo "[$(date +%T)] $*" | tee -a "$WORK/chain.log"; }

if [[ -s "$RUN/metrics.json" ]]; then
  say "train skipped (metrics.json present in $RUN)"
else
  resume_args=()
  if [[ -d "$RUN/short_phase" ]]; then
    latest_checkpoint="$(find "$RUN/short_phase" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V | tail -1)"
    if [[ -n "$latest_checkpoint" ]]; then
      resume_args=(--override "resume_from_checkpoint=$RUN/short_phase/$latest_checkpoint")
      say "TRAIN resume from $latest_checkpoint"
    fi
  fi
  say "TRAIN $MODEL_NAME -> $RUN"
  "$PY" scripts/train_scale.py \
    --data "$DATA" --output-dir "$RUN" --model-name "$MODEL_NAME" \
    --max-epochs 1 --short-max-length 256 --command-max-tokens 128 \
    --max-prefixes 32 --skip-long-phase --vram-gb 40 \
    --micro-batch-size 32 --gradient-accumulation-steps 2 \
    --save-steps 1500 --save-total-limit 4 --logging-steps 50 \
    --dataloader-num-workers 2 \
    --override load_best_model_at_end=false "${resume_args[@]}" >>"$LOG" 2>&1
  rc=$?
  say "train rc=$rc"
  [[ -s "$RUN/metrics.json" ]] || { say "ABORT: training produced no metrics.json"; exit 2; }
fi

say "STAGE A (export -> scoring -> sweep) for v5"
RUN="$RUN" DATA="$DATA" WORK="$WORK" MODEL_VERSION="$MODEL_VERSION" \
  STAGE=a bash scripts/run_v4_posttrain.sh >>"$WORK/chain.stage_a.log" 2>&1
rc=$?
say "stage A rc=$rc log=$WORK/chain.stage_a.log"
if [[ "$rc" -ne 0 || ! -s "$WORK/sweep/sweep.json" ]]; then
  say "ABORT: sweep.json missing after Stage A"; exit 3
fi

say "STAGE B (recalibrated prod artefact + evidence) for v5"
RUN="$RUN" DATA="$DATA" WORK="$WORK" CELL="$CELL" PROD="$PROD" \
  MODEL_VERSION="$MODEL_VERSION" \
  bash scripts/run_v4_stage_b_evidence.sh >>"$WORK/chain.stage_b.log" 2>&1
rc=$?
say "stage B rc=$rc prod=$PROD"
if [[ "$rc" -ne 0 || ! -s "$PROD/onnx/export_manifest.json" ]]; then
  say "ABORT: v5 Stage B failed rc=$rc or manifest missing"
  exit 4
fi
say "V5_CHAIN_DONE"

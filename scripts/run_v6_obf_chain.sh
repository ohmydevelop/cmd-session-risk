#!/usr/bin/env bash
# v6 chain: continue-train from v5 on the obfuscation-augmented corpus,
# reuse the v4/v5 posttrain stages unchanged, then apply the NEW v6 gates
# (critical recall / obfuscation recall / benign-twin FPR).
#
# Everything is additive: no v3/v4/v5 artifact is touched; the run is
# idempotent and re-runnable (each stage resumes where it left off).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN="${RUN:-artifacts/cmd-session-risk-v6-l8-h256-obf-20260923}"
DATA="${DATA:-data/processed/v6_obf}"
EVAL_DIR="${EVAL_DIR:-data/processed/v6_obf_evals}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922/short_phase}"
MODEL_DIR="${MODEL_DIR:-artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922/short_phase}"
WORK="${WORK:-/tmp/recal_v6}"
MODEL_VERSION="${MODEL_VERSION:-cmd-session-risk-v6-l8-h256-obf-20260923}"
CELL="${CELL:-$WORK/sweep/fpr_0.01}"
PROD="${PROD:-$RUN-recal-prod-session1p-20260923}"
LOG="${LOG:-logs/train_v6_obf.log}"
PY="${PY:-.venv/bin/python}"

mkdir -p logs "$WORK"
say() { echo "[$(date +%T)] $*" | tee -a "$WORK/chain.log"; }

if [[ -s "$RUN/metrics.json" ]]; then
  say "train skipped (metrics.json present in $RUN)"
else
  say "TRAIN base=$BASE_CHECKPOINT data=$DATA -> $RUN"
  "$PY" scripts/train_scale.py \
    --data "$DATA" --output-dir "$RUN" \
    --model-name "$MODEL_DIR" --base-checkpoint "$BASE_CHECKPOINT" \
    --max-epochs 1 --short-max-length 256 --command-max-tokens 128 \
    --max-prefixes 32 --skip-long-phase --vram-gb 40 \
    --micro-batch-size 32 --gradient-accumulation-steps 2 \
    --save-steps 1500 --save-total-limit 4 --logging-steps 50 \
    --dataloader-num-workers 2 \
    --override load_best_model_at_end=false >>"$LOG" 2>&1
  rc=$?
  say "train rc=$rc"
  [[ -s "$RUN/metrics.json" ]] || { say "ABORT: training produced no metrics.json"; exit 2; }
fi

say "STAGE A (export -> scoring -> sweep)"
RUN="$RUN" DATA="$DATA" WORK="$WORK" MODEL_VERSION="$MODEL_VERSION" \
  STAGE=a bash scripts/run_v4_posttrain.sh >>"$WORK/chain.stage_a.log" 2>&1
rc=$?
say "stage A rc=$rc"
[[ "$rc" -eq 0 && -s "$WORK/sweep/sweep.json" ]] || { say "ABORT: sweep.json missing"; exit 3; }

say "STAGE B (recalibrated prod artifact + evidence)"
RUN="$RUN" DATA="$DATA" WORK="$WORK" CELL="$CELL" PROD="$PROD" \
  MODEL_VERSION="$MODEL_VERSION" \
  bash scripts/run_v4_stage_b_evidence.sh >>"$WORK/chain.stage_b.log" 2>&1
rc=$?
say "stage B rc=$rc"
[[ "$rc" -eq 0 && -s "$PROD/onnx/export_manifest.json" ]] || { say "ABORT: stage B incomplete"; exit 4; }

say "STAGE C (v6 gates: critical/obfuscation/benign-twin)"
TH="$("$PY" - "$PROD/onnx/calibration.json" <<'PY' 2>/dev/null || echo 0.0001179935351014905
import json,sys
c=json.load(open(sys.argv[1]))
print(c["malicious_threshold"])
PY
)"
TEMP="$("$PY" - "$PROD/onnx/calibration.json" <<'PY' 2>/dev/null || echo 0.7969617216679871
import json,sys
c=json.load(open(sys.argv[1]))
print(c["malicious_temperature"])
PY
)"
"$PY" scripts/evaluate_v6_gates.py \
  --artifact-dir "$PROD/onnx" \
  --threshold "$TH" --temperature "$TEMP" \
  --eval-dir "$EVAL_DIR" \
  --out "$PROD/v6_gates.json" >>"$WORK/chain.stage_c.log" 2>&1
rc=$?
say "stage C rc=$rc"
cp "$EVAL_DIR"/*.jsonl "$PROD/" 2>/dev/null || true
if [[ "$rc" -ne 0 ]]; then
  say "ABORT: v6 gates FAILED (see $PROD/v6_gates.json)"
  exit 5
fi
say "V6_CHAIN_DONE prod=$PROD"

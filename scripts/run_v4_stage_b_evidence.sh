#!/usr/bin/env bash
# Stage B + evidence collection for the million-scale v4 run.
#
# Waits for Stage A (export -> FP32/INT8 flat scoring -> recalibration sweep)
# to finish, then:
#   1. materialises the recalibrated *production* artifact from the chosen
#      sweep cell (default fpr_0.01, the same budget used by the v3 prod run so
#      the v3<->v4 comparison stays apples-to-apples),
#   2. re-evaluates the already-scored ONNX prefixes at the prod temperature /
#      threshold (prefix + session operating point, FP32 vs INT8),
#   3. runs the deployment-aligned session operating point sweep,
#   4. validates INT8 vs FP32 on the full v4 test split,
#   5. re-measures 512-token latency on the frozen graphs,
#   6. refreshes the manifest LAST so every evidence file is covered.
#
# Every step is skipped when its output already exists, so the script can be
# re-run after an interruption.  Nothing here mutates v3 artefacts, the v4
# trained run, or the v4 sweep directory.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN="${RUN:-artifacts/cmd-session-risk-v4-l6-h256-million-20260921}"
DATA="${DATA:-data/processed/v4}"
WORK="${WORK:-/tmp/recal_v4}"
CELL="${CELL:-$WORK/sweep/fpr_0.01}"
PROD="${PROD:-$RUN-recal-prod-20260921}"
MODEL_VERSION="${MODEL_VERSION:-cmd-session-risk-v4-l6-h256-million-20260921}"
RECAL_SUFFIX="${RECAL_SUFFIX:--recal-prod-20260921}"
SESSION_FPR_GRID="${SESSION_FPR_GRID:-0.0025,0.005,0.0075,0.01,0.015,0.02,0.03}"
PY="${PY:-.venv/bin/python}"
LATENCY_LENGTHS="${LATENCY_LENGTHS:-512}"
INT8_VALIDATOR="${INT8_VALIDATOR:-scripts/validate_int8.py}"
FAILED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK" logs
LOG="$WORK/posttrain_b.log"
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
mark_failure() { FAILED=1; }

echo "[$(date +%T)] stage-b driver start (run=$RUN cell=$CELL)" >> "$LOG"

# ---------------------------------------------------------------- gate on A
say "waiting for Stage A (sweep.json)"
while [[ ! -s "$WORK/sweep/sweep.json" ]]; do
  if ! pgrep -f "run_v4_posttrain.sh|wait_and_posttrain.sh|train_scale.py" >/dev/null 2>&1; then
    # nothing is running any more: only keep waiting if a partial sweep exists
    if [[ ! -s "$WORK/sweep/sweep.json" ]]; then
      say "WARNING: no producer process found and sweep.json is missing; aborting Stage B"
      exit 3
    fi
  fi
  sleep 60
done

if [[ ! -d "$CELL" ]]; then
  say "ERROR: chosen sweep cell missing: $CELL"
  exit 2
fi

# ------------------------------------------------------- derived constants
read_temperature() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['malicious_temperature'])" "$1"; }
read_threshold()   { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['malicious_threshold'])"   "$1"; }

SRC_CAL="$RUN/calibration.json"
[[ -f "$SRC_CAL" ]] || { say "ERROR: missing $SRC_CAL"; exit 2; }
T_SOURCE="$(read_temperature "$SRC_CAL")"
T_TARGET="$(read_temperature "$CELL/calibration.json")"
THRESHOLD="$(read_threshold "$CELL/calibration.json")"
say "T_source=$T_SOURCE T_target=$T_TARGET threshold=$THRESHOLD"

# ------------------------------------------------------------- 1) prod run
if [[ ! -d "$PROD" ]]; then
  say "STEP build_recal_artifact -> $PROD"
  "$PY" scripts/build_recal_artifact.py \
    --source-run "$RUN" --cell "$CELL" --output-run "$PROD" \
    --model-version "${MODEL_VERSION}${RECAL_SUFFIX}" >>"$LOG" 2>&1
  rc=$?
  [[ "$rc" -eq 0 && -d "$PROD" ]] || mark_failure
  say "build_recal rc=$rc"
  if [[ "$rc" -eq 0 && -s "$PROD/onnx/export_manifest.json" ]]; then
    "$PY" scripts/refresh_artifact_manifest.py \
      --artifact-dir "$PROD/onnx" --int8-model model.int8.onnx >>"$LOG" 2>&1
    say "post-build manifest refresh rc=$?"
  fi
else
  say "build_recal skipped (exists)"
fi

# ------------------------------------------------------- 2) prefix/session eval
PROD_EVAL="$PROD/recalibrated_onnx_evaluation.json"
if [[ ! -s "$PROD_EVAL" ]]; then
  say "STEP evaluate_recalibrated_onnx_scores"
  "$PY" scripts/evaluate_recalibrated_onnx_scores.py \
    --run-dir "$PROD" \
    --dev-data "$DATA/dev.jsonl" --test-data "$DATA/test.jsonl" \
    --fp32-dev "$WORK/dev_fp32.jsonl" --fp32-test "$WORK/test_fp32.jsonl" \
    --int8-dev "$WORK/dev_int8.jsonl" --int8-test "$WORK/test_int8.jsonl" \
    --source-temperature "$T_SOURCE" --target-temperature "$T_TARGET" \
    --threshold "$THRESHOLD" --output "$WORK/prod_onnx_evaluation.json" >>"$LOG" 2>&1
  rc=$?
  [[ "$rc" -eq 0 && -s "$WORK/prod_onnx_evaluation.json" ]] || mark_failure
  [[ -s "$WORK/prod_onnx_evaluation.json" ]] && {
    cp -f "$WORK/prod_onnx_evaluation.json" "$PROD_EVAL"
    cp -f "$WORK/prod_onnx_evaluation.json" "$PROD/onnx/recalibrated_onnx_evaluation.json"
  }
  say "evaluate rc=$rc"
else
  say "evaluate skipped (exists)"
fi

# --------------------------------------------------- 3) session operating point
PROD_SESSION="$PROD/session_operating_point_sweep.json"
if [[ ! -s "$PROD_SESSION" ]]; then
  say "STEP session_operating_point_sweep"
  "$PY" scripts/session_operating_point_sweep.py \
    --run-dir "$PROD" \
    --dev-data "$DATA/dev.jsonl" --test-data "$DATA/test.jsonl" \
    --fp32-dev "$WORK/dev_fp32.jsonl" --fp32-test "$WORK/test_fp32.jsonl" \
    --source-temperature "$T_SOURCE" --target-temperature "$T_TARGET" \
    --session-fpr-grid "$SESSION_FPR_GRID" \
    --output "$WORK/prod_session_sweep.json" >>"$LOG" 2>&1
  rc=$?
  [[ "$rc" -eq 0 && -s "$WORK/prod_session_sweep.json" ]] || mark_failure
  [[ -s "$WORK/prod_session_sweep.json" ]] && cp -f "$WORK/prod_session_sweep.json" "$PROD_SESSION"
  say "session sweep rc=$rc"
else
  say "session sweep skipped (exists)"
fi

# ------------------------------------------------------------- 4) INT8 check
# Defensive: build_recal_artifact copies the whole source tree, so an
# int8_validation.json left behind by the *source* run (measured at the source
# threshold) would be inherited verbatim and would silently freeze the wrong
# operating point into the factory evidence.  Discard any inherited or
# threshold-mismatched report and recompute at the prod threshold.
INT8_VAL="$PROD/onnx/int8_validation.json"
if [[ -s "$INT8_VAL" ]]; then
  verdict="$("$PY" - "$INT8_VAL" "$PROD/onnx" "$THRESHOLD" <<'INT8CHECK'
import json, os, sys

path, artifact, threshold = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    data = json.loads(open(path, encoding="utf-8").read())
except Exception:
    print("unreadable")
    raise SystemExit(0)
if not isinstance(data, dict):
    print("not_an_object")
    raise SystemExit(0)
try:
    got = float(data.get("threshold"))
except Exception:
    print("threshold_missing")
    raise SystemExit(0)
if abs(got - float(threshold)) > 1e-9:
    print("threshold_mismatch")
    raise SystemExit(0)
recorded = data.get("artifact_dir")
if recorded and os.path.realpath(recorded) != os.path.realpath(artifact):
    print("artifact_dir_mismatch")
    raise SystemExit(0)
print("ok")
INT8CHECK
)"
  if [[ "$verdict" != "ok" ]]; then
    say "discarding inherited int8_validation.json ($verdict; expected threshold=$THRESHOLD)"
    rm -f "$INT8_VAL"
  fi
fi
if [[ ! -s "$INT8_VAL" ]]; then
  say "STEP validate_int8 (prod, full prefix)"
  "$PY" "$INT8_VALIDATOR" \
    --artifact-dir "$PROD/onnx" --input "$DATA/test.jsonl" >>"$LOG" 2>&1
  rc=$?
  [[ "$rc" -eq 0 && -s "$INT8_VAL" ]] || mark_failure
  say "validate_int8 rc=$rc"
else
  say "validate_int8 skipped (exists, threshold verified)"
fi

# ---------------------------------------------------------------- 5) latency
if [[ ! -s "$PROD/onnx/latency_report.json" ]]; then
  say "STEP latency_bench (${LATENCY_LENGTHS} tokens)"
  "$PY" scripts/latency_bench.py --artifact-dir "$PROD/onnx" \
    --lengths "$LATENCY_LENGTHS" --warmup 10 --iterations 50 --format json \
    --output "$PROD/onnx/latency_report.json" --update-manifest >>"$LOG" 2>&1
  rc=$?
  [[ "$rc" -eq 0 && -s "$PROD/onnx/latency_report.json" ]] || mark_failure
  say "latency rc=$rc"
else
  say "latency skipped (exists)"
fi

# ------------------------------------------------------- 6) sweep provenance
if [[ -s "$WORK/sweep/sweep.json" ]]; then
  cp -f "$WORK/sweep/sweep.json" "$PROD/onnx/recalibration_sweep.json"
  cp -f "$WORK/sweep/sweep.json" "$PROD/recalibration_sweep.json"
fi

# ------------------------------------------------- 7) manifest refresh last
say "STEP manifest refresh (last)"
"$PY" scripts/refresh_artifact_manifest.py \
  --artifact-dir "$PROD/onnx" --int8-model model.int8.onnx >>"$LOG" 2>&1
rc=$?
[[ "$rc" -eq 0 && -s "$PROD/onnx/export_manifest.json" ]] || mark_failure
say "manifest rc=$rc"

say "SUMMARY prod=$PROD cell=$CELL T_source=$T_SOURCE T_target=$T_TARGET threshold=$THRESHOLD"
if [[ "$FAILED" -ne 0 ]]; then
  say "STAGE_B_EVIDENCE_FAILED"
  exit 1
fi
say "STAGE_B_EVIDENCE_DONE"

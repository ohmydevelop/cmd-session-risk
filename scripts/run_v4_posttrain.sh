#!/usr/bin/env bash
# Post-training chain for the million-scale v4 run.
#
# Stage A (default) runs the deterministic half once training has finished:
#   export FP32+INT8 ONNX -> FP32/INT8 flat scoring -> recalibration sweep
#   -> INT8 agreement -> 512-token latency -> manifest refresh.
# Stage B (STAGE=b, CELL=<path>) materialises the chosen recalibrated
# production artifact and the final evidence bundle.
#
# Every step is skipped when its output already exists, so the script is
# re-runnable after an interruption.  Nothing here mutates the v3 artefacts.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

RUN="${RUN:-artifacts/cmd-session-risk-v4-l6-h256-million-20260921}"
DATA="${DATA:-data/processed/v4}"
WORK="${WORK:-/tmp/recal_v4}"
MODEL_VERSION="${MODEL_VERSION:-cmd-session-risk-v4-l6-h256-million-20260921}"
STAGE="${STAGE:-a}"
CELL="${CELL:-}"
FPR_GRID="${FPR_GRID:-0.005,0.0075,0.01,0.0125,0.015,0.0175,0.02,0.025,0.03}"
PROXY_AUDIT="${PROXY_AUDIT:-}"
LATENCY_LENGTHS="${LATENCY_LENGTHS:-512}"
PY="${PY:-.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$WORK" logs
LOG="$WORK/posttrain_${STAGE}.log"
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
step() { say "STEP $*"; }

# wait for an in-flight training process to exit
wait_for_children() {
  local pattern="$1"
  while pgrep -f "$pattern" >/dev/null 2>&1; do sleep 60; done
}

# Flat scoring emits exactly one row per command in the input session file.
# That identity is what makes "did this stream actually finish?" checkable, so a
# job that dies half-way can never be mistaken for a finished one: `-s` alone
# only proves a file is non-empty, and silently scoring on a truncated stream
# would corrupt every downstream number.
count_commands() {
  "$PY" - "$1" <<'COUNTCMD'
import json, sys

total = 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        total += len(json.loads(line).get("commands", []))
print(total)
COUNTCMD
}

# Drop a score file whose row count does not match the input, so the parallel
# block below re-scores it instead of skipping it.
prune_incomplete() {
  local path="$1" expected="$2" label="$3"
  [[ -e "$path" ]] || return 0
  local got
  got="$(wc -l < "$path" 2>/dev/null || echo 0)"
  if [[ "$got" != "$expected" ]]; then
    say "discarding incomplete ${label} scores: $path ($got rows, expected $expected)"
    rm -f "$path"
  fi
}

stage_a() {
  step "export ONNX (FP32+INT8)"
  if [[ ! -f "$RUN/onnx/model.onnx" ]]; then
    "$PY" kaggle/export_onnx.py \
      --checkpoint "$RUN/short_phase" \
      --output-dir "$RUN/onnx" \
      --tokenizer-source "$RUN/short_phase" \
      --calibration "$RUN/calibration.json" \
      --metrics "$RUN/metrics.json" \
      --model-version "$MODEL_VERSION" >>"$LOG" 2>&1
    say "export rc=$?"
  else
    say "export skipped (exists)"
  fi

  local EXPECTED_DEV EXPECTED_TEST
  EXPECTED_DEV="$(count_commands "$DATA/dev.jsonl")"
  EXPECTED_TEST="$(count_commands "$DATA/test.jsonl")"
  say "expected flat rows: dev=$EXPECTED_DEV test=$EXPECTED_TEST"
  prune_incomplete "$WORK/dev_fp32.jsonl"  "$EXPECTED_DEV"  "dev_fp32"
  prune_incomplete "$WORK/test_fp32.jsonl" "$EXPECTED_TEST" "test_fp32"
  prune_incomplete "$WORK/dev_int8.jsonl"  "$EXPECTED_DEV"  "dev_int8"
  prune_incomplete "$WORK/test_int8.jsonl" "$EXPECTED_TEST" "test_int8"

  # The four flat-scoring passes are independent, so run them concurrently.
  #
  # Thread budget matters more than raw core count here.  ORT sizes its intra-op
  # pool from the physical core count when left unpinned, so four un-pinned jobs
  # spawn 40 spinning threads on a 10-core box and collapse to roughly serial
  # throughput.  Measured on this host (1905 prefixes, 4-way parallel):
  #   default (4x10t) 60s | 4x4t 39s | 4x3t 34s | 4x2t 31s | 1x10t 53s
  # A single job also runs just as fast on 3 threads as on 10, i.e. this small
  # model barely scales intra-op -- the win comes purely from packing distinct
  # jobs onto distinct cores.  4x2 = 8 threads leaves head-room for the OS.
  # Every configuration produced byte-identical score JSONL.
  local SCORE_THREADS="${SCORE_THREADS:-2}"
  step "flat scoring FP32/INT8 dev/test (4-way parallel, ${SCORE_THREADS} intra-op threads each)"
  local pids=()
  if [[ ! -s "$WORK/dev_fp32.jsonl" ]]; then
    "$PY" scripts/score_fp32.py \
      --artifact-dir "$RUN/onnx" --input "$DATA/dev.jsonl" \
      --output "$WORK/dev_fp32.jsonl" --batch-size 16 \
      --chunk-sessions 64 --intra-op-threads "$SCORE_THREADS" >>"$LOG.dev_fp32" 2>&1 &
    pids+=($!)
  fi
  if [[ ! -s "$WORK/test_fp32.jsonl" ]]; then
    "$PY" scripts/score_fp32.py \
      --artifact-dir "$RUN/onnx" --input "$DATA/test.jsonl" \
      --output "$WORK/test_fp32.jsonl" --batch-size 16 \
      --chunk-sessions 64 --intra-op-threads "$SCORE_THREADS" >>"$LOG.test_fp32" 2>&1 &
    pids+=($!)
  fi
  if [[ ! -s "$WORK/dev_int8.jsonl" ]]; then
    "$PY" -m cmdrisk.cli score \
      --artifact-dir "$RUN/onnx" --input "$DATA/dev.jsonl" \
      --output "$WORK/dev_int8.jsonl" --flat --batch-size 16 \
      --intra-op-threads "$SCORE_THREADS" >>"$LOG.dev_int8" 2>&1 &
    pids+=($!)
  fi
  if [[ ! -s "$WORK/test_int8.jsonl" ]]; then
    "$PY" -m cmdrisk.cli score \
      --artifact-dir "$RUN/onnx" --input "$DATA/test.jsonl" \
      --output "$WORK/test_int8.jsonl" --flat --batch-size 16 \
      --intra-op-threads "$SCORE_THREADS" >>"$LOG.test_int8" 2>&1 &
    pids+=($!)
  fi
  local rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  say "flat scoring done rc=$rc"

  # Fail closed: a crashed stream must not be silently accepted.
  local bad=0
  check_rows() {
    local path="$1" expected="$2" label="$3" got
    got="$(wc -l < "$path" 2>/dev/null || echo 0)"
    if [[ "$got" != "$expected" ]]; then
      say "ERROR: ${label} scores incomplete: $path ($got rows, expected $expected)"
      bad=1
      rm -f "$path"
    fi
  }
  check_rows "$WORK/dev_fp32.jsonl"  "$EXPECTED_DEV"  "dev_fp32"
  check_rows "$WORK/test_fp32.jsonl" "$EXPECTED_TEST" "test_fp32"
  check_rows "$WORK/dev_int8.jsonl"  "$EXPECTED_DEV"  "dev_int8"
  check_rows "$WORK/test_int8.jsonl" "$EXPECTED_TEST" "test_int8"
  if [[ "$bad" != "0" ]]; then
    say "ABORT: flat scoring incomplete; re-run STAGE=a to redo the missing streams"
    return 2
  fi
  say "flat scoring verified (4/4 streams complete)"

  step "recalibration sweep"
  local audit_args=()
  [[ -n "$PROXY_AUDIT" ]] && audit_args=(--proxy-audit "$PROXY_AUDIT")
  if [[ ! -s "$WORK/sweep/sweep.json" ]]; then
    "$PY" scripts/recalibrate_sweep.py \
      --run-dir "$RUN" --dev "$DATA/dev.jsonl" --test "$DATA/test.jsonl" \
      "${audit_args[@]}" \
      --output-dir "$WORK/sweep" --cache-dir "$WORK/logits" \
      --calibration-fraction 1.0 --fpr-grid "$FPR_GRID" \
      --batch-size 16 >>"$LOG" 2>&1
    say "sweep rc=$?"
  else
    say "sweep skipped (exists)"
  fi

  # Do NOT write onnx/int8_validation.json into the source run by default.
  # build_recal_artifact.py copies small files (it only hardlinks >=5MB), so a
  # source-threshold int8_validation.json produced here would be inherited by
  # the -recal-prod artifact and Stage B would then skip recomputing it at the
  # production threshold -- i.e. the shipped evidence would silently be
  # measured at the wrong operating point.  The byte-identical INT8 graph is
  # validated in Stage B against the production threshold instead.  Set
  # SOURCE_INT8_VALIDATION=1 to also measure it here (diagnostic only).
  if [[ "${SOURCE_INT8_VALIDATION:-0}" == "1" ]]; then
    step "INT8 agreement on test (full prefix, source threshold, diagnostic)"
    [[ -s "$RUN/onnx/int8_validation.json" ]] || "$PY" scripts/validate_int8.py \
      --artifact-dir "$RUN/onnx" --input "$DATA/test.jsonl" >>"$LOG" 2>&1
    say "validate_int8 rc=$?"
  else
    say "source-threshold INT8 validation skipped (Stage B validates PROD)"
  fi

  step "manifest refresh"
  "$PY" scripts/refresh_artifact_manifest.py \
    --artifact-dir "$RUN/onnx" --int8-model model.int8.onnx >>"$LOG" 2>&1
  say "manifest rc=$?"

  step "${LATENCY_LENGTHS}-token latency"
  "$PY" scripts/latency_bench.py --artifact-dir "$RUN/onnx" \
    --lengths "$LATENCY_LENGTHS" --warmup 10 --iterations 50 --format json \
    --output "$RUN/onnx/latency_report.json" --update-manifest >>"$LOG" 2>&1
  say "latency rc=$?"

  step "final manifest refresh"
  "$PY" scripts/refresh_artifact_manifest.py \
    --artifact-dir "$RUN/onnx" --int8-model model.int8.onnx >>"$LOG" 2>&1
  say "STAGE_A_DONE"
}

stage_b() {
  if [[ -z "$CELL" ]]; then echo "CELL=<sweep cell dir> is required" >&2; exit 2; fi
  local PROD="${PROD:-$RUN-recal-prod-20260921}"
  local RECAL="${RECAL:-$WORK/recal-20260921}"

  step "build recalibrated artifact -> $PROD"
  "$PY" scripts/build_recal_artifact.py \
    --source-run "$RUN" --cell "$CELL" --output-run "$PROD" \
    --model-version "${MODEL_VERSION}-recal-prod-20260921" \
    ${FORCE:+--force} >>"$LOG" 2>&1
  say "build_recal rc=$?"
  say "STAGE_B_DONE"
}

# ---------------------------------------------------------------------------
# Dispatch.  Without this block the script only defines the stages and exits 0,
# which can look like a successful Stage A while producing no evidence at all.
case "$STAGE" in
  a|A)
    stage_a
    if [[ ! -s "$WORK/sweep/sweep.json" ]]; then
      say "ERROR: Stage A finished without $WORK/sweep/sweep.json"
      exit 1
    fi
    ;;
  b|B)
    stage_b
    ;;
  *)
    echo "STAGE must be 'a' or 'b' (got '$STAGE')" >&2
    exit 2
    ;;
esac

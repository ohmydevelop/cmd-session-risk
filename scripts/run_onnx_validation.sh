#!/usr/bin/env bash
# Full held-out prefix scoring of the exported ONNX artifact on GPU.
#
# This is the exact command that produced
#   artifacts/<run>/validation_scores.jsonl  (31,467 prefix rows)
# whose metrics are recomputed independently by scripts/evaluate_validation.py
# and recorded in reports/validation_metrics.json.
#
# Run from the repository root.  On a CUDA host the NVIDIA runtime libraries
# shipped with onnxruntime-gpu must be on LD_LIBRARY_PATH.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ARTIFACT_DIR="${1:-artifacts/cmd-session-risk-l4-20260919-0020/onnx}"
INPUT="${2:-data/processed/v2/test.jsonl}"
OUTPUT="${3:-artifacts/cmd-session-risk-l4-20260919-0020/validation_scores.jsonl}"

# onnxruntime-gpu ships its CUDA/cuDNN libraries inside the wheel.
if compgen -G ".venv/lib/python3.*/site-packages/nvidia/*/lib" > /dev/null; then
  LD_LIBRARY_PATH="$(ls -d .venv/lib/python3.*/site-packages/nvidia/*/lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH
fi

date -u +"start %Y-%m-%dT%H:%M:%SZ"
.venv/bin/cmdrisk score \
  --artifact-dir "$ARTIFACT_DIR" \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --flat --batch-size 32 --max-batch-tokens 16384 \
  --provider CUDAExecutionProvider --provider CPUExecutionProvider
date -u +"done %Y-%m-%dT%H:%M:%SZ"

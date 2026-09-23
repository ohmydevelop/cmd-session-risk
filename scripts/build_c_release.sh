#!/usr/bin/env bash
# Build the redistributable C runtime package for cmd-session-risk.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${VERSION:-1.0.0}"
DIST="$ROOT/dist"
REL="$DIST/cmd-risk-c-runtime-$VERSION"
ARCH="$DIST/cmd-risk-c-runtime-$VERSION.tar.gz"

mkdir -p "$ROOT/tools/lib" "$DIST"
# Keep old releases, never rm.
if [[ -e "$REL" ]]; then
  ts="$(date +%Y%m%d-%H%M%S)"
  mv "$REL" "$REL.old-$ts"
fi
mkdir -p "$REL"/{bin,lib,model,etc,licenses,docs}

cp "$ROOT/LICENSE" "$REL/licenses/LICENSE"
cp "$ROOT/tools/third_party/onnxruntime_c_api.h" "$REL/licenses/onnxruntime_c_api.h"
# note: ORT header is MIT-licensed from microsoft/onnxruntime v1.30.0

# Model bundle for the session1p operating point.
MODEL_SRC="$ROOT/artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922/onnx"
for f in model.int8.onnx calibration.json labels.json runtime_config.json tokenizer_config.json command_tokenization.json int8_validation.json; do
  cp "$MODEL_SRC/$f" "$REL/model/$f"
done
cp "$ROOT/data/raw/pretrained/bert_uncased_L-8_H-256_A-4/vocab.txt" "$REL/model/vocab.txt"

# Runtime shared libraries.
cp "$ROOT/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.30.0" "$REL/lib/libonnxruntime.so.1"
cp "$ROOT/.venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime_providers_shared.so" "$REL/lib/"

# Rebuild with relative origin so the customer can relocate the package.
cc -O2 -I "$ROOT/tools/third_party" "$ROOT/tools/onnx_int8_c.c" \
   "$ROOT/tools/lib/libonnxruntime.so.1" \
   -Wl,-rpath,'$ORIGIN/../lib' -lm -o "$REL/bin/onnx_int8_c"
cc -O2 -I "$ROOT/tools/third_party" "$ROOT/tools/cmd_pipeline_c.c" \
   "$ROOT/tools/lib/libonnxruntime.so.1" \
   -Wl,-rpath,'$ORIGIN/../lib' -lm -o "$REL/bin/cmd_pipeline_c"

# Sources and build manifest for source distribution.
for f in onnx_int8_c.c cmd_pipeline_c.c; do cp "$ROOT/tools/$f" "$REL/docs/$f"; done
cp "$ROOT/reports/C_RUNTIME_V5_zh.md" "$REL/docs/COMPARISON.md"
cp "$ROOT/reports/PRODUCTION_READINESS_V5_zh.md" "$REL/docs/PRODUCTION_READINESS.md"

# Smoke inputs.
head -8 "$ROOT/work/c_engine/inputs.tokenids.txt" > "$REL/etc/tokenids.sample"
head -8 "$ROOT/work/c_engine/text_inputs.txt" > "$REL/etc/text_inputs.sample"
printf '0.0001179935351014905\n' > "$REL/etc/threshold"

# Deploy runbook + wrapper.
cat > "$REL/README.md" <<'MD'
# cmd-risk C runtime v1.0.0

CPU inference for `cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922` INT8 ONNX.

## Quick start
```bash
# token-id inference
bin/onnx_int8_c model/model.int8.onnx etc/tokenids.sample

# raw command text -> decision (commands separated by unit separator U+001F; each `cmd<TAB>dt`)
bin/cmd_pipeline_c model/model.int8.onnx model/vocab.txt etc/text_inputs.sample "$(cat etc/threshold)"
```

## Operating point
- malicious_temperature: 0.7969617216679871
- malicious_threshold: 0.0001179935351014905 (session-FPR <= 1% dev fit)
- INT8 graph: model/model.int8.onnx
- Verified: full-prefix decision agreement vs Python 258/258; mal max_abs_diff 3.9e-8 (onnx path); session 256-token p95 6.23ms (idle); RSS ~56MB.

## Runtime requirements
- Linux x86-64, glibc
- libstdc++/libgcc (ORT runtime)
- No Python, no GPU, no AVX-512 required
MD

cat > "$REL/etc/manifest.json" <<'MD'
{"package":"cmd-risk-c-runtime","version":"1.0.0","threshold":0.0001179935351014905,"model":"cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922"}
MD

python3 - "$REL" <<'PY'
import hashlib, json, sys
from pathlib import Path
root=Path(sys.argv[1])
entries={}
for p in sorted(root.rglob('*')):
    if p.is_file() and p.name!='manifest.json':
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        entries[p.relative_to(root).as_posix()]={'sha256':h,'bytes':p.stat().st_size}
(root/'etc/manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
print('wrote manifest',len(entries),'files')
PY

tar -C "$DIST" -czf "$ARCH" "$(basename "$REL")"
python3 - "$ARCH" <<'PY'
import hashlib, sys
h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()
open(sys.argv[1]+'.sha256','w').write(h+'  '+sys.argv[1].split('/')[-1]+'\n')
print('sha256',h)
PY
echo "RELEASE=$ARCH"

#!/usr/bin/env bash
# Bake the v6 model into the single C binary and cut the release.
# Usage: RUN=artifacts/<v6-run> PROD=artifacts/<v6-run>-recal-prod-session1p-* TAG=v1.4.0 scripts/bake_v6_release.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${RUN:?set RUN}" "${PROD:?set PROD}" "${TAG:?set TAG}"

# 1) export C weights from the v6 checkpoint
RUN="$RUN" OUT_DIR=work/c_engine_v6 python3 tools/export_c_weights.py
ls -la work/c_engine_v6/model_f32.bin work/c_engine_v6/config.json

# 2) safety-backup the shipped v5 blob, then install v6 blob + vocab
cp -f ci_assets/model_f32.bin ci_assets/model_f32.v5.backup.bin
cp -f work/c_engine_v6/model_f32.bin ci_assets/model_f32.bin

# 3) update calibration constants in C sources from the PROD artifact
python3 - "$PROD" <<'PY'
import json, re, sys
from pathlib import Path
prod = Path(sys.argv[1])
cal = json.loads((prod/'onnx/calibration.json').read_text())
th, t = cal['malicious_threshold'], cal['malicious_temperature']
p = Path('tools/cmdrisk_calibration.h'); s = p.read_text()
s = re.sub(r'#define MAL_TEMP [0-9.eE+-]+f', f'#define MAL_TEMP {t!r}f', s)
p.write_text(s)
p2 = Path('tools/cmdrisk_single.c'); s = p2.read_text()
s = s.replace('float TH = 0.0001179935351014905f;', f'float TH = {th!r}f;')
p2.write_text(s)
print(f'C constants -> TH={th} MAL_TEMP={t}')
PY

# 4) rebuild + full smoke (8 alerts, 28 critical, 17 borderline controls)
make -f Makefile.release all -B
make -f Makefile.release smoke

# 5) decision parity: C binary vs ground-truth expectation on all gate sets
python3 - <<'PYPY'
import json
from pathlib import Path
for f, label in [('critical_singles_eval',1),('obfuscation_eval',1),('benign_twins_eval',0)]:
    rows=[json.loads(l) for l in open(f'data/processed/v6_obf_evals/{f}.jsonl') if l.strip()]
    with open(f'/tmp/{f}_lines.txt','w') as fh:
        for r in rows:
            fh.write(r['commands'][0]['cmd']+'
')
PYPY
TH_VAL="$(python3 -c "import json;print(json.load(open('$PROD/onnx/calibration.json'))['malicious_threshold'])")"
for f in critical_singles_eval obfuscation_eval benign_twins_eval; do
  ./build/cmdrisk /tmp/${f}_lines.txt "$TH_VAL" > /tmp/${f}_cout.txt 2>/dev/null
done
python3 - <<'PYPY'
labels={'critical_singles_eval':1,'obfuscation_eval':1,'benign_twins_eval':0}
for f, label in labels.items():
    c=[l.rstrip('\n').split('\t') for l in open(f'/tmp/{f}_cout.txt')]
    ok=sum(1 for row in c if int(row[3])==label)
    print(f'{f}: rows={len(c)} expect-match={ok} mismatch={len(c)-ok}')
    assert ok==len(c), f
print('PARITY_OK')
PYPY
sha256sum build/cmdrisk
echo "BAKE_DONE sha256 above; tag=$TAG"

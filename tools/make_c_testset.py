#!/usr/bin/env python3
"""Tokenize session prefixes and produce a Python reference for the C engine."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from cmdrisk.tokenization import CommandTokenizer
from cmdrisk.model_factory import load_risk_model

ROOT = Path(__file__).resolve().parents[1]
ONNX = ROOT/"artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922-recal-prod-session1p-20260922/onnx"
RUN = ROOT/"artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922"
TOK = CommandTokenizer.from_pretrained(ONNX, max_length=256, command_max_tokens=128)
records = [json.loads(l) for l in (ROOT/'/tmp/v5_runtime_sample.jsonl').read_text().splitlines() if l.strip()]
records = records[:6]

model = load_risk_model(RUN/'short_phase')
model.eval()
refs=[]
with open(ROOT/'work/c_engine/inputs.tokenids.txt','w') as f:
    for ri, r in enumerate(records):
        cmds=[(c['cmd'], c.get('dt')) for c in r['commands']]
        for k in range(1,len(cmds)+1):
            enc=TOK.encode([{'cmd':cmd,'dt':dt} for cmd,dt in cmds[:k]])
            ids=list(enc.input_ids)
            f.write(f"{len(ids)}" + "".join(f" {x}" for x in ids) + "\n")
            iids=torch.tensor([ids], dtype=torch.long)
            am=torch.ones_like(iids)
            with torch.no_grad():
                out=model(input_ids=iids, attention_mask=am)
                mal=out.malicious_logits
                tact=out.tactic_logits.squeeze(0)
                tech=out.technique_logits.squeeze(0)
            refs.append({
                'session_index':ri,'prefix_len':k,'mal_logit':float(mal),
                'tact_logit':tact.tolist(),'tech_logit':tech.tolist(),
            })
(ROOT/'work/c_engine/refs.json').write_text(json.dumps(refs))
print('refs',len(refs),'sessions',len(records))
# tiny subset for speed
lines=(ROOT/'work/c_engine/inputs.tokenids.txt').read_text().splitlines()
small=[lines[0], lines[1], lines[len(lines)//2], lines[-1]]
(ROOT/'work/c_engine/inputs.small.tokenids.txt').write_text('\n'.join(small)+'\n')
print('small indices [0,1,mid,last]',[0,1,len(lines)//2,len(lines)-1])

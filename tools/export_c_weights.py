#!/usr/bin/env python3
"""Export the v5 BERT risk checkpoint to a flat C layout binary.

Reads the frozen safetensors, emits:
  work/c_engine/model_f32.bin  (little-endian float32, C row-major)
  work/c_engine/config.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922"
SAFE = RUN / "short_phase/model.safetensors"
OUT_DIR = ROOT / "work/c_engine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with safe_open(str(SAFE), framework="pt") as f:
    keys = sorted(f.keys())
    tensors = {}
    for k in keys:
        t = f.get_slice(k)
        # get_tensor needs tensorflow/pyarrow; use safe_open.get_tensor
        tensors[k] = f.get_tensor(k)

cfg = {
    "vocab_size": int(tensors["bert.embeddings.word_embeddings.weight"].shape[0]),
    "hidden_size": int(tensors["bert.embeddings.word_embeddings.weight"].shape[1]),
    "num_layers": 8,
    "num_heads": 4,
    "intermediate_size": int(tensors["bert.encoder.layer.0.intermediate.dense.weight"].shape[0]),
    "max_seq": 256,
    "layer_norm_eps": 1e-12,
    "num_tactics": 14,
    "num_techniques": 12,
}

blobs: list[bytes] = []
def put(name: str):
    arr = np.ascontiguousarray(tensors[name], dtype=np.float32)
    blobs.append(arr.tobytes())
    print(f"{arr.dtype} {arr.shape} {name}")

put("bert.embeddings.word_embeddings.weight")
put("bert.embeddings.position_embeddings.weight")
put("bert.embeddings.token_type_embeddings.weight")
put("bert.embeddings.LayerNorm.weight")
put("bert.embeddings.LayerNorm.bias")
for l in range(cfg["num_layers"]):
    base = f"bert.encoder.layer.{l}"
    put(f"{base}.attention.self.query.weight")
    put(f"{base}.attention.self.query.bias")
    put(f"{base}.attention.self.key.weight")
    put(f"{base}.attention.self.key.bias")
    put(f"{base}.attention.self.value.weight")
    put(f"{base}.attention.self.value.bias")
    put(f"{base}.attention.output.dense.weight")
    put(f"{base}.attention.output.dense.bias")
    put(f"{base}.attention.output.LayerNorm.weight")
    put(f"{base}.attention.output.LayerNorm.bias")
    put(f"{base}.intermediate.dense.weight")
    put(f"{base}.intermediate.dense.bias")
    put(f"{base}.output.dense.weight")
    put(f"{base}.output.dense.bias")
    put(f"{base}.output.LayerNorm.weight")
    put(f"{base}.output.LayerNorm.bias")
put("bert.pooler.dense.weight")
put("bert.pooler.dense.bias")
put("malicious_head.weight")
put("malicious_head.bias")
put("tactic_head.weight")
put("tactic_head.bias")
put("technique_head.weight")
put("technique_head.bias")

blob = b"".join(blobs)
(OUT_DIR / "model_f32.bin").write_bytes(blob)
(OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
stats = {
    "binary_bytes": len(blob),
    "parameters": int(sum(arr.numel() for arr in [tensors[k] for k in keys])),
    "binary_sha256": __import__("hashlib").sha256(blob).hexdigest(),
    "config": cfg,
}
(OUT_DIR / "export_meta.json").write_text(json.dumps(stats, indent=2) + "\n")
print(stats)

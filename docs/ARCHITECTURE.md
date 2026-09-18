# Architecture

`cmd-session-risk` has four boundaries:

1. Source adapters normalize public corpora into versioned `SessionRecord` values.
2. Teacher/subagent code labels, augments, and audits those sessions while
   preserving provenance and consensus quality.
3. Prefix generation expands a session into all observable command prefixes.
4. A ModernBERT multi-label encoder scores each prefix and is exported to CPU
   INT8 ONNX for alert-only inference.

Raw data, checkpoints, and generated datasets live outside Git. The source
manifest records source URL, revision/hash, license, and redistribution status.
All model artifacts contain the model revision, label vocabulary, calibration
metadata, and schema version.

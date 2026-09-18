# Remote GPU and Runtime Operations

This document covers the ONNX runtime contract and the Lightning remote workflow.
The runtime is alert-only and uses the CPU execution provider.

## ONNX artifact layout

`SessionScorer.from_onnx(artifact_dir)` expects a self-contained directory:

```text
artifact/
  model.onnx                     # or model.int8.onnx / model_int8.onnx
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json        # optional, depending on tokenizer
  command_tokenization.json      # optional; written by CommandTokenizer.save_pretrained
  label_vocab.json               # labels.json is accepted as fallback
  calibration.json               # thresholds.json is accepted as fallback
  runtime_config.json            # optional; config.json is accepted as fallback
```

When present, `model.int8.onnx` is preferred over `model_int8.onnx`, which is
preferred over `model.onnx`.

`label_vocab.json` (or `labels.json`):

```json
{
  "model_version": "modernbert-v1",
  "max_length": 8192,
  "tactics": [
    {"id": "TA0007", "name": "Discovery"}
  ],
  "techniques": [
    {"id": "T1105", "name": "Ingress Tool Transfer"}
  ]
}
```

The training exporter writes bare ATT&CK ids; the runtime resolves their names
from the taxonomy. Both forms are accepted:

```json
{
  "version": 1,
  "tactics": ["TA0007", "TA0002"],
  "techniques": ["T1105"]
}
```

`calibration.json` accepts both the runtime envelope and the flat training
format. The training exporter emits per-head temperatures plus flat id-to-
threshold maps:

```json
{
  "malicious_temperature": 1.0,
  "malicious_threshold": 0.91,
  "tactic_temperature": 1.0,
  "technique_temperature": 1.0,
  "tactic_thresholds": {"TA0007": 0.45},
  "technique_thresholds": {"T1105": 0.60}
}
```

The older runtime envelope, including a shared `temperature`, is still
supported:

```json
{
  "malicious_threshold": 0.91,
  "temperature": 1.0,
  "tactics": {
    "default": 0.5,
    "per_label": {"TA0007": 0.45}
  },
  "techniques": {
    "default": 0.5,
    "per_label": {"T1105": 0.60}
  }
}
```

Optional `runtime_config.json` can override input/output names, and
`command_tokenization.json` carries the truncation budgets written at training
time:

```json
{
  "max_length": 8192,
  "command_max_tokens": 512,
  "session_head_tokens": 1024,
  "session_tail_tokens": 7168
}
```

```json
{
  "max_length": 8192,
  "onnx_inputs": ["input_ids", "attention_mask"],
  "onnx_outputs": {
    "malicious": "malicious_logits",
    "tactics": "tactics_logits",
    "techniques": "techniques_logits"
  }
}
```

The runtime serialises every observed prefix using the training token format:

```text
<SESSION><CMD_START>whoami<CMD_END><DT_0S>...</SESSION>
```

Time buckets are `<DT_0S>`, `<DT_LE_1S>`, `<DT_LE_10S>`, `<DT_LE_1M>`,
`<DT_LE_10M>`, `<DT_GT_10M>`, and `<DT_UNKNOWN>`. Inputs longer than
`max_length` retain the first 1024 tokens and the most recent tokens.

`risk_score` is `malicious_probability * severity(active ATT&CK tactic)`. An
alert is selected only by the calibrated malicious threshold and remains
latched for the rest of the session.

## Python and CLI

```python
from cmdrisk.runtime import SessionScorer

scorer = SessionScorer.from_onnx("artifacts/onnx")
session = scorer.start_session("ssh-123")
session.update("whoami", dt=0)
session.update("curl -o /tmp/a https://example.invalid/a", dt=2.1)
result = session.result()
print(result.alerted, result.peak_probability)
```

Batch mode accepts full `SessionRecord` JSONL or the compact form
`{"session_id": "...", "commands": [{"cmd": "...", "dt": 0}]}`. It emits one
`SequenceScore` per input record by default; `--flat` emits one enriched
`ScoreEntry` per prefix.

```bash
cmdrisk score --artifact-dir artifacts/onnx \
  --input data/sessions.jsonl --output data/scores.jsonl

cmdrisk score --artifact-dir artifacts/onnx --input sessions.jsonl --flat

printf '%s\n' '{"cmd":"whoami","dt":0}' \
  | cmdrisk stream --artifact-dir artifacts/onnx --session-id live-1
```

Exit codes are `0` for success, `2` for invalid input, `3` for artifact load
failure, and `4` for inference failure. Errors are JSON objects on stderr.

## Lightning bootstrap

The bootstrap script never accepts a token argument and never prints
`LIGHTNING_SETUP_URL`. Set the temporary setup URL only in the environment:

```bash
export LIGHTNING_SETUP_URL='<temporary setup URL>'
export LIGHTNING_SSH_USER='s_<studio-user>'
./scripts/bootstrap_lightning.sh
unset LIGHTNING_SETUP_URL
```

If both `~/.ssh/lightning_rsa` and a matching SSH `Host`/`HostName` entry
already exist, the script exits without calling the setup endpoint. Optional
overrides are `LIGHTNING_SSH_KEY` and `LIGHTNING_SSH_USER`. The token URL must
not be committed or recorded in logs.

## Preflight and synchronization

Run preflight both locally and on the remote host:

```bash
python scripts/remote_preflight.py --output remote_environment.json
```

The JSON report includes GPU devices and memory, CUDA/driver versions, CPU
topology and affinity, RAM, disk, and Python runtime information. A host
without a GPU is reported normally with `gpu.count == 0`.

Upload processed Parquet files and their checksum manifest:

```bash
export LIGHTNING_SSH_TARGET='s_<studio-user>@ssh.lightning.ai'
python scripts/sync_remote.py upload \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --remote-root '~/cmd-session-risk' \
  --local-root .
```

Training jobs should create remote manifests before download:

```bash
python scripts/sync_remote.py manifest \
  --root artifacts/checkpoints \
  --manifest manifests/checkpoints.sha256
```

Repeat for `onnx` and `metrics`, then download all three components locally:

```bash
python scripts/sync_remote.py download \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --remote-root '~/cmd-session-risk' \
  --components checkpoints onnx metrics
```

Downloaded files are placed under `artifacts/checkpoints`,
`artifacts/onnx`, and `artifacts/metrics`; remote manifests are copied to
`manifests/`. Every file is SHA256-verified. Missing or mismatched manifests
fail the command unless the explicitly unsafe
`--allow-missing-manifests` option is used. Use `--dry-run` to inspect rsync
transfers. The training host must have rsync installed.

## Latency benchmark

```bash
python scripts/latency_bench.py \
  --artifact-dir artifacts/onnx \
  --lengths 2048 8192 \
  --warmup 5 --iterations 20 \
  --cpu-affinity 0-7 --format json
```

The benchmark reports P50 and P95 end-to-end prefix-scoring latency, along
with actual token counts and command counts. Pinning eight CPUs is optional;
omit `--cpu-affinity` when affinity is managed externally.

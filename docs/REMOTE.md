# Remote GPU and Runtime Operations

This document covers the ONNX runtime contract, Lightning bootstrap and
preflight flow, remote training, latency measurement, and promotion evidence.
The runtime is alert-only and uses the CPU execution provider.

## ONNX artifact layout

`SessionScorer.from_onnx(artifact_dir)` expects a self-contained directory:

```text
artifact/
  export_manifest.json           # required integrity and promotion metadata
  model.onnx                     # or model.int8.onnx / model_int8.onnx
  tokenizer.json
  tokenizer_config.json
  special_tokens_map.json        # optional, depending on tokenizer
  command_tokenization.json      # optional; written by CommandTokenizer.save_pretrained
  label_vocab.json               # labels.json is accepted as fallback
  calibration.json               # thresholds.json is accepted as fallback
  runtime_config.json            # optional; config.json is accepted as fallback
  metrics.json                   # copied from the training run when supplied
  latency_report.json            # attached after benchmark measurement
```

When present, `model.int8.onnx` is preferred over `model_int8.onnx`, which is
preferred over `model.onnx`.

`export_manifest.json` uses schema `cmdrisk.onnx.artifact` version `1`. It
declares the FP32 and INT8 model names, every artifact's byte size and SHA256,
parity errors, latency evidence, and the aggregated promotion decision. The
manifest is mandatory in production; a bundle that is missing it, omits a
declared file, or changes any declared byte is rejected.

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

## Manifest verification

Manifest verification is fail-closed and runs **before** model, tokenizer, or
configuration bytes are read. `SessionScorer.from_onnx()` defaults to
`verify_manifest=True`; the CLI `score` and `stream` commands use the same
default. A missing or malformed manifest, path traversal, undeclared model, size
mismatch, or SHA256 mismatch raises `ArtifactLoadError` and exits with code
`3`.

`verify_manifest=False` exists only for unit-test fakes and explicitly handled
legacy bundles. It must not be used in a deployment path. `manifest_path` can
select an alternate manifest location, but the manifest still has to describe
the same artifact root.

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
or verification failure, and `4` for inference failure. Errors are JSON objects
on stderr.

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
overrides are `LIGHTNING_SSH_KEY` and `LIGHTNING_SSH_USER`. The setup URL and
token must not be committed, echoed, or recorded in logs.

## Local and remote preflight

Collect a structured local report with optional fail-closed minimums:

```bash
python scripts/remote_preflight.py \
  --output remote_environment.json \
  --require-cuda \
  --min-vram-gb 24 \
  --min-free-disk-gb 50 \
  --min-cpu-cores 8 \
  --min-cuda-version 12.1
```

The JSON report includes GPU devices and memory, CUDA/driver versions, CPU
topology and affinity, RAM, disk, Python runtime information, and a `gates`
block. Any configured minimum that cannot be proven fails the command with a
non-zero exit status. A CPU-only host is reported normally only when neither
CUDA nor VRAM gates are requested.

`scripts/sync_remote.py upload` and `download` now run a remote
connectivity/disk preflight by default before transferring data. Add
`--require-cuda` and/or `--min-vram-gb` when the transfer must target a GPU host.
`--skip-preflight` is available only as an explicitly unsafe escape hatch.

The full remote training script writes its environment report to
`$REMOTE_DATA/remote_environment.json`, never into the synced project tree. The
rsync rules in `scripts/train_remote.sh` exclude `remote_environment.json`,
`*.pem`, and `*.key` so secrets and runtime reports are not uploaded with the
source checkout.

## Data and artifact synchronization

Upload processed Parquet files and their checksum manifest:

```bash
export LIGHTNING_SSH_TARGET='s_<studio-user>@ssh.lightning.ai'
python scripts/sync_remote.py upload \
  --ssh-target "$LIGHTNING_SSH_TARGET" \
  --remote-root '~/cmd-session-risk' \
  --min-free-disk-gb 20
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

## Remote training

Run the full workflow locally or against the Lightning host:

```bash
./scripts/train_remote.sh \
  --data-dir data/processed \
  --output-dir artifacts/remote-run \
  --ssh-key ~/.ssh/lightning_rsa \
  --min-free-disk-gb 20
```

The script validates the SSH key and data path, synchronizes the project while
excluding secrets/artifacts, runs fail-closed remote GPU/disk preflight, uploads
the processed data, launches `kaggle/train.py` (including `torchrun` on
multi-GPU hosts), verifies that the expected checkpoint exists, exports ONNX,
and downloads the remote artifacts. Use `--vram-gb` when automatic VRAM
detection needs an explicit value. `--allow-cpu` disables the CUDA requirement
but is not suitable for real training. `--local` runs the same training and
export path in the current environment.

## Latency evidence and promotion

Promotion is fail-closed. See `docs/QUALITY_GATES.md` for the complete metric,
slice, audit, and delay gates. In summary, promotion requires:

- the held-out metric gates, all four evaluation slices, and the proxy audit;
- an on-profile latency report at or below the P95/P99 thresholds plus
  tolerance;
- a complete ONNX export manifest with artifact checksums and parity checks;
- no split leakage across families, parent samples, or synthetic children.

After export, measure latency on the 8-core-or-larger x86-64 AVX2 CPU profile
and attach the evidence to the manifest:

```bash
python scripts/latency_bench.py \
  --artifact-dir artifacts/onnx \
  --lengths 2048 8192 \
  --warmup 5 --iterations 20 \
  --cpu-affinity 0-7 \
  --output artifacts/metrics/latency.json \
  --update-manifest --require-promotion \
  --format json
```

`--update-manifest` requires `--output`. It copies the report into the artifact
directory as `latency_report.json`, recalculates the manifest checksums, and
revalidates the metrics and latency promotion blocks when
`--require-promotion` is supplied. The default latency gates are P95
`50.0 ms`, P99 `70.0 ms`, and `10.0 ms` tolerance; the profile also requires
x86-64, at least 8 logical CPUs, and AVX2. A gate failure exits non-zero.
`--allow-failure` is for diagnostics only and must never be used to promote a
model.

If the latency report already exists during export, it can be embedded in one
step:

```bash
python kaggle/export_onnx.py \
  --checkpoint artifacts/checkpoints/long_phase \
  --output-dir artifacts/onnx \
  --calibration artifacts/calibration.json \
  --metrics artifacts/metrics.json \
  --latency-report artifacts/metrics/latency.json \
  --require-promotion
```

Both `export_onnx(..., require_promotion=True)` and
`update_export_manifest_latency(..., require_promotion=True)` refuse to return a
promotable manifest unless the metrics promotion block and latency evidence
both pass. A normal export still writes `export_manifest.json`; omitting
latency then produces a valid integrity manifest with
`promotion.eligible_for_promotion: false` until latency evidence is attached.

The benchmark reports P50/P95/P99 and mean end-to-end prefix-scoring latency,
actual token and command counts, provider/runtime versions, CPU topology,
affinity, AVX2 status, and the resulting gate decision. Pinning eight CPUs is
recommended for the documented profile; use `--cpu-affinity` explicitly or
manage affinity externally. The runtime always uses the ONNX Runtime CPU
execution provider.

## Post-training operations

Once the remote run has been synchronised back, the remaining steps are
independent and can be replayed without re-training.

### Watchdog sync after a detached run

`logs/watch_and_sync.sh` is a noclobber `flock`-guarded watchdog. It polls the
remote `.train.exit` sentinel every five minutes and, when training *and* the
ONNX export have finished, rsyncs the whole artifact directory back, retrying
five times. It is meant to survive SSH drops and is launched with:

```bash
nohup setsid bash logs/watch_and_sync.sh >> logs/watch_and_sync.log 2>&1 &
```

The sentinel is written by `scripts/train_remote.sh --detach` only after the
export step, so a completed rsync always implies a complete artifact.

### Rebuilding the promotion block after the audit finishes

Training runs launched without `--proxy-audit` record
`promotion.proxy_audit.blockers: ["proxy_audit_missing"]` and
`promotion.passed: false`, even when every point-estimate gate passes.
`scripts/repromote.py` replays only the cheap half of the pipeline — it reloads
the final checkpoint, rebuilds the dev/test prefix examples exactly as
`run_training` does, re-fits calibration and re-evaluates test metrics with a
finished audit report injected:

```bash
python scripts/repromote.py \
  --run-dir artifacts/<run> \
  --data data/processed/v2/train.jsonl \
  --proxy-audit reports/proxy_audit/report.json \
  --output-dir artifacts/<run>/repromoted
```

Outputs (`calibration.json`, `metrics.json`, `repromote_summary.json`) go to a
separate directory so the original run artifact stays byte-for-byte intact.

### Scoring the audit cases

The proxy audit needs one session-level `malicious_probability` per case.
`scripts/score_audit_cases.py` filters the held-out split down to the audit
sessions, scores them with `cmdrisk score`, and maps `peak_probability` (the max
over prefix scores, i.e. "does this session ever look malicious") to
`malicious_probability`:

```bash
python scripts/score_audit_cases.py \
  --cases reports/proxy_audit/cases.jsonl \
  --test data/processed/v2/test.jsonl \
  --artifact-dir artifacts/<run>/onnx \
  --out reports/proxy_audit/predictions.jsonl
```

### Aggregating blind reviewer votes

`scripts/aggregate_proxy_audit.py` replays recorded per-reviewer vote files
through the same `ProxyReviewResponse` schema and `ProxyAuditor` gate used by
`scripts/run_proxy_audit.py`, so a CPU-only, offline review produces an
identical promotion decision:

```bash
python scripts/aggregate_proxy_audit.py \
  --cases reports/proxy_audit/cases.jsonl \
  --votes-dir reports/proxy_audit/votes \
  --predictions reports/proxy_audit/predictions.jsonl \
  --output reports/proxy_audit/report.json \
  --provenance-out reports/proxy_audit/provenance.json \
  --reviewer-label R1=<label> --reviewer-label R2=<label> --reviewer-label R3=<label>
```

The provenance record stores the contract hash, cases hash, blind payload field
list and a per-reviewer vote digest. The audit is a model-proxy gate: it is not
human gold labelling and it does not by itself establish production FPR.

### One-shot delivery validation

`scripts/validate_delivery.sh <RUN_DIR> [TEST_JSONL]` walks the whole delivery:
layout, `SHA256SUMS`, export manifest / parity / promotion, parameter count from
the ONNX initializers, metrics slice summary, `cmdrisk score --flat`, a
`cmdrisk stream` smoke test and a latency benchmark. `PY`, `CLI` and
`ITERATIONS` override the interpreter, CLI entry point and benchmark iterations.
`scripts/validate_int8.py` measures FP32-vs-INT8 decision agreement and
first-alert-index drift; it is decision-level evidence, not `atol` parity.

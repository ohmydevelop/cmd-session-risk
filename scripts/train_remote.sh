#!/usr/bin/env bash
set -euo pipefail

DEFAULT_HOST="s_01m2tmwd7v1mc00aayt5vf82hr@ssh.lightning.ai"
HOST="${LIGHTNING_SSH_HOST:-$DEFAULT_HOST}"
REMOTE_ROOT="${REMOTE_ROOT:-cmd-session-risk}"
SSH_KEY="${LIGHTNING_SSH_KEY:-}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR=""
OUTPUT_DIR=""
VRAM_GB=""
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-20}"
REQUIRE_CUDA=1
EXTRA_ARGS=()
EXPORT_ARGS=()
RUN_LOCAL=0
DETACH=0
POLL_INTERVAL="${POLL_INTERVAL:-30}"
POLL_TIMEOUT="${POLL_TIMEOUT:-21600}"

usage() {
  cat <<'EOF'
Usage: scripts/train_remote.sh --data-dir DIR --output-dir DIR [options]

Options:
  --host HOST             Lightning SSH host (defaults to the project endpoint)
  --remote-root DIR       Remote project directory (default: cmd-session-risk).
                          Use a unique value per run so stale remote artifacts
                          can never be mistaken for a successful new run.
  --ssh-key PATH          SSH private key (or LIGHTNING_SSH_KEY)
  --vram-gb N             Override remote VRAM detection
  --min-free-disk-gb N    Fail if the remote filesystem has less free space
  --skip-int8             Export FP32 ONNX only (default verifies FP32 parity;
                          add this when INT8 has not been validated)
  --detach                Run the remote phase inside a detached tmux session
                          and poll for its exit sentinel. Survives SSH drops.
  --poll-interval SECONDS Poll interval for --detach (default 30)
  --poll-timeout SECONDS  Give up polling after this long (default 21600)
  --allow-cpu             Do not require a CUDA GPU (unsupported for real training)
  --local                 Run in the current environment instead of rsync/SSH
  -- ARGS...              Additional kaggle/train.py arguments
  -h, --help              Show this help

Set LIGHTNING_SETUP_URL to bootstrap a new SSH key. The value is intentionally
never hard-coded, never echoed, and never written into the repository.
Environment reports are written under the remote data directory, never into
the synced project tree.
EOF
}

fail() {
  echo "$1" >&2
  exit 2
}

while (($#)); do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --vram-gb) VRAM_GB="$2"; shift 2 ;;
    --min-free-disk-gb) MIN_FREE_DISK_GB="$2"; shift 2 ;;
    --skip-int8) EXPORT_ARGS+=(--skip-int8); shift ;;
    --detach) DETACH=1; shift ;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
    --poll-timeout) POLL_TIMEOUT="$2"; shift 2 ;;
    --allow-cpu) REQUIRE_CUDA=0; shift ;;
    --local) RUN_LOCAL=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$DATA_DIR" || -z "$OUTPUT_DIR" ]]; then
  echo "--data-dir and --output-dir are required" >&2
  usage >&2
  exit 2
fi
if [[ ! -e "$DATA_DIR" ]]; then
  echo "Data path does not exist: $DATA_DIR" >&2
  exit 2
fi
if [[ -n "$SSH_KEY" && ! -f "$SSH_KEY" ]]; then
  echo "SSH key does not exist: $SSH_KEY" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

export_flags="${EXPORT_ARGS[*]:-}"

run_train_command() {
  local gpu_count
  gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)"
  if [[ -z "$gpu_count" || "$gpu_count" -lt 1 ]]; then
    gpu_count=0
  fi

  local cmd=(uv run --extra train python kaggle/train.py)
  if [[ "$gpu_count" -gt 1 ]]; then
    cmd=(uv run --extra train torchrun --standalone --nproc_per_node="$gpu_count" kaggle/train.py)
  fi
  local args=(--data "$1" --output-dir "$2")
  if [[ -n "$VRAM_GB" ]]; then
    args+=(--vram-gb "$VRAM_GB")
  fi
  args+=("${EXTRA_ARGS[@]}")
  "${cmd[@]}" "${args[@]}"
}

if [[ "$RUN_LOCAL" -eq 1 ]]; then
  cd "$LOCAL_ROOT"
  run_train_command "$DATA_DIR" "$OUTPUT_DIR"
  local_checkpoint="$OUTPUT_DIR/long_phase"
  for arg in "${EXTRA_ARGS[@]}"; do
    [[ "$arg" == "--skip-long-phase" ]] && local_checkpoint="$OUTPUT_DIR/short_phase"
  done
  if [[ ! -d "$local_checkpoint" ]]; then
    fail "training did not produce a checkpoint: $local_checkpoint"
  fi
  # shellcheck disable=SC2086
  uv run --extra train python kaggle/export_onnx.py \
    --checkpoint "$local_checkpoint" \
    --output-dir "$OUTPUT_DIR/onnx" \
    --calibration "$OUTPUT_DIR/calibration.json" \
    --metrics "$OUTPUT_DIR/metrics.json" $export_flags
  exit 0
fi

command -v ssh >/dev/null 2>&1 || fail "ssh is required but was not found"
command -v rsync >/dev/null 2>&1 || fail "rsync is required but was not found"

if [[ -n "${LIGHTNING_SETUP_URL:-}" ]]; then
  # The setup URL is supplied out-of-band by the user and contains a one-time
  # token.  Never echo, log, or persist it.
  curl -fsSL "$LIGHTNING_SETUP_URL" | bash
fi

SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new
  -o KexAlgorithms=curve25519-sha256 -o HostKeyAlgorithms=rsa-sha2-512)
if [[ -n "$SSH_KEY" ]]; then
  SSH_OPTS+=(-i "$SSH_KEY")
fi

ssh "${SSH_OPTS[@]}" "$HOST" "true" >/dev/null 2>&1 \
  || fail "cannot reach remote host: $HOST"

# The Lightning login shell starts in a different directory from $HOME
# (/home/zeus vs /teamspace/studios/this_studio), so a relative remote path
# would resolve differently for rsync and for interactive ssh.  Pin it.
if [[ "$REMOTE_ROOT" != /* ]]; then
  REMOTE_HOME="$(ssh "${SSH_OPTS[@]}" "$HOST" 'printf %s "$HOME"' || true)"
  if [[ -n "$REMOTE_HOME" ]]; then
    REMOTE_ROOT="$REMOTE_HOME/$REMOTE_ROOT"
  else
    echo "warning: could not resolve \$HOME on $HOST; using relative $REMOTE_ROOT" >&2
  fi
fi

REMOTE_DATA="/tmp/cmd-session-risk-data"
REMOTE_OUTPUT="$REMOTE_ROOT/artifacts"
REMOTE_ENVIRONMENT_FILE="$REMOTE_DATA/remote_environment.json"

ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE_ROOT' '$REMOTE_DATA' '$REMOTE_OUTPUT'"

# Lightning images ship without rsync and without CPython development headers.
# Both are needed: rsync for the artifact sync, Python.h for the Triton kernels
# that torch JIT-compiles on the first training step.  Install fail-softly.
if ! ssh "${SSH_OPTS[@]}" "$HOST" "command -v rsync >/dev/null 2>&1"; then
  echo "Installing rsync on $HOST" >&2
  ssh "${SSH_OPTS[@]}" "$HOST" "sudo -n apt-get install -y rsync >/dev/null 2>&1 || { sudo -n apt-get update -qq && sudo -n apt-get install -y rsync >/dev/null 2>&1; }" || true
  ssh "${SSH_OPTS[@]}" "$HOST" "command -v rsync >/dev/null 2>&1" || fail "rsync is not available on $HOST and could not be installed"
fi

if ! ssh "${SSH_OPTS[@]}" "$HOST" "ls /usr/include/python3.*/Python.h >/dev/null 2>&1"; then
  echo "Installing python3-dev + build-essential on $HOST" >&2
  ssh "${SSH_OPTS[@]}" "$HOST" "sudo -n apt-get install -y python3-dev build-essential >/dev/null 2>&1 || { sudo -n apt-get update -qq && sudo -n apt-get install -y python3-dev build-essential >/dev/null 2>&1; }" || true
  ssh "${SSH_OPTS[@]}" "$HOST" "ls /usr/include/python3.*/Python.h >/dev/null 2>&1" || echo "warning: Python.h still missing; Triton kernels may fail to build" >&2
fi

# The project tree must never contain secrets, environment reports, or the
# multi-gigabyte raw corpus (which is not needed to train on processed data).
rsync -az --delete \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.git' --exclude '.venv' --exclude 'artifacts' --exclude '__pycache__' \
  --exclude 'remote_environment.json' --exclude '*.pem' --exclude '*.key' \
  --exclude 'data/' \
  "$LOCAL_ROOT/" "$HOST:$REMOTE_ROOT/"

# Fail-closed GPU/CUDA/disk preflight before uploading data or launching training.
preflight_args=(--path "$REMOTE_DATA" --min-free-disk-gb "$MIN_FREE_DISK_GB" --output "$REMOTE_ENVIRONMENT_FILE")
if [[ "$REQUIRE_CUDA" -eq 1 ]]; then
  preflight_args+=(--require-cuda)
fi
ssh "${SSH_OPTS[@]}" "$HOST" \
  "cd '$REMOTE_ROOT' && python3 scripts/remote_preflight.py ${preflight_args[*]}"

if [[ ! -d "$DATA_DIR" ]]; then
  fail "processed data directory does not exist: $DATA_DIR"
fi

# --delete keeps the remote data directory an exact mirror of DATA_DIR, so a
# stale sessions.jsonl from an earlier run can never be double-counted.
rsync -az --delete "$DATA_DIR/" "$HOST:$REMOTE_DATA/"

REMOTE_RUNNER="$REMOTE_ROOT/.remote_train.sh"
read -r -d '' RUNNER_BODY <<'REMOTE' || true
#!/usr/bin/env bash
set -euo pipefail
remote_data="$1"
output_dir="$2"
vram_gb="$3"
export_flags="$4"
shift 4
remote_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$remote_root"

uv sync --extra train
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ' || true)"
if [[ -z "$gpu_count" ]]; then
  gpu_count=0
fi
command=(uv run --extra train python kaggle/train.py)
if [[ "$gpu_count" -gt 1 ]]; then
  command=(uv run --extra train torchrun --standalone --nproc_per_node="$gpu_count" kaggle/train.py)
fi
args=(--data "$remote_data" --output-dir "$output_dir")
if [[ -n "$vram_gb" ]]; then
  args+=(--vram-gb "$vram_gb")
fi
args+=("$@")
"${command[@]}" "${args[@]}"

checkpoint="$output_dir/long_phase"
for arg in "$@"; do
  [[ "$arg" == "--skip-long-phase" ]] && checkpoint="$output_dir/short_phase"
done
if [[ ! -d "$checkpoint" ]]; then
  echo "training did not produce a checkpoint: $checkpoint" >&2
  exit 3
fi
# shellcheck disable=SC2086
uv run --extra train python kaggle/export_onnx.py \
  --checkpoint "$checkpoint" \
  --output-dir "$output_dir/onnx" \
  --calibration "$output_dir/calibration.json" \
  --metrics "$output_dir/metrics.json" $export_flags
REMOTE

# Feed the runner from a local file instead of a pipeline.  ssh closes stdin as
# soon as the remote ``cat`` finishes, and a SIGPIPE on the writer would kill
# this script with exit code 141 instead of reporting a real failure.
RUNNER_SOURCE="$(mktemp "${TMPDIR:-/tmp}/cmdrisk-remote-runner.XXXXXX")"
printf '%s\n' "$RUNNER_BODY" > "$RUNNER_SOURCE"
if ! ssh "${SSH_OPTS[@]}" "$HOST" \
  "cat > '$REMOTE_RUNNER' && chmod +x '$REMOTE_RUNNER'" < "$RUNNER_SOURCE"; then
  rm -f "$RUNNER_SOURCE"
  fail "failed to install the remote runner on $HOST"
fi
rm -f "$RUNNER_SOURCE"

# Quote every remote argument so paths and flags survive the tmux/SSH layers.
quoted_args=""
for arg in "$REMOTE_DATA" "$REMOTE_OUTPUT" "$VRAM_GB" "$export_flags" "${EXTRA_ARGS[@]}"; do
  quoted_args+=" $(printf '%q' "$arg")"
done

if [[ "$DETACH" -eq 1 ]]; then
  REMOTE_ABS="$(ssh "${SSH_OPTS[@]}" "$HOST" "cd '$REMOTE_ROOT' && pwd")"
  SESSION="cmdrisk-train"
  SENTINEL="$REMOTE_ABS/artifacts/.train.exit"
  LOG="$REMOTE_ABS/artifacts/train.log"

  if ssh "${SSH_OPTS[@]}" "$HOST" "tmux has-session -t '$SESSION' 2>/dev/null"; then
    echo "Reattaching to in-flight remote session '$SESSION' (skipping relaunch)"
  else
    ssh "${SSH_OPTS[@]}" "$HOST" "rm -f '$SENTINEL'"
    ssh "${SSH_OPTS[@]}" "$HOST" \
      "tmux new-session -d -s '$SESSION' -c '$REMOTE_ABS' \
         \"bash '$REMOTE_RUNNER'$quoted_args > '$LOG' 2>&1; echo \\\$? > '$SENTINEL'\""
    echo "Launched detached remote training session '$SESSION' (log: $LOG)"
  fi

  waited=0
  while :; do
    exit_code="$(ssh "${SSH_OPTS[@]}" "$HOST" "cat '$SENTINEL' 2>/dev/null || true" || true)"
    if [[ -n "$exit_code" ]]; then
      break
    fi
    if (( waited >= POLL_TIMEOUT )); then
      fail "timed out after ${POLL_TIMEOUT}s waiting for $SENTINEL (session may still be running)"
    fi
    sleep "$POLL_INTERVAL"
    waited=$(( waited + POLL_INTERVAL ))
    echo "[$waited s] remote session still running..."
  done
  echo "Remote training finished with exit code $exit_code"
  if [[ "$exit_code" != "0" ]]; then
    ssh "${SSH_OPTS[@]}" "$HOST" "tail -n 80 '$LOG'" || true
    fail "remote training failed with exit code $exit_code (see $LOG)"
  fi
else
  ssh "${SSH_OPTS[@]}" "$HOST" "bash '$REMOTE_RUNNER'$quoted_args"
fi

rsync -az -e "ssh ${SSH_OPTS[*]}" "$HOST:$REMOTE_OUTPUT/" "$OUTPUT_DIR/"
echo "Remote artifacts synchronized to $OUTPUT_DIR"

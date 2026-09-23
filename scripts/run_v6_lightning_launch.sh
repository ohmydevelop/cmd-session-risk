#!/usr/bin/env bash
# Launch the v6 chain on a Lightning Studio and monitor its exit sentinel.
# Usage:
#   LIGHTNING_HOST=s_xxx@ssh.lightning.ai scripts/run_v6_lightning_launch.sh
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${LIGHTNING_HOST:-s_01m2tmwd7v1mc00aayt5vf82hr@ssh.lightning.ai}"
REMOTE_ROOT="${REMOTE_ROOT:-/teamspace/studios/this_studio/cmdrisk-v6}"
SSH_KEY="${LIGHTNING_SSH_KEY:-}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15
  -o KexAlgorithms=curve25519-sha256 -o HostKeyAlgorithms=rsa-sha2-512)
[[ -n "$SSH_KEY" ]] && SSH_OPTS+=(-i "$SSH_KEY")

ssh "${SSH_OPTS[@]}" "$HOST" "true" >/dev/null 2>&1 \
  || { echo "cannot reach $HOST yet" >&2; exit 1; }

REMOTE_DATA="/tmp/cmdrisk-v6-data"
ssh "${SSH_OPTS[@]}" "$HOST" "mkdir -p '$REMOTE_ROOT' '$REMOTE_DATA'"

echo "[1/3] rsync project tree (no git/venv/artifacts/data)"
rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'artifacts' --exclude 'data/' --exclude 'remote_environment.json' \
  --exclude '*.pem' --exclude '*.key' \
  "$LOCAL_ROOT/" "$HOST:$REMOTE_ROOT/"

echo "[2/3] rsync data + base checkpoint"
rsync -az -e "ssh ${SSH_OPTS[*]}" \
  "$LOCAL_ROOT/data/processed/v4" \
  "$LOCAL_ROOT/data/processed/v6_obf" \
  "$LOCAL_ROOT/data/processed/v6_obf_evals" \
  "$LOCAL_ROOT/artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922/short_phase" \
  "$HOST:$REMOTE_DATA/" \
  || true
# the second rsync of many sources is ambiguous across hosts; fall back to
# individual transfers on failure
for d in v4 v6_obf v6_obf_evals; do
  rsync -az -e "ssh ${SSH_OPTS[*]}" \
    "$LOCAL_ROOT/data/processed/$d" "$HOST:$REMOTE_DATA/$d" >/dev/null 2>&1 || true
done
rsync -az -e "ssh ${SSH_OPTS[*]}" \
  "$LOCAL_ROOT/artifacts/cmd-session-risk-v5-l8-h256-million-gpu-20260922/short_phase" \
  "$HOST:$REMOTE_DATA/short_phase" >/dev/null 2>&1 || true

echo "[3/3] launch chain detached"
ssh "${SSH_OPTS[@]}" "$HOST" "cd '$REMOTE_ROOT' && uv sync --extra train >/dev/null 2>&1 || true; \
  nohup env DATA='$REMOTE_DATA/v6_obf' \
            EVAL_DIR='$REMOTE_DATA/v6_obf_evals' \
            BASE_CHECKPOINT='$REMOTE_DATA/short_phase' \
            MODEL_DIR='$REMOTE_DATA/short_phase' \
    ./scripts/run_v6_obf_chain.sh >'$REMOTE_ROOT/logs_v6_nohup.log' 2>&1 & echo launched"
echo "launched; poll with: ssh $HOST 'tail -n 20 $REMOTE_ROOT/logs_v6_nohup.log'"

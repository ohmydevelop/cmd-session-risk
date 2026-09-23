#!/usr/bin/env bash
# Pull the finished v6 artifacts + posttrain outputs back from the Lightning box.
set -euo pipefail
HOST="${LIGHTNING_HOST:-s_01m2tmwd7v1mc00aayt5vf82hr@ssh.lightning.ai}"
REMOTE_ROOT="${REMOTE_ROOT:-/teamspace/studios/this_studio/cmdrisk-v6}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15)

RUN_DIR="$1"   # e.g. artifacts/cmd-session-risk-v6-l8-h256-obf-20260923
PROD_DIR="$2"  # e.g. artifacts/...-recal-prod-session1p-20260923
WORK_DIR="${3:-/tmp/recal_v6}"

mkdir -p "$LOCAL_ROOT/$RUN_DIR"
rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude 'checkpoint-*' \
  "$HOST:$REMOTE_ROOT/$RUN_DIR/" "$LOCAL_ROOT/$RUN_DIR/"
mkdir -p "$LOCAL_ROOT/$PROD_DIR"
rsync -az -e "ssh ${SSH_OPTS[*]}" --exclude 'checkpoint-*' --exclude 'short_phase' \
  "$HOST:$REMOTE_ROOT/$PROD_DIR/" "$LOCAL_ROOT/$PROD_DIR/"
mkdir -p "$LOCAL_ROOT/work/recal_v6"
rsync -az -e "ssh ${SSH_OPTS[*]}" "$HOST:$WORK_DIR/" "$LOCAL_ROOT/work/recal_v6/"
echo "COLLECTED: $RUN_DIR $PROD_DIR work/recal_v6"

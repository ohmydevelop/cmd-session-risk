#!/usr/bin/env bash
# Idempotently configure SSH access to a Lightning Studio.
set -euo pipefail

HOME_DIR="${HOME:?HOME is not set}"
SSH_DIR="${HOME_DIR}/.ssh"
SSH_CONFIG="${SSH_DIR}/config"
SSH_KEY="${LIGHTNING_SSH_KEY:-${SSH_DIR}/lightning_rsa}"
SSH_USER="${LIGHTNING_SSH_USER:-}"

has_host_config() {
  [[ -f "${SSH_CONFIG}" ]] || return 1
  grep -Eq '^[[:space:]]*Host[[:space:]]+.*(^|[[:space:]])lightning\.ai([[:space:]]|$)' "${SSH_CONFIG}" \
    || grep -Eq '^[[:space:]]*HostName[[:space:]]+ssh\.lightning\.ai[[:space:]]*$' "${SSH_CONFIG}"
}

if [[ -f "${SSH_KEY}" ]] && has_host_config; then
  printf '%s\n' "Lightning SSH key and host config already exist; skipping bootstrap."
  exit 0
fi

if [[ ! -f "${SSH_KEY}" ]]; then
  if [[ -z "${LIGHTNING_SETUP_URL:-}" ]]; then
    printf '%s\n' "LIGHTNING_SETUP_URL is required when ${SSH_KEY} does not exist." >&2
    exit 1
  fi
  mkdir -p "${SSH_DIR}"
  chmod 700 "${SSH_DIR}"
  printf '%s\n' "Configuring Lightning SSH access."
  curl -fsSL "${LIGHTNING_SETUP_URL}" | bash
fi

if [[ ! -f "${SSH_KEY}" ]]; then
  printf '%s\n' "Lightning bootstrap completed but ${SSH_KEY} was not created." >&2
  exit 1
fi
chmod 600 "${SSH_KEY}"

if ! has_host_config; then
  if [[ -z "${SSH_USER}" ]]; then
    printf '%s\n' "Host config is missing. Set LIGHTNING_SSH_USER to append a lightning.ai entry." >&2
    exit 1
  fi
  touch "${SSH_CONFIG}"
  chmod 600 "${SSH_CONFIG}"
  if ! has_host_config; then
    {
      printf '\n# cmd-session-risk: managed Lightning config\n'
      printf 'Host lightning.ai\n'
      printf '    HostName ssh.lightning.ai\n'
      printf '    User %s\n' "${SSH_USER}"
      printf '    IdentityFile %s\n' "${SSH_KEY}"
      printf '    IdentitiesOnly yes\n'
      printf '    ServerAliveInterval 30\n'
      printf '    ServerAliveCountMax 6\n'
    } >>"${SSH_CONFIG}"
  fi
fi

printf '%s\n' "Lightning SSH configuration is ready."

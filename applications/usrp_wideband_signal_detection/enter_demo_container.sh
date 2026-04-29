#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_repo_guard.sh"
EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${SCRIPT_DIR}")}

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}

resolve_exec_uid() {
  if [[ -n "${EXEC_UID:-}" ]]; then
    echo "${EXEC_UID}"
    return
  fi

  if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    stat -c '%u' "${XAUTHORITY}"
    return
  fi

  if [[ -n "${SUDO_UID:-}" ]]; then
    echo "${SUDO_UID}"
    return
  fi

  id -u
}

resolve_exec_gid() {
  if [[ -n "${EXEC_GID:-}" ]]; then
    echo "${EXEC_GID}"
    return
  fi

  if [[ -n "${XAUTHORITY:-}" && -f "${XAUTHORITY}" ]]; then
    stat -c '%g' "${XAUTHORITY}"
    return
  fi

  if [[ -n "${SUDO_GID:-}" ]]; then
    echo "${SUDO_GID}"
    return
  fi

  id -g
}

EXEC_UID=$(resolve_exec_uid)
EXEC_GID=$(resolve_exec_gid)
XDG_RUNTIME_DIR_VALUE=${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-${EXEC_UID}}

if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  exit 1
fi

ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "Container is stopped; starting ${CONTAINER_NAME}."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
fi

sudo docker exec -u 0:0 "${CONTAINER_NAME}" bash -lc "mkdir -p '${XDG_RUNTIME_DIR_VALUE}' && chown '${EXEC_UID}:${EXEC_GID}' '${XDG_RUNTIME_DIR_VALUE}' && chmod 700 '${XDG_RUNTIME_DIR_VALUE}'"

exec sudo docker exec -it \
  -u "${EXEC_UID}:${EXEC_GID}" \
  -e DISPLAY="${DISPLAY:-}" \
  -e XAUTHORITY="${XAUTHORITY:-}" \
  -e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR_VALUE}" \
  -e USER="${USER:-}" \
  -e LOGNAME="${LOGNAME:-${USER:-}}" \
  -e HOME=/tmp \
  -e PS1='container$ ' \
  -w /workspace/holohub \
  "${CONTAINER_NAME}" bash --noprofile --norc -i

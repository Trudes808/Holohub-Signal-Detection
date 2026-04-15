#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_repo_guard.sh"
EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${SCRIPT_DIR}")}

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}

if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  exit 1
fi

ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "Container is stopped; starting ${CONTAINER_NAME}."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
fi

exec sudo docker exec -it "${CONTAINER_NAME}" bash
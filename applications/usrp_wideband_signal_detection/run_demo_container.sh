#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_repo_guard.sh"
EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${SCRIPT_DIR}")}

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  echo "Run ./build_demo_container.sh first to create and provision it." >&2
  exit 1
fi

ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
  echo "Container ${CONTAINER_NAME} is already running."
else
  sudo docker start "${CONTAINER_NAME}" >/dev/null
  echo "Started container ${CONTAINER_NAME}."
fi

echo "Open a shell with: ./enter_demo_container.sh"
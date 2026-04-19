#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

canon_dir() {
  local target_dir=$1
  python3 - "$target_dir" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

expected_repo_root_from_script_dir() {
  local script_dir=$1
  canon_dir "${script_dir}/../.."
}

container_mount_source_for_workspace() {
  local container_name=$1
  sudo docker inspect -f '{{range .Mounts}}{{if eq .Destination "/workspace/holohub"}}{{.Source}}{{end}}{{end}}' "${container_name}" 2>/dev/null
}

ensure_container_repo_mount_matches() {
  local container_name=$1
  local expected_repo_root=$2
  local actual_mount_source
  local actual_mount_root
  local expected_mount_root

  expected_mount_root=$(canon_dir "${expected_repo_root}")
  actual_mount_source=$(container_mount_source_for_workspace "${container_name}")

  if [[ -z "${actual_mount_source}" ]]; then
    echo "Container ${container_name} does not mount /workspace/holohub." >&2
    echo "Recreate it with ./build_demo_container.sh from ${expected_mount_root}." >&2
    exit 1
  fi

  actual_mount_root=$(canon_dir "${actual_mount_source}")

  if [[ "${actual_mount_root}" != "${expected_mount_root}" ]]; then
    echo "Container ${container_name} is mounted from the wrong checkout." >&2
    echo "Expected: ${expected_mount_root}" >&2
    echo "Actual:   ${actual_mount_root}" >&2
    echo "Recreate it with ./build_demo_container.sh from ${expected_mount_root}." >&2
    exit 1
  fi
}

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
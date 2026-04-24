#!/usr/bin/env bash

if [[ -n "${USRP_WIDEBAND_CONTAINER_REPO_GUARD_SH:-}" ]]; then
  return 0
fi
USRP_WIDEBAND_CONTAINER_REPO_GUARD_SH=1

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
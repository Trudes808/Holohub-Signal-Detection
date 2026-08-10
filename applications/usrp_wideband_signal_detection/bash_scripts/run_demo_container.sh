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

APP_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${APP_DIR}")}

source "${SCRIPT_DIR}/container_env.sh"

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  echo "Run ./build_demo_container.sh first to create and provision it." >&2
  exit 1
fi

ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"

container_has_env_value() {
  local key=$1
  local value=$2
  sudo docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${CONTAINER_NAME}" | grep -Fx "${key}=${value}" >/dev/null
}

container_has_mount_destination() {
  local destination=$1
  sudo docker inspect -f '{{range .Mounts}}{{println .Destination}}{{end}}' "${CONTAINER_NAME}" | grep -Fx "${destination}" >/dev/null
}

container_has_mount_source() {
  local source=$1
  sudo docker inspect -f '{{range .Mounts}}{{println .Source}}{{end}}' "${CONTAINER_NAME}" | grep -Fx "${source}" >/dev/null
}

require_current_display_forwarding() {
  local requested_display=${DISPLAY:-}
  local x11_socket_dir=/tmp/.X11-unix

  if [[ -z "${requested_display}" ]]; then
    return
  fi

  # The X11 socket directory is bind-mounted as a whole (see build_demo_container.sh), so it always
  # exposes the CURRENT session's socket regardless of the display number. That mount -- not the
  # container's baked DISPLAY env -- is the real requirement for visualization.
  if [[ -d "${x11_socket_dir}" ]] && ! container_has_mount_destination "${x11_socket_dir}"; then
    echo "Container ${CONTAINER_NAME} is missing the ${x11_socket_dir} mount needed for visualization." >&2
    echo "Recreate it from a desktop session with:" >&2
    echo "  SKIP_IMAGE_BUILD=1 sudo -E ./build_demo_container.sh" >&2
    exit 1
  fi

  # A baked DISPLAY (or XAUTHORITY) that differs from the current session is NOT a problem: the run
  # wrappers pass `-e DISPLAY=$DISPLAY` / `-e XAUTHORITY` at `docker exec` time, so the app always
  # targets the current display. This lets the container survive a reboot that changes the X display
  # number without a recreate (host-side `xhost +local:root` in after_reboot.sh grants access).
  if ! container_has_env_value DISPLAY "${requested_display}"; then
    echo "Note: ${CONTAINER_NAME} was created with a different DISPLAY than the current ${requested_display};" >&2
    echo "      the run wrappers override DISPLAY at exec time, so no recreate is needed for visualization." >&2
  fi
}

require_current_display_forwarding

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
  echo "Container ${CONTAINER_NAME} is already running."
else
  sudo docker start "${CONTAINER_NAME}" >/dev/null
  echo "Started container ${CONTAINER_NAME}."
fi

echo "Open a shell with: ./enter_demo_container.sh"
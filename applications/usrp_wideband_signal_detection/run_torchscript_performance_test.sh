#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}

if [[ $# -gt 1 ]]; then
	echo "Usage: $0 [config-name.yaml]" >&2
	exit 1
fi

CONFIG_NAME=${1:-${CONFIG_NAME:-config_torchscript_performance.yaml}}

echo "Running usrp_wideband_signal_detection with config: ${CONFIG_NAME}"

"${SCRIPT_DIR}/run_demo_container.sh"

existing_processes=$(sudo docker exec "${CONTAINER_NAME}" bash -lc "pgrep -af '(^|/)usrp_wideband_signal_detection( |$)' || true")
if [[ -n "${existing_processes}" ]]; then
	echo "Another usrp_wideband_signal_detection instance is already running in ${CONTAINER_NAME}:" >&2
	echo "${existing_processes}" >&2
	echo "Stop the stale process before starting a new run." >&2
	echo "Suggested cleanup: sudo docker exec ${CONTAINER_NAME} pkill -f '(^|/)usrp_wideband_signal_detection( |$)'" >&2
	exit 1
fi

exec sudo docker exec -it "${CONTAINER_NAME}" bash -lc "cd ${BUILD_APP_DIR} && ./usrp_wideband_signal_detection ${CONFIG_NAME}"
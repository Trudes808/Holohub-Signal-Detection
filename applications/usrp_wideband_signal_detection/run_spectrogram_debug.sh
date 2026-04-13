#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
CONFIG_NAME=${CONFIG_NAME:-config_spectrogram_debug.yaml}

"${SCRIPT_DIR}/run_demo_container.sh"

exec sudo docker exec -it "${CONTAINER_NAME}" bash -lc "cd ${BUILD_APP_DIR} && ./usrp_wideband_signal_detection ${CONFIG_NAME}"
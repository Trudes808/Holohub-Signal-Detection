#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

APP_NAME=${APP_NAME:-usrp_wideband_signal_detection}
IMAGE_NAME=${IMAGE_NAME:-usrp_x410_signal_detection_demo:latest}
DOCKER_FILE=${DOCKER_FILE:-applications/usrp_freq_detection/Dockerfile}

cd "${REPO_ROOT}"

exec sudo ./holohub build-container "${APP_NAME}" --docker-file "${DOCKER_FILE}" --img "${IMAGE_NAME}" "$@"
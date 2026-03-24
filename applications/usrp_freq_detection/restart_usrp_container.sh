#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

CONTAINER_NAME=${CONTAINER_NAME:-holohub_usrp_freq_detection}
IMAGE_NAME=${IMAGE_NAME:-holohub:usrp_freq_detection}
HUGEPAGES_DIR=${HUGEPAGES_DIR:-/dev/hugepages}

cd "${REPO_ROOT}"

sudo ./holohub build-container usrp_freq_detection
sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

exec sudo docker run \
  --name "${CONTAINER_NAME}" \
  --privileged \
  --net host \
  --interactive \
  --tty \
  -u 0:0 \
  -v "${REPO_ROOT}:/workspace/holohub" \
  -v "${HUGEPAGES_DIR}:/dev/hugepages" \
  -w /workspace/holohub \
  --runtime nvidia \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
  "${IMAGE_NAME}"

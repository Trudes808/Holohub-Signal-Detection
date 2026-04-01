#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
IMAGE_NAME=${IMAGE_NAME:-usrp_x410_signal_detection_demo:latest}
HUGEPAGES_DIR=${HUGEPAGES_DIR:-/dev/hugepages}
DETACH=${DETACH:-1}

cd "${REPO_ROOT}"

sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_RUN_CMD=(
  sudo docker run
  --name "${CONTAINER_NAME}" \
  --privileged \
  --net host \
  -u 0:0 \
  -e HOLOHUB_BUILD_LOCAL=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
  -v "${REPO_ROOT}:/workspace/holohub" \
  -v /dev:/dev \
  -v "${HUGEPAGES_DIR}:/dev/hugepages" \
  -w /workspace/holohub \
  --runtime nvidia \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
)

if [[ "${DETACH}" == "1" ]]; then
  DOCKER_RUN_CMD+=(
    --detach
    "${IMAGE_NAME}"
    bash
    -lc
    "trap : TERM INT; sleep infinity & wait"
  )
  "${DOCKER_RUN_CMD[@]}" >/dev/null
  echo "Started container ${CONTAINER_NAME} from image ${IMAGE_NAME}."
  echo "Attach with: sudo docker exec -it ${CONTAINER_NAME} bash"
  exit 0
fi

DOCKER_RUN_CMD+=(
  --interactive
  --tty
  "${IMAGE_NAME}"
)

if [[ "$#" -gt 0 ]]; then
  DOCKER_RUN_CMD+=("$@")
else
  DOCKER_RUN_CMD+=(bash)
fi

exec "${DOCKER_RUN_CMD[@]}"
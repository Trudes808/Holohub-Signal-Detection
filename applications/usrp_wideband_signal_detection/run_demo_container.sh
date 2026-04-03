#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
IMAGE_NAME=${IMAGE_NAME:-usrp_x410_signal_detection_demo:latest}
HUGEPAGES_DIR=${HUGEPAGES_DIR:-/dev/hugepages}
SPECTROGRAM_HOST_DIR=${SPECTROGRAM_HOST_DIR:-/tmp/usrp_spectrograms}
DINO_MASK_HOST_DIR=${DINO_MASK_HOST_DIR:-/tmp/usrp_dino_masks}
DETACH=${DETACH:-1}
DISPLAY_VALUE=${DISPLAY:-}
XAUTHORITY_VALUE=${XAUTHORITY:-}
X11_SOCKET_DIR=/tmp/.X11-unix

cd "${REPO_ROOT}"

mkdir -p "${SPECTROGRAM_HOST_DIR}" "${DINO_MASK_HOST_DIR}"

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
  -v "${SPECTROGRAM_HOST_DIR}:/workspace/spectrograms" \
  -v "${DINO_MASK_HOST_DIR}:/workspace/dino_masks" \
  -v /dev:/dev \
  -v "${HUGEPAGES_DIR}:/dev/hugepages" \
  -w /workspace/holohub \
  --runtime nvidia \
  --gpus all \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --ipc=host \
)

if [[ -n "${DISPLAY_VALUE}" && -d "${X11_SOCKET_DIR}" ]]; then
  DOCKER_RUN_CMD+=(
    -e DISPLAY="${DISPLAY_VALUE}" \
    -v "${X11_SOCKET_DIR}:${X11_SOCKET_DIR}:rw"
  )

  if [[ -n "${XAUTHORITY_VALUE}" && -f "${XAUTHORITY_VALUE}" ]]; then
    DOCKER_RUN_CMD+=(
      -e XAUTHORITY="${XAUTHORITY_VALUE}" \
      -v "${XAUTHORITY_VALUE}:${XAUTHORITY_VALUE}:ro"
    )
  fi
fi

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
  echo "Host spectrogram output: ${SPECTROGRAM_HOST_DIR}"
  echo "Host DINO mask output: ${DINO_MASK_HOST_DIR}"
  if [[ -n "${DISPLAY_VALUE}" && -d "${X11_SOCKET_DIR}" ]]; then
    echo "Display forwarding enabled for DISPLAY=${DISPLAY_VALUE}."
  else
    echo "Display forwarding not configured; HoloViz viewers will fail unless you launch from a desktop session with DISPLAY set."
  fi
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
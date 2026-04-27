#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_repo_guard.sh"

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${SCRIPT_DIR}")}
ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "Container ${CONTAINER_NAME} is not running. Start it with ./run_demo_container.sh or ./build_demo_container.sh." >&2
  exit 1
fi

CONTAINER_DINOV3_ROOT=${CONTAINER_DINOV3_ROOT:-/workspace/models/dinov3}
CONTAINER_WEIGHT_PATH=${CONTAINER_WEIGHT_PATH:-${CONTAINER_DINOV3_ROOT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
CONTAINER_TRT_EXPORT_SCRIPT=${CONTAINER_TRT_EXPORT_SCRIPT:-/workspace/holohub/applications/usrp_wideband_signal_detection/export_dinov3_single_channel_trt.py}
CONTAINER_TRT_ONNX_PATH=${CONTAINER_TRT_ONNX_PATH:-${CONTAINER_DINOV3_ROOT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.single_channel.onnx}
CONTAINER_TRT_ENGINE_PATH=${CONTAINER_TRT_ENGINE_PATH:-${CONTAINER_DINOV3_ROOT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.single_channel.fp16.engine}
INPUT_HEIGHT=${INPUT_HEIGHT:-256}
INPUT_WIDTH=${INPUT_WIDTH:-512}
MODEL_NAME=${MODEL_NAME:-dinov3_vitb16}
TENSORRT_APT_VERSION=${TENSORRT_APT_VERSION:-}

if ! sudo docker exec "${CONTAINER_NAME}" python3 -c 'import onnx, onnxscript' >/dev/null 2>&1; then
  echo "Installing ONNX export dependencies into ${CONTAINER_NAME}."
  sudo docker exec "${CONTAINER_NAME}" bash -lc 'export PIP_BREAK_SYSTEM_PACKAGES=1 && python3 -m pip install --no-cache-dir onnx onnxscript'
fi

if ! sudo docker exec "${CONTAINER_NAME}" bash -lc 'command -v trtexec >/dev/null 2>&1'; then
  echo "Installing TensorRT tooling into ${CONTAINER_NAME}."
  if [[ -n "${TENSORRT_APT_VERSION}" ]]; then
    sudo docker exec "${CONTAINER_NAME}" bash -lc "set -euo pipefail && apt-get update && apt-get install -y --allow-downgrades --no-install-recommends libnvinfer-bin=${TENSORRT_APT_VERSION} libnvinfer-lean10=${TENSORRT_APT_VERSION} libnvinfer-dispatch10=${TENSORRT_APT_VERSION} libnvinfer-vc-plugin10=${TENSORRT_APT_VERSION} libnvinfer10=${TENSORRT_APT_VERSION} libnvinfer-headers-dev=${TENSORRT_APT_VERSION} libnvinfer-dev=${TENSORRT_APT_VERSION} libnvinfer-headers-plugin-dev=${TENSORRT_APT_VERSION} libnvinfer-plugin10=${TENSORRT_APT_VERSION} libnvinfer-plugin-dev=${TENSORRT_APT_VERSION} libnvinfer-lean-dev=${TENSORRT_APT_VERSION} libnvinfer-dispatch-dev=${TENSORRT_APT_VERSION} libnvinfer-vc-plugin-dev=${TENSORRT_APT_VERSION} libnvonnxparsers10=${TENSORRT_APT_VERSION} libnvonnxparsers-dev=${TENSORRT_APT_VERSION}"
  else
    sudo docker exec "${CONTAINER_NAME}" bash -lc 'set -euo pipefail && apt-get update && apt-get install -y --no-install-recommends libnvinfer-bin libnvinfer-lean10 libnvinfer-dispatch10 libnvinfer-vc-plugin10 libnvinfer10 libnvinfer-headers-dev libnvinfer-dev libnvinfer-headers-plugin-dev libnvinfer-plugin10 libnvinfer-plugin-dev libnvinfer-lean-dev libnvinfer-dispatch-dev libnvinfer-vc-plugin-dev libnvonnxparsers10 libnvonnxparsers-dev'
  fi
fi

sudo docker exec "${CONTAINER_NAME}" python3 "${CONTAINER_TRT_EXPORT_SCRIPT}" \
  --model-repo "${CONTAINER_DINOV3_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --weights-path "${CONTAINER_WEIGHT_PATH}" \
  --output-onnx "${CONTAINER_TRT_ONNX_PATH}" \
  --output-engine "${CONTAINER_TRT_ENGINE_PATH}" \
  --input-height "${INPUT_HEIGHT}" \
  --input-width "${INPUT_WIDTH}" \
  --build-engine \
  "$@"

echo "TensorRT ONNX path: ${CONTAINER_TRT_ONNX_PATH}"
echo "TensorRT engine path: ${CONTAINER_TRT_ENGINE_PATH}"
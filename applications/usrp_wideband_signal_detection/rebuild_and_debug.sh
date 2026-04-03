#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace/holohub}
BUILD_DIR=${BUILD_DIR:-build/usrp_wideband_signal_detection}
APP_NAME=${APP_NAME:-usrp_wideband_signal_detection}
VISUALIZER_NAME=${VISUALIZER_NAME:-offline_spectrogram_visualizer}
CPU_SCRIPT=${CPU_SCRIPT:-/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be_cpu.ts}
CUDA_SCRIPT=${CUDA_SCRIPT:-/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts}
CPU_CONFIG=${CPU_CONFIG:-config_torchscript_cpu_eval.yaml}
CUDA_CONFIG=${CUDA_CONFIG:-config_torchscript_validation.yaml}
RUN_SECONDS=${RUN_SECONDS:-20}
SKIP_CUDA_APP=${SKIP_CUDA_APP:-0}
SKIP_TORCH_DEBUG=${SKIP_TORCH_DEBUG:-0}

torch_available_in_container() {
  sudo docker exec "${CONTAINER_NAME}" bash -lc 'python3 - <<"PY"
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") is not None else 1)
PY'
}

run_in_container() {
  sudo docker exec "${CONTAINER_NAME}" bash -lc "$1"
}

run_timed_app() {
  local config_path=$1
  local label=$2
  local command="cd ${WORKSPACE_DIR}/${BUILD_DIR}/applications/${APP_NAME} && timeout --signal=INT ${RUN_SECONDS}s ./${APP_NAME} --config ${config_path}"

  set +e
  run_in_container "set -euo pipefail && ${command}"
  local status=$?
  set -e

  if [[ ${status} -ne 0 && ${status} -ne 124 && ${status} -ne 130 ]]; then
    echo "${label} failed with exit code ${status}" >&2
    exit ${status}
  fi

  if [[ ${status} -eq 124 ]]; then
    echo "${label} reached the ${RUN_SECONDS}s timeout without crashing."
  elif [[ ${status} -eq 130 ]]; then
    echo "${label} exited after timeout sent SIGINT."
  else
    echo "${label} completed with exit code ${status}."
  fi
}

if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  echo "Start it first with applications/usrp_wideband_signal_detection/run_demo_container.sh" >&2
  exit 1
fi

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "Container is stopped; starting ${CONTAINER_NAME}."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
fi

if [[ "${SKIP_TORCH_DEBUG}" != "1" ]] && ! torch_available_in_container; then
  echo "Torch is not installed in ${CONTAINER_NAME}; skipping Torch-dependent detector rebuild/debug steps." >&2
  echo "Run applications/usrp_wideband_signal_detection/setup_demo_container.sh to install the pinned Torch stack if you want Torch validation." >&2
  SKIP_TORCH_DEBUG=1
fi

echo "==> Regenerating CMake build tree"
run_in_container "set -euo pipefail && cd ${WORKSPACE_DIR} && cmake -S . -B ${BUILD_DIR}"

echo "==> Cleaning prior build outputs"
run_in_container "set -euo pipefail && cd ${WORKSPACE_DIR} && cmake --build ${BUILD_DIR} --target clean"

if [[ "${SKIP_TORCH_DEBUG}" == "1" ]]; then
  echo "==> Rebuilding app and offline visualizer only (Torch-dependent targets skipped)"
  run_in_container "set -euo pipefail && cd ${WORKSPACE_DIR} && cmake --build ${BUILD_DIR} --target ${APP_NAME} ${VISUALIZER_NAME} -j"
else
  echo "==> Rebuilding detector, sandbox, app, and offline visualizer"
  run_in_container "set -euo pipefail && cd ${WORKSPACE_DIR} && cmake --build ${BUILD_DIR} --target ${APP_NAME} ${VISUALIZER_NAME} dinov3_signal_detector dinov3_libtorch_sandbox -j"

  echo "==> Running CPU sandbox"
  run_in_container "set -euo pipefail && ${WORKSPACE_DIR}/${BUILD_DIR}/operators/dinov3_signal_detector/dinov3_libtorch_sandbox --script ${CPU_SCRIPT} --mode cpu"

  echo "==> Running CUDA sandbox"
  run_in_container "set -euo pipefail && ${WORKSPACE_DIR}/${BUILD_DIR}/operators/dinov3_signal_detector/dinov3_libtorch_sandbox --script ${CUDA_SCRIPT} --mode cuda"

  echo "==> Running CPU TorchScript app validation"
  run_timed_app "${CPU_CONFIG}" "CPU validation"

  if [[ "${SKIP_CUDA_APP}" != "1" ]]; then
    echo "==> Running CUDA TorchScript app validation"
    run_timed_app "${CUDA_CONFIG}" "CUDA validation"
  fi
fi

echo "==> Rebuild and debug sequence finished"
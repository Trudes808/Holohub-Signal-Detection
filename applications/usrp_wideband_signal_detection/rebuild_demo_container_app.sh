#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_repo_guard.sh"
EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${SCRIPT_DIR}")}

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
WORKSPACE_DIR=${WORKSPACE_DIR:-/workspace/holohub}
BUILD_DIR=${BUILD_DIR:-build/usrp_wideband_signal_detection}
APP_NAME=${APP_NAME:-usrp_wideband_signal_detection}
VISUALIZER_NAME=${VISUALIZER_NAME:-offline_spectrogram_visualizer}
COHERENT_VALIDATOR_NAME=${COHERENT_VALIDATOR_NAME:-offline_coherent_power_validator}
DINO_VALIDATOR_NAME=${DINO_VALIDATOR_NAME:-offline_dino_validator}
VISUALIZER_TARGET=${VISUALIZER_TARGET:-applications/${APP_NAME}/${VISUALIZER_NAME}}
COHERENT_VALIDATOR_TARGET=${COHERENT_VALIDATOR_TARGET:-applications/${APP_NAME}/${COHERENT_VALIDATOR_NAME}}
DINO_VALIDATOR_TARGET=${DINO_VALIDATOR_TARGET:-applications/${APP_NAME}/${DINO_VALIDATOR_NAME}}
MATX_DIR=${MATX_DIR:-/usr/local/lib/cmake/matx}
BUILD_APP_DIR=${BUILD_APP_DIR:-${WORKSPACE_DIR}/${BUILD_DIR}/applications/${APP_NAME}}
SOURCE_APP_DIR=${SOURCE_APP_DIR:-${WORKSPACE_DIR}/applications/${APP_NAME}}
FORCE_REBUILD=${FORCE_REBUILD:-0}

run_in_container() {
  sudo docker exec "${CONTAINER_NAME}" bash -lc "$1"
}

ensure_vulkan_runtime() {
  run_in_container 'set -euo pipefail
if ! ldconfig -p | grep -q "libvulkan.so.1"; then
  apt-get update
  apt-get install -y --no-install-recommends libvulkan1
fi'
}

torch_available_in_container() {
  sudo docker exec "${CONTAINER_NAME}" bash -lc 'python3 - <<"PY"
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") is not None else 1)
PY'
}

build_targets() {
  if torch_available_in_container >/dev/null 2>&1; then
    echo "${APP_NAME} ${VISUALIZER_TARGET} ${COHERENT_VALIDATOR_TARGET} ${DINO_VALIDATOR_TARGET} coherent_power_signal_detector dinov3_signal_detector dinov3_libtorch_sandbox"
  else
    echo "${APP_NAME} ${VISUALIZER_TARGET} ${COHERENT_VALIDATOR_TARGET} ${DINO_VALIDATOR_TARGET} coherent_power_signal_detector"
  fi
}

ninja_target_exists() {
  local target=$1
  run_in_container "set -euo pipefail && ninja -C ${WORKSPACE_DIR}/${BUILD_DIR} -t targets all | cut -d: -f1 | grep -Fx -- '${target}' >/dev/null"
}

build_auxiliary_targets() {
  local targets
  local resolved_targets=()
  local target

  targets=$(build_targets)

  for target in ${targets}; do
    if ninja_target_exists "${target}"; then
      resolved_targets+=("${target}")
    else
      echo "==> Skipping auxiliary build target not present in this Ninja graph: ${target}" >&2
    fi
  done

  if [[ ${#resolved_targets[@]} -eq 0 ]]; then
    echo "==> No auxiliary Ninja targets were present in the generated build tree" >&2
    return 0
  fi

  run_in_container "set -euo pipefail && ninja -C ${WORKSPACE_DIR}/${BUILD_DIR} ${resolved_targets[*]}"
}

build_tree_uses_torch_stub() {
  run_in_container "test -f ${WORKSPACE_DIR}/${BUILD_DIR}/build.ninja"

  sudo docker exec "${CONTAINER_NAME}" bash -lc "grep -q 'dinov3_torch_runtime_stub.cpp' ${WORKSPACE_DIR}/${BUILD_DIR}/build.ninja"
}

clear_build_tree() {
  run_in_container "rm -rf ${WORKSPACE_DIR}/${BUILD_DIR}"
}

needs_rebuild() {
  local targets
  local dry_run_output
  local status

  if [[ "${FORCE_REBUILD}" == "1" ]]; then
    echo "==> Rebuild forced by FORCE_REBUILD=1"
    return 0
  fi

  if ! run_in_container "test -x ${BUILD_APP_DIR}/${APP_NAME}"; then
    echo "==> Rebuild required because the app binary is missing"
    return 0
  fi

  if ! run_in_container "test -f ${WORKSPACE_DIR}/${BUILD_DIR}/build.ninja"; then
    echo "==> Rebuild required because the Ninja build tree is missing"
    return 0
  fi

  targets=$(build_targets)

  set +e
  dry_run_output=$(sudo docker exec "${CONTAINER_NAME}" bash -lc "set -euo pipefail && ninja -C ${WORKSPACE_DIR}/${BUILD_DIR} -n ${targets}" 2>&1)
  status=$?
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "==> Rebuild required because the dry-run build check could not confirm the current build state"
    return 0
  fi

  if grep -qi "no work to do" <<< "${dry_run_output}"; then
    echo "==> No rebuild needed; tracked targets are already up to date"
    return 1
  fi

  echo "==> Rebuild required because source or build inputs changed"
  return 0
}

sync_runtime_configs() {
  run_in_container "set -euo pipefail && cp ${SOURCE_APP_DIR}/config*.yaml ${BUILD_APP_DIR}/"
}

if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  echo "Create it first with ./build_demo_container.sh" >&2
  exit 1
fi

ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "Container is stopped; starting ${CONTAINER_NAME}."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
fi

ensure_vulkan_runtime

if ! sudo docker exec "${CONTAINER_NAME}" bash -lc 'python3 - <<"PY"
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("torch")
if spec is None or spec.origin is None:
    raise SystemExit(1)

torch_dir = Path(spec.origin).resolve().parent
torch_config = torch_dir / "share" / "cmake" / "Torch" / "TorchConfig.cmake"
if not torch_config.exists():
    raise SystemExit(2)
PY'; then
  echo "Warning: container PyTorch CMake files are missing; the rebuild will compile dinov3_signal_detector without Torch support." >&2
  echo "Recreate the container with ./build_demo_container.sh if you need the Torch-enabled path." >&2
fi

if torch_available_in_container >/dev/null 2>&1; then
  if build_tree_uses_torch_stub >/dev/null 2>&1; then
    echo "==> Existing build tree is configured for dinov3_torch_runtime_stub.cpp; clearing it so CMake can reconfigure with Torch support"
    clear_build_tree
  fi
fi

if needs_rebuild; then
  run_in_container "set -euo pipefail && \
    cd ${WORKSPACE_DIR} && \
    export HOLOHUB_BUILD_LOCAL=1 && \
    ./holohub build ${APP_NAME} --local --configure-args=-Dmatx_DIR=${MATX_DIR}"
fi

build_auxiliary_targets

sync_runtime_configs

run_in_container "ls -lah ${BUILD_APP_DIR}"
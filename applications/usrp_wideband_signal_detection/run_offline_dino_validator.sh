#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_REF_DIR=$(dirname "${BASH_SOURCE[0]}")
WORKING_DIR=$(pwd -P)
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
DEFAULT_CONFIG=${DEFAULT_CONFIG:-${SCRIPT_DIR}/config_torchscript_validation_capture_single_channel.yaml}
RUN_DEMO_CONTAINER=${RUN_DEMO_CONTAINER:-${SCRIPT_REF_DIR}/run_demo_container.sh}
VALIDATOR_BIN=${VALIDATOR_BIN:-}

host_repo_root=${HOST_REPO_ROOT:-$(dirname "$(dirname "${SCRIPT_DIR}")")}

absolutize_host_path() {
  local raw_path=$1
  python3 - "$WORKING_DIR" "$raw_path" <<'PY'
import os
import sys

cwd = sys.argv[1]
raw = sys.argv[2]
print(os.path.realpath(raw if os.path.isabs(raw) else os.path.join(cwd, raw)))
PY
}

map_path_to_container() {
  local raw_path=$1
  if [[ -z "${raw_path}" ]]; then
    return 0
  fi
  case "${raw_path}" in
    /tmp/usrp_spectrograms/*)
      printf '/workspace/spectrograms/%s' "${raw_path#/tmp/usrp_spectrograms/}"
      ;;
    /tmp/usrp_dino_masks/*)
      printf '/workspace/dino_masks/%s' "${raw_path#/tmp/usrp_dino_masks/}"
      ;;
    ${host_repo_root}/*)
      printf '/workspace/holohub/%s' "${raw_path#${host_repo_root}/}"
      ;;
    *)
      printf '%s' "${raw_path}"
      ;;
  esac
}

tensor_path=${TENSOR_PATH:-}
config_path=${CONFIG_PATH:-${DEFAULT_CONFIG}}
live_mask_path=${LIVE_MASK_PATH:-}
output_dir=${OUTPUT_DIR:-}
debug_chunk_index=${DEBUG_CHUNK_INDEX:-13}
verbose=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tensor-npy)
      tensor_path=$2
      shift 2
      ;;
    --config)
      config_path=$2
      shift 2
      ;;
    --live-mask)
      live_mask_path=$2
      shift 2
      ;;
    --output-dir)
      output_dir=$2
      shift 2
      ;;
    --debug-chunk-index)
      debug_chunk_index=$2
      shift 2
      ;;
    --verbose)
      verbose=1
      shift
      ;;
    *)
      echo "Usage: $0 --tensor-npy PATH [--config PATH] [--live-mask PATH] [--output-dir DIR] [--debug-chunk-index N] [--verbose]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${tensor_path}" ]]; then
  echo "--tensor-npy is required" >&2
  exit 1
fi

tensor_path=$(absolutize_host_path "${tensor_path}")
config_path=$(absolutize_host_path "${config_path}")
if [[ -n "${live_mask_path}" ]]; then
  live_mask_path=$(absolutize_host_path "${live_mask_path}")
fi
if [[ -n "${output_dir}" ]]; then
  output_dir=$(absolutize_host_path "${output_dir}")
fi

"${RUN_DEMO_CONTAINER}"

container_tensor_path=$(map_path_to_container "${tensor_path}")
container_config_path=$(map_path_to_container "${config_path}")
container_live_mask_path=
if [[ -n "${live_mask_path}" ]]; then
  container_live_mask_path=$(map_path_to_container "${live_mask_path}")
fi

if [[ -z "${output_dir}" ]]; then
  tensor_basename=$(basename "${tensor_path}")
  tensor_stem=${tensor_basename%.npy}
  output_dir="/tmp/usrp_spectrograms/dino_validator_artifacts/${tensor_stem}"
fi
container_output_dir=$(map_path_to_container "${output_dir}")

container_validator_bin=${VALIDATOR_BIN}
if [[ -z "${container_validator_bin}" ]]; then
  container_validator_bin="__AUTO__"
fi

exec sudo docker exec -it \
  -e VALIDATOR_BIN="${container_validator_bin}" \
  -e BUILD_APP_DIR="${BUILD_APP_DIR}" \
  -e TENSOR_PATH="${container_tensor_path}" \
  -e CONFIG_PATH="${container_config_path}" \
  -e OUTPUT_DIR="${container_output_dir}" \
  -e LIVE_MASK_PATH="${container_live_mask_path}" \
  -e DEBUG_CHUNK_INDEX="${debug_chunk_index}" \
  -e VERBOSE_FLAG="${verbose}" \
  "${CONTAINER_NAME}" \
  bash -lc '
set -euo pipefail

validator_bin="${VALIDATOR_BIN}"
if [[ -z "${validator_bin}" || "${validator_bin}" == "__AUTO__" ]]; then
  candidates=(
    "${BUILD_APP_DIR}/offline_dino_validator"
    "/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/offline_dino_validator"
    "/workspace/holohub/build/applications/usrp_wideband_signal_detection/offline_dino_validator"
    "/workspace/holohub/build/usrp_wideband_signal_detection/offline_dino_validator"
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/offline_dino_validator"
    "/workspace/holohub-dev/build/applications/usrp_wideband_signal_detection/offline_dino_validator"
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/offline_dino_validator"
  )
  validator_bin=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      validator_bin="${candidate}"
      break
    fi
  done
fi

if [[ -z "${validator_bin}" ]]; then
  echo "offline_dino_validator not found in the container." >&2
  echo "Checked BUILD_APP_DIR=${BUILD_APP_DIR} and common build output locations." >&2
  echo "Rebuild it first or set VALIDATOR_BIN to the container path of the binary." >&2
  exit 1
fi

cmd=(
  "${validator_bin}"
  "--tensor-npy" "${TENSOR_PATH}"
  "--config" "${CONFIG_PATH}"
  "--output-dir" "${OUTPUT_DIR}"
  "--debug-chunk-index" "${DEBUG_CHUNK_INDEX}"
)
if [[ -n "${LIVE_MASK_PATH}" ]]; then
  cmd+=("--live-mask" "${LIVE_MASK_PATH}")
fi
if [[ "${VERBOSE_FLAG}" == "1" ]]; then
  cmd+=("--verbose")
fi

exec "${cmd[@]}"
'
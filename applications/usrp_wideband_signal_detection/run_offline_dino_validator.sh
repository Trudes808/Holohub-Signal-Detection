#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
DEFAULT_CONFIG=${DEFAULT_CONFIG:-${SCRIPT_DIR}/config_torchscript_validation_capture_single_channel.yaml}

host_repo_root=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

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
    --verbose)
      verbose=1
      shift
      ;;
    *)
      echo "Usage: $0 --tensor-npy PATH [--config PATH] [--live-mask PATH] [--output-dir DIR] [--verbose]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${tensor_path}" ]]; then
  echo "--tensor-npy is required" >&2
  exit 1
fi

"${SCRIPT_DIR}/run_demo_container.sh"

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

cmd=(
  "cd ${BUILD_APP_DIR}"
  "./offline_dino_validator"
  "--tensor-npy ${container_tensor_path}"
  "--config ${container_config_path}"
  "--output-dir ${container_output_dir}"
)
if [[ -n "${container_live_mask_path}" ]]; then
  cmd+=("--live-mask ${container_live_mask_path}")
fi
if [[ ${verbose} -eq 1 ]]; then
  cmd+=("--verbose")
fi

exec sudo docker exec -it "${CONTAINER_NAME}" bash -lc "$(printf '%q ' "${cmd[@]}")"
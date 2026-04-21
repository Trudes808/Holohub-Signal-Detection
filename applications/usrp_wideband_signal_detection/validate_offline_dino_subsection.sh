#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKING_DIR=$(pwd -P)
REBUILD_SCRIPT=${REBUILD_SCRIPT:-${SCRIPT_DIR}/rebuild_demo_container_app.sh}
VALIDATOR_SCRIPT=${VALIDATOR_SCRIPT:-${SCRIPT_DIR}/run_offline_dino_validator.sh}
COMPARE_SCRIPT=${COMPARE_SCRIPT:-${SCRIPT_DIR}/compare_offline_dino_subsection.py}
DEFAULT_CONFIG=${DEFAULT_CONFIG:-${SCRIPT_DIR}/config_torchscript_validation_capture_single_channel.yaml}
PYTHON_BIN=${PYTHON_BIN:-python3}

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

tensor_path=${TENSOR_PATH:-}
config_path=${CONFIG_PATH:-${DEFAULT_CONFIG}}
output_dir=${OUTPUT_DIR:-}
debug_chunk_index=${DEBUG_CHUNK_INDEX:-13}
skip_rebuild=0
skip_validator=0
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
    --output-dir)
      output_dir=$2
      shift 2
      ;;
    --debug-chunk-index)
      debug_chunk_index=$2
      shift 2
      ;;
    --skip-rebuild)
      skip_rebuild=1
      shift
      ;;
    --skip-validator)
      skip_validator=1
      shift
      ;;
    --verbose)
      verbose=1
      shift
      ;;
    *)
      echo "Usage: $0 --tensor-npy PATH [--config PATH] [--output-dir DIR] [--debug-chunk-index N] [--skip-rebuild] [--skip-validator] [--verbose]" >&2
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
if [[ -n "${output_dir}" ]]; then
  output_dir=$(absolutize_host_path "${output_dir}")
else
  tensor_basename=$(basename "${tensor_path}")
  tensor_stem=${tensor_basename%.npy}
  output_dir="/tmp/usrp_spectrograms/dino_validator_artifacts/${tensor_stem}"
fi

if [[ "${skip_rebuild}" != "1" ]]; then
  "${REBUILD_SCRIPT}"
fi

if [[ "${skip_validator}" != "1" ]]; then
  validator_cmd=(
    "${VALIDATOR_SCRIPT}"
    "--tensor-npy" "${tensor_path}"
    "--config" "${config_path}"
    "--output-dir" "${output_dir}"
    "--debug-chunk-index" "${debug_chunk_index}"
    "--subsection-only-validation"
  )
  if [[ "${verbose}" == "1" ]]; then
    validator_cmd+=("--verbose")
  fi
  "${validator_cmd[@]}"
fi

compare_cmd=(
  "${PYTHON_BIN}"
  "${COMPARE_SCRIPT}"
  "--tensor-npy" "${tensor_path}"
  "--config" "${config_path}"
  "--output-dir" "${output_dir}"
  "--debug-chunk-index" "${debug_chunk_index}"
)
"${compare_cmd[@]}"
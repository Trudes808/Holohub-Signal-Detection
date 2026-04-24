#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKING_DIR=$(pwd -P)
BASE_VALIDATE_SCRIPT=${BASE_VALIDATE_SCRIPT:-${SCRIPT_DIR}/validate_offline_dino_subsection.sh}
CUDA_VALIDATOR_SCRIPT=${CUDA_VALIDATOR_SCRIPT:-${SCRIPT_DIR}/run_offline_dino_cuda_validator.sh}

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
output_dir=${OUTPUT_DIR:-}
has_output_dir=0
forwarded_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tensor-npy)
      tensor_path=$2
      forwarded_args+=("$1" "$2")
      shift 2
      ;;
    --output-dir)
      output_dir=$2
      has_output_dir=1
      forwarded_args+=("$1" "$2")
      shift 2
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${tensor_path}" && "${has_output_dir}" != "1" && -z "${output_dir}" ]]; then
  tensor_path=$(absolutize_host_path "${tensor_path}")
  tensor_basename=$(basename "${tensor_path}")
  tensor_stem=${tensor_basename%.npy}
  output_dir="/tmp/usrp_spectrograms/dino_cuda_validator_artifacts/${tensor_stem}"
  forwarded_args+=("--output-dir" "${output_dir}")
fi

exec env VALIDATOR_SCRIPT="${CUDA_VALIDATOR_SCRIPT}" "${BASE_VALIDATE_SCRIPT}" "${forwarded_args[@]}"
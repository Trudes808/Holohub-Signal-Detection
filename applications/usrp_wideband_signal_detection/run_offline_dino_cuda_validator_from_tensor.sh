#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
VALIDATOR_SCRIPT=${VALIDATOR_SCRIPT:-${SCRIPT_DIR}/run_offline_dino_cuda_validator.sh}

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 TENSOR_NPY [--config PATH] [--output-dir DIR] [--debug-chunk-index N] [--verbose]" >&2
  exit 1
fi

tensor_path=$1
shift

exec "${VALIDATOR_SCRIPT}" --tensor-npy "${tensor_path}" "$@"
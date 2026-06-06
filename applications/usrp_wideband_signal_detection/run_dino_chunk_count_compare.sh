#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKING_DIR=$(pwd -P)
VALIDATION_SCRIPT=${VALIDATION_SCRIPT:-${SCRIPT_DIR}/dino_cuda_validation.sh}
BASELINE_CONFIG=${BASELINE_CONFIG:-${SCRIPT_DIR}/config_cuda_dino_scaffold_single_channel.yaml}
EXPERIMENT_CONFIG=${EXPERIMENT_CONFIG:-${SCRIPT_DIR}/config_cuda_dino_scaffold_single_channel_6_chunks.yaml}
EXPERIMENT_LABEL=${EXPERIMENT_LABEL:-experiment_6_chunks}
BASELINE_DEBUG_CHUNK_INDEX=${BASELINE_DEBUG_CHUNK_INDEX:-12}
EXPERIMENT_DEBUG_CHUNK_INDEX=${EXPERIMENT_DEBUG_CHUNK_INDEX:-2}

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

usage() {
  cat >&2 <<'EOF'
Usage: run_dino_chunk_count_compare.sh --tensor-npy PATH [options]

Runs the standard 25-chunk scaffold validation and the 6-chunk experiment back to back,
writing separate CUDA/reference artifact trees and compare plots for each.

Options:
  --tensor-npy PATH     Input spectrogram tensor (.npy) to validate
  --output-root DIR     Root output directory for both runs
  --experiment-config   Config path for the non-baseline run
  --experiment-label    Output label for the non-baseline run
  --baseline-debug-chunk-index N
                        Debug chunk index for the baseline run
  --experiment-debug-chunk-index N
                        Debug chunk index for the non-baseline run
  --plot-only           Reuse existing artifacts and only regenerate plots
  --verbose             Forward verbose mode to the underlying validation script
  -h, --help            Show this help text
EOF
}

tensor_path=
output_root=
experiment_config=${EXPERIMENT_CONFIG}
experiment_label=${EXPERIMENT_LABEL}
baseline_debug_chunk_index=${BASELINE_DEBUG_CHUNK_INDEX}
experiment_debug_chunk_index=${EXPERIMENT_DEBUG_CHUNK_INDEX}
plot_only=0
verbose=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tensor-npy)
      tensor_path=${2:-}
      shift 2
      ;;
    --output-root)
      output_root=${2:-}
      shift 2
      ;;
    --experiment-config)
      experiment_config=${2:-}
      shift 2
      ;;
    --experiment-label)
      experiment_label=${2:-}
      shift 2
      ;;
    --baseline-debug-chunk-index)
      baseline_debug_chunk_index=${2:-}
      shift 2
      ;;
    --experiment-debug-chunk-index)
      experiment_debug_chunk_index=${2:-}
      shift 2
      ;;
    --plot-only)
      plot_only=1
      shift
      ;;
    --verbose)
      verbose=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$tensor_path" ]]; then
  echo "--tensor-npy is required" >&2
  usage
  exit 1
fi

tensor_path=$(absolutize_host_path "$tensor_path")
if [[ ! -f "$tensor_path" ]]; then
  echo "Tensor file not found: $tensor_path" >&2
  exit 1
fi

experiment_config=$(absolutize_host_path "$experiment_config")
if [[ ! -f "$experiment_config" ]]; then
  echo "Experiment config not found: $experiment_config" >&2
  exit 1
fi

if [[ -z "$experiment_label" ]]; then
  echo "--experiment-label must not be empty" >&2
  exit 1
fi

if [[ -z "$output_root" ]]; then
  tensor_basename=$(basename "$tensor_path")
  tensor_stem=${tensor_basename%.npy}
  output_root="${SCRIPT_DIR}/chunk_compare_artifacts/${tensor_stem}"
else
  output_root=$(absolutize_host_path "$output_root")
fi

run_case() {
  local label=$1
  local config_path=$2
  local debug_chunk_index=$3
  local case_root="${output_root}/${label}"
  local cuda_output_dir="${case_root}/cuda"
  local reference_output_dir="${case_root}/reference"
  local plot_output="${case_root}/${label}_cuda_vs_reference.png"
  local cmd=(
    "$VALIDATION_SCRIPT"
    --tensor-npy "$tensor_path"
    --cuda-config "$config_path"
    --reference-config "$config_path"
    --cuda-output-dir "$cuda_output_dir"
    --reference-output-dir "$reference_output_dir"
    --plot-output "$plot_output"
    --debug-chunk-index "$debug_chunk_index"
  )

  if [[ "$plot_only" == "1" ]]; then
    cmd+=(--plot-only)
  fi
  if [[ "$verbose" == "1" ]]; then
    cmd+=(--verbose)
  fi

  echo "Running ${label} validation:" >&2
  printf '  %q' "${cmd[@]}" >&2
  printf '\n' >&2
  "${cmd[@]}"
}

mkdir -p "$output_root"

run_case baseline_25_chunks "$BASELINE_CONFIG" "$baseline_debug_chunk_index"
run_case "$experiment_label" "$experiment_config" "$experiment_debug_chunk_index"

echo >&2
echo "Chunk-count comparison outputs:" >&2
echo "  baseline plot:   ${output_root}/baseline_25_chunks/baseline_25_chunks_cuda_vs_reference.png" >&2
echo "  experiment plot: ${output_root}/${experiment_label}/${experiment_label}_cuda_vs_reference.png" >&2
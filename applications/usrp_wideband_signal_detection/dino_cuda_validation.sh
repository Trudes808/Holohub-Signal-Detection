#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKING_DIR=$(pwd -P)

CUDA_VALIDATOR_SCRIPT=${CUDA_VALIDATOR_SCRIPT:-${SCRIPT_DIR}/run_offline_cuda_dino_operator_replay.sh}
REFERENCE_VALIDATOR_SCRIPT=${REFERENCE_VALIDATOR_SCRIPT:-${SCRIPT_DIR}/run_offline_dino_validator_performance.sh}
PLOT_SCRIPT=${PLOT_SCRIPT:-${SCRIPT_DIR}/plot_offline_dino_cuda_artifact_compare.py}

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
Usage: dino_cuda_validation.sh --tensor-npy PATH [options]

Runs the replayed live CUDA operator, the reference C++ validator, and the artifact compare plot.

Options:
  --tensor-npy PATH             Input spectrogram tensor (.npy) to validate
  --cuda-config PATH            Config to use for the replayed CUDA operator
  --reference-config PATH       Config to use for the reference validator wrapper
  --cuda-output-dir DIR         CUDA operator output directory
  --reference-output-dir DIR    Reference validator output directory
  --plot-output PATH            Output PNG path for the comparison plot
  --debug-chunk-index N         Debug chunk index for both validators
  --stages K1 K2 ...            Explicit stage keys to pass to the plot script
  --skip-cuda                   Do not run the replayed CUDA operator
  --skip-reference              Do not run the reference validator
  --plot-only                   Only generate the plot from existing outputs
  --verbose                     Forward verbose mode to both validators
  -h, --help                    Show this help text

Examples:
  ./dino_cuda_validation.sh --tensor-npy /tmp/usrp_spectrograms/tensors/foo.npy --verbose
  ./dino_cuda_validation.sh --tensor-npy /tmp/usrp_spectrograms/tensors/foo.npy --plot-only
EOF
}

tensor_path=
cuda_config=
reference_config=
cuda_output_dir=
reference_output_dir=
plot_output=
debug_chunk_index=13
verbose=0
skip_cuda=0
skip_reference=0
plot_only=0
stage_keys=()
cuda_elapsed_ms=
reference_elapsed_ms=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tensor-npy)
      tensor_path=${2:-}
      shift 2
      ;;
    --cuda-config)
      cuda_config=${2:-}
      shift 2
      ;;
    --reference-config)
      reference_config=${2:-}
      shift 2
      ;;
    --cuda-output-dir)
      cuda_output_dir=${2:-}
      shift 2
      ;;
    --reference-output-dir)
      reference_output_dir=${2:-}
      shift 2
      ;;
    --plot-output)
      plot_output=${2:-}
      shift 2
      ;;
    --debug-chunk-index)
      debug_chunk_index=${2:-}
      shift 2
      ;;
    --stages)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        stage_keys+=("$1")
        shift
      done
      ;;
    --skip-cuda)
      skip_cuda=1
      shift
      ;;
    --skip-reference)
      skip_reference=1
      shift
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

if [[ -n "$cuda_config" ]]; then
  cuda_config=$(absolutize_host_path "$cuda_config")
fi
if [[ -n "$reference_config" ]]; then
  reference_config=$(absolutize_host_path "$reference_config")
fi
if [[ -n "$cuda_output_dir" ]]; then
  cuda_output_dir=$(absolutize_host_path "$cuda_output_dir")
fi
if [[ -n "$reference_output_dir" ]]; then
  reference_output_dir=$(absolutize_host_path "$reference_output_dir")
fi
if [[ -n "$plot_output" ]]; then
  plot_output=$(absolutize_host_path "$plot_output")
fi

tensor_basename=$(basename "$tensor_path")
tensor_stem=${tensor_basename%.npy}

if [[ -z "$cuda_output_dir" ]]; then
  cuda_output_dir="/tmp/usrp_spectrograms/dino_cuda_operator_artifacts/${tensor_stem}"
fi
if [[ -z "$reference_output_dir" ]]; then
  reference_output_dir="/tmp/usrp_spectrograms/dino_validator_artifacts/${tensor_stem}"
fi
if [[ -z "$plot_output" ]]; then
  plot_output="${cuda_output_dir}/chunk_debug/cuda_vs_reference_compare.png"
fi

if [[ "$plot_only" == "1" ]]; then
  skip_cuda=1
  skip_reference=1
fi

if [[ "$skip_cuda" != "1" ]]; then
  cuda_cmd=("$CUDA_VALIDATOR_SCRIPT" --tensor-npy "$tensor_path" --output-dir "$cuda_output_dir" --debug-chunk-index "$debug_chunk_index")
  if [[ -n "$cuda_config" ]]; then
    cuda_cmd+=(--config "$cuda_config")
  fi
  if [[ "$verbose" == "1" ]]; then
    cuda_cmd+=(--verbose)
  fi

  echo "Running replayed CUDA operator:" >&2
  printf '  %q' "${cuda_cmd[@]}" >&2
  printf '\n' >&2
  cuda_start_ns=$(date +%s%N)
  "${cuda_cmd[@]}"
  cuda_end_ns=$(date +%s%N)
  cuda_elapsed_ms=$(( (cuda_end_ns - cuda_start_ns) / 1000000 ))
fi

if [[ "$skip_reference" != "1" ]]; then
  reference_cmd=("$REFERENCE_VALIDATOR_SCRIPT" --tensor-npy "$tensor_path" --output-dir "$reference_output_dir" --debug-chunk-index "$debug_chunk_index")
  if [[ -n "$reference_config" ]]; then
    reference_cmd+=(--config "$reference_config")
  fi
  if [[ "$verbose" == "1" ]]; then
    reference_cmd+=(--verbose)
  fi

  echo "Running reference validator:" >&2
  printf '  %q' "${reference_cmd[@]}" >&2
  printf '\n' >&2
  reference_start_ns=$(date +%s%N)
  "${reference_cmd[@]}"
  reference_end_ns=$(date +%s%N)
  reference_elapsed_ms=$(( (reference_end_ns - reference_start_ns) / 1000000 ))
fi

if [[ ! -d "$cuda_output_dir" ]]; then
  echo "CUDA output directory not found for plotting: $cuda_output_dir" >&2
  exit 1
fi
if [[ ! -d "$reference_output_dir" ]]; then
  echo "Reference output directory not found for plotting: $reference_output_dir" >&2
  exit 1
fi

plot_cmd=("$PLOT_SCRIPT" --cuda-output-dir "$cuda_output_dir" --reference-output-dir "$reference_output_dir" --output "$plot_output")
if [[ -n "$cuda_elapsed_ms" ]]; then
  plot_cmd+=(--cuda-elapsed-ms "$cuda_elapsed_ms")
fi
if [[ -n "$reference_elapsed_ms" ]]; then
  plot_cmd+=(--reference-elapsed-ms "$reference_elapsed_ms")
fi
if [[ ${#stage_keys[@]} -gt 0 ]]; then
  plot_cmd+=(--stages "${stage_keys[@]}")
fi

echo "Generating comparison plot:" >&2
printf '  %q' "${plot_cmd[@]}" >&2
printf '\n' >&2
"${plot_cmd[@]}"

cat <<EOF
CUDA output dir: $cuda_output_dir
Reference output dir: $reference_output_dir
Comparison plot: $plot_output
EOF
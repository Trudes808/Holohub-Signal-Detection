#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKING_DIR=$(pwd -P)
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
DEFAULT_CONFIG=${DEFAULT_CONFIG:-${SCRIPT_DIR}/config_cuda_dino_scaffold_single_channel.yaml}
RUN_DEMO_CONTAINER=${RUN_DEMO_CONTAINER:-${SCRIPT_DIR}/run_demo_container.sh}
REPLAY_BIN=${REPLAY_BIN:-}
REPLAY_NAME=${REPLAY_NAME:-offline_cuda_dino_operator_replay}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-$(dirname "$(dirname "${SCRIPT_DIR}")")}

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
    ${HOST_REPO_ROOT}/*)
      printf '/workspace/holohub/%s' "${raw_path#${HOST_REPO_ROOT}/}"
      ;;
    *)
      printf '%s' "${raw_path}"
      ;;
  esac
}

require_container_mapped_path() {
  local label=$1
  local raw_path=$2
  if [[ "$(map_path_to_container "${raw_path}")" == "${raw_path}" ]]; then
    echo "Error: ${label} is not mounted into container ${CONTAINER_NAME}: ${raw_path}" >&2
    echo "Use a path under /tmp/usrp_spectrograms or ${HOST_REPO_ROOT}." >&2
    exit 1
  fi
}

tensor_path=${TENSOR_PATH:-}
config_path=${CONFIG_PATH:-${DEFAULT_CONFIG}}
output_dir=${OUTPUT_DIR:-}
debug_chunk_index=${DEBUG_CHUNK_INDEX:-13}
tensor_axis_order=${TENSOR_AXIS_ORDER:-auto}
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
    --tensor-axis-order)
      tensor_axis_order=$2
      shift 2
      ;;
    --verbose)
      verbose=1
      shift
      ;;
    *)
      echo "Usage: $0 --tensor-npy PATH [--config PATH] [--output-dir DIR] [--debug-chunk-index N] [--tensor-axis-order auto|time_frequency|frequency_time] [--verbose]" >&2
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
fi

if [[ ! -f "${tensor_path}" ]]; then
  echo "Tensor snapshot not found: ${tensor_path}" >&2
  exit 1
fi
if [[ ! -f "${config_path}" ]]; then
  echo "Replay config not found: ${config_path}" >&2
  exit 1
fi

if [[ -z "${output_dir}" ]]; then
  tensor_basename=$(basename "${tensor_path}")
  tensor_stem=${tensor_basename%.npy}
  output_dir="/tmp/usrp_spectrograms/dino_cuda_operator_artifacts/${tensor_stem}"
fi

require_container_mapped_path "tensor snapshot path" "${tensor_path}"
require_container_mapped_path "config path" "${config_path}"
require_container_mapped_path "operator output directory" "${output_dir}"

"${RUN_DEMO_CONTAINER}"

container_tensor_path=$(map_path_to_container "${tensor_path}")
container_config_path=$(map_path_to_container "${config_path}")
container_output_dir=$(map_path_to_container "${output_dir}")
container_replay_bin=${REPLAY_BIN:-__AUTO__}

mkdir -p "${output_dir}"

docker_exec_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
  docker_exec_flags=(-it)
fi

exec sudo docker exec "${docker_exec_flags[@]}" \
  -e REPLAY_BIN="${container_replay_bin}" \
  -e REPLAY_NAME="${REPLAY_NAME}" \
  -e BUILD_APP_DIR="${BUILD_APP_DIR}" \
  -e TENSOR_PATH="${container_tensor_path}" \
  -e CONFIG_PATH="${container_config_path}" \
  -e OUTPUT_DIR="${container_output_dir}" \
  -e DEBUG_CHUNK_INDEX="${debug_chunk_index}" \
  -e TENSOR_AXIS_ORDER="${tensor_axis_order}" \
  -e VERBOSE_FLAG="${verbose}" \
  "${CONTAINER_NAME}" \
  bash -lc '
set -euo pipefail

replay_bin="${REPLAY_BIN}"
preferred_replay_bin=""
discovered_candidates=()
if [[ -z "${replay_bin}" || "${replay_bin}" == "__AUTO__" ]]; then
  preferred_replay_bin="${BUILD_APP_DIR}/${REPLAY_NAME}"
  candidates=(
    "${preferred_replay_bin}"
    "/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/${REPLAY_NAME}"
    "/workspace/holohub/build/applications/usrp_wideband_signal_detection/${REPLAY_NAME}"
    "/workspace/holohub/build/usrp_wideband_signal_detection/${REPLAY_NAME}"
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/${REPLAY_NAME}"
    "/workspace/holohub-dev/build/applications/usrp_wideband_signal_detection/${REPLAY_NAME}"
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/${REPLAY_NAME}"
  )
  replay_bin=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      discovered_candidates+=("${candidate}")
    fi
  done
  if [[ -x "${preferred_replay_bin}" ]]; then
    replay_bin="${preferred_replay_bin}"
  elif [[ ${#discovered_candidates[@]} -gt 0 ]]; then
    replay_bin="${discovered_candidates[0]}"
  fi
fi

if [[ -z "${replay_bin}" ]]; then
  echo "${REPLAY_NAME} not found in the container." >&2
  echo "Rebuild it first or set REPLAY_BIN to the container path of the binary." >&2
  exit 1
fi

echo "Using ${REPLAY_NAME}: ${replay_bin}" >&2
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cmd=(
  "${replay_bin}"
  "--tensor-npy" "${TENSOR_PATH}"
  "--config" "${CONFIG_PATH}"
  "--output-dir" "${OUTPUT_DIR}"
  "--debug-chunk-index" "${DEBUG_CHUNK_INDEX}"
  "--tensor-axis-order" "${TENSOR_AXIS_ORDER}"
)

if [[ "${VERBOSE_FLAG}" == "1" ]]; then
  echo "Running operator replay command:" >&2
  printf "  %q" "${cmd[@]}" >&2
  printf "\n" >&2
fi

"${cmd[@]}"
'

OUTPUT_DIR_FOR_VALIDATION="${output_dir}" python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR_FOR_VALIDATION"]).resolve()
summary_path = output_dir / "offline_validation_summary.json"
chunk_debug_summary_path = output_dir / "chunk_debug" / "chunk_debug_summary.json"

failures = []
if not summary_path.exists():
    failures.append(f"missing summary: {summary_path}")
    summary = {}
else:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

if not chunk_debug_summary_path.exists():
    failures.append(f"missing chunk debug summary: {chunk_debug_summary_path}")
    debug_summary = {}
else:
    debug_summary = json.loads(chunk_debug_summary_path.read_text(encoding="utf-8"))

required_summary_artifacts = [
    "projected_grouped_mask_npy",
    "projected_grouped_score_npy",
    "merged_box_mask_npy",
    "final_mask_npy",
    "chunk_plan_json",
    "projected_boxes_json",
    "merged_boxes_json",
]
required_debug_artifacts = [
    "corrected_resized_npy",
    "dino_score_raw_npy",
    "dino_score_raw_deweighted_npy",
    "coherence_gate_npy",
    "combined_score_npy",
    "grouped_mask_npy",
    "grouped_boxes_json",
    "final_mask_npy",
    "final_mask_source_npy",
    "final_mask_projected_npy",
]

for key in required_summary_artifacts:
    path = Path(str(summary.get(key, "") or ""))
    if not path.exists():
        failures.append(f"missing summary artifact for {key}: {path}")

for key in required_debug_artifacts:
    path = Path(str(debug_summary.get(key, "") or ""))
    if not path.exists():
        failures.append(f"missing debug artifact for {key}: {path}")

manifest = {
    "bundle_kind": "offline_cuda_dino_operator_bundle_v1",
    "output_dir": str(output_dir),
    "summary_json": str(summary_path),
    "chunk_debug_summary_json": str(chunk_debug_summary_path),
    "summary_artifacts": {key: str(summary.get(key, "")) for key in required_summary_artifacts},
    "debug_artifacts": {key: str(debug_summary.get(key, "")) for key in required_debug_artifacts},
}
manifest_path = output_dir / "cuda_artifact_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if failures:
    print("CUDA operator replay completed, but the artifact bundle is incomplete:", file=sys.stderr)
    for failure in failures:
        print(f"- {failure}", file=sys.stderr)
    sys.exit(1)

print(f"Validated CUDA operator artifact bundle: {manifest_path}", file=sys.stderr)
PY
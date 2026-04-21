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
VALIDATOR_NAME=${VALIDATOR_NAME:-offline_dino_validator_performance}

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
subsection_only_validation=0

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
    --subsection-only-validation)
      subsection_only_validation=1
      shift
      ;;
    *)
      echo "Usage: $0 --tensor-npy PATH [--config PATH] [--live-mask PATH] [--output-dir DIR] [--debug-chunk-index N] [--verbose] [--subsection-only-validation]" >&2
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
  -e VALIDATOR_NAME="${VALIDATOR_NAME}" \
  -e BUILD_APP_DIR="${BUILD_APP_DIR}" \
  -e TENSOR_PATH="${container_tensor_path}" \
  -e CONFIG_PATH="${container_config_path}" \
  -e OUTPUT_DIR="${container_output_dir}" \
  -e LIVE_MASK_PATH="${container_live_mask_path}" \
  -e DEBUG_CHUNK_INDEX="${debug_chunk_index}" \
  -e VERBOSE_FLAG="${verbose}" \
  -e SUBSECTION_ONLY_VALIDATION_FLAG="${subsection_only_validation}" \
  "${CONTAINER_NAME}" \
  bash -lc '
set -euo pipefail

validator_bin="${VALIDATOR_BIN}"
preferred_validator_bin=""
discovered_candidates=()
if [[ -z "${validator_bin}" || "${validator_bin}" == "__AUTO__" ]]; then
  preferred_validator_bin="${BUILD_APP_DIR}/${VALIDATOR_NAME}"
  candidates=(
    "${preferred_validator_bin}"
    "/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/${VALIDATOR_NAME}"
    "/workspace/holohub/build/applications/usrp_wideband_signal_detection/${VALIDATOR_NAME}"
    "/workspace/holohub/build/usrp_wideband_signal_detection/${VALIDATOR_NAME}"
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/${VALIDATOR_NAME}"
    "/workspace/holohub-dev/build/applications/usrp_wideband_signal_detection/${VALIDATOR_NAME}"
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/${VALIDATOR_NAME}"
  )
  validator_bin=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      already_listed=0
      for discovered in "${discovered_candidates[@]}"; do
        if [[ "${discovered}" == "${candidate}" ]]; then
          already_listed=1
          break
        fi
      done
      if [[ "${already_listed}" != "1" ]]; then
        discovered_candidates+=("${candidate}")
      fi
    fi
  done

  if [[ -x "${preferred_validator_bin}" ]]; then
    validator_bin="${preferred_validator_bin}"
  elif [[ ${#discovered_candidates[@]} -gt 0 ]]; then
    validator_bin="${discovered_candidates[0]}"
  fi
fi

if [[ -z "${validator_bin}" ]]; then
  echo "${VALIDATOR_NAME} not found in the container." >&2
  echo "Checked BUILD_APP_DIR=${BUILD_APP_DIR} and common build output locations." >&2
  echo "Rebuild it first or set VALIDATOR_BIN to the container path of the binary." >&2
  exit 1
fi

if [[ -n "${preferred_validator_bin:-}" && -x "${preferred_validator_bin}" && "${validator_bin}" != "${preferred_validator_bin}" ]]; then
  echo "Warning: selected validator does not match preferred build output ${preferred_validator_bin}" >&2
fi

if [[ ${#discovered_candidates[@]} -gt 0 ]]; then
  echo "Discovered ${VALIDATOR_NAME} candidates:" >&2
  for candidate in "${discovered_candidates[@]}"; do
    if command -v stat >/dev/null 2>&1; then
      printf "  - %s (%s)\\n" "${candidate}" "$(stat -c "%y" "${candidate}" 2>/dev/null || printf "%s" "mtime unavailable")" >&2
    else
      printf "  - %s\\n" "${candidate}" >&2
    fi
  done
fi

echo "Using ${VALIDATOR_NAME}: ${validator_bin}" >&2
if command -v stat >/dev/null 2>&1; then
  stat -c "Validator mtime: %y" "${validator_bin}" >&2 || true
fi

if command -v stat >/dev/null 2>&1; then
  source_candidates=(
    "/workspace/holohub/applications/usrp_wideband_signal_detection/offline_dino_validator_performance.cpp"
    "/workspace/holohub/operators/dinov3_signal_detector/dinov3_torch_runtime.cpp"
    "/workspace/holohub/operators/dinov3_signal_detector/dinov3_torch_runtime.hpp"
    "/workspace/holohub-dev/applications/usrp_wideband_signal_detection/offline_dino_validator_performance.cpp"
    "/workspace/holohub-dev/operators/dinov3_signal_detector/dinov3_torch_runtime.cpp"
    "/workspace/holohub-dev/operators/dinov3_signal_detector/dinov3_torch_runtime.hpp"
  )
  stale_against=()
  validator_mtime=$(stat -c '%Y' "${validator_bin}" 2>/dev/null || printf '0')
  for source_path in "${source_candidates[@]}"; do
    if [[ -f "${source_path}" ]]; then
      source_mtime=$(stat -c '%Y' "${source_path}" 2>/dev/null || printf '0')
      if [[ "${source_mtime}" -gt "${validator_mtime}" ]]; then
        stale_against+=("${source_path}")
      fi
    fi
  done
  if [[ ${#stale_against[@]} -gt 0 ]]; then
    echo "Selected ${VALIDATOR_NAME} is older than patched source files and is likely stale." >&2
    for source_path in "${stale_against[@]}"; do
      printf "  - newer source: %s (%s)\\n" "${source_path}" "$(stat -c "%y" "${source_path}" 2>/dev/null || printf "%s" "mtime unavailable")" >&2
    done
    echo "Rebuild the preferred build tree or set VALIDATOR_BIN explicitly to the freshly rebuilt container binary." >&2
    exit 1
  fi
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
if [[ "${SUBSECTION_ONLY_VALIDATION_FLAG}" == "1" ]]; then
  cmd+=("--subsection-only-validation")
fi

echo "Resetting validator output directory: ${OUTPUT_DIR}" >&2
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

set +e
"${cmd[@]}"
validator_status=$?
set -e

if [[ ${validator_status} -ne 0 ]]; then
  if [[ ${validator_status} -eq 137 ]]; then
    echo "${VALIDATOR_NAME} was killed with exit code 137 (SIGKILL), which usually means the container hit an OOM or cgroup memory limit." >&2
    echo "The validator now only keeps heavyweight debug buffers for the selected debug chunk, so rerun after rebuilding this latest binary before investigating algorithm parity further." >&2
  fi
  exit ${validator_status}
fi

OUTPUT_DIR_FOR_VALIDATION="${OUTPUT_DIR}" python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR_FOR_VALIDATION"])
chunk_debug_dir = output_dir / "chunk_debug"
summary_path = chunk_debug_dir / "chunk_debug_summary.json"
required_files = [
  chunk_debug_dir / "chunk_runtime_input_gray.npy",
  chunk_debug_dir / "chunk_dino_score_raw.npy",
  chunk_debug_dir / "chunk_patch_features.npy",
  output_dir / "offline_corrected_resized.npy",
  output_dir / "offline_final_mask.npy",
]

failures = []
if not summary_path.exists():
  failures.append(f"missing summary: {summary_path}")
  summary = {}
else:
  summary = json.loads(summary_path.read_text(encoding="utf-8"))

for path in required_files:
  if not path.exists():
    failures.append(f"missing artifact: {path}")

artifact_contract = str(summary.get("artifact_contract", "") or "")
if artifact_contract != "chunk_no_extra_sideband_crop_v2":
  failures.append(
    f"unexpected artifact_contract {artifact_contract!r} in {summary_path}; expected 'chunk_no_extra_sideband_crop_v2'"
  )

runtime_rows = int(summary.get("runtime_input_gray_rows", 0) or 0)
runtime_cols = int(summary.get("runtime_input_gray_cols", 0) or 0)
src_rows = int(summary.get("src_rows", 0) or 0)
src_cols = int(summary.get("src_cols", 0) or 0)
ignore_bins = int(summary.get("ignore_bins_per_side", 0) or 0)
patch_rows = int(summary.get("patch_rows", 0) or 0)
patch_cols = int(summary.get("patch_cols", 0) or 0)
feature_dim = int(summary.get("feature_dim", 0) or 0)

if ignore_bins == 0 and src_rows > 0 and src_cols > 0 and (runtime_rows, runtime_cols) != (src_rows, src_cols):
  failures.append(
    f"runtime_input_gray shape {(runtime_rows, runtime_cols)} does not match source chunk {(src_rows, src_cols)} when ignore_bins_per_side=0"
  )

if patch_rows <= 0 or patch_cols <= 0 or feature_dim <= 0:
  failures.append(
    f"invalid patch metadata patch_rows={patch_rows}, patch_cols={patch_cols}, feature_dim={feature_dim}"
  )

if failures:
  print("Fresh validator run completed, but the produced artifact bundle is still stale or incomplete:", file=sys.stderr)
  for failure in failures:
    print(f"- {failure}", file=sys.stderr)
  sys.exit(1)

print(f"Validated fresh artifact bundle: {summary_path}", file=sys.stderr)
PY
'
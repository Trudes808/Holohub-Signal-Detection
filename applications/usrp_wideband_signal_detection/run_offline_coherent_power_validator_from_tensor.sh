#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_REF_DIR=$(dirname "${BASH_SOURCE[0]}")
WORKING_DIR=$(pwd -P)
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
DEFAULT_CONFIG=${DEFAULT_CONFIG:-${SCRIPT_DIR}/config_coherent_power_validation.yaml}
RUN_DEMO_CONTAINER=${RUN_DEMO_CONTAINER:-${SCRIPT_REF_DIR}/run_demo_container.sh}
VALIDATOR_BIN=${VALIDATOR_BIN:-}
VALIDATOR_NAME=${VALIDATOR_NAME:-offline_coherent_power_validator}

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
    /tmp/coherent_power_masks/*)
      printf '/workspace/coherent_power_masks/%s' "${raw_path#/tmp/coherent_power_masks/}"
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
output_root=${OUTPUT_ROOT:-}
validator_output_dir=${OUTPUT_DIR:-}
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
    --output-root)
      output_root=$2
      shift 2
      ;;
    --output-dir)
      validator_output_dir=$2
      shift 2
      ;;
    --verbose)
      verbose=1
      shift
      ;;
    *)
      echo "Usage: $0 --tensor-npy PATH [--config PATH] [--output-root DIR] [--output-dir DIR] [--verbose]" >&2
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

echo "Using tensor: ${tensor_path}" >&2
echo "Using coherent config: ${config_path}" >&2

if [[ -z "${output_root}" ]]; then
  tensor_basename=$(basename "${tensor_path}")
  tensor_stem=${tensor_basename%.npy}
  output_root="/tmp/usrp_spectrograms/coherent_power_validator_artifacts/${tensor_stem}"
fi
output_root=$(absolutize_host_path "${output_root}")

if [[ -z "${validator_output_dir}" ]]; then
  validator_output_dir="${output_root}/offline_validator"
fi
validator_output_dir=$(absolutize_host_path "${validator_output_dir}")

mkdir -p "${output_root}"

container_tensor_path=$(map_path_to_container "${tensor_path}")
container_output_root=$(map_path_to_container "${output_root}")
container_validator_output_dir=$(map_path_to_container "${validator_output_dir}")
metadata_path="${output_root}/coherent_power_input_snapshot.json"
container_metadata_path="${container_output_root}/coherent_power_input_snapshot.json"

echo "Preparing coherent snapshot metadata: ${metadata_path}" >&2

python3 - "${tensor_path}" "${container_tensor_path}" "${config_path}" "${metadata_path}" <<'PY'
import ast
import json
import struct
import sys
from pathlib import Path


def load_npy_header(path: Path):
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not a valid .npy file")
        major = handle.read(1)[0]
        minor = handle.read(1)[0]
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in (2, 3):
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise ValueError(f"Unsupported .npy version {(major, minor)}")
        header = ast.literal_eval(handle.read(header_len).decode("latin1"))
        return header


def parse_scalar(raw: str):
    text = raw.strip()
    if not text:
        return None
    if text.startswith(('"', "'")) and text.endswith(('"', "'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(char in text for char in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def parse_required_sections(config_path: Path):
    sections = {"fft": {}, "coherent_power_signal_detector": {}}
    current_section = None
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            current_section = line[:-1] if line.endswith(":") else None
            continue
        if current_section not in sections:
            continue
        if not line.startswith("  ") or line.startswith("    "):
            continue
        stripped = line.strip()
        if stripped.startswith("- ") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        parsed_value = parse_scalar(value)
        if parsed_value is not None:
            sections[current_section][key.strip()] = parsed_value
    return sections


tensor_path = Path(sys.argv[1])
container_tensor_path = sys.argv[2]
config_path = Path(sys.argv[3])
metadata_path = Path(sys.argv[4])

header = load_npy_header(tensor_path)
shape = tuple(int(dim) for dim in header.get("shape", ()))
descr = str(header.get("descr", ""))
if len(shape) != 2:
    raise ValueError(f"Expected a 2D tensor snapshot, got shape {shape}")
if "c" not in descr:
    raise ValueError(f"Expected a complex tensor snapshot, got dtype {descr}")

rows, cols = shape
sections = parse_required_sections(config_path)
fft = sections["fft"]
coherent = sections["coherent_power_signal_detector"]

tensor_axis_order = "time_frequency" if rows < cols else "frequency_time"
freq_bins = cols if tensor_axis_order == "time_frequency" else rows

span_hz = float(fft.get("span", 0.0))
resolution_hz = float(fft.get("resolution", 0.0))
if resolution_hz <= 0.0 and span_hz > 0.0 and freq_bins > 0:
  resolution_hz = span_hz / float(freq_bins)

config_keys = [
    "chunk_bandwidth_hz",
    "chunk_overlap_hz",
    "uncalibrated_chunk_fraction",
    "uncalibrated_overlap_fraction",
    "ignore_sideband_percent",
    "ignore_sideband_hz",
    "frontend_row_q",
    "frontend_reference_q",
    "frontend_smooth_sigma",
    "frontend_max_boost_db",
    "coherence_weight",
    "power_weight",
    "power_assist_mode",
    "power_floor_time_q",
    "power_floor_global_q",
    "power_excess_start_db",
    "power_excess_full_db",
    "power_local_blend",
    "coherence_gate_start",
    "coherence_gate_full",
    "coherence_bridge_bias",
    "coherence_power_joint_weight",
    "coherence_power_support_q",
    "coherence_power_q",
    "min_component_size",
    "filter_detection_mask",
    "grouping_seed_score_q",
    "grouping_bridge_freq_px",
    "grouping_bridge_time_px",
    "grouping_min_component_size",
    "grouping_min_freq_span_px",
    "grouping_min_time_span_px",
    "grouping_min_density",
    "grouping_time_continuity_ratio",
]

metadata = {
    "rows": rows,
    "cols": cols,
    "input_height": int(coherent.get("input_height", rows)),
    "input_width": int(coherent.get("input_width", cols)),
    "resolution_hz": resolution_hz,
    "sample_rate_hz": span_hz,
    "span_hz": span_hz,
    "tensor_axis_order": tensor_axis_order,
    "tensor_snapshot_path": container_tensor_path,
    "power_db_snapshot_path": None,
    "config": {key: coherent[key] for key in config_keys if key in coherent},
}

metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
PY

echo "Prepared metadata sidecar." >&2

echo "Ensuring demo container is running..." >&2

"${RUN_DEMO_CONTAINER}"

echo "Container ready: ${CONTAINER_NAME}" >&2

container_validator_bin=${VALIDATOR_BIN}
if [[ -z "${container_validator_bin}" ]]; then
  container_validator_bin="__AUTO__"
fi

docker_exec_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
  docker_exec_flags=(-it)
fi

exec sudo docker exec "${docker_exec_flags[@]}" \
  -e VALIDATOR_BIN="${container_validator_bin}" \
  -e VALIDATOR_NAME="${VALIDATOR_NAME}" \
  -e BUILD_APP_DIR="${BUILD_APP_DIR}" \
  -e SNAPSHOT_JSON="${container_metadata_path}" \
  -e OUTPUT_DIR="${container_validator_output_dir}" \
  -e VERBOSE_FLAG="${verbose}" \
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

if command -v stat >/dev/null 2>&1; then
  source_candidates=(
    "/workspace/holohub/applications/usrp_wideband_signal_detection/offline_coherent_power_validator.cpp"
    "/workspace/holohub/operators/coherent_power_signal_detector/coherent_power_signal_detector.cu"
    "/workspace/holohub/operators/coherent_power_signal_detector/coherent_power_signal_detector.hpp"
    "/workspace/holohub-dev/applications/usrp_wideband_signal_detection/offline_coherent_power_validator.cpp"
    "/workspace/holohub-dev/operators/coherent_power_signal_detector/coherent_power_signal_detector.cu"
    "/workspace/holohub-dev/operators/coherent_power_signal_detector/coherent_power_signal_detector.hpp"
  )
  stale_against=()
  validator_mtime=$(stat -c "%Y" "${validator_bin}" 2>/dev/null || printf "0")
  for source_path in "${source_candidates[@]}"; do
    if [[ -f "${source_path}" ]]; then
      source_mtime=$(stat -c "%Y" "${source_path}" 2>/dev/null || printf "0")
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

echo "Using ${VALIDATOR_NAME}: ${validator_bin}" >&2
echo "Snapshot JSON inside container: ${SNAPSHOT_JSON}" >&2

cmd=(
  "${validator_bin}"
  "--snapshot-json" "${SNAPSHOT_JSON}"
  "--output-dir" "${OUTPUT_DIR}"
)
if [[ "${VERBOSE_FLAG}" == "1" ]]; then
  cmd+=("--verbose")
fi

echo "Resetting validator output directory: ${OUTPUT_DIR}" >&2
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Starting coherent offline validator..." >&2
start_epoch=$(date +%s)

"${cmd[@]}" &
validator_pid=$!

while kill -0 "${validator_pid}" 2>/dev/null; do
  now_epoch=$(date +%s)
  elapsed=$((now_epoch - start_epoch))
  echo "coherent validator still running (${elapsed}s elapsed)..." >&2
  sleep 15
done

wait "${validator_pid}"
validator_status=$?
end_epoch=$(date +%s)
elapsed_total=$((end_epoch - start_epoch))

if [[ "${validator_status}" -ne 0 ]]; then
  echo "coherent validator failed after ${elapsed_total}s" >&2
  exit "${validator_status}"
fi

echo "coherent validator completed in ${elapsed_total}s" >&2
  '
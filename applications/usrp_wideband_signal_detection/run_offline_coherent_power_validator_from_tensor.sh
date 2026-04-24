#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKING_DIR=$(pwd -P)
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
RUN_DEMO_CONTAINER=${RUN_DEMO_CONTAINER:-${SCRIPT_DIR}/run_demo_container.sh}
VALIDATOR_BIN=${VALIDATOR_BIN:-}
VALIDATOR_NAME=${VALIDATOR_NAME:-offline_coherent_power_validator}
LIVE_CONFIG=${LIVE_CONFIG:-${SCRIPT_DIR}/config_coherent_power_live_timing_reference.yaml}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-$(dirname "$(dirname "${SCRIPT_DIR}")")}

resolve_latest_coherent_snapshot_json() {
  local snapshot_dir=${1:-/tmp/coherent_power_snapshots}
  python3 - "$snapshot_dir" <<'PY'
from pathlib import Path
import sys

snapshot_dir = Path(sys.argv[1])
paths = sorted(snapshot_dir.glob('coherent_power_snapshot_ch*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if not paths:
    raise SystemExit(f"No coherent snapshot JSON found in {snapshot_dir}")
print(paths[0])
PY
}

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

resolve_tensor_snapshot_path() {
  local raw_path=$1
  local candidate_path
  local -a search_roots=(
    "$WORKING_DIR"
    "$SCRIPT_DIR"
    "/tmp/usrp_spectrograms/tensors"
    "/tmp/usrp_spectrograms"
  )

  if [[ "${raw_path}" = /* ]]; then
    printf '%s\n' "${raw_path}"
    return 0
  fi

  for search_root in "${search_roots[@]}"; do
    candidate_path=$(realpath -m "${search_root}/${raw_path}")
    if [[ -f "${candidate_path}" ]]; then
      printf '%s\n' "${candidate_path}"
      return 0
    fi
  done

  printf '%s\n' "$(absolutize_host_path "${raw_path}")"
}

map_path_to_container() {
  local raw_path=$1
  case "${raw_path}" in
    /tmp/usrp_spectrograms/*)
      printf '/workspace/spectrograms/%s' "${raw_path#/tmp/usrp_spectrograms/}"
      ;;
    /tmp/coherent_power_snapshots/*)
      printf '/workspace/coherent_power_snapshots/%s' "${raw_path#/tmp/coherent_power_snapshots/}"
      ;;
    /tmp/coherent_power_masks/*)
      printf '/workspace/coherent_power_masks/%s' "${raw_path#/tmp/coherent_power_masks/}"
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
    echo "Use a path under /tmp/usrp_spectrograms, /tmp/coherent_power_snapshots, /tmp/coherent_power_masks, or ${HOST_REPO_ROOT}." >&2
    exit 1
  fi
}

verbose=0
tensor_path=
snapshot_json=

while [[ $# -gt 0 ]]; do
  case "$1" in
    --verbose)
      verbose=1
      shift
      ;;
    --latest-snapshot)
      snapshot_json=$(resolve_latest_coherent_snapshot_json)
      shift
      ;;
    --snapshot-json)
      if [[ $# -lt 2 ]]; then
        echo "Usage: $0 [TENSOR_NPY] [--latest-snapshot] [--snapshot-json PATH] [--verbose]" >&2
        exit 1
      fi
      snapshot_json=$(absolutize_host_path "$2")
      shift 2
      ;;
    --*)
      echo "Usage: $0 [TENSOR_NPY] [--latest-snapshot] [--snapshot-json PATH] [--verbose]" >&2
      exit 1
      ;;
    *)
      if [[ -n "${tensor_path}" || -n "${snapshot_json}" ]]; then
        echo "Usage: $0 [TENSOR_NPY] [--latest-snapshot] [--snapshot-json PATH] [--verbose]" >&2
        exit 1
      fi
      tensor_path=$(resolve_tensor_snapshot_path "$1")
      shift
      ;;
  esac
done

if [[ -z "${tensor_path}" && -z "${snapshot_json}" ]]; then
  snapshot_json=$(resolve_latest_coherent_snapshot_json)
fi

if [[ -n "${tensor_path}" && -n "${snapshot_json}" ]]; then
  echo "Usage: $0 [TENSOR_NPY] [--latest-snapshot] [--snapshot-json PATH] [--verbose]" >&2
  exit 1
fi

config_path=$(absolutize_host_path "${LIVE_CONFIG}")
if [[ ! -f "${config_path}" ]]; then
  echo "Live coherent config not found: ${config_path}" >&2
  exit 1
fi

if [[ -n "${tensor_path}" ]]; then
  if [[ ! -f "${tensor_path}" ]]; then
    echo "Tensor snapshot not found: ${tensor_path}" >&2
    exit 1
  fi

  tensor_stem=$(basename "${tensor_path}" .npy)
  output_root="/tmp/usrp_spectrograms/coherent_power_validator_artifacts/${tensor_stem}"
  validator_output_dir="${output_root}/operator_live_validator"
  metadata_path="${output_root}/coherent_power_input_snapshot.json"

  echo "Using tensor: ${tensor_path}" >&2
  echo "Using live coherent config: ${config_path}" >&2
  echo "Preparing coherent snapshot metadata: ${metadata_path}" >&2

  require_container_mapped_path "tensor snapshot path" "${tensor_path}"
  require_container_mapped_path "validator output root" "${output_root}"

  mkdir -p "${output_root}"

  container_tensor_path=$(map_path_to_container "${tensor_path}")
  container_output_root=$(map_path_to_container "${output_root}")
  container_metadata_path="${container_output_root}/coherent_power_input_snapshot.json"
  container_validator_output_dir=$(map_path_to_container "${validator_output_dir}")

  python3 - "${tensor_path}" "${container_tensor_path}" "${config_path}" "${metadata_path}" <<'PY'
import ast
import json
import struct
import sys
from pathlib import Path


def load_npy_header(path: Path):
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path} is not a valid .npy file")
        major = handle.read(1)[0]
        handle.read(1)
        header_len = struct.unpack("<H", handle.read(2))[0] if major == 1 else struct.unpack("<I", handle.read(4))[0]
        return ast.literal_eval(handle.read(header_len).decode("latin1"))


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
        return float(text) if any(char in text for char in (".", "e", "E")) else int(text)
    except ValueError:
        return text


def parse_sections(config_path: Path):
    sections = {"fft": {}, "coherent_power_signal_detector": {}}
    current_section = None
    for raw_line in config_path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" "):
            current_section = line[:-1] if line.endswith(":") else None
            continue
        if current_section not in sections or not line.startswith("  ") or line.startswith("    "):
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
if len(shape) != 2 or "c" not in str(header.get("descr", "")):
    raise ValueError("Expected a 2D complex tensor snapshot")

rows, cols = shape
sections = parse_sections(config_path)
fft = sections["fft"]
coherent = sections["coherent_power_signal_detector"]
tensor_axis_order = "time_frequency" if rows < cols else "frequency_time"
freq_bins = cols if tensor_axis_order == "time_frequency" else rows
span_hz = float(fft.get("span", 0.0))
resolution_hz = float(fft.get("resolution", 0.0))
if resolution_hz <= 0.0 and span_hz > 0.0 and freq_bins > 0:
    resolution_hz = span_hz / float(freq_bins)

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
    "config": {
        key: coherent[key]
        for key in (
        "fast_performance", "chunk_bandwidth_hz", "chunk_overlap_hz", "uncalibrated_chunk_fraction",
            "uncalibrated_overlap_fraction", "ignore_sideband_percent", "ignore_sideband_hz", "frontend_row_q",
            "frontend_reference_q", "frontend_smooth_sigma", "frontend_max_boost_db", "coherence_weight",
            "power_weight", "power_assist_mode", "power_floor_time_q", "power_floor_global_q",
            "power_excess_start_db", "power_excess_full_db", "power_local_blend", "coherence_source_mode",
            "coherence_gate_start", "coherence_gate_full", "coherence_bridge_bias", "coherence_power_joint_weight",
            "score_threshold_mode", "fixed_score_threshold", "coherence_power_support_q", "coherence_power_q",
            "min_component_size", "filter_detection_mask", "grouping_seed_score_q", "grouping_bridge_freq_px",
            "grouping_bridge_time_px", "grouping_min_component_size", "grouping_min_freq_span_px",
            "grouping_min_time_span_px", "grouping_min_density", "grouping_time_continuity_ratio",
        )
        if key in coherent
    },
}
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
PY

  echo "Prepared metadata sidecar." >&2
else
  if [[ ! -f "${snapshot_json}" ]]; then
    echo "Coherent snapshot JSON not found: ${snapshot_json}" >&2
    exit 1
  fi

  snapshot_stem=$(basename "${snapshot_json}" .json)
  output_root="/tmp/coherent_power_snapshots/validator_regression_check"
  validator_output_dir="${output_root}/${snapshot_stem}_operator_live_replay"

  echo "Using latest coherent snapshot JSON: ${snapshot_json}" >&2

  require_container_mapped_path "snapshot JSON path" "${snapshot_json}"
  require_container_mapped_path "validator output directory" "${validator_output_dir}"

  mkdir -p "${output_root}"

  container_metadata_path=$(map_path_to_container "${snapshot_json}")
  container_validator_output_dir=$(map_path_to_container "${validator_output_dir}")
fi
echo "Ensuring demo container is running..." >&2
"${RUN_DEMO_CONTAINER}"
echo "Container ready: ${CONTAINER_NAME}" >&2

container_validator_bin=${VALIDATOR_BIN:-__AUTO__}
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
if [[ -z "${validator_bin}" || "${validator_bin}" == "__AUTO__" ]]; then
  for candidate in \
    "${BUILD_APP_DIR}/${VALIDATOR_NAME}" \
    "/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/${VALIDATOR_NAME}" \
    "/workspace/holohub-dev/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/${VALIDATOR_NAME}"; do
    if [[ -x "${candidate}" ]]; then
      validator_bin="${candidate}"
      break
    fi
  done
fi

if [[ -z "${validator_bin}" ]]; then
  echo "${VALIDATOR_NAME} not found in the container." >&2
  exit 1
fi

echo "Using ${VALIDATOR_NAME}: ${validator_bin}" >&2
echo "Snapshot JSON inside container: ${SNAPSHOT_JSON}" >&2
rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

cmd=("${validator_bin}" "--snapshot-json" "${SNAPSHOT_JSON}" "--output-dir" "${OUTPUT_DIR}")
if [[ "${VERBOSE_FLAG}" == "1" ]]; then
  cmd+=("--verbose")
fi
"${cmd[@]}"
'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
WORKING_DIR=$(pwd -P)

source "${APP_DIR}/bash_scripts/container_repo_guard.sh"

EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${APP_DIR}")}
HOST_REPO_ROOT=${HOST_REPO_ROOT:-${EXPECTED_REPO_ROOT}}
source "${APP_DIR}/bash_scripts/container_env.sh"
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
EVAL_NAME=${EVAL_NAME:-run_offline_cuda_detector_eval}
EVAL_BIN=${EVAL_BIN:-}
BASE_CONFIG=${BASE_CONFIG:-${SCRIPT_DIR}/config_cuda_dino_performance_single_channel_offline_eval.yaml}
STAGED_INPUT_ROOT=${STAGED_INPUT_ROOT:-/tmp/usrp_spectrograms/offline_eval_inputs_refactor}
SAFE_OUTPUT_ROOT_PREFIX=${SAFE_OUTPUT_ROOT_PREFIX:-/tmp/usrp_spectrograms/offline_cuda_detector_eval}

usage() {
  cat >&2 <<'EOF'
Usage: run_offline_cuda_detector_eval_refactor.sh --input-file PATH [options]

Options:
  --input-file PATH      SigMF IQ data file, usually *.sigmf-data. Required.
  --config PATH          Offline CUDA DINO config to replay.
                         Default: config_cuda_dino_performance_single_channel_offline_eval.yaml
  --output-root DIR      Host output root for notebook artifacts.
                         Default: /tmp/usrp_spectrograms/offline_cuda_detector_eval/<input-stem>
  --progress-every N     Offline app progress log stride.
  --verbose              Print the exact container command.
  --help                 Show this message.

This script does not build. Rebuild first from applications/usrp_wideband_signal_detection with:
  sudo ./rebuild_demo_container_app.sh
EOF
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

map_path_to_container() {
  local raw_path=$1
  if [[ -z "${raw_path}" ]]; then
    return 0
  fi

  case "${raw_path}" in
    /tmp/usrp_spectrograms/*)
      printf '/workspace/spectrograms/%s' "${raw_path#/tmp/usrp_spectrograms/}"
      ;;
    /tmp/usrp_spectrograms)
      printf '/workspace/spectrograms'
      ;;
    "${HOST_REPO_ROOT}"/*)
      printf '/workspace/holohub/%s' "${raw_path#${HOST_REPO_ROOT}/}"
      ;;
    "${HOST_REPO_ROOT}")
      printf '/workspace/holohub'
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
    echo "Use a path under /tmp/usrp_spectrograms or under ${HOST_REPO_ROOT}." >&2
    exit 1
  fi
}

sigmf_meta_for_data_path() {
  local data_path=$1
  python3 - "$data_path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if path.name.endswith('.sigmf-data'):
    print(path.with_name(path.name[:-len('.sigmf-data')] + '.sigmf-meta'))
else:
    print(path.with_suffix('.sigmf-meta'))
PY
}

validate_sigmf_input() {
  local data_path=$1
  local meta_path=$2
  python3 - "$data_path" "$meta_path" <<'PY'
from pathlib import Path
import json
import math
import sys

data_path = Path(sys.argv[1])
meta_path = Path(sys.argv[2])
if not meta_path.is_file():
    raise SystemExit(f"Missing SigMF metadata sidecar: {meta_path}")

with meta_path.open("r", encoding="utf-8") as handle:
    meta = json.load(handle)

global_info = meta.get("global", {})
datatype = global_info.get("core:datatype")
sample_rate = global_info.get("core:sample_rate")
if not isinstance(datatype, str) or not datatype:
    raise SystemExit(f"{meta_path}: missing global.core:datatype")
try:
    sample_rate_hz = float(sample_rate)
except (TypeError, ValueError):
    raise SystemExit(f"{meta_path}: invalid global.core:sample_rate") from None
if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
    raise SystemExit(f"{meta_path}: invalid global.core:sample_rate")

base = datatype[:-3] if datatype.endswith(("_le", "_be")) else datatype
if not base.startswith("c") or len(base) < 3:
    raise SystemExit(f"{meta_path}: offline CUDA DINO replay requires a complex datatype, got {datatype!r}")
scalar_kind = base[1]
try:
    scalar_bits = int(base[2:])
except ValueError:
    raise SystemExit(f"{meta_path}: unparseable datatype bit width in {datatype!r}") from None
if scalar_kind not in {"i", "u", "f"} or scalar_bits <= 0 or scalar_bits % 8 != 0:
    raise SystemExit(f"{meta_path}: unsupported datatype {datatype!r}")

num_channels = int(global_info.get("core:num_channels", 1))
if num_channels != 1:
    raise SystemExit(f"{meta_path}: this replay path expects core:num_channels == 1, got {num_channels}")

bytes_per_complex = (scalar_bits // 8) * 2 * num_channels
file_size = data_path.stat().st_size
if file_size == 0:
    raise SystemExit(f"{data_path}: input file is empty")
if file_size % bytes_per_complex:
    raise SystemExit(
        f"{data_path}: file size {file_size} is not aligned to {bytes_per_complex} bytes/complex sample for {datatype}"
    )
total_complex_samples = file_size // bytes_per_complex

captures = meta.get("captures", [])
capture = captures[0] if captures else {}
capture_sample_start = int(capture.get("core:sample_start", 0))
center_frequency = capture.get("core:frequency")
annotations = meta.get("annotations", [])

print("SigMF input check:")
print(f"  data: {data_path}")
print(f"  meta: {meta_path}")
print(f"  datatype: {datatype}")
print(f"  sample_rate_hz: {sample_rate_hz:g}")
print(f"  num_channels: {num_channels}")
print(f"  capture_sample_start: {capture_sample_start}")
print(f"  center_frequency_hz: {center_frequency if center_frequency is not None else 'not set'}")
print(f"  annotations: {len(annotations) if isinstance(annotations, list) else 0}")
print(f"  total_complex_samples: {total_complex_samples}")
PY
}

stage_input_into_mounted_dir() {
  local raw_input_path=$1
  local raw_meta_path=$2
  local staged_dir="${STAGED_INPUT_ROOT}"
  local input_basename
  local staged_path
  local staged_meta_path

  input_basename=$(basename "${raw_input_path}")
  mkdir -p "${staged_dir}"
  staged_path="${staged_dir}/${input_basename}"
  staged_meta_path="${staged_dir}/$(basename "${raw_meta_path}")"

  rm -f -- "${staged_path}" "${staged_meta_path}"

  if ! ln "${raw_input_path}" "${staged_path}" 2>/dev/null; then
    cp --reflink=auto --sparse=always "${raw_input_path}" "${staged_path}"
  fi
  if ! ln "${raw_meta_path}" "${staged_meta_path}" 2>/dev/null; then
    cp --reflink=auto --sparse=always "${raw_meta_path}" "${staged_meta_path}"
  fi

  printf '%s' "${staged_path}"
}

reset_output_root() {
  local raw_output_root=$1
  case "${raw_output_root}" in
    "${SAFE_OUTPUT_ROOT_PREFIX}"/*)
      ;;
    *)
      echo "Refusing to clear output root outside ${SAFE_OUTPUT_ROOT_PREFIX}: ${raw_output_root}" >&2
      echo "Choose an output root under ${SAFE_OUTPUT_ROOT_PREFIX} or clear it manually." >&2
      exit 1
      ;;
  esac

  if [[ -e "${raw_output_root}" ]]; then
    rm -rf -- "${raw_output_root}"
  fi
  mkdir -p "${raw_output_root}"
}

ensure_container_ready() {
  if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
    echo "Container not found: ${CONTAINER_NAME}" >&2
    echo "Create it first with ./build_demo_container.sh" >&2
    exit 1
  fi
  ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"
  if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    sudo docker start "${CONTAINER_NAME}" >/dev/null
  fi
}

audit_manifest() {
  local output_root=$1
  python3 - "$output_root" <<'PY'
from pathlib import Path
import csv
import json
import sys

output_root = Path(sys.argv[1])
summary_path = output_root / "offline_eval_summary.json"
manifest_path = output_root / "frame_manifest.csv"
if not summary_path.is_file():
    raise SystemExit(f"Missing summary: {summary_path}")
if not manifest_path.is_file():
    raise SystemExit(f"Missing manifest: {manifest_path}")

summary = json.loads(summary_path.read_text())
with manifest_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit(f"Manifest has no frame rows: {manifest_path}")

expected_frames = int(summary.get("full_frame_count", summary.get("total_frames", 0)))
samples_per_frame = int(summary.get("samples_per_frame", 0))
processed_complex_samples = int(summary.get("processed_complex_samples", summary.get("total_complex_samples", 0)))
input_total_complex_samples = int(summary.get("input_total_complex_samples", processed_complex_samples))
dropped_tail_complex_samples = int(
  summary.get("dropped_tail_complex_samples", input_total_complex_samples - processed_complex_samples)
)
global_sample_start = int(summary.get("global_sample_start", rows[0]["file_offset_complex"]))
global_sample_end = int(summary.get("global_sample_end", global_sample_start + processed_complex_samples))
input_global_sample_end = int(summary.get("input_global_sample_end", global_sample_start + input_total_complex_samples))

if expected_frames <= 0:
    raise SystemExit("Summary reports zero complete frames; no offline eval artifacts should be reviewed")
if samples_per_frame <= 0:
    raise SystemExit("Summary is missing a positive samples_per_frame")
if processed_complex_samples != expected_frames * samples_per_frame:
    raise SystemExit(
        f"Processed sample count {processed_complex_samples} does not equal "
        f"expected_frames * samples_per_frame ({expected_frames} * {samples_per_frame})"
    )
if input_total_complex_samples != processed_complex_samples + dropped_tail_complex_samples:
    raise SystemExit(
        "Input sample accounting is inconsistent: "
        f"input_total={input_total_complex_samples}, processed={processed_complex_samples}, "
        f"dropped_tail={dropped_tail_complex_samples}"
    )
if global_sample_end != global_sample_start + processed_complex_samples:
    raise SystemExit(
        f"global_sample_end {global_sample_end} does not match start + processed samples "
        f"{global_sample_start + processed_complex_samples}"
    )
if input_global_sample_end != global_sample_start + input_total_complex_samples:
    raise SystemExit(
        f"input_global_sample_end {input_global_sample_end} does not match start + input samples "
        f"{global_sample_start + input_total_complex_samples}"
    )
if len(rows) != expected_frames:
  raise SystemExit(f"Manifest has {len(rows)} rows but expected {expected_frames} complete-frame rows")

required = [
    "frame_number",
    "file_offset_complex",
    "data_end_complex",
    "frame_end_complex",
    "complex_samples_read",
    "complex_samples_padded",
    "samples_per_row",
]
missing = [name for name in required if name not in rows[0]]
if missing:
    raise SystemExit(f"Manifest is missing refactor audit columns: {missing}")

previous_frame_end = None
for index, row in enumerate(rows, start=1):
    frame_number = int(row["frame_number"])
    frame_start = int(row["file_offset_complex"])
    data_end = int(row["data_end_complex"])
    frame_end = int(row["frame_end_complex"])
    samples_read = int(row["complex_samples_read"])
    samples_padded = int(row["complex_samples_padded"])
    samples_per_row = int(row["samples_per_row"])
    expected_frame_start = global_sample_start + (index - 1) * samples_per_frame
    expected_frame_end = expected_frame_start + samples_per_frame
    if frame_number != index:
        raise SystemExit(f"Frame numbering is not contiguous at row {index}: got frame {frame_number}")
    if frame_start != expected_frame_start:
        raise SystemExit(f"Frame {frame_number}: frame_start {frame_start} does not match expected {expected_frame_start}")
    if data_end != expected_frame_end:
        raise SystemExit(f"Frame {frame_number}: data_end {data_end} does not match expected {expected_frame_end}")
    if frame_end != expected_frame_end:
        raise SystemExit(f"Frame {frame_number}: frame_end {frame_end} does not match expected {expected_frame_end}")
    if data_end - frame_start != samples_read:
        raise SystemExit(f"Frame {frame_number}: data_end - frame_start does not equal complex_samples_read")
    if frame_end - frame_start != samples_read + samples_padded:
        raise SystemExit(f"Frame {frame_number}: frame span does not equal read + padded samples")
    if samples_read != samples_per_frame:
        raise SystemExit(f"Frame {frame_number}: complex_samples_read {samples_read} does not equal full frame size {samples_per_frame}")
    if samples_padded != 0:
        raise SystemExit(f"Frame {frame_number}: complete-frame replay should not pad samples, got {samples_padded}")
    if samples_per_row <= 0:
        raise SystemExit(f"Frame {frame_number}: samples_per_row must be positive")
    if previous_frame_end is not None and frame_start != previous_frame_end:
        raise SystemExit(f"Frame {frame_number}: frame_start {frame_start} does not match prior frame_end {previous_frame_end}")
    previous_frame_end = frame_end

if previous_frame_end != global_sample_end:
    raise SystemExit(f"Manifest ends at sample {previous_frame_end}, expected processed end {global_sample_end}")

artifact_columns = [
    "spectrogram_preview_pgm",
    "spectrogram_tensor_npy",
    "mask_preview_pgm",
    "mask_npy",
    "gt_annotations_json",
    "gt_mask_npy",
]
for row in rows:
    for column in artifact_columns:
        value = row.get(column, "").strip()
        if not value:
            raise SystemExit(f"Frame {row['frame_number']}: missing artifact path in {column}")
        artifact_path = output_root / value
        if not artifact_path.is_file():
            raise SystemExit(f"Frame {row['frame_number']}: artifact listed in {column} does not exist: {artifact_path}")

first = rows[0]
last = rows[-1]
print("Offline eval artifact audit:")
print(f"  output_root: {output_root}")
print(f"  complete_frames: {len(rows)}")
print(f"  samples_per_frame: {samples_per_frame}")
print(f"  input_total_complex_samples: {input_total_complex_samples}")
print(f"  processed_complex_samples: {processed_complex_samples}")
print(f"  global_sample_start: {first['file_offset_complex']}")
print(f"  global_frame_end: {last['frame_end_complex']}")
print(
    f"  dropped_tail_complex_samples: {dropped_tail_complex_samples} "
    f"(not enough to make a full frame of {samples_per_frame} samples)"
)
print(f"  sample_rate_hz: {summary.get('input_sample_rate_hz')}")
print(f"  manifest: {manifest_path}")
print(f"  notebook OUTPUT_ROOT: {output_root}")
print(
  f"Dropped {dropped_tail_complex_samples} samples from the end of the file because they were not "
  f"enough to make a full frame of {samples_per_frame} samples."
)
PY
}

input_file_path=${INPUT_FILE_PATH:-}
config_path=${CONFIG_PATH:-${BASE_CONFIG}}
output_root=${OUTPUT_ROOT:-}
progress_every=${PROGRESS_EVERY:-}
verbose=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-file)
      input_file_path=$2
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
    --progress-every)
      progress_every=$2
      shift 2
      ;;
    --verbose)
      verbose=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${input_file_path}" ]]; then
  usage
  exit 1
fi

input_file_path=$(absolutize_host_path "${input_file_path}")
config_path=$(absolutize_host_path "${config_path}")
if [[ ! -f "${input_file_path}" ]]; then
  echo "Offline IQ file not found: ${input_file_path}" >&2
  exit 1
fi
if [[ ! -f "${config_path}" ]]; then
  echo "Config not found: ${config_path}" >&2
  exit 1
fi

meta_path=$(sigmf_meta_for_data_path "${input_file_path}")
validate_sigmf_input "${input_file_path}" "${meta_path}"

if [[ -z "${output_root}" ]]; then
  input_basename=$(basename "${input_file_path}")
  input_stem=${input_basename%.sigmf-data}
  input_stem=${input_stem%.*}
  output_root="${SAFE_OUTPUT_ROOT_PREFIX}/${input_stem}"
else
  output_root=$(absolutize_host_path "${output_root}")
fi

require_container_mapped_path "config" "${config_path}"
require_container_mapped_path "output root" "${output_root}"

if [[ "$(map_path_to_container "${input_file_path}")" == "${input_file_path}" ]]; then
  echo "Staging input and SigMF sidecar into mounted scratch space: ${STAGED_INPUT_ROOT}" >&2
  input_file_path=$(stage_input_into_mounted_dir "${input_file_path}" "${meta_path}")
  meta_path=$(sigmf_meta_for_data_path "${input_file_path}")
fi

require_container_mapped_path "offline IQ file" "${input_file_path}"
reset_output_root "${output_root}"
ensure_container_ready

container_input_file=$(map_path_to_container "${input_file_path}")
container_config_path=$(map_path_to_container "${config_path}")
container_output_root=$(map_path_to_container "${output_root}")
container_eval_bin=${EVAL_BIN:-__AUTO__}

docker_exec_flags=(-i)
if [[ -t 0 && -t 1 ]]; then
  docker_exec_flags=(-it)
fi

sudo docker exec "${docker_exec_flags[@]}" \
  -e EVAL_BIN="${container_eval_bin}" \
  -e EVAL_NAME="${EVAL_NAME}" \
  -e BUILD_APP_DIR="${BUILD_APP_DIR}" \
  -e INPUT_FILE_PATH="${container_input_file}" \
  -e CONFIG_PATH="${container_config_path}" \
  -e OUTPUT_ROOT="${container_output_root}" \
  -e PROGRESS_EVERY="${progress_every}" \
  -e VERBOSE_FLAG="${verbose}" \
  "${CONTAINER_NAME}" \
  bash -lc '
set -euo pipefail

eval_bin="${EVAL_BIN}"
if [[ -z "${eval_bin}" || "${eval_bin}" == "__AUTO__" ]]; then
  candidates=(
    "${BUILD_APP_DIR}/${EVAL_NAME}"
    "/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/${EVAL_NAME}"
    "/workspace/holohub/build/applications/usrp_wideband_signal_detection/${EVAL_NAME}"
    "/workspace/holohub/build/usrp_wideband_signal_detection/${EVAL_NAME}"
  )
  eval_bin=""
  for candidate in "${candidates[@]}"; do
    if [[ -x "${candidate}" ]]; then
      eval_bin="${candidate}"
      break
    fi
  done
fi

if [[ -z "${eval_bin}" ]]; then
  echo "${EVAL_NAME} was not found in the container build tree." >&2
  echo "Rebuild first with: sudo ./rebuild_demo_container_app.sh" >&2
  exit 1
fi

cmd=(
  "${eval_bin}"
  "--config" "${CONFIG_PATH}"
  "--input-file" "${INPUT_FILE_PATH}"
  "--output-root" "${OUTPUT_ROOT}"
)
if [[ -n "${PROGRESS_EVERY}" ]]; then
  cmd+=("--progress-every" "${PROGRESS_EVERY}")
fi

if [[ "${VERBOSE_FLAG}" == "1" ]]; then
  echo "Running offline refactor eval command:" >&2
  printf "  %q" "${cmd[@]}" >&2
  printf "\n" >&2
fi

"${cmd[@]}"
'

audit_manifest "${output_root}"
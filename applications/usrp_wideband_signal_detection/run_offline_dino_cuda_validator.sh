#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
WORKING_DIR=$(pwd -P)
BASE_VALIDATOR_SCRIPT=${BASE_VALIDATOR_SCRIPT:-${SCRIPT_DIR}/run_offline_dino_validator_performance.sh}
VALIDATOR_NAME=${VALIDATOR_NAME:-offline_dino_validator_performance}
DEFAULT_CONFIG=${DEFAULT_CONFIG:-${SCRIPT_DIR}/config_cuda_dino_scaffold_single_channel.yaml}

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

env VALIDATOR_NAME="${VALIDATOR_NAME}" DEFAULT_CONFIG="${DEFAULT_CONFIG}" "${BASE_VALIDATOR_SCRIPT}" "${forwarded_args[@]}"

if [[ -z "${output_dir}" ]]; then
  echo "cuda validator wrapper could not determine output_dir" >&2
  exit 1
fi

OUTPUT_DIR_FOR_VALIDATION="${output_dir}" DEFAULT_CONFIG_FOR_VALIDATION="${DEFAULT_CONFIG}" SCRIPT_DIR_FOR_VALIDATION="${SCRIPT_DIR}" python3 - <<'PY'
import json
import os
import sys
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR_FOR_VALIDATION"]).resolve()
default_config = Path(os.environ["DEFAULT_CONFIG_FOR_VALIDATION"]).resolve()
script_dir = Path(os.environ["SCRIPT_DIR_FOR_VALIDATION"]).resolve()
repo_root = script_dir.parents[1]
summary_path = output_dir / "offline_validation_summary.json"
chunk_debug_dir = output_dir / "chunk_debug"
chunk_debug_summary_path = chunk_debug_dir / "chunk_debug_summary.json"


def to_host_path(raw: str) -> Path:
  path = Path(str(raw or ""))
  text = str(path)
  if not text:
    return path
  if text.startswith("/workspace/spectrograms/"):
    return Path("/tmp/usrp_spectrograms") / text.removeprefix("/workspace/spectrograms/")
  if text == "/workspace/spectrograms":
    return Path("/tmp/usrp_spectrograms")
  if text.startswith("/workspace/dino_masks/"):
    return Path("/tmp/usrp_dino_masks") / text.removeprefix("/workspace/dino_masks/")
  if text == "/workspace/dino_masks":
    return Path("/tmp/usrp_dino_masks")
  if text.startswith("/workspace/holohub/"):
    return repo_root / text.removeprefix("/workspace/holohub/")
  if text == "/workspace/holohub":
    return repo_root
  if text.startswith("/workspace/holohub-dev/"):
    return repo_root / text.removeprefix("/workspace/holohub-dev/")
  if text == "/workspace/holohub-dev":
    return repo_root
  return path

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
  "corrected_resized_npy",
  "projected_grouped_mask_npy",
  "projected_grouped_score_npy",
  "merged_box_mask_npy",
  "final_mask_npy",
  "chunk_plan_json",
  "chunk_results_json",
  "projected_boxes_json",
  "merged_boxes_json",
]
required_debug_artifacts = [
  "corrected_resized_npy",
  "runtime_input_gray_npy",
  "dino_score_raw_npy",
  "dino_score_raw_deweighted_npy",
  "coherence_gate_npy",
  "hybrid_contrib_npy",
  "combined_score_npy",
  "valid_mask_npy",
  "bridged_mask_npy",
  "grouped_mask_npy",
  "grouped_boxes_json",
  "final_mask_npy",
  "final_mask_source_npy",
  "final_mask_projected_npy",
]

for key in required_summary_artifacts:
  path = to_host_path(str(summary.get(key, "") or ""))
  if not path.exists():
    failures.append(f"missing summary artifact for {key}: {path}")

for key in required_debug_artifacts:
  path = to_host_path(str(debug_summary.get(key, "") or ""))
  if not path.exists():
    failures.append(f"missing debug artifact for {key}: {path}")

artifact_contract = str(debug_summary.get("artifact_contract", "") or "")
if artifact_contract not in {"chunk_fixed_detector_grid_v1", "chunk_no_extra_sideband_crop_v2"}:
  failures.append(
    f"unexpected artifact_contract {artifact_contract!r} in {chunk_debug_summary_path}"
  )

component_count = int(debug_summary.get("component_count", 0) or 0)
grouped_box_count = int(debug_summary.get("grouped_box_count", 0) or 0)
patch_rows = int(debug_summary.get("patch_rows", 0) or 0)
patch_cols = int(debug_summary.get("patch_cols", 0) or 0)
feature_dim = int(debug_summary.get("feature_dim", 0) or 0)
if patch_rows <= 0 or patch_cols <= 0 or feature_dim <= 0:
  failures.append(
    f"invalid patch metadata patch_rows={patch_rows}, patch_cols={patch_cols}, feature_dim={feature_dim}"
  )

manifest = {
  "bundle_kind": "offline_dino_cuda_validator_bundle_v1",
  "default_config": str(default_config),
  "output_dir": str(output_dir),
  "summary_json": str(summary_path),
  "chunk_debug_summary_json": str(chunk_debug_summary_path),
  "artifact_contract": artifact_contract,
  "component_count": component_count,
  "grouped_box_count": grouped_box_count,
  "summary_artifacts": {key: str(to_host_path(str(summary.get(key, "") or ""))) for key in required_summary_artifacts},
  "debug_artifacts": {key: str(to_host_path(str(debug_summary.get(key, "") or ""))) for key in required_debug_artifacts},
}
manifest_path = output_dir / "cuda_artifact_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

if failures:
  print("CUDA validator run completed, but the artifact bundle is incomplete:", file=sys.stderr)
  for failure in failures:
    print(f"- {failure}", file=sys.stderr)
  sys.exit(1)

print(f"Validated CUDA artifact bundle: {manifest_path}", file=sys.stderr)
PY
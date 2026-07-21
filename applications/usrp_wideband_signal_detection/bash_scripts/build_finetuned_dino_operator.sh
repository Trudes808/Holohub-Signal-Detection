#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash (this script uses bash-only syntax).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Build the native finetuned_dino_detector operator into the container app.
#
# Prereqs (authored separately -- see ../../../build_instructions.md):
#   1. operators/finetuned_dino_detector/{finetuned_dino_detector.hpp,.cu,CMakeLists.txt,metadata.json}
#   2. app CMakeLists: add holoscan::ops::finetuned_dino_detector to DEPENDS OPERATORS + target_link,
#      and the operator dir under add_subdirectory / the operators path list.
#   3. a finetuned_dino_detector: config block + DetectorAdapter registration in
#      run_offline_cuda_detector_eval.cpp (and main.cpp for live).
#   4. exported weights: run dino_fine_tuning/export_finetuned_models.sh first.
#
# This wraps the standard rebuild (in-container ninja). Needs docker/sudo (lab-admin).
# Usage: sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./build_finetuned_dino_operator.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd -P)"

OP_DIR="${REPO_ROOT}/operators/finetuned_dino_detector"
if [[ ! -f "${OP_DIR}/finetuned_dino_detector.cu" ]]; then
  echo "WARNING: ${OP_DIR}/finetuned_dino_detector.cu not found." >&2
  echo "  The operator C++ source hasn't been authored yet -- see ${REPO_ROOT}/build_instructions.md" >&2
  echo "  (rebuild will still run, but it won't produce the new detector until the source + CMake exist)." >&2
fi
for tag in m1 m2; do
  w="${REPO_ROOT}/dino_fine_tuning/weights/finetuned_dino_${tag}.ts"
  [[ -f "$w" ]] || echo "WARNING: missing $w -- run dino_fine_tuning/export_finetuned_models.sh" >&2
done

echo "=== rebuilding container app (CONTAINER_NAME=${CONTAINER_NAME:-<container_env default>}) ==="
exec "${SCRIPT_DIR}/rebuild_demo_container_app.sh"

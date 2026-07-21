#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash (this script uses bash-only syntax).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Build ALL native detector operators in ONE container rebuild: finetuned_dino_detector + yolo_detector
# (plus the existing coherent_power / cuda_dino). The in-container ninja build compiles every operator
# wired into the app CMake, so this is a single rebuild once the operators + CMake + configs exist.
#
# Prereqs (see ../../../build_instructions.md):
#   - operators/{finetuned_dino_detector,yolo_detector}/*  authored
#   - app CMakeLists: DEPENDS OPERATORS + target_link + operator dirs added
#   - detector config blocks + DetectorAdapter registrations
#   - weights exported: dino_fine_tuning/export_finetuned_models.sh + yolo_training/export_yolo_models.sh
#
# Needs docker/sudo (lab-admin). Usage:
#   sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./build_all_detector_operators.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd -P)"

echo "--- checking exported weights ---"
for w in dino_fine_tuning/weights/finetuned_dino_m1.ts dino_fine_tuning/weights/finetuned_dino_m2.ts \
         yolo_training/weights/yolo26s.torchscript yolo_training/weights/yolo26m.torchscript; do
  [[ -f "${REPO_ROOT}/${w}" ]] && echo "  ok   ${w}" || echo "  MISSING ${w} -- run the matching export script" >&2
done
echo "--- checking operator sources ---"
for op in finetuned_dino_detector yolo_detector; do
  if [[ -f "${REPO_ROOT}/operators/${op}/${op}.cu" ]]; then echo "  ok   operators/${op}"
  else echo "  MISSING operators/${op}/${op}.cu -- author it (see build_instructions.md)" >&2; fi
done
echo "--- rebuilding container app (compiles all wired operators) ---"
exec "${SCRIPT_DIR}/rebuild_demo_container_app.sh"

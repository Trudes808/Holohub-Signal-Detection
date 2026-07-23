#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash.
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Produce masks for all detectors into ONE batch root, reusing whatever already exists.
#
# The C++ detectors (coherent_power, cuda_dino) come from the container binary (run_batch_offline_eval.py)
# and carry the shared GT/manifest -- they are REUSED if already present (default root already has them).
# The baselines (3dB_power, blob_detection) + ML detectors (dino_finetuned[_m1], yolo, yolo26s) are then
# produced by the collaborator's run_full_comparison.py stages, mirroring that GT.
#
# Usage:
#   ./create_all_masks.sh                       # report + run baselines+ml+eval into the default root
#   BATCH_ROOT=/path ./create_all_masks.sh      # different root
#   STAGES="baselines ml" ./create_all_masks.sh # subset (default: baselines ml eval)
#   SNIP_ENV=dinov3 ./create_all_masks.sh        # conda env with numpy+scipy+torch+ultralytics
# If coherent_power/cuda_dino are missing from the root, produce them first (needs the container):
#   (cd ../signal_detection_experiments && sudo env CONTAINER_NAME=... python3 run_batch_offline_eval.py \
#        --captures-dir /home/bqn82/captures --detectors coherent_power cuda_dino --run-id <id> \
#        --output-root <root> --repack-masks)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
CMP_DIR="$(cd "${SCRIPT_DIR}/../baseline_comparisons" && pwd -P)"
BATCH_ROOT="${BATCH_ROOT:-/tmp/usrp_spectrograms/all_detectors}"
CONFIG="${CONFIG:-${CMP_DIR}/comparison_config.yaml}"
STAGES="${STAGES:-baselines ml eval}"
SNIP_ENV="${SNIP_ENV:-dinov3}"

echo "=== existing masks under ${BATCH_ROOT} ==="
python3 "${SCRIPT_DIR}/find_masks.py" "${BATCH_ROOT}" || true

if [[ ! -d "${BATCH_ROOT}/coherent_power" || ! -d "${BATCH_ROOT}/cuda_dino" ]]; then
  echo "WARNING: coherent_power/cuda_dino not found in ${BATCH_ROOT}." >&2
  echo "  Produce them first with the container (see the header of this script)." >&2
fi

echo "=== activating env ${SNIP_ENV} (needs numpy+scipy+torch+ultralytics) ==="
source ~/miniforge3/etc/profile.d/conda.sh; conda activate "${SNIP_ENV}"
python3 - <<'PYCHK'
mods=[]
for m in ("numpy","scipy","torch","ultralytics"):
    try: __import__(m)
    except Exception: mods.append(m)
print("  missing deps:", mods or "none")
PYCHK

echo "=== run_full_comparison stages: ${STAGES} (baselines + ML into ${BATCH_ROOT}) ==="
python3 "${CMP_DIR}/run_full_comparison.py" --config "${CONFIG}" \
    --batch-root "${BATCH_ROOT}" --stages ${STAGES}
echo "=== final inventory ==="
python3 "${SCRIPT_DIR}/find_masks.py" "${BATCH_ROOT}"

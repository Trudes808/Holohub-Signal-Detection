#!/usr/bin/env bash
set -euo pipefail

# Label antenna-ON OTA captures with the COHERENT detector (high-precision teacher) for the DINO-FT
# domain-adaptation fine-tune. Runs the offline eval binary (same operators as live), dumping masks
# only (--no-tensors) to <LABELS_DIR>/<stem>/mask_arrays/. build_dataset_ota.py then reads those.
#
#   ./label_ota.sh                 # all *_on_* captures in ~/Documents/captures
# Env: CAP_DIR, LABELS_DIR (must be under /tmp/usrp_spectrograms or the repo), TEACHER_CFG, GLOB.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP_DIR="${APP_DIR:-${SCRIPT_DIR}/../applications/usrp_wideband_signal_detection}"
USER_HOME=$(getent passwd "${SUDO_USER:-${USER}}" | cut -d: -f6)
CAP_DIR="${CAP_DIR:-${USER_HOME}/Documents/captures}"
LABELS_DIR="${LABELS_DIR:-/tmp/usrp_spectrograms/ota_labels}"
TEACHER_CFG="${TEACHER_CFG:-${APP_DIR}/old_configs/config_coherent_teacher.yaml}"
GLOB="${GLOB:-x410_ota_*_on_*.sigmf-data}"   # ON captures only (both bands, both gains)

sudo mkdir -p "${LABELS_DIR}" && sudo chmod 1777 "${LABELS_DIR}" 2>/dev/null || true
cd "${APP_DIR}"
shopt -s nullglob
caps=(${CAP_DIR}/${GLOB})
echo "==> ${#caps[@]} ON captures to label with teacher ${TEACHER_CFG##*/}"
for cap in "${caps[@]}"; do
  stem=$(basename "$cap" .sigmf-data)
  if [ -d "${LABELS_DIR}/${stem}/mask_arrays" ] && [ -n "$(ls -A "${LABELS_DIR}/${stem}/mask_arrays" 2>/dev/null)" ]; then
    echo "==> ${stem}: already labeled, skipping"; continue
  fi
  echo "==> labeling ${stem}"
  python3 run_cuda_dino_offline_file.py "$cap" --detector coherent_power \
    --config "${TEACHER_CFG}" --no-tensors --output-root "${LABELS_DIR}/${stem}" \
    --progress-every 50 2>&1 | grep -iE "processed [0-9]+ masks|complete|ERROR|error" | tail -3
done
echo "==> labels in ${LABELS_DIR}"

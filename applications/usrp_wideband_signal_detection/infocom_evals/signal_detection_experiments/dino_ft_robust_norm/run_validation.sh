#!/usr/bin/env bash
# Robust-normalization validation driver.
# Runs the DINO-FT offline eval in 3 normalization modes (fixed / adapt / robust) across 3 scenes
# (dense-noiseless "clean", dense-low-SNR "snr0", sparse-realistic OTA capture), dumping the [0,1]
# model input + mask for a handful of frames each. The dumps land per-(scene,mode) so analyze.py can
# quantify whether robust normalization collapses on the degenerate scenes (it must not) and whether it
# still adapts on the sparse scene (it should). See README.md.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}/../../.."
cd "${APP_DIR}"

SNR_BENCH=/tmp/usrp_spectrograms/snr_bench
SPARSE_CAP="${HOME}/Documents/captures/x410_ota_2g4_gain10_20260908.sigmf-data"
DUMP_ROOT=/tmp/usrp_spectrograms/robust_val
OUT_ROOT=/tmp/usrp_spectrograms/robust_val_out
DET=cuda_dino_finetuned

declare -A SCENE_INPUT=(
  [clean]="${SNR_BENCH}/clean/comprehensive_ordered_clean.sigmf-data"
  [snr0]="${SNR_BENCH}/snr0/comprehensive_ordered_snr0.sigmf-data"
  [sparse]="${SPARSE_CAP}"
)

sudo mkdir -p "${DUMP_ROOT}" "${OUT_ROOT}"

for scene in clean snr0 sparse; do
  IN="${SCENE_INPUT[$scene]}"
  if [[ ! -f "${IN}" ]]; then echo "!! missing input for ${scene}: ${IN}"; continue; fi
  for mode in fixed adapt robust; do
    echo "==================== ${scene} / ${mode} ===================="
    # clear the per-mode dump dir the operator writes into (owned by root -> sudo)
    sudo rm -rf "${DUMP_ROOT}/${mode}"
    python3 run_cuda_dino_offline_file.py "${IN}" \
      --detector "${DET}" \
      --config "${SCRIPT_DIR}/config_${mode}.yaml" \
      --output-root "${OUT_ROOT}/${scene}_${mode}" \
      --progress-every 0 2>&1 | tail -8
    # relocate the dump to a per-(scene,mode) dir for analysis
    sudo rm -rf "${DUMP_ROOT}/${scene}_${mode}"
    if [[ -d "${DUMP_ROOT}/${mode}" ]]; then
      sudo mv "${DUMP_ROOT}/${mode}" "${DUMP_ROOT}/${scene}_${mode}"
      echo "  dumped -> ${DUMP_ROOT}/${scene}_${mode}"
    else
      echo "  !! no dump produced for ${scene}/${mode}"
    fi
  done
done
# make the dumps readable by the analysis (they were written by the container's root)
sudo chown -R "$(id -u):$(id -g)" "${DUMP_ROOT}" 2>/dev/null || true
echo "ALL DONE"

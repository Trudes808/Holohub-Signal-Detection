#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash (this script uses bash arrays).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Run ALL detectors offline over one or more captures -> masks under OUT_ROOT/<detector>/<stem>/.
# Skips any detector whose base config isn't present yet (operator not wired), so it works today for
# coherent_power + cuda_dino and auto-includes finetuned_dino/yolo once they're built. Needs docker/sudo.
#
# Usage:
#   sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./run_all_detectors.sh                 # FULL SNR sweep (default)
#   sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./run_all_detectors.sh a.sigmf-data b.sigmf-data
# Default captures = the full attenuation/SNR sweep the notebook uses (0-60 dB incl. 30 dB v2) so the
# measurements are accurate across SNR (not a single point). Pass explicit files to override, or set
# ATTENS / CAPTURES_DIR. NOTE: full sweep = 14 captures x up to 6 detectors x ~7 s replay each -- long.
# Env: DETECTORS, ATTENS, CAPTURES_DIR, OUT_ROOT=/tmp/usrp_spectrograms/all_detectors, EXTRA_ARGS="--no-tensors"
# Flags: --repack (pack masks .npy -> .packed.npz per run so the notebook/snipper can read them)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
#USER_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"
USER_HOME="/home/bqn82/"


DETECTORS="${DETECTORS:-coherent_power cuda_dino finetuned_dino finetuned_dino_m2 yolo26s yolo26m}"
OUT_ROOT="${OUT_ROOT:-/tmp/usrp_spectrograms/all_detectors}"
EXTRA_ARGS="${EXTRA_ARGS:---no-tensors}"
REPACK="${REPACK:-0}"                    # --repack (or REPACK=1): pack masks .npy -> .packed.npz per run
_pos=()
for _a in "$@"; do case "$_a" in --repack) REPACK=1 ;; *) _pos+=("$_a") ;; esac; done
if [[ ${#_pos[@]} -gt 0 ]]; then set -- "${_pos[@]}"; else set --; fi
declare -A CFG=(
  [coherent_power]="old_configs/config_coherent_power_performance_single_channel.yaml"
  [cuda_dino]="config_cuda_dino_performance_single_channel.yaml"
  [finetuned_dino]="config_finetuned_dino_single_channel.yaml"
  [finetuned_dino_m2]="config_finetuned_dino_m2_single_channel.yaml"
  [yolo26s]="config_yolo26s_single_channel.yaml"
  [yolo26m]="config_yolo26m_single_channel.yaml"
)
CAPTURES_DIR="${CAPTURES_DIR:-${USER_HOME}/captures}"
ATTENS="${ATTENS:-0 5 10 15 20 25 30 30_v2 35 40 45 50 55 60}"   # notebook's SNR sweep (30_v2 averaged into 30 dB)
CLEANUP_STAGED="${CLEANUP_STAGED:-1}"   # rm each capture's staged copy after use so /tmp doesn't fill (set 0 to keep)
if [[ $# -gt 0 ]]; then
  CAPS=("$@")
else
  CAPS=(); for a in ${ATTENS}; do CAPS+=("${CAPTURES_DIR}/attenuation_dB_${a}.sigmf-data"); done
fi

# runnable detectors = those whose base config exists (operator wired)
RUN_DETS=()
for det in ${DETECTORS}; do
  if [[ -f "${APP_DIR}/${CFG[$det]:-__none__}" ]]; then RUN_DETS+=("$det")
  else echo "SKIP ${det} (config not found: ${CFG[$det]:-<unknown>} -- operator not wired yet)"; fi
done
[[ ${#RUN_DETS[@]} -gt 0 ]] || { echo "no runnable detectors -- nothing to do" >&2; exit 1; }

avail_gb="$(df -BG --output=avail /tmp/usrp_spectrograms 2>/dev/null | tail -1 | tr -dc '0-9')"
echo "detectors: ${RUN_DETS[*]}"
echo "captures:  ${#CAPS[@]} file(s) (SNR sweep)"
echo "out root:  ${OUT_ROOT}   | /tmp avail: ${avail_gb:-?} GB   | cleanup staged: ${CLEANUP_STAGED}"
[[ -n "${avail_gb}" && "${avail_gb}" -lt 40 ]] && echo "WARNING: < 40 GB free on /tmp -- staged inputs may fill it" >&2

# capture-outer: stage each capture ONCE, run all detectors on it, then free the staged copy.
# (Peak staging = one capture (~14-19 GB) instead of the whole ~196 GB sweep.)
for cap in "${CAPS[@]}"; do
  [[ -f "$cap" ]] || { echo "MISSING capture: $cap (skip)" >&2; continue; }
  stem="$(basename "$cap")"; stem="${stem%.sigmf-data}"
  for det in "${RUN_DETS[@]}"; do
    echo "=== ${det} on ${stem} ==="
    python3 "${APP_DIR}/run_cuda_dino_offline_file.py" "$cap" \
        --detector "${det}" --config "${APP_DIR}/${CFG[$det]}" \
        --output-root "${OUT_ROOT}/${det}/${stem}" ${EXTRA_ARGS}
    if [[ "${REPACK}" == "1" ]]; then
      python3 "${APP_DIR}/repack_offline_masks.py" "${OUT_ROOT}/${det}/${stem}" || echo "  repack warning (${det}/${stem})" >&2
    fi
  done
  if [[ "${CLEANUP_STAGED}" == "1" ]]; then
    echo "--- freeing staged input /tmp/usrp_spectrograms/offline_inputs/${stem} ---"
    sudo rm -rf "/tmp/usrp_spectrograms/offline_inputs/${stem}"
  fi
done
echo "DONE -> ${OUT_ROOT}/<detector>/<stem>/mask_arrays/"

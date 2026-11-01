#!/usr/bin/env bash
# Offline reference legs: the SAME detector configs the loopback legs use, run by the offline eval
# binary over the SigMF file (clean, deterministic ingest). Produces per-frame GT masks + offline
# masks under $OUT/<leg>/<capture_stem>/ for scoring the loopback masks (score_leg.py).
#   ./run_offline_refs.sh <capture.sigmf-data> [leg ...]
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP=$(cd "${HERE}/../../.." && pwd -P)
CAP=${1:?capture.sigmf-data}; shift
LEGS=("$@"); [[ ${#LEGS[@]} -eq 0 ]] && LEGS=(dinoft_rt_m5s coherent dinoft_native_m2dr dinoft_native_m5s dinoft_rt_m2dr)
OUT=${OUT:-/tmp/usrp_spectrograms/rt_stress/offline}
STEM=$(basename "${CAP}" .sigmf-data)
sudo mkdir -p "${OUT}" && sudo chmod 1777 "${OUT}"
det_for() { case "$1" in coherent*) echo coherent_power ;; dinoft*) echo cuda_dino_finetuned ;; *) echo "" ;; esac; }
for leg in "${LEGS[@]}"; do
  cfg="${HERE}/configs/config_lb_${leg}.yaml"
  out="${OUT}/${leg}/${STEM}"
  if [[ -f "${out}/frame_manifest.csv" ]]; then echo "==> [offline] ${leg}: already done (${out})"; continue; fi
  det=$(det_for "${leg}"); [[ -n "${det}" ]] || { echo "!! unknown leg ${leg}"; continue; }
  echo "==> [offline] ${leg} (${det}) -> ${out}   $(date +%T)"
  t0=$(date +%s)
  ( cd "${APP}" && sudo python3 run_cuda_dino_offline_file.py "${CAP}" --detector "${det}" \
      --config "${cfg}" --output-root "${out}" --no-tensors --progress-every 50 ) > "${OUT}/${leg}_offline.log" 2>&1
  rc=$?
  echo "    exit=${rc} wall=$(( $(date +%s) - t0 ))s  masks=$(ls "${out}/mask_arrays" 2>/dev/null | wc -l) gt=$(ls "${out}/gt_masks" 2>/dev/null | wc -l)"
  grep -E "Starting offline CUDA detector eval|frames processed|masks emitted|downsample: FFT|native: FFT|Offline eval complete|summary" "${OUT}/${leg}_offline.log" | tail -4 | sed 's/^/    /'
done
echo "==> offline refs done $(date +%T)"

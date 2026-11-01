#!/usr/bin/env bash
# Run a series of loopback legs back to back (GPU + NIC are exclusive, so legs never overlap), then
# collect metrics and -- when an offline reference for the leg exists -- score the emitted masks.
#   ./run_series.sh <rate_hz> <pps> <loops> <pcap> <capture.sigmf-meta> <suffix> <leg> [leg ...]
#   e.g. ./run_series.sh 245760000 240000 1 $PCAP $META 245 dinoft_rt_m5s coherent dinoft_native_m2dr
#        ./run_series.sh 491520000 480000 4 $PCAP $META 491 dinoft_rt_m5s coherent dinoft_native_m2dr
# Env: OUT, OFFLINE_ROOT (/tmp/usrp_spectrograms/rt_stress/offline), SCORE (1 = score vs offline),
#      FRAMES_PER_LOOP (set for multi-loop runs at the offline rate; default: no modulo).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
RATE_HZ=${1:?}; PPS=${2:?}; LOOPS=${3:?}; PCAP=${4:?}; META=${5:?}; SUFFIX=${6:?}; shift 6
LEGS=("$@")
OUT=${OUT:-/tmp/usrp_spectrograms/rt_stress}
OFFLINE_ROOT=${OFFLINE_ROOT:-/tmp/usrp_spectrograms/rt_stress/offline}
SCORE=${SCORE:-1}
STEM=$(basename "${META}" .sigmf-meta)
for leg in "${LEGS[@]}"; do
  run="${leg}_${SUFFIX}"
  echo; echo "######## ${run}  $(date +%T) ########"
  "${HERE}/run_leg.sh" "${run}" "${leg}" "${RATE_HZ}" "${PPS}" "${LOOPS}" "${PCAP}" || { echo "!! leg ${run} failed"; continue; }
  python3 "${HERE}/collect_metrics.py" "${OUT}/${run}"
  base_leg="${leg%%_nosnip*}"   # load-ablation variants (*_nosnip, *_nosnip_noviz) share the base detector's offline reference
  if [[ "${SCORE}" == "1" && -f "${OFFLINE_ROOT}/${base_leg}/${STEM}/frame_manifest.csv" ]]; then
    fpl_arg=(); [[ -n "${FRAMES_PER_LOOP:-}" ]] && fpl_arg=(--frames-per-loop "${FRAMES_PER_LOOP}")
    python3 "${HERE}/score_leg.py" "${OUT}/${run}" --offline "${OFFLINE_ROOT}/${base_leg}/${STEM}" --meta "${META}" --leg "${leg}" "${fpl_arg[@]}" | tail -20
  fi
  sleep 3
done
echo; echo "######## series ${SUFFIX} done $(date +%T) ########"

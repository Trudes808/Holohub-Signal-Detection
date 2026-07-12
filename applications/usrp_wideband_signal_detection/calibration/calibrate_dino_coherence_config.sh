#!/usr/bin/env bash
# STAGE 1 of the CUDA DINO coherence-gate calibration: measure the per-frequency noise
# power floor (dB) and coherence-gate threshold, for inspection before wiring Stage 2.
#
# Mirrors the coherent_power calibration flow, but for the DINO detector's OWN coherence
# gate (it does NOT change the DINO method):
#   1. cut annotation-free noise regions out of the captures (reused extractor)
#   2. run the CUDA DINO detector over the noise concat with save_coherence_stats: true
#      -> per-frame corrected_db (full frame) + packed coherence gate .npy
#   3. fit per-frequency power floor + gate threshold and write .npy + diagnostics
#
# Requires the container rebuilt after the operator stats-dump change:
#     ./rebuild_demo_container_app.sh
# Then run from the app dir (sudo so container exec + root-owned /tmp artifacts are readable):
#     sudo ./calibrate_dino_coherence_config.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# Run from the app root so generated_inputs/, config paths, and the offline driver resolve.
cd "${APP_DIR}"
# Pick up the container identity (CONTAINER_NAME) the offline driver targets.
source "${APP_DIR}/bash_scripts/container_env.sh"

# --- config (override via env) ---
INPUT_GLOBS="${INPUT_GLOBS:-generated_inputs/attenuation_dB_45_*.sigmf-data generated_inputs/attenuation_dB_50_*.sigmf-data generated_inputs/attenuation_dB_55_*.sigmf-data generated_inputs/attenuation_dB_60_*.sigmf-data}"
NOISE_DIR="${NOISE_DIR:-/tmp/usrp_spectrograms/calibration_noise}"
RUN_ROOT="${RUN_ROOT:-/tmp/usrp_spectrograms/offline_cuda_dino}"
# HOST view of the stats dump. The dump config writes to the CONTAINER path
# /workspace/spectrograms/dino_coherence_cal/run, bind-mounted to this host dir.
CAL_ROOT="${CAL_ROOT:-/tmp/usrp_spectrograms/dino_coherence_cal}"
STATS_RUN_DIR="${CAL_ROOT}/run"
BASE_CONFIG="${BASE_CONFIG:-config_cuda_dino_performance_single_channel.yaml}"
DUMP_CONFIG="${DUMP_CONFIG:-calibration/config_cuda_dino_coherence_calibration_dump_single_channel.yaml}"
POWER_FLOOR_FP="${POWER_FLOOR_FP:-0.02}"
GATE_FP="${GATE_FP:-0.02}"
POWER_FLOOR_OUT="${POWER_FLOOR_OUT:-calibration/dino_coherence_per_freq_power_floor.npy}"
GATE_THRESHOLD_OUT="${GATE_THRESHOLD_OUT:-calibration/dino_coherence_per_freq_gate_threshold.npy}"
OUTPUT_CONFIG="${OUTPUT_CONFIG:-calibration/config_cuda_dino_coherence_calibrated_single_channel.yaml}"
# Static coherence_band_threshold floor in the emitted config; per-row threshold is max'd with
# this. 0.0 lets the per-frequency values govern fully (incl. rows calibrated below the old 0.05).
COHERENCE_BAND_THRESHOLD="${COHERENCE_BAND_THRESHOLD:-0.0}"
FLOOR_OFFSET_DB="${FLOOR_OFFSET_DB:-0.0}"

NOISE_DATA="${NOISE_DIR}/dino_coherence_cal_noise.sigmf-data"
mkdir -p "${NOISE_DIR}" "${CAL_ROOT}"

echo "=== [1/4] extracting annotation-free noise regions ==="
# shellcheck disable=SC2086
python3 calibration/extract_noise_regions.py --inputs ${INPUT_GLOBS} --output "${NOISE_DATA}"

echo
echo "=== [2/4] generating dump config from ${BASE_CONFIG} ==="
python3 - "${BASE_CONFIG}" "${DUMP_CONFIG}" <<'PY'
import re, sys
base, out = sys.argv[1], sys.argv[2]
text = open(base).read()
inject = ('  save_coherence_stats: true\n'
          '  coherence_stats_dir: "/workspace/spectrograms/dino_coherence_cal/run"\n')
new, n = re.subn(r'(^cuda_dino_detector:\n)', r'\1' + inject, text, count=1, flags=re.M)
if n != 1:
    raise SystemExit("could not find 'cuda_dino_detector:' block header in base config")
open(out, 'w').write(new)
print(f"  wrote {out} (save_coherence_stats + coherence_stats_dir injected)")
PY

echo
echo "=== [3/4] running CUDA DINO over noise (coherence-stats dump) ==="
rm -rf "${STATS_RUN_DIR}"
mkdir -p "${STATS_RUN_DIR}"
python3 run_cuda_dino_offline_file.py "${NOISE_DATA}" --detector cuda_dino \
    --config "${DUMP_CONFIG}" \
    --output-root "${RUN_ROOT}/dino_coherence_noise" --progress-every 50
rm -rf "${CAL_ROOT}/noise"
mv "${STATS_RUN_DIR}" "${CAL_ROOT}/noise"
if [[ ! -f "${CAL_ROOT}/noise/meta.json" ]]; then
    echo "ERROR: no coherence stats written (${CAL_ROOT}/noise is empty)." >&2
    echo "       The DINO binary ignored save_coherence_stats — rebuild first:" >&2
    echo "         ./rebuild_demo_container_app.sh   (or FORCE_REBUILD=1 ./rebuild_demo_container_app.sh)" >&2
    exit 1
fi
echo "  stats -> ${CAL_ROOT}/noise ($(ls "${CAL_ROOT}/noise"/coherence_stats_*_coherence_gate.npy 2>/dev/null | wc -l) frame dumps)"

echo
echo "=== [4/4] fitting per-frequency floor + gate threshold ==="
python3 calibration/calibrate_dino_coherence_config.py \
    --stats-run-dir "${CAL_ROOT}/noise" \
    --power-floor-fp "${POWER_FLOOR_FP}" \
    --gate-fp "${GATE_FP}" \
    --power-floor-out "${POWER_FLOOR_OUT}" \
    --gate-threshold-out "${GATE_THRESHOLD_OUT}" \
    --base-config "${BASE_CONFIG}" \
    --output-config "${OUTPUT_CONFIG}" \
    --coherence-band-threshold "${COHERENCE_BAND_THRESHOLD}" \
    --floor-offset-db "${FLOOR_OFFSET_DB}" \
    --plots-dir calibration/diagnostics

echo
echo "DONE."
echo "  power floor      : ${POWER_FLOOR_OUT}"
echo "  gate threshold   : ${GATE_THRESHOLD_OUT}"
echo "  calibrated config: ${OUTPUT_CONFIG}  (per-frequency gate enabled)"
echo "  diagnostics      : calibration/diagnostics/dino_coherence_per_freq.png"
echo "  stats            : ${CAL_ROOT}/noise"
echo
echo "Run it through the eval with --config ${OUTPUT_CONFIG} and compare against the base."

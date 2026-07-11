#!/usr/bin/env bash
# Build the DINO positional (RoPE) noise template from GUARANTEED-noise regions:
# it cuts the annotation-free sample spans out of the low-SNR captures, stitches them
# into one noise-only capture, runs that through the detector (template disabled, prenorm
# dumps on -> masks are also saved under the run dir), then fits the template.
#
# Run from the app dir. Intended to be run with sudo so the container exec and the
# root-owned /tmp artifacts are all readable in a single pass:
#     sudo ./calibration_script.sh
#
# Override defaults via env, e.g. to widen the input set (noise floor is attenuation-
# independent, so more files = more frames = a more stable template):
#     sudo INPUT_GLOBS="generated_inputs/attenuation_dB_*_*.sigmf-data" ./calibration_script.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"

# --- config (override via env) ---
INPUT_GLOBS="${INPUT_GLOBS:-generated_inputs/attenuation_dB_45_*.sigmf-data generated_inputs/attenuation_dB_50_*.sigmf-data generated_inputs/attenuation_dB_55_*.sigmf-data generated_inputs/attenuation_dB_60_*.sigmf-data}"
NOISE_DIR="${NOISE_DIR:-/tmp/usrp_spectrograms/calibration_noise}"
RUN_ROOT="${RUN_ROOT:-/tmp/usrp_spectrograms/offline_cuda_dino}"
RUN_NAME="${RUN_NAME:-cal_noise_concat}"
CAL_CONFIG="${CAL_CONFIG:-config_cuda_dino_calibration_dump_single_channel.yaml}"
TEMPLATE_OUT="${TEMPLATE_OUT:-calibration/dino_vitb16_noise_sigma_64x64.npy}"
REDUCE="${REDUCE:-median}"          # median | quantile | mean
QUANTILE="${QUANTILE:-0.30}"        # used when REDUCE=quantile
GUARD="${GUARD:-40960}"             # shrink each gap by this many samples per side
MIN_GAP="${MIN_GAP:-262144}"        # discard noise gaps shorter than this (after guard)
MIN_SAMPLES="${MIN_SAMPLES:-4}"     # calibrate hard-floor on total frame*chunk samples

mkdir -p "${NOISE_DIR}"
NOISE_DATA="${NOISE_DIR}/${RUN_NAME}.sigmf-data"

echo "=== [1/3] extracting annotation-free noise regions ==="
# shellcheck disable=SC2086
python3 extract_noise_regions.py --inputs ${INPUT_GLOBS} \
    --output "${NOISE_DATA}" --guard-samples "${GUARD}" --min-gap-samples "${MIN_GAP}"

echo
echo "=== [2/3] running the detector on the noise capture (prenorm dumps + masks) ==="
python3 run_cuda_dino_offline_file.py "${NOISE_DATA}" --detector cuda_dino \
    --config "${CAL_CONFIG}" \
    --output-root "${RUN_ROOT}/${RUN_NAME}" --progress-every 50

echo
echo "=== [3/3] fitting the positional template ==="
python3 calibrate_dino_positional_template.py \
    --run-dir "${RUN_ROOT}/${RUN_NAME}" \
    --output "${TEMPLATE_OUT}" \
    --expect-deweight 0.75 --reduce "${REDUCE}" --quantile "${QUANTILE}" \
    --min-samples "${MIN_SAMPLES}" \
    --plots-dir calibration/diagnostics

echo
echo "DONE."
echo "  template : ${TEMPLATE_OUT}"
echo "  masks    : ${RUN_ROOT}/${RUN_NAME}/mask_arrays/  (should be ~empty on noise)"
echo "  diagnostics: calibration/diagnostics/{template_heatmap,ring_profile}.png"
echo
echo "Enable it by setting, in config_cuda_dino_performance_single_channel.yaml:"
echo "  raw_dino_positional_template_path: \"/workspace/holohub/applications/usrp_wideband_signal_detection/${TEMPLATE_OUT}\""
echo
echo "If it reported too few samples, widen the input set (noise is attenuation-independent):"
echo "  sudo INPUT_GLOBS=\"generated_inputs/attenuation_dB_*_*.sigmf-data\" ./calibration_script.sh"

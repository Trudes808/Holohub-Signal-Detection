#!/usr/bin/env bash
# Calibrate the coherent-power fast-path thresholds from measured NOISE (and SIGNAL)
# statistics, and write a calibrated config.
#
# Mirrors calibration_script.sh (the DINO positional template flow):
#   1. cut annotation-free noise regions + annotated signal regions out of the captures
#   2. run the coherent-power detector over each concat with the fast-path stats dump on
#      (save_coherent_power_stats: true) -> per-frame corrected_db + local background .npy
#   3. fit fast_power_floor_db / fast_power_span_db / fast_score_threshold from those
#      distributions and emit config_coherent_power_calibrated_single_channel.yaml
#
# Requires the container to be rebuilt after the operator stats-dump change:
#     ./rebuild_demo_container_app.sh
# Then run from the app dir (sudo so container exec + root-owned /tmp artifacts are all
# readable in one pass):
#     sudo ./calibrate_coherent_power_config.sh
#
# Override the input set via env (noise floor is attenuation-independent; signal level
# should come from mid/low-SNR captures that actually carry annotations):
#     sudo INPUT_GLOBS="generated_inputs/attenuation_dB_*_*.sigmf-data" ./calibrate_coherent_power_config.sh
#
# ---------------------------------------------------------------------------------------
# Per-frequency floor mode: calibrated vs dynamic vs static
# ---------------------------------------------------------------------------------------
# The per-frequency fill (OR-ed into the fast mask; fires where corrected_db exceeds a
# per-row noise floor by per_freq_threshold_offset_db) can source its floor three ways,
# selected in the config's `coherent_power_signal_detector` block via
# `per_freq_threshold_mode`. This offline script produces the CALIBRATED floor; the other
# two modes need no calibration run.
#
#   per_freq_threshold_mode: "calibrated"   (what this script targets)
#     Loads the static per-row floor .npy this script writes
#     (calibration/coherent_power_per_freq_floor.npy, referenced by per_freq_threshold_path).
#     Best when the noise environment matches the capture set you calibrated against.
#     Recalibrate (re-run this script) when the front end, gain, or band changes.
#
#   per_freq_threshold_mode: "dynamic"       (no calibration run; learns live)
#     Learns the per-row floor online from the running stream: each bin starts at a high
#     bar (dynamic_floor_init_db) and only descends toward the quietest power it observes,
#     so it self-calibrates to the current noise floor without a capture set. An
#     always-on signal never presents a quiet frame, so its bin stays high and is ignored
#     by design. Re-seeds to the high bar on app reset and on a center-frequency change.
#     Tuning knobs (config block):
#       dynamic_floor_init_db      high starting bar per bin, in dB (default 40).
#       dynamic_floor_std_k        per-frame per-row statistic = mean + k*std of
#                                  corrected_db; k approximates the noise high-quantile this
#                                  script measures. Raise to reduce false positives (default 2.0).
#       dynamic_floor_window_slots number of sub-window minima kept per bin. The floor is the
#                                  min across all slots; stale lows age out once every slot
#                                  rotates, which bounds the slow downward creep a pure global
#                                  minimum would accumulate over a long run (default 8).
#       dynamic_floor_slot_frames  frames each slot accumulates before the cursor rotates. The
#                                  effective sliding window = window_slots * slot_frames frames
#                                  (default 16 -> 8*16 = 128 frames). Shorter window = more
#                                  responsive + less creep but a noisier floor; longer = smoother
#                                  and closer to the calibrated floor but slower to adapt.
#       dynamic_floor_warmup_frames frames to accumulate before the learned floor is allowed to
#                                  feed the fill (default 0; the high init bar already keeps early
#                                  frames conservative).
#     per_freq_threshold_offset_db still sets the firing margin above the floor, same as in
#     calibrated mode, and also absorbs any residual bias in the learned floor.
#     Ready-made config: config_coherent_power_perf_dynamic_single_channel.yaml.
#
#   per_freq_threshold_mode: "static"        (per-frequency fill disabled)
#     No per-row floor; only the global fast_power_floor_db / fast_score_threshold path runs.
#
#   (empty)  -> legacy behavior: derives from per_freq_threshold_enable
#               (true -> calibrated, false -> static).
# ---------------------------------------------------------------------------------------
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
SIGNAL_DIR="${SIGNAL_DIR:-/tmp/usrp_spectrograms/calibration_signal}"
RUN_ROOT="${RUN_ROOT:-/tmp/usrp_spectrograms/offline_coherent_power}"
# HOST view of the stats dumps. The dump config writes to the CONTAINER path
# /workspace/spectrograms/coherent_power_cal/run, which is bind-mounted to
# host /tmp/usrp_spectrograms/coherent_power_cal/run (= STATS_RUN_DIR here).
CAL_ROOT="${CAL_ROOT:-/tmp/usrp_spectrograms/coherent_power_cal}"
STATS_RUN_DIR="${CAL_ROOT}/run"
DUMP_CONFIG="${DUMP_CONFIG:-calibration/config_coherent_power_calibration_dump_single_channel.yaml}"
BASE_CONFIG="${BASE_CONFIG:-old_configs/config_coherent_power_performance_single_channel.yaml}"
OUTPUT_CONFIG="${OUTPUT_CONFIG:-calibration/config_coherent_power_calibrated_single_channel.yaml}"

# Calibration policy (passed through to calibrate_coherent_power_config.py).
FP_RATE="${FP_RATE:-1e-3}"
FLOOR_Q="${FLOOR_Q:-0.90}"
SIGNAL_Q="${SIGNAL_Q:-0.90}"
SIGNAL_SUPPORT_TARGET="${SIGNAL_SUPPORT_TARGET:-0.90}"
THRESHOLD_MARGIN_DB="${THRESHOLD_MARGIN_DB:-0.0}"
# Span policy: which fast_power_span_db to WRITE (both candidates are always printed).
# 'signal' sizes span from the measured signal level; 'fixed' uses a fixed operating zone.
# Masks are identical either way (threshold preserves the FP-calibrated boundary T); span
# only reshapes the soft score that feeds grouping.
SPAN_MODE="${SPAN_MODE:-signal}"
OPERATING_RANGE_DB="${OPERATING_RANGE_DB:-40.0}"
# Per-frequency noise-floor fill: floor at PER_FREQ_FP false-positive rate, fires where
# corrected_db > floor[row] + PER_FREQ_OFFSET_DB. Fills in signal interiors the local box
# hollows out. OR-ed into the box mask; does not touch live_emit_always_on_*.
PER_FREQ_FP="${PER_FREQ_FP:-0.02}"
PER_FREQ_OFFSET_DB="${PER_FREQ_OFFSET_DB:-2.0}"

NOISE_DATA="${NOISE_DIR}/coherent_cal_noise.sigmf-data"
SIGNAL_DATA="${SIGNAL_DIR}/coherent_cal_signal.sigmf-data"
mkdir -p "${NOISE_DIR}" "${SIGNAL_DIR}" "${CAL_ROOT}"

run_detector() {  # $1 = input .sigmf-data, $2 = dest stats dir name
    local input="$1" dest="$2"
    rm -rf "${STATS_RUN_DIR}"
    mkdir -p "${STATS_RUN_DIR}"
    python3 run_cuda_dino_offline_file.py "${input}" --detector coherent_power \
        --config "${DUMP_CONFIG}" \
        --output-root "${RUN_ROOT}/${dest}" --progress-every 50
    rm -rf "${CAL_ROOT}/${dest}"
    mv "${STATS_RUN_DIR}" "${CAL_ROOT}/${dest}"
    if [[ ! -f "${CAL_ROOT}/${dest}/meta.json" ]]; then
        echo "ERROR: no stats were written for the '${dest}' run (${CAL_ROOT}/${dest} is empty)." >&2
        echo "       The detector binary ignored save_coherent_power_stats — the coherent_power" >&2
        echo "       operator was not recompiled with the stats-dump change. Rebuild first:" >&2
        echo "         ./rebuild_demo_container_app.sh   (or FORCE_REBUILD=1 ./rebuild_demo_container_app.sh)" >&2
        echo "       then re-run this script." >&2
        exit 1
    fi
    echo "  stats -> ${CAL_ROOT}/${dest} ($(ls "${CAL_ROOT}/${dest}"/coherent_power_stats_*_corrected_sxx_db.npy 2>/dev/null | wc -l) frame dumps)"
}

echo "=== [1/4] extracting annotation-free noise regions ==="
# shellcheck disable=SC2086
python3 calibration/extract_noise_regions.py --inputs ${INPUT_GLOBS} --output "${NOISE_DATA}"

echo
echo "=== [2/4] extracting annotated signal regions ==="
# shellcheck disable=SC2086
python3 calibration/extract_signal_regions.py --inputs ${INPUT_GLOBS} --output "${SIGNAL_DATA}"

echo
echo "=== [3/4] running detector on noise + signal (fast-path stats dump) ==="
run_detector "${NOISE_DATA}" noise
run_detector "${SIGNAL_DATA}" signal

echo
echo "=== [4/4] fitting calibrated fast-path thresholds ==="
python3 calibration/calibrate_coherent_power_config.py \
    --noise-run-dir "${CAL_ROOT}/noise" \
    --signal-run-dir "${CAL_ROOT}/signal" \
    --base-config "${BASE_CONFIG}" \
    --output-config "${OUTPUT_CONFIG}" \
    --noise-false-positive-rate "${FP_RATE}" \
    --floor-quantile "${FLOOR_Q}" \
    --signal-quantile "${SIGNAL_Q}" \
    --signal-support-target "${SIGNAL_SUPPORT_TARGET}" \
    --threshold-margin-db "${THRESHOLD_MARGIN_DB}" \
    --span-mode "${SPAN_MODE}" \
    --operating-range-db "${OPERATING_RANGE_DB}" \
    --per-freq-fp "${PER_FREQ_FP}" \
    --per-freq-offset-db "${PER_FREQ_OFFSET_DB}"

echo
echo "DONE."
echo "  calibrated config : ${OUTPUT_CONFIG}"
echo "  sidecar           : ${OUTPUT_CONFIG%.yaml}.calibration.json"
echo "  noise stats       : ${CAL_ROOT}/noise"
echo "  signal stats      : ${CAL_ROOT}/signal"
echo
echo "Validate before adopting: run the batch eval with --config ${OUTPUT_CONFIG} and compare"
echo "P/R/F1 + the noise-only false-positive fraction against the base config."

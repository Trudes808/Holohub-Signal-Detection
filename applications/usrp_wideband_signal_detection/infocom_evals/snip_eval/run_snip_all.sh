#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Run the REAL mask_replay -> signal_snipper over every (detector, capture) under a batch root, for
# one or more snipper MODES, in a single invocation. Footprint-only by default: the config sets
# sigmf_file_sink.write_iq=false, so only tiny .sigmf-meta files are kept (each carries sample_count
# / rate => exact bytes); the .sigmf-data IQ is written then deleted per snippet (peak ~1 snippet).
# Speckle is gated by signal_snipper.min_box_pixels (256) so at most a few snippets/frame.
#
# Per capture: materialize .packed.npz masks -> .npy ONCE, run the snip for each MODE (reading raw IQ
# in place from the /workspace/captures mount -- no 14 GB copy), write our Python detections meta
# once, then delete the transient .npy. Needs the container with mask_replay compiled + ~/captures.
#
# Usage (BOTH modes, one command):
#   sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 BATCH_ROOT=/tmp/ds_batch \
#        DETECTORS="ground_truth" CAPTURES_DIR=/home/bqn82/captures ./run_snip_all.sh
# Env: MODES (default "frequency time_only"), BATCH_ROOT, SNIP_OUT, CAPTURES_DIR, DETECTORS, CONFIG, PYTHON.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
BATCH_ROOT="${BATCH_ROOT:-/tmp/usrp_spectrograms/all_detectors}"
SNIP_OUT="${SNIP_OUT:-/tmp/usrp_spectrograms/snipped}"
CAPTURES_DIR="${CAPTURES_DIR:-/home/bqn82/captures}"
CONFIG="${CONFIG:-${APP_DIR}/config_mask_replay_snip_single_channel.yaml}"
MODES="${MODES:-frequency time_only}"
PYTHON="${PYTHON:-python3}"
export HOST_CAPTURES_ROOT="${CAPTURES_DIR}"

# Build a config per mode. The base config is signal_snipper.mode: "frequency"; for any other mode we
# sed ONLY that unique line (sigmf_file_sink.mode is "per_signal", so there is no collision).
mkdir -p "${SNIP_OUT}/_mode_configs"
declare -A MODE_CFG
for MODE in ${MODES}; do
  if [[ "${MODE}" == "frequency" ]]; then
    MODE_CFG[$MODE]="${CONFIG}"
  else
    cfg="${SNIP_OUT}/_mode_configs/$(basename "${CONFIG%.yaml}")_${MODE}.yaml"
    sed 's/^  mode: "frequency"/  mode: "'"${MODE}"'"/' "${CONFIG}" > "${cfg}"
    MODE_CFG[$MODE]="${cfg}"
  fi
done

if [[ -n "${DETECTORS:-}" ]]; then DETS=(${DETECTORS}); else
  DETS=(); for d in "${BATCH_ROOT}"/*/; do DETS+=("$(basename "$d")"); done
fi
echo "batch_root=${BATCH_ROOT}  snip_out=${SNIP_OUT}  captures=${CAPTURES_DIR}"
echo "modes: ${MODES}   detectors: ${DETS[*]}"
for det in "${DETS[@]}"; do
  for capdir in "${BATCH_ROOT}/${det}"/*/; do
    [[ -d "${capdir}mask_arrays" ]] || continue
    stem="$(basename "${capdir%/}")"
    cap="${CAPTURES_DIR}/${stem}.sigmf-data"
    if [[ ! -f "$cap" ]]; then echo "  skip ${det}/${stem} (no capture ${cap} -- stale dir?)"; continue; fi
    "${PYTHON}" "${SCRIPT_DIR}/materialize_npy.py" "${BATCH_ROOT}/${det}/${stem}"      # .packed.npz -> .npy (once)
    for MODE in ${MODES}; do
      out="${SNIP_OUT}/${MODE}/${det}/${stem}"
      # Resumable: a (mode,detector,capture) is DONE once its eval completes -- INCLUDING a
      # zero-snippet result (legitimate when a size/SNR threshold leaves nothing to snip). Completion
      # is recorded with a .snip_complete marker so zero-result captures are not re-run every pass.
      # Fallback: runs predating the marker are still recognized as done if they have snippet metas.
      if [ -f "${out}/.snip_complete" ] || compgen -G "${out}/snippets/*.sigmf-meta" > /dev/null 2>&1; then
        echo "  skip [${MODE}] ${det}/${stem} (already done)"; continue
      fi
      echo "=== snip [${MODE}] ${det}/${stem} ==="
      # --snippets-only => the binary writes ONLY the snippet metas; no mask_arrays/gt_masks/previews
      # are written in the first place (needs the rebuilt container).
      # Tolerant: a single failed run logs + continues (does NOT abort the whole batch); re-run to retry.
      if ! python3 "${APP_DIR}/run_cuda_dino_offline_file.py" "$cap" \
          --detector mask_replay --config "${MODE_CFG[$MODE]}" \
          --mask-dir "${BATCH_ROOT}/${det}/${stem}/mask_arrays" \
          --output-root "${out}" \
          --captures-mounted --no-tensors --snippets-only; then
        echo "  FAILED [${MODE}] ${det}/${stem} — skipping (re-run resumes it)"
        rm -rf "${out}/snippets" 2>/dev/null || true   # drop partial so the resume-check re-attempts it
        rm -f "${out}/.snip_complete" 2>/dev/null || true
        continue
      fi
      mkdir -p "${out}" && touch "${out}/.snip_complete"   # mark done (0 or more snippets) -> won't re-run
    done
    # Our Python detections-annotation meta (mode-independent), for cross-check / labelling.
    "${PYTHON}" "${SCRIPT_DIR}/snip_annotations.py" --run-dir "${BATCH_ROOT}/${det}/${stem}" \
        --captures-dir "${CAPTURES_DIR}" --out-dir "${SNIP_OUT}/annotations/${det}/${stem}_snipped" || true
    rm -f "${BATCH_ROOT}/${det}/${stem}/mask_arrays/"*.npy                             # drop transient masks
    echo "  -> ${SNIP_OUT}/<mode>/${det}/${stem}/snippets/*.sigmf-meta"
  done
done
echo "DONE. metas at ${SNIP_OUT}/<mode>/<detector>/<stem>/snippets/  ->  run verify_snip.py"

#!/usr/bin/env bash
# BER sweep driver for ONE detector across attenuation levels.
#   run_ber_sweep.sh <ground_truth|coherent_power|finetuned_dino_m2>
#
# Snippet generation (coherent/dino) is SERIAL (one container/GPU job at a time);
# the MATLAB BER evals run in a bounded PARALLEL pool (MAXEVAL) because levels are
# independent and eval is the CPU bottleneck. Each eval writes only its own
# per-level result files (race-free) and then deletes that level's bulky /tmp
# snippets. ground_truth reads captures directly (no gen). Everything under /tmp.
#
# Env knobs: LEVELS, MAXEVAL (default 5), BER_THREADS (default 6), CONTAINER_NAME,
#   SNIP_EXTRA  extra soft_label_pipeline flags, e.g. the 75 kHz + 1 ms snipper gate:
#               SNIP_EXTRA="--min-mask-bandwidth-hz 75000 --min-bandwidth-hz 75000 --min-duration-s 0.001"
#   RESULTS_DIR results folder for a variant sweep (default results/)
#   TAG         suffix for the /tmp snippet root so a variant never collides
set -uo pipefail

DET="${1:?usage: run_ber_sweep.sh <ground_truth|coherent_power|finetuned_dino_m2>}"
BER="/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/ber_eval"
PIPE="/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/soft_label_pipeline"
CAPS="/home/bqn82/captures"
ALLDET="/tmp/usrp_spectrograms/all_detectors"
LEVELS="${LEVELS:-5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80}"
MAXEVAL="${MAXEVAL:-5}"
export BER_THREADS="${BER_THREADS:-6}"
export CONTAINER_NAME="${CONTAINER_NAME:-usrp_x410_sig_det_bqn82}"
SNIP_EXTRA="${SNIP_EXTRA:-}"
RESULTS_DIR="${RESULTS_DIR:-$BER/results}"
TAG="${TAG:-}"

case "$DET" in
    coherent_power)    OUTROOT="/tmp/usrp_spectrograms/ber_eval/coherent_power${TAG}" ;;
    finetuned_dino_m2) OUTROOT="/tmp/usrp_spectrograms/ber_eval/finetuned_dino_m2${TAG}" ;;
    ground_truth)      OUTROOT="" ;;
    *) echo "bad detector '$DET'"; exit 2 ;;
esac
mkdir -p "$RESULTS_DIR"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate dinov3
mkdir -p "$BER/results"

running_evals() { jobs -rp | wc -l; }

eval_one() {   # $1=stem  (run backgrounded)
    local stem="$1"
    local sroot=""
    [ -n "$OUTROOT" ] && sroot="${OUTROOT}/iq/${stem}/snippets"
    ( cd "$BER" && matlab -batch "ber_sweep_one('${stem}','${DET}','${RESULTS_DIR}','${sroot}')" ) \
        >> "$RESULTS_DIR/eval_${DET}_${stem}.log" 2>&1
    if [ -n "$OUTROOT" ]; then
        rm -rf "${OUTROOT}/iq/${stem}" \
               "${OUTROOT}/masks/coherent_power/${stem}" \
               "${OUTROOT}/masks/cuda_dino/${stem}" \
               "${OUTROOT}/masks/dino_finetuned/${stem}"
    fi
    echo "[$(date -u +%TZ)] EVAL DONE ${DET} ${stem}  (/tmp free $(df -h --output=avail /tmp | tail -1 | tr -d ' '))"
}

gen_coherent() {  # $1=stem -> returns rc
    local stem="$1" maskarg=()
    [ -d "${ALLDET}/coherent_power/${stem}/mask_arrays" ] && maskarg=(--mask-root "$ALLDET")
    ( cd "$PIPE" && python soft_label_pipeline.py --waveform-dir "$CAPS" --glob "${stem}.sigmf-data" \
        --detector coherent_power --outputs iq --mode frequency --min-box-pixels 256 "${maskarg[@]}" \
        $SNIP_EXTRA --output-root "$OUTROOT" )
}

gen_dino() {  # $1=stem -> returns rc
    local stem="$1"
    local src="${ALLDET}/cuda_dino/${stem}" dst="${OUTROOT}/masks/cuda_dino/${stem}"
    if [ -f "${src}/frame_manifest.csv" ] && ls "${src}/mask_arrays/"* >/dev/null 2>&1; then
        mkdir -p "$dst"; cp -rf "${src}/." "${dst}/"        # reuse foundation -> skip a GPU pass
    fi
    ( cd "$PIPE" && python soft_label_pipeline.py --waveform-dir "$CAPS" --glob "${stem}.sigmf-data" \
        --detector dino_finetuned --outputs iq --mode frequency --min-box-pixels 256 \
        $SNIP_EXTRA --output-root "$OUTROOT" )
}

echo "=== sweep ${DET} start $(date -u +%FT%TZ)  levels: ${LEVELS}  MAXEVAL=${MAXEVAL} threads=${BER_THREADS} ==="
echo "    results -> ${RESULTS_DIR}"
[ -n "$SNIP_EXTRA" ] && echo "    snipper gate: ${SNIP_EXTRA}"
[ -n "$OUTROOT" ] && echo "    snippets -> ${OUTROOT}"
for L in $LEVELS; do
    stem="attenuation_dB_${L}"
    [ -f "${CAPS}/${stem}.sigmf-data" ] || { echo "!! missing capture ${stem} -> skip"; continue; }

    # Idempotent resume: if this level's snippets are already on disk (a previous
    # run generated them but its eval was interrupted), skip straight to the eval
    # instead of redoing the container/GPU pass.
    have=0
    if [ -n "$OUTROOT" ] && [ -f "${OUTROOT}/iq/${stem}/frame_manifest.csv" ] && \
       [ "$(ls "${OUTROOT}/iq/${stem}/snippets/"*.sigmf-meta 2>/dev/null | wc -l)" -gt 0 ]; then
        have=1
        echo "[$(date -u +%TZ)] GEN skipped for ${stem} (snippets already present)"
    fi
    if [ $have -eq 0 ]; then
      case "$DET" in
        coherent_power)
            echo "[$(date -u +%TZ)] GEN coherent ${stem}"
            if ! gen_coherent "$stem"; then echo "!! GEN FAILED coherent ${stem} -> skip"; continue; fi ;;
        finetuned_dino_m2)
            echo "[$(date -u +%TZ)] GEN dino ${stem}"
            if ! gen_dino "$stem"; then echo "!! GEN FAILED dino ${stem} -> skip"; continue; fi ;;
      esac
    fi
    if [ -n "$OUTROOT" ]; then
        n=$(ls "${OUTROOT}/iq/${stem}/snippets/"*.sigmf-meta 2>/dev/null | wc -l)
        echo "[$(date -u +%TZ)] ${stem}: ${n} snippets generated"
        # A selective snipper gate can legitimately emit NOTHING (e.g. the 75 kHz +
        # 1 ms gate at deep noise, where every detection was a clock-spur artifact).
        # The snipper then never creates snippets/, which the harness would read as
        # "pipeline never ran" and throw. Gen succeeded here, so materialize the empty
        # dir to assert "ran, saved nothing" -> scored as all-miss instead of skipped.
        if [ "$n" -eq 0 ]; then
            mkdir -p "${OUTROOT}/iq/${stem}/snippets"
            echo "         (zero detections -> empty snippets/ created; scores as all-miss)"
        fi
    fi

    while [ "$(running_evals)" -ge "$MAXEVAL" ]; do sleep 5; done
    echo "[$(date -u +%TZ)] DISPATCH eval ${stem} (pool $(running_evals)/${MAXEVAL})"
    eval_one "$stem" &
done
wait
echo "=== sweep ${DET} done $(date -u +%FT%TZ) ==="

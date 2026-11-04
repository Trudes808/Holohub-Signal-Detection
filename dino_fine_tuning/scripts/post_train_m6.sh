#!/usr/bin/env bash
# Everything after train.py finishes: evaluations (M6-S + M5-S baseline), legacy held-out, export, qualitative
# overlays on the unlabelled recordings, comparison tables for the experiment record. Resumable; waits for the
# trainer's DONE marker first.
set -uo pipefail
FT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${PY:-/home/genesys-dgx1/Documents/holoscan_waveform_generation/.venv-ml/bin/python}
NAME=${NAME:-M6S_multirate}
SKIND=${SKIND:-scenes}
COMPARE=${COMPARE:-M5S}   # baseline names for compare_evals / overlay copies
EXP=$FT/../applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/dino_ft_multirate_scenes
export PYTHONPATH="$FT/src:${PYTHONPATH:-}"
until [ -f "$FT/checkpoints/$NAME/DONE" ]; do sleep 30; done
echo "[post] training done: $(cat $FT/checkpoints/$NAME/DONE | tr '\n' ' ')"
cd "$FT"
"$FT/scripts/run_m6.sh" eval
"$FT/scripts/run_m6.sh" export
[ -f "eval_out/unlabelled_$NAME/summary.md" ] || $PY src/eval_scenes.py --model "checkpoints/$NAME/best.pt" --name "$NAME" \
    --sigmf data/real_unlabelled --real --out "eval_out/unlabelled_$NAME" --max-frames 4 --png-per-scenario 8
$PY src/compare_evals.py --names $COMPARE "$NAME" --scenes-kind "$SKIND" --out "$EXP/results/comparison_${NAME}.md"
mkdir -p "$EXP/results/overlays"
for n in $COMPARE "$NAME"; do
  for f in eval_out/hard_$n/overlays/*.png eval_out/unlabelled_$n/overlays/*.png; do
    [ -f "$f" ] && cp "$f" "$EXP/results/overlays/${n}_$(basename "$f")"
  done
  for sc in cellular_adjacent drone_ism band_edge_focus hoppers_focus ism_collisions; do
    f=$(ls eval_out/${SKIND}_$n/overlays/${sc}_*.png 2>/dev/null | head -1); [ -n "$f" ] && cp "$f" "$EXP/results/overlays/${n}_$(basename "$f")"
  done
done
cp checkpoints/$NAME/history.json "$EXP/results/${NAME}_history.json"
cp eval_out/${SKIND}_$NAME/summary.md "$EXP/results/${SKIND}_${NAME}_summary.md"
[ -f checkpoints/$NAME/heldout_eval/heldout_summary.json ] && cp checkpoints/$NAME/heldout_eval/heldout_summary.json "$EXP/results/${NAME}_legacy491_heldout.json"
echo "[post] ALL DONE"

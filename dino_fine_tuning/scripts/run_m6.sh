#!/usr/bin/env bash
# M6-S multi-rate / multi-scenario retrain of the deployed DINO-FT detector (successor of M5-S).
#
#   1. combine  : data/dataset_scenes (gen_scenes.py synthetic, 5-500 MSps) + data/dataset_ota (real X410 OTA,
#                 coherent-teacher labels + antenna-off negatives)        -> data/dataset_m6
#   2. train    : ViT-S/16 ft_lastN, the M5-S recipe (configs/train.yaml) -> checkpoints/M6S_multirate/best.pt
#   3. evaluate : held-out scenes (new), the legacy 491.52 held-out (eval_heldout.py), the real held-out
#                 (ARGUS LTE OOD + hard_test_cases), each also for M5-S as the baseline   -> eval_out/
#   4. export   : TorchScript for the live operator                        -> weights/finetuned_dino_m6s_multirate_bf16.ts
#
# Usage (from anywhere): scripts/run_m6.sh [combine|train|eval|export|all]   (default all; every step resumable)
set -euo pipefail
FT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${PY:-/home/genesys-dgx1/Documents/holoscan_waveform_generation/.venv-ml/bin/python}
export PYTHONPATH="$FT/src:${PYTHONPATH:-}"
NAME=${NAME:-M6S_multirate}
DATASET=${DATASET:-$FT/data/dataset_m6}
SCENES=${SCENES:-$FT/data/dataset_scenes}      # synthetic scene dataset (train/val tiles + heldout_sigmf)
SKIND=${SKIND:-scenes}                         # eval_out/<SKIND>_<model> for the held-out scene eval
BASELINES=${BASELINES:-M5S:$FT/checkpoints/M5S_491_ota/best.pt}   # extra name:ckpt pairs to score on the same sets
VITS=${VITS:-$FT/../../dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}
BASE_CKPT=$FT/checkpoints/M5S_491_ota/best.pt
step=${1:-all}

combine() {
  [ -f "$DATASET/dataset_meta.json" ] && { echo "[combine] exists: $DATASET"; return; }
  $PY "$FT/src/combine_datasets.py" --inputs "$SCENES" "$FT/data/dataset_ota" --out "$DATASET"
}
train() {
  [ -f "$FT/checkpoints/$NAME/DONE" ] && { echo "[train] done: $FT/checkpoints/$NAME"; return; }
  cd "$FT" && $PY src/train.py --config configs/train.yaml --dataset "$DATASET" --mode ft_lastN \
      --backbone vits16 --weights-path "$VITS" --name "$NAME" --out "checkpoints/$NAME" --resume
}
evaluate() {
  cd "$FT"
  for tag in "$NAME:$FT/checkpoints/$NAME/best.pt" $BASELINES; do
    n=${tag%%:*}; ck=${tag#*:}
    [ -f "eval_out/${SKIND}_$n/summary.md" ] || $PY src/eval_scenes.py --model "$ck" --name "$n" \
        --sigmf "$SCENES/heldout_sigmf" --out "eval_out/${SKIND}_$n" --max-frames 4 --png-per-scenario 2
    [ -f "eval_out/argus_lte_$n/summary.md" ] || $PY src/eval_scenes.py --model "$ck" --name "$n" \
        --sigmf data/real_heldout --real --out "eval_out/argus_lte_$n" --max-frames 6 --png-per-scenario 4
    [ -f "eval_out/hard_$n/summary.md" ] || $PY src/eval_scenes.py --model "$ck" --name "$n" \
        --sigmf data/hard_test_cases --real --assume-continuous --out "eval_out/hard_$n" --max-frames 8 --png-per-scenario 3
  done
  # legacy 491.52 held-out (the M3/M4/M5 per-SNR table) for the new model
  [ -f "checkpoints/$NAME/heldout_eval/heldout_summary.json" ] || \
      $PY src/eval_heldout.py --ckpt "checkpoints/$NAME/best.pt" --dataset data/dataset_491 --out "checkpoints/$NAME/heldout_eval"
}
export_ts() {
  cd "$FT"
  out=weights/finetuned_dino_$(echo "$NAME" | tr 'A-Z' 'a-z')_bf16.ts
  [ -f "$out" ] && { echo "[export] exists: $out"; return; }
  $PY ../applications/usrp_wideband_signal_detection/export_dinov3_finetuned_torchscript.py \
      --ckpt "checkpoints/$NAME/best.pt" --output "$out" --autocast bf16 --sample-rate-hz 491.52e6 \
      --rate-range-hz 5e6 500e6 --fft-window hann --threshold 0.6 --dinov3-repo "$(dirname "$(dirname "$VITS")")"
}
case "$step" in
  combine) combine ;; train) train ;; eval) evaluate ;; export) export_ts ;;
  all) combine; train; evaluate; export_ts ;;
  *) echo "unknown step $step"; exit 1 ;;
esac

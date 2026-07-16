#!/usr/bin/env bash
# Fine-tune YOLO26 detection (signal/noise), BOTH sizes, sequentially (single GPU).
# STABILITY FIX (v2): the v1 run diverged to NaN ~epoch 20 (fp16 AMP overflow + auto lr0=0.01).
#   -> amp=False (fp32, no overflow), optimizer=SGD lr0=0.005, warmup_epochs=5, batch=8 (fits fp32).
# Identical params across s/m so it stays a clean size comparison. rect=True respects 256x1024.
set -uo pipefail
cd "$(dirname "$0")/.."
YOLO=/home/bqn82/miniforge3/envs/yolo/bin/yolo
SUMMARY="${SUMMARY:-runs/train_summary.log}"
mkdir -p runs; : > "$SUMMARY"
for m in yolo26s yolo26m; do
  echo "START $m $(date '+%F %T')" >> "$SUMMARY"
  "$YOLO" detect train model=${m}.pt data=configs/dataset.yaml \
      imgsz=1024 epochs=100 batch=8 rect=True \
      amp=False optimizer=SGD lr0=0.005 warmup_epochs=5 \
      name=${m}_signal device=0 exist_ok=True > "runs/train_${m}.log" 2>&1
  rc=$?
  bp=$(find runs -name best.pt -path "*${m}_signal*" 2>/dev/null | head -1)
  if [ $rc -eq 0 ] && [ -n "$bp" ]; then
    echo "PASS $m $(date '+%F %T') -> $bp" >> "$SUMMARY"
  else
    echo "FAIL $m rc=$rc $(date '+%F %T') (see runs/train_${m}.log)" >> "$SUMMARY"
  fi
done
echo "ALLDONE $(date '+%F %T')" >> "$SUMMARY"

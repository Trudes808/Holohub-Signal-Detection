#!/usr/bin/env bash
# Full end-to-end run: dataset -> 4 models -> 5 evals -> report.
# Resumable: training uses --resume; re-running skips a completed dataset build
# unless FORCE_BUILD=1. Progress is written to reports/STATUS.txt.
set -uo pipefail

ROOT=/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning
PY=/home/bqn82/miniforge3/envs/dinov3/bin/python
export PYTHONPATH="/home/bqn82/dinov3:$ROOT/src"
cd "$ROOT"

DATASET="$ROOT/data/dataset"
STATUS="$ROOT/reports/STATUS.txt"
mkdir -p "$ROOT/reports"

stamp(){ date "+%Y-%m-%d %H:%M:%S"; }
status(){ echo "[$(stamp)] $*" | tee -a "$STATUS"; }

status "=== FULL RUN START ==="

# 1) DATASET ---------------------------------------------------------------
if [[ "${FORCE_BUILD:-0}" == "1" || ! -f "$DATASET/dataset_meta.json" ]]; then
  status "STEP 1/4 build dataset (all captures)"
  $PY src/build_dataset.py --config configs/dataset.yaml 2>&1 | tee -a reports/build.log
  [[ -f "$DATASET/dataset_meta.json" ]] || { status "FATAL: dataset build failed"; exit 1; }
else
  status "STEP 1/4 dataset already built, skipping (FORCE_BUILD=1 to rebuild)"
fi
status "dataset counts: $(grep -o '\"counts\":[^}]*}' $DATASET/dataset_meta.json)"

# 2) TRAIN -----------------------------------------------------------------
train(){  # name  mode  extra...
  local name=$1 mode=$2; shift 2
  if [[ -f "checkpoints/$name/DONE" && "${FORCE_TRAIN:-0}" != "1" ]]; then
    status "  [$name] already complete (DONE), skipping"; return
  fi
  status "  [$name] training (mode=$mode $*)"
  $PY src/train.py --config configs/train.yaml --dataset "$DATASET" \
      --mode "$mode" --name "$name" --out "checkpoints/$name" --resume "$@" \
      2>&1 | tee -a "reports/train_$name.log"
}
status "STEP 2/4 train 4 models"
train M1_frozen frozen   --atten-max 30
train M1_ft     ft_lastN --atten-max 30
train M2_frozen frozen
train M2_ft     ft_lastN

# 3) EVAL ------------------------------------------------------------------
status "STEP 3/4 evaluate on shared all-dB test split"
evalm(){ # name  args...
  local name=$1; shift
  status "  [eval $name]"
  $PY src/evaluate.py --config configs/train.yaml --dataset "$DATASET" \
      --name "$name" --out eval_out "$@" 2>&1 | tee -a reports/eval.log
}
evalm energy    --baseline energy
evalm M1_frozen --ckpt checkpoints/M1_frozen/best.pt
evalm M1_ft     --ckpt checkpoints/M1_ft/best.pt
evalm M2_frozen --ckpt checkpoints/M2_frozen/best.pt
evalm M2_ft     --ckpt checkpoints/M2_ft/best.pt

# 4) REPORT ----------------------------------------------------------------
status "STEP 4/4 build report figures + tables"
$PY src/report.py --eval-root eval_out \
    --models energy,M1_frozen,M2_frozen,M1_ft,M2_ft \
    --reports reports --heatmap-model M2_ft 2>&1 | tee -a reports/report.log

status "=== FULL RUN DONE ==="

#!/usr/bin/env bash
# Convert (jupytext) + execute both comparison notebooks with the dinov3 kernel.
set -uo pipefail
BIN=/home/bqn82/miniforge3/envs/dinov3/bin
NBDIR=/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks
EVAL=/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments
export PYTHONPATH="/home/bqn82/dinov3:/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/src:$EVAL"
cd "$NBDIR"

for nb in batch_eval_review_three_detectors compare_three_detectors; do
  echo "=== $nb: jupytext convert ==="
  "$BIN/jupytext" --to notebook "$nb.py"
  echo "=== $nb: execute ==="
  "$BIN/jupyter" nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=python3 "$nb.ipynb"
  echo "$nb exit=$?"
done
echo ALLDONE

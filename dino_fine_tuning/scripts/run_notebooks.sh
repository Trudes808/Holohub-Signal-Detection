#!/usr/bin/env bash
# Convert (jupytext) + execute both comparison notebooks with the dinov3 kernel.
set -uo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)            # dino_fine_tuning/
REPO=$(cd "$ROOT/.." && pwd -P)
BIN=${BIN:-$(dirname "$(command -v jupyter)")}                       # env bin dir holding jupyter + jupytext
NBDIR=${NBDIR:-$REPO/notebooks/dino_fine_tuning_evals}
EVAL=$REPO/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments
DINOV3_ROOT=${DINOV3_ROOT:-$REPO/../dinov3}
export PYTHONPATH="$DINOV3_ROOT:$ROOT/src:$EVAL${PYTHONPATH:+:$PYTHONPATH}"
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

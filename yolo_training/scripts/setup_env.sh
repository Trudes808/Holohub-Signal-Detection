#!/usr/bin/env bash
# Create the 'yolo' conda env for Ultralytics YOLO26 on RF spectrograms.
# cu124 torch wheels (RTX 4000 Ada, sm_89), matching the dinov3 env.
set -euo pipefail
ENV_NAME=yolo
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[setup] env ${ENV_NAME} already exists; reusing"
else
  echo "[setup] creating env ${ENV_NAME} (python 3.11)"
  mamba create -y -n "${ENV_NAME}" python=3.11
fi
conda activate "${ENV_NAME}"
echo "[setup] installing torch (cu124) + ultralytics (YOLO26) + jupyter"
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
pip install -U ultralytics jupyter ipykernel jupytext pandas numpy matplotlib
python -m ipykernel install --user --name yolo --display-name "Python (yolo)"
echo "[setup] freezing exact versions -> requirements.txt"
pip freeze > "$(dirname "$0")/../requirements.txt"
echo "[setup] verifying"
python - <<'PY'
import torch, ultralytics
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("ultralytics", ultralytics.__version__)
PY
echo "[setup] DONE"

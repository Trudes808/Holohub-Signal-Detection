#!/usr/bin/env bash
# Create the 'dinov3' conda env for fine-tuning DINOv3 on RF spectrograms.
# Pick the torch wheel index for your driver: CUDA_WHEEL_INDEX=cu124 (x86 RTX 4000 Ada, default),
# cu130 (DGX Spark / CUDA 13), ...
set -euo pipefail

ENV_NAME=dinov3
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo "[setup] env ${ENV_NAME} already exists; reusing"
else
  echo "[setup] creating env ${ENV_NAME} (python 3.11)"
  mamba create -y -n "${ENV_NAME}" python=3.11
fi

conda activate "${ENV_NAME}"

echo "[setup] installing torch (cu124) + deps via pip"
pip install --upgrade pip
pip install --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL_INDEX:-cu124}" torch torchvision
pip install \
  torchmetrics omegaconf numpy scipy pandas scikit-learn \
  matplotlib seaborn tqdm pillow einops ftfy regex termcolor iopath pyyaml

echo "[setup] verifying torch + cuda + dinov3 backbone load"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

echo "[setup] DONE"

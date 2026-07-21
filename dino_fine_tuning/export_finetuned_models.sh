#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash (this script uses bash-only syntax).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Export the fine-tuned DINO segmenters (M1, M2) to TorchScript for the native
# finetuned_dino_detector operator. Verified TorchScript == eager (parity check inside the exporter).
#
# Writes into dino_fine_tuning/weights/, which is under the repo mounted at /workspace/holohub in the
# container -> the operator config's model_script_path is:
#   /workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m{1,2}.ts
#
# Run on the HOST (no container needed):  conda activate dinov3;  ./export_finetuned_models.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${DINO_ENV:-dinov3}"

declare -A CKPT=( [m1]="checkpoints/M1_ft/best.pt" [m2]="checkpoints/M2_ft/best.pt" )
declare -A META=( [m1]="eval_out/M1_ft/eval_meta.json" [m2]="eval_out/M2_ft/eval_meta.json" )
mkdir -p weights
for tag in m1 m2; do
  echo "=== exporting finetuned_dino_${tag}  (ckpt=${CKPT[$tag]}) ==="
  python src/export_finetuned_torchscript.py \
      --ckpt "${CKPT[$tag]}" --out "weights/finetuned_dino_${tag}.ts" --eval-meta "${META[$tag]}"
done
echo
echo "DONE. Container model_script_path values (repo is mounted at /workspace/holohub):"
echo "  finetuned_dino     -> /workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m1.ts"
echo "  finetuned_dino_m2  -> /workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m2.ts"

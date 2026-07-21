#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash (this script uses bash-only syntax).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Export trained YOLO26 s/m to TorchScript for the native yolo_detector operator.
# Writes into yolo_training/weights/ (under the repo mounted at /workspace/holohub in the container).
# Run on the HOST:  conda activate yolo;  ./export_yolo_models.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
cd "${SCRIPT_DIR}"
source ~/miniforge3/etc/profile.d/conda.sh
conda activate "${YOLO_ENV:-yolo}"
mkdir -p weights
declare -A CKPT=( [yolo26s]="runs/detect/yolo26s_signal/weights/best.pt" [yolo26m]="runs/detect/yolo26m_signal/weights/best.pt" )
for tag in yolo26s yolo26m; do
  echo "=== exporting ${tag}  (ckpt=${CKPT[$tag]}) ==="
  python src/export_yolo_torchscript.py --ckpt "${CKPT[$tag]}" --out "weights/${tag}.torchscript" --imgsz "${IMGSZ:-1024}"
done
echo
echo "DONE. Container model_script_path values (repo mounted at /workspace/holohub):"
echo "  yolo26s -> /workspace/holohub/yolo_training/weights/yolo26s.torchscript"
echo "  yolo26m -> /workspace/holohub/yolo_training/weights/yolo26m.torchscript"

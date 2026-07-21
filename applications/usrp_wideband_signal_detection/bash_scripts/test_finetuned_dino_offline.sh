#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash (this script uses bash-only syntax).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Smoke-test the native finetuned_dino_detector offline on a capture, then (optional) compare its masks
# to the Python reference masks in notebooks/yolo_evals/sweeps/sweep_all/<detector>.
# Needs docker/sudo (lab-admin) + a built container with the operator.
# Usage: sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./test_finetuned_dino_offline.sh [capture.sigmf-data] [detector]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
REPO_ROOT="$(cd "${APP_DIR}/../.." && pwd -P)"
USER_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"

DETECTOR="${2:-finetuned_dino}"
CAP="${1:-${USER_HOME}/captures/attenuation_dB_30.sigmf-data}"
OUT="${SNIP_OUT:-/tmp/usrp_spectrograms/optest/${DETECTOR}}"
echo "=== offline ${DETECTOR} on $(basename "$CAP") -> ${OUT} ==="
python3 "${APP_DIR}/run_cuda_dino_offline_file.py" "$CAP" --detector "${DETECTOR}" \
    --output-root "${OUT}" --progress-every 50

# optional IoU vs the Python reference masks (same stem under sweep_all)
STEM="$(basename "$CAP")"; STEM="${STEM%.sigmf-data}"
REF="${REPO_ROOT}/notebooks/yolo_evals/sweeps/sweep_all/${DETECTOR}/${STEM}/mask_arrays"
NEW="${OUT}/${DETECTOR}/${STEM}/mask_arrays"
if [[ -d "$REF" && -d "$NEW" ]]; then
  echo "=== IoU vs Python reference (${DETECTOR}/${STEM}) ==="
  python3 - "$REF" "$NEW" <<'PY'
import sys, numpy as np
from pathlib import Path
def load(p):
    z=np.load(p); return np.unpackbits(z["packed"])[:int(z["rows"])*int(z["cols"])].reshape(int(z["rows"]),int(z["cols"]))
ref,new=Path(sys.argv[1]),Path(sys.argv[2]); ious=[]
for rf in sorted(ref.glob("mask_ch0_f*.packed.npz"))[:50]:
    nf=new/rf.name
    if not nf.exists(): continue
    a,b=load(rf).astype(bool),load(nf).astype(bool)
    u=(a|b).sum(); ious.append(1.0 if u==0 else (a&b).sum()/u)
print(f"mean IoU over {len(ious)} frames = {np.mean(ious):.3f}" if ious else "no overlapping frames")
PY
else
  echo "(skip IoU compare: reference or new masks not found)"
fi

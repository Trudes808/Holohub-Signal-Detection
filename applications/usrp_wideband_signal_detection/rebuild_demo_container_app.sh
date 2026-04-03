#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
APP_NAME=${APP_NAME:-usrp_wideband_signal_detection}
MATX_DIR=${MATX_DIR:-/usr/local/lib/cmake/matx}

if ! sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container not found: ${CONTAINER_NAME}" >&2
  echo "Start it first with applications/usrp_wideband_signal_detection/run_demo_container.sh" >&2
  exit 1
fi

if [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  echo "Container is stopped; starting ${CONTAINER_NAME}."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
fi

sudo docker exec "${CONTAINER_NAME}" bash -lc 'set -euo pipefail
if ! ldconfig -p | grep -q "libvulkan.so.1"; then
  apt-get update
  apt-get install -y --no-install-recommends libvulkan1
fi'

if ! sudo docker exec "${CONTAINER_NAME}" bash -lc 'python3 - <<"PY"
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("torch")
if spec is None or spec.origin is None:
    raise SystemExit(1)

torch_dir = Path(spec.origin).resolve().parent
torch_config = torch_dir / "share" / "cmake" / "Torch" / "TorchConfig.cmake"
if not torch_config.exists():
    raise SystemExit(2)
PY'; then
  echo "Warning: container PyTorch CMake files are missing; the rebuild will compile dinov3_signal_detector without Torch support." >&2
  echo "Run applications/usrp_wideband_signal_detection/setup_demo_container.sh to install the pinned CUDA PyTorch stack and export the TorchScript artifact before rebuilding." >&2
fi

sudo docker exec "${CONTAINER_NAME}" bash -lc "set -euo pipefail && \
  cd /workspace/holohub && \
  export HOLOHUB_BUILD_LOCAL=1 && \
  ./holohub build ${APP_NAME} --local --configure-args=-Dmatx_DIR=${MATX_DIR}"

sudo docker exec "${CONTAINER_NAME}" bash -lc "ls -lah /workspace/holohub/build/${APP_NAME}/applications/${APP_NAME}"
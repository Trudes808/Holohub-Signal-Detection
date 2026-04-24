#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)
WORKSPACE_ROOT=$(cd "${REPO_ROOT}/.." && pwd -P)

APP_NAME=${APP_NAME:-usrp_wideband_signal_detection}
IMAGE_NAME=${IMAGE_NAME:-usrp_x410_signal_detection_demo:latest}
DOCKER_FILE=${DOCKER_FILE:-applications/usrp_freq_detection/Dockerfile}
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
HUGEPAGES_DIR=${HUGEPAGES_DIR:-/dev/hugepages}
SPECTROGRAM_HOST_DIR=${SPECTROGRAM_HOST_DIR:-/tmp/usrp_spectrograms}
DINO_MASK_HOST_DIR=${DINO_MASK_HOST_DIR:-/tmp/usrp_dino_masks}
COHERENT_SNAPSHOT_HOST_DIR=${COHERENT_SNAPSHOT_HOST_DIR:-/tmp/coherent_power_snapshots}
COHERENT_MASK_HOST_DIR=${COHERENT_MASK_HOST_DIR:-/tmp/coherent_power_masks}
HOST_DINOV3_ROOT=${HOST_DINOV3_ROOT:-${WORKSPACE_ROOT}/dinov3}
HOST_WEIGHT_PATH=${HOST_WEIGHT_PATH:-${HOST_DINOV3_ROOT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
CONTAINER_DINOV3_ROOT=${CONTAINER_DINOV3_ROOT:-/workspace/models/dinov3}
CONTAINER_WEIGHT_PATH=${CONTAINER_WEIGHT_PATH:-${CONTAINER_DINOV3_ROOT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth}
CONTAINER_TORCHSCRIPT_PATH=${CONTAINER_TORCHSCRIPT_PATH:-${CONTAINER_DINOV3_ROOT}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts}
CONTAINER_EXPORT_SCRIPT=${CONTAINER_EXPORT_SCRIPT:-/workspace/holohub/applications/usrp_wideband_signal_detection/export_dinov3_torchscript.py}
PYTORCH_INDEX_URL=${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}
PYTORCH_VERSION=${PYTORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.25.0}
MATX_VERSION=${MATX_VERSION:-0.9.2}
BUILD_APP_IN_CONTAINER=${BUILD_APP_IN_CONTAINER:-1}
INSTALL_PYTHON_DEPS=${INSTALL_PYTHON_DEPS:-1}
SKIP_IMAGE_BUILD=${SKIP_IMAGE_BUILD:-0}
ENSURE_VULKAN_RUNTIME=${ENSURE_VULKAN_RUNTIME:-1}
DISPLAY_VALUE=${DISPLAY:-}
XAUTHORITY_VALUE=${XAUTHORITY:-}
X11_SOCKET_DIR=/tmp/.X11-unix

run_in_container() {
	sudo docker exec "${CONTAINER_NAME}" bash -lc "$1"
}

install_pytorch_cuda_stack() {
	run_in_container "export PIP_BREAK_SYSTEM_PACKAGES=1 && \
		python3 -m pip uninstall -y torch torchvision torchaudio || true && \
		python3 -m pip install --no-cache-dir --index-url ${PYTORCH_INDEX_URL} torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION}"
}

install_matx() {
	echo "Installing MatX ${MATX_VERSION} into ${CONTAINER_NAME}."
	run_in_container "set -euo pipefail && \
		command -v curl >/dev/null 2>&1 || (apt-get update && apt-get install -y --no-install-recommends curl) && \
		rm -rf /tmp/matx && mkdir -p /tmp/matx && cd /tmp/matx && \
		curl -OL https://github.com/NVIDIA/MatX/archive/refs/tags/v${MATX_VERSION}.tar.gz && \
		tar -xzf v${MATX_VERSION}.tar.gz && cd MatX-${MATX_VERSION} && \
		mkdir -p build && cd build && cmake .. && make -j\$(nproc) && make install && \
		rm -rf /tmp/matx"
}

ensure_vulkan_runtime() {
	run_in_container 'set -euo pipefail
if ! ldconfig -p | grep -q "libvulkan.so.1"; then
	apt-get update
	apt-get install -y --no-install-recommends libvulkan1
fi'
}

ensure_nvjitlink_symlink() {
	run_in_container 'set -euo pipefail
for libdir in \
	/usr/local/cuda/targets/x86_64-linux/lib \
	/usr/local/cuda/lib64 \
	/usr/local/cuda-12.6/targets/x86_64-linux/lib \
	/usr/local/cuda-12.6/lib64; do
	if [[ -f "$libdir/libnvJitLink.so.12" && ! -e "$libdir/libnvJitLink.so" ]]; then
		ln -s libnvJitLink.so.12 "$libdir/libnvJitLink.so"
	fi
done'
}

require_network_build_deps() {
	if ! run_in_container 'pkg-config --exists libdpdk'; then
		echo "Container image ${CONTAINER_NAME} is missing DPDK build dependencies (pkg-config could not find libdpdk)." >&2
		echo "Rebuild the image from the MatX/USRP-enabled base and rerun this script." >&2
		exit 1
	fi
}

cd "${REPO_ROOT}"

if [[ ! -d "${HOST_DINOV3_ROOT}" ]]; then
	echo "Host DINOv3 repo not found: ${HOST_DINOV3_ROOT}" >&2
	exit 1
fi

if [[ ! -f "${HOST_WEIGHT_PATH}" ]]; then
	echo "Host DINOv3 weight not found: ${HOST_WEIGHT_PATH}" >&2
	exit 1
fi

mkdir -p "${SPECTROGRAM_HOST_DIR}" "${DINO_MASK_HOST_DIR}" "${COHERENT_SNAPSHOT_HOST_DIR}" "${COHERENT_MASK_HOST_DIR}"

echo "Using repo checkout: ${REPO_ROOT}"

if [[ "${SKIP_IMAGE_BUILD}" != "1" ]]; then
	sudo ./holohub build-container "${APP_NAME}" --docker-file "${DOCKER_FILE}" --img "${IMAGE_NAME}" "$@"
fi

sudo docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

DOCKER_RUN_CMD=(
	sudo docker run
	--name "${CONTAINER_NAME}" \
	--privileged \
	--net host \
	-u 0:0 \
	-e HOLOHUB_BUILD_LOCAL=1 \
	-e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
	-v "${REPO_ROOT}:/workspace/holohub" \
	-v "${SPECTROGRAM_HOST_DIR}:/workspace/spectrograms" \
	-v "${DINO_MASK_HOST_DIR}:/workspace/dino_masks" \
	-v "${COHERENT_SNAPSHOT_HOST_DIR}:/workspace/coherent_power_snapshots" \
	-v "${COHERENT_MASK_HOST_DIR}:/workspace/coherent_power_masks" \
	-v /dev:/dev \
	-v "${HUGEPAGES_DIR}:/dev/hugepages" \
	-w /workspace/holohub \
	--runtime nvidia \
	--gpus all \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	--ipc=host \
)

if [[ -n "${DISPLAY_VALUE}" && -d "${X11_SOCKET_DIR}" ]]; then
	DOCKER_RUN_CMD+=(
		-e DISPLAY="${DISPLAY_VALUE}" \
		-v "${X11_SOCKET_DIR}:${X11_SOCKET_DIR}:rw"
	)

	if [[ -n "${XAUTHORITY_VALUE}" && -f "${XAUTHORITY_VALUE}" ]]; then
		DOCKER_RUN_CMD+=(
			-e XAUTHORITY="${XAUTHORITY_VALUE}" \
			-v "${XAUTHORITY_VALUE}:${XAUTHORITY_VALUE}:ro"
		)
	fi
fi

DOCKER_RUN_CMD+=(
	--detach
	"${IMAGE_NAME}"
	bash
	-lc
	'trap : TERM INT; sleep infinity & wait'
)

"${DOCKER_RUN_CMD[@]}" >/dev/null

echo "Started container ${CONTAINER_NAME} from image ${IMAGE_NAME}."
echo "Mounted host repo: ${REPO_ROOT} -> /workspace/holohub"
echo "Host spectrogram output: ${SPECTROGRAM_HOST_DIR}"
echo "Host DINO mask output: ${DINO_MASK_HOST_DIR}"
echo "Host coherent snapshot output: ${COHERENT_SNAPSHOT_HOST_DIR}"
echo "Host coherent mask output: ${COHERENT_MASK_HOST_DIR}"

sudo docker exec "${CONTAINER_NAME}" rm -rf "${CONTAINER_DINOV3_ROOT}"
sudo docker exec "${CONTAINER_NAME}" mkdir -p "${CONTAINER_DINOV3_ROOT}/weights"

tar --exclude='./weights' -C "${HOST_DINOV3_ROOT}" -cf - . \
	| sudo docker exec -i "${CONTAINER_NAME}" tar -xf - -C "${CONTAINER_DINOV3_ROOT}"

sudo docker cp "${HOST_WEIGHT_PATH}" "${CONTAINER_NAME}:${CONTAINER_WEIGHT_PATH}"

echo "Verifying NVIDIA runtime inside ${CONTAINER_NAME}."
run_in_container 'nvidia-smi'

if [[ "${INSTALL_PYTHON_DEPS}" == "1" ]]; then
	if ! run_in_container 'python3 -c "import torch, torchvision, ftfy, omegaconf, regex, submitit, termcolor, torchmetrics"' >/dev/null 2>&1; then
		echo "Installing DINOv3 Python requirements into ${CONTAINER_NAME}."
		run_in_container 'export PIP_BREAK_SYSTEM_PACKAGES=1 && python3 -m pip install --no-cache-dir ftfy omegaconf regex scikit-learn submitit termcolor torchmetrics'
		install_pytorch_cuda_stack
	fi

	if ! sudo docker exec "${CONTAINER_NAME}" python3 - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

if not str(torch.version.cuda).startswith("12.6"):
		sys.exit(1)

if not torch.cuda.is_available():
		sys.exit(1)
PY
	then
		echo "Reinstalling pinned PyTorch ${PYTORCH_VERSION} / torchvision ${TORCHVISION_VERSION} from ${PYTORCH_INDEX_URL}."
		install_pytorch_cuda_stack
	fi
fi

echo "Checking CUDA availability in PyTorch inside ${CONTAINER_NAME}."
sudo docker exec "${CONTAINER_NAME}" python3 - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")

if not torch.cuda.is_available():
		raise SystemExit(
				"PyTorch still cannot initialize CUDA inside the container. "
				"This usually means the selected wheel is incompatible with the host driver or the container GPU runtime is not configured correctly."
		)

print(f"cuda_device_count={torch.cuda.device_count()}")
print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
PY

ensure_nvjitlink_symlink

if [[ "${ENSURE_VULKAN_RUNTIME}" == "1" ]]; then
	ensure_vulkan_runtime
else
	echo "Skipping Vulkan runtime installation check because ENSURE_VULKAN_RUNTIME=${ENSURE_VULKAN_RUNTIME}."
fi

if ! run_in_container 'test -f /usr/local/lib/cmake/matx/matx-config.cmake'; then
	install_matx
fi

run_in_container 'ls -lah /usr/local/lib/cmake/matx'
require_network_build_deps

sudo docker exec "${CONTAINER_NAME}" python3 "${CONTAINER_EXPORT_SCRIPT}" \
	--repo "${CONTAINER_DINOV3_ROOT}" \
	--weights "${CONTAINER_WEIGHT_PATH}" \
	--output "${CONTAINER_TORCHSCRIPT_PATH}" \
	--model-name dinov3_vitb16 \
	--height 256 \
	--width 512 \
	--device cuda

run_in_container "ls -lah ${CONTAINER_DINOV3_ROOT}/weights"

if [[ "${BUILD_APP_IN_CONTAINER}" == "1" ]]; then
	run_in_container "cd /workspace/holohub && export HOLOHUB_BUILD_LOCAL=1 && ./holohub build ${APP_NAME} --local --configure-args=-Dmatx_DIR=/usr/local/lib/cmake/matx"
	run_in_container "ls -lah /workspace/holohub/build/${APP_NAME}/applications/${APP_NAME}"
fi

echo "Initial container setup is complete."
echo "Next host commands:"
echo "  ./run_demo_container.sh"
echo "  ./enter_demo_container.sh"
echo "  ./rebuild_demo_container_app.sh"
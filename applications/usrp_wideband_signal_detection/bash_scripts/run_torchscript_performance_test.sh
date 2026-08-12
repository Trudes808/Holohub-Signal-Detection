#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/container_env.sh"
CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
SOURCE_APP_DIR=${SOURCE_APP_DIR:-/workspace/holohub/applications/usrp_wideband_signal_detection}
REBUILD_DEMO_CONTAINER_APP=${REBUILD_DEMO_CONTAINER_APP:-${SCRIPT_DIR}/rebuild_demo_container_app.sh}
DISPLAY_VALUE=${DISPLAY:-}
XAUTHORITY_VALUE=${XAUTHORITY:-}
XDG_RUNTIME_DIR_VALUE=${XDG_RUNTIME_DIR:-/tmp/xdg-runtime-root}

if [[ $# -gt 1 ]]; then
	echo "Usage: $0 [config-name.yaml]" >&2
	exit 1
fi

CONFIG_NAME=${1:-${CONFIG_NAME:-config_cuda_dino_performance_single_channel.yaml}}

if [[ $# -eq 0 ]]; then
	echo "NOTE: no config argument provided — using: ${CONFIG_NAME}" >&2
	echo "      (a CONFIG_NAME=... prefix does NOT survive sudo; pass the config as the first argument instead)" >&2
fi

echo "Running usrp_wideband_signal_detection with config: ${CONFIG_NAME}"

"${REBUILD_DEMO_CONTAINER_APP}"

if ! sudo docker exec "${CONTAINER_NAME}" bash -lc "test -f ${SOURCE_APP_DIR}/${CONFIG_NAME}"; then
	echo "Config not found in source tree: ${SOURCE_APP_DIR}/${CONFIG_NAME}" >&2
	exit 1
fi

sudo docker exec "${CONTAINER_NAME}" bash -lc "set -euo pipefail && cp ${SOURCE_APP_DIR}/${CONFIG_NAME} ${BUILD_APP_DIR}/${CONFIG_NAME}"

existing_processes=$(sudo docker exec "${CONTAINER_NAME}" bash -lc "pgrep -af '(^|/)usrp_wideband_signal_detection( |$)' || true")
if [[ -n "${existing_processes}" ]]; then
	echo "Another usrp_wideband_signal_detection instance is already running in ${CONTAINER_NAME}:" >&2
	echo "${existing_processes}" >&2
	echo "Stop the stale process before starting a new run." >&2
	echo "Suggested cleanup: sudo docker exec ${CONTAINER_NAME} pkill -f '(^|/)usrp_wideband_signal_detection( |$)'" >&2
	exit 1
fi

sudo docker exec -u 0:0 "${CONTAINER_NAME}" bash -lc "mkdir -p '${XDG_RUNTIME_DIR_VALUE}' && chown 0:0 '${XDG_RUNTIME_DIR_VALUE}' && chmod 700 '${XDG_RUNTIME_DIR_VALUE}'"

# Auto-adopt the stream rate/center from the sidecar written by rx_to_remote_udp.py (live) or
# replay_rx_to_buff.py (loopback), unless already set explicitly in the environment. This is what
# makes a new radio rate/center flow through the pipeline with no config edit. It is consumed once
# (deleted after reading) so it is strictly a fresh handoff from the sender/replay to THIS launch and
# never silently overrides a later run of a different config; when absent, the config's
# channel_sample_rates_hz (else nominal fft.span) is used.
STREAM_PARAMS_SIDECAR="${USRP_STREAM_PARAMS_FILE:-/tmp/usrp_stream_params.json}"
if [[ -f "${STREAM_PARAMS_SIDECAR}" ]]; then
	if [[ -z "${USRP_SAMPLE_RATE_HZ:-}" ]]; then
		USRP_SAMPLE_RATE_HZ=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('sample_rate_hz',''))" "${STREAM_PARAMS_SIDECAR}" 2>/dev/null || true)
	fi
	if [[ -z "${USRP_CENTER_FREQ_HZ:-}" ]]; then
		USRP_CENTER_FREQ_HZ=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('center_freq_hz',''))" "${STREAM_PARAMS_SIDECAR}" 2>/dev/null || true)
	fi
	echo "Using stream params from ${STREAM_PARAMS_SIDECAR}: USRP_SAMPLE_RATE_HZ=${USRP_SAMPLE_RATE_HZ:-} USRP_CENTER_FREQ_HZ=${USRP_CENTER_FREQ_HZ:-} (consuming sidecar)"
	sudo rm -f "${STREAM_PARAMS_SIDECAR}" 2>/dev/null || rm -f "${STREAM_PARAMS_SIDECAR}" 2>/dev/null || true
fi

# The sidecar center is a single (channel-0) value. Multi-channel configs carry their own
# per-channel centers (chdr_converter.channel_center_frequencies_hz), and a global override
# would mislabel every channel with channel 0's frequency — so only apply it to
# single-channel configs. The rate override stays (all channels share one rate).
HOST_CONFIG_PATH="${SCRIPT_DIR}/../${CONFIG_NAME}"
if [[ -n "${USRP_CENTER_FREQ_HZ:-}" && -f "${HOST_CONFIG_PATH}" ]]; then
	CONFIG_NUM_CHANNELS=$(awk '/^chdr_converter:/{f=1;next} f&&/^[^[:space:]]/{exit} f&&$1=="num_channels:"{print $2; exit}' "${HOST_CONFIG_PATH}")
	if [[ "${CONFIG_NUM_CHANNELS:-1}" =~ ^[0-9]+$ && "${CONFIG_NUM_CHANNELS:-1}" -gt 1 ]]; then
		echo "Multi-channel config (num_channels=${CONFIG_NUM_CHANNELS}): ignoring single-value USRP_CENTER_FREQ_HZ=${USRP_CENTER_FREQ_HZ}; per-channel centers come from the config."
		USRP_CENTER_FREQ_HZ=""
	fi
fi

exec sudo docker exec -it \
	-e DISPLAY="${DISPLAY_VALUE}" \
	-e XAUTHORITY="${XAUTHORITY_VALUE}" \
	-e XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR_VALUE}" \
	-e USRP_SAMPLE_RATE_HZ="${USRP_SAMPLE_RATE_HZ:-}" \
	-e USRP_CENTER_FREQ_HZ="${USRP_CENTER_FREQ_HZ:-}" \
	"${CONTAINER_NAME}" bash -lc "cd ${BUILD_APP_DIR} && ./usrp_wideband_signal_detection ${CONFIG_NAME}"
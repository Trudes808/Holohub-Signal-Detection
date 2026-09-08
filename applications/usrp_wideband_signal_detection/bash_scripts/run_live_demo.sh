#!/usr/bin/env bash
set -euo pipefail

# One-command live demo launcher (DGX Spark bench).
#
# Starts the Holoscan app, waits until DPDK is armed, starts the X410 over-the-air
# stream, mirrors the app log to this terminal, and tears everything down on Ctrl-C.
#
#   sudo ./bash_scripts/run_live_demo.sh            # dual channel (2400 + 1000 MHz)
#   sudo ./bash_scripts/run_live_demo.sh single     # single channel @ 2400 MHz
#   sudo ./bash_scripts/run_live_demo.sh <config.yaml>
#
# Radio knobs are the same env vars as start_radio_stream.sh (FREQS, GAIN, CHANNELS,
# DEST_PORTS, DURATION, ...), e.g.:
#   sudo env FREQS="915e6 1800e6" GAIN=40 ./bash_scripts/run_live_demo.sh
# (Use `sudo env VAR=...` — a plain VAR=... prefix does not survive sudo.)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_env.sh"

BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
SOURCE_APP_DIR=${SOURCE_APP_DIR:-/workspace/holohub/applications/usrp_wideband_signal_detection}
APP_LOG=${APP_LOG:-/tmp/live_demo_app.log}   # path inside the container

MODE=${1:-dual}
case "${MODE}" in
  dual)
    CONFIG_NAME=config_coherent_power_performance_emit_stride1_two_channel.yaml
    ;;
  single)
    CONFIG_NAME=config_coherent_power_perf_dynamic_single_channel.yaml
    export CHANNELS=${CHANNELS:-0} FREQS=${FREQS:-2400e6} DEST_PORTS=${DEST_PORTS:-1234}
    ;;
  v3)
    # Dashboard v3: detector + classifier pipeline (no decode). Snipper +
    # compression feed the classify-only daemon; DEMO CONTROLS drive the
    # classifier checklist live and detector switches via the conductor.
    CONFIG_NAME=config_live_v3_single_channel.yaml
    export CHANNELS=${CHANNELS:-0} FREQS=${FREQS:-2400e6} DEST_PORTS=${DEST_PORTS:-1234}
    V3_STACK=1
    ;;
  *.yaml)
    CONFIG_NAME=${MODE}
    ;;
  *)
    echo "Usage: sudo $0 [single|dual|<config.yaml>]" >&2
    exit 1
    ;;
esac

# The CG_400 image always negotiates 491.52 Msps, and the app launches BEFORE the radio,
# so the stream-params sidecar can't deliver the true rate in time. Hand it over directly
# so the FFT geometry / frequency axis is exact (config fft.span says 500e6).
export USRP_SAMPLE_RATE_HZ=${USRP_SAMPLE_RATE_HZ:-491520000}
if [[ "${MODE}" == "single" ]]; then
  # Single value is safe for one channel; multi-channel centers come from the config.
  export USRP_CENTER_FREQ_HZ=${USRP_CENTER_FREQ_HZ:-${FREQS}}
fi

if pgrep -f "rx_to_remote_udp.py" >/dev/null 2>&1; then
  echo "A radio stream (rx_to_remote_udp.py) is already running — stop it first:" >&2
  echo "  pkill -f 'python3 rx_to_remote_udp'" >&2
  exit 1
fi

# Verify the physical radio topology BEFORE launching (cables/IPs get shuffled on a shared
# bench; a crossed topology otherwise fails as a silent 0-packet run). Also enforces the
# canonical host addressing when the wiring checks out. Skip with SKIP_TOPOLOGY_CHECK=1.
if [[ "${SKIP_TOPOLOGY_CHECK:-0}" != "1" ]]; then
  echo "==> Checking radio topology"
  if ! "${SCRIPT_DIR}/check_radio_topology.sh"; then
    echo "Radio topology check failed — fix the wiring above (or SKIP_TOPOLOGY_CHECK=1 to override)." >&2
    exit 1
  fi
fi

echo "==> Syncing configs / rebuilding if needed"
"${SCRIPT_DIR}/rebuild_demo_container_app.sh"

stop_app() {
  sudo docker exec "${CONTAINER_NAME}" bash -lc '
    p=$(ps -eo pid,comm | awk "\$2 ~ /^usrp_wideband/ {print \$1}")
    [ -n "$p" ] && kill $p 2>/dev/null || true
    for i in $(seq 1 30); do ps -eo comm | grep -q "^usrp_wideband" || break; sleep 1; done
    rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null
    rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true' 2>/dev/null || true
}

RADIO_PID=""
TAIL_PID=""
DAEMON_PID=""
CONDUCTOR_PID=""
JANITOR_PID=""
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "==> Shutting down the demo"
  [ -n "${RADIO_PID}" ] && kill "${RADIO_PID}" 2>/dev/null || true
  [ -n "${TAIL_PID}" ] && kill "${TAIL_PID}" 2>/dev/null || true
  [ -n "${DAEMON_PID}" ] && kill "${DAEMON_PID}" 2>/dev/null || true
  [ -n "${CONDUCTOR_PID}" ] && kill "${CONDUCTOR_PID}" 2>/dev/null || true
  [ -n "${JANITOR_PID}" ] && kill "${JANITOR_PID}" 2>/dev/null || true
  stop_app
  echo "==> Demo stopped."
}
trap cleanup INT TERM EXIT

echo "==> Stopping any previous app instance"
stop_app

if [[ "${V3_STACK:-0}" == "1" ]]; then
  echo "==> v3: control state + snippet scratch (before the app starts writing)"
  sudo mkdir -p /tmp/usrp_spectrograms && sudo chmod 1777 /tmp/usrp_spectrograms
  sudo rm -rf /tmp/usrp_spectrograms/snippets 2>/dev/null || true
  printf '{"gate": "tprime", "snr": "clean", "detector": "coherent_power", "classifiers": "vtcnn2,resnet1d,tprime", "seq": 1}\n' \
    | sudo tee /tmp/usrp_spectrograms/demo_control.json > /dev/null
  sudo chmod 666 /tmp/usrp_spectrograms/demo_control.json
fi

echo "==> Launching the app (config: ${CONFIG_NAME})"
sudo docker exec -d \
  -e DISPLAY="${DISPLAY:-}" \
  -e USRP_SAMPLE_RATE_HZ="${USRP_SAMPLE_RATE_HZ}" \
  -e USRP_CENTER_FREQ_HZ="${USRP_CENTER_FREQ_HZ:-}" \
  "${CONTAINER_NAME}" bash -lc "
  cp '${SOURCE_APP_DIR}/${CONFIG_NAME}' '${BUILD_APP_DIR}/' 2>/dev/null || true
  mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root
  export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root
  cd '${BUILD_APP_DIR}'
  exec ./usrp_wideband_signal_detection '${CONFIG_NAME}' > '${APP_LOG}' 2>&1"

echo "==> Waiting for DPDK to arm (RX flows)"
for i in $(seq 1 60); do
  if sudo docker exec "${CONTAINER_NAME}" bash -lc "grep -q 'Adding RX flow' '${APP_LOG}' 2>/dev/null"; then
    break
  fi
  if ! sudo docker exec "${CONTAINER_NAME}" bash -lc 'ps -eo comm | grep -q "^usrp_wideband"'; then
    echo "App exited during startup — last log lines:" >&2
    sudo docker exec "${CONTAINER_NAME}" bash -lc "tail -25 '${APP_LOG}'" >&2 || true
    exit 1
  fi
  sleep 1
done

# Mirror the app log into this terminal alongside the radio output.
sudo docker exec "${CONTAINER_NAME}" tail -f "${APP_LOG}" &
TAIL_PID=$!

if [[ "${V3_STACK:-0}" == "1" ]]; then
  APP_DIR_HOST=$(cd "${SCRIPT_DIR}/.." && pwd -P)
  # under sudo, HOME=/root — resolve the desktop user's home for the ML venv
  V3_USER_HOME=$(getent passwd "${SUDO_USER:-${USER}}" | cut -d: -f6)
  VENV_PY="${VENV_PY:-${V3_USER_HOME}/Documents/holoscan_waveform_generation/.venv-ml/bin/python}"
  echo "==> v3: classify-only AMC daemon (THROUGHPUT + COMPUTE panels)"
  # stale metrics from a previous run must not paint the panels before the
  # daemon's first write
  sudo rm -f /tmp/usrp_spectrograms/rt_metrics.json /tmp/usrp_spectrograms/daemon_live.log
  (cd "${APP_DIR_HOST}/infocom_evals/pycodec_e2e" && \
   PYCODEC_ROOT="${V3_USER_HOME}/Documents/holoscan_waveform_generation" \
   exec "${VENV_PY}" rt_decode_daemon.py --snips /tmp/usrp_spectrograms/snippets \
     --classify-only --no-band-truth \
     --metrics-out /tmp/usrp_spectrograms/rt_metrics.json \
     > /tmp/usrp_spectrograms/daemon_live.log 2>&1) &
  DAEMON_PID=$!

  echo "==> v3: conductor (detector switches from DEMO CONTROLS; no replay)"
  (cd "${APP_DIR_HOST}/infocom_evals/pycodec_e2e" && \
   exec sudo python3 demo_conductor.py --no-replay --display "${DISPLAY:-:1}" \
     --rate-hz "${USRP_SAMPLE_RATE_HZ}" \
     > /tmp/usrp_spectrograms/conductor.log 2>&1) &
  CONDUCTOR_PID=$!

  echo "==> v3: snippet janitor (keep 5 min)"
  (sudo bash -c 'while true; do
     find /tmp/usrp_spectrograms/snippets -name "snip_pack*" -mmin +5 -delete 2>/dev/null
     sleep 30; done' > /dev/null 2>&1) &
  JANITOR_PID=$!
fi

echo "==> Starting the over-the-air radio stream (Ctrl-C stops everything)"
"${SCRIPT_DIR}/start_radio_stream.sh" &
RADIO_PID=$!
wait "${RADIO_PID}" || true

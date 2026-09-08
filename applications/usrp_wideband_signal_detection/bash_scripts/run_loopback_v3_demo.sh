#!/usr/bin/env bash
# Loopback replay demo with the v3 dashboard (detector + classifier pipeline,
# throughput/GPU/compute HUD, classifier checklist -- NO decode/BER). Same
# dashboard as `run_live_demo.sh v3`, but fed by tcpreplay over the Spark's own
# QSFP loopback cable instead of the USRP, so it needs no radio.
#
# Requires the two CX7 ports cabled to each OTHER (not to the USRP): tcpreplay
# sends on SENDER_IFACE, DPDK receives on the app's data port (0000:01:00.0).
#
# Run WITHOUT sudo (it sudo's the pieces that need it; xhost must run as you):
#     ./bash_scripts/run_loopback_v3_demo.sh
# Stop everything:
#     ./bash_scripts/stop_loopback_demo.sh   (also tears this down)
#
# Knobs (env):
#   SENDER_IFACE  loopback sender port           (default enp1s0f1np1)
#   DEMO_DISPLAY  X display for the window        (default $DISPLAY or :1)
#   PCAP          composite pcap to loop          (default snr_clean_burst.pcap)
#   PPS           tcpreplay packet rate           (default 240000 = full 245.76 MSps)
#   CONFIG_NAME   initial v3 config               (default config_loopback_v3_single_channel.yaml)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=container_env.sh
source "${SCRIPT_DIR}/container_env.sh"

SENDER_IFACE="${SENDER_IFACE:-enp1s0f1np1}"
DEMO_DISPLAY="${DEMO_DISPLAY:-${DISPLAY:-:1}}"
# The composites are 245.76 MSps -- which is ALSO the DINO-FT (M2_dr) trained
# rate, so DINO-FT runs at its native trained geometry here (looks its best).
RATE_HZ=245760000
CENTER_HZ=2400000000
PPS="${PPS:-240000}"
COMPOSITES="/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites"
PCAP="${PCAP:-snr_clean_burst.pcap}"
PCAP_PATH="${COMPOSITES}/${PCAP}"
CONFIG_NAME="${CONFIG_NAME:-config_loopback_v3_single_channel.yaml}"
BUILD_APP_DIR="/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection"
SOURCE_APP_DIR="/workspace/holohub/applications/usrp_wideband_signal_detection"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this WITHOUT sudo -- xhost must run as the desktop user (the script sudo's what needs root)." >&2
  exit 1
fi
if [[ ! -f "${PCAP_PATH}" ]]; then
  echo "Missing pcap: ${PCAP_PATH}" >&2
  echo "Available:"; ls "${COMPOSITES}"/*.pcap 2>/dev/null >&2 || true
  exit 1
fi

# under sudo, HOME=/root -- resolve the desktop user's home for the ML venv
V3_USER_HOME=$(getent passwd "${SUDO_USER:-${USER}}" | cut -d: -f6)
VENV_PY="${VENV_PY:-${V3_USER_HOME}/Documents/holoscan_waveform_generation/.venv-ml/bin/python}"

RADIO_PID=""      # tcpreplay
cleanup() {
  trap - INT TERM EXIT
  echo
  echo "==> Shutting down the loopback v3 demo"
  [ -n "${RADIO_PID}" ] && sudo kill "${RADIO_PID}" 2>/dev/null || true
  sudo pkill -f '^tcpreplay' 2>/dev/null || true
  sudo pkill -f 'demo_conduc[t]or' 2>/dev/null || true
  pkill -f 'rt_decode_dae[m]on.py --snips' 2>/dev/null || true
  sudo pkill -f 'usrp_spectrograms/snippets.*-mmin' 2>/dev/null || true
  sudo docker exec "${CONTAINER_NAME}" bash -lc \
    "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true" 2>/dev/null || true
  echo "==> Loopback v3 demo stopped."
}
trap cleanup INT TERM EXIT

echo "==> [1/7] X access for the container window (resets on login/lock)"
DISPLAY="${DEMO_DISPLAY}" xhost +local: >/dev/null

echo "==> [2/7] sender port ${SENDER_IFACE} up @ MTU 9000 (replay frames are ~4.2 KB)"
sudo ip link set "${SENDER_IFACE}" up mtu 9000

echo "==> [3/7] v3 scratch + control state (classifiers field drives the checklist)"
sudo mkdir -p /tmp/usrp_spectrograms && sudo chmod 1777 /tmp/usrp_spectrograms
sudo rm -rf /tmp/usrp_spectrograms/snippets 2>/dev/null || true
sudo rm -f /tmp/usrp_stream_params.json
# stale metrics must not paint the panels before the daemon's first write
sudo rm -f /tmp/usrp_spectrograms/rt_metrics.json /tmp/usrp_spectrograms/daemon_live.log \
  /tmp/usrp_spectrograms/snip_stats*.json
printf '{"gate": "tprime", "snr": "clean", "detector": "coherent_power", "classifiers": "vtcnn2,resnet1d,tprime", "seq": 1}\n' \
  | sudo tee /tmp/usrp_spectrograms/demo_control.json > /dev/null
sudo chmod 666 /tmp/usrp_spectrograms/demo_control.json

echo "==> [4/7] app (v3 pipeline: detector + snipper + compression + viz) on ${DEMO_DISPLAY}"
sudo docker exec "${CONTAINER_NAME}" bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; \
   for i in \$(seq 1 30); do pgrep -f '(^|/)usrp_wideband_signal_detection( |\$)' >/dev/null || break; sleep 1; done; \
   rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"
sudo docker exec "${CONTAINER_NAME}" bash -lc \
  "cp '${SOURCE_APP_DIR}/${CONFIG_NAME}' '${BUILD_APP_DIR}/' 2>/dev/null || true"
sudo docker exec -d -e DISPLAY="${DEMO_DISPLAY}" \
  -e USRP_SAMPLE_RATE_HZ="${RATE_HZ}" -e USRP_CENTER_FREQ_HZ="${CENTER_HZ}" \
  "${CONTAINER_NAME}" bash -lc \
  "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && \
   export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && cd '${BUILD_APP_DIR}' && \
   exec ./usrp_wideband_signal_detection '${CONFIG_NAME}' > /workspace/spectrograms/demo_app.log 2>&1"

echo "==> waiting for DPDK to arm (RX flows)"
for i in $(seq 1 60); do
  if sudo docker exec "${CONTAINER_NAME}" bash -lc "grep -q 'Adding RX flow' /workspace/spectrograms/demo_app.log 2>/dev/null"; then
    break
  fi
  if ! sudo docker exec "${CONTAINER_NAME}" bash -lc 'ps -eo comm | grep -q "^usrp_wideband"'; then
    echo "App exited during startup -- last log lines:" >&2
    sudo docker exec "${CONTAINER_NAME}" bash -lc 'tail -25 /workspace/spectrograms/demo_app.log' >&2 || true
    exit 1
  fi
  sleep 1
done

echo "==> [5/7] classify-only AMC daemon (THROUGHPUT + COMPUTE panels, no decode)"
(cd "${APP_DIR}/infocom_evals/pycodec_e2e" && \
 PYCODEC_ROOT="${V3_USER_HOME}/Documents/holoscan_waveform_generation" \
 "${VENV_PY}" rt_decode_daemon.py --snips /tmp/usrp_spectrograms/snippets \
   --classify-only --no-band-truth \
   --metrics-out /tmp/usrp_spectrograms/rt_metrics.json \
   > /tmp/usrp_spectrograms/daemon_live.log 2>&1 &)

echo "==> [6/7] conductor (--loopback: detector dropdown -> loopback config map; we drive the pcap)"
sudo bash -c "cd '${APP_DIR}/infocom_evals/pycodec_e2e' && \
  (python3 demo_conductor.py --loopback --display '${DEMO_DISPLAY}' \
   --rate-hz ${RATE_HZ} --center-hz ${CENTER_HZ} \
   > /tmp/usrp_spectrograms/conductor.log 2>&1 &)"

echo "==> [7/7] snippet janitor (keep 5 min) + tcpreplay loop"
sudo bash -c "(while true; do \
  find /tmp/usrp_spectrograms/snippets -name 'snip_pack*' -mmin +5 -delete 2>/dev/null; \
  sleep 30; done > /dev/null 2>&1 &)"

echo "==> Looping ${PCAP} on ${SENDER_IFACE} @ ${PPS} pps (Ctrl-C stops everything)"
sudo tcpreplay --preload-pcap --loop 0 --pps "${PPS}" -i "${SENDER_IFACE}" "${PCAP_PATH}" &
RADIO_PID=$!
wait "${RADIO_PID}" || true

#!/usr/bin/env bash
# One-command loopback replay demo: dashboard + detector + snipper + AMC decode
# daemon + replay conductor, no USRP needed (QSFP between the Spark's own two
# CX7 ports). See notes/no_radio_loopback_demo.md for the anatomy.
#
# Run WITHOUT sudo (it sudo's the pieces that need it; xhost must run as you):
#     ./bash_scripts/run_loopback_demo.sh
# Stop everything:
#     ./bash_scripts/stop_loopback_demo.sh
#
# Knobs (env): SENDER_IFACE (default enp1s0f1np1), DEMO_DISPLAY (default $DISPLAY or :1)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=container_env.sh
source "${SCRIPT_DIR}/container_env.sh"

SENDER_IFACE="${SENDER_IFACE:-enp1s0f1np1}"
DEMO_DISPLAY="${DEMO_DISPLAY:-${DISPLAY:-:1}}"
RATE_HZ=245760000
CENTER_HZ=2400000000
BUILD_APP_DIR="/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection"
VENV_PY="${HOME}/Documents/holoscan_waveform_generation/.venv-ml/bin/python"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this WITHOUT sudo — xhost must run as the desktop user (the script sudo's what needs root)." >&2
  exit 1
fi

echo "==> [1/7] X access for the container window (resets on login/lock)"
DISPLAY="${DEMO_DISPLAY}" xhost +local: >/dev/null

echo "==> [2/7] sender port ${SENDER_IFACE} up @ MTU 9000 (replay frames are ~4.2 KB)"
sudo ip link set "${SENDER_IFACE}" up mtu 9000

echo "==> [3/7] scratch dirs + demo control state (fresh snippet dir for clean accounting)"
sudo mkdir -p /tmp/usrp_spectrograms && sudo chmod 1777 /tmp/usrp_spectrograms
sudo rm -rf /tmp/usrp_spectrograms/snippets
printf '{"gate": "tprime", "snr": "clean", "detector": "coherent_power", "seq": 1}\n' \
  | sudo tee /tmp/usrp_spectrograms/demo_control.json > /dev/null
sudo chmod 666 /tmp/usrp_spectrograms/demo_control.json

echo "==> [4/7] app (viz + snipper) on ${DEMO_DISPLAY}"
sudo docker exec "${CONTAINER_NAME}" bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true"
sleep 2
sudo docker exec "${CONTAINER_NAME}" bash -lc \
  "cp '/workspace/holohub/applications/usrp_wideband_signal_detection/config_snipper_viz_demo.yaml' '${BUILD_APP_DIR}/' 2>/dev/null || true"
sudo docker exec -d -e DISPLAY="${DEMO_DISPLAY}" \
  -e USRP_SAMPLE_RATE_HZ="${RATE_HZ}" -e USRP_CENTER_FREQ_HZ="${CENTER_HZ}" \
  "${CONTAINER_NAME}" bash -lc \
  "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && \
   export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && cd '${BUILD_APP_DIR}' && \
   exec ./usrp_wideband_signal_detection config_snipper_viz_demo.yaml \
   > /workspace/spectrograms/demo_app.log 2>&1"

echo "==> [5/7] replay conductor (SNR pcaps + detector switches from DEMO CONTROLS)"
sudo bash -c "cd '${APP_DIR}/infocom_evals/pycodec_e2e' && \
  (python3 demo_conductor.py --iface '${SENDER_IFACE}' --rate-hz ${RATE_HZ} \
   --display '${DEMO_DISPLAY}' > /tmp/usrp_spectrograms/conductor.log 2>&1 &)"

echo "==> [6/7] AMC decode daemon (LIVE DECODE + CLASSIFIER panels)"
(cd "${APP_DIR}/infocom_evals/pycodec_e2e" && \
 "${VENV_PY}" rt_decode_daemon.py --snips /tmp/usrp_spectrograms/snippets \
   --metrics-out /tmp/usrp_spectrograms/rt_metrics.json \
   > /tmp/usrp_spectrograms/daemon_live.log 2>&1 &)

echo "==> [7/7] snippet janitor (looped replay writes snippets fast; keep 5 min)"
sudo bash -c "(while true; do \
  find /tmp/usrp_spectrograms/snippets -name 'snip_pack*' -mmin +5 -delete 2>/dev/null; \
  sleep 30; done > /dev/null 2>&1 &)"

sleep 10
if DISPLAY="${DEMO_DISPLAY}" xwininfo -root -children 2>/dev/null | grep -q "USRP Wideband Spectrogram"; then
  echo "DEMO UP: window on ${DEMO_DISPLAY}; replay looping the clean composite."
  echo "  logs: /tmp/usrp_spectrograms/{conductor,daemon_live}.log, container demo_app.log"
  echo "  stop: ${SCRIPT_DIR}/stop_loopback_demo.sh"
else
  echo "WARNING: no dashboard window detected — check container demo_app.log and xhost." >&2
  exit 1
fi

#!/usr/bin/env bash
# Persistent SINGLE-channel M5-S DINO loopback demo (ch0 only, udp 1234) over the Spark's QSFP
# loopback cable. Sibling of run_loopback_m5s_dual.sh; used by loopback_channel_supervisor.sh when
# the DEMO CONTROLS "Channels" toggle is set to Single. On single channel the M5-S detector runs
# emit_stride=1 (per-frame) so the ROI ribbon is smooth (no dual striding).
#
#   Launch:  ./bash_scripts/run_loopback_m5s_single.sh
#   Stop:    ./bash_scripts/stop_loopback_m5s_dual.sh   (tears down app + all tcpreplay)
#
# Env knobs: CONFIG_NAME (default config_live_v3_dino_ft_m5s.yaml), PCAP_CH0, PPS (default 480000),
#            RATE_HZ (491520000), CENTER_HZ (2400000000), DEMO_DISPLAY.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_env.sh"
CN="${CONTAINER_NAME:-usrp_x410_sig_det_sat3737}"

SENDER_IFACE="${SENDER_IFACE:-enp1s0f1np1}"
DEMO_DISPLAY="${DEMO_DISPLAY:-${DISPLAY:-:1}}"
CONFIG_NAME="${CONFIG_NAME:-config_live_v3_dino_ft_m5s.yaml}"
RATE_HZ="${RATE_HZ:-491520000}"
CENTER_HZ="${CENTER_HZ:-2400000000}"
PPS="${PPS:-480000}"
COMPOSITES="${COMPOSITES:-/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites}"
PCAP_CH0="${PCAP_CH0:-x410_ota_2g4_gain10_20260908_46f.spark.pcap}"
P0="${COMPOSITES}/${PCAP_CH0}"
BUILD_APP_DIR="/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection"
SOURCE_APP_DIR="/workspace/holohub/applications/usrp_wideband_signal_detection"
JOB=/tmp/usrp_spectrograms

[[ -f "$P0" ]] || { echo "!! missing pcap: $P0"; exit 1; }

echo "==> [1/5] X access + sender iface ${SENDER_IFACE} up @ MTU 9000"
DISPLAY="${DEMO_DISPLAY}" xhost +local: >/dev/null 2>&1 || true
sudo ip link set "${SENDER_IFACE}" up mtu 9000

echo "==> [2/5] stop any stale app + tcpreplay, clean DPDK hugepages"
sudo pkill -f 'tcpreplay --preload-pcap' 2>/dev/null || true  # matches the sudo/setsid/nohup wrapper + child
sudo docker exec "${CN}" bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; \
   for i in \$(seq 1 30); do pgrep -f '(^|/)usrp_wideband_signal_detection( |\$)' >/dev/null || break; sleep 1; done; \
   rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"

echo "==> [3/5] launch M5-S single app (${CONFIG_NAME}) detached on ${DEMO_DISPLAY} @ ${RATE_HZ} Hz"
USRP_OVERLAY="${USRP_OVERLAY:-1}"
USRP_CLASS_COLORS="${USRP_CLASS_COLORS:-0}"   # class-color overlay retired; controls hidden (default off)
USRP_CLASSIFIER_MODELS="${USRP_CLASSIFIER_MODELS:-resnet1d,cnn1d,tprime_lg}"
USRP_SNIP_ADDITIVE_GUARD_HZ="${USRP_SNIP_ADDITIVE_GUARD_HZ:-10e6}"
sudo mkdir -p "${JOB}" && sudo chmod 1777 "${JOB}" 2>/dev/null || true
# Seed the control file only if absent, so a supervisor-driven relaunch preserves the user's live
# detector/channel/color selections (the supervisor owns the file when it is managing the demo).
if [[ ! -f "${JOB}/demo_control.json" ]]; then
  printf '{"gate": "resnet1d", "snr": "clean", "detector": "cuda_dino_finetuned", "classifiers": "resnet1d,cnn1d,tprime_lg", "color_model": "resnet1d", "channel_mode": "single", "seq": 1}\n' \
    | sudo tee "${JOB}/demo_control.json" > /dev/null
  sudo chmod 666 "${JOB}/demo_control.json"
fi
sudo docker exec "${CN}" bash -lc "cp '${SOURCE_APP_DIR}/${CONFIG_NAME}' '${BUILD_APP_DIR}/' 2>/dev/null || true"
sudo docker exec -d -e DISPLAY="${DEMO_DISPLAY}" \
  -e USRP_SAMPLE_RATE_HZ="${RATE_HZ}" -e USRP_CENTER_FREQ_HZ="${CENTER_HZ}" \
  -e USRP_OVERLAY="${USRP_OVERLAY}" -e USRP_CLASS_COLORS="${USRP_CLASS_COLORS}" \
  -e USRP_CLASSIFIER_MODELS="${USRP_CLASSIFIER_MODELS}" \
  -e USRP_SNIP_ADDITIVE_GUARD_HZ="${USRP_SNIP_ADDITIVE_GUARD_HZ}" \
  "${CN}" bash -lc \
  "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && \
   export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && cd '${BUILD_APP_DIR}' && \
   exec ./usrp_wideband_signal_detection '${CONFIG_NAME}' > /workspace/spectrograms/demo_app.log 2>&1"

echo "==> [4/5] waiting up to 90 s for DPDK RX flows"
for i in $(seq 1 90); do
  sudo docker exec "${CN}" bash -lc "grep -q 'Adding RX flow' /workspace/spectrograms/demo_app.log 2>/dev/null" && { echo "  armed"; break; }
  sudo docker exec "${CN}" bash -lc 'ps -eo comm | grep -q "^usrp_wideband"' || { echo "  !! app exited during startup:"; sudo docker exec "${CN}" bash -lc 'tail -25 /workspace/spectrograms/demo_app.log'; exit 1; }
  sleep 1
done
sleep 2

echo "==> [5/5] starting ONE detached tcpreplay loop (ch0 udp1234) @ ${PPS} pps"
sudo setsid nohup tcpreplay --preload-pcap --loop 0 --pps "${PPS}" -i "${SENDER_IFACE}" "${P0}" \
  > "${JOB}/tr_ch0.log" 2>&1 < /dev/null &
sleep 1
echo "==> single M5-S loopback is UP (detached). tcpreplay procs:"
pgrep -a -x tcpreplay | sed 's/^/    /' || echo "    (none yet -- check ${JOB}/tr_ch0.log)"

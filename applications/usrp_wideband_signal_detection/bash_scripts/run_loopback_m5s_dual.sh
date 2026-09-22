#!/usr/bin/env bash
# Persistent DUAL-channel M5-S DINO loopback demo over the Spark's own QSFP loopback cable
# (CX7 sender enp1s0f1np1 -> CX7 data port 0000:01:00.0; NO radio). Replays the 491.52 MSps
# composite on BOTH channels: ch0 (udp 1234) + ch1 (udp 1235, portmap copy) simultaneously on the
# one wire; the DPDK receive-port flows split them. Unlike run_loopback_dual_dino_ft.sh (a 45 s
# self-tearing test on the M3/M4 dual config), this launches the M5-S dual config and STAYS UP:
# the app runs detached in the container and the two tcpreplay loops run detached on the host, so
# the dashboard keeps scrolling until you stop it.
#
#   Launch:  ./bash_scripts/run_loopback_m5s_dual.sh
#   Stop:    ./bash_scripts/stop_loopback_m5s_dual.sh   (or stop_loopback_demo.sh)
#
# Env knobs:
#   CONFIG_NAME  dual config to run     (default config_live_v3_dino_ft_m5s_two_channel.yaml)
#   PCAP_CH0     ch0 pcap basename      (default x410_ota_2g4_gain10_20260908_46f.spark.pcap)
#   PCAP_CH1     ch1 pcap basename      (default x410_ota_ch1_p1235_46f.spark.pcap)
#   PPS          tcpreplay pkt rate/ch  (default 480000 = full 491.52 MSps)
#   RATE_HZ      stream sample rate     (default 491520000)
#   CENTER_HZ    ch0 center label       (default 2400000000)
#   DEMO_DISPLAY X display              (default $DISPLAY or :1)
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
# shellcheck source=container_env.sh
source "${SCRIPT_DIR}/container_env.sh"
CN="${CONTAINER_NAME:-usrp_x410_sig_det_sat3737}"

SENDER_IFACE="${SENDER_IFACE:-enp1s0f1np1}"
DEMO_DISPLAY="${DEMO_DISPLAY:-${DISPLAY:-:1}}"
CONFIG_NAME="${CONFIG_NAME:-config_live_v3_dino_ft_m5s_two_channel.yaml}"
RATE_HZ="${RATE_HZ:-491520000}"
CENTER_HZ="${CENTER_HZ:-2400000000}"
PPS="${PPS:-480000}"
COMPOSITES="${COMPOSITES:-/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites}"
PCAP_CH0="${PCAP_CH0:-x410_ota_2g4_gain10_20260908_46f.spark.pcap}"
PCAP_CH1="${PCAP_CH1:-x410_ota_ch1_p1235_46f.spark.pcap}"
P0="${COMPOSITES}/${PCAP_CH0}"
P1="${COMPOSITES}/${PCAP_CH1}"
BUILD_APP_DIR="/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection"
SOURCE_APP_DIR="/workspace/holohub/applications/usrp_wideband_signal_detection"
JOB=/tmp/usrp_spectrograms

for p in "$P0" "$P1"; do
  [[ -f "$p" ]] || { echo "!! missing pcap: $p"; echo "Available:"; ls "${COMPOSITES}"/*.pcap 2>/dev/null; exit 1; }
done

echo "==> [1/5] X access + sender iface ${SENDER_IFACE} up @ MTU 9000"
DISPLAY="${DEMO_DISPLAY}" xhost +local: >/dev/null 2>&1 || true
sudo ip link set "${SENDER_IFACE}" up mtu 9000

echo "==> [2/5] stop any stale app + tcpreplay, clean DPDK hugepages"
sudo pkill -f 'tcpreplay --preload-pcap' 2>/dev/null || true  # matches the sudo/setsid/nohup wrapper + child
sudo docker exec "${CN}" bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; \
   for i in \$(seq 1 30); do pgrep -f '(^|/)usrp_wideband_signal_detection( |\$)' >/dev/null || break; sleep 1; done; \
   rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"

echo "==> [3/5] launch M5-S dual app (${CONFIG_NAME}) detached on ${DEMO_DISPLAY} @ ${RATE_HZ} Hz"
# v5 dashboard env (mirror run_live_demo.sh v5dual): detection overlay ON, class-colored masks, the
# release_v2 classifier taxonomy (so the panels read QPSK/802.11AX/BLE/5G_NR/BFM), additive snip guard.
USRP_OVERLAY="${USRP_OVERLAY:-1}"
USRP_CLASS_COLORS="${USRP_CLASS_COLORS:-0}"   # class-color overlay retired; controls hidden (default off)
USRP_CLASSIFIER_MODELS="${USRP_CLASSIFIER_MODELS:-resnet1d,cnn1d,tprime_lg}"
USRP_SNIP_ADDITIVE_GUARD_HZ="${USRP_SNIP_ADDITIVE_GUARD_HZ:-10e6}"
# seed the DEMO CONTROLS control file so the checklist/labels paint before the daemon's first write
sudo mkdir -p "${JOB}" && sudo chmod 1777 "${JOB}" 2>/dev/null || true
# Seed the control file only if absent, so a supervisor-driven relaunch preserves the user's live
# detector/channel/color selections (the supervisor owns the file when it is managing the demo).
if [[ ! -f "${JOB}/demo_control.json" ]]; then
  printf '{"gate": "resnet1d", "snr": "clean", "detector": "cuda_dino_finetuned", "classifiers": "resnet1d,cnn1d,tprime_lg", "color_model": "resnet1d", "channel_mode": "dual", "seq": 1}\n' \
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

echo "==> [5/5] starting TWO detached tcpreplay loops (ch0 udp1234 + ch1 udp1235) @ ${PPS} pps each"
sudo setsid nohup tcpreplay --preload-pcap --loop 0 --pps "${PPS}" -i "${SENDER_IFACE}" "${P0}" \
  > "${JOB}/tr_ch0.log" 2>&1 < /dev/null &
sudo setsid nohup tcpreplay --preload-pcap --loop 0 --pps "${PPS}" -i "${SENDER_IFACE}" "${P1}" \
  > "${JOB}/tr_ch1.log" 2>&1 < /dev/null &
sleep 1
echo "==> dual M5-S loopback is UP (detached). tcpreplay procs:"
pgrep -a -x tcpreplay | sed 's/^/    /' || echo "    (none yet -- check ${JOB}/tr_ch*.log)"
echo "    app log:   sudo docker exec ${CN} tail -f /workspace/spectrograms/demo_app.log"
echo "    stop with: ${SCRIPT_DIR}/stop_loopback_m5s_dual.sh"

#!/usr/bin/env bash
# One N320 preliminary live leg through the real wrapper (run_live_demo.sh, direct-config mode).
# Topology + radio settings are explained in README.md next to this script. Needs passwordless sudo.
# Usage: n320_leg.sh <leg> <rate_hz> <channels> <freqs> <dest_ports> [gain] [duration_s]
#   e.g. n320_leg.sh n320_m7s_single 245.76e6 "0" "2440e6" "1234"
set -uo pipefail
LEG=$1 RATE=$2 CHANNELS=$3 FREQS=$4 DEST_PORTS=$5 GAIN=${6:-30} DUR=${7:-60}
APP=/home/genesys-dgx1/Documents/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
CFG=/workspace/holohub/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/n320_live_m7s_prelim/configs/${LEG}.yaml
OUT=/tmp/usrp_spectrograms/n320_live/${LEG}
RATE_INT=$(python3 -c "print(int(float('${RATE}')))")
sudo -n mkdir -p "${OUT}" && sudo -n chmod 1777 /tmp/usrp_spectrograms/n320_live "${OUT}"
sudo -n rm -rf "${OUT}/masks" "${OUT}"/*.log "${OUT}"/*.txt "${OUT}"/*.csv 2>/dev/null
ctr() { sudo -n ethtool -S enP2p1s0f1np1 | grep -E '^\s+(rx_vport_unicast_packets|rx_vport_unicast_bytes|rx_out_of_buffer|rx_discards_phy):' | tr -s ' '; }
ctr > "${OUT}/nic_before.txt"
nvidia-smi --query-gpu=timestamp,utilization.gpu --format=csv,noheader -lms 1000 > "${OUT}/gpu.csv" 2>/dev/null &
GPU_PID=$!
SINGLE_CENTER=""; [[ "${CHANNELS}" != *" "* ]] && SINGLE_CENTER="USRP_CENTER_FREQ_HZ=${FREQS}"
cd "${APP}"
sudo -n env DISPLAY=:1 SKIP_TOPOLOGY_CHECK=1 USRP_OVERLAY=1 \
  USRP_SAMPLE_RATE_HZ="${RATE_INT}" ${SINGLE_CENTER} \
  CTRL_ADDR="192.168.10.2,master_clock_rate=245.76e6" ADAPTER=sfp0 \
  DEST_ADDR=192.168.10.1 DEST_MAC=4c:bb:47:2c:45:18 ANTENNA=TX/RX KEEP_HDR=1 \
  RATE="${RATE}" CHANNELS="${CHANNELS}" FREQS="${FREQS}" DEST_PORTS="${DEST_PORTS}" GAIN="${GAIN}" \
  DURATION="${DUR}" \
  ./bash_scripts/run_live_demo.sh "${CFG}" > "${OUT}/wrapper.log" 2>&1
echo "wrapper exit=$?" >> "${OUT}/wrapper.log"
kill "${GPU_PID}" 2>/dev/null
ctr > "${OUT}/nic_after.txt"
sudo -n docker exec usrp_x410_sig_det_sat3737 cat /tmp/live_demo_app.log > "${OUT}/app.log" 2>/dev/null
echo "== ${LEG}: NIC deltas on enP2p1s0f1np1 (DPDK PF) (radio stream ${DUR}s)"
paste "${OUT}/nic_before.txt" "${OUT}/nic_after.txt" | awk '{printf "  %-20s %14d\n", $1, $4-$2}'
echo "== wrapper tail"; grep -vE '^\s*$' "${OUT}/wrapper.log" | grep -E '==>|exit=|Error|error|Traceback|RX antenna|sampling rate|Stream started' | tail -14
echo "== results in ${OUT}"

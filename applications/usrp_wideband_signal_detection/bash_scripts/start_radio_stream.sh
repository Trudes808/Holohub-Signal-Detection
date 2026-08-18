#!/usr/bin/env bash
set -euo pipefail

# Radio-control / OTA collection step for the live pipeline (DGX Spark bench defaults).
#
# This commands the X410 to tune its RF frontends, receive OVER THE AIR through the
# antennas, and stream the received IQ samples out its data QSFP (sfp0) into the DPDK
# data port the Holoscan app is listening on. It is NOT a synthetic/demo signal source —
# it IS the real data collection. It is a separate process by design: the Holoscan app is
# a pure DPDK data-plane consumer and cannot own the UHD/RFNoC control session (control
# rides the kernel-owned QSFP, sfp1).
#
# Run the app first (its window opens and waits), then this script; the visualization
# comes alive as soon as samples flow. Ctrl-C stops the radio.
#
# Defaults = the validated dual-channel bench setup. Override via env, e.g.:
#   single channel:  CHANNELS="0" FREQS="2400e6" DEST_PORTS="1234" ./start_radio_stream.sh
#   other centers:   FREQS="915e6 1800e6" ./start_radio_stream.sh
#   timed run:       DURATION=60 ./start_radio_stream.sh
# Note: this X410's CG_400 image runs at a fixed 491.52 Msps regardless of --rate.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

FREQS=${FREQS:-"2400e6 1000e6"}          # per-channel RF center frequencies
CHANNELS=${CHANNELS:-"0 1"}              # X410 RF channels (antenna ports)
RATE=${RATE:-500e6}                      # requested rate (CG_400 negotiates 491.52 Msps)
GAIN=${GAIN:-30}                         # RX gain in dB
DEST_PORTS=${DEST_PORTS:-"1234 1235"}    # one UDP dst port per channel (must match config flows)
DEST_ADDR=${DEST_ADDR:-192.168.10.1}     # host data-port IP (enp1s0f0np0)
DEST_MAC=${DEST_MAC:-4c:bb:47:2c:45:13}  # host data-port MAC (enp1s0f0np0)
CTRL_ADDR=${CTRL_ADDR:-192.168.21.2}     # X410 control address (sfp1, kernel link)
ADAPTER=${ADAPTER:-sfp0}                 # X410 QSFP the data exits (the data link)
SPP=${SPP:-1024}                         # samples per packet (must match chdr_converter)
DURATION=${DURATION:-}                   # empty = stream until Ctrl-C

cd "${SCRIPT_DIR}/../../usrp_freq_detection"

# UHD's python bindings install to /usr/local site-packages, which root's
# python3 does not search — without this, the sudo'd run_live_demo.sh path
# dies with "No module named 'uhd'" while a user shell works fine.
export PYTHONPATH="/usr/local/lib/python3.12/site-packages${PYTHONPATH:+:${PYTHONPATH}}"

# shellcheck disable=SC2086  # word-splitting of the multi-value vars is intentional
exec python3 rx_to_remote_udp.py \
  --args "addr=${CTRL_ADDR}" \
  --freq ${FREQS} \
  --rate "${RATE}" \
  --gain "${GAIN}" \
  --channels ${CHANNELS} \
  --adapter "${ADAPTER}" \
  --dest-addr "${DEST_ADDR}" \
  --dest-port ${DEST_PORTS} \
  --dest-mac-addr "${DEST_MAC}" \
  --spp "${SPP}" \
  ${DURATION:+--duration "${DURATION}"}

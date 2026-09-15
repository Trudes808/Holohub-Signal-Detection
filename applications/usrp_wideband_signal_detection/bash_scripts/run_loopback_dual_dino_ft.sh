#!/usr/bin/env bash
# Dual-channel loopback over ONE wire: replay ch0 (udp 1234) + ch1 (udp 1235) simultaneously on the
# single sender port; the DPDK receive-port flows split them into ch0/ch1. Validates that two 491.52 MSps
# streams through the full v3 DINO-FT M3 pipeline keep up (partial_drops, panic_resets) and mask BOTH
# channels. No conductor/daemon (pure detector+viz+snipper load).
set -uo pipefail
#
# PREREQUISITE (once): make the ch1 pcap by remapping the UDP port 1234->1235 on the ch0 pcap:
#   tcprewrite --portmap=1234:1235 \
#     --infile=<composites>/x410_ota_2g4_gain10_20260908.spark.pcap \
#     --outfile=<composites>/x410_ota_ch1_p1235.spark.pcap
# Both streams share the one loopback wire; the DPDK receive-port flows split them by udp_dst
# (1234->ch0, 1235->ch1). Aggregate 2x491.52 MSps sc16 ~= 31 Gbps, well under the 100GbE link.
#
APP=/home/genesys-dgx1/Documents/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
JOB=/tmp/usrp_spectrograms/dual_loop_logs
COMP=/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites
mkdir -p "$JOB" 2>/dev/null; cd "$APP"; source bash_scripts/container_env.sh 2>/dev/null
CN=${CONTAINER_NAME:-usrp_x410_sig_det_sat3737}
SENDER=${SENDER_IFACE:-enp1s0f1np1}
BT=/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
CFG=config_live_v3_dino_ft_two_channel.yaml
APPLOG=/tmp/usrp_spectrograms/demo_app.log
DUMP=/tmp/usrp_spectrograms/dual_loop/dino
PPS=${PPS:-480000}

sudo rm -rf "$DUMP" 2>/dev/null; sudo rm -f "$APPLOG" 2>/dev/null
echo "==> X access + sender iface up (mtu 9000)"
DISPLAY=:1 xhost +local: >/dev/null 2>&1 || true
sudo ip link set "$SENDER" up mtu 9000

echo "==> stop any stale app + launch dual DINO-FT M3 app (DISPLAY=:1, 491.52)"
sudo docker exec "$CN" bash -lc "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; sleep 2; rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"
sudo docker exec -d -e DISPLAY=:1 -e USRP_SAMPLE_RATE_HZ=491520000 "$CN" bash -lc \
  "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && cd '$BT' && exec ./usrp_wideband_signal_detection '$CFG' > /workspace/spectrograms/demo_app.log 2>&1"

echo "==> waiting for DPDK RX flows (up to 90s)"
for i in $(seq 1 90); do
  sudo docker exec "$CN" bash -lc "grep -q 'Adding RX flow' /workspace/spectrograms/demo_app.log 2>/dev/null" && { echo "  armed"; break; }
  sudo docker exec "$CN" bash -lc 'ps -eo comm | grep -q "^usrp_wideband"' || { echo "  !! app exited:"; sudo docker exec "$CN" bash -lc 'tail -20 /workspace/spectrograms/demo_app.log'; exit 1; }
  sleep 1
done
sleep 3
echo "==> starting TWO tcpreplay streams: ch0 (1234) + ch1 (1235) @ ${PPS} pps each"
sudo tcpreplay --preload-pcap --loop 0 --pps "$PPS" -i "$SENDER" "$COMP/x410_ota_2g4_gain10_20260908.spark.pcap" > "$JOB/tr_ch0.log" 2>&1 &
TR0=$!
sudo tcpreplay --preload-pcap --loop 0 --pps "$PPS" -i "$SENDER" "$COMP/x410_ota_ch1_p1235.spark.pcap" > "$JOB/tr_ch1.log" 2>&1 &
TR1=$!
echo "  tcpreplay ch0 pid $TR0, ch1 pid $TR1"
echo "==> running dual for 45s (accumulating masks + CHDR stats)"; sleep 45

echo "==> per-channel CHDR summary (keeping up? drops?):"
sudo docker exec "$CN" bash -lc "grep -aE 'CHDR summary ch=' /workspace/spectrograms/demo_app.log | tail -8"
echo "==> detector lines:"
sudo docker exec "$CN" bash -lc "grep -aE 'finetuned_dino_detector|FFT ingress live' /workspace/spectrogram*/demo_app.log 2>/dev/null | grep -avE 'downsample: FFT' | tail -6; grep -aE 'robust normalization ON' /workspace/spectrograms/demo_app.log | head -2"

echo "==> teardown"
sudo kill "$TR0" "$TR1" 2>/dev/null || true
sudo pkill -f '^tcpreplay' 2>/dev/null || true
sudo docker exec "$CN" bash -lc "pkill -9 -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true; echo cleaned"
sleep 2
sudo cp "$APPLOG" "$JOB/dual_demo_app.log" 2>/dev/null || true
sudo chown -R "$(id -u):$(id -g)" "$DUMP" "$JOB/dual_demo_app.log" 2>/dev/null || true
echo "==> dumped masks ch0: $(ls "$DUMP"/mask_ch0_*.npy 2>/dev/null | wc -l)  ch1: $(ls "$DUMP"/mask_ch1_*.npy 2>/dev/null | wc -l)"
echo "DUAL_LOOP_DONE"

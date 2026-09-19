#!/usr/bin/env bash
# Capture the class-color overlay live: run the v3 loopback pipeline (aligned pcap, 491.52) with a given
# CONFIG + optional classifier daemon, wait for the overlay to populate, screenshot the GUI, snapshot
# metrics. Env: CONFIG_NAME, DAEMON=1|0, LABEL, PCAP, RATE_HZ, PPS.
set -uo pipefail
APP=/home/genesys-dgx1/Documents/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
SP=/tmp/claude-1000/-home-genesys-dgx1-Documents-Holohub-Signal-Detection/323ea032-eab1-4437-b121-04a515160984/scratchpad
COMP=/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites
cd "$APP"; source bash_scripts/container_env.sh 2>/dev/null
CN=${CONTAINER_NAME:-usrp_x410_sig_det_sat3737}
SENDER=${SENDER_IFACE:-enp1s0f1np1}
BT=/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
VENV_PY=/home/genesys-dgx1/Documents/holoscan_waveform_generation/.venv-ml/bin/python
CONFIG_NAME=${CONFIG_NAME:?set CONFIG_NAME}
DAEMON=${DAEMON:-1}
LABEL=${LABEL:-cap}
PCAP=${PCAP:-x410_ota_2g4_gain10_20260908_46f.spark.pcap}
RATE_HZ=${RATE_HZ:-491520000}; PPS=${PPS:-480000}; CENTER_HZ=2400000000
OUT="$SP/shots"; mkdir -p "$OUT"
DISP=:1

echo "=== [$LABEL] config=$CONFIG_NAME daemon=$DAEMON pcap=$PCAP ==="
# clean state
sudo pkill -f '^tcpreplay' 2>/dev/null || true
sudo pkill -f 'rt_decode_daemon' 2>/dev/null || true
sudo docker exec "$CN" bash -lc "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; sleep 2; rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"
sudo rm -rf /tmp/usrp_spectrograms/snippets 2>/dev/null || true
sudo rm -f /tmp/usrp_spectrograms/rt_metrics.json /tmp/usrp_spectrograms/daemon_live.log 2>/dev/null || true
sudo mkdir -p /tmp/usrp_spectrograms && sudo chmod 1777 /tmp/usrp_spectrograms
printf '{"gate": "tprime", "snr": "clean", "detector": "coherent_power", "classifiers": "vtcnn2,resnet1d,tprime", "seq": 1}\n' | sudo tee /tmp/usrp_spectrograms/demo_control.json >/dev/null
sudo chmod 666 /tmp/usrp_spectrograms/demo_control.json
DISPLAY=$DISP xhost +local: >/dev/null 2>&1 || true
sudo ip link set "$SENDER" up mtu 9000

echo "==> sync config + launch app"
sudo docker exec "$CN" bash -lc "cp '/workspace/holohub/applications/usrp_wideband_signal_detection/$CONFIG_NAME' '$BT/'"
sudo docker exec -d -e DISPLAY=$DISP -e USRP_SAMPLE_RATE_HZ=$RATE_HZ -e USRP_CENTER_FREQ_HZ=$CENTER_HZ "$CN" bash -lc \
  "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && cd '$BT' && exec ./usrp_wideband_signal_detection '$CONFIG_NAME' > /workspace/spectrograms/demo_app.log 2>&1"
for i in $(seq 1 90); do
  sudo docker exec "$CN" bash -lc "grep -q 'Adding RX flow' /workspace/spectrograms/demo_app.log 2>/dev/null" && { echo "  armed"; break; }
  sudo docker exec "$CN" bash -lc 'ps -eo comm | grep -q "^usrp_wideband"' || { echo "  !! app exited:"; sudo docker exec "$CN" bash -lc 'tail -20 /workspace/spectrograms/demo_app.log'; exit 1; }
  sleep 1
done

if [ "$DAEMON" = "1" ]; then
  echo "==> start classify-only AMC daemon"
  (cd "$APP/infocom_evals/pycodec_e2e" && PYCODEC_ROOT=/home/genesys-dgx1/Documents/holoscan_waveform_generation \
    "$VENV_PY" rt_decode_daemon.py --snips /tmp/usrp_spectrograms/snippets --classify-only --no-band-truth \
    --metrics-out /tmp/usrp_spectrograms/rt_metrics.json > /tmp/usrp_spectrograms/daemon_live.log 2>&1 &)
fi

echo "==> tcpreplay $PCAP @ $PPS pps --loop 0"
sudo tcpreplay --preload-pcap --loop 0 --pps "$PPS" -i "$SENDER" "$COMP/$PCAP" > "$SP/tr_$LABEL.log" 2>&1 &
TR=$!

echo "==> wait for signal + (if daemon) class markers (up to 90s)"
for i in $(seq 1 45); do
  sleep 2
  n=$($VENV_PY -c "import json,sys;
try:
  d=json.load(open('/tmp/usrp_spectrograms/rt_metrics.json'));
  print(len(d.get('recent_decodes',[])))
except Exception: print(0)" 2>/dev/null)
  [ "$DAEMON" = "1" ] && [ "${n:-0}" -ge 2 ] && { echo "  markers=$n at ${i}x2s"; break; }
  [ "$DAEMON" = "0" ] && [ "$i" -ge 12 ] && { echo "  (no-daemon warmup done)"; break; }
done

echo "==> screenshots"
for k in 1 2 3; do
  DISPLAY=$DISP gnome-screenshot -f "$OUT/${LABEL}_${k}.png" 2>/dev/null || \
    { xwd -root -display $DISP -silent 2>/dev/null > "$OUT/${LABEL}_${k}.xwd"; echo "  (used xwd fallback)"; }
  sleep 4
done
ls -la "$OUT"/${LABEL}_* 2>/dev/null

echo "==> metrics snapshot"
sudo cp /tmp/usrp_spectrograms/rt_metrics.json "$SP/rtm_$LABEL.json" 2>/dev/null || echo "  (no rt_metrics.json)"
sudo chown "$(id -u):$(id -g)" "$SP/rtm_$LABEL.json" 2>/dev/null || true
$VENV_PY -c "import json;
try:
  d=json.load(open('$SP/rtm_$LABEL.json'));
  r=d.get('recent_decodes',[]);
  from collections import Counter; c=Counter(x['mod'] for x in r)
  print('  recent_decodes:',len(r),'classes:',dict(c))
except Exception as e: print('  metrics:',e)"
echo "==> CHDR health:"; sudo docker exec "$CN" bash -lc "grep -aE 'CHDR summary ch=0' /workspace/spectrograms/demo_app.log | tail -1"

echo "==> teardown"
sudo kill "$TR" 2>/dev/null; sudo pkill -f '^tcpreplay' 2>/dev/null || true
sudo pkill -f 'rt_decode_daemon' 2>/dev/null || true
sudo docker exec "$CN" bash -lc "pkill -9 -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"
sleep 2
echo "CAPTURE_DONE_$LABEL"

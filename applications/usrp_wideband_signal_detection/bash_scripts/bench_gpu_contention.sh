#!/usr/bin/env bash
# No-hardware GPU-contention benchmark: N concurrent offline replays of the real
# fft->detector chain at live dual frame geometry (512x20480, per-frame detection, dual
# detector profile), using the offline binary's loop-preload mode (zero per-frame host I/O).
# Metric: per-instance and aggregate frames/s vs the live real-time bar (46.9 f/s per channel;
# dual stride-1 needs ~94 aggregate). Use nsys per-kernel sums as the primary ranking signal;
# this aggregate is the end-to-end sanity check (the offline graph carries ~20 ms/frame of
# serial host overhead the live app does not, so absolute f/s undershoots live).
#
# One-time input setup (hardlink + 491.52 Msps relabel of the staged frozen capture):
#   sudo mkdir -p /tmp/usrp_spectrograms/offline_inputs/bench491
#   sudo ln -f /tmp/usrp_spectrograms/offline_inputs/comprehensive_ordered/comprehensive_ordered.sigmf-data \
#        /tmp/usrp_spectrograms/offline_inputs/bench491/bench491.sigmf-data
#   sudo python3 -c "import json;m=json.load(open('/tmp/usrp_spectrograms/offline_inputs/comprehensive_ordered/comprehensive_ordered.sigmf-meta'));m['global']['core:sample_rate']=491520000.0;json.dump(m,open('/tmp/usrp_spectrograms/offline_inputs/bench491/bench491.sigmf-meta','w'))"
#
# Usage: sudo bash_scripts/bench_gpu_contention.sh <tag> [instances=2]
set -euo pipefail
TAG=${1:?tag}
N=${2:-2}
CONTAINER=usrp_x410_sig_det_sat3737
BIN=/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/run_offline_cuda_detector_eval
CFG=/workspace/holohub/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/gpu_bench/config_bench491_gpu_contention.yaml

PIDS=()
for i in $(seq 1 "$N"); do
  docker exec ${CONTAINER} bash -lc "
    rm -rf /workspace/spectrograms/bench491_out_${TAG}_${i}
    start=\$(date +%s.%N)
    ${BIN} --config ${CFG} \
      --input-file /workspace/spectrograms/offline_inputs/bench491/bench491.sigmf-data \
      --output-root /workspace/spectrograms/bench491_out_${TAG}_${i} \
      --detector coherent_power > /tmp/bench_${TAG}_${i}.log 2>&1
    end=\$(date +%s.%N)
    echo \"instance ${i} wall_s=\$(echo \"\$end \$start\" | awk '{printf \"%.2f\", \$1-\$2}')\" >> /tmp/bench_${TAG}_${i}.log" &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done

echo "== bench ${TAG} (${N} instances) =="
TOTAL_FPS=0
for i in $(seq 1 "$N"); do
  LOG=/tmp/bench_${TAG}_${i}.log
  FRAMES=$(docker exec ${CONTAINER} bash -lc "grep -oE 'expected [0-9]+ real frames' ${LOG} | grep -oE '[0-9]+' | head -1 || true")
  WALL=$(docker exec ${CONTAINER} bash -lc "grep -oE 'wall_s=[0-9.]+' ${LOG} | cut -d= -f2")
  # Wall time includes ~app graph setup + 12 GB file read; frame-rate from the frame-processing
  # window: first to last 'processed frame' progress lines when present, else wall.
  FIRST=$(docker exec ${CONTAINER} bash -lc "grep -m1 -oE 'frame [0-9]+/[0-9]+' ${LOG} || true")
  FPS=$(echo "$FRAMES $WALL" | awk '{if ($2>0) printf "%.1f", $1/$2; else print 0}')
  echo "  instance $i: frames=${FRAMES} wall=${WALL}s -> ${FPS} f/s (${FIRST:-no-progress-lines})"
  TOTAL_FPS=$(echo "$TOTAL_FPS $FPS" | awk '{printf "%.1f", $1+$2}')
done
echo "  aggregate: ${TOTAL_FPS} f/s   (real-time bar: 46.9 f/s per live channel; dual stride-1 needs ~94 aggregate)"

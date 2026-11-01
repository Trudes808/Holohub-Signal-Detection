#!/usr/bin/env bash
# One loopback leg: launch the app on a leg config with the emitted-mask dump enabled, replay a
# frame-aligned CHDR pcap over the QSFP loopback at a given packet rate, wait for the pipeline to
# drain, stop the app, and leave everything under one results directory:
#   <OUT>/<run_name>/app.log            full app log (DPDK/CHDR/FFT/detector/viz counters)
#   <OUT>/<run_name>/masks/             mask_ch0_*.npy + mask_dump_manifest.csv (emitted masks)
#   <OUT>/<run_name>/gpu.csv            nvidia-smi utilization/power samples during the replay
#   <OUT>/<run_name>/tcpreplay.log      sender statistics (packets, duration, achieved pps)
#   <OUT>/<run_name>/leg_meta.json      what was run
#
#   ./run_leg.sh <run_name> <leg> <rate_hz> <pps> <loops> <pcap>
#     leg   = coherent | dinoft_native_m2dr | dinoft_rt_m5s | dinoft_native_m5s | dinoft_rt_m2dr
#             (configs/config_lb_<leg>.yaml)
#     e.g.  ./run_leg.sh rt_m5s_245 dinoft_rt_m5s 245760000 240000 1 $COMP/capture_snr_p24dB_171f.spark.pcap
#           ./run_leg.sh rt_m5s_491 dinoft_rt_m5s 491520000 480000 4 $COMP/capture_snr_p24dB_171f.spark.pcap
# Env: OUT (/tmp/usrp_spectrograms/rt_stress), SENDER_IFACE (enP2p1s0f0np0: NIC-internal loopback to the DPDK PF; a cabled loopback would use enp1s0f1np1), DEMO_DISPLAY (:1),
#      CENTER_HZ (2400000000), MASK_MAX_FRAMES (0 = all), DRAIN_S (max seconds to wait after replay),
#      APP_CORES (0-17: affinity of the app's floating threads; DPDK lcores 9/10 pin themselves),
#      SENDER_CORES (18,19: cores the tcpreplay sender is pinned to, normal priority),
#      DINO_DEBUG_STALL_FRAME / DINO_DEBUG_STALL_MS (test only: make the DINO detector sleep on one frame),
#      CHDR_DEBUG_FATAL_AT_BATCH (test only: CHDR converter simulates a fatal CUDA error at batch N),
#      NSYS_CAPTURE_S / NSYS_DELAY_S (profile a steady-state window with Nsight Systems -> <run>/profile.nsys-rep),
#      SENDER_EXTRA (extra tcpreplay flags, e.g. "--pps-multi=32" to send packet bursts per pacing tick),
#      MASK_DUMP (1 = dump every emitted mask for scoring [default]; 0 = production-like, no dump: the
#      dump's per-frame D2H + disk write is itself load).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP=$(cd "${HERE}/../../.." && pwd -P)
source "${APP}/bash_scripts/container_env.sh"
CN="${CONTAINER_NAME}"

RUN=${1:?run_name}; LEG=${2:?leg}; RATE_HZ=${3:?rate_hz}; PPS=${4:?pps}; LOOPS=${5:?loops}; PCAP=${6:?pcap}
OUT=${OUT:-/tmp/usrp_spectrograms/rt_stress}
SENDER_IFACE=${SENDER_IFACE:-enP2p1s0f0np0}   # PF 0002:01:00.0 = physical port p0, switched in-NIC to the DPDK PF 0000:01:00.0
DEMO_DISPLAY=${DEMO_DISPLAY:-:1}
CENTER_HZ=${CENTER_HZ:-2400000000}
MASK_MAX_FRAMES=${MASK_MAX_FRAMES:-0}
DRAIN_S=${DRAIN_S:-40}
BUILD_APP_DIR=/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
# Host checkout is bind-mounted at /workspace/holohub; the leg config is referenced by its container path.
CFG_HOST="${HERE}/configs/config_lb_${LEG}.yaml"
CFG_CONT="/workspace/holohub/${CFG_HOST#$(cd "${APP}/../.." && pwd -P)/}"
[[ -f "${CFG_HOST}" ]] || { echo "!! no config ${CFG_HOST}"; exit 1; }
[[ -f "${PCAP}" ]] || { echo "!! no pcap ${PCAP}"; exit 1; }

HOST_RUN="${OUT}/${RUN}"; CONT_RUN="/workspace/spectrograms/rt_stress/${RUN}"
sudo rm -rf "${HOST_RUN}"; sudo mkdir -p "${HOST_RUN}/masks"; sudo chmod -R 1777 "${HOST_RUN}"

echo "==> [${RUN}] leg=${LEG} rate=${RATE_HZ} pps=${PPS} loops=${LOOPS} pcap=$(basename "${PCAP}")"
DISPLAY="${DEMO_DISPLAY}" xhost +local: >/dev/null 2>&1 || true
sudo ip link set "${SENDER_IFACE}" up mtu 9000

# stale app / replay / DPDK locks
sudo pkill -f 'tcpreplay --preload-pcap' 2>/dev/null || true
sudo docker exec "${CN}" bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; \
   for i in \$(seq 1 30); do pgrep -f '(^|/)usrp_wideband_signal_detection( |\$)' >/dev/null || break; sleep 1; done; \
   rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"
# demo-control file so the DEMO CONTROLS panel paints like the demo (no conductor is running)
[[ -f /tmp/usrp_spectrograms/demo_control.json ]] || printf '{"gate": "resnet1d", "snr": "clean", "detector": "cuda_dino_finetuned", "channel_mode": "single", "seq": 1}\n' | sudo tee /tmp/usrp_spectrograms/demo_control.json >/dev/null

# Optional Nsight Systems capture of a steady-state window (NSYS_CAPTURE_S seconds starting
# NSYS_DELAY_S seconds into the replay): the app runs under `nsys launch`, collection is started/stopped
# from outside so startup and drain are excluded. Report: <run>/profile.nsys-rep.
# NSYS_MODE=profile (default) uses `nsys profile --delay=NSYS_ABS_DELAY_S` counted from app launch
# (make the replay long enough with LOOPS); NSYS_MODE=launch uses an nsys session started/stopped
# around the replay (failed on GB10 2026-10-21: matxOutOfMemory in the converter's first allocation).
NSYS_PREFIX=""
NSYS_MODE=${NSYS_MODE:-profile}
if [[ -n "${NSYS_CAPTURE_S:-}" && "${NSYS_MODE}" == "launch" ]]; then
  NSYS_PREFIX="nsys launch --session-new=rtprof --trace=cuda,nvtx"
elif [[ -n "${NSYS_CAPTURE_S:-}" ]]; then
  NSYS_PREFIX="nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --kill=none --delay=${NSYS_ABS_DELAY_S:-50} --duration=${NSYS_CAPTURE_S} --output=${CONT_RUN}/profile --force-overwrite=true"
fi
echo "==> launching app (${CFG_CONT})"
T_LAUNCH=$(date +%s)
sudo docker exec -d -e DISPLAY="${DEMO_DISPLAY}" \
  -e USRP_SAMPLE_RATE_HZ="${RATE_HZ}" -e USRP_CENTER_FREQ_HZ="${CENTER_HZ}" \
  -e USRP_MASK_DUMP_DIR="$([[ "${MASK_DUMP:-1}" == "1" ]] && echo "${CONT_RUN}/masks")" \
  -e USRP_OVERLAY=1 -e USRP_CLASS_COLORS=0 -e USRP_CLASSIFIER_MODELS="resnet1d,cnn1d,tprime_lg" \
  -e USRP_SNIP_ADDITIVE_GUARD_HZ=10e6 \
  -e DINO_DEBUG_STALL_FRAME="${DINO_DEBUG_STALL_FRAME:-}" -e DINO_DEBUG_STALL_MS="${DINO_DEBUG_STALL_MS:-}" \
  -e CHDR_DEBUG_FATAL_AT_BATCH="${CHDR_DEBUG_FATAL_AT_BATCH:-}" \
  "${CN}" bash -lc \
  "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && \
   cd '${BUILD_APP_DIR}' && taskset -c '${APP_CORES:-0-17}' ${NSYS_PREFIX} ./usrp_wideband_signal_detection '${CFG_CONT}' > '${CONT_RUN}/app.log' 2>&1; \
   echo \"[run_leg] app exit status=\$?\" >> '${CONT_RUN}/app.log'"

echo "==> waiting for DPDK RX flows + detector ready"
armed=0
for i in $(seq 1 150); do
  if sudo grep -q 'Adding RX flow' "${HOST_RUN}/app.log" 2>/dev/null; then armed=1; break; fi
  if ! sudo docker exec "${CN}" bash -lc 'ps -eo comm | grep -qE "^(usrp_wideband|nsys)"'; then
    echo "!! app exited during startup:"; sudo tail -30 "${HOST_RUN}/app.log"; exit 1; fi
  sleep 1
done
[[ ${armed} -eq 1 ]] || { echo "!! DPDK never armed"; sudo tail -30 "${HOST_RUN}/app.log"; exit 1; }
if [[ "${LEG}" == dinoft* ]]; then
  for i in $(seq 1 60); do sudo grep -q 'warmup done' "${HOST_RUN}/app.log" 2>/dev/null && break; sleep 1; done
fi
sudo grep -q 'Emitted-mask dump ENABLED' "${HOST_RUN}/app.log" || echo "!! WARNING: mask dump not enabled in app log"
sleep 4

echo "==> replaying $(date +%T) ($(( $(date +%s) - T_LAUNCH )) s after app launch)"
( nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw --format=csv,noheader -lms 250 > "${HOST_RUN}/gpu.csv" 2>/dev/null ) &
GPU_PID=$!
T_START=$(date +%s.%N)
# CPU isolation (measured 2026-10-21): unpinned, the app's host threads slowed the sender to ~400 kpps
# at a 480 kpps setpoint (alone it holds 480 kpps exactly); a SCHED_FIFO sender instead STARVED
# whichever app thread the scheduler had placed on its core (legs collapsed). So: normal priority,
# sender on SENDER_CORES, and the app launched with its affinity restricted to APP_CORES (above).
sudo taskset -c "${SENDER_CORES:-18,19}" tcpreplay --preload-pcap --loop "${LOOPS}" --pps "${PPS}" ${SENDER_EXTRA:-} -i "${SENDER_IFACE}" "${PCAP}" > "${HOST_RUN}/tcpreplay.log" 2>&1 &
TR_PID=$!
if [[ -n "${NSYS_CAPTURE_S:-}" && "${NSYS_MODE}" == "launch" ]]; then
  sleep "${NSYS_DELAY_S:-3}"
  sudo docker exec "${CN}" bash -lc "nsys start --session=rtprof --sample=none --cpuctxsw=none --output='${CONT_RUN}/profile' --force-overwrite=true" \
    && echo "    nsys capture started (${NSYS_CAPTURE_S}s)"
  sleep "${NSYS_CAPTURE_S}"
  sudo docker exec "${CN}" bash -lc "nsys stop --session=rtprof" >/dev/null 2>&1 && echo "    nsys capture stopped"
fi
wait "${TR_PID}"
TR_RC=$?
T_END=$(date +%s.%N)
echo "    tcpreplay exit=${TR_RC}: $(grep -E 'Actual|Rated|Statistics|Successful|Failed' "${HOST_RUN}/tcpreplay.log" | tr '\n' ' ')"

echo "==> draining (manifest count must hold still 4 s; max ${DRAIN_S}s)"
prev=-1; still=0
for i in $(seq 1 "${DRAIN_S}"); do
  n=$(sudo wc -l < "${HOST_RUN}/masks/mask_dump_manifest.csv" 2>/dev/null || echo 0)
  if [[ "${n}" == "${prev}" ]]; then still=$((still+1)); else still=0; fi
  prev=${n}
  [[ ${still} -ge 4 ]] && break
  sleep 1
done
kill "${GPU_PID}" 2>/dev/null; wait "${GPU_PID}" 2>/dev/null
echo "    masks dumped: $((prev-1))"

echo "==> stopping app (graceful: final DPDK + CHDR exit reports land in app.log)"
sudo docker exec "${CN}" bash -lc "pkill -INT -f '(^|/)usrp_wideband_signal_detection( |\$)' || true"
for i in $(seq 1 40); do sudo docker exec "${CN}" bash -lc 'ps -eo comm | grep -qE "^(usrp_wideband|nsys)"' || break; sleep 1; done
sudo docker exec "${CN}" bash -lc "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' 2>/dev/null; rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null; true"

cat > "${HOST_RUN}/leg_meta.json" <<JSON
{"run": "${RUN}", "leg": "${LEG}", "config": "${CFG_CONT}", "rate_hz": ${RATE_HZ}, "pps": ${PPS}, "loops": ${LOOPS},
 "pcap": "${PCAP}", "pcap_packets": $(capinfos -c -M "${PCAP}" 2>/dev/null | awk '/Number of packets/{print $NF}' || echo 0),
 "replay_start_epoch": ${T_START}, "replay_end_epoch": ${T_END}, "tcpreplay_rc": ${TR_RC}, "masks_dumped": $((prev-1)),
 "sender_iface": "${SENDER_IFACE}", "center_hz": ${CENTER_HZ}, "host": "$(hostname)", "date": "$(date -Is)"}
JSON
echo "==> [${RUN}] done -> ${HOST_RUN}"

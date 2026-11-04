#!/usr/bin/env bash
set -euo pipefail
# Short OTA snippets from the USRP N320 (serial 3459703) to SigMF (ci16_le), for detector stress tests.
#   ch 0 = RF0 (VERT2450 on TX/RX), ch 1 = RF1 (VERT900 on TX/RX)   -- set with --channel
# Streams to /dev/shm (RAM) then moves the pair to --out-dir with a descriptive name:
#   n320_<label>_c<freqMHz>_r<rateMHz>_g<gain>_ch<ch>.sigmf-{data,meta}
# Usage:
#   ./bash_scripts/capture_n320_sigmf.sh --freq 881.5e6 --rate 25e6 --gain 30 --dur 1 --channel 1 \
#        --label cell850_downlink_869-894MHz --desc "LTE/CDMA downlinks, band 5" [--mcr 200e6|245.76e6|250e6]
RX="${RX_BIN:-/usr/local/lib/uhd/examples/rx_samples_to_file}"
ADDR="${ADDR:-addr=192.168.10.2}"
FREQ=915e6; RATE=25e6; GAIN=30; DUR=1; CH=1; ANT=TX/RX; LABEL=cap; DESC=""; MCR=""
OUT_DIR="${OUT_DIR:-$HOME/Datasets/ARGUS/OOD_captures/OTA}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --freq) FREQ="$2"; shift 2;; --rate) RATE="$2"; shift 2;; --gain) GAIN="$2"; shift 2;; --dur) DUR="$2"; shift 2;;
    --channel) CH="$2"; shift 2;; --ant) ANT="$2"; shift 2;; --label) LABEL="$2"; shift 2;; --desc) DESC="$2"; shift 2;;
    --mcr) MCR="$2"; shift 2;; --out-dir) OUT_DIR="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done
ARGS="${ADDR}"; [ -n "${MCR}" ] && ARGS="${ARGS},master_clock_rate=${MCR}"
FREQ_HZ=$(python3 -c "print(int(float('${FREQ}')))"); RATE_HZ=$(python3 -c "print(float('${RATE}'))")
FTAG=$(python3 -c "print(f'{float(\"${FREQ}\")/1e6:g}'.replace('.','p'))"); RTAG=$(python3 -c "print(f'{float(\"${RATE}\")/1e6:g}'.replace('.','p'))")
ANTMODEL=$([ "${CH}" = "0" ] && echo "VERT2450" || echo "VERT900")
NAME="n320_${LABEL}_c${FTAG}M_r${RTAG}M_g${GAIN}_ch${CH}"
SHM="/dev/shm/${NAME}.sigmf-data"; mkdir -p "${OUT_DIR}"
trap 'rm -f "${SHM}" 2>/dev/null || true' EXIT INT TERM
NEED=$(python3 -c "print(int(float('${DUR}')*float('${RATE}')*4*1.05))"); FREE=$(df -B1 --output=avail /dev/shm | tail -1 | tr -d ' ')
[ "${NEED}" -gt "${FREE}" ] && { echo "!! /dev/shm too small for ${DUR}s @ ${RATE}" >&2; exit 1; }
echo "==> ${NAME}: ${DUR}s @ ${FREQ} Hz, ${RATE} Sps, gain ${GAIN} dB, ch${CH} ${ANT} (${ANTMODEL})"
LOG=$(mktemp)
"${RX}" --args "${ARGS}" --freq "${FREQ}" --rate "${RATE}" --gain "${GAIN}" --duration "${DUR}" \
        --channel "${CH}" --ant "${ANT}" --type short --file "${SHM}" 2>&1 | tee "${LOG}" | grep -iE "Actual RX (Rate|Freq|Gain)|Done|Error|error" || true
OVF=$(grep -c "^O\|OO\| O " "${LOG}" 2>/dev/null || true); ACT_RATE=$(grep -i "Actual RX Rate" "${LOG}" | tail -1 | grep -oE "[0-9.]+ Msps" | head -1)
ACT_FREQ=$(grep -i "Actual RX Freq" "${LOG}" | tail -1 | grep -oE "[0-9.]+ MHz" | head -1); rm -f "${LOG}"
BYTES=$(stat -c%s "${SHM}" 2>/dev/null || echo 0); SAMPS=$((BYTES/4))
[ "${BYTES}" -eq 0 ] && { echo "!! empty capture" >&2; exit 1; }
# peak / floor sanity from the first 4M samples (clipping or too-low gain)
STATS=$(python3 - "${SHM}" <<'PY'
import numpy as np, sys
raw = np.fromfile(sys.argv[1], dtype="<i2", count=8_000_000)
iq = raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)
peak = float(np.max(np.abs(raw))) / 32767.0
rms = float(np.sqrt(np.mean(np.abs(iq) ** 2))) / 32767.0
print(f"peak {peak:.3f} FS, rms {20*np.log10(max(rms,1e-9)):.1f} dBFS, clipped {100*np.mean(np.abs(raw) >= 32700):.4f}%")
PY
)
echo "==> ${BYTES} bytes (${SAMPS} samples); overflow lines: ${OVF}; ${STATS}"
python3 - "$FREQ_HZ" "$RATE_HZ" "$GAIN" "$DUR" "$CH" "$ANT" "$ANTMODEL" "$LABEL" "$DESC" "$OVF" "$STATS" "${MCR}" "${ACT_RATE}" "${ACT_FREQ}" "${OUT_DIR}/${NAME}.sigmf-meta" <<'PY'
import json, sys, datetime
freq, rate, gain, dur, ch, ant, antmodel, label, desc, ovf, stats, mcr, act_rate, act_freq, out = sys.argv[1:]
meta = {"global": {"core:datatype": "ci16_le", "core:sample_rate": float(rate), "core:version": "1.0.0", "core:num_channels": 1,
                   "core:hw": f"USRP N320 (serial 3459703), Rhodium RX ch{ch} antenna port {ant} ({antmodel}), gain {gain} dB" + (f", master_clock_rate {mcr}" if mcr else ""),
                   "core:author": "Brandon Nguyen", "core:recorder": "UHD 4.7 rx_samples_to_file -> /dev/shm (bash_scripts/capture_n320_sigmf.sh)",
                   "core:description": f"{dur} s OTA snippet, {label}: {desc}. DGX Spark bench, {datetime.datetime.now().isoformat(timespec='seconds')}. "
                                       f"Overflow lines {ovf}; {stats}" + (f"; UHD actual rate {act_rate}, freq {act_freq}" if act_rate else ""),
                   "argus:band_label": label, "argus:expected_waveforms": desc, "argus:antenna": antmodel, "argus:rx_channel": int(ch),
                   "argus:gain_db": float(gain), "argus:overflow_events": int(ovf)},
        "captures": [{"core:sample_start": 0, "core:frequency": int(freq), "core:datetime": datetime.datetime.utcnow().isoformat() + "Z"}],
        "annotations": []}
open(out, "w").write(json.dumps(meta, indent=2)); print("wrote", out)
PY
DEST="${OUT_DIR}/${NAME}.sigmf-data"
mv "${SHM}" "${DEST}" && trap - EXIT && echo "==> saved ${DEST}"

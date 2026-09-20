#!/usr/bin/env bash
set -euo pipefail

# Capture overflow-free OTA IQ from the X410 to SigMF, for DINO-FT domain-adaptation training data.
#
# Streams to /dev/shm (RAM) so the host can sustain 491.52 MSps (~1.97 GB/s) with ZERO overflows
# (a disk at ~400 MB/s cannot), then writes the SigMF .sigmf-meta and moves the pair to the captures
# dir. Antenna state (ON for signal / OFF or 50 ohm load for noise negatives) is YOURS to set between
# runs -- record it in --label.
#
# Usage (run under sudo -- UHD needs raw socket access):
#   sudo ./bash_scripts/capture_ota_sigmf.sh --freq 2.4e9 --gain 10 --dur 2 --label 2g4_on
#   sudo ./bash_scripts/capture_ota_sigmf.sh --freq 1.0e9 --gain 10 --dur 2 --label 1g0_on
#   # remove the antenna (or fit a 50 ohm load), then:
#   sudo ./bash_scripts/capture_ota_sigmf.sh --freq 2.4e9 --gain 10 --dur 2 --label 2g4_noise
#
# Env overrides: OUT_DIR (default ~/Documents/captures), RATE (491.52e6), ADDR, RX_BIN.

FREQ=2.4e9; GAIN=10; DUR=2; RATE="${RATE:-491.52e6}"; LABEL=cap
RX="${RX_BIN:-/usr/local/lib/uhd/examples/rx_samples_to_file}"
ADDR="${ADDR:-addr=192.168.10.2,mgmt_addr=192.168.21.2}"
# under sudo, HOME is /root; resolve the desktop user's home for the default out dir
USER_HOME=$(getent passwd "${SUDO_USER:-${USER}}" | cut -d: -f6)
OUT_DIR="${OUT_DIR:-${USER_HOME}/Documents/captures}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --freq)  FREQ="$2"; shift 2;;
    --gain)  GAIN="$2"; shift 2;;
    --dur)   DUR="$2";  shift 2;;
    --rate)  RATE="$2"; shift 2;;
    --label) LABEL="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
# Hz integers for the SigMF meta + a compact freq tag (e.g. 2400M) for the filename.
FREQ_HZ=$(python3 -c "print(int(float('${FREQ}')))")
RATE_HZ=$(python3 -c "print(float('${RATE}'))")
FREQ_TAG=$(python3 -c "print(f'{int(float(\"${FREQ}\")/1e6)}M')")
NAME="x410_ota_${FREQ_TAG}_g${GAIN}_${LABEL}_${TS}"

SHM="/dev/shm/${NAME}.sigmf-data"
mkdir -p "${OUT_DIR}"

# Always free the RAM file on exit BEFORE a successful move (error / Ctrl-C / partial capture) so an
# aborted run never leaves a multi-GB file sitting in /dev/shm. Cleared after a verified move below.
SHM_TO_CLEAN="${SHM}"
cleanup() { [ -n "${SHM_TO_CLEAN:-}" ] && rm -f "${SHM_TO_CLEAN}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Pre-check that RAM has room for the whole capture (+5% margin) so we never half-fill /dev/shm.
NEED=$(python3 -c "print(int(float('${DUR}')*float('${RATE}')*4*1.05))")
FREE=$(df -B1 --output=avail /dev/shm | tail -1 | tr -d ' ')
if [ "${NEED}" -gt "${FREE}" ]; then
  echo "!! Not enough /dev/shm: need ~$((NEED/1000000)) MB, have $((FREE/1000000)) MB free." >&2
  echo "   Reduce --dur, or clear stale files first:  ls -la /dev/shm" >&2
  exit 1
fi

echo "==> Capturing ${DUR}s @ ${FREQ} (${FREQ_TAG}), gain ${GAIN} dB, ${RATE} Msps -> RAM"
LOG=$(mktemp)
sudo "${RX}" --args "${ADDR}" --freq "${FREQ}" --rate "${RATE}" --gain "${GAIN}" \
     --duration "${DUR}" --type short --file "${SHM}" 2>&1 | tee "${LOG}" | \
     grep -iE "Actual RX (Rate|Freq|Gain)|Done" || true

# Overflow check: rx_samples_to_file prints 'O' per overflow event.
if grep -q "^O\|OO\| O " "${LOG}" 2>/dev/null || grep -qi "overflow occurred" "${LOG}" 2>/dev/null; then
  echo "!! WARNING: overflow(s) detected during capture -- data may have gaps. Re-run (shorter --dur?)." >&2
fi
rm -f "${LOG}"

BYTES=$(stat -c%s "${SHM}" 2>/dev/null || echo 0)
SAMPS=$(( BYTES / 4 ))
echo "==> Captured ${BYTES} bytes (${SAMPS} complex samples, ~$(python3 -c "print(f'{${SAMPS}/${RATE_HZ}:.3f}')")s)"

# Write the SigMF sidecar.
python3 - "$FREQ_HZ" "$RATE_HZ" "$GAIN" "$DUR" "$LABEL" "$TS" "${OUT_DIR}/${NAME}.sigmf-meta" <<'PY'
import json, sys
freq, rate, gain, dur, label, ts, out = sys.argv[1:]
meta = {
  "global": {
    "core:datatype": "ci16_le",
    "core:sample_rate": float(rate),
    "core:version": "1.0.0",
    "core:num_channels": 1,
    "core:hw": f"USRP X410 (CG_400, serial 3415B36), RX ch0, gain {gain} dB",
    "core:description": (f"{dur} s OTA capture at {int(freq)} Hz via UHD rx_samples_to_file to /dev/shm "
                         f"(RAM, overflow-free); label={label}; DGX Spark bench {ts}"),
  },
  "captures": [{"core:sample_start": 0, "core:frequency": int(freq)}],
  "annotations": [],
}
open(out, "w").write(json.dumps(meta, indent=2))
print(f"wrote {out}")
PY

# Move the data out of RAM onto disk (cross-filesystem mv = copy+delete -> the RAM copy is freed).
DEST="${OUT_DIR}/${NAME}.sigmf-data"
if mv "${SHM}" "${DEST}" && [ "$(stat -c%s "${DEST}" 2>/dev/null || echo 0)" = "${BYTES}" ]; then
  SHM_TO_CLEAN=""   # moved + size-verified; RAM is clear, nothing for the trap to remove
  chown "${SUDO_USER:-${USER}}:${SUDO_USER:-${USER}}" "${DEST}" "${OUT_DIR}/${NAME}.sigmf-meta" 2>/dev/null || true
  echo "==> Saved ${DEST%.sigmf-data}.sigmf-{data,meta}  (RAM freed; /dev/shm avail now $(df -h /dev/shm | awk 'NR==2{print $4}'))"
else
  echo "!! Move to disk FAILED (disk full?). Capture kept in RAM at ${SHM}." >&2
  echo "   Free disk space and move it manually:  sudo mv ${SHM} ${DEST}" >&2
  SHM_TO_CLEAN=""   # keep the RAM copy so the capture isn't lost; you move it manually
  exit 1
fi

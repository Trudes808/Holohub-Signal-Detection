#!/usr/bin/env bash
# OTA stress-test capture matrix on the N320 (2026-11-03): several bands, each at 2-3 sample rates and
# centre offsets so the detector is scored off-centre and off-rate. Each capture lands in its own folder
# <OUT_ROOT>/<stem>/ (the detections + masks are written next to it later by run_ota_detections.py).
# Gain is probed per band (0.2 s at the start gain; target rms -40..-20 dBFS, peak < 0.5 FS).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
OUT_ROOT="${OUT_ROOT:-$HOME/Datasets/ARGUS/OOD_captures/OTA}"
PROBE_DIR=$(mktemp -d /dev/shm/n320_probe.XXXX)
cap() {  # ch freq rate mcr dur gain0 label desc
  local ch=$1 freq=$2 rate=$3 mcr=$4 dur=$5 gain=$6 label=$7 desc=$8
  local mcr_arg=(); [ -n "$mcr" ] && mcr_arg=(--mcr "$mcr")
  # gain probe (up to 3 adjustments)
  for i in 1 2 3; do
    local out; out=$(OUT_DIR="$PROBE_DIR" "$HERE/capture_n320_sigmf.sh" --freq "$freq" --rate "$rate" --gain "$gain" --dur 0.2 --channel "$ch" --label probe "${mcr_arg[@]}" 2>&1 | grep -oE "peak [0-9.]+ FS, rms -?[0-9.]+ dBFS" | tail -1)
    rm -f "$PROBE_DIR"/*.sigmf-* 2>/dev/null
    local peak rms; peak=$(echo "$out" | grep -oE "peak [0-9.]+" | awk '{print $2}'); rms=$(echo "$out" | grep -oE "rms -?[0-9.]+" | awk '{print $2}')
    [ -z "$peak" ] && { echo "!! probe failed ($label)"; break; }
    if python3 -c "import sys; sys.exit(0 if float('$peak') > 0.5 else 1)"; then gain=$((gain-15)); [ $gain -lt 0 ] && gain=0; echo "   probe $label: peak $peak -> gain $gain"; continue; fi
    if python3 -c "import sys; sys.exit(0 if float('$rms') < -40 else 1)"; then local g2=$((gain+15)); [ $g2 -gt 60 ] && g2=60; [ $g2 -eq $gain ] && break; gain=$g2; echo "   probe $label: rms $rms dBFS -> gain $gain"; continue; fi
    break
  done
  local stem="n320_${label}"; local dir="$OUT_ROOT/tmp_$$"
  OUT_DIR="$dir" "$HERE/capture_n320_sigmf.sh" --freq "$freq" --rate "$rate" --gain "$gain" --dur "$dur" --channel "$ch" --label "$label" --desc "$desc" "${mcr_arg[@]}" 2>&1 | grep -E "^==>|!!" | tail -2
  # move the pair into its own folder named after the stem
  for f in "$dir"/*.sigmf-meta; do [ -f "$f" ] || continue; local s; s=$(basename "$f" .sigmf-meta); mkdir -p "$OUT_ROOT/$s"; mv "$dir/$s".sigmf-* "$OUT_ROOT/$s/"; echo "   -> $OUT_ROOT/$s/"; done
  rmdir "$dir" 2>/dev/null || true
}
# ---------------- VERT900 on RF1 (ch 1) ----------------
cap 1 98e6    24.576e6 245.76e6 1   40 fm_broadcast_88-108MHz        "FM broadcast stations on the 200 kHz raster (75 kHz dev stereo + RDS); VERT900 is off-resonance here"
cap 1 103e6   10e6     200e6    2   40 fm_broadcast_98-108MHz_partial "upper half of the FM band, narrow look"
cap 1 94e6    50e6     200e6    0.5 40 fm_broadcast_69-119MHz_wide    "FM band inside a wide 50 MHz look (band edges in view)"
cap 1 530e6   100e6    200e6    0.5 40 hdtv_uhf_480-580MHz_scan       "ATSC 6 MHz HDTV channels (UHF 14-36), wide scan"
cap 1 515e6   31.25e6  250e6    1   40 hdtv_uhf_499-531MHz            "ATSC HDTV channels, 31.25 MSps"
cap 1 575e6   20e6     200e6    1   40 hdtv_uhf_565-585MHz            "ATSC HDTV channels, 20 MSps, likely one channel cut at the edge"
cap 1 742e6   31.25e6  250e6    1   40 cell700_downlink_726-758MHz    "LTE band 12/13/17 downlinks (5/10 MHz carriers) + part of band 13"
cap 1 707e6   20e6     200e6    2   45 cell700_uplink_697-717MHz      "LTE band 12/17 uplinks: sporadic narrow UE bursts, low duty"
cap 1 881.5e6 25e6     200e6    1   40 cell850_downlink_869-894MHz    "cellular band 5 downlinks (LTE/NR, CDMA legacy) centred on the band"
cap 1 875e6   50e6     200e6    0.5 40 cell850_downlink_850-900MHz_offcentre "band 5 downlink in a wide off-centre look: UL/DL guard + ISM edge"
cap 1 890e6   12.5e6   200e6    2   40 cell850_downlink_884-896MHz_edge "upper band-5 downlink carriers, one cut by the band edge"
cap 1 836.5e6 25e6     200e6    1   45 cell850_uplink_824-849MHz      "band 5 uplinks: low-duty UE bursts, PUCCH edges, PRACH"
cap 1 830e6   10.24e6  245.76e6 2   45 cell850_uplink_825-835MHz_narrow "band 5 uplink, narrow look, 10.24 MSps"
cap 1 845e6   61.44e6  245.76e6 0.5 45 cell850_uplink_814-876MHz_wide "band 5 uplink + guard + lower downlink in one wide look"
cap 1 915e6   30.72e6  245.76e6 1   45 ism900_900-930MHz              "900 MHz ISM: LoRa, FHSS smart meters, Z-Wave, pagers nearby"
cap 1 920e6   12.5e6   200e6    2   45 ism900_914-926MHz_narrow       "900 MHz ISM, narrow look"
cap 1 1090e6  24.576e6 245.76e6 1   45 adsb_dme_1078-1102MHz          "ADS-B / Mode S replies (1090 MHz pulses), DME, radar"
cap 1 1350e6  25e6     200e6    1   45 quiet_1338-1362MHz             "expected quiet band (L-band radar region): false-alarm reference"
# ---------------- VERT2450 on RF0 (ch 0) ----------------
cap 0 2437e6  100e6    200e6    0.5 35 ism2g4_2387-2487MHz_full       "2.4 GHz ISM: WiFi ch 1-14, Bluetooth, BLE, Zigbee, microwave ovens"
cap 0 2450e6  200e6    200e6    0.5 35 ism2g4_2350-2550MHz_wide       "2.4 GHz ISM inside a 200 MHz look (band edges and out-of-band in view)"
cap 0 2412e6  40e6     200e6    1   35 ism2g4_2392-2432MHz_ch1edge    "WiFi channel 1 at the centre, channel 6 cut at the edge"
cap 0 2462e6  20e6     200e6    1   35 ism2g4_2452-2472MHz_ch11       "WiFi channel 11 only, 20 MSps"
cap 0 2442e6  61.44e6  245.76e6 1   35 ism2g4_2411-2473MHz            "WiFi ch 1-11 at 61.44 MSps (odd rate for the detector)"
cap 0 2425e6  12.5e6   200e6    2   35 ism2g4_2419-2431MHz_narrow     "between WiFi ch 3 and 5: BT hops and packet edges, 12.5 MSps"
cap 0 5200e6  100e6    200e6    0.5 35 wifi5g_5150-5250MHz_unii1      "5 GHz UNII-1 WiFi (ch 36-48, 80 MHz BSSs)"
cap 0 5500e6  200e6    200e6    0.5 35 wifi5g_5400-5600MHz_unii2e     "5 GHz UNII-2 extended (DFS) in a 200 MHz look"
cap 0 5775e6  62.5e6   250e6    1   35 wifi5g_5744-5806MHz_unii3      "5 GHz UNII-3 WiFi (ch 149-165)"
cap 0 5180e6  20e6     200e6    1   35 wifi5g_5170-5190MHz_ch36       "single 5 GHz channel 36, 20 MSps"
cap 0 1960e6  61.44e6  245.76e6 1   40 cell1900_pcs_downlink_1929-1991MHz "PCS band 2 downlinks (VERT2450 off-resonance)"
cap 0 2132e6  50e6     200e6    1   40 cell2100_aws_downlink_2107-2157MHz "AWS band 4/66 downlinks (VERT2450 off-resonance)"
rm -rf "$PROBE_DIR"
echo "MATRIX DONE"

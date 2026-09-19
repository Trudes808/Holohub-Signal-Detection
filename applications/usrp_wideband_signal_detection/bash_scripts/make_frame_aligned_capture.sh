#!/usr/bin/env bash
# Make a FRAME-ALIGNED canonical loopback test capture from the OTA capture.
#
# WHY: the raw OTA capture (x410_ota_2g4_gain10_20260908.spark.pcap) is 480000 packets =
# exactly 46.875 detector frames (10240 packets/frame, 1024 samples/packet, 512 rows x 20480
# fft_size = 10,485,760 samples/frame). 46.875 is NOT a whole number, so `tcpreplay --loop`
# re-sends the pcap starting 0.875 of a frame into frame 47 -> every frame after the first pass
# is a shifted splice of two capture frames, and the online (loopback RT) masks stop matching the
# offline masks 1:1 (mean IoU collapses from 0.998 to ~0.02). See notes/dino_ft_finetune_plan.md
# (2026-09-19) and memory loopback-46875-frame-seam.
#
# FIX: trim to a whole number of frames so every tcpreplay loop starts exactly on a frame
# boundary. Then online frame k == offline frame (k mod FRAMES) for ALL loops, and the loopback
# masks reproduce the offline masks bit-for-bit regardless of --loop count.
#
# Produces both channels (ch0 dst port 1234, ch1 dst port 1235 via portmap) so it is a drop-in
# for both the single-channel exact comparison and bash_scripts/run_loopback_dual_dino_ft.sh.
set -euo pipefail

COMP=${COMP:-/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites}
SRC=${SRC:-$COMP/x410_ota_2g4_gain10_20260908.spark.pcap}
PKTS_PER_FRAME=10240        # 10,485,760 samples/frame / 1024 samples/packet
FRAMES=${FRAMES:-46}        # max whole frames in the 46.875-frame capture
NPKTS=$((FRAMES * PKTS_PER_FRAME))

CH0_OUT=${CH0_OUT:-$COMP/x410_ota_2g4_gain10_20260908_${FRAMES}f.spark.pcap}
CH1_OUT=${CH1_OUT:-$COMP/x410_ota_ch1_p1235_${FRAMES}f.spark.pcap}

for t in editcap tcprewrite capinfos; do command -v "$t" >/dev/null || { echo "!! missing $t (apt-get install wireshark-common tshark)"; exit 1; }; done
[ -f "$SRC" ] || { echo "!! source pcap not found: $SRC"; exit 1; }

echo "==> source: $SRC ($(capinfos -c -M "$SRC" | awk '/Number of packets/{print $NF}') packets)"
echo "==> keeping first $NPKTS packets = $FRAMES frames = $((NPKTS * 1024)) samples"

echo "==> [1/2] trim ch0 (dst port 1234)"
editcap -r "$SRC" "$CH0_OUT" 1-"$NPKTS"

echo "==> [2/2] portmap ch0 -> ch1 (UDP dst 1234 -> 1235)"
tcprewrite --portmap=1234:1235 --infile="$CH0_OUT" --outfile="$CH1_OUT"

echo "==> verify (both must read exactly $NPKTS packets):"
for f in "$CH0_OUT" "$CH1_OUT"; do
  n=$(capinfos -c -M "$f" | awk '/Number of packets/{print $NF}')
  printf "   %-70s %s packets %s\n" "$(basename "$f")" "$n" "$([ "$n" = "$NPKTS" ] && echo OK || echo MISMATCH)"
done
echo "==> ch1 dst port (expect 1235):"
tshark -r "$CH1_OUT" -c 1 -T fields -e udp.dstport 2>/dev/null | sed 's/^/   /'
echo "DONE: $CH0_OUT"
echo "DONE: $CH1_OUT"

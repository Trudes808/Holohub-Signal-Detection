#!/usr/bin/env bash
# After capture_n320_matrix.sh: detections (M5S/M6S/M7S) for every capture, synthetic-UAV overlays on the
# 2.4 GHz captures + their scoring, and the markdown summary. Resumable; waits for "MATRIX DONE".
set -uo pipefail
FT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
PY=${PY:-/home/genesys-dgx1/Documents/holoscan_waveform_generation/.venv-ml/bin/python}
export PYTHONPATH="$FT/src:${PYTHONPATH:-}"
OTA=${OTA:-$HOME/Datasets/ARGUS/OOD_captures/OTA}
EXP=$FT/../applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/dino_ft_multirate_scenes
MODELS="M5S=$FT/checkpoints/M5S_491_ota/best.pt M6S=$FT/checkpoints/M6S_multirate/best.pt M7S=$FT/checkpoints/M7S_scenes_v2/best.pt"
until grep -q "MATRIX DONE" "$OTA"/n320_capture_matrix_*.log 2>/dev/null; do sleep 20; done
echo "[ota] matrix done: $(ls -d $OTA/n320_*/ | wc -l) capture folders"
cd "$FT"
$PY src/run_ota_detections.py --root "$OTA" --models $MODELS --max-frames 12 --overlay-frames 2
# ---- synthetic UAV overlays on the 2.4 GHz captures (mixed real + synthetic)
mk() {  # glob mode bw snr fc_offset seed tag
  local d; d=$(ls -d $OTA/$1 2>/dev/null | head -1); [ -z "$d" ] && { echo "[ota] no capture for $1"; return; }
  local meta; meta=$(ls "$d"/*.sigmf-meta | grep -v synth | head -1)
  $PY src/overlay_synthetic_uav.py --capture "$meta" --out "$d/synth_uav" --mode "$2" --video-bw "$3" --snr "$4" "--fc-offset-hz=$5" --seed "$6" --tag "$7" --max-seconds 0.5
}
mk "n320_ism2g4_2387-2487MHz_full_*"      ofdm 20e6 25 7e6    3 "synthUAV_ofdm20M_snr25_off+7M"
mk "n320_ism2g4_2387-2487MHz_full_*"      ofdm 10e6 8  -22e6  4 "synthUAV_ofdm10M_snr8_off-22M"
mk "n320_ism2g4_2350-2550MHz_wide_*"      ofdm 40e6 30 -60e6  5 "synthUAV_ofdm40M_snr30_off-60M"
mk "n320_ism2g4_2411-2473MHz_*"           ofdm 20e6 12 12e6   6 "synthUAV_ofdm20M_snr12_off+12M"
mk "n320_ism2g4_2392-2432MHz_ch1edge_*"   fm   20e6 20 15e6   7 "synthUAV_fm20M_snr20_edge+15M"
mk "n320_ism2g4_2452-2472MHz_ch11_*"      ofdm 20e6 5  -3e6   8 "synthUAV_ofdm20M_snr5_off-3M"
mk "n320_ism2g4_2419-2431MHz_narrow_*"    ofdm 10e6 18 2e6    9 "synthUAV_ofdm10M_snr18_narrowband-cut"
mk "n320_wifi5g_5150-5250MHz_unii1_*"     ofdm 20e6 22 -30e6  10 "synthUAV_ofdm20M_snr22_5GHz"
# score the synthetic bursts in the real background with every model (region det + overlays)
for tag in M5S:$FT/checkpoints/M5S_491_ota/best.pt M6S:$FT/checkpoints/M6S_multirate/best.pt M7S:$FT/checkpoints/M7S_scenes_v2/best.pt; do
  n=${tag%%:*}; ck=${tag#*:}
  $PY src/eval_scenes.py --model "$ck" --name "$n" --sigmf $(ls -d $OTA/n320_*/synth_uav 2>/dev/null) --out "eval_out/ota_synth_uav_$n" --max-frames 6 --png-per-scenario 0
done
$PY src/compare_evals.py --names M5S M6S M7S --scenes-kind ota_synth_uav --out "$EXP/results/ota_synth_uav_comparison.md" || true
# 3-model panels for the synthetic-UAV files
for m in $(ls $OTA/n320_*/synth_uav/*.sigmf-meta 2>/dev/null); do
  d=$(dirname "$m"); s=$(basename "$m" .sigmf-meta)
  $PY src/compare_overlays.py --a $FT/checkpoints/M5S_491_ota/best.pt --b $FT/checkpoints/M6S_multirate/best.pt --c $FT/checkpoints/M7S_scenes_v2/best.pt \
      --names M5-S M6-S M7-S --out "$d/panels_$s" --scene "$m:0" --title "$s"
done
$PY src/summarize_ota_detections.py --root "$OTA" --out "$OTA/README_n320_detections_2026-11-03.md"
cp "$OTA/README_n320_detections_2026-11-03.md" "$EXP/results/ota_n320_detections.md"
echo "[ota] ALL DONE"

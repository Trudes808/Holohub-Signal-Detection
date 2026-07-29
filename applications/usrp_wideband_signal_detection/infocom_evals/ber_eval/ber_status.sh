#!/usr/bin/env bash
# BER sweep live status dashboard. Run anytime:  bash ber_status.sh
# (from a Claude prompt:  ! bash applications/.../infocom_evals/ber_eval/ber_status.sh)
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results"
LEVELS="0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80"
DETS="ground_truth coherent_power finetuned_dino_m2"
declare -A SHORT=( [ground_truth]=GT [coherent_power]=COH [finetuned_dino_m2]=DINO )

ber_of() {  # $1=det $2=level  -> BER from per-signal CSV, or "" if not done
  local f="$R/ber_${1}_attenuation_dB_${2}.csv"
  [ -f "$f" ] || { echo ""; return; }
  awk -F, 'NR>1 && $7!="NaN" && $7!="" {e+=$7; b+=$8} END{ if(b>0) printf "%.3f", e/b; else printf "n/a" }' "$f"
}
running_stem() {  # echo "det:level" pairs currently in a MATLAB -batch
  for p in $(pgrep -f "R2025b/bin/glnxa64/MATLAB -batch" 2>/dev/null); do
    c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
    lv=$(echo "$c" | grep -oE "attenuation_dB_[0-9]+" | head -1 | grep -oE "[0-9]+$")
    for d in $DETS; do echo "$c" | grep -q "$d" && [ -n "$lv" ] && echo "$d:$lv"; done
  done
}

RUN=$(running_stem)
echo "================================================================================"
echo "  BER SWEEP STATUS  —  $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"
printf "  %-6s |" "level"; for L in $LEVELS; do printf " %4s" "$L"; done; echo "   (dB atten)"
printf "  %-6s |" "SNR"; for L in $LEVELS; do printf " %4s" "$((54-L))"; done; echo "   (approx dB)"
echo "  -------+$(for L in $LEVELS; do printf -- "-----"; done)"
total=0; done=0
for d in $DETS; do
  printf "  %-6s |" "${SHORT[$d]}"
  for L in $LEVELS; do
    total=$((total+1))
    b=$(ber_of "$d" "$L")
    if [ -n "$b" ]; then done=$((done+1)); printf " %4s" "$b"
    elif echo "$RUN" | grep -q "^$d:$L$"; then printf " %4s" "~~"
    else printf " %4s" "·"; fi
  done
  echo ""
done
echo "  -------+$(for L in $LEVELS; do printf -- "-----"; done)"
pct=$(( done * 100 / total ))
echo "  legend: number=done(BER)   ~~=running   ·=pending"
echo ""
echo "  PROGRESS: ${done}/${total} detector-levels complete  = ${pct}%"
# per-detector tally
for d in $DETS; do
  n=0; for L in $LEVELS; do [ -n "$(ber_of "$d" "$L")" ] && n=$((n+1)); done
  printf "     %-18s %2d/17\n" "${SHORT[$d]}:" "$n"
done
echo ""
echo "  live: load=$(cut -d' ' -f1-3 /proc/loadavg)   MATLAB evals=$(pgrep -fc 'R2025b/bin/glnxa64/MATLAB -batch' 2>/dev/null)   /tmp free=$(df -h --output=avail /tmp | tail -1 | tr -d ' ')"
drv=$(pgrep -af 'run_ber_sweep.sh' 2>/dev/null | grep -oE 'run_ber_sweep.sh [a-z_]+' | sort -u | tr '\n' ' ')
echo "  drivers running: ${drv:-none}"
[ -n "$RUN" ] && echo "  evaluating now : $(echo "$RUN" | sed 's/ground_truth/GT/;s/coherent_power/COH/;s/finetuned_dino_m2/DINO/' | tr '\n' ' ')"
echo "================================================================================"

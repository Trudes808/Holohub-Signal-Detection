#!/usr/bin/env bash
LOG=/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/reports/resource.log
echo "ts,mem_used_gb,mem_avail_gb,swap_used_gb,gpu_mem_mb,gpu_util,load1,my_rss_gb,warn" > "$LOG"
for i in $(seq 1 480); do
  read mu ma <<<"$(free -m | awk '/Mem:/{print $3, $7}')"
  sw=$(free -m | awk '/Swap:/{print $3}')
  gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d ' ')
  load=$(awk '{print $1}' /proc/loadavg)
  myrss=$(ps -u bqn82 -o rss= 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/1048576}')
  warn=""
  [ "$ma" -lt 8000 ] 2>/dev/null && warn="LOW_MEM"
  [ "$sw" -gt 7500 ] 2>/dev/null && warn="${warn}:HIGH_SWAP"
  echo "$(date +%H:%M:%S),$(awk "BEGIN{printf \"%.1f\",$mu/1024}"),$(awk "BEGIN{printf \"%.1f\",$ma/1024}"),$(awk "BEGIN{printf \"%.1f\",$sw/1024}"),${gpu},${load},${myrss},${warn}" >> "$LOG"
  sleep 60
done

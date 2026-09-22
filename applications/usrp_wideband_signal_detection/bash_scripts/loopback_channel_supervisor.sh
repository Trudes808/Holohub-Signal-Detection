#!/usr/bin/env bash
# Loopback single<->dual channel supervisor. Backs the DEMO CONTROLS "Channels" toggle: the viz
# button writes "channel_mode": "single"|"dual" into demo_control.json; this watcher relaunches the
# loopback app + tcpreplay with the matching config when the value changes (~15 s reconfigure, the
# same UX as the detector switch). Run it in the background (it launches the initial mode, then loops):
#
#   DEMO_DISPLAY=:1 sudo -E ./bash_scripts/loopback_channel_supervisor.sh [single|dual]   # default dual
#
# Stop: kill this process, then ./bash_scripts/stop_loopback_m5s_dual.sh
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export DEMO_DISPLAY="${DEMO_DISPLAY:-${DISPLAY:-:1}}"
CTL="${CTL:-/tmp/usrp_spectrograms/demo_control.json}"
current="${1:-dual}"

launch() {
  case "$1" in
    single) "${SCRIPT_DIR}/run_loopback_m5s_single.sh" ;;
    dual)   "${SCRIPT_DIR}/run_loopback_m5s_dual.sh" ;;
    *) echo "!! unknown mode '$1'"; return 1 ;;
  esac
}
read_mode() { grep -oE '"channel_mode"[[:space:]]*:[[:space:]]*"(single|dual)"' "$CTL" 2>/dev/null | grep -oE '(single|dual)' | tail -1; }

echo "==> [supervisor] initial channel mode = ${current}"
launch "${current}" || exit 1

echo "==> [supervisor] watching ${CTL} for channel_mode changes (Ctrl-C to stop)"
while true; do
  sleep 1
  want=$(read_mode)
  [[ -z "$want" || "$want" == "$current" ]] && continue
  echo "==> [supervisor] channel_mode: ${current} -> ${want}; relaunching loopback"
  if launch "$want"; then
    current="$want"
  else
    echo "!! [supervisor] relaunch to ${want} failed; staying on ${current}"
  fi
done

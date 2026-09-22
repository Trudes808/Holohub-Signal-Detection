#!/usr/bin/env bash
# Loopback demo supervisor (mini-conductor). Owns the whole loopback demo lifecycle so the DEMO
# CONTROLS panel is fully live:
#   * launches the classify-only AMC daemon ONCE (feeds the CLASSIFIER OUTPUTS + COMPUTE panels;
#     survives app relaunches),
#   * seeds + owns demo_control.json,
#   * watches BOTH "detector" (coherent_power <-> cuda_dino_finetuned) and "channel_mode"
#     (single <-> dual) and relaunches the app + tcpreplay with the matching config on either change
#     (~15-25 s reconfigure, the same UX as the live conductor's detector switch).
#
# Run it in the background:
#   DEMO_DISPLAY=:1 sudo -E ./bash_scripts/loopback_channel_supervisor.sh [single|dual]   # default dual
# Stop: kill this process, then ./bash_scripts/stop_loopback_m5s_dual.sh (and pkill rt_decode_daemon).
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
export DEMO_DISPLAY="${DEMO_DISPLAY:-${DISPLAY:-:1}}"
JOB=/tmp/usrp_spectrograms
CTL="${CTL:-${JOB}/demo_control.json}"
init_mode="${1:-dual}"

# --- config matrix: (detector, channel_mode) -> "runner config.yaml" -------------------------------
config_for() {
  case "$1/$2" in
    coherent_power/single)      echo "single config_live_v3_single_channel.yaml" ;;
    coherent_power/dual)        echo "dual config_live_v3_two_channel.yaml" ;;
    cuda_dino_finetuned/single) echo "single config_live_v3_dino_ft_m5s.yaml" ;;
    cuda_dino_finetuned/dual)   echo "dual config_live_v3_dino_ft_m5s_two_channel.yaml" ;;
    *) echo "" ;;
  esac
}
launch() {  # $1=detector $2=channel_mode
  local m; m=$(config_for "$1" "$2"); [[ -z "$m" ]] && { echo "!! [supervisor] bad detector/mode '$1/$2'"; return 1; }
  local runner="${m%% *}" cfg="${m##* }"
  echo "==> [supervisor] launch ${runner} config=${cfg} (detector=$1 channels=$2)"
  CONFIG_NAME="${cfg}" "${SCRIPT_DIR}/run_loopback_m5s_${runner}.sh"
}
read_field() { grep -oE "\"$1\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" "$CTL" 2>/dev/null | sed -E 's/.*"([^"]*)"[[:space:]]*$/\1/' | tail -1; }

# --- seed the control file ONCE (supervisor owns it; runners only seed if absent) ------------------
sudo mkdir -p "${JOB}/snippets" && sudo chmod -R 1777 "${JOB}" 2>/dev/null || true
printf '{"gate": "resnet1d", "snr": "clean", "detector": "cuda_dino_finetuned", "classifiers": "resnet1d,cnn1d,tprime_lg", "color_model": "resnet1d", "channel_mode": "%s", "seq": 1}\n' "${init_mode}" \
  | sudo tee "${CTL}" > /dev/null
sudo chmod 666 "${CTL}"

# --- classify-only AMC daemon (release_v2 taxonomy) ONCE; independent of the app, survives relaunch -
V3_USER_HOME=$(getent passwd "${SUDO_USER:-${USER}}" | cut -d: -f6)
VENV_PY="${VENV_PY:-${V3_USER_HOME}/Documents/holoscan_waveform_generation/.venv-ml/bin/python}"
sudo pkill -f 'rt_decode_dae[m]on.py' 2>/dev/null || true
sudo rm -f "${JOB}/rt_metrics.json" "${JOB}/daemon_live.log" "${JOB}"/snip_stats*.json 2>/dev/null || true
echo "==> [supervisor] launching classify-only AMC daemon (release_v2; ~20-30 s torch import)"
( cd "${APP_DIR}/infocom_evals/pycodec_e2e" && \
  PYCODEC_ROOT="${V3_USER_HOME}/Documents/holoscan_waveform_generation" \
  setsid nohup "${VENV_PY}" rt_decode_daemon.py --snips "${JOB}/snippets" \
    --classify-only --no-band-truth --amc-v2 --color-model resnet1d \
    --classify-min-interval-ms "${AMC_MIN_INTERVAL_MS:-100}" \
    --metrics-out "${JOB}/rt_metrics.json" > "${JOB}/daemon_live.log" 2>&1 < /dev/null & )

# --- snippet janitor (keep last 5 min) -------------------------------------------------------------
( setsid nohup bash -c 'while true; do find /tmp/usrp_spectrograms/snippets -name "snip_pack*" -mmin +5 -delete 2>/dev/null; sleep 30; done' \
    >/dev/null 2>&1 < /dev/null & )

# --- initial app launch ----------------------------------------------------------------------------
cur_det=$(read_field detector);       [[ -z "$cur_det" ]] && cur_det="cuda_dino_finetuned"
cur_mode=$(read_field channel_mode);  [[ -z "$cur_mode" ]] && cur_mode="${init_mode}"
launch "$cur_det" "$cur_mode" || exit 1

echo "==> [supervisor] watching ${CTL} for detector + channel_mode changes (Ctrl-C to stop)"
while true; do
  sleep 1
  d=$(read_field detector); m=$(read_field channel_mode)
  [[ -z "$d" || -z "$m" ]] && continue
  if [[ "$d" != "$cur_det" || "$m" != "$cur_mode" ]]; then
    echo "==> [supervisor] change: detector ${cur_det}->${d}, channels ${cur_mode}->${m}; relaunching"
    if launch "$d" "$m"; then cur_det="$d"; cur_mode="$m"; else echo "!! [supervisor] relaunch failed; keeping ${cur_det}/${cur_mode}"; fi
  fi
done

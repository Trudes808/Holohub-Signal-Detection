#!/usr/bin/env bash
# Re-exec under bash if started with sh/dash (this script uses bash-only syntax).
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Offline signal_snipper runner (data-saving eval).
#
# Replays SigMF captures through the coherent_power detector + signal_snipper + sigmf_file_sink,
# writing snipped SigMF per capture to a HOST-VISIBLE dir that the data-saving notebook reads to
# compute the *measured* resample+filter footprint (replacing the raw-mask-coverage projection).
#
# Snippets land at:  ${SNIP_ROOT}/${DETECTOR}/<capture-stem>/snippets/*.sigmf-data
# which maps 1:1 into the container mount (/workspace/spectrograms -> /tmp/usrp_spectrograms).
#
# Usage (lab-admin, needs docker/sudo; CONTAINER_NAME must match the built container):
#   sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_offline_snipper.sh
#   sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_offline_snipper.sh /path/a.sigmf-data /path/b.sigmf-data
# Env knobs: DETECTOR (default coherent_power), CONFIG_NAME, SNIP_ROOT, CAPTURES_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

DETECTOR="${DETECTOR:-coherent_power}"
CONFIG_NAME="${CONFIG_NAME:-config_signal_snipper_single_channel.yaml}"
SNIP_ROOT="${SNIP_ROOT:-/tmp/usrp_spectrograms/snippets_eval}"
USER_HOME="$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)"
CAPTURES_DIR="${CAPTURES_DIR:-${USER_HOME}/captures/live_data/sigmf_out}"

if [[ $# -gt 0 ]]; then
  CAPTURES=("$@")
else
  shopt -s nullglob
  CAPTURES=("${CAPTURES_DIR}"/*.sigmf-data)
  shopt -u nullglob
fi
if [[ ${#CAPTURES[@]} -eq 0 ]]; then
  echo "No captures found (pass files as args or set CAPTURES_DIR=<dir with *.sigmf-data>)." >&2
  exit 1
fi

echo "detector=${DETECTOR}  config=${CONFIG_NAME}  container=${CONTAINER_NAME:-<from container_env.sh>}"
echo "captures (${#CAPTURES[@]}): ${CAPTURES[*]}"
for cap in "${CAPTURES[@]}"; do
  stem="$(basename "$cap")"; stem="${stem%.sigmf-data}"
  out="${SNIP_ROOT}/${DETECTOR}/${stem}"
  echo "=== signal_snipper offline: ${stem} -> ${out}/snippets ==="
  python3 "${APP_DIR}/run_cuda_dino_offline_file.py" "$cap" \
      --detector "${DETECTOR}" \
      --config "${APP_DIR}/${CONFIG_NAME}" \
      --output-root "${out}" \
      --no-tensors
done
echo "DONE — snippets under ${SNIP_ROOT}/${DETECTOR}/<stem>/snippets/"

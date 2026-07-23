#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
# Resilient launcher for the full snip. run_snip_all.sh is already resumable (skips done captures),
# per-run tolerant (one failed run doesn't abort), and footprint-only (--snippets-only). This wrapper
# re-invokes it in a loop until a full pass makes NO new progress (i.e. everything done, or stuck),
# so transient per-run kills self-heal. Launch it DETACHED so a dying su/tmux shell can't take it down:
#
#   cd .../infocom_evals/snip_eval
#   sudo nohup setsid ./launch_snip.sh > /tmp/snip_run.log 2>&1 &
#   tail -f /tmp/snip_run.log          # Ctrl-C the tail anytime; the run keeps going
#
# If the WHOLE process tree is still killed (e.g. OOM), just re-launch the same command — it resumes.
set -uo pipefail   # NOT -e: the loop must survive a failed/killed pass
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export CONTAINER_NAME="${CONTAINER_NAME:-usrp_x410_sig_det_bqn82}"
export BATCH_ROOT="${BATCH_ROOT:-${SCRIPT_DIR}/snip_run}"
export CAPTURES_DIR="${CAPTURES_DIR:-/home/bqn82/captures}"
export SNIP_OUT="${SNIP_OUT:-/tmp/usrp_spectrograms/snip_eval}"
export MODES="${MODES:-frequency time_only}"
MAX_PASSES="${MAX_PASSES:-30}"

count_done() { find "${SNIP_OUT}" -mindepth 4 -maxdepth 4 -name snippets -type d 2>/dev/null | wc -l; }

prev=-1
for pass in $(seq 1 "${MAX_PASSES}"); do
  d0="$(count_done)"
  echo "===== snip pass ${pass}/${MAX_PASSES}  start $(date -u +%H:%M:%S)  done=${d0} ====="
  bash "${SCRIPT_DIR}/run_snip_all.sh" || echo "  (run_snip_all exited non-zero / was killed — resuming next pass)"
  d1="$(count_done)"
  echo "===== pass ${pass} end $(date -u +%H:%M:%S)  done=${d1} ====="
  # A completed pass either did new work (d1 > d0) or, if everything was already done, added nothing.
  if [ "${d1}" -le "${prev}" ] || { [ "${d1}" -eq "${d0}" ] && [ "${pass}" -gt 1 ]; }; then
    echo "no new progress this pass (done=${d1}) -> finished or stuck. Stopping."
    break
  fi
  prev="${d1}"
done
echo "LAUNCHER DONE $(date -u +%H:%M:%S)  done=$(count_done) snippet dirs."
echo "Next: python3 ${SCRIPT_DIR}/verify_snip.py --snip-out ${SNIP_OUT}"

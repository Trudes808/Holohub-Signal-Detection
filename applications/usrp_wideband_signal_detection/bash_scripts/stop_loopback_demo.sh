#!/usr/bin/env bash
# Tear down everything run_loopback_demo.sh started. Each pkill runs as its
# own command with an anchored/specific pattern (a compound line whose text
# contains the pattern would match and kill your own shell).
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
# shellcheck source=container_env.sh
source "${SCRIPT_DIR}/container_env.sh"

echo "==> stopping tcpreplay"
sudo pkill -f '^tcpreplay' 2>/dev/null || true
echo "==> stopping conductor"
sudo pkill -f '^python3 demo_conductor' 2>/dev/null || true
echo "==> stopping decode daemon"
pkill -f 'rt_decode_daemon.py --snips' 2>/dev/null || true
echo "==> stopping snippet janitor"
sudo pkill -f 'usrp_spectrograms/snippets.*-mmin' 2>/dev/null || true
echo "==> stopping the app"
sudo docker exec "${CONTAINER_NAME}" bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true" 2>/dev/null || true

sleep 1
LEFT=$(pgrep -af 'tcpreplay|demo_conductor|rt_decode_daemon' | grep -v grep || true)
if [[ -n "${LEFT}" ]]; then
  echo "still running:"; echo "${LEFT}"
else
  echo "DEMO STOPPED (snippets left in /tmp/usrp_spectrograms/snippets; janitor gone)."
fi

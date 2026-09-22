#!/usr/bin/env bash
# Tear down the persistent dual M5-S loopback demo started by run_loopback_m5s_dual.sh:
# stop both tcpreplay loops and the container-side app.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
source "${SCRIPT_DIR}/container_env.sh" 2>/dev/null || true
CN="${CONTAINER_NAME:-usrp_x410_sig_det_sat3737}"
echo "==> stopping tcpreplay loops"
sudo pkill -f 'tcpreplay --preload-pcap' 2>/dev/null || true  # matches the sudo/setsid/nohup wrapper + child
echo "==> stopping container-side app"
sudo docker exec "${CN}" bash -lc \
  "pkill -f '(^|/)usrp_wideband_signal_detection( |\$)' || true; \
   for i in \$(seq 1 20); do pgrep -f '(^|/)usrp_wideband_signal_detection( |\$)' >/dev/null || break; sleep 1; done; \
   rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true; echo cleaned" 2>/dev/null || true
echo "==> dual M5-S loopback stopped."

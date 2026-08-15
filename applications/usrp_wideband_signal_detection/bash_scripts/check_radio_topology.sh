#!/usr/bin/env bash
# Radio-topology preflight: discover which host QSFP port is actually wired to which X410
# interface (by ARP, no assumptions), instead of streaming into the wrong port after cables
# or IPs get shuffled (2026-08-15 incident: swapped cables + foreign IPs -> silent 0-packet runs).
#
#   sudo ./bash_scripts/check_radio_topology.sh          # probe, report, fix IPs if canonical
#   sudo ./bash_scripts/check_radio_topology.sh --quiet  # exit code only (for wrappers)
#
# Exit codes: 0 = canonical topology verified + host IPs set
#             2 = radio found but CROSSED (control/data cables swapped) — prints the fix
#             3 = radio not reachable on either port (off / booting / unplugged)
#
# Canonical topology (see notes/RUNNING_ON_DGX_SPARK.md):
#   host enp1s0f0np0 (0000:01:00.0, DPDK)  <->  X410 sfp0 data    (192.168.10.2, host 192.168.10.1/24)
#   host enp1s0f1np1 (0000:01:00.1, kernel) <->  X410 sfp1 control (192.168.21.2, host 192.168.21.1/24)
set -euo pipefail

DATA_IF=${DATA_IF:-enp1s0f0np0}
CTRL_IF=${CTRL_IF:-enp1s0f1np1}
RADIO_DATA_IP=${RADIO_DATA_IP:-192.168.10.2}
RADIO_CTRL_IP=${RADIO_CTRL_IP:-192.168.21.2}
HOST_DATA_ADDR=${HOST_DATA_ADDR:-192.168.10.1/24}
HOST_CTRL_ADDR=${HOST_CTRL_ADDR:-192.168.21.1/24}
QUIET=${1:-}

log() { [ "${QUIET}" = "--quiet" ] || echo "$@"; }

if [ "$(id -u)" -ne 0 ]; then
  echo "must run as root (manipulates interface addresses for probing)" >&2
  exit 1
fi

# probe <iface> <target_ip> <host_addr_cidr>: is target_ip on the wire behind iface?
# Sends one ping to trigger ARP and checks the neighbor table for a resolved lladdr — this
# works even when the radio's own ARP cache is stale from a previous topology (its ARP
# replies come back regardless, only higher-layer traffic breaks).
probe() {
  local iface=$1 target=$2 host_addr=$3 added=0 found=1
  ip link set "$iface" up 2>/dev/null || return 1
  if ! ip -o addr show dev "$iface" | grep -q "inet ${host_addr%/*}/"; then
    ip addr add "$host_addr" dev "$iface" 2>/dev/null && added=1
  fi
  ip neigh del "$target" dev "$iface" 2>/dev/null || true
  ping -c1 -W1 -I "$iface" "$target" > /dev/null 2>&1 || true
  if ip neigh show "$target" dev "$iface" 2>/dev/null | grep -q lladdr; then
    found=0
  fi
  if [ "$added" -eq 1 ]; then
    ip addr del "$host_addr" dev "$iface" 2>/dev/null || true
  fi
  return $found
}

ctrl_on_ctrl=1; data_on_data=1; ctrl_on_data=1; data_on_ctrl=1
probe "$CTRL_IF" "$RADIO_CTRL_IP" "$HOST_CTRL_ADDR" && ctrl_on_ctrl=0 || true
probe "$DATA_IF" "$RADIO_DATA_IP" "$HOST_DATA_ADDR" && data_on_data=0 || true
if [ $ctrl_on_ctrl -ne 0 ] || [ $data_on_data -ne 0 ]; then
  probe "$DATA_IF" "$RADIO_CTRL_IP" "$HOST_CTRL_ADDR" && ctrl_on_data=0 || true
  probe "$CTRL_IF" "$RADIO_DATA_IP" "$HOST_DATA_ADDR" && data_on_ctrl=0 || true
fi

if [ $ctrl_on_ctrl -eq 0 ] && [ $data_on_data -eq 0 ]; then
  # Canonical: enforce the host addressing that goes with it.
  ip addr flush dev "$DATA_IF" 2>/dev/null || true
  ip addr flush dev "$CTRL_IF" 2>/dev/null || true
  ip addr add "$HOST_DATA_ADDR" dev "$DATA_IF"
  ip addr add "$HOST_CTRL_ADDR" dev "$CTRL_IF"
  ip link set "$DATA_IF" up; ip link set "$CTRL_IF" up
  log "TOPOLOGY OK: ${DATA_IF} <-> X410 data (${RADIO_DATA_IP}), ${CTRL_IF} <-> X410 control (${RADIO_CTRL_IP})."
  log "Host addressing enforced: ${DATA_IF}=${HOST_DATA_ADDR}, ${CTRL_IF}=${HOST_CTRL_ADDR}."
  exit 0
fi

if [ $ctrl_on_data -eq 0 ] || [ $data_on_ctrl -eq 0 ]; then
  echo "TOPOLOGY CROSSED: the two X410 cables are swapped relative to the validated setup." >&2
  [ $ctrl_on_data -eq 0 ] && echo "  - X410 CONTROL (${RADIO_CTRL_IP}) answers on ${DATA_IF} (should be ${CTRL_IF})" >&2
  [ $data_on_ctrl -eq 0 ] && echo "  - X410 DATA (${RADIO_DATA_IP}) answers on ${CTRL_IF} (should be ${DATA_IF})" >&2
  echo "  Fix: swap the two QSFP connectors with each other, at EITHER end (host or radio)." >&2
  echo "  (The DPDK data path is pinned to ${DATA_IF} / 0000:01:00.0 in the app configs.)" >&2
  exit 2
fi

echo "RADIO NOT FOUND: neither X410 interface (${RADIO_CTRL_IP}, ${RADIO_DATA_IP}) answers ARP on ${DATA_IF} or ${CTRL_IF}." >&2
echo "  Radio off or still booting (X410 takes ~2-3 min), cables unplugged, or the radio's own IPs changed." >&2
exit 3

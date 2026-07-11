#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
APP_DIR=$(cd "${SCRIPT_DIR}/.." && pwd -P)
source "${SCRIPT_DIR}/container_repo_guard.sh"
EXPECTED_REPO_ROOT=${EXPECTED_REPO_ROOT:-$(expected_repo_root_from_script_dir "${APP_DIR}")}

source "${SCRIPT_DIR}/container_env.sh"

CONTAINER_NAME=${CONTAINER_NAME:-usrp_x410_signal_detection_demo}
HUGEPAGES_DIR=${HUGEPAGES_DIR:-/dev/hugepages}
HUGEPAGE_PAGE_SIZE=${HUGEPAGE_PAGE_SIZE:-1G}
HUGEPAGE_DIR_MODE=${HUGEPAGE_DIR_MODE:-1777}
HUGEPAGES_COUNT=${HUGEPAGES_COUNT:-}
DEFAULT_HUGEPAGES_COUNT=${DEFAULT_HUGEPAGES_COUNT:-3}
PERSIST_BOOT_CONFIG=${PERSIST_BOOT_CONFIG:-0}
RESET_MLX_PORTS=${RESET_MLX_PORTS:-1}
RECREATE_CONTAINER=${RECREATE_CONTAINER:-0}
START_CONTAINER=${START_CONTAINER:-1}
CLEAN_DPDK_STATE=${CLEAN_DPDK_STATE:-1}
XHOST_LOCAL_ROOT=${XHOST_LOCAL_ROOT:-1}
SKIP_IMAGE_BUILD=${SKIP_IMAGE_BUILD:-1}
MLX_PORTS=${MLX_PORTS:-"ens4f0np0 ens4f1np1"}
MLX_DEVICES=${MLX_DEVICES:-"pci/0000:a2:00.0 pci/0000:a2:00.1"}
MLX_PCI_FUNCTIONS=${MLX_PCI_FUNCTIONS:-"0000:a2:00.0 0000:a2:00.1"}
BRING_MLX_PORTS_UP=${BRING_MLX_PORTS_UP:-0}
BUILD_APP_DIR=${BUILD_APP_DIR:-/workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection}
DEFAULT_CONFIG_PATH=${DEFAULT_CONFIG_PATH:-/workspace/holohub/applications/usrp_wideband_signal_detection/config_cuda_dino_performance_single_channel.yaml}

log() {
  echo "==> $*"
}

warn() {
  echo "Warning: $*" >&2
}

die() {
  echo "Error: $*" >&2
  exit 1
}

require_sudo() {
  if ! sudo -n true >/dev/null 2>&1; then
    log "This recovery path needs sudo for host and container setup."
    sudo -v
  fi
}

nvidia_peermem_loaded() {
  lsmod | awk '$1 == "nvidia_peermem" { found = 1 } END { exit found ? 0 : 1 }'
}

container_exists() {
  sudo docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

container_running() {
  [[ "$(sudo docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null || echo false)" == "true" ]]
}

stop_container_if_running() {
  if ! container_exists; then
    return
  fi

  if container_running; then
    log "Stopping ${CONTAINER_NAME} before host-side NIC recovery"
    sudo docker stop "${CONTAINER_NAME}" >/dev/null
  fi
}

hugepages_total() {
  awk '/HugePages_Total:/ {print $2}' /proc/meminfo
}

mount_fstype() {
  findmnt -n -o FSTYPE --target "$1" 2>/dev/null || true
}

validate_positive_integer() {
  local label=$1
  local value=$2

  if ! [[ "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    die "${label} must be a positive integer, got '${value}'"
  fi
}

configured_hugepages_count() {
  if [[ ! -r /etc/sysctl.d/90-holohub-usrp.conf ]]; then
    return 1
  fi

  local configured_count
  configured_count=$(awk -F= '/^[[:space:]]*vm\.nr_hugepages[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2; exit}' \
    /etc/sysctl.d/90-holohub-usrp.conf)

  if [[ -z "${configured_count}" ]]; then
    return 1
  fi

  if [[ "${configured_count}" == "<COUNT>" ]]; then
    warn "Ignoring placeholder vm.nr_hugepages=<COUNT> in /etc/sysctl.d/90-holohub-usrp.conf"
    return 1
  fi

  if ! [[ "${configured_count}" =~ ^[0-9]+$ ]] || (( configured_count < 1 )); then
    warn "Ignoring invalid vm.nr_hugepages='${configured_count}' in /etc/sysctl.d/90-holohub-usrp.conf"
    return 1
  fi

  echo "${configured_count}"
}

resolve_target_hugepages_count() {
  if [[ -n "${HUGEPAGES_COUNT}" ]]; then
    validate_positive_integer "HUGEPAGES_COUNT" "${HUGEPAGES_COUNT}"
    echo "${HUGEPAGES_COUNT}"
    return 0
  fi

  local configured_count
  configured_count=$(configured_hugepages_count || true)
  if [[ -n "${configured_count}" ]]; then
    echo "${configured_count}"
    return 0
  fi

  local current_count
  current_count=$(hugepages_total)
  if [[ "${current_count}" =~ ^[0-9]+$ ]] && (( current_count > 0 )); then
    echo "${current_count}"
    return 0
  fi

  validate_positive_integer "DEFAULT_HUGEPAGES_COUNT" "${DEFAULT_HUGEPAGES_COUNT}"
  log "Falling back to DEFAULT_HUGEPAGES_COUNT=${DEFAULT_HUGEPAGES_COUNT}"
  echo "${DEFAULT_HUGEPAGES_COUNT}"
}

ensure_not_in_container() {
  if [[ -f /.dockerenv ]]; then
    die "Run after_reboot.sh on the host, not inside the demo container."
  fi
}

ensure_nvidia_peermem_loaded() {
  if nvidia_peermem_loaded; then
    log "nvidia-peermem is already loaded"
    return
  fi

  log "Loading nvidia-peermem"
  local modprobe_output=""
  if ! modprobe_output=$(sudo modprobe nvidia-peermem 2>&1); then
    warn "modprobe nvidia-peermem reported: ${modprobe_output}"
    modprobe_output=$(sudo modprobe nvidia_peermem 2>&1 || true)
    if [[ -n "${modprobe_output}" ]]; then
      warn "modprobe nvidia_peermem reported: ${modprobe_output}"
    fi
  fi

  if ! nvidia_peermem_loaded; then
    die "Failed to load nvidia-peermem. Run 'sudo modprobe nvidia-peermem' manually and inspect its error output."
  fi

  log "nvidia-peermem is now loaded"
}

ensure_hugepages_reserved() {
  local current_count
  current_count=$(hugepages_total)
  if ! [[ "${current_count}" =~ ^[0-9]+$ ]]; then
    die "Could not read HugePages_Total from /proc/meminfo"
  fi

  local target_count
  if ! target_count=$(resolve_target_hugepages_count); then
    die "HugePages_Total is 0 and no configured hugepage count was found. Re-run with HUGEPAGES_COUNT=<COUNT>."
  fi

  if (( current_count >= target_count )); then
    log "HugePages_Total=${current_count}"
    return
  fi

  log "Increasing HugePages_Total from ${current_count} to ${target_count}"
  sudo sysctl -w "vm.nr_hugepages=${target_count}" >/dev/null

  current_count=$(hugepages_total)
  if ! [[ "${current_count}" =~ ^[0-9]+$ ]] || (( current_count < target_count )); then
    die "HugePages_Total is ${current_count} after attempting to reserve ${target_count} hugepages"
  fi
}

ensure_hugepages_mounted() {
  sudo mkdir -p "${HUGEPAGES_DIR}"

  if mountpoint -q "${HUGEPAGES_DIR}"; then
    local fstype
    fstype=$(mount_fstype "${HUGEPAGES_DIR}")
    if [[ "${fstype}" != "hugetlbfs" ]]; then
      die "${HUGEPAGES_DIR} is mounted, but not as hugetlbfs"
    fi
    log "hugetlbfs is already mounted at ${HUGEPAGES_DIR}"
    return
  fi

  log "Mounting hugetlbfs at ${HUGEPAGES_DIR}"
  sudo mount -t hugetlbfs -o "pagesize=${HUGEPAGE_PAGE_SIZE}" nodev "${HUGEPAGES_DIR}"
}

ensure_hugepages_permissions() {
  log "Setting ${HUGEPAGES_DIR} permissions to ${HUGEPAGE_DIR_MODE}"
  sudo chown root:root "${HUGEPAGES_DIR}"
  sudo chmod "${HUGEPAGE_DIR_MODE}" "${HUGEPAGES_DIR}"
}

persist_boot_config() {
  if [[ "${PERSIST_BOOT_CONFIG}" != "1" ]]; then
    return
  fi

  local target_count
  target_count=$(resolve_target_hugepages_count)

  log "Persisting reboot-sensitive host config"
  echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf >/dev/null
  printf 'vm.nr_hugepages=%s\n' "${target_count}" | sudo tee /etc/sysctl.d/90-holohub-usrp.conf >/dev/null

  if grep -qE "^[^#].*[[:space:]]${HUGEPAGES_DIR}[[:space:]]+hugetlbfs[[:space:]].*mode=${HUGEPAGE_DIR_MODE}" /etc/fstab; then
    log "Found an existing fstab entry for ${HUGEPAGES_DIR} with mode=${HUGEPAGE_DIR_MODE}"
    return
  fi

  if grep -qE "^[^#].*[[:space:]]${HUGEPAGES_DIR}[[:space:]]+hugetlbfs[[:space:]]" /etc/fstab; then
    warn "An /etc/fstab entry for ${HUGEPAGES_DIR} already exists without mode=${HUGEPAGE_DIR_MODE}; current boot is fixed, but update that line if you want mount permissions to survive reboot exactly."
    return
  fi

  printf 'nodev %s hugetlbfs defaults,pagesize=%s,mode=%s 0 0\n' \
    "${HUGEPAGES_DIR}" "${HUGEPAGE_PAGE_SIZE}" "${HUGEPAGE_DIR_MODE}" | \
    sudo tee -a /etc/fstab >/dev/null
}

reset_mlx_ports_if_requested() {
  if [[ "${RESET_MLX_PORTS}" != "1" ]]; then
    log "Skipping dedicated Mellanox port reset because RESET_MLX_PORTS=${RESET_MLX_PORTS}"
    return
  fi

  log "Resetting dedicated Mellanox ports"
  local port
  for port in ${MLX_PORTS}; do
    sudo ip link set "${port}" down
  done

  local device
  for device in ${MLX_DEVICES}; do
    sudo devlink dev reload "${device}"
  done

  local pci_function
  for pci_function in ${MLX_PCI_FUNCTIONS}; do
    if [[ -w "/sys/bus/pci/devices/${pci_function}/reset" ]]; then
      log "Resetting PCI function ${pci_function}"
      echo 1 | sudo tee "/sys/bus/pci/devices/${pci_function}/reset" >/dev/null
    else
      warn "PCI function reset is not available for ${pci_function}"
    fi
  done

  if command -v udevadm >/dev/null 2>&1; then
    sudo udevadm settle
  fi

  if [[ "${BRING_MLX_PORTS_UP}" == "1" ]]; then
    for port in ${MLX_PORTS}; do
      sudo ip link set "${port}" up || warn "Failed to bring ${port} back up after devlink reload"
    done
  else
    log "Leaving dedicated Mellanox ports administratively down for DPDK ownership"
  fi
}

grant_local_root_display_access() {
  if [[ "${XHOST_LOCAL_ROOT}" != "1" ]]; then
    return
  fi

  if [[ -z "${DISPLAY:-}" ]]; then
    log "DISPLAY is not set; skipping xhost update"
    return
  fi

  if ! command -v xhost >/dev/null 2>&1; then
    warn "xhost is not available; skipping X11 access update"
    return
  fi

  if xhost +local:root >/dev/null 2>&1; then
    log "Granted local root access to the active X server"
  else
    warn "xhost +local:root failed; visualization may still fail until local root is allowed to connect"
  fi
}

detach_mlx_ports_from_network_manager() {
  if ! command -v nmcli >/dev/null 2>&1; then
    return
  fi

  if ! systemctl is-active --quiet NetworkManager 2>/dev/null; then
    return
  fi

  log "Detaching dedicated Mellanox ports from NetworkManager"
  local port
  for port in ${MLX_PORTS}; do
    sudo nmcli device disconnect "${port}" >/dev/null 2>&1 || true
    sudo nmcli device set "${port}" managed no >/dev/null 2>&1 || warn "Failed to mark ${port} unmanaged in NetworkManager"
    sudo ip addr flush dev "${port}" >/dev/null 2>&1 || warn "Failed to flush IP addresses from ${port}"
  done
}

recreate_container_if_requested() {
  if [[ "${RECREATE_CONTAINER}" != "1" ]]; then
    return
  fi

  log "Recreating ${CONTAINER_NAME}"
  (
    cd "${SCRIPT_DIR}"
    SKIP_IMAGE_BUILD="${SKIP_IMAGE_BUILD}" sudo -E ./build_demo_container.sh
  )
}

start_container_if_requested() {
  if [[ "${START_CONTAINER}" != "1" ]]; then
    return
  fi

  if ! container_exists; then
    die "Container ${CONTAINER_NAME} does not exist. Re-run with RECREATE_CONTAINER=1 or create it with ./build_demo_container.sh first."
  fi

  ensure_container_repo_mount_matches "${CONTAINER_NAME}" "${EXPECTED_REPO_ROOT}"

  log "Starting or validating ${CONTAINER_NAME}"
  (
    cd "${SCRIPT_DIR}"
    ./run_demo_container.sh
  )
}

clean_dpdk_runtime_state() {
  if [[ "${CLEAN_DPDK_STATE}" != "1" ]]; then
    return
  fi

  if container_exists; then
    if ! container_running; then
      if [[ "${START_CONTAINER}" == "1" || "${RECREATE_CONTAINER}" == "1" ]]; then
        sudo docker start "${CONTAINER_NAME}" >/dev/null
      else
        log "Container ${CONTAINER_NAME} is not running; skipping in-container DPDK cleanup"
        sudo rm -f "${HUGEPAGES_DIR}"/nwlrbbmqbh*
        return
      fi
    fi

    log "Removing stale DPDK runtime files from ${CONTAINER_NAME}"
    sudo docker exec -u 0:0 "${CONTAINER_NAME}" bash -lc '
pkill -f "(^|/)usrp_wideband_signal_detection( |$)" || true
rm -rf /tmp/xdg-runtime-*/dpdk/*
rm -f /dev/hugepages/nwlrbbmqbh*
'
  else
    log "Container ${CONTAINER_NAME} does not exist yet; skipping in-container DPDK cleanup"
  fi

  log "Removing stale host hugepage files"
  sudo rm -f "${HUGEPAGES_DIR}"/nwlrbbmqbh*
}

print_verification() {
  log "Verification summary"
  lsmod | grep nvidia_peermem || true
  grep -E 'HugePages_Total|HugePages_Free|Hugepagesize' /proc/meminfo || true
  mount | grep ' on /dev/hugepages type hugetlbfs' || true
  ls -ld "${HUGEPAGES_DIR}" || true

  if container_exists; then
    sudo docker inspect -f 'container={{.Name}} running={{.State.Running}}' "${CONTAINER_NAME}" || true
  fi
}

print_next_steps() {
  cat <<EOF

Recovery completed.

Next steps:
  cd ${SCRIPT_DIR}
  ./enter_demo_container.sh

Then inside the container:
  cd ${BUILD_APP_DIR}
  ./usrp_wideband_signal_detection ${DEFAULT_CONFIG_PATH}

Optional flags:
  HUGEPAGES_COUNT=<COUNT> ./after_reboot.sh
  PERSIST_BOOT_CONFIG=1 HUGEPAGES_COUNT=<COUNT> ./after_reboot.sh
  RESET_MLX_PORTS=0 ./after_reboot.sh
  BRING_MLX_PORTS_UP=1 ./after_reboot.sh
  RECREATE_CONTAINER=1 ./after_reboot.sh
EOF
}

main() {
  ensure_not_in_container
  require_sudo
  ensure_nvidia_peermem_loaded
  ensure_hugepages_reserved
  ensure_hugepages_mounted
  ensure_hugepages_permissions
  persist_boot_config
  stop_container_if_running
  detach_mlx_ports_from_network_manager
  reset_mlx_ports_if_requested
  grant_local_root_display_access
  recreate_container_if_requested
  start_container_if_requested
  clean_dpdk_runtime_state
  print_verification
  print_next_steps
}

main "$@"
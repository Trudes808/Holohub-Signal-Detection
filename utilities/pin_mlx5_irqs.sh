#!/bin/bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: sudo ./utilities/pin_mlx5_irqs.sh <pci-bdf> [cpu-list] [--dry-run]

Examples:
  sudo ./utilities/pin_mlx5_irqs.sh 0000:a2:00.1 0-2
  sudo ./utilities/pin_mlx5_irqs.sh 0000:a2:00.1 0,1,2 --dry-run

This script finds all IRQs for the given mlx5 PCI function and pins them to the
requested CPU list by writing /proc/irq/<irq>/smp_affinity_list.
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 1
fi

pci_bdf=""
cpu_list="0-2"
dry_run=0

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            dry_run=1
            ;;
        0000:*)
            pci_bdf="$arg"
            ;;
        *)
            cpu_list="$arg"
            ;;
    esac
done

if [[ -z "$pci_bdf" ]]; then
    echo "error: missing PCI BDF" >&2
    usage
    exit 1
fi

mapfile -t irq_lines < <(python3 - "$pci_bdf" <<'PY'
import re
import sys
from pathlib import Path

pci_bdf = sys.argv[1]
pattern = f"@pci:{pci_bdf}"
for line in Path("/proc/interrupts").read_text().splitlines():
    if pattern not in line:
        continue
    match = re.match(r"\s*(\d+):", line)
    if not match:
        continue
    irq = match.group(1)
    fields = line.split(":", 1)[1].split()
    name = fields[-1]
    total = 0
    for token in fields[:-3]:
        try:
            total += int(token)
        except ValueError:
            pass
    print(f"{irq}\t{name}\t{total}")
PY
)

if [[ ${#irq_lines[@]} -eq 0 ]]; then
    echo "No IRQs found for $pci_bdf" >&2
    exit 1
fi

printf "%-6s %-28s %-14s %-14s %-14s\n" "IRQ" "NAME" "TOTAL" "CURRENT" "TARGET"
for entry in "${irq_lines[@]}"; do
    irq=${entry%%$'\t'*}
    rest=${entry#*$'\t'}
    name=${rest%%$'\t'*}
    total=${entry##*$'\t'}
    current=$(<"/proc/irq/${irq}/smp_affinity_list")
    printf "%-6s %-28s %-14s %-14s %-14s\n" "$irq" "$name" "$total" "$current" "$cpu_list"
done

if [[ $dry_run -eq 1 ]]; then
    exit 0
fi

if [[ $EUID -ne 0 ]]; then
    echo "error: root is required to update IRQ affinity; rerun with sudo" >&2
    exit 1
fi

for entry in "${irq_lines[@]}"; do
    irq=${entry%%$'\t'*}
    echo "$cpu_list" >"/proc/irq/${irq}/smp_affinity_list"
done

echo
echo "Updated IRQ affinities for $pci_bdf to $cpu_list"
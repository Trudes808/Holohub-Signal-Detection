#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "setup_demo_container.sh is deprecated; use ./build_demo_container.sh for initial container creation and provisioning." >&2
exec "${SCRIPT_DIR}/build_demo_container.sh" "$@"
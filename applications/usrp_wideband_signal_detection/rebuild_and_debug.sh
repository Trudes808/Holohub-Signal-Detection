#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "rebuild_and_debug.sh is deprecated; use ./rebuild_demo_container_app.sh, then ./enter_demo_container.sh, and rerun the app manually inside the container." >&2
exec "${SCRIPT_DIR}/rebuild_demo_container_app.sh" "$@"
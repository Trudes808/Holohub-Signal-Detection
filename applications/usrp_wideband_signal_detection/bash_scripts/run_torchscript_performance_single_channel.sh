#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/run_torchscript_performance_test.sh" old_configs/config_torchscript_performance_single_channel.yaml
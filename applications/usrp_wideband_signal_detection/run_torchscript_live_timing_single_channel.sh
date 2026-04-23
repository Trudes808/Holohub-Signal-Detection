#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/run_torchscript_performance_test.sh" config_torchscript_live_timing_single_channel.yaml
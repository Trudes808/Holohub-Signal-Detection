#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec "${SCRIPT_DIR}/run_torchscript_performance_test.sh" config_torchscript_validation_capture_single_channel_2400mhz.yaml
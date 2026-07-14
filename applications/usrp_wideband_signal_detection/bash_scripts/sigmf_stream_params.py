#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 National Instruments Corporation
# SPDX-License-Identifier: Apache-2.0
"""Print USRP_SAMPLE_RATE_HZ / USRP_CENTER_FREQ_HZ from a SigMF recording.

Use this to feed the live/loopback pipeline the stream's true rate/center so it
adapts automatically (no config edit) -- mirrors what the offline eval already
derives from the SigMF. Example (cable-loopback replay of a capture):

    eval "$(bash_scripts/sigmf_stream_params.py my_capture.sigmf-meta)"
    sudo env USRP_SAMPLE_RATE_HZ="$USRP_SAMPLE_RATE_HZ" \
             USRP_CENTER_FREQ_HZ="$USRP_CENTER_FREQ_HZ" \
        ./bash_scripts/run_torchscript_performance_test.sh \
        config_signal_snipper_single_channel.yaml

Accepts either the .sigmf-meta or the .sigmf-data path.
"""
import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sigmf_stream_params.py <file.sigmf-meta|file.sigmf-data>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if path.suffix == ".sigmf-data":
        path = path.with_suffix(".sigmf-meta")
    if not path.exists():
        print(f"SigMF meta not found: {path}", file=sys.stderr)
        return 1

    meta = json.loads(path.read_text())
    g = meta.get("global", {})
    rate = g.get("core:sample_rate")
    captures = meta.get("captures", [{}])
    center = captures[0].get("core:frequency", 0.0) if captures else 0.0

    if rate is None:
        print("SigMF global is missing core:sample_rate", file=sys.stderr)
        return 1

    # Shell-eval-able assignments.
    print(f"USRP_SAMPLE_RATE_HZ={rate:g}")
    print(f"USRP_CENTER_FREQ_HZ={center:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

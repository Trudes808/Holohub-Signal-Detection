#!/usr/bin/env python3
"""Demo conductor: honors the dashboard's DEMO CONTROLS selections.

Watches the demo_control.json the visualizer writes and:
  - snr      -> switches which replay pcap tcpreplay loops on the loopback
                cable (clean composite / single-SNR holds / staircase)
  - detector -> restarts the app container-side with the other pipeline
                config (coherent_power <-> cuda_dino); the dashboard blinks
                for the restart, selections persist via the control file
  - gate     -> handled directly by rt_decode_daemon.py (not this script)

Run on the host with sudo (tcpreplay + docker):
    sudo python3 demo_conductor.py [--dry]

The gate switch is instant; an SNR switch has a ~1 s gap while tcpreplay
restarts; a detector switch takes ~15 s (app restart).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

COMPOSITES = "/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites"  # fixed: ~ expands to /root under sudo
PCAP_BY_SNR = {
    "clean": "comprehensive_4class_py.pcap",
    "staircase": "snr_staircase_4class.pcap",
    **{db: f"snr_single_{db}db.pcap" for db in ("30", "20", "15", "12", "9", "6")},
}
CONFIG_BY_DETECTOR = {
    "coherent_power": "config_signal_snipper_single_channel.yaml",
    "cuda_dino": "config_cuda_dino_performance_single_channel.yaml",  # live air (bench-era floors)
}
# Loopback replay runs at the composites' rate, which is what the DINO
# coherence gate was calibrated against (2026-08-18) — use the calibrated
# config there; per-frequency floors are span-specific so live air keeps
# the base config.
CUDA_DINO_CALIBRATED = ("/workspace/holohub/applications/usrp_wideband_signal_detection/"
                        "calibration/config_cuda_dino_coherence_calibrated_single_channel.yaml")
CONTAINER = os.environ.get("CONTAINER_NAME", "usrp_x410_sig_det_sat3737")
BUILD_APP_DIR = ("/workspace/holohub/build/usrp_wideband_signal_detection/"
                 "applications/usrp_wideband_signal_detection")


class Conductor:
    def __init__(self, args):
        self.args = args
        self.replay: subprocess.Popen | None = None
        self.snr = None
        self.detector = None

    def run_or_print(self, cmd, **kw):
        print(f"[conductor] {'DRY: ' if self.args.dry else ''}{' '.join(cmd)}", flush=True)
        if not self.args.dry:
            return subprocess.Popen(cmd, **kw) if kw.pop("background", False) else \
                subprocess.run(cmd, check=False)
        return None

    def set_snr(self, snr: str):
        if self.args.no_replay:
            if snr != self.snr:
                print(f"[conductor] snr '{snr}' noted — replay disabled (--no-replay: "
                      "live-radio mode; SNR selection drives the loopback pcaps only)",
                      flush=True)
            self.snr = snr
            return
        pcap = PCAP_BY_SNR.get(snr)
        if pcap is None:
            print(f"[conductor] unknown snr '{snr}', ignoring", flush=True)
            return
        path = os.path.join(self.args.pcap_dir, pcap)
        if not os.path.exists(path):
            print(f"[conductor] missing pcap {path}, ignoring", flush=True)
            return
        if self.replay is not None and self.replay.poll() is None:
            print("[conductor] stopping current tcpreplay", flush=True)
            if not self.args.dry:
                self.replay.send_signal(signal.SIGINT)
                try:
                    self.replay.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.replay.kill()
        cmd = ["tcpreplay", "--preload-pcap", "--loop", "0", "--pps", "240000",
               "-i", self.args.iface, path]
        print(f"[conductor] {'DRY: ' if self.args.dry else ''}{' '.join(cmd)}", flush=True)
        if not self.args.dry:
            self.replay = subprocess.Popen(cmd)
        self.snr = snr

    def set_detector(self, detector: str):
        cfg = CONFIG_BY_DETECTOR.get(detector)
        if cfg is None:
            print(f"[conductor] unknown detector '{detector}', ignoring", flush=True)
            return
        if detector == "cuda_dino" and not self.args.no_replay:
            cfg = CUDA_DINO_CALIBRATED
        self.run_or_print(["docker", "exec", CONTAINER, "bash", "-lc",
                           "pkill -f '(^|/)usrp_wideband_signal_detection( |$)' || true"])
        time.sleep(2 if not self.args.dry else 0)
        # mirror run_live_demo.sh's launch env: stream rate/center + XDG dir
        self.run_or_print(["docker", "exec", "-d",
                           "-e", f"DISPLAY={self.args.display}",
                           "-e", f"USRP_SAMPLE_RATE_HZ={self.args.rate_hz:.0f}",
                           "-e", f"USRP_CENTER_FREQ_HZ={self.args.center_hz:.0f}",
                           CONTAINER, "bash", "-lc",
                           "mkdir -p /tmp/xdg-runtime-root && chmod 700 /tmp/xdg-runtime-root && "
                           "export XDG_RUNTIME_DIR=/tmp/xdg-runtime-root && "
                           f"cd {BUILD_APP_DIR} && exec ./usrp_wideband_signal_detection {cfg} "
                           f"> /workspace/spectrograms/demo_app.log 2>&1"])
        self.detector = detector

    def watch(self):
        print(f"[conductor] watching {self.args.control} (iface {self.args.iface}, "
              f"pcaps {self.args.pcap_dir}, container {CONTAINER}"
              f"{', DRY RUN' if self.args.dry else ''})", flush=True)
        mtime = 0.0
        while True:
            try:
                mt = os.path.getmtime(self.args.control)
                if mt != mtime:
                    mtime = mt
                    ctl = json.load(open(self.args.control))
                    snr = str(ctl.get("snr", "clean"))
                    det = str(ctl.get("detector", "coherent_power"))
                    if snr != self.snr:
                        self.set_snr(snr)
                    if self.detector is not None and det != self.detector:
                        self.set_detector(det)
                    elif self.detector is None:
                        self.detector = det   # adopt initial state, no restart
            except (OSError, json.JSONDecodeError, ValueError):
                pass
            time.sleep(self.args.poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", default="/tmp/usrp_spectrograms/demo_control.json")
    ap.add_argument("--pcap-dir", default=COMPOSITES)
    ap.add_argument("--iface", default="enP2p1s0f0np0")
    ap.add_argument("--display", default=os.environ.get("DISPLAY", ":1"))
    ap.add_argument("--rate-hz", type=float, default=491.52e6,
                    help="stream rate the relaunched app adopts "
                         "(491.52e6 live radio, 245.76e6 loopback replay)")
    ap.add_argument("--center-hz", type=float, default=2.4e9)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--dry", action="store_true", help="print actions instead of executing")
    ap.add_argument("--no-replay", action="store_true",
                    help="live-radio mode: honor detector switches but ignore SNR "
                         "selections (those drive the loopback replay pcaps)")
    args = ap.parse_args()
    c = Conductor(args)
    try:
        c.watch()
    except KeyboardInterrupt:
        if c.replay is not None and c.replay.poll() is None:
            c.replay.send_signal(signal.SIGINT)
        print("\n[conductor] stopped", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

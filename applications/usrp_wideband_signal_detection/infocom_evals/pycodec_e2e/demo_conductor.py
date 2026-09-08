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
    "clean": "snr_clean_burst.pcap",   # noise-free staircase-family capture (same bands/bursts); the composer composite pcap remains for manual replay
    "staircase": "snr_staircase_4class.pcap",
    **{db: f"snr_single_{db}db.pcap" for db in ("30", "20", "15", "12", "9", "6", "0")},
    "-5": "snr_single_m5db.pcap",
    "-10": "snr_single_m10db.pcap",
}
CONFIG_BY_DETECTOR = {
    "coherent_power": "config_snipper_viz_demo.yaml",  # viz-enabled; the plain snipper config is HEADLESS
    "cuda_dino": "config_cuda_dino_performance_single_channel.yaml",  # live air (bench-era floors)
    # Fine-tuned DINOv3 segmenter (M2_dr): its emit_stride=16 handles the GB10 compute budget,
    # so replay stays at the full 240 kpps (fresh mask every ~340 ms).
    "cuda_dino_finetuned": "config_dino_finetuned_viz_demo.yaml",
}
# Live radio (--no-replay): coherent_power uses the v3 live config (dynamic
# floor + snipper + compression). NOTE the DINO-FT M2_dr checkpoint was trained
# for 20.48-245.76 MS/s; at the live 491.52 MS/s it runs outside its
# rate-invariance envelope (expect degraded masks until a retrain).
CONFIG_BY_DETECTOR_LIVE = {
    "coherent_power": "config_live_v3_single_channel.yaml",
    "cuda_dino": "config_live_v3_dino.yaml",
    "cuda_dino_finetuned": "config_live_v3_dino_ft.yaml",
    # Same detector + ignore_sideband_percent 3.0: trims the band-edge mask
    # columns the rolloff cliff fires (dashboard A/B against the base variant).
    "cuda_dino_finetuned_sb": "config_live_v3_dino_ft_sb.yaml",
}
# Loopback replay of dense composites with the v3 dashboard (run_loopback_v3_demo.sh).
# Same 4 detectors as live, but coherent_power's emit-occupancy guard is OFF (dense
# composites legitimately exceed the live 0.35 cap; the guard is a live garbage-flood
# defense and the loopback transport is clean). The DINO configs are reused verbatim
# (no emit guard) and, at the composite's 245.76 MSps, DINO-FT runs at its trained
# native geometry.
CONFIG_BY_DETECTOR_LOOPBACK = {
    "coherent_power": "config_loopback_v3_single_channel.yaml",
    "cuda_dino": "config_loopback_v3_dino.yaml",
    "cuda_dino_finetuned": "config_loopback_v3_dino_ft.yaml",
    "cuda_dino_finetuned_sb": "config_loopback_v3_dino_ft_sb.yaml",
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
        self.current_cfg = None
        self.pps = 240000

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
        cmd = ["tcpreplay", "--preload-pcap", "--loop", "0", "--pps", str(self.pps),
               "-i", self.args.iface, path]
        print(f"[conductor] {'DRY: ' if self.args.dry else ''}{' '.join(cmd)}", flush=True)
        if not self.args.dry:
            self.replay = subprocess.Popen(cmd)
        self.snr = snr

    def _launch_app(self, cfg: str):
        # Kill the old app, WAIT for its DPDK teardown, and clear the EAL
        # runtime state — relaunching after a fixed 2 s hit "EAL: Cannot
        # create lock on /var/run/dpdk/nwlrbbmqbh/config" (old process still
        # holding the primary lock) and the new app died instantly, which
        # read on the dashboard as "detector switch does nothing".
        self.run_or_print(["docker", "exec", CONTAINER, "bash", "-lc",
                           "pkill -f '(^|/)usrp_wideband_signal_detection( |$)' || true; "
                           "for i in $(seq 1 30); do "
                           "  pgrep -f '(^|/)usrp_wideband_signal_detection( |$)' >/dev/null || break; "
                           "  sleep 1; done; "
                           "rm -f /dev/hugepages/nwlrbbmqbh* 2>/dev/null; "
                           "rm -rf /var/run/dpdk/nwlrbbmqbh 2>/dev/null || true"])
        time.sleep(1 if not self.args.dry else 0)
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

    def _app_alive(self) -> bool:
        if self.args.dry:
            return True
        r = subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc",
                            "pgrep -f '(^|/)usrp_wideband_signal_detection( |$)' >/dev/null"],
                           check=False)
        return r.returncode == 0

    def config_map(self):
        # --loopback: v3 dashboard fed by loopback replay (external tcpreplay) -> the
        # loopback config set (coherent guard off). Implies the --no-replay app-launch
        # path (we drive the pcap, not the conductor).
        if getattr(self.args, "loopback", False):
            return CONFIG_BY_DETECTOR_LOOPBACK
        return CONFIG_BY_DETECTOR_LIVE if self.args.no_replay else CONFIG_BY_DETECTOR

    def set_detector(self, detector: str):
        cfg = self.config_map().get(detector)
        if cfg is None:
            print(f"[conductor] unknown detector '{detector}', ignoring", flush=True)
            return
        if detector == "cuda_dino" and not self.args.no_replay:
            cfg = CUDA_DINO_CALIBRATED
        # DINO cannot sustain the full-rate replay on GB10 (the converter's
        # degraded-shutdown watchdog exits the app) — halve the packet rate
        want_pps = 120000 if detector == "cuda_dino" else 240000
        if want_pps != self.pps and not self.args.no_replay:
            self.pps = want_pps
            print(f"[conductor] replay rate -> {self.pps} pps for {detector}", flush=True)
            if self.snr is not None:
                snr, self.snr = self.snr, None
                self.set_snr(snr)
        prev_cfg = self.current_cfg
        self._launch_app(cfg)
        self.current_cfg = cfg
        # survival check: if the new pipeline dies (e.g. can't keep up), roll
        # back instead of leaving the demo dead with no window
        if not self.args.dry:
            time.sleep(10)
            if not self._app_alive():
                print(f"[conductor] APP DIED after switching to {detector} ({cfg}) — "
                      f"rolling back to {prev_cfg or self.config_map()['coherent_power']}",
                      flush=True)
                # preserve the dying app's log before the rollback launch
                # overwrites it (post-mortems were impossible without this)
                self.run_or_print(["docker", "exec", CONTAINER, "bash", "-lc",
                                   "cp /workspace/spectrograms/demo_app.log "
                                   f"/workspace/spectrograms/demo_app_died_{detector}.log || true"])
                self.pps = 240000
                if self.snr is not None:
                    snr, self.snr = self.snr, None
                    self.set_snr(snr)
                self._launch_app(prev_cfg or self.config_map()["coherent_power"])
                self.current_cfg = prev_cfg or self.config_map()["coherent_power"]
                return
        self.detector = detector

    def watch(self):
        print(f"[conductor] watching {self.args.control} (iface {self.args.iface}, "
              f"pcaps {self.args.pcap_dir}, container {CONTAINER}"
              f"{', DRY RUN' if self.args.dry else ''})", flush=True)
        mtime = 0.0
        alive_check = 0.0
        while True:
            # keep-alive: ESC or the window X gracefully kills the app (HoloViz
            # quits on ESC even when it was aimed at a dropdown) — revive it.
            # Intentional stops kill this conductor first (stop script order).
            now = time.time()
            if self.detector is not None and now - alive_check > 5.0:
                alive_check = now
                if not self._app_alive():
                    cfg = self.current_cfg or self.config_map().get(
                        self.detector, self.config_map()["coherent_power"])
                    print(f"[conductor] app not running — reviving with {cfg}", flush=True)
                    self._launch_app(cfg)
                    self.current_cfg = cfg
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
    ap.add_argument("--loopback", action="store_true",
                    help="v3 dashboard fed by EXTERNAL loopback replay (run_loopback_v3_demo.sh "
                         "drives tcpreplay): use the loopback config set (coherent emit guard off) "
                         "and the direct app-launch path. Implies --no-replay.")
    args = ap.parse_args()
    if args.loopback:
        args.no_replay = True  # we don't manage tcpreplay; the launcher does
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

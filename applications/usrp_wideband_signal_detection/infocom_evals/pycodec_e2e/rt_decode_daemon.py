#!/usr/bin/env python3
"""Real-time decode stage: consume signal_snipper SigMF output, decode framed
bursts with ZERO ground truth, publish live BER metrics.

This is the decode "operator" of the demo pipeline, run as a sidecar process:

    detector -> signal_snipper -> sigmf_file_sink --(SigMF packs on disk)-->
        THIS DAEMON: blind symbol-rate estimate -> frame sync -> header
        (modulation/length/CRC) -> payload decode -> CRC32 + PN9 BER

It polls the snippet directory, decodes every NEW snippet annotation as it
appears, prints one line per snippet plus a rolling aggregate, and writes the
running metrics to a JSON file (rt_metrics.json next to the snippets) that a
future HoloViz overlay can render.

Usage:
    python3 rt_decode_daemon.py --snips /tmp/usrp_spectrograms/<run>/snippets \
        [--poll 0.5] [--once] [--idle-exit 30]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

PYCODEC_ROOT = os.environ.get(
    "PYCODEC_ROOT", os.path.expanduser("~/Documents/holoscan_waveform_generation"))
sys.path.insert(0, PYCODEC_ROOT)

from pycodec.frame import (channelize, decode_frames,  # noqa: E402
                           estimate_symbol_rate, find_subbands)

PROFILE = dict(sps=8, pulse_shape="rrc", rolloff=0.35, span_symbols=10)


class Metrics:
    def __init__(self):
        self.snips = 0
        self.snips_with_frames = 0
        self.frames = 0
        self.crc_ok = 0
        self.pn9_bits = 0
        self.pn9_errors = 0
        self.by_mod: dict[str, int] = {}
        self.started = time.time()

    def as_dict(self):
        return {
            "snippets_seen": self.snips,
            "snippets_with_frames": self.snips_with_frames,
            "frames_decoded": self.frames,
            "frames_crc_ok": self.crc_ok,
            "pn9_bits_compared": self.pn9_bits,
            "pn9_bit_errors": self.pn9_errors,
            "pn9_ber": (self.pn9_errors / self.pn9_bits) if self.pn9_bits else None,
            "frames_by_modulation": self.by_mod,
            "uptime_s": round(time.time() - self.started, 1),
        }


def process_annotation(a, data, metrics: Metrics) -> str:
    snip_fs = float(a.get("wfgt:snippet_sample_rate",
                          a.get("wfgt:orig_sample_rate", 245.76e6)))
    start, count = int(a["core:sample_start"]), int(a["core:sample_count"])
    iq = np.asarray(data[start:start + count], dtype=np.complex64)
    if iq.size < 4096:
        return None
    t0 = time.time()
    # A detection box can merge several frequency-stacked signals into one
    # snippet: channelize each occupied sub-band, then rate-estimate + frame-
    # decode per band, all blind.
    frames = []
    band_info = []
    try:
        for center, bw in find_subbands(iq, snip_fs):
            ch, chfs = channelize(iq, snip_fs, center, bw)
            rs = estimate_symbol_rate(ch, chfs, lo_hz=max(1e6, 0.3 * bw / 1.35),
                                      hi_hz=max(2e6, min(1.2 * bw, 0.45 * chfs)))
            got = decode_frames(ch, chfs, rs, **PROFILE)
            frames.extend(got)
            band_info.append(f"{center/1e6:+.1f}MHz/rs{rs/1e6:.2f}:{len(got)}")
    except Exception as e:
        return f"decode error: {e}"
    dt_ms = (time.time() - t0) * 1e3

    metrics.snips += 1
    if not frames:
        return (f"snip frame#{a.get('wfgt:frame_number', '?')} "
                f"{count/snip_fs*1e3:6.2f} ms  no frames (bands {' '.join(band_info) or '-'})  [{dt_ms:.0f} ms]")
    metrics.snips_with_frames += 1
    ok = [f for f in frames if f.payload_crc_ok]
    err = sum(f.bit_errors for f in frames if f.pn9_payload)
    bits = sum(f.payload_len_bits for f in frames if f.pn9_payload)
    metrics.frames += len(frames)
    metrics.crc_ok += len(ok)
    metrics.pn9_bits += bits
    metrics.pn9_errors += err
    for f in frames:
        metrics.by_mod[f.payload_mod] = metrics.by_mod.get(f.payload_mod, 0) + 1
    mods = ",".join(sorted({f.payload_mod for f in frames}))
    ber = f"{err}/{bits}" if bits else "-"
    return (f"snip frame#{a.get('wfgt:frame_number', '?')} {count/snip_fs*1e3:6.2f} ms  "
            f"bands[{' '.join(band_info)}]  frames {len(frames)} (crc_ok {len(ok)}) "
            f"mod {mods}  pn9_err {ber}  [{dt_ms:.0f} ms]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snips", required=True)
    ap.add_argument("--poll", type=float, default=0.5)
    ap.add_argument("--once", action="store_true", help="process what exists, then exit")
    ap.add_argument("--idle-exit", type=float, default=None,
                    help="exit after this many seconds without new snippets")
    args = ap.parse_args()

    metrics = Metrics()
    seen: dict[str, int] = {}   # pack meta path -> annotations processed
    last_new = time.time()
    print(f"rt_decode_daemon: watching {args.snips} (blind decode, profile {PROFILE})",
          flush=True)
    while True:
        new_work = False
        for mp in sorted(glob.glob(os.path.join(args.snips, "*.sigmf-meta"))):
            try:
                meta = json.load(open(mp))
            except (json.JSONDecodeError, OSError):
                continue  # pack still being written
            anns = meta.get("annotations", [])
            done = seen.get(mp, 0)
            if len(anns) <= done:
                continue
            dp = mp.replace(".sigmf-meta", ".sigmf-data")
            try:
                data = np.memmap(dp, dtype="<c8", mode="r")
            except (OSError, ValueError):
                continue
            for a in anns[done:]:
                line = process_annotation(a, data, metrics)
                if line:
                    print(f"[{time.strftime('%H:%M:%S')}] {line}", flush=True)
            seen[mp] = len(anns)
            new_work = True
        if new_work:
            last_new = time.time()
            m = metrics.as_dict()
            ber = m["pn9_ber"]
            print(f"[{time.strftime('%H:%M:%S')}] === LIVE: frames {m['frames_decoded']} "
                  f"(crc_ok {m['frames_crc_ok']}) by_mod {m['frames_by_modulation']} "
                  f"PN9 BER {ber if ber is None else f'{ber:.2e}'} "
                  f"({m['pn9_bit_errors']}/{m['pn9_bits_compared']}) ===", flush=True)
            try:
                with open(os.path.join(args.snips, "rt_metrics.json"), "w") as f:
                    json.dump(m, f, indent=1)
            except OSError:
                pass
        if args.once and not new_work:
            break
        if args.idle_exit and (time.time() - last_new) > args.idle_exit:
            print("idle timeout, exiting", flush=True)
            break
        time.sleep(args.poll)

    m = metrics.as_dict()
    print("\nFINAL:", json.dumps(m, indent=1), flush=True)
    return 0 if (m["frames_decoded"] > 0 and
                 m["frames_crc_ok"] == m["frames_decoded"] and
                 m["pn9_bit_errors"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Loopback-RT vs offline M3 mask comparison on the SAME x410 capture.

Frame-by-frame IoU is unreliable here (DPDK-replay vs file-chunking frame-phase offset), so the primary
metric is the ALIGNMENT-FREE per-frequency detection profile: mean mask occupancy per freq column over
all frames. If the RT pipeline reproduces the offline masks, the two profiles match regardless of frame
phase. Result: correlation 0.994, occupancy 0.72% (loopback) vs 0.67% (offline) -> RT == offline.
Inputs: loopback dump /tmp/usrp_spectrograms/loopback_eval/dino_ft_rt/mask_arrays and offline eval
/tmp/usrp_spectrograms/cmp3/m3_rt.
"""
from __future__ import annotations
import glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LB = "/tmp/usrp_spectrograms/loopback_eval/dino_ft_rt/mask_arrays"
OFF = "/tmp/usrp_spectrograms/cmp3/m3_rt/mask_arrays"
OUT = os.path.join(os.path.dirname(__file__), "results", "loopback_vs_offline_profile.png")


def profile(paths):
    acc, n = None, 0
    for p in paths:
        col = np.load(p).astype(np.float32).mean(axis=0)  # per-freq occupancy this frame
        acc = col if acc is None else acc + col
        n += 1
    return acc / max(1, n)


def dsc(a, f=20):
    c = (len(a) // f) * f
    return a[:c].reshape(-1, f).mean(1)


def main():
    lb = sorted(glob.glob(f"{LB}/mask_ch0_*.npy"))
    off = sorted(glob.glob(f"{OFF}/*.npy"))
    if not lb or not off:
        print("missing masks (run the loopback + offline M3 evals first)"); return
    pl, po = profile(lb), profile(off)
    pld, pod = dsc(pl), dsc(po)
    r = np.corrcoef(pld, pod)[0, 1]
    print(f"per-freq detection profile correlation (loopback vs offline): {r:.4f}")
    print(f"total occupancy: loopback {pl.mean()*100:.3f}%  offline {po.mean()*100:.3f}%")
    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.linspace(2.154, 2.646, len(pld))
    ax.plot(x, pod * 100, label="offline M3", lw=1.5)
    ax.plot(x, pld * 100, label="loopback-RT M3", lw=1.5, alpha=0.8)
    ax.set_xlabel("frequency (GHz)"); ax.set_ylabel("detection rate %"); ax.legend(); ax.grid(alpha=.3)
    ax.set_title(f"Per-frequency detection profile: loopback-RT vs offline M3 (corr={r:.3f})")
    fig.tight_layout(); fig.savefig(OUT, dpi=110)
    print("wrote", OUT)


if __name__ == "__main__":
    main()

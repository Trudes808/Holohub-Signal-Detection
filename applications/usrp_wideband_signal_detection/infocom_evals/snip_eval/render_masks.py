#!/usr/bin/env python3
"""Render detector masks (bit-packed .packed.npz, 512x10240) as time x freq images, to see WHAT each
detector flags at low vs high SNR. rows = time (0..~21.3 ms), cols = freq (-fs/2..+fs/2). Yellow = detected.

Usage: python3 render_masks.py   (defaults: coherent_power + finetuned_dino_m2 @ atten 70 vs 0)
"""
from __future__ import annotations
import numpy as np, glob
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SE = Path(__file__).resolve().parent
OUT = SE / "figs_minsize"; OUT.mkdir(exist_ok=True)
FS = 245.76e6
FRAME_MS = 10240 / FS * 512 * 1e3        # ~21.33 ms per frame
EXT = [-FS/2/1e6, FS/2/1e6, FRAME_MS, 0]

def load(f):
    z = np.load(f); return np.unpackbits(z["packed"])[:int(z["rows"])*int(z["cols"])].reshape(int(z["rows"]), int(z["cols"]))

def masks(det, atten):
    return sorted(glob.glob(str(SE / f"snip_run/{det}/attenuation_dB_{atten}/mask_arrays/*.packed.npz")))

def frame(det, atten, fr):
    fl = masks(det, atten)
    cand = [f for f in fl if f"_f{fr}_" in f]
    return cand[0] if cand else (fl[0] if fl else None)

# (detector, atten, SNR dB, frame, subtitle)
panels = [
    ("coherent_power",    70, -16, 100, "Coherent Power @ -16 dB"),
    ("coherent_power",    70, -16, 160, "Coherent Power @ -16 dB"),
    ("coherent_power",     0,  54, 100, "Coherent Power @ +54 dB"),
    ("finetuned_dino_m2", 70, -16, 100, "DINO FT @ -16 dB"),
    ("finetuned_dino_m2", 70, -16, 160, "DINO FT @ -16 dB"),
    ("finetuned_dino_m2",  0,  54, 100, "DINO FT @ +54 dB"),
]
fig, axs = plt.subplots(2, 3, figsize=(16, 7)); axs = axs.ravel()
for ax, (det, at, snr, fr, sub) in zip(axs, panels):
    f = frame(det, at, fr)
    if f is None:
        ax.set_visible(False); continue
    m = load(f); cov = m.mean() * 100
    ax.imshow(m, aspect="auto", cmap="magma", extent=EXT, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{sub}  (frame {fr})\n{cov:.1f}% of pixels flagged", fontsize=9)
    ax.set_xlabel("freq (MHz)"); ax.set_ylabel("time (ms)")
fig.suptitle("Detector masks — LOW SNR (−16 dB) is sparse for BOTH; coherent's few hits are wide, solid, "
             "persistent bars (clear the 100 kHz/5 ms gate). At HIGH SNR coherent floods the band.",
             fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "masks_coherent_lowsnr.png", dpi=150)
print("wrote", OUT / "masks_coherent_lowsnr.png")

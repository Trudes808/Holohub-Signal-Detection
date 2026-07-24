#!/usr/bin/env python3
"""Before/after figure for the implemented min_mask_bandwidth_hz fix: real measured snip footprint
(100 kHz/5 ms gate) from real_snip_metrics_minsize.csv (before, spur-fusion artifact present) vs
real_snip_metrics_minsize_v2.csv (after, mask pre-filter enabled), coherent_power vs
finetuned_dino_m2, both snipper modes. -> figs_minsize/maskfilter_before_after.png

Run: ~/miniforge3/envs/dinov3/bin/python plot_maskfilter_before_after.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SE = Path(__file__).resolve().parent
SNR0 = 54
FLOOR = 5e-4  # display floor for true zeros on the log axis


def series(csv_path, mode, det):
    pts = []
    for r in csv.DictReader(open(csv_path)):
        if r["mode"] == mode and r["detector"] == det and r["attenuation_db"]:
            pts.append((SNR0 - int(r["attenuation_db"]),
                        max(float(r["decimated_TB_per_hour"]), FLOOR)))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


before, after = SE / "real_snip_metrics_minsize.csv", SE / "real_snip_metrics_minsize_v2.csv"
fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, mode, title in [(axs[0], "frequency", "frequency mode (snip + resample)"),
                        (axs[1], "time_only", "time_only mode (full-band time slices)")]:
    x, y = series(before, mode, "coherent_power")
    ax.plot(x, y, "o--", color="#d62728", alpha=0.8, label="coherent BEFORE (spur-fusion artifact)")
    x, y = series(after, mode, "coherent_power")
    ax.plot(x, y, "o-", color="#2ca02c", lw=2, label="coherent AFTER (min_mask_bandwidth_hz)")
    x, y = series(after, mode, "finetuned_dino_m2")
    ax.plot(x, y, "^-", color="#7f7f7f", alpha=0.8, label="DINO-FT m2 (reference, unchanged)")
    ax.axhline(7.078, color="k", lw=1, alpha=0.5)
    ax.text(ax.get_xlim()[0], 7.078, " save-all 7.08 TB/hr", fontsize=8, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB) = 54 − attenuation")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, which="both")
axs[0].set_ylabel("REAL measured stored footprint (TB/hr, log)")
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Real signal_snipper footprint, 100 kHz/5 ms gate, before vs after the pre-labeling "
             "mask filter\n(points at 5e-4 are true zeros; the low-SNR plateau was the 2048 MHz "
             "receiver spur fused into full-height boxes)", fontsize=10)
fig.tight_layout()
fig.savefig(SE / "figs_minsize" / "maskfilter_before_after.png", dpi=150)
print("wrote", SE / "figs_minsize" / "maskfilter_before_after.png")

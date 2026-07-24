#!/usr/bin/env python3
"""Summary figure for fix_quantification.csv: coherent_power's snip footprint vs SNR, current
snipper behavior vs the persistent-column split (and spur notch for time mode), with
finetuned_dino_m2 as the honest reference. -> figs_minsize/fix_quantification.png

Run: ~/miniforge3/envs/dinov3/bin/python plot_fix_quantification.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SE = Path(__file__).resolve().parent
SNR0 = 54  # snr_db = 54 - attenuation_db
FLOOR = 5e-4  # display floor for true zeros on the log axis

rows = list(csv.DictReader(open(SE / "fix_quantification.csv")))


def series(det, gate, strat, col):
    pts = sorted((SNR0 - int(r["attenuation_db"]), max(float(r[col]), FLOOR))
                 for r in rows if r["detector"] == det and r["gate"] == gate
                 and r["strategy"] == strat)
    return [p[0] for p in pts], [p[1] for p in pts]


fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
for ax, mode_col, title in [(axs[0], "freq_TB_hr", "frequency mode (snip + resample)"),
                            (axs[1], "time_TB_hr", "time_only mode (full-band time slices)")]:
    for gate, ls in [("minsize_100k_5ms", "-"), ("default", "--")]:
        glabel = "100kHz/5ms gate" if gate == "minsize_100k_5ms" else "no size gate"
        x, y = series("coherent_power", gate, "current", mode_col)
        ax.plot(x, y, "o" + ls, color="#d62728", label=f"coherent CURRENT ({glabel})")
        x, y = series("coherent_power", gate, "split", mode_col)
        ax.plot(x, y, "s" + ls, color="#2ca02c", label=f"coherent SPLIT ({glabel})")
        if mode_col == "time_TB_hr" and gate == "default":
            x, y = series("coherent_power", gate, "suppress", mode_col)
            ax.plot(x, y, "d:", color="#17becf", label="coherent SPLIT + spur notch (no gate)")
        x, y = series("finetuned_dino_m2", gate, "current", mode_col)
        ax.plot(x, y, "^" + ls, color="#7f7f7f", alpha=0.8, label=f"DINO-FT m2 ({glabel})")
    ax.axhline(7.078, color="k", lw=1, alpha=0.5)
    ax.text(ax.get_xlim()[0], 7.078, " save-all 7.08 TB/hr", fontsize=8, va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("SNR (dB) = 54 − attenuation")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3, which="both")
axs[0].set_ylabel("stored footprint (TB/hr, log)")
axs[0].legend(fontsize=7.5, loc="lower right")
axs[1].legend(fontsize=7.5, loc="lower right")
fig.suptitle("Persistent-column split removes the spur-fusion artifact\n"
             "(with a size gate coherent_power hits the same ~0 as DINO-FT; gate-free time mode also "
             "needs the calibrated spur notch; points at 5e-4 are true zeros)", fontsize=10)
fig.tight_layout()
fig.savefig(SE / "figs_minsize" / "fix_quantification.png", dpi=150)
print("wrote", SE / "figs_minsize" / "fix_quantification.png")

#!/usr/bin/env python3
"""Mixed-source storage figure in the notebook Figure-1 house style.

One figure, both collection strategies, curated detector mix pulled from TWO measurement CSVs:
  - Naive save-all           : computed (245.76 MHz cf32 -> 7,078 GB/hr, flat, SNR-independent)
  - Ground truth  (ceilings) : real_snip_metrics.csv        (both strategies, flat mean over SNR)
  - DINO FT (M2, relabeled)  : real_snip_metrics.csv        (both strategies, curve vs SNR)
  - Coherent Power           : real_snip_metrics_75k_v2.csv (both strategies, curve vs SNR)
                               ^ the min_mask_bandwidth_hz-fixed run (75 kHz / 1 ms gate), so the
                                 receiver-clock-spur over-collection is gone -> ~0 stored at low SNR.

Two lines per detector: solid = time slice (time_only mode); dashed = time slice + downsample
(frequency mode -- was called "snip": it keeps the time slice AND resamples each box down to its
bandwidth). The solid/dashed = strategy rule holds for every line, GT ceilings included. Coherent
Power = blue circles, DINO FT = brown crosses. Y = GB/hour, log (plain-number ticks); X = SNR (dB).

Provenance note: GT and DINO FT come from the default-gate sweep; Coherent Power from the 75 kHz/1 ms
mask-filtered sweep. GT/DINO are gate-robust (real signals, no spur fusion), so this mix is a fair
apples-to-apples storage comparison; Coherent is shown at the operating point where the fix applies.
All three mask sets are the same staged July batch family (see mask_provenance.md).

Run: ~/miniforge3/envs/dinov3/bin/python plot_mixed_storage.py   ->  figs/fig_mixed_storage_vs_snr.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

DS = Path(__file__).resolve().parent
OUT = DS / "figs"; OUT.mkdir(exist_ok=True)
BASE = DS / "real_snip_metrics.csv"           # ground_truth + finetuned_dino_m2
K75 = DS / "real_snip_metrics_75k_v2.csv"     # coherent_power (mask-filter fixed, 75 kHz/1 ms)

SNR0 = float(json.load(open(DS / "snr_calibration.json"))["snr0_ref_db"]) if (DS / "snr_calibration.json").exists() else 54.0
NAIVE_GB = 245.76e6 * 8 * 3600 / 1e9          # save-all, GB/hr = 7077.89
FLOOR_GB = 0.01                               # exact zeros land here (log axis can't show 0)

# Canonical detector style shared by the baseline/SNR/latency figures outside snip_eval
# (plot_eval_results.py / plot_snr_results.py / plot_latency_results.py): coherent_power = blue "o",
# dino_finetuned = brown "P". Kept identical here so this figure matches the rest of the paper.
COL_COH, COL_DINO = "#1f77b4", "#8c564b"      # coherent = blue, DINO FT = brown
MRK_COH, MRK_DINO = "o", "P"                   # coherent circles, DINO FT plus-crosses
LS_TS, LS_DS = "-", (0, (5, 2))               # solid = time slice, dashed = time slice + downsample

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

snr = lambda a: SNR0 - np.asarray(a, dtype=float)
gb = lambda tb: np.asarray(tb, dtype=float) * 1000.0     # TB/hr -> GB/hr


def plain_log_y(ax):
    """Log y with tick labels ONLY at powers of 10 (minor gridlines kept, but unlabeled)."""
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def series(df, det, mode):
    """Per-SNR mean GB/hr for one detector+strategy, dup 30 dB retake collapsed, zeros -> floor."""
    d = df[(df.detector == det) & (df["mode"] == mode)]
    d = d[d.file_stem != "attenuation_dB_30_v2"]
    d = d.groupby("attenuation_db", as_index=False)["decimated_TB_per_hour"].mean().sort_values("attenuation_db")
    return snr(d.attenuation_db.values), np.maximum(gb(d.decimated_TB_per_hour.values), FLOOR_GB)


def gt_ceiling(df, mode):
    """Flat GT ceiling: GT is the same transmitted waveforms at every SNR, so mean over SNR."""
    d = df[(df.detector == "ground_truth") & (df["mode"] == mode)]
    d = d[d.file_stem != "attenuation_dB_30_v2"]
    per_atten = d.groupby("attenuation_db")["decimated_TB_per_hour"].mean()
    return gb(per_atten.mean())


base = pd.read_csv(BASE)
k75 = pd.read_csv(K75)
gt_ts = gt_ceiling(base, "time_only")
gt_ds = gt_ceiling(base, "frequency")

fig, ax = plt.subplots(figsize=(9.0, 5.6))

# --- references ---
ax.axhline(NAIVE_GB, color="#0b0b0b", ls="-", lw=2.6, zorder=2,
           label=f"Naive save-all ({NAIVE_GB:,.0f} GB/hr)")
ax.axhline(gt_ts, color="#4d4b47", ls=LS_TS, lw=1.8, zorder=3,
           label=f"Ground truth · time slice ({gt_ts:,.0f} GB/hr)")
ax.axhline(gt_ds, color="#7a7873", ls=LS_DS, lw=1.8, zorder=3,
           label=f"Ground truth · time slice + downsample ({gt_ds:,.0f} GB/hr)")

# --- detectors: solid = time slice, dashed = time slice + downsample ---
for label, df, det, color, mrk in [
    ("Coherent Power", k75, "coherent_power", COL_COH, MRK_COH),
    ("DINO FT", base, "finetuned_dino_m2", COL_DINO, MRK_DINO),
]:
    xs, ys = series(df, det, "time_only")
    ax.plot(xs, ys, ls=LS_TS, marker=mrk, color=color, lw=2.0, ms=6, zorder=5,
            label=f"{label} · time slice")
    xf, yf = series(df, det, "frequency")
    ax.plot(xf, yf, ls=LS_DS, marker=mrk, color=color, lw=2.0, ms=6, zorder=5,
            label=f"{label} · time slice + downsample")

ax.set_xlim(-20, 40)
ax.set_ylim(FLOOR_GB * 0.6, NAIVE_GB * 1.6)
plain_log_y(ax)
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("Stored Data (GB / hour, log scale)")
ax.grid(alpha=.3, which="both")
ax.set_title("Stored Data per Hour vs. SNR")
leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.92,
                title="solid = time slice   ·   dashed = time slice + downsample")
leg.get_title().set_fontsize(8)
ax.text(0.02, 0.15, "Coherent Power markers at 0.01\n= exact zeros (nothing stored)",
        transform=ax.transAxes, fontsize=8, color="#555", ha="left", va="center")

fname = OUT / "fig_mixed_storage_vs_snr.png"
fig.tight_layout()
fig.savefig(fname)
plt.close(fig)
print("wrote", fname)

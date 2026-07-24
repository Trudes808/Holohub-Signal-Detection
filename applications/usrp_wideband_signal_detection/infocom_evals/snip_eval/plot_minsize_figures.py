#!/usr/bin/env python3
"""Figures for the min-size-filter snip eval (coherent_power + FT-DINO M2).

Reads:
  real_snip_metrics.csv          -> DEFAULT run (min_box_pixels=256 only; all detectors)
  real_snip_metrics_minsize.csv  -> 100 kHz + 5 ms filter run (2 detectors)

Produces (in figs_minsize/):
  fig1_100k5ms.png            both strategies (time-slice + snip), 2 detectors, vs SNR
  fig2_100k5ms_timeslice.png  time-slice footprint, 2 detectors
  fig3_100k5ms_snip.png       snip footprint, 2 detectors
  cmp_snip.png                snip footprint: default (256) vs 100 kHz/5 ms, per detector
  cmp_timeslice.png           time-slice footprint: default vs 100 kHz/5 ms
  cmp_count.png               snippet COUNT: default vs 100 kHz/5 ms (how many boxes the gate drops)

GB/hour, log y (plain numbers). Zero-footprint points (e.g. FT-DINO M2 at -16 dB, where nothing
clears the gate) are floored + annotated "0" since log can't show zero.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

SE  = Path(__file__).resolve().parent
OUT = SE / "figs_minsize"; OUT.mkdir(exist_ok=True)
DEFAULT = pd.read_csv(SE / "real_snip_metrics.csv")
MIN     = pd.read_csv(SE / "real_snip_metrics_minsize.csv")
for df in (DEFAULT, MIN):                                   # drop the duplicate 30 dB re-take
    df.drop(df[df.file_stem.astype(str).str.contains("_v2")].index, inplace=True)

SNR0 = float(json.load(open(SE / "snr_calibration.json"))["snr0_ref_db"]) if (SE/"snr_calibration.json").exists() else 54.02380742487758
NAIVE_GB = 245.76e6 * 8 * 3600 / 1e9      # 7078
FLOOR = 0.03                               # GB/hr floor for plotting zeros on a log axis
def snr(a): return SNR0 - np.asarray(a, float)
def gb(tb): return np.asarray(tb, float) * 1000.0

def gt_ceiling(mode):
    d = DEFAULT[(DEFAULT["mode"] == mode) & (DEFAULT.detector == "ground_truth")]
    return gb(d.groupby("attenuation_db").decimated_TB_per_hour.mean().mean())
GT_TS, GT_SNIP = gt_ceiling("time_only"), gt_ceiling("frequency")

DETS  = ["coherent_power", "finetuned_dino_m2"]
LABEL = {"coherent_power": "Coherent Power", "finetuned_dino_m2": "DINO FT"}
COLOR = {"coherent_power": "#4a3aa7", "finetuned_dino_m2": "#2a78d6"}
MARK  = {"coherent_power": "o", "finetuned_dino_m2": "D"}

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

def _ser(df, det, mode, col="decimated_TB_per_hour", scale=True):
    d = (df[(df["mode"] == mode) & (df.detector == det)]
         .groupby("attenuation_db", as_index=False)[col].mean().sort_values("attenuation_db"))
    y = gb(d[col].values) if scale else d[col].values.astype(float)
    return snr(d.attenuation_db.values), y

def _plain_log(ax):
    ax.set_yscale("log"); ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))

def _line(ax, x, y, color, marker, floor=FLOOR, **kw):
    """Plot with zeros floored to `floor` and annotated, so a real 0 shows on a log axis."""
    yy = np.where(np.asarray(y) <= 0, floor, y)
    ax.plot(x, yy, marker=marker, color=color, **kw)
    for xi, yi in zip(x, y):
        if yi <= 0:
            ax.annotate("0", (xi, floor), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=8, color=color, fontweight="bold")

def _finish(ax, title, fname, ymin=FLOOR*0.7, ymax=NAIVE_GB*1.35):
    ax.axhline(NAIVE_GB, color="#0b0b0b", ls="-", lw=2.4, zorder=1, label=f"Naive save-all ({NAIVE_GB:g})")
    ax.set_xlim(-20, 40); ax.set_ylim(ymin, ymax); _plain_log(ax)
    ax.set_xlabel("SNR (dB)  [higher → cleaner]"); ax.set_ylabel("stored data (GB / hour, log)")
    ax.grid(alpha=.3, which="both"); ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    f = ax.get_figure(); f.tight_layout(); f.savefig(OUT / fname); plt.close(f)
    print("wrote", OUT / fname)

# ---- Fig 1: both strategies, 2 detectors (100 kHz / 5 ms) ----
fig, ax = plt.subplots(figsize=(8.6, 5.4))
ax.axhline(GT_TS,   color="#4d4b47", ls=(0, (6, 3)), lw=1.8, zorder=2, label=f"GT time-slice ({GT_TS:g})")
ax.axhline(GT_SNIP, color="#7a7873", ls=(0, (1, 1)), lw=1.8, zorder=2, label=f"GT snip ({GT_SNIP:g})")
for det in DETS:
    x, y = _ser(MIN, det, "time_only"); _line(ax, x, y, COLOR[det], MARK[det], lw=1.8, ms=5, ls="-",  label=f"{LABEL[det]} · time-slice")
    x, y = _ser(MIN, det, "frequency"); _line(ax, x, y, COLOR[det], MARK[det], lw=1.8, ms=5, ls=(0, (1, 1)), mfc="none", label=f"{LABEL[det]} · snip")
_finish(ax, "Data stored/hr — 100 kHz + 5 ms gate", "fig1_100k5ms.png")

# ---- Fig 2 / Fig 3: single strategy ----
for mode, gtline, gtlab, title, fname in [
    ("time_only", GT_TS,   "Ground truth time-slice", "Time-slice storage vs SNR (100 kHz + 5 ms)", "fig2_100k5ms_timeslice.png"),
    ("frequency", GT_SNIP, "Ground truth snip",        "Snip storage vs SNR (100 kHz + 5 ms)",       "fig3_100k5ms_snip.png")]:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.axhline(gtline, color="#4d4b47", ls=(0, (6, 3)), lw=2.2, zorder=2, label=f"{gtlab} ({gtline:g})")
    for det in DETS:
        x, y = _ser(MIN, det, mode); _line(ax, x, y, COLOR[det], MARK[det], lw=1.9, ms=6, ls="-", label=LABEL[det])
    _finish(ax, title, fname)

# ---- Comparison: default (256 only) vs 100 kHz/5 ms, per detector ----
for mode, title, fname in [("frequency", "Snip footprint: default vs 100 kHz + 5 ms", "cmp_snip.png"),
                           ("time_only", "Time-slice footprint: default vs 100 kHz + 5 ms", "cmp_timeslice.png")]:
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for det in DETS:
        xd, yd = _ser(DEFAULT, det, mode); ax.plot(xd, yd, ls=(0, (5, 2)), marker=MARK[det], color=COLOR[det], lw=1.6, ms=4, alpha=.55, mfc="none")
        xm, ym = _ser(MIN,     det, mode); _line(ax, xm, ym, COLOR[det], MARK[det], lw=2.0, ms=6, ls="-")
    handles = [Line2D([], [], color=COLOR[d], marker=MARK[d], ls="-", label=LABEL[d]) for d in DETS]
    handles += [Line2D([], [], color="#666", ls=(0, (5, 2)), label="default (256 px only)"),
                Line2D([], [], color="#666", ls="-", label="100 kHz + 5 ms gate")]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    ax.axhline(NAIVE_GB, color="#0b0b0b", ls="-", lw=2.0, zorder=1)
    ax.set_xlim(-20, 40); ax.set_ylim(FLOOR*0.7, NAIVE_GB*1.35); _plain_log(ax)
    ax.set_xlabel("SNR (dB)  [higher → cleaner]"); ax.set_ylabel("stored data (GB / hour, log)")
    ax.grid(alpha=.3, which="both"); ax.set_title(title)
    f = ax.get_figure(); f.tight_layout(); f.savefig(OUT / fname); plt.close(f); print("wrote", OUT / fname)

# ---- Comparison: snippet COUNT (how many boxes survive the gate) ----
fig, ax = plt.subplots(figsize=(8.8, 5.4))
for det in DETS:
    xd, yd = _ser(DEFAULT, det, "frequency", col="n_snippets", scale=False)
    ax.plot(xd, yd, ls=(0, (5, 2)), marker=MARK[det], color=COLOR[det], lw=1.6, ms=4, alpha=.55, mfc="none")
    xm, ym = _ser(MIN, det, "frequency", col="n_snippets", scale=False)
    _line(ax, xm, ym, COLOR[det], MARK[det], lw=2.0, ms=6, ls="-", floor=0.7)
handles = [Line2D([], [], color=COLOR[d], marker=MARK[d], ls="-", label=LABEL[d]) for d in DETS]
handles += [Line2D([], [], color="#666", ls=(0, (5, 2)), label="default (256 px only)"),
            Line2D([], [], color="#666", ls="-", label="100 kHz + 5 ms gate")]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
ax.set_xlim(-20, 40); ax.set_yscale("log"); ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax.set_xlabel("SNR (dB)  [higher → cleaner]"); ax.set_ylabel("snippets per capture (log)")
ax.grid(alpha=.3, which="both"); ax.set_title("Boxes kept: default vs 100 kHz + 5 ms gate (snip mode)")
fig.tight_layout(); fig.savefig(OUT / "cmp_count.png"); plt.close(fig); print("wrote", OUT / "cmp_count.png")
print("done")

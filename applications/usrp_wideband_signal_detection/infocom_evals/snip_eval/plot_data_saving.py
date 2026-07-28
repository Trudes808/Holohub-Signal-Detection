#!/usr/bin/env python3
"""Figures 1-3 from the REAL signal_snipper measurements (verify_snip.py -> real_snip_metrics.csv).

Time-slice strategy = the operator's time_only footprint; snip strategy = the operator's frequency
footprint. Y-axis in GIGABYTES/hour (log scale, plain numbers - no scientific notation); X-axis SNR.
Two versions each: (a) all detectors, (b) curated = ground truth + naive + coherent_power + FT-DINO.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HSD   = Path.home() / "Holohub-Signal-Detection"
DS    = HSD / "applications/usrp_wideband_signal_detection/infocom_evals/snip_eval"
METRICS = Path("/tmp/usrp_spectrograms/snip_eval/real_snip_metrics.csv")
if not METRICS.exists():
    METRICS = HSD / "applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/real_snip_metrics.csv"
OUT   = DS / "figs"; OUT.mkdir(exist_ok=True)

SNR0 = float(json.load(open(DS / "snr_calibration.json"))["snr0_ref_db"]) if (DS/"snr_calibration.json").exists() else 54.0
NAIVE_GB = 245.76e6 * 8 * 3600 / 1e9        # save-all, GB/hr = 7078
def snr(atten): return SNR0 - np.asarray(atten, dtype=float)
def gb(tbhr):   return np.asarray(tbhr, dtype=float) * 1000.0   # TB/hr -> GB/hr

DETS  = ["coherent_power","cuda_dino","finetuned_dino","finetuned_dino_m2","yolo26s","yolo26m","3dB_power","blob_detection"]
LABEL = {"coherent_power":"Coherent Power","cuda_dino":"Zero-shot DINOv3","finetuned_dino":"FT-DINO M1",
         "finetuned_dino_m2":"FT-DINO M2","yolo26s":"YOLO26s","yolo26m":"YOLO26m",
         "3dB_power":"3 dB Power","blob_detection":"Blob Detection"}
COLOR = {"coherent_power":"#4a3aa7","cuda_dino":"#1baf7a","finetuned_dino":"#eb6834","finetuned_dino_m2":"#2a78d6",
         "yolo26s":"#e34948","yolo26m":"#eda100","3dB_power":"#6f6d68","blob_detection":"#9c6b3f"}
MARKER= {"coherent_power":"o","cuda_dino":"s","finetuned_dino":"^","finetuned_dino_m2":"D",
         "yolo26s":"v","yolo26m":"P","3dB_power":"X","blob_detection":"*"}
CURATED = ["coherent_power","finetuned_dino_m2"]   # + naive + ground truth ceilings
CURATED_LABEL = dict(LABEL); CURATED_LABEL["finetuned_dino_m2"] = "DINO FT"   # curated: show M2 only, labelled just "DINO FT"

plt.rcParams.update({"figure.dpi":120,"savefig.dpi":220,"savefig.bbox":"tight","font.size":11,
                     "axes.spines.top":False,"axes.spines.right":False})

m = pd.read_csv(METRICS)
m = m[m["file_stem"] != "attenuation_dB_30_v2"]   # drop the duplicate 30 dB re-take (keep one capture per SNR)
FREQ = m[m["mode"] == "frequency"].copy()      # snip footprint
TONLY = m[m["mode"] == "time_only"].copy()     # time-slice footprint
def _gt_ceiling(df):   # avg duplicate captures per attenuation first, then mean across SNR (flat GT)
    per_atten = df[df.detector == "ground_truth"].groupby("attenuation_db")["decimated_TB_per_hour"].mean()
    return gb(per_atten.mean())
GT_TS_GB   = _gt_ceiling(TONLY)
GT_SNIP_GB = _gt_ceiling(FREQ)

def _plain_log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda y, _: f"{y:g}" if y in (2,3,5,20,30,50,200,300,500,2000,3000,5000) else ""))

def _ser(df, det, sort=True):
    # Collapse duplicate captures at the same attenuation (e.g. two takes at 30 dB) into one mean point.
    d = df[df.detector == det].groupby("attenuation_db", as_index=False)["decimated_TB_per_hour"].mean()
    d = d.sort_values("attenuation_db")
    return snr(d.attenuation_db.values), gb(d.decimated_TB_per_hour.values)

def _finish(ax, title, fname):
    ax.axhline(NAIVE_GB, color="#0b0b0b", ls="-", lw=2.6, zorder=1, label=f"Naive save-all ({NAIVE_GB:g} GB/hr)")
    ax.set_xlim(-20, 40); _plain_log_y(ax)
    ax.set_xlabel("SNR (dB)  [higher → cleaner]"); ax.set_ylabel("stored data (GB / hour, log)")
    ax.grid(alpha=.3, which="both"); ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(OUT / fname); plt.close(fig)
    print(f"wrote {OUT/fname}")

def fig1(dets, fname, title, lbl=LABEL):
    fig, ax = plt.subplots(figsize=(9.6, 5.8) if len(dets) > 4 else (8.2, 5.2))
    ax.axhline(GT_TS_GB,   color="#4d4b47", ls=(0,(6,3)), lw=2.0, zorder=2, label=f"GT time-slice ({GT_TS_GB:g} GB/hr)")
    ax.axhline(GT_SNIP_GB, color="#7a7873", ls=(0,(1,1)), lw=2.0, zorder=2, label=f"GT snip ({GT_SNIP_GB:g} GB/hr)")
    for det in dets:
        xs, ys = _ser(TONLY, det); ax.plot(xs, ys, ls="-", marker=MARKER[det], color=COLOR[det], lw=1.8, ms=5, label=f"{lbl[det]} · time-slice")
        xf, yf = _ser(FREQ,  det); ax.plot(xf, yf, ls=(0,(1,1)), marker=MARKER[det], color=COLOR[det], lw=1.8, ms=5, mfc="none", label=f"{lbl[det]} · snip")
    _finish(ax, title, fname)

def fig_strategy(df, gt_gb, gt_label, dets, fname, title, lbl=LABEL):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    ax.axhline(gt_gb, color="#4d4b47", ls=(0,(6,3)), lw=2.6, zorder=2, label=f"Ground truth ({gt_gb:g} GB/hr)")
    for det in dets:
        xs, ys = _ser(df, det); ax.plot(xs, ys, "-", marker=MARKER[det], color=COLOR[det], lw=1.8, ms=5, label=lbl[det])
    _finish(ax, title, fname)

for suffix, dets in (("all", DETS), ("curated", CURATED)):
    lbl = LABEL if suffix == "all" else CURATED_LABEL
    fig1(dets, f"fig1_bytes_{suffix}.png", "Data stored/hr — naive vs time-slice vs snip"
         + ("" if suffix == "all" else " (curated)"), lbl=lbl)
    fig_strategy(TONLY, GT_TS_GB,   "GT", dets, f"fig2_timeslice_{suffix}.png",
                 "Time-slice storage vs SNR" + ("" if suffix == "all" else " (curated)"), lbl=lbl)
    fig_strategy(FREQ,  GT_SNIP_GB, "GT", dets, f"fig3_snip_{suffix}.png",
                 "Snip (resample+filter) storage vs SNR" + ("" if suffix == "all" else " (curated)"), lbl=lbl)
print("done")

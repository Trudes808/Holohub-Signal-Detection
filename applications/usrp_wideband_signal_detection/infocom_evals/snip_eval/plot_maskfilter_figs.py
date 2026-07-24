#!/usr/bin/env python3
"""Before/after figures for the min_mask_bandwidth_hz fix in the data_saving_eval_review house
style (Figures 1-3: GB/hr on a plain-number log axis vs SNR, naive save-all line, GT ceiling,
same detector colors/markers, legend outside right).

Per gate config (100 kHz/5 ms and 75 kHz/1 ms), two figures matching fig2/fig3:
  fig2_timeslice_before_after_<gate>.png   (time_only mode)
  fig3_snip_before_after_<gate>.png        (frequency mode)
Lines: Coherent Power BEFORE the fix (spur-fusion artifact), Coherent Power AFTER, DINO FT
(reference; the mask filter provably removes 0.00% of its pixels so before == after), plus the
naive save-all and a ground-truth ceiling. Sources: real verify_snip CSVs where a real run exists;
the 75k BEFORE and both GT ceilings come from the offline replication that matches the real
pipeline to three decimals (labeled "replicated").

Run: ~/miniforge3/envs/dinov3/bin/python plot_maskfilter_figs.py   -> figs_minsize/
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DS = Path(__file__).resolve().parent
OUT = DS / "figs_minsize"
SNR0 = float(json.load(open(DS / "snr_calibration.json"))["snr0_ref_db"]) if (DS / "snr_calibration.json").exists() else 54.0
NAIVE_GB = 245.76e6 * 8 * 3600 / 1e9

COL_COH, COL_DINO = "#4a3aa7", "#2a78d6"
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

snr = lambda a: SNR0 - np.asarray(a, dtype=float)
gb = lambda tb: np.asarray(tb, dtype=float) * 1000.0


def plain_log_y(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(
        lambda y, _: f"{y:g}" if y in (2, 3, 5, 20, 30, 50, 200, 300, 500, 2000, 3000, 5000) else ""))


def ser(df, det, mode, floor_gb=1e-2):
    d = df[(df.detector == det) & (df["mode"] == mode)]
    d = d[d.file_stem != "attenuation_dB_30_v2"]
    d = d.groupby("attenuation_db", as_index=False)["decimated_TB_per_hour"].mean().sort_values("attenuation_db")
    return snr(d.attenuation_db.values), np.maximum(gb(d.decimated_TB_per_hour.values), floor_gb)


def gt_ceiling(gate_key, mode_col):
    q = pd.read_csv(DS / "fix_quantification.csv")
    q = q[(q.detector == "ground_truth") & (q.strategy == "current") & (q.gate == gate_key)]
    return gb(q[mode_col].mean())


CFGS = {
    "100k5ms": dict(gate_key="minsize_100k_5ms", label="100 kHz / 5 ms gate",
                    before=DS / "real_snip_metrics_minsize.csv",
                    after=DS / "real_snip_metrics_minsize_v2.csv", before_replicated=False),
    "75k1ms": dict(gate_key="75k_1ms", label="75 kHz / 1 ms gate",
                   before=DS / "real_snip_metrics_75k_before_replicated.csv",
                   after=DS / "real_snip_metrics_75k_v2.csv", before_replicated=True),
}


def make_fig(cfg_key, cfg, mode, fig_stem, strategy_title):
    before = pd.read_csv(cfg["before"])
    after = pd.read_csv(cfg["after"])
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    gt_gb = gt_ceiling(cfg["gate_key"], "freq_TB_hr" if mode == "frequency" else "time_TB_hr")
    ax.axhline(NAIVE_GB, color="#0b0b0b", ls="-", lw=2.6, zorder=1,
               label=f"Naive save-all ({NAIVE_GB:g} GB/hr)")
    ax.axhline(gt_gb, color="#4d4b47", ls=(0, (6, 3)), lw=2.2, zorder=2,
               label=f"Ground truth, replicated ({gt_gb:g} GB/hr)")
    rep = " (replicated)" if cfg["before_replicated"] else ""
    xs, ys = ser(before, "coherent_power", mode)
    ax.plot(xs, ys, ls=(0, (4, 2)), marker="o", mfc="none", color=COL_COH, lw=1.8, ms=5,
            label=f"Coherent Power · BEFORE fix{rep}")
    xs, ys = ser(after, "coherent_power", mode)
    ax.plot(xs, ys, "-", marker="o", color=COL_COH, lw=2.2, ms=5,
            label="Coherent Power · AFTER fix")
    xs, ys = ser(after, "finetuned_dino_m2", mode)
    ax.plot(xs, ys, "-", marker="D", color=COL_DINO, lw=1.8, ms=5, label="DINO FT")
    ax.set_xlim(-20, 40)
    plain_log_y(ax)
    ax.set_xlabel("SNR (dB)  [higher → cleaner]")
    ax.set_ylabel("stored data (GB / hour, log)")
    ax.grid(alpha=.3, which="both")
    ax.set_title(f"{strategy_title} — before vs after the mask pre-filter ({cfg['label']})\n"
                 f"points at 0.01 GB/hr are true zeros", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, .5), fontsize=8)
    fig.tight_layout()
    fname = OUT / f"{fig_stem}_before_after_{cfg_key}.png"
    fig.savefig(fname)
    plt.close(fig)
    print("wrote", fname)


for cfg_key, cfg in CFGS.items():
    make_fig(cfg_key, cfg, "time_only", "fig2_timeslice", "Time-slice storage vs SNR")
    make_fig(cfg_key, cfg, "frequency", "fig3_snip", "Snip (resample+filter) storage vs SNR")
print("done")

#!/usr/bin/env python3
"""Live (over-the-air) data-saving comparison for the two 500 MSps X410 captures.

Three-way comparison per capture: the ORIGINAL save-all footprint vs the coherent-power snip
pipeline under two snipper gate settings (256-pixel component gate only, and + a 75 kHz bandwidth
filter + 1 ms minimum duration). Reads the two snip_pipeline pipeline_metrics.csv files and writes:
  ota_live_eval.csv   — combined table (original / snip-256px / snip-75kHz+1ms, both captures)
  ota_live_eval.png   — stored GB/hr (log) vs the save-all line, with snippet count + reduction

Run: ~/miniforge3/envs/dinov3/bin/python make_live_eval.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
SRC_256 = HERE / "ota_metrics_snip_256px.csv"
SRC_75K = HERE / "ota_metrics_snip_75kHz_1ms.csv"

BYTES_PER_SAMPLE = 8                                    # cf32
RATE_HZ = 500e6
DURATION_S = 10.0
SAVE_ALL_GB_HR = RATE_HZ * BYTES_PER_SAMPLE * 3600 / 1e9   # 14,400 GB/hr at 500 MSps cf32
ORIG_MB_10S = RATE_HZ * DURATION_S * BYTES_PER_SAMPLE / 1e6  # 40,000 MB

CAP_LABEL = {
    "ota_x410_cf2400MHz_500Msps_cf32_10s.sigmf-data": "2.4 GHz",
    "ota_x410_cf1000MHz_500Msps_cf32_10s.sigmf-data": "1.0 GHz",
}
STAGES = [("snip · 256px", SRC_256), ("snip · 75kHz+1ms", SRC_75K)]

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def load(p):
    return {r["capture"]: r for r in csv.DictReader(open(p))}


metrics = {label: load(src) for label, src in STAGES}
captures = list(next(iter(metrics.values())).keys())

# ---- combined table ----
rows = []
for cap in captures:
    lab = CAP_LABEL.get(cap, cap)
    rows.append(dict(capture=lab, stage="original (save-all)", n_snippets="",
                     stored_MB_10s=round(ORIG_MB_10S, 1), stored_GB_per_hr=round(SAVE_ALL_GB_HR, 1),
                     reduction_x=1.0, mask_coverage_pct=""))
    for stage_label, _ in STAGES:
        r = metrics[stage_label][cap]
        ss = int(r["stored_samples"])
        rows.append(dict(capture=lab, stage=stage_label, n_snippets=int(r["n_snippets"]),
                         stored_MB_10s=round(ss * BYTES_PER_SAMPLE / 1e6, 1),
                         stored_GB_per_hr=round(float(r["stored_GB_per_hour"]), 2),
                         reduction_x=round(float(r["reduction_x"]), 1),
                         mask_coverage_pct=float(r["mask_coverage_pct"])))
out_csv = HERE / "ota_live_eval.csv"
with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print("wrote", out_csv)

# ---- two-channel system totals (both captures = the 2-channel 500 MSps system over 10 s) ----
N_CH = 2
saveall_2ch_gb = RATE_HZ * DURATION_S * BYTES_PER_SAMPLE * N_CH / 1e9   # 80.0 GB (both channels, 10 s)
sys_rows = []
for stage_label, _ in STAGES:
    m = metrics[stage_label]
    tot_samp = sum(int(m[c]["stored_samples"]) for c in captures)
    tot_snips = sum(int(m[c]["n_snippets"]) for c in captures)
    gb = tot_samp * BYTES_PER_SAMPLE / 1e9
    sys_rows.append(dict(config=stage_label,
                         saveall_two_channel_GB_10s=round(saveall_2ch_gb, 1),
                         stored_GB_10s=round(gb, 3), stored_MB_10s=round(gb * 1000, 1),
                         reduction_x=round(saveall_2ch_gb / gb, 1),
                         n_recordings=tot_snips, n_files_data_plus_meta=tot_snips * 2))
sys_csv = HERE / "ota_system_totals.csv"
with open(sys_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(sys_rows[0].keys()))
    w.writeheader(); w.writerows(sys_rows)
print("wrote", sys_csv)
for r in sys_rows:
    print(f"  [{r['config']:>10}] 2-ch save-all {r['saveall_two_channel_GB_10s']} GB -> stored "
          f"{r['stored_GB_10s']} GB ({r['reduction_x']}x less), "
          f"{r['n_recordings']:,} recordings ({r['n_files_data_plus_meta']:,} files)")

# ---- figure: stored GB/hr (log) per capture, two gate settings, vs the save-all line ----
cap_labels = [CAP_LABEL.get(c, c) for c in captures]
x = np.arange(len(captures))
w = 0.34
COLORS = {"snip · 256px": "#4a3aa7", "snip · 75kHz+1ms": "#2a78d6"}

fig, ax = plt.subplots(figsize=(8.4, 5.4))
ax.axhline(SAVE_ALL_GB_HR, color="#0b0b0b", lw=2.6, zorder=1,
           label=f"original / save-all ({SAVE_ALL_GB_HR:,.0f} GB/hr)")
for i, (stage_label, _) in enumerate(STAGES):
    vals = [float(metrics[stage_label][c]["stored_GB_per_hour"]) for c in captures]
    snips = [int(metrics[stage_label][c]["n_snippets"]) for c in captures]
    red = [float(metrics[stage_label][c]["reduction_x"]) for c in captures]
    off = (i - 0.5) * w
    bars = ax.bar(x + off, vals, w, color=COLORS[stage_label], zorder=3,
                  label=stage_label.replace("snip · ", "snip · gate "))
    for xi, v, n, rr in zip(x + off, vals, snips, red):
        ax.text(xi, v * 1.08, f"{v:.0f} GB/hr\n{n:,} snips\n×{rr:g} less",
                ha="center", va="bottom", fontsize=7.5)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
ax.set_ylim(30, SAVE_ALL_GB_HR * 3)
ax.set_xticks(x, cap_labels)
ax.set_xlabel("capture center frequency (500 MSps, 10 s, over-the-air)")
ax.set_ylabel("Stored Data (GB / hour, log scale)")
ax.set_title("Live OTA data-saving: coherent-power snip vs save-all")
ax.grid(alpha=.3, which="both", axis="y")
ax.legend(loc="upper right", fontsize=8)
r75 = next(r for r in sys_rows if "75" in r["config"])
fig.text(0.5, -0.015,
         f"Two-channel system, 10 s:  save-all {saveall_2ch_gb:.0f} GB  →  coherent-power snip "
         f"{r75['stored_GB_10s']:.2f} GB ({r75['reduction_x']:g}× less) in "
         f"{r75['n_recordings']:,} recordings / {r75['n_files_data_plus_meta']:,} files [75 kHz+1 ms]",
         ha="center", va="top", fontsize=9)
fig.tight_layout()
out_png = HERE / "ota_live_eval.png"
fig.savefig(out_png)
plt.close(fig)
print("wrote", out_png)

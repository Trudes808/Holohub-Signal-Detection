#!/usr/bin/env python3
"""Detection-rate + data-transfer-rate analysis for the OTA 75 kHz/1 ms snip pipeline.

For each capture (2.4 GHz, 1.0 GHz) and the modeled two-channel system, bin the snipped detections
by the detector's native frame (10,485,760 samples = 20.97 ms at 500 MSps) and quantify how the
stored-data transfer rate and detection rate vary across the ~10 s experiment. Emits:
  ota_rate_timeseries.csv         per-frame ndet / stored bytes / rate (Gbps, MB/s) x {2.4,1.0,2ch}
  ota_rate_stats.csv              per-series min/mean/max/std/median/p5/p95 rate + totals + det-rate
  ota_detection_property_stats.csv per-detection bandwidth / duration / stored-size / decimation stats
  ota_rate_vs_time.png            data-rate (Gbps) and detection-rate (det/s) vs time
  ota_detection_distributions.png histograms of per-detection bandwidth / duration / stored size

NOTE the 1.0 GHz series is the band-edge spur (see README), not real signal. The two captures were
taken sequentially; the "2-channel" series sums them frame-by-frame as the dual-500-MSps config would
capture them simultaneously (valid for totals/means; the combined per-frame peak is a model estimate).

Run: ~/miniforge3/envs/dinov3/bin/python analyze_rate.py
"""
from __future__ import annotations
import csv, glob, json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
SNIP_ROOT = Path("/tmp/usrp_spectrograms/ota_snip_pipeline_75k1ms/snip")

FS = 500e6
BYTES = 8                                   # cf32
SAMPLES_PER_FRAME = 10_485_760              # 512 rows x 20480
FRAME_S = SAMPLES_PER_FRAME / FS            # 20.9715 ms
N_FRAMES = 476
DUR_S = N_FRAMES * FRAME_S                  # ~9.98 s processed
SAVEALL_GBPS_CH = FS * BYTES * 8 / 1e9      # 32.0 Gbps per channel
SAVEALL_GBPS_2CH = 2 * SAVEALL_GBPS_CH      # 64.0 Gbps

CAPS = [("ota_x410_cf2400MHz_500Msps_cf32_10s", "2.4 GHz", "#2a78d6", False),
        ("ota_x410_cf1000MHz_500Msps_cf32_10s", "1.0 GHz", "#d1651a", True)]   # True = spur


def load_dets(stem):
    """Return list of dicts per detection: frame, stored_bytes, bw_hz, dur_s, decim, stored_kb."""
    out = []
    for mp in glob.glob(f"{SNIP_ROOT}/{stem}/snippets/*.sigmf-meta"):
        for a in json.load(open(mp))["annotations"]:
            out.append(dict(
                frame=int(a["wfgt:frame_number"]),
                stored_bytes=int(a["core:sample_count"]) * BYTES,
                bw_hz=float(a["core:freq_upper_edge"]) - float(a["core:freq_lower_edge"]),
                dur_s=int(a["wfgt:orig_sample_count"]) / FS,
                decim=float(a["wfgt:decimation_factor"]),
            ))
    return out


def per_frame(dets):
    """Arrays over frames 1..N_FRAMES (0 for empty frames)."""
    nbytes = np.zeros(N_FRAMES); ndet = np.zeros(N_FRAMES, dtype=int)
    for d in dets:
        f = d["frame"] - 1
        if 0 <= f < N_FRAMES:
            nbytes[f] += d["stored_bytes"]; ndet[f] += 1
    gbps = nbytes * 8 / FRAME_S / 1e9
    detps = ndet / FRAME_S
    return nbytes, ndet, gbps, detps


def stats(x):
    return dict(min=float(np.min(x)), mean=float(np.mean(x)), max=float(np.max(x)),
                std=float(np.std(x)), median=float(np.median(x)),
                p5=float(np.percentile(x, 5)), p95=float(np.percentile(x, 95)))


# ---- load + per-frame ----
series = {}   # label -> dict
frames_bytes = {}
for stem, label, color, spur in CAPS:
    dets = load_dets(stem)
    nbytes, ndet, gbps, detps = per_frame(dets)
    frames_bytes[label] = nbytes
    series[label] = dict(label=label, color=color, spur=spur, dets=dets,
                         nbytes=nbytes, ndet=ndet, gbps=gbps, detps=detps)
# combined 2-channel = frame-wise sum
comb_bytes = sum(frames_bytes.values())
comb_gbps = comb_bytes * 8 / FRAME_S / 1e9
comb_ndet = sum(series[l]["ndet"] for l in series)
comb_detps = comb_ndet / FRAME_S
series["2-channel"] = dict(label="2-channel (modeled)", color="#4a3aa7", spur=False, dets=None,
                           nbytes=comb_bytes, ndet=comb_ndet, gbps=comb_gbps, detps=comb_detps)

t = np.arange(N_FRAMES) * FRAME_S    # frame start time (s)

# ---- CSV 1: per-frame time series ----
ts_csv = HERE / "ota_rate_timeseries.csv"
with open(ts_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["frame", "time_s",
                "cf2400_ndet", "cf2400_stored_bytes", "cf2400_Gbps", "cf2400_MBps",
                "cf1000_ndet", "cf1000_stored_bytes", "cf1000_Gbps", "cf1000_MBps",
                "comb2ch_ndet", "comb2ch_stored_bytes", "comb2ch_Gbps", "comb2ch_MBps"])
    s24, s10, sc = series["2.4 GHz"], series["1.0 GHz"], series["2-channel"]
    for i in range(N_FRAMES):
        w.writerow([i + 1, round(t[i], 5),
                    int(s24["ndet"][i]), int(s24["nbytes"][i]), round(s24["gbps"][i], 5), round(s24["nbytes"][i] / FRAME_S / 1e6, 3),
                    int(s10["ndet"][i]), int(s10["nbytes"][i]), round(s10["gbps"][i], 5), round(s10["nbytes"][i] / FRAME_S / 1e6, 3),
                    int(sc["ndet"][i]), int(sc["nbytes"][i]), round(sc["gbps"][i], 5), round(sc["nbytes"][i] / FRAME_S / 1e6, 3)])
print("wrote", ts_csv)

# ---- CSV 2: summary rate stats ----
st_csv = HERE / "ota_rate_stats.csv"
with open(st_csv, "w", newline="") as fh:
    cols = ["series", "spur", "n_detections", "total_stored_MB", "effective_seconds",
            "mean_Gbps", "min_Gbps", "max_Gbps", "std_Gbps", "median_Gbps", "p5_Gbps", "p95_Gbps",
            "peak_to_mean", "mean_MBps", "mean_GB_per_hr", "mean_det_per_s", "max_det_per_s",
            "saveall_Gbps", "reduction_vs_saveall_x"]
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for key in ["2.4 GHz", "1.0 GHz", "2-channel"]:
        s = series[key]; g = s["gbps"]; rs = stats(g)
        total_mb = float(s["nbytes"].sum()) / 1e6
        saveall = SAVEALL_GBPS_2CH if key == "2-channel" else SAVEALL_GBPS_CH
        w.writerow(dict(series=s["label"], spur=s["spur"],
                        n_detections=int(s["ndet"].sum()), total_stored_MB=round(total_mb, 1),
                        effective_seconds=round(DUR_S, 3),
                        mean_Gbps=round(rs["mean"], 4), min_Gbps=round(rs["min"], 4), max_Gbps=round(rs["max"], 4),
                        std_Gbps=round(rs["std"], 4), median_Gbps=round(rs["median"], 4),
                        p5_Gbps=round(rs["p5"], 4), p95_Gbps=round(rs["p95"], 4),
                        peak_to_mean=round(rs["max"] / rs["mean"], 1) if rs["mean"] else float("nan"),
                        mean_MBps=round(total_mb / DUR_S, 2), mean_GB_per_hr=round(total_mb / 1e3 / DUR_S * 3600, 1),
                        mean_det_per_s=round(float(s["ndet"].sum()) / DUR_S, 1), max_det_per_s=round(float(s["detps"].max()), 1),
                        saveall_Gbps=round(saveall, 1),
                        reduction_vs_saveall_x=round(saveall / rs["mean"], 1) if rs["mean"] else float("nan")))
print("wrote", st_csv)

# ---- CSV 3: per-detection property stats ----
pp_csv = HERE / "ota_detection_property_stats.csv"
with open(pp_csv, "w", newline="") as fh:
    cols = ["capture", "property", "unit", "n", "min", "mean", "median", "max", "p95"]
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
    for key in ["2.4 GHz", "1.0 GHz"]:
        dets = series[key]["dets"]
        props = [("bandwidth", "MHz", np.array([d["bw_hz"] for d in dets]) / 1e6),
                 ("duration", "ms", np.array([d["dur_s"] for d in dets]) * 1e3),
                 ("stored_size", "KB", np.array([d["stored_bytes"] for d in dets]) / 1e3),
                 ("decimation", "x", np.array([d["decim"] for d in dets]))]
        for name, unit, arr in props:
            w.writerow(dict(capture=key, property=name, unit=unit, n=len(arr),
                            min=round(float(arr.min()), 3), mean=round(float(arr.mean()), 3),
                            median=round(float(np.median(arr)), 3), max=round(float(arr.max()), 3),
                            p95=round(float(np.percentile(arr, 95)), 3)))
print("wrote", pp_csv)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def roll(x, k=9):
    if k <= 1: return x
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


# ---- FIG 1: data rate + detection rate vs time ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
for key in ["2.4 GHz", "1.0 GHz", "2-channel"]:
    s = series[key]
    lbl = s["label"] + (" [spur]" if s["spur"] else "")
    ax1.plot(t, s["gbps"], color=s["color"], lw=0.8, alpha=0.35)
    ax1.plot(t, roll(s["gbps"]), color=s["color"], lw=2.0, label=f"{lbl}  (mean {s['gbps'].mean():.2f} Gbps)")
ax1.set_ylabel("data transfer rate (Gbps)")
ax1.set_title(f"OTA 75 kHz/1 ms snip — stored-data rate over {DUR_S:.1f} s, per {FRAME_S*1e3:.1f} ms frame\n"
              f"(faint = per-frame, bold = 9-frame rolling mean; two-channel save-all = {SAVEALL_GBPS_2CH:.0f} Gbps, off scale)",
              fontsize=10)
ax1.grid(alpha=.3); ax1.legend(fontsize=8, loc="upper right"); ax1.set_ylim(bottom=0)
for key in ["2.4 GHz", "1.0 GHz"]:
    s = series[key]
    lbl = s["label"] + (" [spur]" if s["spur"] else "")
    ax2.plot(t, s["detps"], color=s["color"], lw=0.8, alpha=0.35)
    ax2.plot(t, roll(s["detps"]), color=s["color"], lw=2.0, label=f"{lbl}  (mean {s['ndet'].sum()/DUR_S:.0f}/s)")
ax2.set_ylabel("detection rate (detections/s)"); ax2.set_xlabel("time in experiment (s)")
ax2.grid(alpha=.3); ax2.legend(fontsize=8, loc="upper right"); ax2.set_ylim(bottom=0); ax2.set_xlim(0, DUR_S)
fig.tight_layout(); fig.savefig(HERE / "ota_rate_vs_time.png"); plt.close(fig)
print("wrote", HERE / "ota_rate_vs_time.png")

# ---- FIG 2: per-detection distributions ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
defs = [("bandwidth", "MHz", 1e6, True), ("duration", "ms", 1e-3, True), ("stored_size", "KB", 1e3, True)]
for ax, (name, unit, scale, logx) in zip(axes, defs):
    for key in ["2.4 GHz", "1.0 GHz"]:
        s = series[key]
        if name == "bandwidth": arr = np.array([d["bw_hz"] for d in s["dets"]]) / scale
        elif name == "duration": arr = np.array([d["dur_s"] for d in s["dets"]]) / scale
        else: arr = np.array([d["stored_bytes"] for d in s["dets"]]) / scale
        bins = np.logspace(np.log10(max(arr.min(), 1e-3)), np.log10(arr.max()), 40) if logx else 40
        ax.hist(arr, bins=bins, histtype="step", lw=1.8, color=s["color"],
                label=s["label"] + (" [spur]" if s["spur"] else ""))
    if logx: ax.set_xscale("log")
    ax.set_xlabel(f"{name.replace('_',' ')} ({unit})"); ax.set_ylabel("detections")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.suptitle("Per-detection distributions — OTA 75 kHz/1 ms snip (log x)", fontsize=11)
fig.tight_layout(); fig.savefig(HERE / "ota_detection_distributions.png"); plt.close(fig)
print("wrote", HERE / "ota_detection_distributions.png")

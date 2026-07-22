#!/usr/bin/env python3
"""Plot the per-frame latency + compute-load eval from a serialized LatencyResults.

Importable helpers (usable straight from the notebook on a loaded object) + a CLI that
writes PNGs. Stdlib + numpy + matplotlib; no pandas/seaborn.

Figures:
  * fig_latency_hist(rate)          -- per-detector CPU-vs-GPU latency histograms at one
                                       rate, with that rate's horizontal real-time budget line.
  * fig_latency_hist_by_rate(device)-- per-detector latency histograms across all rates
                                       (one device), with every rate's real-time budget as a
                                       horizontal dashed reference line.
  * fig_latency_vs_rate()           -- per-detector median latency vs sample rate (CPU & GPU,
                                       p05-p95 band) with the real-time budget curve.
  * fig_compute_load()              -- GFLOPs/frame + peak GPU memory bars (the compute-load eval).

Detector colors/order match baseline_comparisons/plot_snr_results.py so the two evals
read as one system.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from latency_results import LatencyResults

# Detector styling — kept in sync with baseline_comparisons/plot_snr_results.py.
DETECTOR_STYLE = {
    "coherent_power": {"color": "#1f77b4", "marker": "o"},
    "cuda_dino": {"color": "#d62728", "marker": "s"},
    "3dB_power": {"color": "#2ca02c", "marker": "^"},
    "blob_detection": {"color": "#9467bd", "marker": "D"},
    "yolo": {"color": "#ff7f0e", "marker": "v"},
    "dino_finetuned": {"color": "#8c564b", "marker": "P"},
    "dino_finetuned_rt": {"color": "#17becf", "marker": "X"},  # real-time (downsample); matches SNR plot
    "dino_finetuned_opt": {"color": "#e377c2", "marker": "*"},
}
DETECTOR_ORDER = ["coherent_power", "cuda_dino", "3dB_power", "blob_detection",
                  "yolo", "dino_finetuned", "dino_finetuned_rt", "dino_finetuned_opt"]
DETECTOR_LABELS = {"cuda_dino": "zero_shot_dino",
                   "dino_finetuned_rt": "dino_finetuned_rt (real-time)",
                   "dino_finetuned_opt": "dino_finetuned +compile"}
# per-sample-rate colors (low -> high rate)
RATE_COLORS = ["#4575b4", "#91bfdb", "#fc8d59", "#d73027"]
# real-time budget lines are keyed to FFT bin size (frequency resolution), which sets the
# per-frame deadline independent of sample rate: budget = num_ffts_per_batch / bin_size.
DEFAULT_BUDGET_BINS_HZ = (20e3, 50e3, 100e3, 200e3)
DEVICE_STYLE = {"cpu": {"color": "#d95f02", "hatch": "//"},
                "cuda": {"color": "#1b9e77", "hatch": None}}
DEVICE_LABEL = {"cpu": "CPU", "cuda": "GPU"}
# Total device memory (GB) for the peak-GPU-memory reference lines. Distinct real SKUs so each
# is its own line; edit / pass gpu_mem_gb= to fig_compute_load to change the reference set.
# Jetson modules are unified LPDDR5 (shared CPU+GPU), i.e. the total system budget, not dedicated VRAM.
GPU_MEM_GB = {
    "Jetson Orin Nano 8 GB (edge)": 8,
    "RTX 4000 Ada (this machine) 20 GB": 20,
    "A100 40 GB": 40,
    "H100 80 GB": 80,
    "DGX Spark / Jetson AGX Thor 128 GB": 128,
}
# ESTIMATED live-pipeline-minus-detector GPU footprint (MB) per sample rate, for the grey
# "pipeline (est.)" bars — an exemplary full-system load. From the cuda_dino config: CUDA context
# ~625 MB + DPDK RX GPU region 512 MiB + display history 1024 MB (all ~fixed) + FFT/spectrogram
# working tensors that scale with fft_size (~rate). NOT measured live; see latency_eval.tex.
PIPELINE_MEM_MB_EST = {20e6: 2260.0, 100e6: 2340.0, 250e6: 2440.0, 500e6: 2640.0}


def label_for(det: str) -> str:
    return DETECTOR_LABELS.get(det, det)


def _style(det: str) -> dict:
    return DETECTOR_STYLE.get(det, {"color": "#555555", "marker": "x"})


def order_detectors(dets) -> list:
    dets = list(dets)
    return sorted(dets, key=lambda d: (DETECTOR_ORDER.index(d) if d in DETECTOR_ORDER
                                       else len(DETECTOR_ORDER), d))


def _rate_color(rate_hz: float, rates: list[float]) -> str:
    i = rates.index(rate_hz) if rate_hz in rates else 0
    return RATE_COLORS[i % len(RATE_COLORS)]


def _grid(n: int, ncol: int = 3):
    import matplotlib.pyplot as plt
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow),
                             layout="constrained", squeeze=False)
    return fig, axes, nrow, ncol


# --------------------------------------------------------------------------- #
# 1. per-detector CPU vs GPU latency histograms at ONE rate + RT budget line
# --------------------------------------------------------------------------- #
def fig_latency_hist(results: LatencyResults, rate_hz: float, bins: int = 28,
                     devices=("cpu", "cuda"), log_y: bool = True):
    """Grid of per-detector panels; horizontal histograms of CPU & GPU per-frame latency
    at ``rate_hz``, with a horizontal dashed line at that rate's real-time budget."""
    import matplotlib.pyplot as plt
    dets = order_detectors(results.detectors())
    budget = results.budget_ms(rate_hz)
    fig, axes, nrow, ncol = _grid(len(dets))

    for k, det in enumerate(dets):
        ax = axes[k // ncol][k % ncol]
        allv = []
        for dev in devices:
            lat = results.latency_samples(det, rate_hz, dev)
            if lat.size == 0:
                continue
            allv.append(lat)
            st = DEVICE_STYLE[dev]
            ax.hist(lat, bins=bins, orientation="horizontal", histtype="stepfilled",
                    alpha=0.45, color=st["color"], hatch=st["hatch"],
                    edgecolor=st["color"], label=f"{DEVICE_LABEL[dev]} (med {np.median(lat):.2f} ms)")
        ax.axhline(budget, ls="--", lw=1.4, color="k",
                   label=f"real-time budget {budget:.1f} ms")
        if log_y and allv:
            ax.set_yscale("log")
        ax.set_title(label_for(det), fontsize=10, fontweight="bold")
        ax.set_xlabel("count"); ax.set_ylabel("per-frame latency (ms)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", alpha=0.25)
    for j in range(len(dets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Per-frame latency — CPU vs GPU @ {rate_hz/1e6:.0f} MHz "
                 f"(fft={results.geometry[str(int(rate_hz))]['actual_fft_size']})",
                 fontsize=12, fontweight="bold")
    return fig


# --------------------------------------------------------------------------- #
# 2. per-detector histograms across all rates (one device) + all RT budget lines
# --------------------------------------------------------------------------- #
def fig_latency_hist_by_rate(results: LatencyResults, device: str = "cuda",
                             bins: int = 28, log_y: bool = True):
    """Per-detector horizontal latency histograms for every rate on one device, with a
    horizontal dashed real-time-budget line for each rate (a few threshold lines for
    reference)."""
    import matplotlib.pyplot as plt
    dets = order_detectors(results.detectors())
    rates = results.sample_rates()
    fig, axes, nrow, ncol = _grid(len(dets))
    seen_budget = {}

    for k, det in enumerate(dets):
        ax = axes[k // ncol][k % ncol]
        for r in rates:
            lat = results.latency_samples(det, r, device)
            if lat.size == 0:
                continue
            c = _rate_color(r, rates)
            ax.hist(lat, bins=bins, orientation="horizontal", histtype="stepfilled",
                    alpha=0.40, color=c, edgecolor=c, label=f"{r/1e6:.0f} MHz")
        for r in rates:
            b = results.budget_ms(r)
            key = round(b, 2)
            seen_budget[key] = r
            ax.axhline(b, ls="--", lw=1.1, color="0.35", alpha=0.8)
        if log_y:
            ax.set_yscale("log")
        ax.set_title(label_for(det), fontsize=10, fontweight="bold")
        ax.set_xlabel("count"); ax.set_ylabel("per-frame latency (ms)")
        ax.legend(fontsize=7, loc="upper right", title="sample rate")
        ax.grid(True, axis="y", alpha=0.25)
    for j in range(len(dets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    blurb = ", ".join(f"{v:.1f} ms" for v in sorted(seen_budget))
    fig.suptitle(f"Per-frame latency histograms on {DEVICE_LABEL.get(device, device)} — "
                 f"dashed = real-time budgets ({blurb})", fontsize=12, fontweight="bold")
    return fig


# --------------------------------------------------------------------------- #
# 3. median latency vs sample rate (CPU & GPU) + real-time budget curve
# --------------------------------------------------------------------------- #
def fig_latency_vs_rate(results: LatencyResults, log_y: bool = True,
                        budget_bin_sizes_hz=DEFAULT_BUDGET_BINS_HZ):
    """Per-detector median detector-only latency vs sample rate for CPU and GPU (p05-p95
    band), with the real-time budget for each FFT bin size drawn as a horizontal reference
    line (detector is real-time-capable for a given resolution where its line sits below
    that bin's budget)."""
    import matplotlib.pyplot as plt
    dets = order_detectors(results.detectors())
    rates = results.sample_rates()
    fig, axes, nrow, ncol = _grid(len(dets))
    rate_mhz = [r / 1e6 for r in rates]
    budgets = _bin_budgets(results, budget_bin_sizes_hz)

    for k, det in enumerate(dets):
        ax = axes[k // ncol][k % ncol]
        for dev in ("cpu", "cuda"):
            med, lo, hi, xs = [], [], [], []
            for r in rates:
                lat = results.latency_samples(det, r, dev)
                if lat.size == 0:
                    continue
                xs.append(r / 1e6)
                med.append(np.median(lat)); lo.append(np.percentile(lat, 5)); hi.append(np.percentile(lat, 95))
            if not xs:
                continue
            st = DEVICE_STYLE[dev]
            ax.plot(xs, med, "-o", color=st["color"], ms=4, lw=1.6, label=DEVICE_LABEL[dev])
            ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.18)
        for b_hz, ms in sorted(budgets.items()):
            ax.axhline(ms, ls="--", lw=1.0, color="0.4", zorder=1)
            ax.text(rate_mhz[-1], ms, f" {1e6/b_hz:.0f} µs FFT", va="bottom", ha="right",
                    fontsize=6.5, color="0.3")
        if log_y:
            ax.set_yscale("log")
        ax.set_xscale("log"); ax.set_xticks(rate_mhz); ax.set_xticklabels([f"{m:.0f}" for m in rate_mhz])
        ax.set_title(label_for(det), fontsize=10, fontweight="bold")
        ax.set_xlabel("sample rate (MHz)"); ax.set_ylabel("latency (ms)")
        ax.legend(fontsize=7, loc="upper left"); ax.grid(True, which="both", alpha=0.2)
    for j in range(len(dets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle("Detector-only latency vs sample rate "
                 "(dashed = real-time budget by FFT bin size)", fontsize=12, fontweight="bold")
    return fig


# --------------------------------------------------------------------------- #
# 3b. max real-time sample rate: latency-vs-rate crossing the frame budget
# --------------------------------------------------------------------------- #
def _budget0_ms(results: LatencyResults) -> float:
    """The ~constant per-frame real-time budget (frame duration) at the deployed design
    resolution: num_ffts_per_batch / reference_bin. This is the flat line detectors race."""
    p = results.params
    ref_bin = float(p.get("reference_span_hz", 500e6)) / float(p.get("reference_fft_size", 20480))
    return float(p.get("num_ffts_per_batch", 512)) / ref_bin * 1e3


def max_realtime_rate_mhz(results: LatencyResults, det: str, device: str = "cuda",
                          metric: str = "lat_min_ms") -> float:
    """Sample rate (MHz) at which the detector's latency reaches the real-time budget, from a
    log-log fit of measured latency vs rate. Interpolates within the tested range, extrapolates
    (power-law) beyond it. inf if the detector clears the budget with a non-positive slope."""
    rates = np.array(results.sample_rates(), dtype=float)
    L = np.array([_cell_value(results, det, r, metric, device) for r in rates])
    ok = np.isfinite(L) & (L > 0)
    if ok.sum() < 2:
        return float("nan")
    m, c = np.polyfit(np.log(rates[ok]), np.log(L[ok]), 1)
    if m <= 1e-9:
        return float("inf")
    rate_mhz = float(np.exp((np.log(_budget0_ms(results)) - c) / m) / 1e6)
    # A near-flat latency curve (bounded, rate-independent cost — e.g. the downsample path, whose
    # tile count doesn't grow with rate) has a tiny positive slope from measurement noise, which
    # extrapolates to an absurd crossing. Treat anything past a sane ceiling as effectively unbounded
    # (real-time at all practical rates) rather than printing a meaningless number.
    return float("inf") if rate_mhz > 1.0e4 else rate_mhz


def fig_max_rate(results: LatencyResults, device: str = "cuda", metric: str = "lat_min_ms",
                 rate_cap_mhz: float = 3000.0):
    """Latency vs sample rate, all detectors overlaid on one device, with the ~constant frame
    real-time budget as a single horizontal line. Each detector's line crossing the budget =
    its max feasible sample rate (read off the x-axis, and printed in the legend). The rising
    lines show the more-rate -> more-work-per-frame relationship."""
    import matplotlib.pyplot as plt
    dets = order_detectors(results.detectors())
    rates = np.array(results.sample_rates(), dtype=float)
    B0 = _budget0_ms(results)
    max_rates = {d: max_realtime_rate_mhz(results, d, device, metric) for d in dets}
    finite = [v for v in max_rates.values() if np.isfinite(v)]
    x_hi = min(rate_cap_mhz, max(finite) * 1.3) if finite else rates.max() / 1e6
    xs = np.logspace(np.log10(rates.min() / 1e6 * 0.8), np.log10(max(x_hi, rates.max() / 1e6 * 1.1)), 240)

    fig, ax = plt.subplots(figsize=(15, 8.7), layout="constrained")
    for det in dets:
        L = np.array([_cell_value(results, det, r, metric, device) for r in rates])
        ok = np.isfinite(L) & (L > 0)
        st = _style(det)
        mr = max_rates[det]
        lbl = f"{label_for(det)}  —  ≤ {mr:.0f} MS/s" if np.isfinite(mr) else f"{label_for(det)}  —  n/a"
        ax.plot(rates[ok] / 1e6, L[ok], st["marker"] + "-", color=st["color"], ms=6, lw=1.9, label=lbl)
        if ok.sum() >= 2:
            m, c = np.polyfit(np.log(rates[ok]), np.log(L[ok]), 1)
            ax.plot(xs, np.exp(m * np.log(xs * 1e6) + c), ":", color=st["color"], lw=1.1, alpha=0.65)
            if np.isfinite(mr) and mr <= xs.max():
                ax.plot([mr], [B0], st["marker"], color=st["color"], ms=12, mec="k", mew=0.7, zorder=6)
    ax.axhline(B0, ls="--", lw=1.8, color="k", zorder=5)
    ax.text(0.5, B0, f" real-time budget ~{B0:.0f} ms/frame* ",
            transform=ax.get_yaxis_transform(), va="bottom", ha="center", fontsize=19,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9))
    ax.text(0.99, 0.02,
            "* In our real-time pipeline we scale fft size with sample rate to standardize\n"
            "  the input time window maintain approximately 25kHz per frequency bin",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=16, color="0.3")
    for rm in rates / 1e6:
        ax.axvline(rm, color="0.85", lw=0.8, zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([20, 100, 250, 500, 1000, 2000])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.set_xlabel("sample rate (MS/s)", fontsize=20)
    ax.set_ylabel("detector per-frame latency (ms)", fontsize=20)
    ax.set_title("Real-time sample rate per detector", fontsize=24, fontweight="bold")
    ax.legend(fontsize=13, title="detector — max real-time rate", title_fontsize=14,
              loc="upper left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.18)
    return fig


# --------------------------------------------------------------------------- #
# 4. compute load: GFLOPs/frame + peak GPU memory
# --------------------------------------------------------------------------- #
def _cell_value(results: LatencyResults, det: str, rate_hz: float, column: str,
                device: str = "cuda") -> float:
    d = np.asarray(results.cells["detector"]).astype(str)
    rt = np.asarray(results.cells["sample_rate_hz"], dtype=float)
    dv = np.asarray(results.cells["device"]).astype(str)
    hit = np.nonzero((d == det) & np.isclose(rt, rate_hz) & (dv == device))[0]
    if hit.size == 0:
        return float("nan")
    return float(np.asarray(results.cells[column], dtype=float)[hit[0]])


def fig_compute_load(results: LatencyResults, device: str = "cuda",
                     log_flops: bool = True, log_mem: bool = True,
                     gpu_mem_gb: dict | None = None,
                     pipeline_mem_mb: dict | None = None):
    """Two panels: FLOPs per frame (GFLOPs) and peak GPU memory (MB), grouped bars over
    detectors, one bar per sample rate. This is the compute-load eval. Both axes log by
    default (values span ~5 orders of magnitude across detectors). The peak-memory panel draws
    horizontal reference lines for a few GPUs' total memory (``gpu_mem_gb``), and an extra grey
    ``pipeline (est.)`` cluster (``pipeline_mem_mb``) showing the estimated live-pipeline-minus-
    detector GPU footprint, so an example full-system load can be read alongside the detectors.
    See latency_eval.tex for the memory accounting the bars omit."""
    import matplotlib.pyplot as plt
    if gpu_mem_gb is None:
        gpu_mem_gb = GPU_MEM_GB
    if pipeline_mem_mb is None:
        pipeline_mem_mb = PIPELINE_MEM_MB_EST
    dets = order_detectors(results.detectors())
    rates = results.sample_rates()
    fig, axes = plt.subplots(1, 2, figsize=(7.0 * 2, 4.6), layout="constrained")
    x = np.arange(len(dets))
    w = 0.8 / max(1, len(rates))

    for panel, (col, title, ylabel, use_log) in enumerate(
            [("gflops", "Compute per frame", "GFLOPs / frame", log_flops),
             ("peak_mem_mb", "Peak GPU memory per frame", "GPU memory / frame (MB)", log_mem)]):
        ax = axes[panel]
        for i, r in enumerate(rates):
            off = (i - (len(rates) - 1) / 2) * w
            vals = np.array([_cell_value(results, det, r, col, device) for det in dets])
            ax.bar(x + off, vals, w, color=_rate_color(r, rates), label=f"{r/1e6:.0f} MHz", zorder=3)
            if col == "peak_mem_mb" and pipeline_mem_mb:   # stack grey pipeline overhead on top
                ov = float(pipeline_mem_mb.get(float(r), np.nan))
                ax.bar(x + off, np.full(len(dets), ov), w, bottom=vals,
                       color="0.6", edgecolor="0.4", linewidth=0.25, zorder=2,
                       label=("pipeline additional overhead" if i == 0 else "_nolegend_"))
        if col == "peak_mem_mb" and gpu_mem_gb:            # total-memory reference lines
            tform = ax.get_yaxis_transform()
            for name, gb in sorted(gpu_mem_gb.items(), key=lambda kv: kv[1]):
                ax.axhline(gb * 1000.0, ls="--", lw=1.0, color="0.35", zorder=1)
                ax.text(0.015, gb * 1000.0, f"{name}", transform=tform, va="bottom", ha="left",
                        fontsize=7.0, color="0.2",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
        if use_log:
            ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels([label_for(d) for d in dets], rotation=30, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_ylabel(ylabel)
        if col == "peak_mem_mb":                    # move pipeline overhead to end; legend top-right
            h, lab = ax.get_legend_handles_labels()
            order = [k for k in range(len(lab)) if lab[k] != "pipeline additional overhead"] + \
                    [k for k in range(len(lab)) if lab[k] == "pipeline additional overhead"]
            ax.legend([h[k] for k in order], [lab[k] for k in order],
                      fontsize=8, title="sample rate", loc="upper right")
        else:
            ax.legend(fontsize=8, title="sample rate", loc="lower right")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Compute load per frame ({DEVICE_LABEL.get(device, device)})",
                 fontsize=12, fontweight="bold")
    return fig


# --------------------------------------------------------------------------- #
# 5. clustered bars: mean per-frame latency, detector clusters x (rate, device)
# --------------------------------------------------------------------------- #
def _bin_budgets(results: LatencyResults, bin_sizes_hz) -> dict:
    """{bin_size_hz: real-time budget ms} for the given FFT bin sizes."""
    nfb = int(results.params.get("num_ffts_per_batch", 512))
    return {float(b): nfb / float(b) * 1e3 for b in bin_sizes_hz}


def _draw_bin_budgets(ax, budgets: dict, fontsize: float = 8.0, alternate: bool = True):
    """Draw horizontal dashed real-time-budget lines, labeled on-plot by FFT dwell time (the
    rate-independent knob that increases with the budget: budget = 512 x dwell = 512 / bin)."""
    tform = ax.get_yaxis_transform()   # x axes-fraction, y data
    for i, (b_hz, ms) in enumerate(sorted(budgets.items())):
        dwell_us = 1.0 / b_hz * 1e6
        ax.axhline(ms, ls="--", lw=1.2, color="0.30", zorder=4)
        left = (i % 2 == 0) if alternate else True
        ax.text(0.012 if left else 0.988, ms,
                f"real-time budget — {dwell_us:.0f} µs FFT window "
                f"({b_hz/1e3:.0f} kHz bin) → {ms:.2f} ms",
                transform=tform, va="bottom", ha="left" if left else "right",
                fontsize=fontsize, color="0.12", zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))


def fig_latency_bars(results: LatencyResults, metric: str = "lat_min_ms",
                     devices=("cpu", "cuda"), log_y: bool = True,
                     budget_bin_sizes_hz=DEFAULT_BUDGET_BINS_HZ, show_mean: bool = True):
    """Steady-state per-frame latency as clustered bars: one cluster per detector, and inside
    each cluster a bar for every sample rate x device (CPU then GPU). Bar height is the warm
    **min** (best achievable) by default; a black tick overlays the **mean** of the timed reps
    so you can see they agree (a large gap flags warmup/jitter). The real-time budget is drawn
    as horizontal dashed line(s), labeled on the plot.

    Latency spans ~5 orders of magnitude (GPU sub-ms .. CPU seconds), so the y-axis is log
    by default; pass ``log_y=False`` for a linear axis.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    dets = order_detectors(results.detectors())
    rates = results.sample_rates()
    present = [d for d in devices if d in np.asarray(results.cells["device"]).astype(str)]
    nbar = len(rates) * len(present)
    x = np.arange(len(dets))
    total_w = 0.86
    w = total_w / max(1, nbar)

    fig, ax = plt.subplots(figsize=(2.1 * len(dets) + 2.5, 5.4), layout="constrained")
    vmin = np.inf
    slot = 0
    for r in rates:                                   # sample rate outer, device inner (cpu then gpu)
        for dev in present:
            vals = np.array([_cell_value(results, det, r, metric, dev) for det in dets])
            means = np.array([_cell_value(results, det, r, "lat_mean_ms", dev) for det in dets])
            finite = vals[np.isfinite(vals) & (vals > 0)]
            if finite.size:
                vmin = min(vmin, float(finite.min()))
            offset = (slot - (nbar - 1) / 2) * w
            ax.bar(x + offset, vals, w, color=_rate_color(r, rates),
                   alpha=0.55 if dev == "cpu" else 1.0,
                   hatch="////" if dev == "cpu" else None,
                   edgecolor="black", linewidth=0.3, zorder=3)
            if show_mean:                             # mean overlay tick (should sit at bar top)
                ax.scatter(x + offset, means, marker="_", s=90, color="black",
                           linewidths=1.0, zorder=6)
            slot += 1

    if log_y:
        ax.set_yscale("log")
        if np.isfinite(vmin):
            ax.set_ylim(bottom=vmin * 0.5)
    # real-time budget lines keyed to FFT bin size (labeled on-plot, alternating sides)
    _draw_bin_budgets(ax, _bin_budgets(results, budget_bin_sizes_hz))

    ax.set_xticks(x); ax.set_xticklabels([label_for(d) for d in dets], rotation=20, ha="right")
    ax.set_ylabel("per-frame latency (ms) — bar = warm min, tick = mean")
    ax.set_title("Per-frame latency (steady-state min; mean tick)", fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    # legend: rate colors + device (hatch)
    from matplotlib.lines import Line2D
    rate_handles = [Patch(fc=_rate_color(r, rates), ec="black", lw=0.3, label=f"{r/1e6:.0f} MHz")
                    for r in rates]
    dev_handles = []
    if "cpu" in present:
        dev_handles.append(Patch(fc="0.7", hatch="////", ec="black", lw=0.3, alpha=0.55, label="CPU"))
    if "cuda" in present:
        dev_handles.append(Patch(fc="0.7", ec="black", lw=0.3, label="GPU"))
    if show_mean:
        dev_handles.append(Line2D([0], [0], marker="_", color="black", lw=0, mew=1.4, label="mean"))
    leg1 = ax.legend(handles=rate_handles, title="sample rate", fontsize=8,
                     loc="upper left", ncol=1)
    ax.add_artist(leg1)
    ax.legend(handles=dev_handles, title="device", fontsize=8, loc="upper left",
              bbox_to_anchor=(0.16, 1.0))
    return fig


# --------------------------------------------------------------------------- #
def make_all_figures(results: LatencyResults) -> dict:
    """The default figure set. `fig_latency_hist*` remain importable for a distributional
    view but are no longer emitted by default (superseded by the clustered bar chart)."""
    return {
        "max_rate": fig_max_rate(results),
        "latency_bars": fig_latency_bars(results),
        "latency_vs_rate": fig_latency_vs_rate(results),
        "compute_load": fig_compute_load(results),
    }


def print_min_vs_mean(results: LatencyResults, device: str = "cuda", warn_ratio: float = 1.25) -> None:
    """Report warm min vs mean per (detector, rate) on one device to confirm they agree (a
    large mean/min ratio flags residual warmup or run-to-run jitter, i.e. not steady state)."""
    dets = order_detectors(results.detectors())
    rates = results.sample_rates()
    print(f"\nSteady-state check on {DEVICE_LABEL.get(device, device)} "
          f"(mean/min > {warn_ratio:g} => not fully warm / jittery):")
    print(f"  {'detector':22s} {'rate':>7} {'min ms':>10} {'mean ms':>10} {'mean/min':>9} {'n':>4}")
    for det in dets:
        for r in rates:
            mn = _cell_value(results, det, r, "lat_min_ms", device)
            me = _cell_value(results, det, r, "lat_mean_ms", device)
            n = _cell_value(results, det, r, "n_reps", device)
            if not np.isfinite(mn) or mn <= 0:
                continue
            ratio = me / mn
            flag = "  <-- check" if ratio > warn_ratio else ""
            print(f"  {label_for(det):22s} {r/1e6:>5.0f}MHz {mn:>10.3f} {me:>10.3f} "
                  f"{ratio:>9.3f} {int(n):>4}{flag}")


def print_summary(results: LatencyResults, budget_bin_sizes_hz=DEFAULT_BUDGET_BINS_HZ) -> None:
    print("Per-rate frame geometry (deployed auto-FFT):")
    for r in results.sample_rates():
        g = results.geometry[str(int(r))]
        print(f"  {r/1e6:>5.0f} MHz  fft={g['actual_fft_size']:>6d}  "
              f"samples/frame={g['samples_per_frame']:>9d}  bin={g['resolution_hz']/1e3:6.2f} kHz")
    nfb = int(results.params.get("num_ffts_per_batch", 512))
    print(f"\nReal-time budget by FFT window (= num_ffts_per_batch[{nfb}] x window, "
          f"window = 1/bin_size; sample-rate independent; longer window -> more budget):")
    for b, ms in sorted(_bin_budgets(results, budget_bin_sizes_hz).items(), reverse=True):
        print(f"  {1e6/b:>5.0f} us window ({b/1e3:>3.0f} kHz bin) -> {ms:8.3f} ms")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="LatencyResults base/.npz/.json path.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Where to write PNGs (default <results>/latency_plots).")
    ap.add_argument("--dpi", type=int, default=120)
    args = ap.parse_args()

    results = LatencyResults.load(args.results)
    print_summary(results)
    out_dir = args.out_dir or (Path(args.results).parent / "latency_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in make_all_figures(results).items():
        p = out_dir / f"{name}.png"
        fig.savefig(p, dpi=args.dpi, bbox_inches="tight")
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

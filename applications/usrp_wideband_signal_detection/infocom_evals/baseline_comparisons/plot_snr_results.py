#!/usr/bin/env python3
"""Plot detector performance against SNR from a serialized SnrResults object.

This is the SNR-axis counterpart to ``signal_detection_experiments/plot_eval_results.py``
(which plots vs *attenuation*). It reads the reloadable object written by
``build_snr_results.py`` -- so re-styling, re-binning, or adding lines needs NO
recompute of the captures/fact tables -- and renders:

  * detection rate vs SNR, faceted by signal class / bandwidth (region level)
  * frame-level pixel metrics (precision/recall/F1/IoU) + false-positive area vs
    the per-frame mean-signal SNR

SNR is continuous, so the "vs SNR" curves bin the SNR axis (default 5 dB bins,
matching the physical attenuator step) and aggregate within each bin.

Importable helpers (usable straight from the notebook on a loaded object) + a CLI
that writes PNGs. Stdlib + numpy + matplotlib; no pandas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import snr_measurement as sm

BW_ORDER = ["<2MHz", "2-10MHz", "10-25MHz", "25-60MHz", ">=60MHz"]
DEFAULT_SNR_RANGE = (-20.0, 40.0)  # shared x-axis so facets/figures compare visually
DETECTOR_STYLE = {  # consistent colors/markers across every figure + the notebook
    # trained / deployed
    "coherent_power": {"color": "#1f77b4", "marker": "o"},
    "cuda_dino": {"color": "#d62728", "marker": "s"},       # zero-shot DINOv3
    # non-ML baselines
    "3dB_power": {"color": "#2ca02c", "marker": "^"},        # moving-average power
    "blob_detection": {"color": "#9467bd", "marker": "D"},   # image-processing blobs
    # fine-tuned ML models
    "yolo": {"color": "#ff7f0e", "marker": "v"},             # fine-tuned YOLO26
    "dino_finetuned": {"color": "#8c564b", "marker": "P"},   # fine-tuned DINOv3
    "dino_finetuned_rt": {"color": "#17becf", "marker": "X"},  # real-time (downsample) DINOv3
}
# aliases so the styles hold no matter which run-dir name a detector was written under
_STYLE_ALIASES = {
    "yolo26m": "yolo", "yolo26s": "yolo",
    "finetuned_dino": "dino_finetuned", "finetuned_dino_m2": "dino_finetuned",
}
# canonical left-to-right / legend order (unknown detectors sort to the end, by name)
DETECTOR_ORDER = ["coherent_power", "cuda_dino", "3dB_power", "blob_detection",
                  "yolo", "dino_finetuned", "dino_finetuned_rt"]
# display labels for plots (internal run-dir names stay as-is everywhere else).
# Keep in sync with plot_eval_results.DETECTOR_LABELS + eval_viz.DETECTOR_LABELS.
DETECTOR_LABELS = {"cuda_dino": "zero_shot_dino",
                   "dino_finetuned": "dino_finetuned (native)",
                   "dino_finetuned_rt": "dino_finetuned_rt (real-time)"}


def label_for(det) -> str:
    return DETECTOR_LABELS.get(det, det)


def _style(det):
    return DETECTOR_STYLE.get(det, DETECTOR_STYLE.get(_STYLE_ALIASES.get(det, ""),
                                                      {"color": None, "marker": "x"}))


def order_detectors(dets) -> list:
    """Sort detectors by DETECTOR_ORDER (canonical), unknowns alphabetically after."""
    dets = list(dets)
    return sorted(dets, key=lambda d: (DETECTOR_ORDER.index(d) if d in DETECTOR_ORDER
                                       else len(DETECTOR_ORDER), d))


# --------------------------------------------------------------------------- #
# binning + aggregation (operate directly on the results column arrays)
# --------------------------------------------------------------------------- #
def snr_bins(snr: np.ndarray, width: float,
             snr_range: tuple[float, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin_index_per_row, bin_center_values) for finite, in-range SNRs.

    A row's bin index is -1 when its SNR is NaN **or falls outside ``snr_range``**,
    so callers omit it. When ``snr_range`` is None the bins span the data extent.
    """
    finite = snr[np.isfinite(snr)]
    if snr_range is not None:
        lo, hi = float(snr_range[0]), float(snr_range[1])
    elif finite.size:
        lo = float(np.floor(np.nanmin(finite) / width) * width)
        hi = float(np.ceil(np.nanmax(finite) / width) * width)
    else:
        return np.full(snr.shape, -1, dtype=int), np.zeros(0)
    edges = np.arange(lo, hi + width, width)
    if len(edges) < 2:
        return np.full(snr.shape, -1, dtype=int), np.zeros(0)
    centers = edges[:-1] + width / 2.0
    idx = np.full(snr.shape, -1, dtype=int)
    ok = np.isfinite(snr) & (snr >= edges[0]) & (snr < edges[-1])  # out-of-range -> -1
    idx[ok] = np.clip(np.digitize(snr[ok], edges) - 1, 0, len(centers) - 1)
    return idx, centers


def detection_rate_vs_snr(region: dict, threshold: float, snr_bin_width: float,
                          facet_col: str | None = None,
                          snr_range: tuple[float, float] | None = None):
    """{(detector[, facet]): (centers, rates, counts)} using coverage >= threshold."""
    det = region["detector"]
    snr = region["snr_db"]
    cov = region["coverage"]
    bin_idx, centers = snr_bins(snr, snr_bin_width, snr_range)
    valid = (bin_idx >= 0) & np.isfinite(cov)
    facets = region[facet_col] if facet_col else np.array([""] * len(det))

    out: dict = {}
    keys = sorted(set(zip(det[valid], facets[valid])))
    for d, fv in keys:
        sel = valid & (det == d) & (facets == fv)
        rates = np.full(len(centers), np.nan)
        counts = np.zeros(len(centers), dtype=int)
        for b in range(len(centers)):
            m = sel & (bin_idx == b)
            n = int(m.sum())
            counts[b] = n
            if n:
                rates[b] = float((cov[m] >= threshold).mean())
        key = (d, fv) if facet_col else d
        out[key] = (centers, rates, counts)
    return out


def mean_metric_vs_snr(frame: dict, field: str, snr_bin_width: float,
                       snr_range: tuple[float, float] | None = None):
    """{detector: (centers, means, counts)} of a frame metric vs per-frame mean SNR."""
    det = frame["detector"]
    snr = frame["frame_snr_db"]
    val = frame[field]
    bin_idx, centers = snr_bins(snr, snr_bin_width, snr_range)
    valid = (bin_idx >= 0) & np.isfinite(val)
    out: dict = {}
    for d in sorted(set(det[valid])):
        sel = valid & (det == d)
        means = np.full(len(centers), np.nan)
        counts = np.zeros(len(centers), dtype=int)
        for b in range(len(centers)):
            m = sel & (bin_idx == b)
            n = int(m.sum())
            counts[b] = n
            if n:
                means[b] = float(np.mean(val[m]))
        out[d] = (centers, means, counts)
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_rate_vs_snr_by(results: sm.SnrResults, facet_col: str, facet_order=None,
                       threshold=0.1, snr_bin_width=5.0, title=None,
                       snr_range=DEFAULT_SNR_RANGE):
    import matplotlib.pyplot as plt

    region = results.region
    agg = detection_rate_vs_snr(region, threshold, snr_bin_width, facet_col, snr_range)
    present = sorted({fv for (_d, fv) in agg})
    facets = [f for f in (facet_order or present) if f in present and f not in ("", "unknown")]
    dets = order_detectors({d for (d, _fv) in agg})
    ncol = min(4, len(facets)) or 1
    nrow = (len(facets) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow),
                             squeeze=False, layout="constrained")
    for i, fv in enumerate(facets):
        ax = axes[i // ncol][i % ncol]
        for d in dets:
            if (d, fv) not in agg:
                continue
            centers, rates, _counts = agg[(d, fv)]
            ax.plot(centers, rates, label=label_for(d), **_style(d))
        ax.set_title(f"{facet_col}={fv}", fontsize=9)
        ax.set_xlabel("SNR (dB)  [higher → cleaner]", fontsize=8)
        ax.set_ylabel(f"detection rate (cov≥{threshold})", fontsize=8)
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        if snr_range is not None:
            ax.set_xlim(*snr_range)
    for j in range(len(facets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    if facets:
        axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(title or f"Detection rate vs SNR, by {facet_col}", fontsize=12)
    return fig


def fig_rate_vs_snr_overall(results: sm.SnrResults, threshold=0.1, snr_bin_width=5.0,
                            title=None, snr_range=DEFAULT_SNR_RANGE):
    import matplotlib.pyplot as plt

    agg = detection_rate_vs_snr(results.region, threshold, snr_bin_width, None, snr_range)
    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    for d in order_detectors(agg):
        centers, rates, _counts = agg[d]
        ax.plot(centers, rates, label=label_for(d), **_style(d))
    ax.set_xlabel("SNR (dB)  [higher → cleaner]")
    ax.set_ylabel(f"detection rate (cov≥{threshold})")
    ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    if snr_range is not None:
        ax.set_xlim(*snr_range)
    fig.suptitle(title or "Detection rate vs SNR (all signals)")
    return fig


def fig_frame_metrics_vs_snr(results: sm.SnrResults, snr_bin_width=5.0, title=None,
                             snr_range=DEFAULT_SNR_RANGE):
    import matplotlib.pyplot as plt

    metrics = [("precision", "precision"), ("recall", "recall"), ("f1", "F1"),
               ("iou", "pixel IoU"), ("fp_area_fraction", "false-positive area frac")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.8),
                             squeeze=False, layout="constrained")
    for ax, (field, label) in zip(axes[0], metrics):
        agg = mean_metric_vs_snr(results.frame, field, snr_bin_width, snr_range)
        for d in order_detectors(agg):
            centers, means, _counts = agg[d]
            ax.plot(centers, means, label=label_for(d), **_style(d))
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("per-frame mean SNR (dB)", fontsize=8)
        ax.grid(alpha=0.3)
        if field != "fp_area_fraction":
            ax.set_ylim(-0.02, 1.02)
        if snr_range is not None:
            ax.set_xlim(*snr_range)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title or "Frame-level pixel metrics + false-positive area vs SNR", fontsize=12)
    return fig


def make_all_figures(results: sm.SnrResults, threshold=0.1, snr_bin_width=5.0,
                     snr_range=DEFAULT_SNR_RANGE) -> dict:
    """All SNR figures as {name: Figure} -- the entry point the notebook reuses.

    ``snr_range`` (default (-20, 40) dB) pins a shared x-axis on every figure and
    omits any data outside it, so panels/figures line up for visual comparison. Pass
    ``snr_range=None`` to auto-fit each axis to the data instead.
    """
    return {
        "rate_vs_snr_by_class": fig_rate_vs_snr_by(
            results, "signal_class", None, threshold, snr_bin_width,
            "Detection rate vs SNR, by signal class", snr_range),
        "rate_vs_snr_by_bandwidth": fig_rate_vs_snr_by(
            results, "bandwidth", BW_ORDER, threshold, snr_bin_width,
            "Detection rate vs SNR, by bandwidth", snr_range),
        "rate_vs_snr_overall": fig_rate_vs_snr_overall(
            results, threshold, snr_bin_width, snr_range=snr_range),
        "frame_metrics_vs_snr": fig_frame_metrics_vs_snr(
            results, snr_bin_width, snr_range=snr_range),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True,
                    help="SnrResults base path / .npz / .json from build_snr_results.py.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Where to write PNGs (default <results-dir>/snr_plots).")
    ap.add_argument("--det-threshold", type=float, default=0.1,
                    help="Coverage threshold for 'detected' (GT masks are filled boxes).")
    ap.add_argument("--snr-bin-width", type=float, default=5.0,
                    help="SNR bin width in dB (default 5 = the physical attenuator step).")
    ap.add_argument("--snr-range", type=float, nargs=2, metavar=("LO", "HI"),
                    default=list(DEFAULT_SNR_RANGE),
                    help="Shared SNR x-axis limits in dB (default -20 40); data outside "
                         "is omitted. Pass e.g. --snr-range -30 50 to widen.")
    args = ap.parse_args()
    snr_range = tuple(args.snr_range)

    results = sm.SnrResults.load(args.results)
    print(f"loaded {len(results.region.get('snr_db', []))} region rows, "
          f"{len(results.frame.get('frame_snr_db', []))} frame rows, "
          f"{len(results.calibration)} calibration keys")

    import matplotlib
    matplotlib.use("Agg")

    out = args.out_dir or (Path(args.results).with_suffix("").parent / "snr_plots")
    out.mkdir(parents=True, exist_ok=True)
    figs = make_all_figures(results, args.det_threshold, args.snr_bin_width, snr_range)
    for name, fig in figs.items():
        p = out / f"{name}.png"
        fig.savefig(p, dpi=110)
        print(f"wrote {p}")
    print(f"\nAll SNR figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

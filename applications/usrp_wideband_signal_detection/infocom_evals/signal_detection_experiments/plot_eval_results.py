#!/usr/bin/env python3
"""Plot detector-evaluation results from the tidy fact tables.

Reads ``region_metrics.csv`` (per-annotation) and ``frame_pixel_metrics.csv`` (per-frame),
both produced by ``eval_detector_masks.py``, and renders comparison figures:

  * perf vs POWER (attenuation) faceted by signal class / bandwidth / pulse length, line per detector
  * perf vs BANDWIDTH and vs PULSE LENGTH, line per detector
  * frame-level pixel metrics (precision/recall/F1/IoU) and FALSE-POSITIVE area vs power

Metric sources:
  - Region-level (has per-signal attributes): detection_rate (coverage >= threshold),
    mean coverage, mean box_iou -> breakdowns by class / bandwidth / pulse-length / power.
  - Frame-level (whole-grid): precision, recall, f1, iou, fp_area_fraction -> vs power only
    (false-positive area is not attributable to a single signal's bandwidth/duration).

Stdlib + numpy + matplotlib (no pandas). Importable helpers + a CLI that writes PNGs.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

# canonical bucket orderings (BUCKETERS produce these labels)
BW_ORDER = ["<2MHz", "2-10MHz", "10-25MHz", "25-60MHz", ">=60MHz"]
LEN_ORDER = ["<10k", "10k-100k", "100k-1M", "1M-5M", ">=5M"]
DETECTOR_STYLE = {  # consistent colors/markers across all figures
    "coherent_power": {"color": "#1f77b4", "marker": "o"},
    "cuda_dino": {"color": "#d62728", "marker": "s"},
    "3dB_power": {"color": "#2ca02c", "marker": "^"},       # baseline: moving-average power
    "blob_detection": {"color": "#9467bd", "marker": "D"},  # baseline: image-processing blobs
}


def _f(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def load_region(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "detector": r["detector"],
                "attenuation_db": _f(r.get("attenuation_db")),
                "coverage": _f(r.get("coverage")),
                "box_iou": _f(r.get("box_iou")),
                "signal_class": r.get("bucket_signal_class", "unknown"),
                "bandwidth": r.get("bucket_bandwidth", "unknown"),
                "pulse_length": r.get("bucket_pulse_length", "unknown"),
            })
    return rows


def load_frame(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "detector": r["detector"],
                "attenuation_db": _f(r.get("attenuation_db")),
                "precision": _f(r.get("precision")),
                "recall": _f(r.get("recall")),
                "f1": _f(r.get("f1")),
                "iou": _f(r.get("iou")),
                "fp_area_fraction": _f(r.get("fp_area_fraction")),
                "mask_present": str(r.get("mask_present", "True")).lower() == "true",
            })
    return rows


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def detection_rate(rows, group_keys, threshold: float):
    """{group_tuple: (rate, n)} using coverage >= threshold; NaN coverage skipped."""
    tot = defaultdict(int); hit = defaultdict(int)
    for r in rows:
        cov = r["coverage"]
        if cov is None:
            continue
        key = tuple(r[k] for k in group_keys)
        tot[key] += 1
        hit[key] += 1 if cov >= threshold else 0
    return {k: (hit[k] / tot[k], tot[k]) for k in tot}


def mean_metric(rows, group_keys, field):
    """{group_tuple: (mean, n)} over non-NaN values of field."""
    acc = defaultdict(list)
    for r in rows:
        x = r[field]
        if x is None:
            continue
        acc[tuple(r[k] for k in group_keys)].append(x)
    return {k: (float(np.mean(v)), len(v)) for k, v in acc.items()}


def _attn_axis(rows):
    return sorted({r["attenuation_db"] for r in rows if r["attenuation_db"] is not None})


def _detectors(rows):
    return sorted({r["detector"] for r in rows})


def _style(det):
    return DETECTOR_STYLE.get(det, {"color": None, "marker": "^"})


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_rate_vs_power_by(region, facet_key, facet_order=None, threshold=0.1, title=None):
    """Detection rate vs attenuation, one subplot per facet value, line per detector."""
    import matplotlib.pyplot as plt

    dets = _detectors(region)
    attn = _attn_axis(region)
    facets = facet_order or sorted({r[facet_key] for r in region if r[facet_key] not in (None, "unknown")})
    facets = [f for f in facets if f not in (None, "unknown")]
    ncol = min(4, len(facets)) or 1
    nrow = (len(facets) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow),
                             squeeze=False, layout="constrained")
    agg = detection_rate(region, ("detector", facet_key, "attenuation_db"), threshold)
    for i, fv in enumerate(facets):
        ax = axes[i // ncol][i % ncol]
        for det in dets:
            ys = [agg.get((det, fv, a), (np.nan, 0))[0] for a in attn]
            ax.plot(attn, ys, label=det, **_style(det))
        ax.set_title(f"{facet_key}={fv}", fontsize=9)
        ax.set_xlabel("attenuation (dB)  [louder → quieter]", fontsize=8)
        ax.set_ylabel(f"detection rate (cov≥{threshold})", fontsize=8)
        ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    for j in range(len(facets), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(title or f"Detection rate vs power, by {facet_key}", fontsize=12)
    return fig


def fig_metric_vs_bucket(region, bucket_key, bucket_order, threshold=0.1, title=None):
    """Detection rate + mean box_iou + mean coverage vs a bucket axis, line per detector."""
    import matplotlib.pyplot as plt

    dets = _detectors(region)
    buckets = [b for b in bucket_order if any(r[bucket_key] == b for r in region)]
    rate = detection_rate(region, ("detector", bucket_key), threshold)
    iou = mean_metric(region, ("detector", bucket_key), "box_iou")
    cov = mean_metric(region, ("detector", bucket_key), "coverage")
    panels = [(f"detection rate (cov≥{threshold})", rate), ("mean box IoU", iou), ("mean coverage", cov)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False, layout="constrained")
    x = np.arange(len(buckets))
    for ax, (ylabel, agg) in zip(axes[0], panels):
        for det in dets:
            ys = [agg.get((det, b), (np.nan, 0))[0] for b in buckets]
            ax.plot(x, ys, label=det, **_style(det))
        ax.set_xticks(x); ax.set_xticklabels(buckets, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9); ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
    axes[0][0].set_xlabel(bucket_key); axes[0][0].legend(fontsize=8)
    fig.suptitle(title or f"Performance vs {bucket_key}", fontsize=12)
    return fig


def fig_frame_metrics_vs_power(frame, title=None):
    """precision / recall / F1 / IoU / false-positive-area vs attenuation, line per detector."""
    import matplotlib.pyplot as plt

    dets = _detectors(frame)
    attn = _attn_axis(frame)
    metrics = [("precision", "precision"), ("recall", "recall"), ("f1", "F1"),
               ("iou", "pixel IoU"), ("fp_area_fraction", "false-positive area frac")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 3.8),
                             squeeze=False, layout="constrained")
    for ax, (field, label) in zip(axes[0], metrics):
        agg = mean_metric(frame, ("detector", "attenuation_db"), field)
        for det in dets:
            ys = [agg.get((det, a), (np.nan, 0))[0] for a in attn]
            ax.plot(attn, ys, label=det, **_style(det))
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("attenuation (dB)", fontsize=8)
        ax.grid(alpha=0.3)
        if field != "fp_area_fraction":
            ax.set_ylim(-0.02, 1.02)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title or "Frame-level pixel metrics + false-positive area vs power", fontsize=12)
    return fig


ALL_FIGURES = {
    "rate_vs_power_by_class": lambda reg, frm, t: fig_rate_vs_power_by(reg, "signal_class", None, t),
    "rate_vs_power_by_bandwidth": lambda reg, frm, t: fig_rate_vs_power_by(reg, "bandwidth", BW_ORDER, t),
    "rate_vs_power_by_pulse_length": lambda reg, frm, t: fig_rate_vs_power_by(reg, "pulse_length", LEN_ORDER, t),
    "perf_vs_bandwidth": lambda reg, frm, t: fig_metric_vs_bucket(reg, "bandwidth", BW_ORDER, t),
    "perf_vs_pulse_length": lambda reg, frm, t: fig_metric_vs_bucket(reg, "pulse_length", LEN_ORDER, t),
    "frame_metrics_vs_power": lambda reg, frm, t: fig_frame_metrics_vs_power(frm, t),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", required=True, help="Dir with region_metrics.csv + frame_pixel_metrics.csv.")
    ap.add_argument("--out-dir", default=None, help="Where to write PNGs (default <tables-dir>/plots).")
    ap.add_argument("--det-threshold", type=float, default=0.1,
                    help="Coverage threshold for 'detected' (GT masks are filled boxes, so ~0.1 is a "
                         "reasonable 'found the signal'; try 0.05-0.3).")
    args = ap.parse_args()

    tdir = Path(args.tables_dir)
    out = Path(args.out_dir) if args.out_dir else tdir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    region = load_region(tdir / "region_metrics.csv")
    frame = load_frame(tdir / "frame_pixel_metrics.csv")
    print(f"loaded {len(region)} region rows, {len(frame)} frame rows")

    import matplotlib
    matplotlib.use("Agg")

    # patch the detection threshold into the region figures
    def _rate_by(key, order, title):
        return fig_rate_vs_power_by(region, key, order, args.det_threshold, title)
    figs = {
        "rate_vs_power_by_class": _rate_by("signal_class", None, "Detection rate vs power, by signal class"),
        "rate_vs_power_by_bandwidth": _rate_by("bandwidth", BW_ORDER, "Detection rate vs power, by bandwidth"),
        "rate_vs_power_by_pulse_length": _rate_by("pulse_length", LEN_ORDER, "Detection rate vs power, by pulse length"),
        "perf_vs_bandwidth": fig_metric_vs_bucket(region, "bandwidth", BW_ORDER, args.det_threshold,
                                                  "Performance vs signal bandwidth"),
        "perf_vs_pulse_length": fig_metric_vs_bucket(region, "pulse_length", LEN_ORDER, args.det_threshold,
                                                     "Performance vs pulse length (time duration)"),
        "frame_metrics_vs_power": fig_frame_metrics_vs_power(frame),
    }
    for name, fig in figs.items():
        p = out / f"{name}.png"
        fig.savefig(p, dpi=110)
        print(f"wrote {p}")
    print(f"\nAll figures in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

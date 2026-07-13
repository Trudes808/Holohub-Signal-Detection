"""Aggregate per-model eval CSVs into figures + tables for the report.

Reads eval_out/<model>/{frame_metrics,region_metrics}.csv for a list of models,
writes:
  reports/figs/*.png
  reports/summary.json           (machine-readable aggregates)
  reports/metrics_tables.md      (markdown tables to embed in report.md)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DET_THRESHOLD = 0.3  # a GT region is "detected" if coverage >= this

# consistent model colors/order
def _style(models):
    palette = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#666666", "#66a61e"]
    return {m: palette[i % len(palette)] for i, m in enumerate(models)}


def load(eval_root, models):
    frames, regions = {}, {}
    for m in models:
        fp = Path(eval_root) / m / "frame_metrics.csv"
        rp = Path(eval_root) / m / "region_metrics.csv"
        frames[m] = pd.read_csv(fp) if fp.exists() and fp.stat().st_size else pd.DataFrame()
        regions[m] = pd.read_csv(rp) if rp.exists() and rp.stat().st_size else pd.DataFrame()
    return frames, regions


def _mean(s):
    s = pd.to_numeric(s, errors="coerce")
    return float(np.nanmean(s)) if len(s) else float("nan")


# --------------------------------------------------------------------------- #
def fig_pixel_vs_atten(frames, colors, figs):
    metrics = [("iou", "Pixel IoU"), ("recall", "Recall"),
               ("precision", "Precision"), ("f1", "F1")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        for m, df in frames.items():
            if df.empty:
                continue
            d = df[df["is_signal"] == 1]
            g = d.groupby("attenuation_db")[col].apply(_mean).sort_index()
            ax.plot(g.index, g.values, "-o", ms=4, label=m, color=colors[m])
        ax.set_title(f"{title} vs attenuation (signal frames)")
        ax.set_xlabel("attenuation (dB)  — higher = weaker signal / lower SNR")
        ax.set_ylabel(title); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); p = figs / "pixel_metrics_vs_atten.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig_fp_vs_atten(frames, colors, figs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in frames.items():
        if df.empty:
            continue
        d = df[df["is_signal"] == 0]  # noise-only frames
        if d.empty:
            continue
        g = d.groupby("attenuation_db")["fp_area_fraction"].apply(_mean).sort_index()
        ax.plot(g.index, g.values, "-o", ms=4, label=m, color=colors[m])
    ax.set_title("False-positive area fraction on NOISE-ONLY frames vs attenuation")
    ax.set_xlabel("attenuation (dB)"); ax.set_ylabel("mean FP area fraction")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); p = figs / "fp_rate_vs_atten.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig_detection_vs_atten(regions, colors, figs):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, df in regions.items():
        if df.empty:
            continue
        df = df.copy(); df["det"] = pd.to_numeric(df["coverage"], errors="coerce") >= DET_THRESHOLD
        g = df.groupby("attenuation_db")["det"].mean().sort_index()
        ax.plot(g.index, g.values, "-o", ms=4, label=m, color=colors[m])
    ax.set_title(f"Region detection rate vs attenuation (coverage ≥ {DET_THRESHOLD})")
    ax.set_xlabel("attenuation (dB)"); ax.set_ylabel("detection rate")
    ax.grid(alpha=0.3); ax.set_ylim(-0.02, 1.02); ax.legend(fontsize=8)
    fig.tight_layout(); p = figs / "detection_vs_atten.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig_detection_by_bucket(regions, colors, figs, bucket_col, title, fname, order=None):
    models = [m for m in regions if not regions[m].empty]
    if not models:
        return None
    cats = order or sorted(set().union(*[set(regions[m][bucket_col].dropna()) for m in models]))
    x = np.arange(len(cats)); w = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(cats)), 5))
    for j, m in enumerate(models):
        df = regions[m].copy(); df["det"] = pd.to_numeric(df["coverage"], errors="coerce") >= DET_THRESHOLD
        rates = [df[df[bucket_col] == c]["det"].mean() if (df[bucket_col] == c).any() else np.nan for c in cats]
        ax.bar(x + j * w, rates, w, label=m, color=colors[m])
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
    ax.set_title(title); ax.set_ylabel("detection rate"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout(); p = figs / fname; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def fig_class_atten_heatmap(regions, figs, model):
    df = regions.get(model)
    if df is None or df.empty:
        return None
    df = df.copy(); df["det"] = pd.to_numeric(df["coverage"], errors="coerce") >= DET_THRESHOLD
    piv = df.pivot_table(index="signal_class", columns="attenuation_db", values="det", aggfunc="mean")
    piv = piv.sort_index()
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * piv.shape[1] + 3), 0.5 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=8)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_xlabel("attenuation (dB)"); ax.set_title(f"Detection rate: class × attenuation — {model}")
    for i in range(piv.shape[0]):
        for k in range(piv.shape[1]):
            v = piv.values[i, k]
            if not np.isnan(v):
                ax.text(k, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.02); fig.tight_layout()
    p = figs / f"class_atten_heatmap_{model}.png"; fig.savefig(p, dpi=110); plt.close(fig)
    return p


def summary_tables(frames, regions):
    lines = ["## Overall metrics (signal frames, pixel-level; test split)\n",
             "| model | IoU | F1 | precision | recall | FP-area (noise frames) |",
             "|---|---|---|---|---|---|"]
    summ = {}
    for m, df in frames.items():
        if df.empty:
            continue
        sig = df[df["is_signal"] == 1]; noi = df[df["is_signal"] == 0]
        row = {"iou": _mean(sig["iou"]), "f1": _mean(sig["f1"]),
               "precision": _mean(sig["precision"]), "recall": _mean(sig["recall"]),
               "fp_area": _mean(noi["fp_area_fraction"])}
        summ[m] = row
        lines.append(f"| {m} | {row['iou']:.3f} | {row['f1']:.3f} | {row['precision']:.3f} "
                     f"| {row['recall']:.3f} | {row['fp_area']:.4f} |")
    # detection rate by attenuation
    lines += ["", "## Region detection rate by attenuation (coverage ≥ %.2f)\n" % DET_THRESHOLD]
    attens = sorted(set().union(*[set(pd.to_numeric(r["attenuation_db"], errors="coerce").dropna().astype(int))
                                  for r in regions.values() if not r.empty])) if regions else []
    header = "| model | " + " | ".join(f"{a}dB" for a in attens) + " |"
    lines += [header, "|" + "---|" * (len(attens) + 1)]
    det_by_atten = {}
    for m, df in regions.items():
        if df.empty:
            continue
        df = df.copy(); df["det"] = pd.to_numeric(df["coverage"], errors="coerce") >= DET_THRESHOLD
        df["attenuation_db"] = pd.to_numeric(df["attenuation_db"], errors="coerce")
        g = df.groupby("attenuation_db")["det"].mean()
        det_by_atten[m] = {int(k): float(v) for k, v in g.items()}
        cells = " | ".join(f"{g.get(a, float('nan')):.2f}" for a in attens)
        lines.append(f"| {m} | {cells} |")
    return "\n".join(lines), {"overall": summ, "detection_by_atten": det_by_atten}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model dir names")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--heatmap-model", default=None)
    args = ap.parse_args()

    models = args.models.split(",")
    frames, regions = load(args.eval_root, models)
    colors = _style(models)
    reports = Path(args.reports); figs = reports / "figs"; figs.mkdir(parents=True, exist_ok=True)

    made = []
    made.append(fig_pixel_vs_atten(frames, colors, figs))
    made.append(fig_fp_vs_atten(frames, colors, figs))
    made.append(fig_detection_vs_atten(regions, colors, figs))
    made.append(fig_detection_by_bucket(regions, colors, figs, "signal_class",
                f"Detection rate by waveform class (coverage ≥ {DET_THRESHOLD})", "detection_by_class.png"))
    made.append(fig_detection_by_bucket(regions, colors, figs, "bandwidth_bucket",
                "Detection rate by bandwidth", "detection_by_bandwidth.png",
                order=["<2MHz", "2-10MHz", "10-25MHz", "25-60MHz", ">=60MHz", "unknown"]))
    made.append(fig_detection_by_bucket(regions, colors, figs, "length_bucket",
                "Detection rate by pulse length (samples)", "detection_by_length.png",
                order=["<10k", "10k-100k", "100k-1M", "1M-5M", ">=5M", "unknown"]))
    hm = args.heatmap_model or models[-1]
    made.append(fig_class_atten_heatmap(regions, figs, hm))

    tables_md, summ = summary_tables(frames, regions)
    (reports / "metrics_tables.md").write_text(tables_md)
    (reports / "summary.json").write_text(json.dumps(summ, indent=2))
    made = [str(p) for p in made if p]
    print("[report] figures:", *made, sep="\n  ")
    print("[report] wrote metrics_tables.md + summary.json")


if __name__ == "__main__":
    main()

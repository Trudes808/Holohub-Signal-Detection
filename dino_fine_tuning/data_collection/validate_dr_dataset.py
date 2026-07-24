#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 National Instruments Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Sanity-check the domain-randomized dataset against the sweep: reconstruct a few hundred AUGMENTED
training frames (raw float16 dB -> the exact dataset.py dB-domain augmentation -> clip) and compare
their power/floor/washout/envelope stats to the real ranges the sweep observed. Flags outliers:
frames whose noise floor falls outside the swept floor range, or that wash out (saturate) unrealistically.

Usage:
  ./validate_dr_dataset.py --dataset <data/dataset_dr> --sweep-stats <sweep/stats> --n 400 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import RFSegDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sweep-stats", default=None, help="defaults to the dataset meta's sweep_stats_dir.")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=None, help="where to write plots (default: <dataset>/validation).")
    ap.add_argument("--floor-pct", type=float, default=20.0)
    ap.add_argument("--peak-pct", type=float, default=99.5)
    args = ap.parse_args()

    ds = RFSegDataset(args.dataset, "train", augment=True, seed=0)
    if ds.storage != "float16_db":
        print(f"dataset storage is '{ds.storage}', not float16_db -> nothing to validate."); return
    out = Path(args.out or (Path(args.dataset) / "validation")); out.mkdir(parents=True, exist_ok=True)
    ssd = args.sweep_stats or ds.meta.get("sweep_stats_dir")
    print(f"dataset={args.dataset}\n  storage={ds.storage} vmin/vmax={ds.vmin:.2f}/{ds.vmax:.2f} "
          f"level-aug=[{ds.level_lo:.2f},{ds.level_hi:.2f}] dB  sweep_stats={ssd}")

    # ---- sweep observed floor range (the truth we must stay inside) ----------------------------
    sweep_lo = sweep_hi = sweep_med = None
    if ssd and (Path(ssd) / "floor_stats.json").exists():
        fs = json.loads((Path(ssd) / "floor_stats.json").read_text())["floor_stats"]
        floors_min = [v["min"] for role in fs for v in fs[role].values()]
        floors_max = [v["max"] for role in fs for v in fs[role].values()]
        floors_med = [v["median"] for role in fs for v in fs[role].values()]
        sweep_lo, sweep_hi, sweep_med = min(floors_min), max(floors_max), float(np.median(floors_med))
        print(f"  sweep floor range (dB): [{sweep_lo:.1f}, {sweep_hi:.1f}], median {sweep_med:.1f}")

    # ---- reconstruct augmented frames + stats --------------------------------------------------
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    floors, peaks, washouts, edge_droops = [], [], [], []
    example_imgs = []
    for k, i in enumerate(idx):
        pos = int(ds.rows[i]["mem_pos"])
        raw = np.asarray(ds.frames[pos], dtype=np.float32)          # dB, pre-clip, pre-aug
        mask = np.asarray(ds.masks[pos], dtype=np.float32)
        db, _ = ds._augment_db(raw, mask)                           # EXACT training aug (dB domain)
        floors.append(np.percentile(db, args.floor_pct))
        peaks.append(np.percentile(db, args.peak_pct))
        # envelope droop: band-edge floor vs center floor (per-column low percentile, smoothed ends)
        col_floor = np.percentile(db, 10, axis=0)
        edge = 0.5 * (col_floor[:col_floor.size // 16].mean() + col_floor[-col_floor.size // 16:].mean())
        edge_droops.append(edge - np.median(col_floor))
        img = np.clip((db - ds.vmin) / ds.db_span, 0.0, 1.0)        # what the model sees
        washouts.append(float((img >= 0.99).mean()))
        if len(example_imgs) < 8:
            example_imgs.append((img, float(floors[-1]), float(washouts[-1]),
                                 float(ds.rows[i].get("emulated_rate_hz", 0)) / 1e6))
    floors = np.array(floors); peaks = np.array(peaks); washouts = np.array(washouts)
    edge_droops = np.array(edge_droops)

    # ---- report --------------------------------------------------------------------------------
    print(f"\naugmented frames (n={len(idx)}):")
    print(f"  floor dB  (p{args.floor_pct:.0f}): min {floors.min():.1f}  med {np.median(floors):.1f}  "
          f"max {floors.max():.1f}")
    print(f"  peak dB   (p{args.peak_pct:.0f}): min {peaks.min():.1f}  med {np.median(peaks):.1f}  "
          f"max {peaks.max():.1f}  (peaks exceed sweep because training frames carry SIGNALS)")
    print(f"  edge droop (band-edge floor - center): med {np.median(edge_droops):.1f} dB")
    print(f"  washout (frac px >= 0.99): med {np.median(washouts)*100:.2f}%  "
          f"max {washouts.max()*100:.1f}%  frames >5% washed: {(washouts>0.05).sum()}")
    if sweep_lo is not None:
        out_lo = int((floors < sweep_lo).sum()); out_hi = int((floors > sweep_hi).sum())
        print(f"  FLOOR vs sweep: {out_lo} below {sweep_lo:.1f} dB, {out_hi} above {sweep_hi:.1f} dB "
              f"-> {100*(out_lo+out_hi)/len(floors):.1f}% outside the swept floor range "
              f"({'OK' if out_lo+out_hi == 0 else 'OUTLIERS'})")

    # ---- plots ---------------------------------------------------------------------------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(matplotlib unavailable: {exc}; skipping plots)"); return
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    ax[0].hist(floors, bins=40, color="steelblue", alpha=0.8)
    if sweep_lo is not None:
        for v, c, l in ((sweep_lo, "r", "sweep min"), (sweep_hi, "r", "sweep max"), (sweep_med, "g", "sweep med")):
            ax[0].axvline(v, color=c, ls="--", lw=1.2, label=l)
        ax[0].legend(fontsize=8)
    ax[0].set_title("augmented frame noise floor (dB) vs sweep range"); ax[0].set_xlabel("floor dB")
    ax[1].hist(washouts * 100, bins=40, color="indianred", alpha=0.8)
    ax[1].set_title("washout: % pixels saturated (>=0.99)"); ax[1].set_xlabel("% of frame washed out")
    ax[1].axvline(5, color="k", ls=":", lw=1, label="5% flag"); ax[1].legend(fontsize=8)
    ax[2].hist(edge_droops, bins=40, color="seagreen", alpha=0.8)
    ax[2].set_title("envelope droop (edge - center floor, dB)"); ax[2].set_xlabel("dB")
    fig.tight_layout(); p1 = out / "dr_stats_vs_sweep.png"; fig.savefig(p1, dpi=120); plt.close(fig)

    n = len(example_imgs)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.2 * ((n + 1) // 2), 6), squeeze=False)
    for a, (img, fl, wo, rate) in zip(axes.ravel(), example_imgs):
        a.imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        a.set_title(f"{rate:.1f} MS/s  floor{fl:.0f}dB  wash{wo*100:.0f}%", fontsize=8)
        a.axis("off")
    for a in axes.ravel()[n:]:
        a.axis("off")
    fig.tight_layout(); p2 = out / "dr_example_frames.png"; fig.savefig(p2, dpi=110); plt.close(fig)
    print(f"\nwrote {p1}\n      {p2}")


if __name__ == "__main__":
    main()

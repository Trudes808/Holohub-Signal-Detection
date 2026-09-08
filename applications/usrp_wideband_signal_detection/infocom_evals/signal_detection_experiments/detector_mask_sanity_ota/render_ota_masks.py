#!/usr/bin/env python3
"""Render 4-detector mask panels for the 1 s OTA capture (no ground truth).

The DINO runs use 10,485,760-sample frames (46/capture); the coherent run uses
5,242,880 (93/capture), so each reference (DINO) frame stacks the two coherent
frames covering its sample window. Panel layout per frame: raw spectrogram
(absolute RF axis); the four detector masks overlaid; DINO-FT RT-vs-native
disagreement. Also writes a per-frame occupancy + agreement CSV over all frames.

Usage: python3 render_ota_masks.py [frame ...]  (default: auto-pick by activity)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mask_eval_metrics as mem  # noqa: E402

BATCH = Path("/tmp/usrp_spectrograms/offline_eval")
STEM = "x410_ota_2g4_gain10_20260908"
DETS = {
    "coherent_power": "coherent_power",
    "cuda_dino (zero-shot)": "cuda_dino",
    "DINO-FT native (off-distribution @491.52)": "cuda_dino_finetuned",
    "DINO-FT RT (live path)": "cuda_dino_finetuned_rt",
}
REF = "cuda_dino_finetuned_rt"   # spectrogram + geometry reference (live path, wide FFT)
GRID_ROWS, GRID_COLS = 512, 1024
CENTER_MHZ, SPAN_MHZ = 2400.0, 491.52
FRAME_MS = 10485760 / 491.52e6 * 1e3

OUT = HERE


def manifest(det: str) -> dict[int, dict]:
    return {int(r["frame_number"]): r for r in mem.load_manifest(BATCH / det / STEM)}


def mask_for(det: str, row: dict) -> np.ndarray | None:
    m = mem.load_mask_any(BATCH / det / STEM / row["mask_npy"])
    return None if m is None else mem.resize_mask_nearest(m, GRID_ROWS, GRID_COLS)


def mask_for_ref_window(det: str, mani: dict[int, dict], ref_row: dict) -> np.ndarray | None:
    """Mask on the reference frame's sample window; stacks smaller frames (coherent)."""
    start, end = int(ref_row["file_offset_complex"]), int(ref_row["frame_end_complex"])
    rows = [r for r in mani.values()
            if int(r["file_offset_complex"]) >= start and int(r["frame_end_complex"]) <= end]
    if not rows:
        return None
    if len(rows) == 1:
        return mask_for(det, rows[0])
    rows.sort(key=lambda r: int(r["file_offset_complex"]))
    part_rows = GRID_ROWS // len(rows)
    parts = []
    for r in rows:
        m = mem.load_mask_any(BATCH / det / STEM / r["mask_npy"])
        if m is None:
            return None
        parts.append(mem.resize_mask_nearest(m, part_rows, GRID_COLS))
    return np.vstack(parts)[:GRID_ROWS]


def spec_db(row: dict) -> np.ndarray:
    t = np.load(BATCH / REF / STEM / row["spectrogram_tensor_npy"])
    db = (10.0 * np.log10(np.abs(t) ** 2 + 1e-12)).astype(np.float32)
    rows, cols = db.shape
    rr, cc = max(1, rows // GRID_ROWS), max(1, cols // GRID_COLS)
    db = db[: (rows // rr) * rr, : (cols // cc) * cc]
    db = db.reshape(rows // rr, rr, cols // cc, cc).max(axis=(1, 3))
    if db.shape[0] < GRID_ROWS:  # 256-row frames -> repeat rows up to grid
        db = np.repeat(db, GRID_ROWS // db.shape[0], axis=0)
    return db


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    manis = {d: manifest(d) for d in DETS.values()}
    frames = sorted(manis[REF])

    # per-frame occupancy + RT-vs-native agreement
    stats = []
    for fn in frames:
        row = {"frame": fn, "t_ms": (fn - 1) * FRAME_MS}
        masks = {}
        ref_row = manis[REF][fn]
        for det in DETS.values():
            m = mask_for_ref_window(det, manis[det], ref_row)
            masks[det] = m
            row[f"occ_{det}"] = float((m != 0).mean()) if m is not None else np.nan
        a = masks["cuda_dino_finetuned"]
        b = masks["cuda_dino_finetuned_rt"]
        if a is not None and b is not None:
            ga, gb = a != 0, b != 0
            union = np.logical_or(ga, gb).sum()
            row["rt_native_iou"] = float(np.logical_and(ga, gb).sum() / union) if union else 1.0
        stats.append(row)
    with open(OUT / "ota_frame_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        w.writeheader()
        w.writerows(stats)
    occ = np.array([s["occ_cuda_dino_finetuned_rt"] for s in stats])
    ious = np.array([s.get("rt_native_iou", np.nan) for s in stats], dtype=float)
    print(f"frames={len(stats)}  RT occupancy mean={occ.mean()*100:.2f}%  "
          f"RT-vs-native IoU mean={np.nanmean(ious):.3f} median={np.nanmedian(ious):.3f}")

    if len(sys.argv) > 1:
        picks = [int(a) for a in sys.argv[1:]]
    else:
        by_occ = sorted(stats, key=lambda s: -s["occ_cuda_dino_finetuned_rt"])
        picks = sorted({by_occ[0]["frame"], by_occ[len(by_occ)//3]["frame"],
                        by_occ[2*len(by_occ)//3]["frame"], by_occ[-1]["frame"]})
    print("rendering frames:", picks)

    extent = [CENTER_MHZ - SPAN_MHZ/2, CENTER_MHZ + SPAN_MHZ/2, FRAME_MS, 0.0]
    for fn in picks:
        db = spec_db(manis[REF][fn])
        vmin, vmax = np.percentile(db, 5), np.percentile(db, 99.8)
        masks = {label: mask_for_ref_window(det, manis[det], manis[REF][fn])
                 for label, det in DETS.items()}

        fig, axes = plt.subplots(2, 3, figsize=(19, 10), layout="constrained")
        axes = axes.ravel()

        def draw(ax, title):
            ax.imshow(db, aspect="auto", extent=extent, origin="upper",
                      cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("frequency (MHz)")
            ax.set_ylabel("time in frame (ms)")

        def overlay(ax, m, rgb, alpha=0.45):
            r, g, b = mcolors.to_rgb(rgb)
            rgba = np.zeros((*m.shape, 4), np.float32)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
            rgba[..., 3] = (m != 0) * alpha
            ax.imshow(rgba, aspect="auto", extent=extent, origin="upper")

        t0 = (fn - 1) * FRAME_MS
        draw(axes[0], f"{STEM}\nframe {fn} (t={t0:.1f} ms) — raw spectrogram")
        for ax, (label, det) in zip(axes[1:5], DETS.items()):
            m = masks[label]
            if m is None:
                draw(ax, f"{label} (MISSING)")
                continue
            draw(ax, f"{label} — mask on {100*(m != 0).mean():.2f}%")
            overlay(ax, m, (1.0, 0.25, 0.25))

        a = masks["DINO-FT native (off-distribution @491.52)"]
        b = masks["DINO-FT RT (live path)"]
        ga, gb = (a != 0), (b != 0)
        union = np.logical_or(ga, gb).sum()
        iou = (np.logical_and(ga, gb).sum() / union) if union else 1.0
        draw(axes[5], f"DINO-FT RT vs native — IoU {iou:.3f} "
                      f"(yellow=both, blue=native only, red=RT only)")
        overlay(axes[5], (ga & gb), (1.0, 0.9, 0.1), alpha=0.55)
        overlay(axes[5], (ga & ~gb), (0.2, 0.4, 1.0), alpha=0.65)
        overlay(axes[5], (~ga & gb), (1.0, 0.2, 0.2), alpha=0.65)

        out = OUT / f"ota_panels_frame{fn:03d}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

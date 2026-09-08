#!/usr/bin/env python3
"""Render cross-detector mask sanity panels for one capture.

Aligns detector runs with DIFFERENT frame sizes by absolute sample offset
(coherent runs at 2.62M samples/frame are stacked two-per-DINO-frame of 5.24M)
and renders, for a handful of frames: raw spectrogram + GT boxes, then each
detector's mask overlaid on the SAME spectrogram, plus an RT-vs-native DINO-FT
disagreement panel. Also writes a per-frame RT-vs-native mask agreement CSV.

Usage: python3 render_mask_sanity.py  (paths are baked for the comprehensive_ordered run)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mask_eval_metrics as mem  # noqa: E402

BATCH = Path("/tmp/usrp_spectrograms/offline_eval")
STEM = "comprehensive_ordered"
REF_DET = "cuda_dino_finetuned"          # reference geometry: 5.24M samples/frame
COHERENT_DET = "coherent_power_noguard"  # 2.62M samples/frame -> stack 2 per ref frame
OTHER_DINO = "cuda_dino"
RT_DET = "cuda_dino_finetuned_rt"
GRID_ROWS, GRID_COLS = 512, 1024         # common display/compare grid

OUT_DIR = HERE
FRAMES = [20, 60, 110, 160, 210, 260]


def manifest_by_frame(det: str) -> dict[int, dict]:
    run = BATCH / det / STEM
    return {int(r["frame_number"]): r for r in mem.load_manifest(run)}


def load_mask(det: str, row: dict) -> np.ndarray | None:
    return mem.load_mask_any(BATCH / det / STEM / row["mask_npy"])


def to_grid(mask: np.ndarray) -> np.ndarray:
    return mem.resize_mask_nearest(mask, GRID_ROWS, GRID_COLS)


def coherent_mask_for_ref_frame(cmani: dict[int, dict], ref_row: dict) -> np.ndarray | None:
    """Stack the two coherent frames covering the reference frame's sample window."""
    start = int(ref_row["file_offset_complex"])
    end = int(ref_row["frame_end_complex"])
    parts = []
    for fn in sorted(cmani):
        r = cmani[fn]
        s, e = int(r["file_offset_complex"]), int(r["frame_end_complex"])
        if s >= start and e <= end:
            m = load_mask(COHERENT_DET, r)
            if m is None:
                return None
            parts.append(mem.resize_mask_nearest(m, GRID_ROWS // 2, GRID_COLS))
    if not parts:
        return None
    return np.vstack(parts)[:GRID_ROWS]


def spectrogram_db(ref_row: dict) -> np.ndarray:
    t = np.load(BATCH / REF_DET / STEM / ref_row["spectrogram_tensor_npy"])
    p = np.abs(t) ** 2 + 1e-12
    db = (10.0 * np.log10(p)).astype(np.float32)
    # downsample rows to the display grid by max-pooling (keeps short bursts visible)
    rows, cols = db.shape
    rr = rows // GRID_ROWS
    db = db[: rr * GRID_ROWS].reshape(GRID_ROWS, rr, cols).max(axis=1)
    if cols != GRID_COLS:
        cc = cols // GRID_COLS
        db = db[:, : cc * GRID_COLS].reshape(GRID_ROWS, GRID_COLS, cc).max(axis=2)
    return db


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    from matplotlib.patches import Rectangle

    ref_mani = manifest_by_frame(REF_DET)
    rt_mani = manifest_by_frame(RT_DET)
    dino_mani = manifest_by_frame(OTHER_DINO)
    coh_mani = manifest_by_frame(COHERENT_DET)

    # ---- 1) RT vs native DINO-FT agreement over ALL frames ----
    agree_rows = []
    for fn in sorted(ref_mani):
        if fn not in rt_mani:
            continue
        a = load_mask(REF_DET, ref_mani[fn])
        b = load_mask(RT_DET, rt_mani[fn])
        if a is None or b is None:
            continue
        ga, gb = to_grid(a) != 0, to_grid(b) != 0
        inter, union = np.logical_and(ga, gb).sum(), np.logical_or(ga, gb).sum()
        agree_rows.append({
            "frame": fn,
            "native_on": int(ga.sum()), "rt_on": int(gb.sum()),
            "iou": (inter / union) if union else 1.0,
            "rt_only_frac": ((gb & ~ga).sum() / max(1, gb.sum())),
            "native_only_frac": ((ga & ~gb).sum() / max(1, ga.sum())),
        })
    with open(OUT_DIR / "dinoft_rt_vs_native_agreement.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agree_rows[0].keys()))
        w.writeheader()
        w.writerows(agree_rows)
    ious = np.array([r["iou"] for r in agree_rows])
    print(f"RT vs native DINO-FT mask agreement over {len(agree_rows)} frames: "
          f"mean IoU={ious.mean():.3f} median={np.median(ious):.3f} "
          f"p5={np.percentile(ious, 5):.3f} min={ious.min():.3f}")

    # ---- 2) panel figures ----
    spec_grid_note = None
    for fn in FRAMES:
        ref_row = ref_mani[fn]
        db = spectrogram_db(ref_row)
        vmin, vmax = np.percentile(db, 5), np.percentile(db, 99.7)

        gt = to_grid(mem.load_mask_any(BATCH / REF_DET / STEM / ref_row["gt_mask_npy"]))
        gt_payload = json.loads((BATCH / REF_DET / STEM / ref_row["gt_annotations_json"]).read_text())

        masks = {
            "coherent_power": coherent_mask_for_ref_frame(coh_mani, ref_row),
            "cuda_dino (zero-shot)": to_grid(load_mask(OTHER_DINO, dino_mani[fn])),
            "DINO-FT native (offline)": to_grid(load_mask(REF_DET, ref_row)),
            "DINO-FT RT (live path)": to_grid(load_mask(RT_DET, rt_mani[fn])),
        }

        span_mhz = float(gt_payload.get("span_hz") or 245.76e6) / 1e6
        n_samps = int(ref_row["frame_end_complex"]) - int(ref_row["file_offset_complex"])
        dur_ms = n_samps / 245.76e6 * 1e3
        t0_ms = int(ref_row["file_offset_complex"]) / 245.76e6 * 1e3
        extent = [-span_mhz / 2, span_mhz / 2, dur_ms, 0.0]

        fig, axes = plt.subplots(2, 3, figsize=(19, 10), layout="constrained")
        axes = axes.ravel()

        def draw(ax, title):
            ax.imshow(db, aspect="auto", extent=extent, origin="upper",
                      cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("freq (MHz, baseband)")
            ax.set_ylabel("time in frame (ms)")

        def overlay(ax, m, rgb, alpha=0.45):
            r, g, b = mcolors.to_rgb(rgb)
            rgba = np.zeros((*m.shape, 4), np.float32)
            rgba[..., 0], rgba[..., 1], rgba[..., 2] = r, g, b
            rgba[..., 3] = (m != 0) * alpha
            ax.imshow(rgba, aspect="auto", extent=extent, origin="upper")

        draw(axes[0], f"{STEM} frame {fn} (t={t0_ms:.1f} ms) — GT boxes")
        overlay(axes[0], gt, (0.1, 1.0, 0.3), alpha=0.30)
        for item in gt_payload.get("items", []):
            cs, ce = int(item.get("col_start", 0)), int(item.get("col_stop", 0))
            rs, re = int(item.get("row_start", 0)), int(item.get("row_stop", 0))
            gr, gc = gt.shape
            if ce <= cs or re <= rs:
                continue
            # gt grid indices are on the ORIGINAL gt grid; ours is resized -> scale
            orig_r = int(ref_row["fft_rows"]) * (int(ref_row["samples_per_row"]) and 1) or gr
            # annotations json rows/cols are on fft_rows x fft_cols of the ref run
            fr, fc = int(ref_row["fft_rows"]), int(ref_row["fft_cols"])
            x0 = -span_mhz / 2 + (cs / fc) * span_mhz
            w = ((ce - cs) / fc) * span_mhz
            y0 = (rs / fr) * dur_ms
            h = ((re - rs) / fr) * dur_ms
            axes[0].add_patch(Rectangle((x0, y0), w, h, fill=False,
                                        edgecolor="red", linewidth=0.9))
            axes[0].text(x0, y0, item.get("label", ""), color="red", fontsize=6,
                         va="bottom", ha="left", clip_on=True)

        order = ["coherent_power", "cuda_dino (zero-shot)",
                 "DINO-FT native (offline)", "DINO-FT RT (live path)"]
        for ax, name in zip(axes[1:5], order):
            m = masks[name]
            if m is None:
                draw(ax, f"{name} (MISSING)")
                continue
            on = 100.0 * (m != 0).mean()
            draw(ax, f"{name} — mask on {on:.1f}%")
            overlay(ax, m, (1.0, 0.25, 0.25))

        # panel 6: RT vs native disagreement
        a = masks["DINO-FT native (offline)"] != 0
        b = masks["DINO-FT RT (live path)"] != 0
        inter, union = (a & b).sum(), (a | b).sum()
        iou = inter / union if union else 1.0
        draw(axes[5], f"DINO-FT RT vs native — IoU {iou:.3f} "
                      f"(yellow=both, blue=native only, red=RT only)")
        overlay(axes[5], (a & b), (1.0, 0.9, 0.1), alpha=0.55)
        overlay(axes[5], (a & ~b), (0.2, 0.4, 1.0), alpha=0.65)
        overlay(axes[5], (~a & b), (1.0, 0.2, 0.2), alpha=0.65)

        out = OUT_DIR / f"panels_frame{fn:03d}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""3-detector mask comparison on the real OTA capture: coherent_power vs DINO-FT M2_rt vs M3_rt.

For each detector's offline-eval run dir, pair the emitted mask (mask_arrays) with the same frame's FFT
(spectrogram_tensors), render the dB spectrogram with the mask outlined, across many frames (rows) x the
3 detectors (cols). Purpose: check whether M3 over-fires (large false positives) vs coherent (energy
reference) and M2 on identical real input. Writes results/cmp3_masks.png + per-frame occupancy stats.
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CMP = Path("/tmp/usrp_spectrograms/cmp3")
OUT = Path(__file__).resolve().parent / "results"
DETS = [("coherent", "coherent_power"), ("m2_rt", "DINO-FT M2_dr (rt)"), ("m3_rt", "DINO-FT M3_491 (rt)")]
N_FRAMES = 20


def load_maps(run: Path):
    """frame_number -> (mask_path, tensor_path) for frames that have BOTH."""
    man = run / "frame_manifest.csv"
    if not man.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(man)):
        try:
            fn = int(r.get("frame_number", -1))
        except ValueError:
            continue
        mk = r.get("mask_npy") or ""
        tn = r.get("spectrogram_tensor_npy") or r.get("spectrogram_tensor") or ""
        mp = run / mk if mk else None          # manifest stores paths relative to run dir (incl. subdir)
        tp = run / tn if tn else None
        # fall back to conventional subdirs
        if mp is None or not mp.exists():
            cand = list((run / "mask_arrays").glob(f"*{fn}*.npy")) if (run / "mask_arrays").is_dir() else []
            mp = cand[0] if cand else None
        if tp is None or not tp.exists():
            cand = list((run / "spectrogram_tensors").glob(f"*{fn}*.npy")) if (run / "spectrogram_tensors").is_dir() else []
            tp = cand[0] if cand else None
        if mp and mp.exists() and tp and tp.exists():
            out[fn] = (mp, tp)
    return out


DISP_COLS = 2048   # downsample freq for a compact artifact-embeddable image


def _ds_cols(a, agg):
    """Block-reduce the freq axis of [rows, cols] to ~DISP_COLS (agg = np.mean or np.max)."""
    cols = a.shape[1]
    if cols <= DISP_COLS:
        return a
    f = cols // DISP_COLS
    keep = (cols // f) * f
    return agg(a[:, :keep].reshape(a.shape[0], cols // f, f), axis=2)


def db_of(tensor_path):
    t = np.load(tensor_path)
    if np.iscomplexobj(t):
        p = (t.real.astype(np.float32) ** 2 + t.imag.astype(np.float32) ** 2) + 1e-12
        db = 10.0 * np.log10(p)
    else:
        db = t.astype(np.float32)
    return db


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    maps = {k: load_maps(CMP / k) for k, _ in DETS}
    for k, _ in DETS:
        print(f"{k}: {len(maps[k])} frames with mask+tensor "
              f"(frame#s {sorted(maps[k])[:5]}{'...' if len(maps[k])>5 else ''})")
    # common frame numbers across all three
    common = sorted(set(maps["coherent"]) & set(maps["m2_rt"]) & set(maps["m3_rt"]))
    if len(common) < 3:
        # fall back: use m3's frames; other detectors show nearest available
        print("few common frames; using m3 frames + nearest for others")
        common = sorted(maps["m3_rt"])
    idx = np.linspace(0, len(common) - 1, min(N_FRAMES, len(common))).round().astype(int)
    frames = [common[i] for i in sorted(set(idx.tolist()))]
    print(f"rendering {len(frames)} frames: {frames}")

    # global dB range for a consistent colormap (percentiles over a sample)
    sample = db_of(maps["m3_rt"][frames[0]][1])
    vmin, vmax = np.percentile(sample, 5), np.percentile(sample, 99.5)

    occ = {k: [] for k, _ in DETS}
    n = len(frames)
    fig, axes = plt.subplots(n, 3, figsize=(18, 2.5 * n), squeeze=False)
    for i, fn in enumerate(frames):
        for j, (k, label) in enumerate(DETS):
            ax = axes[i][j]
            m = maps[k]
            use_fn = fn if fn in m else (min(m, key=lambda x: abs(x - fn)) if m else None)
            if use_fn is None:
                ax.axis("off"); continue
            mask_p, tens_p = m[use_fn]
            db = db_of(tens_p)
            mask = np.load(mask_p).astype(np.uint8)
            if mask.shape != db.shape:  # coherent may be on a different grid
                from math import gcd
                mask = np.repeat(np.repeat(mask, max(1, db.shape[0] // mask.shape[0]), 0),
                                 max(1, db.shape[1] // mask.shape[1]), 1)
                mask = mask[:db.shape[0], :db.shape[1]]
            o = 100 * mask.mean()               # occupancy from the FULL-res mask
            db_d = _ds_cols(db, np.mean)         # compact display
            mask_d = _ds_cols(mask.astype(np.float32), np.max)
            ax.imshow(db_d, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            if mask_d.any():
                ax.contour(mask_d, levels=[0.5], colors="cyan", linewidths=0.5)
            occ[k].append(o)
            if i == 0:
                ax.set_title(label, fontsize=11, fontweight="bold")
            ax.set_ylabel(f"f{fn}\nocc {o:.2f}%" if j == 0 else "", fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("3-detector masks on real 2.4 GHz OTA capture (cyan = detection) — many frames",
                 fontsize=13, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.995])
    p = OUT / "cmp3_masks.png"
    fig.savefig(p, dpi=80); plt.close(fig)
    summary = {k: {"mean_occ_pct": float(np.mean(occ[k])), "max_occ_pct": float(np.max(occ[k])),
                   "frames": len(occ[k])} for k in occ if occ[k]}
    (OUT / "cmp3_summary.json").write_text(json.dumps(summary, indent=2))
    print("\noccupancy summary:", json.dumps(summary, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    raise SystemExit(main())

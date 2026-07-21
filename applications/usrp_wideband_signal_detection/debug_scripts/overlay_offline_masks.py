#!/usr/bin/env python3
"""Render offline-eval detector masks as PNGs you can eyeball in VS Code.

For each --root (an offline eval --output-root) loads the predicted mask (mask_arrays/), the
ground-truth mask (gt_masks/), and — if available and non-zero — the spectrogram
(spectrogram_tensors/), resamples the prediction onto the GT grid, and reports agreement.

Two layouts:
  * default: one PNG per root, panels [spectrogram | GT | predicted | agreement].
  * --combined: ALL roots in a single figure (one row each) + a printed summary table — so
    high-SNR, low-SNR, and noise-only regimes are compared together (don't tunnel-vision on one).

Metrics: signal frames report IoU/TP/FP/FN vs GT; a NOISE-ONLY frame (empty GT) reports the
false-positive fraction (pred coverage) instead — IoU is meaningless there.
Agreement colors: green = TP, red = FP, blue = FN.

Usage (non-owner shell needs sudo + the venv python by absolute path):
  sudo /home/sat3737/holohub-dev/.venv/bin/python .../debug_scripts/overlay_offline_masks.py --combined \\
    --root /tmp/.../native/atten0:native@0dB   --root /tmp/.../downsample/atten0:ds@0dB \\
    --root /tmp/.../native/atten55:native@55dB --root /tmp/.../downsample/atten55:ds@55dB \\
    --root /tmp/.../native/noise:native@noise  --root /tmp/.../downsample/noise:ds@noise \\
    --out /tmp/usrp_spectrograms/overlays
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _first_npy(subdir: str, frame: int | None):
    files = sorted(glob.glob(os.path.join(subdir, "*.npy")))
    if frame is not None:
        hit = [f for f in files if f"_f{frame}_" in os.path.basename(f)]
        if hit:
            return hit[0]
    return files[0] if files else None


def _resize_nearest(mask: np.ndarray, shape) -> np.ndarray:
    if mask.shape == tuple(shape):
        return mask
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
    return (np.asarray(img) > 0).astype(np.uint8)


def _spectrogram_db(root: str, frame: int | None, shape):
    f = _first_npy(os.path.join(root, "spectrogram_tensors"), frame)
    if not f:
        return None
    a = np.load(f)
    if np.iscomplexobj(a):
        a = 10.0 * np.log10(np.abs(a) ** 2 + 1e-12)
    if not np.isfinite(a).any() or float(np.nanmax(a) - np.nanmin(a)) < 1e-6:
        return None
    if a.shape != tuple(shape):
        a = np.asarray(Image.fromarray(a.astype(np.float32)).resize((shape[1], shape[0]), Image.BILINEAR))
    return a


def load_root(root: str, label: str, frame: int | None):
    pred_f = _first_npy(os.path.join(root, "mask_arrays"), frame)
    gt_f = _first_npy(os.path.join(root, "gt_masks"), frame)
    if pred_f is None or gt_f is None:
        print(f"[{label}] missing mask_arrays or gt_masks under {root}; skipping")
        return None
    gt = (np.load(gt_f) > 0).astype(np.uint8)
    pred = _resize_nearest(np.load(pred_f), gt.shape)
    spec = _spectrogram_db(root, frame, gt.shape)
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    is_noise = int(gt.sum()) == 0
    iou = None if is_noise else tp / max(tp + fp + fn, 1)
    m = {"label": label, "gt": gt, "pred": pred, "spec": spec, "tp": tp, "fp": fp, "fn": fn,
         "iou": iou, "is_noise": is_noise, "pred_cov": float(pred.mean()), "gt_cov": float(gt.mean()),
         "fp_frac": float((pred & ~gt).mean()), "pred_f": os.path.basename(pred_f)}
    metric = f"FP={100*m['fp_frac']:.2f}% (noise)" if is_noise else f"IoU={iou:.3f}"
    print(f"[{label}] pred={100*m['pred_cov']:.2f}% gt={100*m['gt_cov']:.2f}% {metric} "
          f"TP={tp} FP={fp} FN={fn}  ({m['pred_f']})")
    return m


def _agree(pred, gt):
    a = np.zeros((*gt.shape, 3), np.uint8)
    a[(pred & gt) > 0] = (0, 200, 0)      # TP
    a[(pred & ~gt) > 0] = (220, 0, 0)     # FP
    a[(~pred & gt) > 0] = (0, 0, 220)     # FN
    return a


def _row(axrow, m):
    title = f"{m['label']}: " + (f"FP {100*m['fp_frac']:.2f}%" if m["is_noise"] else f"IoU {m['iou']:.3f}")
    if m["spec"] is not None:
        axrow[0].imshow(m["spec"], aspect="auto", cmap="viridis")
    axrow[0].set_title(title + ("" if m["spec"] is not None else " [no spec]"), fontsize=9)
    axrow[1].imshow(m["gt"], aspect="auto", cmap="gray"); axrow[1].set_title("ground truth", fontsize=9)
    axrow[2].imshow(m["pred"], aspect="auto", cmap="gray"); axrow[2].set_title("predicted", fontsize=9)
    axrow[3].imshow(_agree(m["pred"], m["gt"]), aspect="auto")
    axrow[3].set_title("green=TP red=FP blue=FN", fontsize=9)
    for a in axrow:
        a.set_xticks([]); a.set_yticks([])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", action="append", required=True, help="output-root, optionally 'path:label'.")
    p.add_argument("--frame", type=int, default=None)
    p.add_argument("--out", default="/tmp/usrp_spectrograms/overlays")
    p.add_argument("--combined", action="store_true", help="one figure with a row per root + summary.")
    args = p.parse_args()

    mets = []
    for spec in args.root:
        if ":" in spec and "/" not in spec.rsplit(":", 1)[1]:
            root, label = spec.rsplit(":", 1)
        else:
            root, label = spec, os.path.basename(os.path.normpath(spec))
        m = load_root(root, label, args.frame)
        if m:
            mets.append(m)
    if not mets:
        print("no roots loaded"); return 1
    os.makedirs(args.out, exist_ok=True)

    if args.combined:
        n = len(mets)
        fig, ax = plt.subplots(n, 4, figsize=(24, 3.2 * n), squeeze=False)
        for i, m in enumerate(mets):
            _row(ax[i], m)
        fig.tight_layout()
        out = os.path.join(args.out, "overlay_combined.png")
        fig.savefig(out, dpi=100); plt.close(fig)
        print(f"\nwrote {out}")
        print("\n== summary ==")
        print(f"{'label':<22}{'gt%':>8}{'pred%':>8}{'IoU/FP':>14}")
        for m in mets:
            val = f"FP {100*m['fp_frac']:.2f}%" if m["is_noise"] else f"IoU {m['iou']:.3f}"
            print(f"{m['label']:<22}{100*m['gt_cov']:>7.2f} {100*m['pred_cov']:>7.2f} {val:>14}")
    else:
        for m in mets:
            fig, ax = plt.subplots(1, 4, figsize=(26, 4))
            _row(ax, m)
            out = os.path.join(args.out, f"overlay_{m['label']}.png")
            fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
            print(f"[{m['label']}] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

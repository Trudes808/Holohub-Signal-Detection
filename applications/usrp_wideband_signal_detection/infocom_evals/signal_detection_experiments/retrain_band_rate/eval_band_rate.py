#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 National Instruments Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Rate-sweep + quiet-false-positive eval for the band/rate-invariant DINO retrain.

For each TorchScript model and each sample rate, emulate the capture at that rate (capture-chain
emulation, same code as training -> rate_augment) and run the model, comparing the predicted mask to
the rate-remapped GT. Reports, per (model, rate):
  - labeled capture (has annotations): pixel precision / recall / F1 / IoU  (regression across rate)
  - quiet capture (no annotations): false-positive rate = fired px / total px  (should be ~0)

This is the headline test: a band/rate-invariant model should hold F1 flat across rate on a labeled
band (e.g. 2400 ISM) and keep FP-rate ~0 across rate on a quiet band (e.g. 1150). Run the current and
retrained models together to see the improvement.

Usage:
  ./eval_band_rate.py --models /path/cur.ts:current /path/new.ts:retrained \
       --capture /home/bqn82/captures/quiet_1150.sigmf-meta --out ./results_1150 \
       --rates-hz 20.48e6 61.44e6 122.88e6 245.76e6 [--sweep-stats <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# rate_augment + rfdata live in <repo>/dino_fine_tuning/src
_SRC = Path(__file__).resolve().parents[5] / "dino_fine_tuning" / "src"
sys.path.insert(0, str(_SRC))
import rfdata as rf              # noqa: E402
import rate_augment as rate_aug  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="each 'path.ts:label' (or just path.ts).")
    p.add_argument("--capture", required=True, help="SigMF .sigmf-meta (labeled or quiet).")
    p.add_argument("--out", required=True)
    p.add_argument("--rates-hz", type=float, nargs="+",
                   default=[20.48e6, 61.44e6, 122.88e6, 245.76e6])
    p.add_argument("--n-frames", type=int, default=64, help="Frames evaluated per rate.")
    p.add_argument("--sweep-stats", default=None, help="sweep stats dir (envelopes.npz) for reshaping.")
    p.add_argument("--random-center", action="store_true",
                   help="Random center offset per frame (else f_c=0, whole band centered).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(spec):
    path, _, label = spec.partition(":")
    label = label or Path(path).stem
    model = torch.jit.load(path, map_location="cpu").eval()
    meta = {}
    mp = Path(path).with_suffix(".meta.json")
    if not mp.exists() and path.endswith(".ts"):
        mp = Path(path[:-3] + ".meta.json")
    if mp.exists():
        meta = json.loads(mp.read_text())
    return label, model, meta


def load_envelopes(sweep_stats):
    if not sweep_stats:
        return {}
    p = Path(sweep_stats) / "envelopes.npz"
    if not p.exists():
        return {}
    z = np.load(p)
    return {float(k[len("rate_"):]): z[k].astype(np.float32) for k in z.files if k.startswith("rate_")}


@torch.no_grad()
def predict(model, db_frames, vmin, vmax, thr, device):
    """db_frames [N,tile,nfft] float dB -> clip to [0,1] -> model -> uint8 masks [N,tile,nfft]."""
    span = max(vmax - vmin, 1e-6)
    x = np.clip((db_frames - vmin) / span, 0.0, 1.0).astype(np.float32)
    t = torch.from_numpy(x)[:, None].to(device)          # N,1,tile,nfft
    model = model.to(device)
    logits = model(t)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return (torch.sigmoid(logits.float()) >= thr).squeeze(1).cpu().numpy().astype(np.uint8)


def eval_rate(cap, mm, model, meta, rate, n_frames, envelopes, random_center, device, rng):
    nfft = int(meta.get("nfft", 1024))
    tile = int(meta.get("tile_rows", 256))
    vmin = float(meta.get("db_vmin", -46.934)); vmax = float(meta.get("db_vmax", 19.557))
    thr = float(meta.get("threshold", 0.85))
    src_rate = cap.sample_rate
    D = max(1, int(round(src_rate / rate)))
    need = tile * nfft * D
    n_avail = (cap.n_samples - 1) // need
    if n_avail <= 0:
        return None
    starts = np.linspace(0, (n_avail - 1) * need, min(n_frames, n_avail)).astype(np.int64)

    tp = fp = fn = pred_px = gt_px = total_px = 0
    db_batch, gt_batch = [], []
    for s in starts:
        chunk = np.asarray(mm[int(s):int(s) + need], dtype=np.complex64)
        half = (0.4 * src_rate - rate / 2.0)
        f_c = float(rng.uniform(-half, half)) if (random_center and half > 0) else 0.0
        db, gt, _ = rate_aug.emulate_frame(chunk, int(s), cap.annotations, src_rate, rate, f_c,
                                           nfft, tile, envelopes=envelopes, device=device)
        db_batch.append(db); gt_batch.append(gt)
    preds = predict(model, np.stack(db_batch), vmin, vmax, thr, device)
    for pred, gt in zip(preds, gt_batch):
        gt = gt.astype(bool); pr = pred.astype(bool)
        tp += int((pr & gt).sum()); fp += int((pr & ~gt).sum()); fn += int((~pr & gt).sum())
        pred_px += int(pr.sum()); gt_px += int(gt.sum()); total_px += gt.size
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec and rec and prec + rec > 0) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    noise_px = total_px - gt_px
    fp_rate = fp / noise_px if noise_px else float("nan")
    return {"rate_hz": rate, "n_frames": len(starts), "precision": prec, "recall": rec, "f1": f1,
            "iou": iou, "fp_rate": fp_rate, "gt_px": gt_px, "pred_px": pred_px, "tp": tp, "fp": fp, "fn": fn}


def plot(results, out_dir, labeled):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); skipping plot."); return
    metric = "f1" if labeled else "fp_rate"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, rows in results.items():
        rs = [r["rate_hz"] / 1e6 for r in rows]
        ys = [r[metric] for r in rows]
        ax.plot(rs, ys, "o-", label=label)
    ax.set_xlabel("sample rate (MS/s)")
    ax.set_ylabel("pixel F1 (labeled)" if labeled else "false-positive rate (quiet)")
    ax.set_title(("F1 vs rate" if labeled else "false-positive rate vs rate") + " — band/rate invariance")
    ax.grid(True, alpha=0.3); ax.legend()
    if not labeled:
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    p = os.path.join(out_dir, f"band_rate_{'f1' if labeled else 'fprate'}.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"wrote {p}")


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    cap = rf.load_capture(Path(args.capture))
    labeled = len(cap.annotations) > 0
    mm = cap.memmap()
    envelopes = load_envelopes(args.sweep_stats)
    rng = np.random.default_rng(0)
    print(f"capture {cap.stem}: {'LABELED' if labeled else 'QUIET'} ({len(cap.annotations)} anns), "
          f"src {cap.sample_rate/1e6:.2f} MS/s; rates {[r/1e6 for r in args.rates_hz]}")

    results = {}
    for spec in args.models:
        label, model, meta = load_model(spec)
        rows = []
        for rate in args.rates_hz:
            m = eval_rate(cap, mm, model, meta, rate, args.n_frames, envelopes, args.random_center,
                          args.device, rng)
            if m is None:
                print(f"  {label} @ {rate/1e6:.2f} MS/s: SKIP (capture too short for D={round(cap.sample_rate/rate)})")
                continue
            rows.append(m)
            key = "f1" if labeled else "fp_rate"
            print(f"  {label} @ {rate/1e6:7.2f} MS/s: {key}={m[key]:.4f} "
                  f"(prec={m['precision']:.3f} rec={m['recall']:.3f} iou={m['iou']:.3f})")
        results[label] = rows

    json.dump({"capture": cap.stem, "labeled": labeled, "results": results},
              open(os.path.join(args.out, "band_rate_results.json"), "w"), indent=2)
    plot(results, args.out, labeled)
    print(f"-> {args.out}/band_rate_results.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Ground-truth A/B: does each load-bearing normalization option actually help M3_491 detect?

Runs the held-out SigMF composites through the SAME front-end M3 was trained on, toggling one
normalization option off at a time, at the fixed deployment threshold. Metric = per-signal region
detection rate vs the generator's ground-truth annotations (overall + low-SNR), split dense/sparse.
Turning a load-bearing option off should DROP detection (esp. low-SNR / sparse)."""
import sys
sys.path.insert(0, "src")
import torch
import frontend as fe
from eval_heldout import load_model, per_snr_eval

DEVICE = "cuda"
THR = 0.6  # deployment threshold (config_live_v3_dino_ft.yaml)
SIGMF = "data/dataset_491/heldout_sigmf"

model = load_model("checkpoints/M3_491/best.pt", "configs/train_491.yaml", DEVICE)
print(f"model loaded; threshold={THR}; held-out = {SIGMF}\n")


def summarize(agg):
    out = {}
    for kind in ("dense", "sparse"):
        det = tot = ldet = ltot = 0
        for bucket, (d, t) in agg.get(kind, {}).items():
            det += d; tot += t
            if bucket in ("-3..3", "3..10"):   # low-SNR buckets
                ldet += d; ltot += t
        out[kind] = (det, tot, ldet, ltot)
    return out


def pct(d, t):
    return f"{100*d/t:4.0f}% ({d}/{t})" if t else "   -   "


variants = [
    ("baseline (flatten+adaptive+robust)", dict()),
    ("flatten_noise_floor OFF",            dict(flatten=False)),
    ("adaptive_normalization OFF",         dict(adaptive=False)),
    ("adaptive_robust_floor OFF",          dict(adaptive_robust_floor=False)),
    ("ALL norm OFF (flatten+adaptive)",    dict(flatten=False, adaptive=False)),
]

hdr = f"{'variant':38s} | {'dense all':>14s} {'dense lowSNR':>14s} | {'sparse all':>14s} {'sparse lowSNR':>14s}"
print(hdr); print("-" * len(hdr))
for name, over in variants:
    cfg = fe.FrontEndCfg(**over)
    agg = per_snr_eval(model, SIGMF, THR, DEVICE, cfg)
    r = summarize(agg)
    dd = r.get("dense", (0, 0, 0, 0)); ss = r.get("sparse", (0, 0, 0, 0))
    print(f"{name:38s} | {pct(dd[0],dd[1]):>14s} {pct(dd[2],dd[3]):>14s} | "
          f"{pct(ss[0],ss[1]):>14s} {pct(ss[2],ss[3]):>14s}")
print("\n(lowSNR = the -3..3 and 3..10 dB buckets; detection = any mask pixel overlaps the GT box)")

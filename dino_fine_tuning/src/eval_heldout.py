"""Held-out evaluation of a fine-tuned DINO-FT checkpoint on the 491.52 benchmark.

Two evals:
  1. Test-split pixel metrics: sweep the decision threshold over the deployment-front-end test tiles
     (frames_test/masks_test) -> micro IoU/precision/recall/F1; pick the best-F1 deploy threshold.
  2. Per-SNR region detection: run each held-out SigMF composite through the SAME front-end, threshold
     at the chosen value, and mark each annotation "detected" if any mask pixel overlaps its GT box.
     Bucketed by wfgt:power_db (the per-signal SNR set by the generator) -> detection-rate vs SNR, split
     by dense/sparse. This is the weak-signal recovery curve the retrain is meant to lift.

Optionally compares a second checkpoint (e.g. old) with --compare-ts (a traced .ts + its front-end mode).
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import frontend as fe
from model import DinoSegmenter

FFT, ROWS, TILE, FSAMP = 20480, 512, 256, 512 * 20480
FS = 491_520_000.0
SNR_EDGES = [(-3, 3), (3, 10), (10, 20), (20, 32)]


def load_model(ckpt_path, train_yaml, device):
    import yaml
    cfg = yaml.safe_load(open(train_yaml))
    ck = torch.load(ckpt_path, map_location=device)
    m = DinoSegmenter(cfg["weights_path"], feat_layers=tuple(cfg["feat_layers"]),
                      mode=ck.get("mode", "ft_lastN"), unfreeze_last_n=cfg["unfreeze_last_n"]).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def infer(model, tiles, device, bs=16):
    """tiles [N,1,256,1024] float32 [0,1] -> sigmoid prob [N,256,1024]."""
    out = []
    for i in range(0, tiles.shape[0], bs):
        x = tiles[i:i + bs].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.sigmoid(model(x).float())
        out.append(p.squeeze(1).cpu())
    return torch.cat(out)


def test_split_sweep(model, root, device):
    F = np.load(Path(root) / "frames_test.npy", mmap_mode="r")
    M = np.load(Path(root) / "masks_test.npy", mmap_mode="r")
    tiles = torch.from_numpy(np.asarray(F, dtype=np.float32) / 255.0).unsqueeze(1)
    gt = torch.from_numpy(np.asarray(M, dtype=np.float32))
    prob = infer(model, tiles, device)
    rows = []
    best = (0.5, -1)
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (prob >= thr).float()
        inter = (pred * gt).sum().item()
        p = inter / max(pred.sum().item(), 1)
        r = inter / max(gt.sum().item(), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        iou = inter / max((pred + gt - pred * gt).sum().item(), 1)
        rows.append(dict(thr=thr, precision=p, recall=r, f1=f1, iou=iou))
        if f1 > best[1]:
            best = (thr, f1)
    return rows, best[0]


def per_snr_eval(model, sigmf_dir, thr, device, cfg):
    dirp = Path(sigmf_dir)
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # kind -> snr_bucket -> [det,total]
    for meta_p in sorted(dirp.glob("*.sigmf-meta")):
        kind = "dense" if "dense" in meta_p.stem else "sparse"
        meta = json.loads(meta_p.read_text())
        anns = meta["annotations"]
        iq = np.fromfile(str(meta_p.with_suffix(".sigmf-data")), dtype=np.complex64)
        nfr = iq.size // FSAMP
        # per-frame mask on the [512,1024] grid
        for fidx in range(nfr):
            fr = torch.from_numpy(iq[fidx * FSAMP:(fidx + 1) * FSAMP]).to(device).unsqueeze(0)
            out = fe.compute_front_end(fr, FS, cfg, fft_size=FFT)
            tiles = out.tiles.to(device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pr = torch.sigmoid(model(tiles).float()).squeeze(1)  # [n_tiles,256,1024]
            mask = (pr >= thr).to(torch.uint8).cpu().numpy().reshape(ROWS, cfg.nfft)
            f0 = fidx * FSAMP
            for a in anns:
                s = int(a["core:sample_start"]); c = int(a["core:sample_count"])
                if s >= f0 + FSAMP or s + c <= f0:
                    continue
                snr = float(a.get("wfgt:power_db", 0.0))
                bucket = next((f"{lo}..{hi}" for lo, hi in SNR_EDGES if lo <= snr < hi), ">=32" if snr >= 32 else "<-3")
                r0 = max(0, (s - f0) // FFT); r1 = min(ROWS, -(-(s + c - f0) // FFT))
                lo = float(a["core:freq_lower_edge"]); hi = float(a["core:freq_upper_edge"])
                c0 = max(0, int(np.floor((lo + FS / 2) / FS * cfg.nfft)))
                c1 = min(cfg.nfft, int(np.ceil((hi + FS / 2) / FS * cfg.nfft)))
                if r1 <= r0 or c1 <= c0:
                    continue
                det = int(mask[r0:r1, c0:c1].sum() > 0)
                agg[kind][bucket][1] += 1
                agg[kind][bucket][0] += det
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-yaml", default="configs/train_491.yaml")
    ap.add_argument("--dataset", default="data/dataset_491")
    ap.add_argument("--out", default="checkpoints/M3_491/heldout_eval")
    args = ap.parse_args()
    device = "cuda"
    outd = Path(args.out); outd.mkdir(parents=True, exist_ok=True)
    cfg = fe.FrontEndCfg()
    model = load_model(args.ckpt, args.train_yaml, device)

    rows, best_thr = test_split_sweep(model, args.dataset, device)
    print("threshold sweep (test split, micro pixel):")
    for r in rows:
        print(f"  thr={r['thr']:.2f}  P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f} IoU={r['iou']:.3f}")
    print(f"best-F1 deploy threshold = {best_thr}")

    agg = per_snr_eval(model, Path(args.dataset) / "heldout_sigmf", best_thr, device, cfg)
    snr_rows = []
    print(f"\nper-SNR region detection @thr={best_thr}:")
    for kind in ("dense", "sparse"):
        for b in ["<-3", "-3..3", "3..10", "10..20", "20..32", ">=32"]:
            d, n = agg[kind].get(b, [0, 0])
            if n:
                rate = 100 * d / n
                snr_rows.append(dict(kind=kind, snr=b, det=d, total=n, rate=rate))
                print(f"  {kind:>6} SNR {b:>7}: {d}/{n} = {rate:.0f}%")
    (outd / "heldout_summary.json").write_text(json.dumps(
        {"threshold_sweep": rows, "best_thr": best_thr, "per_snr": snr_rows}, indent=2))

    # plot per-SNR detection rate
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order = ["<-3", "-3..3", "3..10", "10..20", "20..32", ">=32"]
    for kind in ("dense", "sparse"):
        xs, ys = [], []
        for i, b in enumerate(order):
            for r in snr_rows:
                if r["kind"] == kind and r["snr"] == b:
                    xs.append(i); ys.append(r["rate"])
        if xs:
            ax.plot(xs, ys, marker="o", label=kind)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order)
    ax.set_xlabel("per-signal SNR (dB)"); ax.set_ylabel("region detection rate %")
    ax.set_ylim(0, 105); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"M3_491 held-out detection vs SNR @thr={best_thr}")
    fig.tight_layout(); fig.savefig(outd / "heldout_snr.png", dpi=110)
    print("\nwrote", outd / "heldout_snr.png")


if __name__ == "__main__":
    main()

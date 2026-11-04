"""Side-by-side M5-S vs M6-S panels on SigMF scenes (real or synthetic): input + GT, each model's mask, and a
difference map (white = both, red = only model A, green = only model B).

  PYTHONPATH=src .venv-ml/bin/python src/compare_overlays.py --a checkpoints/M5S_491_ota/best.pt --b checkpoints/M6S_multirate/best.pt \
      --names M5-S M6-S --out <dir> --scene <x.sigmf-meta>[:frame|:auto] ... [--real] [--assume-continuous]
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import frontend as fe
import rates as R
from eval_scenes import Model, load_iq, parse_annotations


def frame_masks(model, tiles, thr, rows, nfft):
    prob = model.prob(tiles)
    return (prob >= thr).to(torch.uint8).cpu().numpy().reshape(-1, nfft)[:rows]


def pick_frame(spec: str):
    if ":" in spec:
        p, f = spec.rsplit(":", 1)
        return Path(p), f
    return Path(spec), "0"


def stats(mask, gt):
    m = mask.astype(bool); g = gt.astype(bool)
    occ = m.mean(); 
    if g.any():
        rec = (m & g).sum() / g.sum(); prec = (m & g).sum() / max(m.sum(), 1)
        return f"pred occ {100*occ:.1f}%  |  GT covered {100*rec:.0f}%  precision {100*prec:.0f}%"
    return f"pred occ {100*occ:.2f}%  (no labels)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--c", default=None, help="optional third model: rows become input, A, B, C, diff(B vs C)")
    ap.add_argument("--names", nargs="+", default=["M5-S", "M6-S"])
    ap.add_argument("--train-yaml", default="configs/train.yaml")
    ap.add_argument("--scene", nargs="+", required=True, help="<meta>[:frame|:auto]")
    ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--real", action="store_true"); ap.add_argument("--assume-continuous", action="store_true")
    ap.add_argument("--title", nargs="*", default=None, help="optional caption per scene (same order)")
    a = ap.parse_args()
    dev = "cuda"; cfg = fe.FrontEndCfg()
    A = Model(a.a, a.train_yaml, dev); B = Model(a.b, a.train_yaml, dev)
    C = Model(a.c, a.train_yaml, dev) if a.c else None
    names = list(a.names) + ["C"] * (3 - len(a.names)) if C is not None else list(a.names)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    index = []
    for si, spec in enumerate(a.scene):
        meta_p, fsel = pick_frame(spec)
        meta = json.loads(meta_p.read_text()); g = meta["global"]
        rate = float(g["core:sample_rate"]); dtype = g.get("core:datatype", "cf32_le")
        data_p = meta_p.with_suffix(".sigmf-data")
        bps = 4 if dtype.startswith("ci16") else (16 if dtype.startswith("cf64") else 8)
        n_total = data_p.stat().st_size // bps
        F = R.app_fft_size(rate); rows = R.ROWS_WIDE; FS = rows * F
        if n_total < FS:
            rows = max(256, (n_total // F) // 256 * 256); FS = rows * F
        nfr = max(1, n_total // FS)
        anns = parse_annotations(meta, rate, a.real, n_total, a.assume_continuous)
        if fsel == "auto":   # frame with the most GT pixels (synthetic) or the most energy variance (unlabelled)
            best, fidx = -1.0, 0
            for k in range(min(nfr, 8)):
                gt = fe.rasterize_gt(anns, k * FS, F, rows, cfg.nfft, rate).numpy()
                score = float(gt.mean())
                if score > best:
                    best, fidx = score, k
        else:
            fidx = int(fsel)
        iq = load_iq(data_p, dtype, fidx * FS, FS)
        fr = torch.from_numpy(iq).to(dev).unsqueeze(0)
        o = fe.compute_front_end(fr, rate, cfg, fft_size=F)
        img = o.resized[0].clamp(0, 1).cpu().numpy()
        gt = fe.rasterize_gt(anns, fidx * FS, F, rows, cfg.nfft, rate).numpy()
        ma = frame_masks(A, o.tiles, a.thr, rows, cfg.nfft); mb = frame_masks(B, o.tiles, a.thr, rows, cfg.nfft)
        mc = frame_masks(C, o.tiles, a.thr, rows, cfg.nfft) if C is not None else None
        ext = [-rate / 2e6, rate / 2e6, 0, rows * F / rate * 1e3]
        xs = np.linspace(ext[0], ext[1], cfg.nfft); ys = np.linspace(0, ext[3], rows)
        cap = (a.title[si] if a.title and si < len(a.title) else meta_p.stem)
        nrows = 5 if C is not None else 4
        fig, ax = plt.subplots(nrows, 1, figsize=(14, 3.6 * nrows + 0.5), sharex=True)
        ax[0].imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext)
        if gt.any(): ax[0].contour(xs, ys, gt, levels=[0.5], colors="red", linewidths=0.6)
        ax[0].set_title(f"{cap}  —  {rate/1e6:.2f} MSps, frame {fidx}: model input (robust-normalized), ground truth in red"
                        + (f"  [GT occ {100*gt.mean():.1f}%]" if gt.any() else "  [unlabelled]"), fontsize=11)
        mask_rows = [(ax[1], ma, names[0]), (ax[2], mb, names[1])] + ([(ax[3], mc, names[2])] if C is not None else [])
        for axi, m, nm in mask_rows:
            axi.imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
            axi.imshow(np.ma.masked_where(m == 0, m), aspect="auto", origin="lower", cmap="autumn", alpha=0.55, extent=ext, vmin=0, vmax=1)
            if gt.any(): axi.contour(xs, ys, gt, levels=[0.5], colors="cyan", linewidths=0.5)
            axi.set_title(f"{nm} mask (thr {a.thr}):  {stats(m, gt)}", fontsize=11)
        # difference map between the last two models (A vs B, or B vs C when a third model is given)
        d1, d2, n1, n2 = (ma, mb, names[0], names[1]) if C is None else (mb, mc, names[1], names[2])
        diff = np.zeros(img.shape + (3,), dtype=np.float32) + 0.12
        both = (d1 > 0) & (d2 > 0); onlya = (d1 > 0) & (d2 == 0); onlyb = (d2 > 0) & (d1 == 0)
        diff[both] = (1, 1, 1); diff[onlya] = (0.95, 0.25, 0.2); diff[onlyb] = (0.2, 0.9, 0.3)
        axd = ax[nrows - 1]
        axd.imshow(diff, aspect="auto", origin="lower", extent=ext)
        if gt.any(): axd.contour(xs, ys, gt, levels=[0.5], colors="cyan", linewidths=0.5)
        axd.set_title(f"difference: white = both, red = only {n1}, green = only {n2}  "
                      f"(only-{n1} {100*onlya.mean():.2f}%  only-{n2} {100*onlyb.mean():.2f}%  both {100*both.mean():.2f}%)", fontsize=11)
        for axi in ax: axi.set_ylabel("ms")
        axd.set_xlabel("MHz (relative to centre)")
        fig.tight_layout()
        png = out / f"{si:02d}_{meta_p.stem}_f{fidx}.png"
        fig.savefig(png, dpi=92); plt.close(fig)
        index.append({"png": png.name, "scene": meta_p.stem, "caption": cap, "rate_mhz": rate / 1e6, "frame": fidx,
                      "gt_occ": float(gt.mean()), "a": stats(ma, gt), "b": stats(mb, gt), "c": (stats(mc, gt) if mc is not None else None),
                      "names": names, "diff_pair": [n1, n2],
                      "only_a": float(onlya.mean()), "only_b": float(onlyb.mean()), "both": float(both.mean())})
        print("wrote", png, flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()

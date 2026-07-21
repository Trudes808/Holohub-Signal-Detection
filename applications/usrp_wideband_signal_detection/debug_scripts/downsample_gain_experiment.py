#!/usr/bin/env python3
"""Test whether an FFT-length processing-gain correction fixes the downsample low-SNR collapse.

Runs the eager fine-tuned M2 segmenter (no container, no rebuild) on three inputs per capture:
  native      : nfft=1024 FFT, tiled 256-row -> the trained geometry (reference).
  ds_uncorr   : wide FFT (downsample_fft) -> dB -> normalize -> resize freq to 1024 (naive downsample).
  ds_gaincorr : same, but subtract 10*log10(downsample_fft/nfft) dB BEFORE normalize.

Rationale for the correction (OTA-derivable, no per-capture fit): an unnormalized N-point FFT scales
a white-noise bin's power by N, so the noise floor rises 10*log10(N) dB with FFT length. The model was
trained at nfft=1024; the wide FFT sits 10*log10(fft/1024) dB hotter, pushing the input out of the
trained intensity distribution. Subtracting that (== normalizing the wide FFT's power to the 1024-pt
length) re-centers the noise floor. It depends only on the two FFT sizes -> reproducible on any data.

Scores IoU vs the offline binary's saved GT (gt_masks/*.npy), resampling every mask to the GT grid.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

VMIN, VMAX, THRESH, NFFT, TILE = -46.934, 19.557, 0.85, 1024, 256


def spec_power(iq: np.ndarray, nfft: int) -> torch.Tensor:
    rows = len(iq) // nfft
    blk = torch.from_numpy(np.ascontiguousarray(iq[: rows * nfft].reshape(rows, nfft)))
    spec = torch.fft.fftshift(torch.fft.fft(blk, dim=-1), dim=-1)
    return spec.real ** 2 + spec.imag ** 2  # [rows, nfft]


def db(iq: np.ndarray, nfft: int) -> torch.Tensor:
    return 10.0 * torch.log10(spec_power(iq, nfft) + 1e-12)  # [rows, nfft]


def norm01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp((x - VMIN) / (VMAX - VMIN), 0.0, 1.0)


@torch.no_grad()
def run_model(model, img: torch.Tensor, device) -> np.ndarray:
    """img [rows,1024] in [0,1] -> binary mask [rows,1024] uint8 (tile rows into 256, threshold)."""
    rows = img.shape[0]
    B = math.ceil(rows / TILE)
    padded = B * TILE
    if padded != rows:
        img = F.pad(img, (0, 0, 0, padded - rows))
    tiles = img.view(B, 1, TILE, NFFT).to(device)
    outs = []
    for i in range(0, B, 8):
        logits = model(tiles[i:i + 8])
        outs.append((torch.sigmoid(logits.float()) >= THRESH)[:, 0].to(torch.uint8).cpu())
    m = torch.cat(outs).view(padded, NFFT)[:rows]
    return m.numpy()


def to_gt_grid(mask: np.ndarray, shape) -> np.ndarray:
    if mask.shape == tuple(shape):
        return (mask > 0).astype(np.uint8)
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize((shape[1], shape[0]), Image.NEAREST)
    return (np.asarray(img) > 0).astype(np.uint8)


def iou_or_fp(pred: np.ndarray, gt: np.ndarray):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    if gt.sum() == 0:
        return f"FP {100*pred.mean():.2f}%", tp, fp, fn
    return f"IoU {tp/max(tp+fp+fn,1):.3f}", tp, fp, fn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", action="append", required=True,
                   help="'capture.sigmf-data,gt_root,label' (gt_root = an offline output-root with gt_masks/).")
    p.add_argument("--downsample-fft", type=int, default=10240)
    p.add_argument("--ckpt", default="/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/checkpoints/M2_ft/best.pt")
    p.add_argument("--train-yaml", default="/home/sat3737/holohub-dev/dino_fine_tuning/configs/train.yaml")
    p.add_argument("--src-dir", default="/home/sat3737/holohub-dev/dino_fine_tuning/src")
    p.add_argument("--dinov3-repo", default="/home/bqn82/dinov3")
    p.add_argument("--plot-out", default=None, help="If set, write a combined agreement figure here.")
    args = p.parse_args()

    import yaml
    sys.path.insert(0, args.dinov3_repo)
    sys.path.insert(0, args.src_dir)
    from model import DinoSegmenter  # noqa

    device = "cuda"
    tcfg = yaml.safe_load(open(args.train_yaml))
    ck = torch.load(args.ckpt, map_location=device)
    model = DinoSegmenter(tcfg["weights_path"], feat_layers=tuple(tcfg["feat_layers"]),
                          mode=ck.get("mode", "ft_lastN"), unfreeze_last_n=int(tcfg.get("unfreeze_last_n", 4))).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    off = 10.0 * math.log10(args.downsample_fft / NFFT)
    print(f"gain correction = 10*log10({args.downsample_fft}/{NFFT}) = {off:.2f} dB\n")
    print(f"{'case':<10}{'mode':<14}{'metric':>16}{'TP':>12}{'FP':>12}{'FN':>12}")

    cases_plot = []
    for case in args.case:
        cap, gt_root, label = case.split(",")
        iq = np.fromfile(cap, np.complex64)
        gt_f = sorted(glob.glob(os.path.join(gt_root, "gt_masks", "*.npy")))
        gt = (np.load(gt_f[0]) > 0).astype(np.uint8)

        # native (reference)
        native_in = norm01(db(iq, NFFT))
        # downsample uncorrected + gain-corrected (bilinear over dB)
        wide = db(iq, args.downsample_fft)                              # [rows_w, wide]
        ds_uncorr = F.interpolate(norm01(wide)[None, None], size=(wide.shape[0], NFFT),
                                  mode="bilinear", align_corners=False)[0, 0]
        ds_corr = F.interpolate(norm01(wide - off)[None, None], size=(wide.shape[0], NFFT),
                                mode="bilinear", align_corners=False)[0, 0]
        # power-domain: mean-pool POWER over freq groups of D, then gain-correct + dB (linear merge
        # preserves strong narrowband bins better than averaging dB). No resize -> already NFFT wide.
        wideP = spec_power(iq, args.downsample_fft)                     # [rows_w, wide]
        D = args.downsample_fft // NFFT
        Pmean = F.avg_pool1d(wideP[:, None, :], kernel_size=D, stride=D)[:, 0, :]  # [rows_w, NFFT]
        ds_power = norm01(10.0 * torch.log10(Pmean + 1e-12) - off)

        preds = {}
        for mode, inp in [("native", native_in), ("ds_uncorr", ds_uncorr),
                          ("ds_gaincorr", ds_corr), ("ds_power", ds_power)]:
            pred = to_gt_grid(run_model(model, inp, device), gt.shape)
            preds[mode] = pred
            metric, tp, fp, fn = iou_or_fp(pred, gt)
            print(f"{label:<10}{mode:<14}{metric:>16}{tp:>12}{fp:>12}{fn:>12}")
        print()
        cases_plot.append((label, gt, preds))

    if args.plot_out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        modes = ["native", "ds_uncorr", "ds_gaincorr", "ds_power"]
        nrow = len(cases_plot)
        fig, ax = plt.subplots(nrow, 1 + len(modes), figsize=(6 * (1 + len(modes)), 3.4 * nrow), squeeze=False)
        for i, (label, gt, preds) in enumerate(cases_plot):
            ax[i][0].imshow(gt, aspect="auto", cmap="gray"); ax[i][0].set_title(f"{label}  GT", fontsize=10)
            for j, mode in enumerate(modes):
                pred = preds[mode]
                agree = np.zeros((*gt.shape, 3), np.uint8)
                agree[(pred & gt) > 0] = (0, 200, 0)
                agree[(pred & ~gt) > 0] = (220, 0, 0)
                agree[(~pred & gt) > 0] = (0, 0, 220)
                metric, *_ = iou_or_fp(pred, gt)
                ax[i][j + 1].imshow(agree, aspect="auto")
                ax[i][j + 1].set_title(f"{mode}  {metric}", fontsize=10)
            for a in ax[i]:
                a.set_xticks([]); a.set_yticks([])
        fig.suptitle("green=TP  red=FP  blue=FN   (rows: SNR regime; cols: GT, native, downsample uncorrected, "
                     "downsample gain-corrected)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        os.makedirs(os.path.dirname(args.plot_out), exist_ok=True)
        fig.savefig(args.plot_out, dpi=100); plt.close(fig)
        print(f"wrote {args.plot_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-frame latency of the downsample vs native finetuned-DINO path across sample rates.

Uses the deployed dynamic-FFT geometry (fft_sizing.auto_fft_size): a frame is 512 x auto_fft_size, so
  * downsample: freq always resized to 1024, rows always 512 -> 2 tiles of 256x1024, at EVERY rate
    (cost bounded / rate-independent -- the whole point).
  * native: 1024-pt FFT -> rows = 512*auto_fft/1024 -> rows/256 tiles -> cost scales with rate.
Times the full GPU path (FFT + resize + tiled forward) with warmup + reps, vs the real-time frame
budget. Content-independent (random IQ) since latency depends on tensor shapes, not values.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "infocom_evals", "latency_comp_eval"))
from fft_sizing import frame_geometry  # noqa

NFFT, TILE = 1024, 256


def load_model(ckpt, train_yaml, src_dir, dinov3_repo, device):
    sys.path.insert(0, dinov3_repo)
    sys.path.insert(0, src_dir)
    from model import DinoSegmenter  # noqa
    tcfg = yaml.safe_load(open(train_yaml))
    ck = torch.load(ckpt, map_location=device)
    m = DinoSegmenter(tcfg["weights_path"], feat_layers=tuple(tcfg["feat_layers"]),
                      mode=ck.get("mode", "ft_lastN"), unfreeze_last_n=int(tcfg.get("unfreeze_last_n", 4))).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m


@torch.no_grad()
def time_ms(fn, warmup=3, reps=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


@torch.no_grad()
def run_tiles(model, tiles):
    out = []
    for i in range(0, tiles.shape[0], 8):
        out.append(torch.sigmoid(model(tiles[i:i + 8]).float()) >= 0.85)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rates", type=float, nargs="+", default=[20e6, 100e6, 250e6, 500e6])
    p.add_argument("--ckpt", default="/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/checkpoints/M2_ft/best.pt")
    p.add_argument("--train-yaml", default="/home/sat3737/holohub-dev/dino_fine_tuning/configs/train.yaml")
    p.add_argument("--src-dir", default="/home/sat3737/holohub-dev/dino_fine_tuning/src")
    p.add_argument("--dinov3-repo", default="/home/bqn82/dinov3")
    args = p.parse_args()

    device = "cuda"
    model = load_model(args.ckpt, args.train_yaml, args.src_dir, args.dinov3_repo, device)

    print(f"{'rate(MHz)':>9}{'auto_fft':>9}{'budget_ms':>11}{'ds_tiles':>9}{'ds_ms':>9}"
          f"{'nat_tiles':>10}{'nat_ms':>9}{'real-time?':>26}")
    for rate in args.rates:
        g = frame_geometry(rate)
        wide = g.actual_fft_size
        rows_w = g.num_ffts_per_batch                      # 512 by construction
        rows_n = g.samples_per_frame // NFFT
        ds_tiles = math.ceil(rows_w / TILE)                # ~2, constant
        nat_tiles = math.ceil(rows_n / TILE)               # scales with rate

        iq = torch.randn(g.samples_per_frame, dtype=torch.complex64, device=device)

        def ds_path():
            blk = iq[:rows_w * wide].view(rows_w, wide)
            spec = torch.fft.fftshift(torch.fft.fft(blk, dim=-1), dim=-1)
            dbimg = 10.0 * torch.log10(spec.real ** 2 + spec.imag ** 2 + 1e-12)
            img = torch.clamp((dbimg - (-46.934 + 10 * math.log10(wide / NFFT))) / (19.557 + 46.934), 0, 1)
            img = F.interpolate(img[None, None], size=(rows_w, NFFT), mode="bilinear", align_corners=False)[0, 0]
            pad = ds_tiles * TILE - rows_w
            if pad > 0:
                img = F.pad(img, (0, 0, 0, pad))
            run_tiles(model, img.view(ds_tiles, 1, TILE, NFFT))

        def nat_path():
            blk = iq[:rows_n * NFFT].view(rows_n, NFFT)
            spec = torch.fft.fftshift(torch.fft.fft(blk, dim=-1), dim=-1)
            dbimg = 10.0 * torch.log10(spec.real ** 2 + spec.imag ** 2 + 1e-12)
            img = torch.clamp((dbimg + 46.934) / (19.557 + 46.934), 0, 1)
            pad = nat_tiles * TILE - rows_n
            if pad > 0:
                img = F.pad(img, (0, 0, 0, pad))
            run_tiles(model, img.view(nat_tiles, 1, TILE, NFFT))

        ds_ms = time_ms(ds_path)
        nat_ms = time_ms(nat_path)
        rt = f"ds {'OK' if ds_ms < g.frame_budget_ms else 'NO'} / nat {'OK' if nat_ms < g.frame_budget_ms else 'NO'}"
        print(f"{rate/1e6:>9.0f}{wide:>9}{g.frame_budget_ms:>11.2f}{ds_tiles:>9}{ds_ms:>9.2f}"
              f"{nat_tiles:>10}{nat_ms:>9.2f}{rt:>26}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

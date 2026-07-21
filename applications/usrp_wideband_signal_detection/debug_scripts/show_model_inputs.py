#!/usr/bin/env python3
"""Render the ACTUAL spectrogram each detector mode feeds DINO, to see why downsample differs.

Reconstructs, from a capture's IQ, the exact model input for:
  * native     : nfft=1024 FFT -> [rows_native, 1024]           (per-tile native geometry)
  * downsample : wide FFT (downsample_fft_size) -> [rows_wide, wide] -> bilinear resize freq to 1024
                 -> [rows_wide, 1024]                            (what real_time_downsample feeds)
Both use the same fftshift + 10*log10(|.|^2) + clamp((db-vmin)/(vmax-vmin)) as the operator. Panels are
drawn at a COMMON on-screen size (aspect='auto') though the arrays differ dramatically in shape, with
the true resolution in each title — so the low-SNR detail the downsample throws away is visible.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_VMIN, DEFAULT_VMAX = -46.934, 19.557


def db_norm(iq: np.ndarray, nfft: int, vmin: float, vmax: float) -> torch.Tensor:
    rows = len(iq) // nfft
    blk = torch.from_numpy(np.ascontiguousarray(iq[: rows * nfft].reshape(rows, nfft)))
    spec = torch.fft.fftshift(torch.fft.fft(blk, dim=-1), dim=-1)
    db = 10.0 * torch.log10(spec.real ** 2 + spec.imag ** 2 + 1e-12)
    return torch.clamp((db - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)  # [rows, nfft]


def model_inputs(capture: str, nfft: int, wide: int, vmin: float, vmax: float):
    iq = np.fromfile(capture, dtype=np.complex64)
    native = db_norm(iq, nfft, vmin, vmax)                                  # [rows_n, nfft]
    wide_img = db_norm(iq, wide, vmin, vmax)                                # [rows_w, wide]
    ds = F.interpolate(wide_img[None, None], size=(wide_img.shape[0], nfft),
                       mode="bilinear", align_corners=False)[0, 0]          # [rows_w, nfft]
    return native.numpy(), ds.numpy(), wide_img.numpy()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", action="append", required=True, help="cf32 .sigmf-data, optionally 'path:label'.")
    p.add_argument("--nfft", type=int, default=1024)
    p.add_argument("--downsample-fft", type=int, default=10240)
    p.add_argument("--vmin", type=float, default=DEFAULT_VMIN)
    p.add_argument("--vmax", type=float, default=DEFAULT_VMAX)
    p.add_argument("--out", default="/tmp/usrp_spectrograms/overlays/model_inputs.png")
    args = p.parse_args()

    rows = []
    for spec in args.capture:
        if ":" in spec and "/" not in spec.rsplit(":", 1)[1]:
            path, label = spec.rsplit(":", 1)
        else:
            path, label = spec, os.path.basename(spec).split("_samples")[0]
        native, ds, wide_img = model_inputs(path, args.nfft, args.downsample_fft, args.vmin, args.vmax)
        rows.append((label, native, ds, wide_img))
        print(f"[{label}] native input {native.shape}  wide {wide_img.shape}  downsample input {ds.shape}")

    n = len(rows)
    fig, ax = plt.subplots(n, 3, figsize=(18, 4.2 * n), squeeze=False)
    for i, (label, native, ds, wide_img) in enumerate(rows):
        ax[i][0].imshow(native, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax[i][0].set_title(f"{label}  native input {native.shape[0]}x{native.shape[1]}", fontsize=9)
        ax[i][1].imshow(wide_img, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax[i][1].set_title(f"wide spectrogram {wide_img.shape[0]}x{wide_img.shape[1]}", fontsize=9)
        ax[i][2].imshow(ds, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax[i][2].set_title(f"downsample input {ds.shape[0]}x{ds.shape[1]} (fed to DINO)", fontsize=9)
        for a in ax[i]:
            a.set_xlabel("freq"); a.set_ylabel("time"); a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=100)
    plt.close(fig)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Standalone inference with a fine-tuned DINO-FT TorchScript export on a SigMF recording, through the EXACT
deployment front end. Needs only torch + numpy and the two sibling files frontend.py and rates.py (no dinov3 repo).

Geometry per frame at sample rate R: 512 rows x app_fft_size(R) samples -> hann(rms) FFT -> dB - 10log10(F/1024)
-> per-frequency floor flatten -> robust p20-floor normalisation (span 34 dB, floor at 0.12) -> bilinear resize
F -> 1024 columns -> two 256 x 1024 tiles in [0,1] -> model logits -> sigmoid >= threshold.
A 1024-point-per-row tiling with a fixed dB clip is NOT this front end and makes the model paint whole tiles.

  python run_ts_on_sigmf.py --ts finetuned_dino_m7s_scenes_v2_bf16.ts --capture x.sigmf-meta --out out_dir \
         [--thr 0.6] [--max-frames 50] [--png 4]
Writes out_dir/masks.npz (packed [n_frames,512,1024] masks + frame indices), out_dir/occupancy.csv and PNG overlays.
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frontend as fe
import rates as R


def load_iq(path: Path, dtype: str, offset: int, n: int) -> np.ndarray:
    if dtype.startswith("ci16"):
        raw = np.fromfile(str(path), dtype="<i2", count=2 * n, offset=4 * offset); m = (raw.size // 2) * 2
        return ((raw[0:m:2].astype(np.float32) + 1j * raw[1:m:2].astype(np.float32)) / 32767.0).astype(np.complex64)
    if dtype.startswith("ci8"):
        raw = np.fromfile(str(path), dtype=np.int8, count=2 * n, offset=2 * offset); m = (raw.size // 2) * 2
        return ((raw[0:m:2].astype(np.float32) + 1j * raw[1:m:2].astype(np.float32)) / 127.0).astype(np.complex64)
    if dtype.startswith("cf64"):
        return np.fromfile(str(path), dtype=np.complex128, count=n, offset=16 * offset).astype(np.complex64)
    return np.fromfile(str(path), dtype=np.complex64, count=n, offset=8 * offset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", required=True); ap.add_argument("--capture", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.6); ap.add_argument("--max-frames", type=int, default=10**9)
    ap.add_argument("--png", type=int, default=4, help="overlay PNGs for the first N frames")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = a.device; cfg = fe.FrontEndCfg()
    model = torch.jit.load(a.ts, map_location=dev).eval()
    meta_p = Path(a.capture); meta = json.loads(meta_p.read_text()); g = meta["global"]
    rate = float(g["core:sample_rate"]); dtype = g.get("core:datatype", "cf32_le")
    data_p = meta_p.with_suffix(".sigmf-data")
    bps = 4 if dtype.startswith("ci16") else (2 if dtype.startswith("ci8") else (16 if dtype.startswith("cf64") else 8))
    n_total = data_p.stat().st_size // bps
    F = R.app_fft_size(rate); rows = R.ROWS_WIDE; FS = rows * F
    nfr = min(n_total // FS, a.max_frames)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{meta_p.stem}: rate {rate/1e6:.3f} MSps -> wide FFT {F}, frame {FS} samples ({FS/rate*1e3:.2f} ms), {nfr} frames, device {dev}")
    masks = np.zeros((nfr, rows, cfg.nfft), dtype=np.uint8); occ = []
    for k in range(nfr):
        iq = load_iq(data_p, dtype, k * FS, FS)
        o = fe.compute_front_end(torch.from_numpy(iq).to(dev).unsqueeze(0), rate, cfg, fft_size=F)
        with torch.no_grad():
            logits = model(o.tiles.to(dev).float())
        m = (torch.sigmoid(logits.float()) >= a.thr).to(torch.uint8).cpu().numpy().reshape(-1, cfg.nfft)[:rows]
        masks[k] = m; occ.append(float(m.mean()))
        if k < a.png:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            img = o.resized[0].clamp(0, 1).cpu().numpy(); ext = [-rate / 2e6, rate / 2e6, 0, rows * F / rate * 1e3]
            fig, ax = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
            ax[0].imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext); ax[0].set_title(f"{meta_p.stem} frame {k}: model input (deployment front end)")
            ax[1].imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
            ax[1].imshow(np.ma.masked_where(m == 0, m), aspect="auto", origin="lower", cmap="autumn", alpha=0.55, extent=ext, vmin=0, vmax=1)
            ax[1].set_title(f"mask (thr {a.thr}): occupancy {100*m.mean():.2f}%"); ax[1].set_xlabel("MHz")
            fig.tight_layout(); fig.savefig(out / f"frame_{k:04d}.png", dpi=85); plt.close(fig)
        if k % 20 == 0:
            print(f"  frame {k}: occupancy {100*occ[-1]:.2f}%  (floor {o.floor_db[0]:.1f} dB, robust {o.used_robust[0]})", flush=True)
    np.savez_compressed(out / "masks.npz", packed=np.packbits(masks, axis=-1), rows=rows, cols=cfg.nfft, frame_samples=FS, fft_size=F, rate_hz=rate, threshold=a.thr)
    with open(out / "occupancy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "occupancy"]); w.writerows([(k, v) for k, v in enumerate(occ)])
    print(f"done: occupancy median {100*np.median(occ):.2f}% max {100*np.max(occ):.2f}% -> {out}")


if __name__ == "__main__":
    main()

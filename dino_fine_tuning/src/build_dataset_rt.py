"""Build the DINO-FT 491.52 training dataset by streaming synthetic composites through the deployment
front-end (frontend.py, validated corr 1.0 vs the CUDA operator).

For each composite (gen_491.make_composite): frame the IQ into rows_wide x fft_size blocks, run the
front-end per frame (wide FFT -> flatten -> robust norm -> resize freq->nfft), split into 256-row tiles,
and rasterize the GT on the SAME wide-time-row grid. Keep every signal tile + a capped fraction of
noise-only tiles. Composite-level train/val/test split (no frame leakage). A few held-out composites are
also written to SigMF for the offline-eval benchmark.

Output (out_dir): frames_{split}.npy [N,rows,nfft] uint8, masks_{split}.npy [N,rows,nfft] uint8,
dataset_meta.json. Tiles are uint8 [0,255] (model divides by 255 -> [0,1], matching the operator input).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

import gen_491 as g
import frontend as fe

FFT = 20480          # auto wide-FFT @ 491.52
ROWS = 512           # rows_wide per frame
FSAMP = ROWS * FFT   # 10,485,760 samples/frame
TILE = 256


def frames_of(iq: np.ndarray):
    n = iq.size // FSAMP
    for f in range(n):
        yield f, iq[f * FSAMP:(f + 1) * FSAMP]


def build_split(split, specs, lib, cfg, device, rng, min_sig_px, noise_ratio, sigmf_dir=None):
    frames_u8, masks_u8, csv_rows = [], [], []
    n_sig = n_noise = 0
    for (mode, dur_s, seed) in specs:
        t0 = time.time()
        iq, anns = g.make_composite(mode, dur_s, lib, seed)
        if sigmf_dir is not None:
            g.write_sigmf(Path(sigmf_dir) / f"{mode}_s{seed}", iq, anns)
        ntiles_c = 0
        for fidx, fr in frames_of(iq):
            frame = torch.from_numpy(fr).to(device).unsqueeze(0)
            out = fe.compute_front_end(frame, g.FS, cfg, fft_size=FFT)
            resized = (out.resized[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()  # [512,1024]
            gt = fe.rasterize_gt(anns, fidx * FSAMP, FFT, ROWS, cfg.nfft, g.FS).numpy()          # [512,1024]
            for ti in range(ROWS // TILE):
                r0 = ti * TILE
                ftile = resized[r0:r0 + TILE]
                mtile = gt[r0:r0 + TILE]
                is_sig = int(mtile.sum()) >= min_sig_px
                if is_sig:
                    keep = True
                else:
                    keep = (n_noise < noise_ratio * max(1, n_sig)) and (rng.random() < 0.85)
                if not keep:
                    continue
                pos = len(frames_u8)
                frames_u8.append(ftile); masks_u8.append(mtile); ntiles_c += 1
                n_sig += int(is_sig); n_noise += int(not is_sig)
                csv_rows.append({"frame_id": f"{mode}_s{seed}#f{fidx}t{ti}", "stem": f"{mode}_s{seed}",
                                 "split": split, "attenuation_db": "", "frame_index": fidx,
                                 "abs_start": fidx * FSAMP, "is_signal": int(is_sig),
                                 "n_signal_px": int(mtile.sum()), "mem_pos": pos})
        print(f"  {mode} s{seed} ({dur_s}s): {len(anns)} sig, {ntiles_c} tiles kept "
              f"({time.time()-t0:.1f}s)", flush=True)
    if not frames_u8:
        return None
    F = np.stack(frames_u8).astype(np.uint8)
    M = np.stack(masks_u8).astype(np.uint8)
    return F, M, n_sig, n_noise, csv_rows


def make_specs(n_dense, n_sparse, dur_s, seed0):
    specs = [("dense", dur_s, seed0 + i) for i in range(n_dense)]
    specs += [("sparse", dur_s, seed0 + 1000 + i) for i in range(n_sparse)]
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/home/genesys-dgx1/Documents/Holohub-Signal-Detection/dino_fine_tuning/data/dataset_491")
    ap.add_argument("--dur-s", type=float, default=0.4)
    ap.add_argument("--train-dense", type=int, default=30)
    ap.add_argument("--train-sparse", type=int, default=40)
    ap.add_argument("--val-dense", type=int, default=4)
    ap.add_argument("--val-sparse", type=int, default=6)
    ap.add_argument("--test-dense", type=int, default=3)
    ap.add_argument("--test-sparse", type=int, default=3)
    ap.add_argument("--min-sig-px", type=int, default=8)
    ap.add_argument("--noise-ratio", type=float, default=0.5)
    ap.add_argument("--smoke", action="store_true", help="tiny fast build to validate the pipeline")
    args = ap.parse_args()

    if args.smoke:
        args.train_dense, args.train_sparse = 2, 2
        args.val_dense, args.val_sparse = 1, 1
        args.test_dense, args.test_sparse = 1, 1
        args.dur_s = 0.15
        args.out_dir = args.out_dir + "_smoke"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sigmf_dir = out / "heldout_sigmf"
    lib = g.index_library()
    cfg = fe.FrontEndCfg()  # deployment robust front-end
    print(f"library {len(lib)} waveforms | device {device} | out {out}", flush=True)

    splits = {
        "train": (make_specs(args.train_dense, args.train_sparse, args.dur_s, 1), None),
        "val":   (make_specs(args.val_dense, args.val_sparse, args.dur_s, 5000), None),
        "test":  (make_specs(args.test_dense, args.test_sparse, args.dur_s, 9000), sigmf_dir),
    }
    meta = {"nfft": cfg.nfft, "tile_rows": TILE, "fft_size": FFT, "rows_wide": ROWS,
            "sample_rate": g.FS, "front_end": "robust", "counts": {}, "sig_noise": {}}
    all_csv = []
    for sp, (specs, sdir) in splits.items():
        print(f"[{sp}] {len(specs)} composites", flush=True)
        rng = np.random.default_rng(abs(hash(sp)) % (2**32))
        res = build_split(sp, specs, lib, cfg, device, rng, args.min_sig_px, args.noise_ratio, sdir)
        if res is None:
            print(f"[{sp}] EMPTY"); continue
        F, M, ns, nn, csv_rows = res
        np.save(out / f"frames_{sp}.npy", F)
        np.save(out / f"masks_{sp}.npy", M)
        all_csv.extend(csv_rows)
        meta["counts"][sp] = int(F.shape[0])
        meta["sig_noise"][sp] = {"signal": ns, "noise": nn}
        print(f"[{sp}] wrote {F.shape} ({ns} signal, {nn} noise), mask occ mean "
              f"{100*M.mean():.2f}%", flush=True)
    # frames.csv (all splits; RFSegDataset filters by split and indexes by mem_pos)
    import csv as _csv
    with open(out / "frames.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=list(all_csv[0].keys()))
        w.writeheader(); w.writerows(all_csv)
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", meta["counts"], flush=True)


if __name__ == "__main__":
    main()

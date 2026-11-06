"""Run the detector lineage on OTA capture folders and store masks next to the raw data.

For every <root>/<stem>/<stem>.sigmf-meta: frame the recording at its own rate (512 x app_fft_size rows), run
each model through the deployment front-end, and write into the capture folder
    masks/<model>.npz            packed bit masks [n_frames, 512, 1024] (+ frame_index, frame_samples)
    overlays/<model>_f<k>.png    input + mask for a few frames
    panels/f<k>.png              side-by-side panel (all models + difference of the last two)
    detections.json              per model: predicted occupancy per frame, mean, frames with any detection
Annotation-free captures: only occupancy is reported; captures with annotations also get region det@0.1/0.5.

  PYTHONPATH=src .venv-ml/bin/python src/run_ota_detections.py --root ~/Datasets/ARGUS/OOD_captures/OTA \
      --models M5S=checkpoints/M5S_491_ota/best.pt M6S=checkpoints/M6S_multirate/best.pt M7S=checkpoints/M7S_scenes_v2/best.pt \
      [--glob 'n320_*'] [--max-frames 12] [--thr 0.6]
"""
from __future__ import annotations
import argparse, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

import frontend as fe
import rates as R
from eval_scenes import Model, load_iq, parse_annotations

HERE = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--glob", default="n320_*")
    ap.add_argument("--models", nargs="+", required=True, help="NAME=ckpt_or_ts ...")
    ap.add_argument("--thr", type=float, default=0.6); ap.add_argument("--max-frames", type=int, default=12)
    ap.add_argument("--overlay-frames", type=int, default=2); ap.add_argument("--train-yaml", default="configs/train.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these stems")
    ap.add_argument("--fill-guard", action="store_true", help="apply postfilter.fill_guard to every mask before saving")
    a = ap.parse_args()
    dev = "cuda"; cfg = fe.FrontEndCfg()
    models = {}
    for spec in a.models:
        n, p = spec.split("=", 1); models[n] = Model(str(Path(p) if Path(p).is_absolute() else HERE / p), a.train_yaml, dev)
    names = list(models)
    root = Path(a.root).expanduser()
    folders = sorted(d for d in root.glob(a.glob) if d.is_dir() and list(d.glob("*.sigmf-meta")))
    if a.only:
        folders = [d for d in folders if d.name in a.only]
    print(f"{len(folders)} capture folders, models {names}", flush=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for d in folders:
        meta_p = sorted(d.glob("*.sigmf-meta"))[0]
        meta = json.loads(meta_p.read_text()); g = meta["global"]
        rate = float(g["core:sample_rate"]); dtype = g.get("core:datatype", "cf32_le")
        data_p = meta_p.with_suffix(".sigmf-data")
        bps = 4 if dtype.startswith("ci16") else (16 if dtype.startswith("cf64") else 8)
        n_total = data_p.stat().st_size // bps
        F = R.app_fft_size(rate); rows = R.ROWS_WIDE; FS = rows * F
        nfr_all = n_total // FS
        if nfr_all == 0:
            print(f"  skip {d.name}: shorter than one frame"); continue
        idx = np.unique(np.linspace(0, nfr_all - 1, min(nfr_all, a.max_frames)).astype(int))
        anns = parse_annotations(meta, rate, True, n_total, False)
        (d / "masks").mkdir(exist_ok=True); (d / "overlays").mkdir(exist_ok=True); (d / "panels").mkdir(exist_ok=True)
        masks = {n: np.zeros((idx.size, rows, cfg.nfft), dtype=np.uint8) for n in names}
        imgs = {}
        t0 = time.time()
        for k, fidx in enumerate(idx):
            iq = load_iq(data_p, dtype, int(fidx) * FS, FS)
            fr = torch.from_numpy(iq).to(dev).unsqueeze(0)
            o = fe.compute_front_end(fr, rate, cfg, fft_size=F)
            for n, m in models.items():
                masks[n][k] = (m.prob(o.tiles) >= a.thr).to(torch.uint8).cpu().numpy().reshape(-1, cfg.nfft)[:rows]
                if a.fill_guard:
                    from postfilter import fill_guard as _fg
                    masks[n][k], _ = _fg(masks[n][k], o.resized[0].clamp(0, 1).cpu().numpy())
            if k < a.overlay_frames:
                imgs[k] = o.resized[0].clamp(0, 1).cpu().numpy()
        det = {"capture": meta_p.stem, "rate_hz": rate, "fft_size": F, "frame_samples": FS, "frames_total": int(nfr_all),
               "frame_index": idx.tolist(), "threshold": a.thr, "label": g.get("argus:band_label"), "models": {}}
        for n in names:
            occ = masks[n].reshape(idx.size, -1).mean(axis=1)
            np.savez_compressed(d / "masks" / f"{n}.npz", packed=np.packbits(masks[n], axis=-1), frame_index=idx,
                                rows=rows, cols=cfg.nfft, frame_samples=FS, rate_hz=rate, threshold=a.thr)
            det["models"][n] = {"occupancy_per_frame": [float(x) for x in occ], "occupancy_mean": float(occ.mean()),
                                "frames_with_detection": int((occ > 0).sum())}
        # pairwise agreement (IoU of masks) between models
        det["agreement_iou"] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A, B = masks[names[i]].astype(bool), masks[names[j]].astype(bool)
                det["agreement_iou"][f"{names[i]}|{names[j]}"] = float((A & B).sum() / max((A | B).sum(), 1))
        if anns:
            gt_cov = {n: [] for n in names}
            for k, fidx in enumerate(idx):
                gt = fe.rasterize_gt(anns, int(fidx) * FS, F, rows, cfg.nfft, rate).numpy().astype(bool)
                for n in names:
                    m = masks[n][k].astype(bool)
                    gt_cov[n].append([int((m & gt).sum()), int((m & ~gt).sum()), int((~m & gt).sum())])
            for n in names:
                tp, fp, fn = np.sum(gt_cov[n], axis=0)
                det["models"][n]["pixel"] = {"precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)), "iou": float(tp / max(tp + fp + fn, 1))}
        (d / "detections.json").write_text(json.dumps(det, indent=1))
        # overlays + panels
        ext = [-rate / 2e6, rate / 2e6, 0, rows * F / rate * 1e3]
        for k, img in imgs.items():
            for n in names:
                fig, ax = plt.subplots(figsize=(13, 4))
                ax.imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
                ax.imshow(np.ma.masked_where(masks[n][k] == 0, masks[n][k]), aspect="auto", origin="lower", cmap="autumn", alpha=0.55, extent=ext, vmin=0, vmax=1)
                ax.set_title(f"{meta_p.stem}  frame {idx[k]}  {n} (thr {a.thr}): occupancy {100*masks[n][k].mean():.2f}%", fontsize=10)
                ax.set_xlabel("MHz"); ax.set_ylabel("ms"); fig.tight_layout(); fig.savefig(d / "overlays" / f"{n}_f{idx[k]}.png", dpi=85); plt.close(fig)
            nr = 1 + len(names) + 1
            fig, ax = plt.subplots(nr, 1, figsize=(14, 3.4 * nr), sharex=True)
            ax[0].imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext)
            ax[0].set_title(f"{meta_p.stem} — {rate/1e6:.2f} MSps, frame {idx[k]}: model input ({g.get('argus:expected_waveforms','')})", fontsize=10)
            for r_, n in enumerate(names, start=1):
                ax[r_].imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
                ax[r_].imshow(np.ma.masked_where(masks[n][k] == 0, masks[n][k]), aspect="auto", origin="lower", cmap="autumn", alpha=0.55, extent=ext, vmin=0, vmax=1)
                ax[r_].set_title(f"{n} mask: occupancy {100*masks[n][k].mean():.2f}%", fontsize=10)
            A, B = masks[names[-2]][k] > 0, masks[names[-1]][k] > 0
            diff = np.zeros(img.shape + (3,), dtype=np.float32) + 0.12
            diff[A & B] = (1, 1, 1); diff[A & ~B] = (0.95, 0.25, 0.2); diff[~A & B] = (0.2, 0.9, 0.3)
            ax[-1].imshow(diff, aspect="auto", origin="lower", extent=ext)
            ax[-1].set_title(f"difference: white both, red only {names[-2]}, green only {names[-1]}", fontsize=10)
            for x in ax: x.set_ylabel("ms")
            ax[-1].set_xlabel("MHz (relative to centre)"); fig.tight_layout(); fig.savefig(d / "panels" / f"f{idx[k]}.png", dpi=85); plt.close(fig)
        occs = "  ".join(f"{n} {100*det['models'][n]['occupancy_mean']:.2f}%" for n in names)
        print(f"  {d.name}: {idx.size}/{nfr_all} frames @ {rate/1e6:.2f} MSps | occ {occs} | {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

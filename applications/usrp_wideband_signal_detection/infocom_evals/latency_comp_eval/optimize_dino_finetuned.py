#!/usr/bin/env python3
"""Profile + benchmark quick, output-preserving latency optimizations for dino_finetuned.

Loads the real FinetunedDetector (DINOv3 ViT-B/16 + conv seg head), builds a real 500 MHz
frame's worth of 256x1024 tiles from the 20 dB capture, and (1) profiles where the time goes
(backbone vs seg head vs python/tiling) and (2) times a set of inference variants, checking
each variant's mask still matches the baseline (mean IoU) so we only keep speedups that don't
change detections.

Variants (all no-retrain): baseline (chunks of 16, autocast bf16) | single big batch |
channels_last | cudnn.benchmark | torch.compile | combined.

Run:  python3 optimize_dino_finetuned.py            # 500 MHz frame (40 tiles)
      python3 optimize_dino_finetuned.py --rate 250e6
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from fft_sizing import frame_geometry
from latency_detectors import wire_syspath, frames_to_db
from run_latency_eval import load_config, resample_frame

_THIS = Path(__file__).resolve().parent


def _sync():
    torch.cuda.synchronize()


def _time(fn, warmup=8, reps=40):
    for _ in range(warmup):
        fn(); _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); _sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def build_tiles(cfg, rate_hz, device):
    """Real frame -> (B,1,256,1024) normalized tiles + (rows, nfft) for reassembly."""
    g = frame_geometry(rate_hz, int(cfg.get("num_ffts_per_batch", 512)))
    iq = resample_frame(Path(cfg["capture"]), int(cfg["frame_offset_complex"]),
                        cfg.get("datatype", "cf32_le"), float(cfg["native_sample_rate_hz"]),
                        rate_hz, g.samples_per_frame)
    nfft = 1024
    n = (len(iq) // nfft) * nfft
    rows = n // nfft
    iqt = torch.from_numpy(np.ascontiguousarray(iq[:n].astype(np.complex64))).to(device)
    return iqt, nfft, rows, g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_THIS / "latency_config.yaml")
    ap.add_argument("--rate", type=float, default=500e6)
    args = ap.parse_args()
    assert torch.cuda.is_available(), "needs CUDA"
    device = "cuda"

    cfg = load_config(args.config)
    wire_syspath(cfg.get("dinov3_repo"))
    import rfdata as rf  # noqa
    import model  # noqa
    import finetuned_infer as fi

    spec = cfg["detectors"]["dino_finetuned"]
    ds_meta = __import__("json").loads(Path(spec["dataset_meta"]).read_text())
    train_cfg = yaml.safe_load(Path(spec["train_cfg"]).read_text())
    thr = fi.load_threshold(spec["eval_meta"])
    det = fi.FinetunedDetector(spec["ckpt"], train_cfg, ds_meta, device=device, threshold=thr)
    tile, vmin, vmax = det.tile, det.vmin, det.vmax

    iqt, nfft, rows, g = build_tiles(cfg, args.rate, device)
    db = rf.frames_to_db(iqt[None], nfft, rows)[0]
    img = torch.clamp((db - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    spans = [(r0, min(rows, r0 + tile)) for r0 in range(0, rows, tile)]
    batch = []
    for r0, r1 in spans:
        t = img[r0:r1]
        if t.shape[0] < tile:
            t = F.pad(t, (0, 0, 0, tile - t.shape[0]))
        batch.append(t)
    x = torch.stack(batch)[:, None].contiguous()          # (B,1,256,1024)
    B = x.shape[0]
    print(f"rate {args.rate/1e6:.0f} MHz | fft {g.actual_fft_size} | rows {rows} | tiles B={B} "
          f"| threshold {det.threshold:.3f}")

    m = det.model.eval()

    # ---- reference mask (baseline path) + a masks() helper for IoU checks --------- #
    @torch.inference_mode()
    def run_chunked(mdl, xin, chunk=16, amp=True):
        out = []
        for i in range(0, xin.shape[0], chunk):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = mdl(xin[i:i + chunk])
            out.append((torch.sigmoid(logits.float()) >= det.threshold)[:, 0].to(torch.uint8))
        return torch.cat(out)

    @torch.inference_mode()
    def run_single(mdl, xin, amp=True):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = mdl(xin)
        return (torch.sigmoid(logits.float()) >= det.threshold)[:, 0].to(torch.uint8)

    ref = run_chunked(m, x).float()

    def iou_vs_ref(mask):
        a, b = ref, mask.float()
        inter = (a * b).sum().item(); union = ((a + b) >= 1).float().sum().item()
        return 1.0 if union == 0 else inter / union

    # ---- profile the baseline sub-components (single chunk of 16) ------------------ #
    xc = x[:min(16, B)]
    with torch.inference_mode():
        def backbone_only():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                det.model.backbone.get_intermediate_layers(
                    det.model._prep(xc), n=det.model.feat_layers, reshape=True, norm=True)
        def full_16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                det.model(xc)
        tb = _time(backbone_only, warmup=8, reps=40)[0]
        tf = _time(full_16, warmup=8, reps=40)[0]
    print(f"\nProfile (one {xc.shape[0]}-tile chunk): backbone {tb:.2f} ms | full {tf:.2f} ms "
          f"| seg-head+prep {tf - tb:.2f} ms  ({100*tb/tf:.0f}% backbone)")

    # ---- variants (per-tile latency is what sets the max real-time rate) ----------- #
    torch.backends.cudnn.benchmark = True
    x_cl = x.contiguous(memory_format=torch.channels_last)
    results = []

    def add(name, fn, warmup=8, reps=40):
        med, p90 = _time(fn, warmup=warmup, reps=reps)
        results.append((name, med, p90, iou_vs_ref(fn())))

    add("baseline (chunks of 16, bf16)", lambda: run_chunked(m, x))
    add("single batch (all tiles), eager", lambda: run_single(m, x))
    m_cl = det.model.to(memory_format=torch.channels_last)
    for mode in ("max-autotune", "reduce-overhead"):
        try:
            mc = torch.compile(det.model, mode=mode, fullgraph=False)
            for _ in range(4):
                run_single(mc, x_cl); _sync()
            add(f"single + compile({mode})+cl", lambda mc=mc: run_single(mc, x_cl), warmup=5, reps=30)
        except Exception as exc:
            print(f"  [compile {mode}] failed: {type(exc).__name__}: {exc}")

    base = results[0][1]
    print(f"\n{'variant':36s} {'min ms':>9} {'p90 ms':>8} {'ms/tile':>8} {'speedup':>8} {'IoU':>7}")
    for name, med, p90, iou in results:
        print(f"{name:36s} {med:9.2f} {p90:8.2f} {med/B:8.2f} {base/med:7.2f}x {iou:7.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

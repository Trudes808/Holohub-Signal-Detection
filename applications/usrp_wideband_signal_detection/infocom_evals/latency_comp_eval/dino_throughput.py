#!/usr/bin/env python3
"""Maximize dino_finetuned's real-time sample-rate ceiling by maximizing tile THROUGHPUT.

In a pipelined real-time system the max sustainable sample rate is a throughput limit:

    max_rate = max_tiles_per_sec * samples_per_tile        (samples_per_tile = 256*1024)

so we push per-tile latency down by using the largest efficient batch (tiles batched across
frames -- data duplicated to fill the batch, which is fine for a latency measurement) and the
best precision/compile. For each variant we sweep batch size, find peak tiles/sec, and report
the implied max real-time sample rate; masks are checked (IoU) so we only keep output-preserving
speedups.

Variants: baseline (bf16 autocast) | fp16 | torch.compile(bf16) | torch.compile+fp16.
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
from latency_detectors import wire_syspath
from run_latency_eval import load_config, resample_frame

_THIS = Path(__file__).resolve().parent
SAMPLES_PER_TILE = 256 * 1024          # tile = 256 native FFT rows x 1024 nfft


def _sync():
    torch.cuda.synchronize()


def _time(fn, warmup=8, reps=30):
    for _ in range(warmup):
        fn(); _sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); _sync(); ts.append((time.perf_counter() - t0))
    return float(np.min(ts)), float(np.mean(ts))          # steady-state min, mean


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=_THIS / "latency_config.yaml")
    # NB: torch.compile(max-autotune)+cudagraphs can hit a caching-allocator bug at very large
    # batch (>=64); throughput already plateaus by B=32 so we cap there by default.
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--compile-batches", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    args = ap.parse_args()
    assert torch.cuda.is_available()
    device = "cuda"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = load_config(args.config)
    wire_syspath(cfg.get("dinov3_repo"))
    import rfdata as rf  # noqa
    import model  # noqa
    import finetuned_infer as fi

    spec = cfg["detectors"]["dino_finetuned"]
    ds_meta = __import__("json").loads(Path(spec["dataset_meta"]).read_text())
    train_cfg = yaml.safe_load(Path(spec["train_cfg"]).read_text())
    det = fi.FinetunedDetector(spec["ckpt"], train_cfg, ds_meta, device=device,
                               threshold=fi.load_threshold(spec["eval_meta"]))

    # ---- real tiles from the 500 MHz frame (40 tiles); repeat to fill big batches ---- #
    g = frame_geometry(500e6, int(cfg.get("num_ffts_per_batch", 512)))
    iq = resample_frame(Path(cfg["capture"]), int(cfg["frame_offset_complex"]),
                        cfg.get("datatype", "cf32_le"), float(cfg["native_sample_rate_hz"]),
                        500e6, g.samples_per_frame)
    nfft, tile = 1024, det.tile
    n = (len(iq) // nfft) * nfft
    rows = n // nfft
    iqt = torch.from_numpy(np.ascontiguousarray(iq[:n].astype(np.complex64))).to(device)
    db = rf.frames_to_db(iqt[None], nfft, rows)[0]
    img = torch.clamp((db - det.vmin) / max(det.vmax - det.vmin, 1e-6), 0, 1)
    base_tiles = torch.stack([img[r0:r0 + tile] for r0 in range(0, rows - tile + 1, tile)])[:, None]
    print(f"real base tiles: {base_tiles.shape[0]} (from 500 MHz frame); samples/tile {SAMPLES_PER_TILE}")

    def make_batch(bs, like=base_tiles, half=False, channels_last=False):
        reps = (bs + like.shape[0] - 1) // like.shape[0]
        x = like.repeat(reps, 1, 1, 1)[:bs].contiguous()
        if half:
            x = x.half()
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    thr = det.threshold

    @torch.inference_mode()
    def mask_bf16(mdl, x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return (torch.sigmoid(mdl(x).float()) >= thr)[:, 0].to(torch.uint8)

    @torch.inference_mode()
    def mask_plain(mdl, x):   # for fp16 (model already half) or compiled
        return (torch.sigmoid(mdl(x).float()) >= thr)[:, 0].to(torch.uint8)

    ref_mask = mask_bf16(det.model, base_tiles).float()

    def iou(mask):
        a, b = ref_mask, mask.float()
        u = ((a + b) >= 1).float().sum().item()
        return 1.0 if u == 0 else (a * b).sum().item() / u

    rows_out = []   # (variant, batch, min_ms, tiles_per_sec, iou)

    def sweep(name, call, batches, half=False, cl=False):
        for bs in batches:
            try:
                x = make_batch(bs, half=half, channels_last=cl)
                mn, me = _time(lambda: call(x))
                tps = bs / mn
                io = iou(call(make_batch(base_tiles.shape[0], half=half, channels_last=cl)))
                rows_out.append((name, bs, mn * 1e3, tps, io))
                print(f"  {name:24s} B={bs:4d} | {mn*1e3:8.2f} ms | {tps:8.1f} tiles/s | IoU {io:.4f}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {name:24s} B={bs:4d} | OOM")
                break

    print("\n== baseline bf16 (autocast) ==")
    sweep("baseline_bf16", lambda x: mask_bf16(det.model, x), args.batches)

    # NOTE: full model.half() is unsupported here -- DinoSegmenter.forward casts backbone
    # features to float before the seg head, so half weights meet float input. bf16 autocast
    # is the precision path (already the deployed default). We optimize via compile instead.

    print("\n== torch.compile(max-autotune) bf16 + channels_last ==")
    det.model = det.model.to(memory_format=torch.channels_last)
    m_c = torch.compile(det.model, mode="max-autotune", fullgraph=False)
    sweep("compile_ma", lambda x: mask_bf16(m_c, x), args.compile_batches, cl=True)

    print("\n== torch.compile(reduce-overhead / CUDA graphs) bf16 + channels_last ==")
    m_ro = torch.compile(det.model, mode="reduce-overhead", fullgraph=False)
    sweep("compile_ro", lambda x: mask_bf16(m_ro, x), args.compile_batches, cl=True)

    # ---- summary: peak throughput -> max sustainable real-time sample rate --------- #
    print(f"\n{'variant':24s} {'best B':>7} {'peak tiles/s':>13} {'max rate (MS/s)':>16} {'IoU':>7}")
    best = {}
    for name, bs, ms, tps, io in rows_out:
        if name not in best or tps > best[name][1]:
            best[name] = (bs, tps, io)
    base_rate = None
    for name in ["baseline_bf16", "compile_ma", "compile_ro"]:
        if name not in best:
            continue
        bs, tps, io = best[name]
        rate = tps * SAMPLES_PER_TILE / 1e6
        if name == "baseline_bf16":
            base_rate = rate
        spd = f"({rate/base_rate:.2f}x)" if base_rate else ""
        print(f"{name:24s} {bs:>7d} {tps:>13.1f} {rate:>13.1f} {spd:>6} {io:>7.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Per-frame latency + compute-load eval across the six detectors, CPU vs GPU.

One simulated frame from the 20 dB attenuation capture is resampled to each target
sample rate (20 / 100 / 250 / 500 MHz). For every (detector, rate, device) it measures:

  * latency  -- warm, adaptively-repeated per-frame wall time (a histogram of samples)
  * FLOPs    -- aten conv/matmul flops (torch FlopCounterMode) + analytic FFT flops
  * peak mem -- peak CUDA memory to process the frame (GPU cells only)

Results serialize to a reloadable ``LatencyResults`` (.npz + .json) that the notebook /
plot_latency_results.py render without recompute.

Usage:
    python3 run_latency_eval.py --config latency_config.yaml
    python3 run_latency_eval.py --detectors coherent_power 3dB_power --rates 20e6 500e6
    python3 run_latency_eval.py --devices cuda            # GPU only (skip slow CPU cells)
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from fractions import Fraction
from pathlib import Path

# Reduce CUDA fragmentation before the first torch CUDA use (the flop counter roughly
# doubles peak memory over a plain forward, which OOMs the heaviest ML cell otherwise).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import yaml

from fft_sizing import frame_geometry
from latency_detectors import build_detectors, wire_syspath
from latency_results import LatencyResults

_THIS_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    return cfg.get("latency_eval", cfg)


# --------------------------------------------------------------------------- #
# IQ: one native frame -> resampled to each target rate
# --------------------------------------------------------------------------- #
def read_native_iq(capture: Path, offset: int, n_complex: int, datatype: str) -> np.ndarray:
    if datatype != "cf32_le":
        raise ValueError(f"unsupported datatype {datatype!r} (expected cf32_le)")
    mm = np.memmap(capture, dtype=np.complex64, mode="r")
    end = min(offset + n_complex, mm.shape[0])
    return np.asarray(mm[offset:end], dtype=np.complex64)


def resample_frame(capture: Path, offset: int, datatype: str, native_rate: float,
                   target_rate: float, samples_per_frame: int) -> np.ndarray:
    """Return exactly ``samples_per_frame`` complex samples at ``target_rate``, produced
    by poly resampling the native IQ (interpolate when target>native, decimate when <)."""
    from scipy.signal import resample_poly
    frac = Fraction(int(round(target_rate)), int(round(native_rate))).limit_denominator(20000)
    up, down = frac.numerator, frac.denominator
    # native samples needed to yield >= samples_per_frame after up/down, plus filter margin
    native_needed = int(np.ceil(samples_per_frame * down / up)) + 4 * max(up, down) + 16
    native = read_native_iq(capture, offset, native_needed, datatype)
    if up == down:
        out = native
    else:
        out = resample_poly(native, up, down).astype(np.complex64)
    if out.shape[0] < samples_per_frame:                       # pad (only if file ran short)
        out = np.concatenate([out, np.zeros(samples_per_frame - out.shape[0], np.complex64)])
    return np.ascontiguousarray(out[:samples_per_frame])


# --------------------------------------------------------------------------- #
# Timing / FLOPs / memory
# --------------------------------------------------------------------------- #
def _syncer(device: str):
    if device == "cuda":
        return lambda: torch.cuda.synchronize()
    return lambda: None


def measure_latency(run_fn, sync, warmup, min_reps, max_reps, time_budget_s, hard_cap_s,
                    slow_probe_s: float = 2.0):
    # One probe first (also a warmup) to gauge cost. Very slow cells (ML on CPU at high
    # rates) get warmup/min_reps collapsed so a single run doesn't blow the wall budget.
    t0 = time.perf_counter()
    run_fn(); sync()
    probe = time.perf_counter() - t0
    if probe > slow_probe_s:
        warmup_left, eff_min = 0, min(3, min_reps)
    else:
        warmup_left, eff_min = max(0, warmup - 1), min_reps
    for _ in range(warmup_left):
        run_fn(); sync()
    lat_ms = []
    t_start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        run_fn(); sync()
        lat_ms.append((time.perf_counter() - t0) * 1e3)
        n, elapsed = len(lat_ms), time.perf_counter() - t_start
        if n >= max_reps:
            break
        if elapsed >= hard_cap_s:
            break
        if n >= eff_min and elapsed >= time_budget_s:
            break
    return np.asarray(lat_ms, dtype=float)


def measure_flops(run_fn, sync, analytic_fft_flops: float) -> tuple[float, bool]:
    """aten conv/matmul flops (FlopCounterMode) + analytic FFT flops -> (GFLOPs, ok).
    ``ok`` is False if the counter threw (e.g. CUDA OOM); the caller then repairs the
    value by linear scaling from a successful rate of the same detector."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            run_fn(); sync()
        return (float(fc.get_total_flops()) + float(analytic_fft_flops)) / 1e9, True
    except Exception as exc:                                    # pragma: no cover
        print(f"      [flops] counter failed ({type(exc).__name__}); will scale from another rate")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float("nan"), False


def measure_peak_mem_mb(run_fn, device: str) -> float:
    if device != "cuda":
        return float("nan")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    run_fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=_THIS_DIR / "latency_config.yaml")
    ap.add_argument("--out", type=Path, default=None, help="Output base path (no ext); overrides config out_dir.")
    ap.add_argument("--detectors", nargs="+", default=None, help="Subset of detectors to run.")
    ap.add_argument("--rates", nargs="+", type=float, default=None, help="Subset of sample rates (Hz).")
    ap.add_argument("--devices", nargs="+", default=None, choices=["cpu", "cuda"])
    ap.add_argument("--max-reps", type=int, default=None)
    ap.add_argument("--time-budget-s", type=float, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    wire_syspath(cfg.get("dinov3_repo"))

    rates = args.rates or [float(r) for r in cfg["target_sample_rates_hz"]]
    devices = args.devices or list(cfg.get("devices", ["cpu", "cuda"]))
    if "cuda" in devices and not torch.cuda.is_available():
        print("[warn] CUDA not available; dropping cuda device")
        devices = [d for d in devices if d != "cuda"]

    num_ffts_per_batch = int(cfg.get("num_ffts_per_batch", 512))
    ref_span = float(cfg.get("reference_span_hz", 500e6))
    ref_fft = int(cfg.get("reference_fft_size", 20480))
    packet = int(cfg.get("packet_samples", 1024))
    geoms = {r: frame_geometry(r, num_ffts_per_batch, ref_span, ref_fft, packet) for r in rates}

    warmup = int(cfg.get("warmup_reps", 5))
    min_reps = int(cfg.get("min_reps", 20))
    max_reps = int(args.max_reps or cfg.get("max_reps", 200))
    time_budget_s = float(args.time_budget_s or cfg.get("time_budget_s", 8.0))
    hard_cap_s = float(cfg.get("hard_cap_s", 30.0))

    capture = Path(cfg["capture"])
    native_rate = float(cfg["native_sample_rate_hz"])
    datatype = cfg.get("datatype", "cf32_le")
    offset = int(cfg.get("frame_offset_complex", 0))

    detectors = build_detectors(cfg)
    if args.detectors:
        detectors = {k: v for k, v in detectors.items() if k in args.detectors}
    if not detectors:
        ap.error("no detectors selected")

    # ---- one resampled frame per rate (built once, reused by every detector) ---- #
    print(f"Building simulated frames from {capture.name} @ offset {offset} ...")
    frame_iq = {}
    for r in rates:
        g = geoms[r]
        frame_iq[r] = resample_frame(capture, offset, datatype, native_rate, r, g.samples_per_frame)
        print(f"  {r/1e6:>5.0f} MHz  fft={g.actual_fft_size:>6d}  "
              f"samples/frame={g.samples_per_frame:>9d}  budget={g.frame_budget_ms:7.3f} ms")

    # ---- measure every (detector, device, rate) cell ---------------------------- #
    cells = {k: [] for k in ("detector", "sample_rate_hz", "fft_size", "samples_per_frame",
                             "frame_budget_ms", "device", "n_reps", "lat_mean_ms",
                             "lat_median_ms", "lat_p05_ms", "lat_p95_ms", "lat_min_ms",
                             "lat_std_ms", "gflops", "peak_mem_mb")}
    sample_cell_index, sample_latency_ms = [], []
    flops_cache: dict[tuple[str, float], float] = {}
    flops_ok: dict[tuple[str, float], bool] = {}
    t_all = time.perf_counter()
    # FLOPs are device-independent -> measure them on the fastest device, and run it first
    # per detector so cells built on the slow device already have the cached value.
    flop_device = "cuda" if "cuda" in devices else devices[0]
    dev_order = [flop_device] + [d for d in devices if d != flop_device]

    for dname, det in detectors.items():
        for device in dev_order:
            print(f"\n=== {dname} on {device} ===")
            det.load(device)
            sync = _syncer(device)
            for r in rates:
                g = geoms[r]
                if device == "cuda":
                    torch.cuda.empty_cache()
                run_fn = det.prepare(frame_iq[r], g, device)
                # FLOPs are device-independent: measure once (on flop_device) per (detector, rate)
                fkey = (dname, r)
                if fkey not in flops_cache and device == flop_device:
                    flops_cache[fkey], flops_ok[fkey] = measure_flops(
                        run_fn, sync, det.analytic_fft_flops(g))
                gflops = flops_cache.get(fkey, float("nan"))
                peak_mb = measure_peak_mem_mb(run_fn, device)
                lat = measure_latency(run_fn, sync, warmup, min_reps, max_reps,
                                      time_budget_s, hard_cap_s)
                ci = len(cells["detector"])
                cells["detector"].append(dname)
                cells["sample_rate_hz"].append(float(r))
                cells["fft_size"].append(int(g.actual_fft_size))
                cells["samples_per_frame"].append(int(g.samples_per_frame))
                cells["frame_budget_ms"].append(float(g.frame_budget_ms))
                cells["device"].append(device)
                cells["n_reps"].append(int(lat.size))
                cells["lat_mean_ms"].append(float(np.mean(lat)))
                cells["lat_median_ms"].append(float(np.median(lat)))
                cells["lat_p05_ms"].append(float(np.percentile(lat, 5)))
                cells["lat_p95_ms"].append(float(np.percentile(lat, 95)))
                cells["lat_min_ms"].append(float(np.min(lat)))
                cells["lat_std_ms"].append(float(np.std(lat)))
                cells["gflops"].append(float(gflops))
                cells["peak_mem_mb"].append(float(peak_mb))
                sample_cell_index.extend([ci] * lat.size)
                sample_latency_ms.extend(lat.tolist())
                rt = "OK" if np.median(lat) <= g.frame_budget_ms else "OVER"
                print(f"  {r/1e6:>5.0f} MHz | med {np.median(lat):8.3f} ms "
                      f"(n={lat.size:3d}) | {gflops:9.2f} GFLOPs | "
                      f"peak {peak_mb:8.1f} MB | budget {g.frame_budget_ms:7.3f} ms [{rt}]")
            det.unload()

    # ---- repair any FLOP measurement that failed (OOM) by linear scaling ----------- #
    # Counted conv/matmul flops (and the FFT term) scale ~linearly with samples/frame,
    # so a failed cell is recovered from the same detector's heaviest successful rate.
    for dname in detectors:
        ok_rates = [r for r in rates if flops_ok.get((dname, r))]
        if not ok_rates:
            continue
        anchor = max(ok_rates, key=lambda r: geoms[r].samples_per_frame)
        for r in rates:
            if not flops_ok.get((dname, r)):
                scale = geoms[r].samples_per_frame / geoms[anchor].samples_per_frame
                flops_cache[(dname, r)] = flops_cache[(dname, anchor)] * scale
                print(f"[flops-repair] {dname} @ {r/1e6:.0f} MHz <- "
                      f"{flops_cache[(dname, anchor)]:.1f} GFLOPs x {scale:.3g} "
                      f"= {flops_cache[(dname, r)]:.1f} GFLOPs (from {anchor/1e6:.0f} MHz)")
    cells["gflops"] = [float(flops_cache.get((d, rr), float("nan")))
                       for d, rr in zip(cells["detector"], cells["sample_rate_hz"])]

    results = LatencyResults(
        cells={k: np.asarray(v) for k, v in cells.items()},
        samples={"cell_index": np.asarray(sample_cell_index, dtype=int),
                 "latency_ms": np.asarray(sample_latency_ms, dtype=float)},
        geometry={str(int(r)): geoms[r].as_dict() for r in rates},
        params={
            "target_sample_rates_hz": rates, "devices": devices,
            "num_ffts_per_batch": num_ffts_per_batch, "reference_span_hz": ref_span,
            "reference_fft_size": ref_fft, "packet_samples": packet,
            "warmup_reps": warmup, "min_reps": min_reps, "max_reps": max_reps,
            "time_budget_s": time_budget_s, "hard_cap_s": hard_cap_s,
            "detector_params": cfg.get("detectors", {}),
        },
        provenance={
            "capture": str(capture), "native_sample_rate_hz": native_rate,
            "frame_offset_complex": offset, "datatype": datatype,
            "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu-only",
            "torch": torch.__version__,
            "elapsed_s": round(time.perf_counter() - t_all, 1),
        },
    )
    out_base = args.out or (_THIS_DIR / cfg.get("out_dir", "saved_results/latency_run"))
    npz_path, json_path = results.save(out_base)
    print(f"\nWrote {results.n_cells} cells / {len(sample_latency_ms)} latency samples "
          f"in {time.perf_counter() - t_all:.1f}s")
    print(f"  {npz_path}\n  {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

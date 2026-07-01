#!/usr/bin/env python3
"""Alignment self-check: detect the frame<->mask desync (the ring-buffer aliasing bug).

Measures the SYSTEMATIC frame offset between detector masks and the ground truth by scanning
candidate frame offsets k and taking the MEDIAN column-profile correlation across many frames.
A true systematic offset (e.g. the +8 = ring_size aliasing bug) shows up as a sharp peak at
k != 0; periodic spurious matches average out. A healthy run peaks at k = 0.

Usage:
    python3 check_mask_alignment.py --run-dir /tmp/usrp_spectrograms/offline_eval/cuda_dino/attenuation_dB_25
    python3 check_mask_alignment.py --batch-root /tmp/usrp_spectrograms/offline_eval --file-stem attenuation_dB_25

Exit code 0 if aligned (best k == 0), 1 if a systematic offset is detected. Intended as a
post-run gate (wire into run_batch_offline_eval.py) and for manual verification after a rebuild.
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np

import eval_viz as v
import mask_eval_metrics as mem


def _colprof(mask: np.ndarray) -> np.ndarray:
    p = (mask != 0).mean(axis=0)
    s = p.std()
    return (p - p.mean()) / (s + 1e-9)


def measure_offset(run_dir: Path, max_k: int = 12, stride: int = 6):
    """Return (best_k, best_median_corr, curve) for one detector run directory."""
    run_dir = Path(run_dir)
    gt_by_frame: dict[int, str] = {}
    for f in glob.glob(str(run_dir / "gt_masks") + "/*"):
        m = re.search(r"_f(\d+)_", f)
        if m:
            gt_by_frame[int(m.group(1))] = f
    frames = sorted(gt_by_frame)
    if len(frames) < 3 * max_k:
        # too few frames (e.g. single-frame run) to measure an offset meaningfully
        return 0, float("nan"), {}
    lo, hi = frames[0] + max_k, frames[-1] - max_k
    sample = [n for n in range(lo, hi + 1, stride) if n in gt_by_frame]
    gtcp = {n: _colprof(mem.load_mask_any(Path(gt_by_frame[n]))) for n in frames}

    def maskcp(n):
        g = glob.glob(str(run_dir / "mask_arrays") + f"/mask_ch0_f{n}_*")
        return _colprof(mem.load_mask_any(Path(g[0]))) if g else None

    curve = {}
    for k in range(-max_k, max_k + 1):
        cs = []
        for n in sample:
            mc = maskcp(n)
            if mc is None or (n + k) not in gtcp:
                continue
            cs.append(float(np.mean(mc * gtcp[n + k])))
        if cs:
            curve[k] = float(np.median(cs))
    if not curve:
        return 0, float("nan"), {}
    best_k = max(curve, key=curve.get)
    return best_k, curve[best_k], curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", help="A single detector run dir (.../<detector>/<file_stem>/).")
    src.add_argument("--batch-root", help="Batch root; checks every detector for --file-stem.")
    ap.add_argument("--file-stem", help="Capture stem (with --batch-root).")
    ap.add_argument("--max-k", type=int, default=12)
    args = ap.parse_args()

    if args.run_dir:
        runs = [Path(args.run_dir)]
    else:
        layout = v.resolve_layout(Path(args.batch_root), args.file_stem)
        stem = layout["file_stem"]
        runs = [layout["batch_root"] / d / stem for d in layout["detectors"]]

    ok = True
    for run in runs:
        best_k, best_corr, curve = measure_offset(run, max_k=args.max_k)
        zero = curve.get(0, float("nan"))
        status = "PASS" if best_k == 0 else "FAIL"
        if best_k != 0:
            ok = False
        print(f"[{status}] {run}")
        print(f"        best offset k={best_k:+d} (median corr {best_corr:.3f}); k=0 corr {zero:.3f}")
        if best_k != 0:
            print(f"        -> masks lead GT by {best_k} frames (ring aliasing not fixed / stale binary). "
                  f"Rebuild + re-run.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

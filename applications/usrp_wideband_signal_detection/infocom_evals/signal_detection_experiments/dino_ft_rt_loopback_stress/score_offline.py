#!/usr/bin/env python3
"""Score OFFLINE runs (run_offline_refs.sh output) against SigMF ground truth, side by side.

Used to judge front-end variants (e.g. the short-FFT time-pooled path) on accuracy before any live
run: each leg is the same capture through the offline binary with a different detector config.
  python3 score_offline.py --root /tmp/usrp_spectrograms/rt_stress/offline_v3 --stem capture_snr_p24dB \
      --meta <capture>.sigmf-meta  dinoft_rt_m5s dinoft_rt_m5s_sf2048 ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import mask_eval_metrics as mem  # noqa: E402
from score_leg import summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--frames", type=int, default=None, help="score only frames <= N (e.g. the pcap's frames)")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    ap.add_argument("legs", nargs="+")
    a = ap.parse_args()
    cfg = mem.EvalConfig(region_coverage_threshold=0.5)
    res = {}
    print("| leg | frames | det@0.1 | det@0.5 | px prec | px recall | px F1 | px IoU | FP area | NBFM | <2 MHz | 2-10 MHz | >=60 MHz | ZC | Bluetooth | 16QAM |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for leg in a.legs:
        run = Path(a.root) / leg / a.stem
        fr, rg = mem.evaluate_run(run, detector=leg, file_stem=a.stem, sigmf_meta_path=Path(a.meta),
                                  config=cfg, frame_limit=a.frames)
        s = summarize(fr, rg)
        res[leg] = s
        pc, pb = s["per_class"], s["per_bandwidth"]
        g = lambda d, k: (d.get(k) or {}).get("det@0.5")
        f = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"| {leg} | {s['frames_with_mask']} | {f(s['region_det_rate@0.1'])} | {f(s['region_det_rate@0.5'])} | "
              f"{f(s['pixel_precision'])} | {f(s['pixel_recall'])} | {f(s['pixel_f1'])} | {f(s['pixel_iou'])} | "
              f"{s['fp_area_fraction']:.4f} | {f(g(pc,'Narrowband_FM'))} | {f(g(pb,'<2MHz'))} | {f(g(pb,'2-10MHz'))} | "
              f"{f(g(pb,'>=60MHz'))} | {f(g(pc,'ZC'))} | {f(g(pc,'Bluetooth'))} | {f(g(pc,'16QAM'))} |")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

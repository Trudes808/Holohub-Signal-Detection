#!/usr/bin/env python3
"""Discover detector masks under one or more roots laid out <root>/<detector>/<stem>/mask_arrays/
(the shared batch-eval layout; also handles a batch_runs parent <root>/<run_id>/<detector>/<stem>/).

Reports, per (detector, capture): #masks, file format(s), GT/manifest presence, and completeness
(from offline_eval_summary.json when present). Use it to see what's already produced before running
create_all_masks.sh, so finished detectors (e.g. coherent_power/cuda_dino) aren't recomputed.

Usage:
  python3 find_masks.py [ROOT ...]           # default: /tmp/.../all_detectors + ./batch_runs
"""
from __future__ import annotations
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve()
DEFAULT_ROOTS = [
    "/tmp/usrp_spectrograms/all_detectors",
    str(_HERE.parents[1] / "signal_detection_experiments" / "batch_runs"),
]
EXPECTED = ["coherent_power", "cuda_dino", "3dB_power", "blob_detection",
            "dino_finetuned", "dino_finetuned_m1", "yolo", "yolo26s"]


def scan_run(d: Path) -> dict:
    ma = d / "mask_arrays"
    npy = len(list(ma.glob("*.npy"))) if ma.exists() else 0
    npz = len(list(ma.glob("*.packed.npz"))) if ma.exists() else 0
    fmt = "+".join(f for f, n in (("npy", npy), ("packed.npz", npz)) if n) or "-"
    complete = None
    summ = d / "offline_eval_summary.json"
    if summ.exists():
        try:
            complete = json.load(open(summ)).get("manifest_complete")
        except Exception:
            pass
    return dict(masks=npy + npz, fmt=fmt, gt=(d / "gt_masks").exists(),
                man=(d / "frame_manifest.csv").exists(), complete=complete)


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [Path(r) for r in DEFAULT_ROOTS]
    for root in roots:
        print(f"\n=== {root} ===")
        if not root.exists():
            print("  (missing)"); continue
        run_dirs = sorted({Path(p).parent for p in glob.glob(str(root / "**" / "mask_arrays"), recursive=True)})
        if not run_dirs:
            print("  (no mask_arrays/ found)"); continue
        by_det: dict[str, list] = defaultdict(list)
        for rd in run_dirs:
            by_det[rd.parent.name].append((rd.name, scan_run(rd)))
        for det in sorted(by_det):
            runs = sorted(by_det[det])
            print(f"  [{det}]  {len(runs)} capture(s)")
            for stem, s in runs:
                flag = {True: "complete", False: "PARTIAL", None: "no-summary"}[s["complete"]]
                print(f"    {stem:26s} masks={s['masks']:4d} fmt={s['fmt']:10s} "
                      f"gt={int(s['gt'])} man={int(s['man'])} {flag}")
        present = set(by_det)
        missing = [d for d in EXPECTED if d not in present]
        if missing:
            print(f"  -> not yet produced here: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

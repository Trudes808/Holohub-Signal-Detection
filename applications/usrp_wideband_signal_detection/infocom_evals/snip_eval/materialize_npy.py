#!/usr/bin/env python3
"""Materialize .packed.npz masks -> .npy so the C++ mask_replay_detector (which reads .npy) can load
them. Idempotent: skips frames that already have a .npy. Reverse of repack_offline_masks.py.

(Long-term the operator could read .packed.npz directly to avoid this 8x-larger transient copy; until
then, run this over a batch_root before the snip stage.)

Usage:
  python3 materialize_npy.py <ROOT>            # ROOT/<detector>/<stem>/mask_arrays/*.packed.npz -> *.npy
  python3 materialize_npy.py <ROOT>/<det>/<stem>   # a single run dir
  KEEP_NPZ=0 python3 materialize_npy.py <ROOT>  # also delete the .packed.npz (default: keep)
"""
from __future__ import annotations
import glob, os, sys
from pathlib import Path
import numpy as np

def materialize(run_dir: Path, keep_npz: bool) -> int:
    n = 0
    for sub in ("mask_arrays", "gt_masks"):
        d = run_dir / sub
        if not d.exists():
            continue
        for npz in d.glob("*.packed.npz"):
            npy = npz.with_name(npz.name[: -len(".packed.npz")] + ".npy")
            if npy.exists():
                continue
            z = np.load(npz)
            arr = np.unpackbits(z["packed"])[: int(z["rows"]) * int(z["cols"])].reshape(int(z["rows"]), int(z["cols"])).astype(np.uint8)
            np.save(npy, arr)
            if not keep_npz:
                npz.unlink()
            n += 1
    return n

def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/usrp_spectrograms/all_detectors")
    keep = os.environ.get("KEEP_NPZ", "1") != "0"
    run_dirs = sorted({Path(p).parent for p in glob.glob(str(root / "**" / "mask_arrays"), recursive=True)})
    if not run_dirs:
        print(f"no mask_arrays/ under {root}"); return 0
    tot = 0
    for rd in run_dirs:
        c = materialize(rd, keep)
        tot += c
        print(f"{'materialized' if c else 'up-to-date  '} {c:4d}  {rd}")
    print(f"DONE: {tot} .npy written under {root}  (keep_npz={keep})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

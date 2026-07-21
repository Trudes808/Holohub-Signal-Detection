#!/usr/bin/env python3
"""Repack offline-eval masks (.npy -> packbits .packed.npz) so the data-saving notebook + snipper
measurement (which read *.packed.npz) can consume run_all_detectors.sh / run_cuda_dino_offline_file.py
output. Identical format to run_batch_offline_eval.repack_masks (keys: packed/rows/cols; ~8x smaller);
raw .npy deleted after packing. Idempotent -- run dirs with no raw .npy are skipped.

Offline outputs under /tmp/usrp_spectrograms are root-owned, so run under sudo:
  sudo python3 repack_offline_masks.py [ROOT]           # default ROOT=/tmp/usrp_spectrograms/all_detectors
  sudo python3 repack_offline_masks.py <ROOT>/<det>/<stem>   # a single run dir also works
"""
from __future__ import annotations
import glob
import sys
from pathlib import Path

import numpy as np


def repack_run_dir(run_dir: Path) -> int:
    """Pack every mask_arrays/ + gt_masks/ .npy in one run dir into .packed.npz; return #packed."""
    n = 0
    for subdir in ("mask_arrays", "gt_masks"):
        d = run_dir / subdir
        if not d.exists():
            continue
        for npy in d.glob("*.npy"):
            arr = np.load(npy) != 0
            np.savez_compressed(
                npy.with_suffix(".packed.npz"),
                packed=np.packbits(arr.reshape(-1)), rows=arr.shape[0], cols=arr.shape[1],
            )
            npy.unlink()
            n += 1
    return n


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/usrp_spectrograms/all_detectors")
    if not root.exists():
        print(f"root not found: {root}", file=sys.stderr)
        return 1
    # a "run dir" is any dir containing a mask_arrays/ subdir (also matches a single run dir passed directly)
    run_dirs = sorted({Path(p).parent for p in glob.glob(str(root / "**" / "mask_arrays"), recursive=True)})
    if not run_dirs:
        print(f"no mask_arrays/ found under {root}")
        return 0
    tot_dirs = tot_files = 0
    for rd in run_dirs:
        n = repack_run_dir(rd)
        if n:
            tot_dirs += 1
            tot_files += n
            print(f"repacked {n:4d} masks  {rd}")
        else:
            print(f"skip (already packed) {rd}")
    print(f"DONE: repacked {tot_files} masks across {tot_dirs} run dir(s) under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

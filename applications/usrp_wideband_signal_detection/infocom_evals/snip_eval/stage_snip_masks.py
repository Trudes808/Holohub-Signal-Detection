#!/usr/bin/env python3
"""Stage every detector's masks under the mounted repo so run_snip_all.sh can snip them all in one
BATCH_ROOT. The container only mounts the repo, /tmp/usrp_spectrograms, and ~/captures, so masks that
live elsewhere (baselines in /tmp/ds_batch) can't be reached by --mask-dir.

For each (detector, capture) we build  snip_run/<detector>/<stem>/  with:
  * mask_arrays/mask_ch0_f{N}_{H}x{W}.packed.npz  -- SYMLINK when the source is already .packed.npz
    (coherent_power, cuda_dino, yolo*, finetuned*, ground_truth); PACKED here from .npy when the
    source is raw .npy (3dB_power, blob_detection). run_snip_all.materialize_npy then unpacks them to
    real .npy at run time (dereferencing symlinks host-side), which the container reads.
  * frame_manifest.csv (copied from the source run dir; ground_truth borrows cuda_dino's).

Detectors + attenuations are auto-discovered. Ground-truth masks come from cuda_dino's gt_masks.
Idempotent; re-run to refresh. Writes into the repo (gitignored). No sudo needed.
"""
from __future__ import annotations
import csv, re, shutil, sys
from pathlib import Path
import numpy as np

REPO = Path("/home/bqn82/Holohub-Signal-Detection")
OUT = REPO / "applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/snip_run"
ADET = Path("/tmp/usrp_spectrograms/all_detectors")
DSB = Path("/tmp/ds_batch")
SWA = REPO / "notebooks/yolo_evals/sweeps/sweep_all"
SWD = REPO / "notebooks/dino_fine_tuning_evals/sweeps/sweep_detectors"

# detector -> source run-root (each holds <stem>/mask_arrays + frame_manifest.csv)
SRC = {
    "coherent_power": ADET / "coherent_power",
    "cuda_dino": ADET / "cuda_dino",
    "3dB_power": DSB / "3dB_power",
    "blob_detection": DSB / "blob_detection",
    "yolo26s": SWA / "yolo26s",
    "yolo26m": SWA / "yolo26m",
    "finetuned_dino": SWD / "finetuned_dino",
    "finetuned_dino_m2": SWD / "finetuned_dino_m2",
    # ground_truth: masks from cuda_dino's gt_masks/, manifest from cuda_dino
}


def pack_npy_to_npz(npy: Path, dst: Path):
    arr = np.load(npy) != 0
    np.savez_compressed(dst, packed=np.packbits(arr.reshape(-1)), rows=arr.shape[0], cols=arr.shape[1])


def stage_masks(src_files, dst_dir: Path, strip_prefix: str = ""):
    """src_files: iterable of source mask paths. Symlink .packed.npz, pack .npy. Returns count."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src_files:
        name = f.name
        if strip_prefix and name.startswith(strip_prefix):
            name = name[len(strip_prefix):]
        if f.suffix == ".npz":  # .packed.npz -> symlink (materialize dereferences host-side)
            link = dst_dir / name
            if not link.exists():
                link.symlink_to(f.resolve())
            n += 1
        elif f.suffix == ".npy":  # raw .npy -> pack into a real .packed.npz in the repo
            base = re.sub(r"\.npy$", ".packed.npz", name)
            out = dst_dir / base
            if not out.exists():
                pack_npy_to_npz(f, out)
            n += 1
    return n


def main():
    dets = sys.argv[1:] or (list(SRC.keys()) + ["ground_truth"])
    total = 0
    for det in dets:
        is_gt = det == "ground_truth"
        root = SRC["cuda_dino"] if is_gt else SRC.get(det)
        if root is None or not root.exists():
            print(f"SKIP {det}: source root missing ({root})"); continue
        for stemdir in sorted(root.glob("*/")):
            stem = stemdir.name
            if not re.search(r"attenuation_dB_[0-9_v]+$", stem):
                continue
            if is_gt:
                src = sorted((stemdir / "gt_masks").glob("ground_truth_mask_ch0_f*.packed.npz")) \
                    or sorted((stemdir / "gt_masks").glob("ground_truth_mask_ch0_f*.npy"))
                prefix = "ground_truth_"
            else:
                src = sorted((stemdir / "mask_arrays").glob("mask_ch0_f*.packed.npz")) \
                    or sorted((stemdir / "mask_arrays").glob("mask_ch0_f*.npy"))
                prefix = ""
            if not src:
                continue
            dst = OUT / det / stem
            n = stage_masks(src, dst / "mask_arrays", strip_prefix=prefix)
            man = stemdir / "frame_manifest.csv"
            if man.exists():
                shutil.copy2(man, dst / "frame_manifest.csv")
            total += n
            print(f"  {det}/{stem}: {n} masks", flush=True)
    print(f"\nstaged {total} masks under {OUT}")
    print(f"detectors: {sorted(p.name for p in OUT.iterdir() if p.is_dir())}")


if __name__ == "__main__":
    main()

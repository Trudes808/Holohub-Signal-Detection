"""Materialize the fine-tuned detector's masks into a batch-eval-format run dir,
so the CANONICAL eval_detector_masks.py can score it identically to the deployed
detectors (same metrics, same source-SigMF-derived buckets, all frames).

For each capture it builds  <out-root>/finetuned_dino/<stem>/  containing:
  - frame_manifest.csv, gt_masks/, gt_annotations/   (symlinked from cuda_dino -- GT
    and geometry are detector-independent)
  - mask_arrays/mask_ch0_f{N}_{R}x{C}.packed.npz     (our model's mask per frame)

Then point eval_detector_masks.py at a root containing coherent_power, cuda_dino,
and finetuned_dino to get one consistent 3-detector table set.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

FT_ROOT = Path("/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning")
EVAL_DIR = Path("/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                "/infocom_evals/signal_detection_experiments")
for p in ("/home/bqn82/dinov3", str(FT_ROOT / "src"), str(EVAL_DIR)):
    sys.path.insert(0, p)

import eval_viz as v
import mask_eval_metrics as mem
import finetuned_infer as fi


def log(m): print(f"[ftrun] {m}", flush=True)


def save_packed(path: Path, mask: np.ndarray):
    R, C = mask.shape
    packed = np.packbits(mask.astype(np.uint8).ravel())
    np.savez_compressed(path, packed=packed, rows=R, cols=C)  # -> path + ".npz"


def _link(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="/tmp/usrp_spectrograms/batch_eval/sweep_20260630")
    ap.add_argument("--out-root",
                    default=str(FT_ROOT / "notebooks/sweep_detectors"))  # writable (/tmp sweep is root-owned)
    ap.add_argument("--captures-dir", default="/home/bqn82/captures")
    ap.add_argument("--ft-ckpt", default=str(FT_ROOT / "checkpoints/M1_ft/best.pt"))
    ap.add_argument("--detector-name", default="finetuned_dino",
                    help="run-dir name for this model (e.g. finetuned_dino, finetuned_dino_m2)")
    ap.add_argument("--ft-eval-meta", default=str(FT_ROOT / "eval_out/M1_ft/eval_meta.json"),
                    help="eval_meta.json with the model's val-tuned decision threshold")
    ap.add_argument("--ref-detector", default="cuda_dino", help="detector to mirror GT/manifest from")
    ap.add_argument("--stems", default=None)
    args = ap.parse_args()

    sweep = Path(args.sweep)
    out_root = Path(args.out_root)
    caps = [Path(args.captures_dir)]

    # combined root: symlink the two deployed detectors, generate finetuned_dino
    (out_root).mkdir(parents=True, exist_ok=True)
    for d in ("coherent_power", "cuda_dino"):
        _link(sweep / d, out_root / d)

    train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
    ds_meta = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
    thr = fi.load_threshold(args.ft_eval_meta)
    det = fi.FinetunedDetector(args.ft_ckpt, train_cfg, ds_meta, threshold=thr)
    log(f"[{args.detector_name}] ckpt={args.ft_ckpt} threshold={thr:.2f}")

    ref = args.ref_detector
    stems = (args.stems.split(",") if args.stems else
             sorted(p.name for p in (sweep / ref).iterdir() if p.is_dir()))

    for stem in stems:
        src_run = sweep / ref / stem
        dst_run = out_root / args.detector_name / stem
        (dst_run / "mask_arrays").mkdir(parents=True, exist_ok=True)
        for name in ("frame_manifest.csv", "gt_masks", "gt_annotations"):
            _link(src_run / name, dst_run / name)
        dp = v.find_capture_data(stem, caps)
        manifest = list(csv.DictReader(open(src_run / "frame_manifest.csv")))
        made = skipped = 0
        for row in manifest:
            n = int(row.get("complex_samples_read") or 0)
            R, C = int(row["fft_rows"]), int(row["fft_cols"])
            fnum = int(row["frame_number"])
            out_npz = dst_run / "mask_arrays" / f"mask_ch0_f{fnum}_{R}x{C}.packed.npz"
            if out_npz.exists():
                skipped += 1
                continue
            if n < det.nfft:
                continue
            iq = v.read_frame_iq(dp, int(row["local_file_offset_complex"]), n)
            mask = fi.to_display_grid(det.mask_for_iq(iq), R, C)
            save_packed(dst_run / "mask_arrays" / f"mask_ch0_f{fnum}_{R}x{C}.packed", mask)
            made += 1
        log(f"{stem}: {made} masks made, {skipped} already present ({len(manifest)} frames)")
    log(f"DONE -> {out_root} (detectors: coherent_power, cuda_dino, finetuned_dino)")


if __name__ == "__main__":
    main()

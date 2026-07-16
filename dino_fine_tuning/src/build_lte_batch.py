"""Build an LTE (OOD) batch-eval root with GT + M1/M2 fine-tuned masks.

Frames the LTE captures into the SAME 512x10240 geometry the deployed offline
pipeline uses, writes ground-truth masks/annotations + a frame_manifest, and runs
both fine-tuned models per frame. Layout (batch-eval compatible):

    <out-root>/finetuned_dino/<stem>/{frame_manifest.csv, gt_masks/, gt_annotations/, mask_arrays/}
    <out-root>/finetuned_dino_m2/<stem>/{mask_arrays/ + symlinked manifest/gt}

So the collaborator's coherent_power / cuda_dino runs on the LTE captures (same
framing) can be dropped in as sibling detector dirs and scored by the identical
eval_detector_masks.py path.
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
for p in ("/home/bqn82/dinov3", str(FT_ROOT / "src")):
    sys.path.insert(0, p)
import rfdata as rf
import finetuned_infer as fi

FRAME_ROWS, NFFT = 512, 10240          # deployed batch geometry
FRAME_SAMPLES = FRAME_ROWS * NFFT      # 5,242,880 complex samples per frame


def log(m): print(f"[lte] {m}", flush=True)


def save_packed(path: Path, mask: np.ndarray):
    np.savez_compressed(path, packed=np.packbits(mask.astype(np.uint8).ravel()),
                        rows=mask.shape[0], cols=mask.shape[1])


def _link(src: Path, dst: Path):
    if not (dst.exists() or dst.is_symlink()):
        os.symlink(src.resolve(), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default="/home/bqn82/captures/lte")
    ap.add_argument("--out-root", default=str(FT_ROOT / "notebooks/sweep_lte"))
    ap.add_argument("--m1-ckpt", default=str(FT_ROOT / "checkpoints/M1_ft/best.pt"))
    ap.add_argument("--m2-ckpt", default=str(FT_ROOT / "checkpoints/M2_ft/best.pt"))
    ap.add_argument("--stems", default=None)
    args = ap.parse_args()

    caps_dir = Path(args.captures_dir)
    out = Path(args.out_root)
    train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
    ds_meta = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
    M1 = fi.FinetunedDetector(args.m1_ckpt, train_cfg, ds_meta,
                              threshold=fi.load_threshold(FT_ROOT / "eval_out/M1_ft/eval_meta.json"))
    M2 = fi.FinetunedDetector(args.m2_ckpt, train_cfg, ds_meta,
                              threshold=fi.load_threshold(FT_ROOT / "eval_out/M2_ft/eval_meta.json"))
    log(f"M1 thr={M1.threshold:.2f}  M2 thr={M2.threshold:.2f}")

    metas = sorted(caps_dir.glob("*.sigmf-meta"))
    if args.stems:
        keep = set(args.stems.split(","))
        metas = [m for m in metas if m.stem in keep]

    for meta in metas:
        cap = rf.load_capture(meta)
        stem = cap.stem
        d1 = out / "finetuned_dino" / stem
        d2 = out / "finetuned_dino_m2" / stem
        for sub in ("gt_masks", "gt_annotations", "mask_arrays"):
            (d1 / sub).mkdir(parents=True, exist_ok=True)
        (d2 / "mask_arrays").mkdir(parents=True, exist_ok=True)

        mm = cap.memmap()
        n_frames = cap.n_samples // FRAME_SAMPLES
        manifest = []
        made = 0
        for fidx in range(n_frames):
            s = fidx * FRAME_SAMPLES
            fnum = fidx + 1                                  # 1-indexed, matches deployed
            tag = f"ch0_f{fnum}_{FRAME_ROWS}x{NFFT}"
            mask_gt, boxes = rf.build_frame_mask(cap, s, NFFT, FRAME_ROWS)
            save_packed(d1 / "gt_masks" / f"ground_truth_mask_{tag}.packed.npz", mask_gt)
            items = []
            for b in boxes:
                a = cap.annotations[b.ann_idx]
                items.append({
                    "label": b.label, "kind": b.kind,
                    "row_start": b.row0, "row_stop": b.row1, "col_start": b.col0, "col_stop": b.col1,
                    "sample_start": a.sample_start, "sample_count": a.sample_count,
                    "freq_lower_hz": a.freq_lower_hz, "freq_upper_hz": a.freq_upper_hz,
                    "occupied_bw_hz": abs(a.freq_upper_hz - a.freq_lower_hz),
                })
            (d1 / "gt_annotations" / f"ground_truth_{tag}.json").write_text(json.dumps({
                "channel": 0, "frame_number": fnum, "fft_rows": FRAME_ROWS, "fft_cols": NFFT,
                "samples_per_row": NFFT, "sample_rate_hz": cap.sample_rate,
                "span_hz": cap.sample_rate, "center_frequency_hz": cap.center_freq_hz,
                "items": items,
            }))
            iq = np.asarray(mm[s:s + FRAME_SAMPLES], dtype=np.complex64)
            save_packed(d1 / "mask_arrays" / f"mask_{tag}.packed.npz",
                        fi.to_display_grid(M1.mask_for_iq(iq), FRAME_ROWS, NFFT))
            save_packed(d2 / "mask_arrays" / f"mask_{tag}.packed.npz",
                        fi.to_display_grid(M2.mask_for_iq(iq), FRAME_ROWS, NFFT))
            manifest.append({
                "channel": 0, "frame_number": fnum, "file_offset_complex": s,
                "data_end_complex": s + FRAME_SAMPLES, "frame_end_complex": s + FRAME_SAMPLES,
                "complex_samples_read": FRAME_SAMPLES, "complex_samples_padded": 0,
                "partial_frame": "false", "fft_rows": FRAME_ROWS, "fft_cols": NFFT,
                "preview_rows": 256, "preview_cols": 512,
                "spectrogram_preview_pgm": "", "spectrogram_tensor_npy": "",
                "mask_preview_pgm": "", "mask_npy": f"mask_arrays/mask_{tag}.npy",
                "gt_annotations_json": f"gt_annotations/ground_truth_{tag}.json",
                "gt_mask_npy": f"gt_masks/ground_truth_mask_{tag}.npy",
                "global_sample_start": s, "global_data_end_sample": s + FRAME_SAMPLES,
                "global_frame_end_sample": s + FRAME_SAMPLES,
                "local_file_offset_complex": s, "local_data_end_complex": s + FRAME_SAMPLES,
                "local_frame_end_complex": s + FRAME_SAMPLES, "capture_sample_start": 0,
                "samples_per_row": NFFT,
            })
            made += 1
        with open(d1 / "frame_manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
        for name in ("frame_manifest.csv", "gt_masks", "gt_annotations"):
            _link(d1 / name, d2 / name)
        log(f"{stem} ({cap.attenuation_db}dB): {made} frames, {len(cap.annotations)} annotations")

    log(f"DONE -> {out}  (detectors: finetuned_dino, finetuned_dino_m2; add coherent_power/cuda_dino later)")


if __name__ == "__main__":
    main()

"""Generate combined tidy fact tables for THREE detectors over the batch-eval frames,
in the schema expected by plot_eval_results.py (line-per-detector graphs):

  region_metrics.csv : detector, attenuation_db, coverage, box_iou,
                       bucket_signal_class, bucket_bandwidth, bucket_pulse_length
  frame_pixel_metrics.csv : detector, attenuation_db, precision, recall, f1, iou,
                            fp_area_fraction, mask_present

Detectors:
  coherent_power, cuda_dino  -> masks loaded from the batch sweep (deployed detectors)
  finetuned_dino             -> our M1_ft model, run offline on each frame's IQ

Reuses mask_eval_metrics primitives + bucketers so the rows match the deployed eval.
"""
from __future__ import annotations

import argparse
import csv
import json
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


def log(m): print(f"[tables] {m}", flush=True)


def region_buckets(item):
    bw = abs(float(item.get("freq_upper_hz", 0)) - float(item.get("freq_lower_hz", 0)))
    length = item.get("sample_count")
    return {
        "bucket_signal_class": item.get("label", "unknown"),
        "bucket_bandwidth": mem.bucket_bandwidth({"occupied_bw_hz": bw}),
        "bucket_pulse_length": mem.bucket_length({"length_samples": length, "sample_count": length}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", default="/tmp/usrp_spectrograms/batch_eval/sweep_20260630")
    ap.add_argument("--captures-dir", default="/home/bqn82/captures")
    ap.add_argument("--ft-ckpt", default=str(FT_ROOT / "checkpoints/M1_ft/best.pt"))
    ap.add_argument("--out-dir", default=str(FT_ROOT / "notebooks/compare_tables"))
    ap.add_argument("--frames-per-capture", type=int, default=60)
    ap.add_argument("--stems", default=None, help="comma-separated subset (default: all)")
    args = ap.parse_args()

    batch_root = Path(args.batch_root)
    caps = [Path(args.captures_dir)]
    deployed = ["coherent_power", "cuda_dino"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
    ds_meta = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
    thr = fi.load_threshold(FT_ROOT / "eval_out/M1_ft/eval_meta.json")
    ft = fi.FinetunedDetector(args.ft_ckpt, train_cfg, ds_meta, threshold=thr)
    log(f"finetuned threshold={thr:.2f}")

    stems = (args.stems.split(",") if args.stems else
             sorted(p.name for p in (batch_root / "coherent_power").iterdir() if p.is_dir()))
    region_rows, frame_rows = [], []

    for stem in stems:
        atten = mem.parse_attenuation_db(stem)
        # manifests per deployed detector
        manifests = {d: {int(r["frame_number"]): r
                         for r in mem.load_manifest(batch_root / d / stem)} for d in deployed}
        all_frames = sorted(manifests["coherent_power"].keys())
        picks = v.pick_spread(all_frames, args.frames_per_capture)
        dp = v.find_capture_data(stem, caps)

        for fr in picks:
            ref = manifests["coherent_power"].get(fr)
            if ref is None:
                continue
            ref_dir = batch_root / "coherent_power" / stem
            gt = mem.load_mask_any(ref_dir / ref["gt_mask_npy"])
            if gt is None:
                continue
            R, C = gt.shape
            gt_b = (gt > 0).astype(np.uint8)
            gt_items = json.loads((ref_dir / ref["gt_annotations_json"]).read_text()).get("items", [])

            masks = {}
            for d in deployed:
                row = manifests[d].get(fr)
                m = mem.load_mask_any(batch_root / d / stem / row["mask_npy"]) if row and row.get("mask_npy") else None
                masks[d] = mem.resize_mask_nearest(m, R, C) if m is not None else None
            # finetuned: read IQ once, infer, map to grid
            iq = v.read_frame_iq(dp, int(ref["local_file_offset_complex"]), int(ref["complex_samples_read"]))
            masks["finetuned_dino"] = fi.to_display_grid(ft.mask_for_iq(iq), R, C)

            for det, m in masks.items():
                present = m is not None
                pred = (m > 0).astype(np.uint8) if present else np.zeros((R, C), np.uint8)
                pm = mem.pixel_metrics(pred, gt_b)
                frame_rows.append({
                    "detector": det, "file_stem": stem, "frame_number": fr,
                    "attenuation_db": atten, "precision": pm.precision, "recall": pm.recall,
                    "f1": pm.f1, "iou": pm.iou, "fp_area_fraction": pm.fp_area_fraction,
                    "mask_present": present,
                })
                for it in gt_items:
                    item = {"row_start": it["row_start"], "row_stop": it["row_stop"],
                            "col_start": it["col_start"], "col_stop": it["col_stop"]}
                    rr = mem.region_coverage(pred, item, R, C)
                    region_rows.append({
                        "detector": det, "file_stem": stem, "frame_number": fr,
                        "attenuation_db": atten, "coverage": rr.coverage, "box_iou": rr.box_iou,
                        **region_buckets(it),
                    })
        log(f"{stem} ({atten}dB): {len(picks)} frames x 3 detectors")

    _wcsv(out / "region_metrics.csv", region_rows)
    _wcsv(out / "frame_pixel_metrics.csv", frame_rows)
    log(f"DONE -> {out}  ({len(region_rows)} region rows, {len(frame_rows)} frame rows)")


def _wcsv(path, rows):
    if not rows:
        path.write_text(""); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()

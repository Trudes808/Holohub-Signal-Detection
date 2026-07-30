#!/usr/bin/env python3
"""Region-level detection table for the BER eval (results_v2 definition).

Applies the SAME detection rule the other evaluations use: a ground-truth emission
counts as detected when at least `--threshold` (default 0.1) of its area on the FFT
grid is covered by the detector's RAW MASK. This imports
`signal_detection_experiments/mask_eval_metrics.py` and calls its own
`region_coverage` / `resize_mask_nearest`, so the criterion is bit-for-bit the same
rather than a re-implementation.

Difference from that module: it aggregates to the **emission** rather than emitting
one row per (frame, annotation). An emission spanning several frames contributes a
GT box slice to each, so emission coverage = sum(covered px) / sum(box px) over its
frames -- "at least 0.1 of the emission is covered".

Output CSV (one row per emission), keyed so the MATLAB harness can join on the same
triple it already knows for every GT signal:

    sample_start, freq_lower_hz, freq_upper_hz, label, covered_px, box_px,
    coverage, n_frames, detected

Usage:
  python region_detect.py --run-dir <dir with mask_arrays/gt_annotations/frame_manifest>
                          --out <table.csv> [--threshold 0.1]
"""
from pathlib import Path
import argparse, json, math, sys, csv, collections

APP = Path(__file__).resolve().parent.parent.parent          # usrp_wideband_signal_detection
sys.path.insert(0, str(APP / "infocom_evals" / "signal_detection_experiments"))
import mask_eval_metrics as MEM                              # noqa: E402  (path set above)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="emission is 'detected' when covered fraction >= this (matches "
                         "comparison_config.yaml snr.det_threshold)")
    ap.add_argument("--frame-limit", type=int, default=None)
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    manifest = MEM.load_manifest(run_dir)
    if not manifest:
        sys.exit(f"no frame_manifest under {run_dir}")

    # emission key -> accumulator
    acc: dict = collections.OrderedDict()
    frames_seen = frames_no_mask = 0

    for record in manifest:
        if a.frame_limit is not None and record["frame_number"] > a.frame_limit:
            continue
        gt_rel = record.get("gt_mask_npy")
        if not gt_rel:
            continue
        gt = MEM.load_mask_any(MEM._artifact_path(run_dir, gt_rel))
        if gt is None:
            continue
        mask_rel = record.get("mask_npy")
        pred = MEM.load_mask_any(MEM._artifact_path(run_dir, mask_rel)) if mask_rel else None
        frames_seen += 1
        if pred is None:
            # Detector emitted no mask for this frame (usually a pipeline-drain tail
            # frame). mask_eval_metrics skips region rows here rather than scoring them
            # as misses; do the same so a drain artifact cannot bias detection.
            frames_no_mask += 1
            continue
        pred_on_gtgrid = MEM.resize_mask_nearest(pred, gt.shape[0], gt.shape[1])

        gt_ann_rel = record.get("gt_annotations_json")
        if not gt_ann_rel:
            continue
        p = MEM._artifact_path(run_dir, gt_ann_rel)
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        for item in payload.get("items", []):
            if item.get("kind") != "waveform":
                continue                      # data emissions only, as the BER eval scopes
            rr = MEM.region_coverage(pred_on_gtgrid, item, gt.shape[0], gt.shape[1])
            if rr.box_pixels <= 0:
                continue
            key = (int(item["sample_start"]),
                   int(item["freq_lower_hz"]), int(item["freq_upper_hz"]))
            e = acc.setdefault(key, {"label": item.get("label", ""), "covered": 0,
                                     "box": 0, "frames": 0})
            e["covered"] += int(rr.covered_pixels)
            e["box"] += int(rr.box_pixels)
            e["frames"] += 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ndet = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_start", "freq_lower_hz", "freq_upper_hz", "label",
                    "covered_px", "box_px", "coverage", "n_frames", "detected"])
        for (ss, lo, hi), e in acc.items():
            cov = e["covered"] / e["box"] if e["box"] else math.nan
            det = (not math.isnan(cov)) and cov >= a.threshold
            ndet += bool(det)
            w.writerow([ss, lo, hi, e["label"], e["covered"], e["box"],
                        f"{cov:.6f}", e["frames"], int(det)])
    print(f"region_detect: {run_dir.name}  emissions={len(acc)}  detected(>= {a.threshold})={ndet}"
          f"  frames={frames_seen} (no-mask {frames_no_mask})  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

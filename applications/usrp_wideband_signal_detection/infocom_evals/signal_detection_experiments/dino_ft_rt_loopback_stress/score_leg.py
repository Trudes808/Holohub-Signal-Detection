#!/usr/bin/env python3
"""Score a loopback leg's emitted masks against SigMF ground truth, next to its offline twin.

The offline eval binary (run_offline_refs.sh) wrote, for the SAME capture and the SAME detector
config, per-frame GT masks on the detector's grid + offline masks. The loopback run (run_leg.sh)
dumped the masks the live pipeline emitted, keyed by CHDR batch index (1-based, like the offline
frame_number, since the pcap is frame-aligned and a loop restarts on a frame boundary).

Steps: (1) align loopback->offline frames (small offset search on mask IoU; expect k=0),
(2) build an eval run dir that pairs the offline GT with the loopback masks (frames the live
pipeline never emitted are recorded as missing), (3) run mask_eval_metrics on both the loopback
and the offline run, (4) write tables + summary.json (+ summary.md) under <run>/eval/.

  python3 score_leg.py <run_dir> --offline <offline>/<leg>/<stem> --meta <stem>.sigmf-meta
                       [--frames-per-loop N]   # loopback frame -> offline frame modulo (multi-loop runs)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import mask_eval_metrics as mem  # noqa: E402

DET_THRESHOLDS = (0.1, 0.5)


def load_lb_manifest(masks_dir: Path) -> dict[int, Path]:
    man = masks_dir / "mask_dump_manifest.csv"
    out: dict[int, Path] = {}
    with man.open() as f:
        for r in csv.DictReader(f):
            out[int(r["frame_number"])] = masks_dir / r["mask_npy"]
    return out


def iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = mem.resize_mask_nearest(b.astype(np.uint8), a.shape[0], a.shape[1])
    a = a != 0
    b = b != 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def align(lb: dict[int, Path], off_by_frame: dict[int, dict], off_dir: Path, fpl: int | None,
          max_k: int = 3, sample: int = 24) -> tuple[int, dict]:
    """Best offset k such that loopback frame f matches offline frame map(f)+k."""
    def to_off(f: int) -> int:
        return ((f - 1) % fpl) + 1 if fpl else f
    cands = sorted(lb)
    if len(cands) > sample:
        idx = np.linspace(0, len(cands) - 1, sample).astype(int)
        cands = [cands[i] for i in idx]
    curve = {}
    for k in range(-max_k, max_k + 1):
        vals = []
        for f in cands:
            of = to_off(f) + k
            if of not in off_by_frame or not off_by_frame[of].get("mask_npy"):
                continue
            a = mem.load_mask_any(lb[f])
            b = mem.load_mask_any(off_dir / off_by_frame[of]["mask_npy"])
            if a is None or b is None:
                continue
            if (a != 0).sum() == 0 and (b != 0).sum() == 0:
                continue
            vals.append(iou(a, b))
        if vals:
            curve[k] = float(np.median(vals))
    best = max(curve, key=curve.get) if curve else 0
    return best, curve


def build_eval_run(run_dir: Path, leg: str, stem: str, off_dir: Path, lb: dict[int, Path],
                   off_rows: list[dict], k: int, fpl: int | None) -> tuple[Path, dict]:
    eval_run = run_dir / "eval_run" / f"{leg}_loopback" / stem
    if eval_run.exists():
        import shutil
        shutil.rmtree(eval_run)
    (eval_run / "mask_arrays").mkdir(parents=True)
    for sub in ("gt_masks", "gt_annotations"):
        os.symlink(off_dir / sub, eval_run / sub)
    # offline frame -> loopback frame(s) that map onto it (first occurrence wins for multi-loop)
    off_to_lb: dict[int, int] = {}
    for f in sorted(lb):
        of = (((f - 1) % fpl) + 1 if fpl else f) + k
        off_to_lb.setdefault(of, f)
    rows_out = []
    present = 0
    for r in off_rows:
        r = dict(r)
        of = int(r["frame_number"])
        f = off_to_lb.get(of)
        if f is not None:
            link = eval_run / "mask_arrays" / f"mask_ch0_f{of}_lb.npy"
            os.symlink(lb[f], link)
            r["mask_npy"] = f"mask_arrays/{link.name}"
            present += 1
        else:
            r["mask_npy"] = ""
        rows_out.append(r)
    with (eval_run / "frame_manifest.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    return eval_run, {"offline_frames": len(rows_out), "frames_with_loopback_mask": present}


def summarize(frame_rows: list[dict], region_rows: list[dict]) -> dict:
    fr = [r for r in frame_rows if r.get("mask_present")]
    def mean(key):
        vals = [r[key] for r in fr if isinstance(r.get(key), float) and not math.isnan(r[key])]
        return float(np.mean(vals)) if vals else None
    out = {
        "frames_total": len(frame_rows), "frames_with_mask": len(fr),
        "pixel_precision": mean("precision"), "pixel_recall": mean("recall"),
        "pixel_f1": mean("f1"), "pixel_iou": mean("iou"), "fp_area_fraction": mean("fp_area_fraction"),
    }
    # region-level: only frames that have a mask (so accuracy is per processed frame), with
    # a separate "effective" rate that counts regions in unprocessed frames as missed.
    present_frames = {r["frame_number"] for r in fr}
    regions = [r for r in region_rows if r["frame_number"] in present_frames]
    out["regions_total"] = len(region_rows)
    out["regions_in_processed_frames"] = len(regions)
    for t in DET_THRESHOLDS:
        det = [(r["coverage"] >= t) for r in regions if not math.isnan(r["coverage"])]
        out[f"region_det_rate@{t}"] = float(np.mean(det)) if det else None
        out[f"region_detected@{t}"] = int(sum(det))
        # placeholder: main() rewrites this against the OFFLINE leg's full region count (regions in
        # frames the live pipeline never emitted are misses; evaluate_run drops those frames' rows)
        out[f"region_det_rate_effective@{t}"] = (float(sum(det) / len(region_rows)) if region_rows else None)
    out["region_mean_coverage"] = float(np.mean([r["coverage"] for r in regions if not math.isnan(r["coverage"])])) if regions else None
    per_class: dict = {}
    for cls in sorted({r["bucket_signal_class"] for r in regions}):
        rs = [r for r in regions if r["bucket_signal_class"] == cls]
        per_class[cls] = {"n": len(rs),
                          **{f"det@{t}": float(np.mean([r["coverage"] >= t for r in rs])) for t in DET_THRESHOLDS},
                          "mean_coverage": float(np.mean([r["coverage"] for r in rs]))}
    out["per_class"] = per_class
    per_bw: dict = {}
    for bw in sorted({r["bucket_bandwidth"] for r in regions}):
        rs = [r for r in regions if r["bucket_bandwidth"] == bw]
        per_bw[bw] = {"n": len(rs),
                      **{f"det@{t}": float(np.mean([r["coverage"] >= t for r in rs])) for t in DET_THRESHOLDS}}
    out["per_bandwidth"] = per_bw
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--offline", required=True, help="<offline root>/<leg>/<capture stem> run dir")
    ap.add_argument("--meta", required=True, help="source .sigmf-meta (breakdown attributes)")
    ap.add_argument("--frames-per-loop", type=int, default=None)
    ap.add_argument("--leg", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    off_dir = Path(args.offline).resolve()
    meta = Path(args.meta).resolve()
    stem = off_dir.name
    leg = args.leg or json.loads((run_dir / "leg_meta.json").read_text()).get("leg", off_dir.parent.name)
    out_dir = run_dir / "eval"
    out_dir.mkdir(exist_ok=True)

    off_rows_all = mem.load_manifest(off_dir)
    lb = load_lb_manifest(run_dir / "masks")
    # Frames per replay loop = what the pcap actually carries (frame-aligned pcap), derived from the
    # offline frame geometry; the offline reference is restricted to those frames so both sides are
    # scored on the SAME data (the offline run also sees the capture's tail frames the pcap drops).
    fpl = args.frames_per_loop
    meta_json = json.loads((run_dir / "leg_meta.json").read_text()) if (run_dir / "leg_meta.json").exists() else {}
    if fpl is None and meta_json.get("pcap_packets") and off_rows_all:
        spf = int(off_rows_all[0]["fft_rows"]) * int(off_rows_all[0]["samples_per_row"])
        fpl = int(meta_json["pcap_packets"]) * 1024 // spf
        print(f"[{run_dir.name}] frames per loop derived from pcap: {fpl} (offline has {len(off_rows_all)} frames)")
    off_rows = [r for r in off_rows_all if fpl is None or int(r["frame_number"]) <= fpl]
    off_by_frame = {int(r["frame_number"]): r for r in off_rows}
    k, curve = align(lb, off_by_frame, off_dir, fpl)
    print(f"[{run_dir.name}] alignment: best k={k:+d}  curve={ {kk: round(v, 3) for kk, v in curve.items()} }")

    eval_run, pairing = build_eval_run(run_dir, leg, stem, off_dir, lb, off_rows, k, fpl)

    # online-vs-offline agreement at the chosen alignment (all paired frames)
    agree = []
    for of, r in off_by_frame.items():
        link = eval_run / "mask_arrays" / f"mask_ch0_f{of}_lb.npy"
        if link.exists() and r.get("mask_npy"):
            a = mem.load_mask_any(link)
            b = mem.load_mask_any(off_dir / r["mask_npy"])
            if a is not None and b is not None:
                agree.append(iou(a, b))
    agree = np.array(agree)

    cfg = mem.EvalConfig(region_coverage_threshold=0.5)
    lb_frames, lb_regions = mem.evaluate_run(eval_run, detector=f"{leg}_loopback", file_stem=stem,
                                             sigmf_meta_path=meta, config=cfg)
    off_frames, off_regions = mem.evaluate_run(off_dir, detector=f"{leg}_offline", file_stem=stem,
                                               sigmf_meta_path=meta, config=cfg, frame_limit=fpl)
    mem.write_tables(lb_frames + off_frames, lb_regions + off_regions, out_dir)

    # Validity: the CHDR converter assembles a batch from the next N packets regardless of sequence
    # gaps, so after the FIRST dropped packet every later frame holds a shifted slice of the stream and
    # no frame-indexed ground truth applies any more. A clean pass reproduces offline bit-exactly
    # (median IoU 1.0); a median far below that means per-frame GT scores are NOT meaningful.
    # judged on the frames actually paired for scoring (first replay loop), not the sampled curve
    best_iou = float(np.median(agree)) if agree.size else 0.0
    gt_scoring_valid = best_iou >= 0.5
    summary = {
        "run": run_dir.name, "leg": leg, "stem": stem, "offline_dir": str(off_dir),
        "alignment_k": k, "alignment_curve": curve, "pairing": pairing,
        "gt_scoring_valid": gt_scoring_valid,
        "gt_scoring_note": (("" if (agree.size and float((agree >= 0.999).mean()) > 0.95) else
                             f"only {100*float((agree >= 0.999).mean()):.0f}% of paired frames are identical to offline "
                             "(frames after the first packet drop hold shifted samples); loopback GT metrics are a lower bound")
                            if gt_scoring_valid else
                            "frame alignment lost after packet drops (median online-vs-offline IoU "
                            f"{best_iou:.3f}); per-frame GT metrics below are not meaningful"),
        "online_vs_offline_iou": ({"n": int(agree.size), "mean": float(agree.mean()), "median": float(np.median(agree)),
                                   "p5": float(np.percentile(agree, 5)), "min": float(agree.min()),
                                   "frac_identical": float((agree >= 0.999).mean())} if agree.size else None),
        "frames_per_loop": fpl,
        "loopback": summarize(lb_frames, lb_regions),
        "offline": summarize(off_frames, off_regions),
    }
    # effective detection = detected regions / ALL ground-truth regions in the frames that were sent
    # (the offline leg processed every frame, so its region count is the denominator for both sides)
    total_regions = summary["offline"]["regions_total"]
    for side in ("loopback", "offline"):
        for t in DET_THRESHOLDS:
            summary[side][f"region_det_rate_effective@{t}"] = (
                summary[side][f"region_detected@{t}"] / total_regions if total_regions else None)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def row(name, s):
        return (f"| {name} | {s['frames_with_mask']}/{s['frames_total']} | "
                f"{s['region_det_rate@0.1']:.3f} | {s['region_det_rate@0.5']:.3f} | {s['region_det_rate_effective@0.5']:.3f} | "
                f"{s['pixel_precision']:.3f} | {s['pixel_recall']:.3f} | {s['pixel_f1']:.3f} | {s['pixel_iou']:.3f} | {s['fp_area_fraction']:.4f} |")
    lines = [f"# {run_dir.name} ({leg}) vs ground truth -- {stem}", "",
             ("" if gt_scoring_valid else f"**INVALID FOR GT SCORING**: {summary['gt_scoring_note']}"), 
             f"alignment k={k:+d}; online-vs-offline mask IoU median={summary['online_vs_offline_iou']['median']:.3f} "
             f"mean={summary['online_vs_offline_iou']['mean']:.3f} p5={summary['online_vs_offline_iou']['p5']:.3f} "
             f"identical={summary['online_vs_offline_iou']['frac_identical']*100:.0f}% (n={summary['online_vs_offline_iou']['n']})"
             if summary["online_vs_offline_iou"] else "no paired frames", "",
             "| leg | frames w/ mask | region det@0.1 | region det@0.5 | effective det@0.5 | px precision | px recall | px F1 | px IoU | FP area |",
             "|---|---|---|---|---|---|---|---|---|---|",
             row("loopback", summary["loopback"]), row("offline", summary["offline"]), "",
             "per class (region det@0.5, loopback / offline):", ""]
    for cls in summary["offline"]["per_class"]:
        lo = summary["loopback"]["per_class"].get(cls, {}).get("det@0.5", float("nan"))
        of = summary["offline"]["per_class"][cls]["det@0.5"]
        n = summary["offline"]["per_class"][cls]["n"]
        lines.append(f"- {cls:<14} n={n:<5} loopback {lo:.3f}  offline {of:.3f}")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

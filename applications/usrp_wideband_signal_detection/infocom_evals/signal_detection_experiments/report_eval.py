#!/usr/bin/env python3
"""Generate a Markdown evaluation report from the tidy fact tables.

Reads ``frame_pixel_metrics.csv`` + ``region_metrics.csv`` (produced by
``eval_detector_masks.py``) and writes ``report.md`` with:

  * per-detector pixel-metric summary (precision / recall / F1 / IoU, FP area),
  * per-detector detection-rate tables for every breakdown dimension in
    ``mask_eval_metrics.BUCKETERS`` (signal class, bandwidth, pulse length,
    power level, attenuation, time group),
  * a head-to-head detector comparison.

Stdlib-only (csv + statistics) so it runs anywhere; for richer interactive
analysis use ``mask_eval_metrics.detection_rate_by`` in a pandas notebook.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import mask_eval_metrics as mem


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value: str):
    try:
        v = float(value)
        return v if v == v else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _mean(values) -> float:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else float("nan")


def pixel_summary(frame_rows: list[dict]) -> dict:
    by_det: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in frame_rows:
        grouped[row["detector"]].append(row)
    for detector, rows in grouped.items():
        by_det[detector] = {
            "frames": len(rows),
            "precision": _mean(_f(r["precision"]) for r in rows),
            "recall": _mean(_f(r["recall"]) for r in rows),
            "f1": _mean(_f(r["f1"]) for r in rows),
            "iou": _mean(_f(r["iou"]) for r in rows),
            "fp_area_fraction": _mean(_f(r["fp_area_fraction"]) for r in rows),
        }
    return by_det


def detection_rate_table(region_rows: list[dict], dimension: str) -> dict:
    """Return {(detector, bucket): (n_regions, n_detected, mean_coverage)}."""
    col = f"bucket_{dimension}"
    agg: dict[tuple, list] = defaultdict(lambda: [0, 0, []])
    for row in region_rows:
        bucket = row.get(col, "unknown")
        detected = str(row.get("detected", "")).lower() == "true"
        cov = _f(row.get("coverage"))
        key = (row["detector"], bucket)
        agg[key][0] += 1
        agg[key][1] += 1 if detected else 0
        if cov is not None:
            agg[key][2].append(cov)
    return agg


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables-dir", required=True,
                        help="Directory containing frame_pixel_metrics.csv + region_metrics.csv.")
    parser.add_argument("--out", default=None, help="Output markdown path (default <tables-dir>/report.md).")
    args = parser.parse_args()

    tables_dir = Path(args.tables_dir)
    frame_rows = read_csv(tables_dir / "frame_pixel_metrics.csv")
    region_rows = read_csv(tables_dir / "region_metrics.csv")
    out_path = Path(args.out) if args.out else tables_dir / "report.md"

    lines: list[str] = ["# Detector evaluation report", ""]
    detectors = sorted({r["detector"] for r in frame_rows} | {r["detector"] for r in region_rows})
    lines.append(f"Detectors: {', '.join(detectors) or '(none)'}  ")
    lines.append(f"Frames evaluated: {len(frame_rows)}  ")
    lines.append(f"Annotation regions evaluated: {len(region_rows)}")
    lines.append("")

    # --- pixel summary ---
    lines.append("## Pixel-level metrics (mean over frames)")
    lines.append("")
    px = pixel_summary(frame_rows)
    rows = [[d, px[d]["frames"],
             f"{px[d]['precision']:.4f}", f"{px[d]['recall']:.4f}",
             f"{px[d]['f1']:.4f}", f"{px[d]['iou']:.4f}",
             f"{px[d]['fp_area_fraction']:.5f}"]
            for d in sorted(px)]
    lines.append(md_table(["detector", "frames", "precision", "recall", "f1", "iou", "fp_area_frac"], rows))
    lines.append("")

    # --- detection rate per breakdown dimension ---
    for dimension in mem.BUCKETERS:
        agg = detection_rate_table(region_rows, dimension)
        if not agg:
            continue
        lines.append(f"## Detection rate by {dimension}")
        lines.append("")
        table_rows = []
        for (detector, bucket), (n, n_det, covs) in sorted(agg.items()):
            rate = n_det / n if n else float("nan")
            table_rows.append([detector, bucket, n, n_det, f"{rate:.3f}", f"{_mean(covs):.3f}"])
        lines.append(md_table(["detector", dimension, "n_regions", "n_detected", "detection_rate", "mean_coverage"],
                              table_rows))
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

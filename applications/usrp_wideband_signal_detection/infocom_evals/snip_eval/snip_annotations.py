#!/usr/bin/env python3
"""Build the overall detections annotation file for a (detector, capture) run.

Clusters each frame's mask into signal boxes using the SAME rule as the C++ signal_snipper
(4-connected components >= min_box_pixels, then coalesce boxes within merge_gap_{rows,cols}),
maps every box to (sample_start, sample_count, freq_lower/upper_edge) on the ORIGINAL capture
timeline, and writes a SigMF-style `<stem>_snipped.sigmf-meta` whose `annotations[]` are labelled
`detected_waveform`. Because the clustering matches the snipper, these annotations line up with the
per-snippet recordings the snipper emits.

Usage:
  python3 snip_annotations.py --run-dir <batch_root>/<detector>/<stem> \
      --captures-dir /home/bqn82/captures --out-dir <out>/<detector>/<stem>_snipped
"""
from __future__ import annotations
import argparse, csv, glob, json
from pathlib import Path
import numpy as np
from scipy import ndimage

def load_mask(p: Path) -> np.ndarray:
    if p.suffix == ".npy":
        return (np.load(p) != 0)
    z = np.load(p)
    return np.unpackbits(z["packed"])[:int(z["rows"]) * int(z["cols"])].reshape(int(z["rows"]), int(z["cols"])) != 0

def boxes_from_mask(m, min_box_pixels, gap_rows, gap_cols, max_boxes=1500):
    """4-conn CC >= min_box_pixels -> bboxes, coalesced within gaps (like the snipper). Fast:
    vectorized size gate + single-pass union-find; saturated frames collapse to one full-extent box."""
    lab, n = ndimage.label(m)  # default = 4-connectivity
    if n == 0:
        return []
    sizes = np.bincount(lab.ravel())                      # sizes[k] = component k pixel count
    objs = ndimage.find_objects(lab)
    boxes = [[o[0].start, o[0].stop - 1, o[1].start, o[1].stop - 1]
             for k, o in enumerate(objs, start=1) if o is not None and sizes[k] >= min_box_pixels]
    if not boxes:
        return []
    if len(boxes) > max_boxes:                            # saturated -> avoid O(n^2); one full box
        return [(0, m.shape[0] - 1, 0, m.shape[1] - 1)]
    parent = list(range(len(boxes)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(len(boxes)):
        bi = boxes[i]
        for j in range(i + 1, len(boxes)):
            bj = boxes[j]
            if (max(0, max(bi[0], bj[0]) - min(bi[1], bj[1])) <= gap_rows and
                    max(0, max(bi[2], bj[2]) - min(bi[3], bj[3])) <= gap_cols):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i, b in enumerate(boxes):
        r = find(i); g = groups.get(r)
        if g is None:
            groups[r] = list(b)
        else:
            g[0] = min(g[0], b[0]); g[1] = max(g[1], b[1]); g[2] = min(g[2], b[2]); g[3] = max(g[3], b[3])
    return [tuple(v) for v in groups.values()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path, help="<batch_root>/<detector>/<stem>")
    ap.add_argument("--captures-dir", type=Path, default=Path("/home/bqn82/captures"))  # lab-admin home has no captures/
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-box-pixels", type=int, default=256)
    ap.add_argument("--merge-gap-rows", type=int, default=16)
    ap.add_argument("--merge-gap-cols", type=int, default=80)
    a = ap.parse_args()

    run = a.run_dir
    stem = run.name
    detector = run.parent.name
    manifest = {int(r["frame_number"]): r for r in csv.DictReader(open(run / "frame_manifest.csv"))}
    # fs/center from the source SigMF meta
    src_meta = json.load(open(a.captures_dir / f"{stem}.sigmf-meta"))
    g = src_meta.get("global", {})
    fs = float(g.get("core:sample_rate")); center = float(g.get("core:frequency", 0.0))

    ann = []
    for mf in sorted((run / "mask_arrays").glob("mask_ch0_f*.*")):
        num = int(mf.name.split("_f")[1].split("_")[0])
        row = manifest.get(num)
        if row is None:
            continue
        m = load_mask(mf); rows, cols = m.shape
        fsc = int(float(row["complex_samples_read"]))
        gstart = int(float(row["global_sample_start"]))
        for r0, r1, c0, c1 in boxes_from_mask(m, a.min_box_pixels, a.merge_gap_rows, a.merge_gap_cols):
            local_start = int(np.floor((r0 / rows) * fsc))
            local_end = int(np.ceil(((r1 + 1) / rows) * fsc))
            ann.append({
                "core:sample_start": gstart + local_start,
                "core:sample_count": max(1, local_end - local_start),
                "core:freq_lower_edge": center + ((c0 / cols) - 0.5) * fs,
                "core:freq_upper_edge": center + (((c1 + 1) / cols) - 0.5) * fs,
                "core:label": "detected_waveform",
                "wfgt:detector": detector, "wfgt:frame_number": num,
            })

    out_dir = a.out_dir or (run.parent.parent / f"{detector}_snipped" / f"{stem}_snipped")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}_snipped.sigmf-meta"
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": fs, "core:frequency": center,
                       "core:description": f"{detector} detections on {stem} (annotations only)"},
            "captures": [{"core:sample_start": 0}],
            "annotations": sorted(ann, key=lambda x: x["core:sample_start"])}
    out.write_text(json.dumps(meta, indent=2))
    print(f"{detector}/{stem}: {len(ann)} detections -> {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

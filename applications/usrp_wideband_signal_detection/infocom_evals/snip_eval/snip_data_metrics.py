#!/usr/bin/env python3
"""Metadata-only data-saving eval: per (detector, capture), cluster the detector's masks into
detection boxes (SAME rule as snip_annotations / the C++ signal_snipper), and from those boxes emit

  1. ``<stem>_detected.sigmf-meta`` — SigMF annotations = the DETECTOR's detections (not GT), on the
     original capture timeline. No ``.sigmf-data`` is written (avoids N copies of a ~14 GB capture).
  2. a row of data-saving metrics: how many bytes a resample+filter collector would keep if it saved
     only the detected regions, vs saving everything ("save-all").

The resample+filter footprint per detection box = bandwidth[Hz] * duration[s] * BYTES_PER_SAMPLE.
With bandwidth = (Δcols/cols)*fs and duration = (Δsamples)/fs, fs cancels:
    bytes_box = BYTES_PER_SAMPLE * (Δcols/cols) * Δsamples
Summing per-box matches the snipper's per-snippet output (overlapping detections are saved
separately, as the snipper would). This is the container-free, all-detector analogue of the
notebook's ``resample_meas_TB_hr`` (which reads real snippet .sigmf-data and only exists for the two
container detectors).

Usage:
  python3 snip_data_metrics.py --batch-root /tmp/ds_batch --captures-dir /home/bqn82/captures \
      --out-dir /tmp/ds_batch/detected
  (add --detectors a b ...  to restrict; clustering knobs match snip_annotations defaults)
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

# reuse the exact clustering + mask loader used to build the snipper-aligned annotations
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snip_annotations import load_mask, boxes_from_mask  # noqa: E402

BYTES_PER_SAMPLE = 8          # cf32 (complex float32)
SEC_PER_HR = 3600.0


def _atten(stem: str):
    import re
    m = re.search(r"dB_(\d+)", stem)
    return int(m.group(1)) if m else None


def process_run(run: Path, captures_dir: Path, out_dir: Path,
                min_box_pixels: int, gap_rows: int, gap_cols: int) -> dict | None:
    stem = run.name
    detector = run.parent.name
    manifest_path = run / "frame_manifest.csv"
    src_meta_path = captures_dir / f"{stem}.sigmf-meta"
    if not manifest_path.exists() or not src_meta_path.exists():
        return None
    manifest = {int(r["frame_number"]): r for r in csv.DictReader(open(manifest_path))}
    g = json.load(open(src_meta_path)).get("global", {})
    fs = float(g.get("core:sample_rate"))
    center = float(g.get("core:frequency", 0.0))

    # capture duration (s): prefer the real .sigmf-data size / (bytes*fs), like the notebook
    src_data = captures_dir / f"{stem}.sigmf-data"
    if src_data.exists():
        capture_sec = src_data.stat().st_size / (BYTES_PER_SAMPLE * fs)
    else:  # fallback: last frame's global end / fs
        ends = [int(float(r["global_sample_start"])) + int(float(r["complex_samples_read"]))
                for r in manifest.values()]
        capture_sec = (max(ends) / fs) if ends else float("nan")

    ann, retained_bytes, n_det = [], 0.0, 0
    tf_cells_on = tf_cells_tot = 0
    mfiles = sorted((run / "mask_arrays").glob("mask_ch0_f*.*"))
    for mf in mfiles:
        num = int(mf.name.split("_f")[1].split("_")[0])
        row = manifest.get(num)
        if row is None:
            continue
        m = load_mask(mf); rows, cols = m.shape
        tf_cells_on += int(m.sum()); tf_cells_tot += m.size
        fsc = int(float(row["complex_samples_read"]))
        gstart = int(float(row["global_sample_start"]))
        for r0, r1, c0, c1 in boxes_from_mask(m, min_box_pixels, gap_rows, gap_cols):
            local_start = int(np.floor((r0 / rows) * fsc))
            local_end = int(np.ceil(((r1 + 1) / rows) * fsc))
            samp = max(1, local_end - local_start)
            freq_frac = (c1 + 1 - c0) / cols
            retained_bytes += BYTES_PER_SAMPLE * freq_frac * samp
            n_det += 1
            ann.append({
                "core:sample_start": gstart + local_start,
                "core:sample_count": samp,
                "core:freq_lower_edge": center + ((c0 / cols) - 0.5) * fs,
                "core:freq_upper_edge": center + (((c1 + 1) / cols) - 0.5) * fs,
                "core:label": "detected_waveform",
                "wfgt:detector": detector, "wfgt:frame_number": num,
            })

    # ---- write <stem>_detected.sigmf-meta (annotations only, no .sigmf-data) ----
    det_dir = out_dir / detector
    det_dir.mkdir(parents=True, exist_ok=True)
    meta = {"global": {"core:datatype": "cf32_le", "core:sample_rate": fs, "core:frequency": center,
                       "core:description": f"{detector} detections on {stem} (annotations only; "
                                           f"no IQ — see data_saving_metrics.csv for footprint)"},
            "captures": [{"core:sample_start": 0}],
            "annotations": sorted(ann, key=lambda x: x["core:sample_start"])}
    (det_dir / f"{stem}_detected.sigmf-meta").write_text(json.dumps(meta, indent=2))

    # ---- data-saving metrics ----
    saveall_bytes = BYTES_PER_SAMPLE * fs * capture_sec
    retained_frac = (retained_bytes / saveall_bytes) if saveall_bytes > 0 else float("nan")
    return {
        "detector": detector, "file_stem": stem, "attenuation_db": _atten(stem),
        "n_frames": len(mfiles), "n_detections": n_det,
        "capture_sec": round(capture_sec, 4),
        "saveall_bytes": int(saveall_bytes),
        "retained_bytes": int(retained_bytes),
        "retained_frac": retained_frac,
        "pct_data_saved": 100.0 * (1.0 - retained_frac),
        "reduction_factor": (1.0 / retained_frac) if retained_frac > 0 else float("inf"),
        "retained_TB_per_hour": retained_bytes / capture_sec * SEC_PER_HR / 1e12 if capture_sec > 0 else float("nan"),
        "saveall_TB_per_hour": saveall_bytes / capture_sec * SEC_PER_HR / 1e12 if capture_sec > 0 else float("nan"),
        "tf_coverage": (tf_cells_on / tf_cells_tot) if tf_cells_tot else float("nan"),
        "detected_meta": str(det_dir / f"{stem}_detected.sigmf-meta"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-root", type=Path, default=Path("/tmp/ds_batch"))
    ap.add_argument("--captures-dir", type=Path, default=Path("/home/bqn82/captures"))
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/ds_batch/detected"))
    ap.add_argument("--detectors", nargs="+", default=None, help="default: all under batch-root")
    ap.add_argument("--min-box-pixels", type=int, default=256)
    ap.add_argument("--merge-gap-rows", type=int, default=16)
    ap.add_argument("--merge-gap-cols", type=int, default=80)
    a = ap.parse_args()

    dets = a.detectors or sorted(p.name for p in a.batch_root.iterdir()
                                 if p.is_dir() and any(p.glob("*/mask_arrays")))
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for det in dets:
        for run in sorted((a.batch_root / det).glob("*/")):
            if not (run / "mask_arrays").is_dir():
                continue
            if _atten(run.name) is None:  # skip non-capture dirs
                continue
            r = process_run(run, a.captures_dir, a.out_dir,
                            a.min_box_pixels, a.merge_gap_rows, a.merge_gap_cols)
            if r is None:
                print(f"  skip {det}/{run.name} (no manifest or source meta)")
                continue
            rows.append(r)
            print(f"[{det}/{run.name}] dets={r['n_detections']:5d}  "
                  f"saved={r['pct_data_saved']:.2f}%  reduction={r['reduction_factor']:.1f}x  "
                  f"-> {Path(r['detected_meta']).name}")
    if not rows:
        print("no runs processed", file=sys.stderr); return 1
    cols = list(rows[0].keys())
    csv_path = a.out_dir / "data_saving_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {csv_path}")
    print(f"detected metas -> {a.out_dir}/<detector>/<stem>_detected.sigmf-meta")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

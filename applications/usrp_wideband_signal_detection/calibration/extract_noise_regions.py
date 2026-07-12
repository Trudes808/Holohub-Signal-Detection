#!/usr/bin/env python3
"""Extract annotation-free (noise-only) sample regions from SigMF captures and
concatenate them into a single noise-only SigMF capture for DINO positional-template
calibration.

Each capture's filename/`captures[0].core:sample_start` places it on an absolute
sample timeline; every annotation carries absolute `core:sample_start`/`sample_count`.
A time span that overlaps no annotation is guaranteed noise across all frequencies, so
we cut those spans out of each `.sigmf-data` and stitch them into one long noise file
(discontinuities are irrelevant for a per-position noise-floor estimate). The output
carries an empty annotation list.

Usage:
  python3 extract_noise_regions.py \
      --inputs "generated_inputs/attenuation_dB_5*_*.sigmf-data" \
               "generated_inputs/attenuation_dB_60_*.sigmf-data" \
      --output /tmp/usrp_spectrograms/calibration_noise/noise_concat.sigmf-data
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import sys

# bytes per complex sample by SigMF datatype (only the ones we expect here).
_BYTES_PER_SAMPLE = {"cf32_le": 8, "cf64_le": 16, "ci16_le": 4, "ci8": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch annotation-free noise regions into one SigMF capture.")
    p.add_argument("--inputs", nargs="+", required=True,
                   help="Input .sigmf-data paths or globs (repeatable / space-separated).")
    p.add_argument("--output", required=True, type=Path, help="Output noise .sigmf-data path.")
    p.add_argument("--guard-samples", type=int, default=40960,
                   help="Shrink each noise gap by this many samples on each side to avoid annotation "
                        "edge bleed (default 40960 ~= 2 x 20480-pt FFT).")
    p.add_argument("--min-gap-samples", type=int, default=262144,
                   help="Discard noise gaps shorter than this after guarding (default 262144).")
    return p.parse_args()


def meta_path_for(data_path: Path) -> Path:
    if data_path.name.endswith(".sigmf-data"):
        return data_path.with_name(data_path.name[:-len(".sigmf-data")] + ".sigmf-meta")
    return data_path.with_suffix(".sigmf-meta")


def annotation_free_gaps(file_start: int, file_len: int, annotations: list, guard: int, min_gap: int):
    """Return list of (rel_start, rel_len) noise spans in file-relative sample coords."""
    file_end = file_start + file_len
    covered = []
    for a in annotations:
        ss = a.get("core:sample_start")
        sc = a.get("core:sample_count")
        if ss is None:
            continue
        # Annotations without a count still mark a point/region start; treat as covering
        # to the file end conservatively if count is missing.
        a0 = int(ss)
        a1 = int(ss) + int(sc) if sc else file_end
        lo = max(a0, file_start)
        hi = min(a1, file_end)
        if hi > lo:
            covered.append((lo, hi))
    covered.sort()
    # Merge overlapping covered intervals.
    merged = []
    for lo, hi in covered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    # Complement within [file_start, file_end].
    gaps = []
    cursor = file_start
    for lo, hi in merged:
        if lo > cursor:
            gaps.append((cursor, lo))
        cursor = max(cursor, hi)
    if cursor < file_end:
        gaps.append((cursor, file_end))
    # Guard + min-length filter; convert to file-relative.
    out = []
    for lo, hi in gaps:
        lo_g = lo + guard
        hi_g = hi - guard
        if hi_g - lo_g >= min_gap:
            out.append((lo_g - file_start, hi_g - lo_g))
    return out


def main() -> int:
    args = parse_args()
    inputs: list[Path] = []
    for pat in args.inputs:
        matches = sorted(glob.glob(pat))
        if matches:
            inputs.extend(Path(m) for m in matches)
        elif Path(pat).is_file():
            inputs.append(Path(pat))
    inputs = [p for p in inputs if p.name.endswith(".sigmf-data")]
    if not inputs:
        print("no .sigmf-data inputs matched", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_data = args.output
    out_meta = meta_path_for(args.output)

    global_meta = None
    bytes_per_sample = None
    total_noise_samples = 0
    per_file_report = []

    with open(out_data, "wb") as out_fh:
        for data_path in inputs:
            mpath = meta_path_for(data_path)
            if not mpath.is_file():
                print(f"  skip {data_path.name}: missing meta", file=sys.stderr)
                continue
            meta = json.loads(mpath.read_text())
            g = meta.get("global", {})
            dtype = g.get("core:datatype", "cf32_le")
            if dtype not in _BYTES_PER_SAMPLE:
                print(f"  skip {data_path.name}: unsupported datatype {dtype}", file=sys.stderr)
                continue
            bps = _BYTES_PER_SAMPLE[dtype]
            if global_meta is None:
                global_meta = g
                bytes_per_sample = bps
            elif bps != bytes_per_sample:
                print(f"  skip {data_path.name}: datatype {dtype} != first input", file=sys.stderr)
                continue

            caps = meta.get("captures", [{}])
            file_start = int(caps[0].get("core:sample_start", 0)) if caps else 0
            file_len = os.path.getsize(data_path) // bps
            gaps = annotation_free_gaps(file_start, file_len, meta.get("annotations", []),
                                        args.guard_samples, args.min_gap_samples)
            file_noise = 0
            with open(data_path, "rb") as in_fh:
                for rel_start, glen in gaps:
                    in_fh.seek(rel_start * bps)
                    remaining = glen * bps
                    while remaining > 0:
                        chunk = in_fh.read(min(remaining, 8 << 20))
                        if not chunk:
                            break
                        out_fh.write(chunk)
                        remaining -= len(chunk)
                    file_noise += glen
            total_noise_samples += file_noise
            per_file_report.append((data_path.name, len(gaps), file_noise))
            print(f"  {data_path.name}: {len(gaps)} noise gap(s), {file_noise} samples")

    if total_noise_samples == 0:
        print("ERROR: no annotation-free noise samples found across inputs "
              "(try more/lower-SNR files, or reduce --min-gap-samples)", file=sys.stderr)
        out_data.unlink(missing_ok=True)
        return 3

    sample_rate = float(global_meta.get("core:sample_rate", 0.0)) if global_meta else 0.0
    out_meta_obj = {
        "global": {
            "core:datatype": global_meta.get("core:datatype", "cf32_le"),
            "core:sample_rate": global_meta.get("core:sample_rate", 0),
            "core:version": global_meta.get("core:version", "1.0.0"),
            "core:num_channels": global_meta.get("core:num_channels", 1),
            "core:description": "annotation-free noise regions concatenated for DINO positional calibration",
        },
        "captures": [{"core:sample_start": 0, "core:frequency": 0.0}],
        "annotations": [],
    }
    out_meta.write_text(json.dumps(out_meta_obj, indent=2))

    dur_s = total_noise_samples / sample_rate if sample_rate else 0.0
    print(f"\nwrote {out_data}  ({total_noise_samples} noise samples"
          + (f", {dur_s:.3f} s @ {sample_rate/1e6:.2f} MSps" if sample_rate else "") + ")")
    print(f"      {out_meta.name} (0 annotations)")
    print("  NOTE: more noise samples => more detector frames => a more stable per-position "
          "template. If the downstream calibrate step reports too few samples, add more "
          "low-SNR input files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

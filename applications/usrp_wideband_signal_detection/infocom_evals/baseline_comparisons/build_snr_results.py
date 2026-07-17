#!/usr/bin/env python3
"""Join a per-signal SNR calibration onto the eval fact tables and serialize it.

Reads the tidy fact tables produced by ``eval_detector_masks.py`` --
``region_metrics.csv`` (per annotation) and ``frame_pixel_metrics.csv`` (per
frame) -- calibrates SNR from the 0 dB capture (see :mod:`snr_measurement`), and
writes a single reloadable :class:`~snr_measurement.SnrResults` object (``.npz`` +
``.json`` sidecar). Downstream, ``plot_snr_results.py`` / the notebook load that
object and render SNR-axis figures, so the plots can be re-tweaked (new lines,
different bins, restyled) **without** re-reading the raw captures.

What gets joined:
  * region rows gain ``snr_db = snr0_db(class, bw) - attenuation_db``.
  * frame rows gain ``frame_snr_db`` = the mean SNR of the signals present in that
    frame (per the agreed "per-frame mean-signal SNR" x-axis for pixel plots).

Example
-------
    python3 build_snr_results.py \
        --tables-dir ../signal_detection_experiments/batch_runs/<run_id> \
        --captures-dir /home/bqn82/captures \
        --out ../signal_detection_experiments/batch_runs/<run_id>/snr_results
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

import snr_measurement as sm


def _f(x) -> Optional[float]:
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _find_zero_db_capture(captures_dirs: list[Path], stem: str) -> tuple[Path, Path]:
    for d in captures_dirs:
        data = Path(d) / f"{stem}.sigmf-data"
        meta = Path(d) / f"{stem}.sigmf-meta"
        if data.exists() and meta.exists():
            return data, meta
    raise FileNotFoundError(
        f"could not find {stem}.sigmf-data/.sigmf-meta under {', '.join(map(str, captures_dirs))}")


def _col(rows: list[dict], key: str, cast=str, default=None):
    """Extract one column as a python list, casting/blank-handling per cell."""
    out = []
    for r in rows:
        v = r.get(key, "")
        if v == "" or v is None:
            out.append(default)
        else:
            try:
                out.append(cast(v))
            except (TypeError, ValueError):
                out.append(default)
    return out


def _as_array(values: list, kind: str) -> np.ndarray:
    """Column -> ndarray. NaN-fills numeric Nones; empty-strings text Nones."""
    if kind == "float":
        return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    if kind == "bool":
        return np.array([str(v).lower() == "true" for v in values], dtype=bool)
    return np.array(["" if v is None else str(v) for v in values], dtype=object).astype(str)


def build_results(tables_dir: Path, captures_dirs: list[Path], zero_db_stem: str,
                  cfg: sm.SnrConfig) -> sm.SnrResults:
    region_rows = _load_csv(tables_dir / "region_metrics.csv")
    frame_rows = _load_csv(tables_dir / "frame_pixel_metrics.csv")

    data_path, meta_path = _find_zero_db_capture(captures_dirs, zero_db_stem)
    print(f"calibrating SNR from {data_path.name} ...")
    calib = sm.calibrate_from_capture(data_path, meta_path, cfg)
    lookup = sm.calibration_lookup(calib["calibration"])
    print(f"  {len(calib['calibration'])} (class, bw) keys from "
          f"{len(calib['per_signal'])} measured instances")

    # --- region: snr_db per annotation ------------------------------------- #
    # Only 'waveform' annotations are detection targets with a calibrated SNR; the
    # ZC-preamble / metadata annotations get NaN by design (they are not scored on
    # the SNR axis). Unmatched *waveform* rows would be a real problem, so report
    # them separately.
    region_snr: list[Optional[float]] = []
    n_wf_unmatched = 0
    for r in region_rows:
        cls = r.get("wfgt_class") or r.get("label")
        bw = sm._round_bw(r.get("occupied_bw_hz"))
        atten = _f(r.get("attenuation_db"))
        snr0 = lookup.get((cls, bw))
        if snr0 is None or atten is None:
            region_snr.append(None)
            if r.get("kind") == "waveform":
                n_wf_unmatched += 1
        else:
            region_snr.append(sm.snr_at_attenuation(snr0, atten))
    n_waveform = sum(1 for r in region_rows if r.get("kind") == "waveform")
    print(f"  {n_waveform}/{len(region_rows)} region rows are waveform targets; "
          f"{n_wf_unmatched} of those had no calibration match")
    if n_wf_unmatched:
        print("  WARNING: some waveform rows are uncalibrated (check bw/class join)")

    # --- frame: mean SNR of the signals present (de-duplicated per detector) - #
    # Region rows repeat identically across detectors; the signals-present set for a
    # (file_stem, frame_number) is detector-independent, so average each annotation
    # once using a single reference detector's rows.
    ref_detector = region_rows[0]["detector"] if region_rows else None
    frame_snr_acc: dict[tuple, list[float]] = defaultdict(list)
    for r, snr in zip(region_rows, region_snr):
        if snr is None or r.get("detector") != ref_detector:
            continue
        frame_snr_acc[(r.get("file_stem"), r.get("frame_number"))].append(snr)
    frame_snr_map = {k: float(np.mean(v)) for k, v in frame_snr_acc.items()}
    frame_snr = [frame_snr_map.get((r.get("file_stem"), r.get("frame_number")))
                 for r in frame_rows]

    # --- assemble column-oriented arrays ----------------------------------- #
    region = {
        "detector": _as_array(_col(region_rows, "detector"), "str"),
        "file_stem": _as_array(_col(region_rows, "file_stem"), "str"),
        "kind": _as_array(_col(region_rows, "kind"), "str"),
        "frame_number": _as_array(_col(region_rows, "frame_number", int), "float"),
        "attenuation_db": _as_array(_col(region_rows, "attenuation_db", float), "float"),
        "snr_db": _as_array(region_snr, "float"),
        "coverage": _as_array(_col(region_rows, "coverage", float), "float"),
        "box_iou": _as_array(_col(region_rows, "box_iou", float), "float"),
        "detected": _as_array(_col(region_rows, "detected"), "bool"),
        "signal_class": _as_array(_col(region_rows, "bucket_signal_class"), "str"),
        "bandwidth": _as_array(_col(region_rows, "bucket_bandwidth"), "str"),
        "pulse_length": _as_array(_col(region_rows, "bucket_pulse_length"), "str"),
        "occupied_bw_hz": _as_array(_col(region_rows, "occupied_bw_hz", float), "float"),
    }
    frame = {
        "detector": _as_array(_col(frame_rows, "detector"), "str"),
        "file_stem": _as_array(_col(frame_rows, "file_stem"), "str"),
        "frame_number": _as_array(_col(frame_rows, "frame_number", int), "float"),
        "attenuation_db": _as_array(_col(frame_rows, "attenuation_db", float), "float"),
        "frame_snr_db": _as_array(frame_snr, "float"),
        "precision": _as_array(_col(frame_rows, "precision", float), "float"),
        "recall": _as_array(_col(frame_rows, "recall", float), "float"),
        "f1": _as_array(_col(frame_rows, "f1", float), "float"),
        "iou": _as_array(_col(frame_rows, "iou", float), "float"),
        "fp_area_fraction": _as_array(_col(frame_rows, "fp_area_fraction", float), "float"),
        "mask_present": _as_array(_col(frame_rows, "mask_present"), "bool"),
    }

    from dataclasses import asdict
    return sm.SnrResults(
        region=region, frame=frame, calibration=calib["calibration"],
        params={**asdict(cfg),
                "sample_rate_hz": calib["sample_rate_hz"],
                "datatype": calib["datatype"]},
        provenance={
            "tables_dir": str(tables_dir),
            "zero_db_capture": str(data_path),
            "n_region_rows": len(region_rows),
            "n_frame_rows": len(frame_rows),
            "n_waveform_rows": n_waveform,
            "n_waveform_unmatched": n_wf_unmatched,
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", type=Path, required=True,
                    help="Dir with region_metrics.csv + frame_pixel_metrics.csv.")
    ap.add_argument("--captures-dir", action="append", default=[],
                    help="Dir(s) holding the 0 dB *.sigmf-data/.sigmf-meta (repeatable).")
    ap.add_argument("--zero-db-stem", default="attenuation_dB_0",
                    help="Capture stem to calibrate SNR from (default: attenuation_dB_0).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output results base path (default: <tables-dir>/snr_results).")
    # SNR knobs (default to the agreed method; override for sensitivity studies)
    ap.add_argument("--fft-cols", type=int, default=sm.SnrConfig.fft_cols)
    ap.add_argument("--peak-top-fraction", type=float, default=sm.SnrConfig.peak_top_fraction)
    ap.add_argument("--noise-pre-zc-start-ms", type=float, default=sm.SnrConfig.noise_pre_zc_start_ms)
    ap.add_argument("--noise-pre-zc-stop-ms", type=float, default=sm.SnrConfig.noise_pre_zc_stop_ms)
    ap.add_argument("--max-peak-rows", type=int, default=sm.SnrConfig.max_peak_rows)
    ap.add_argument("--max-instances-per-key", type=int, default=sm.SnrConfig.max_instances_per_key)
    args = ap.parse_args()

    captures_dirs = [Path(d) for d in args.captures_dir] or [Path("/home/bqn82/captures")]
    cfg = sm.SnrConfig(
        fft_cols=args.fft_cols, peak_top_fraction=args.peak_top_fraction,
        noise_pre_zc_start_ms=args.noise_pre_zc_start_ms,
        noise_pre_zc_stop_ms=args.noise_pre_zc_stop_ms,
        max_peak_rows=args.max_peak_rows, max_instances_per_key=args.max_instances_per_key,
    )
    results = build_results(args.tables_dir, captures_dirs, args.zero_db_stem, cfg)
    out = args.out or (args.tables_dir / "snr_results")
    paths = results.save(out)
    print(f"\nwrote {paths['npz']}")
    print(f"wrote {paths['json']}")
    print("\ncalibration (median snr0 at 0 dB):")
    for c in results.calibration:
        bw = "?" if c["occupied_bw_hz"] is None else f"{c['occupied_bw_hz']/1e6:.1f}MHz"
        print(f"  {c['wfgt_class']:14s} {bw:>9s}  snr0={c['snr0_db']:6.2f} dB  "
              f"(peak={c['peak_db']:.1f}, noise={c['noise_db']:.1f}, n={c['n_instances']})")
    print("\nNext: plot from the serialized object (no recompute):")
    print(f"  python3 plot_snr_results.py --results {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

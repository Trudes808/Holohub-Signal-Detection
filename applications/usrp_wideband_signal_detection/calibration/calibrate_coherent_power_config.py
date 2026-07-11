#!/usr/bin/env python3
"""Calibrate the coherent-power fast-path thresholds from measured noise (and optionally
signal) statistics, and emit a calibrated config.

Background
----------
The single-channel live fast path decides per pixel (kernel
``coherent_power_fast_power_assist_score_kernel``):

    support_db = corrected_db - background_db          # background = local box-mean
    support    = clip((support_db - fast_power_floor_db) / fast_power_span_db, 0, 1)
    mask       = support >= fast_score_threshold and support > 0

Because ``mask`` fires exactly when ``support_db >= floor + threshold*span``, the
false-positive rate on noise is governed solely by that *effective dB threshold*
``T = floor + threshold*span``; ``floor`` and ``span`` only shape how fast a real
signal ramps from 0 to 1. This script measures the noise ``support_db`` distribution to
place ``T`` at a target false-positive rate, uses the signal ``support_db`` level (if a
signal run is provided) to size ``span`` so a typical signal reaches a target support,
and back-solves ``fast_score_threshold``.

Run the detector over the noise/signal concats with
``save_coherent_power_stats: true`` first (see ``config_coherent_power_calibration_dump_*``
and ``calibrate_coherent_power_config.sh``); this reads the resulting per-frame
``coherent_power_stats_ch*_f*_*_corrected_sxx_db.npy`` + ``_background_db.npy`` dumps.

Usage:
  python3 calibrate_coherent_power_config.py \
      --noise-run-dir /tmp/usrp_spectrograms/coherent_power_cal/noise \
      --signal-run-dir /tmp/usrp_spectrograms/coherent_power_cal/signal \
      --base-config config_coherent_power_performance_single_channel.yaml \
      --output-config config_coherent_power_calibrated_single_channel.yaml
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import re
from pathlib import Path
import sys

import numpy as np

_CONTRACT = "coherent_power_fast_stats_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate coherent-power fast-path thresholds from noise/signal stats.")
    p.add_argument("--noise-run-dir", required=True, type=Path,
                   help="Stats dir from a noise-only run (has coherent_power_stats_*.npy + meta.json).")
    p.add_argument("--signal-run-dir", type=Path, default=None,
                   help="Stats dir from an annotated-signal run. Omit to keep the base span_db "
                        "(only floor/threshold get recalibrated).")
    p.add_argument("--base-config", required=True, type=Path, help="Base config yaml to derive from.")
    p.add_argument("--output-config", required=True, type=Path, help="Calibrated config yaml to write.")
    # Policy knobs (all have defensible defaults; expose so the operator can trade FP vs sensitivity).
    p.add_argument("--noise-false-positive-rate", type=float, default=1e-3,
                   help="Target fraction of NOISE pixels allowed to cross the mask threshold. Sets the "
                        "effective dB threshold T = quantile(noise support_db, 1 - fp) (default 1e-3).")
    p.add_argument("--floor-quantile", type=float, default=0.90,
                   help="Noise support_db quantile mapped to support=0 (fast_power_floor_db). Noise below "
                        "this contributes zero support (default 0.90).")
    p.add_argument("--signal-quantile", type=float, default=0.90,
                   help="Per-frame spatial quantile of support_db in SIGNAL frames taken as the 'typical "
                        "signal level' (default 0.90).")
    p.add_argument("--signal-support-target", type=float, default=0.90,
                   help="Normalized support a typical signal should reach; sizes fast_power_span_db "
                        "(default 0.90). Must exceed the resulting threshold for the signal to be kept.")
    p.add_argument("--span-mode", choices=["signal", "fixed"], default="signal",
                   help="Which fast_power_span_db candidate to WRITE. 'signal' (default) sizes span from "
                        "the measured signal level; 'fixed' uses --operating-range-db. Both candidates are "
                        "always printed and recorded in the sidecar. NOTE: threshold is derived to preserve "
                        "the FP-calibrated boundary T, so both modes yield IDENTICAL detection masks — span "
                        "only reshapes the soft score that feeds grouping (grouping_seed_score_q).")
    p.add_argument("--operating-range-db", type=float, default=40.0,
                   help="Fixed operating zone in dB above the floor for --span-mode fixed: a signal this many "
                        "dB above floor saturates support at 1.0 (default 40).")
    p.add_argument("--threshold-margin-db", type=float, default=0.0,
                   help="Extra dB added to the effective threshold T (raise to reduce false positives).")
    p.add_argument("--min-span-db", type=float, default=1.0,
                   help="Lower clamp for fast_power_span_db to avoid a degenerate hard step (default 1.0).")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Cap frames loaded per run (0 = all). Useful to bound memory on long captures.")
    # Per-frequency noise-floor fill (fills in signal interiors the local box hollows out).
    p.add_argument("--per-freq-fp", type=float, default=0.02,
                   help="False-positive rate for the per-row noise floor: floor[f] = quantile(noise "
                        "corrected_db[f], 1 - per_freq_fp) (default 0.02).")
    p.add_argument("--per-freq-offset-db", type=float, default=2.0,
                   help="dB above the per-row floor required to fire the per-frequency fill (default 2.0). "
                        "Written to the config as per_freq_threshold_offset_db (tunable without recalibrating).")
    p.add_argument("--per-freq-out", type=Path, default=Path("calibration/coherent_power_per_freq_floor.npy"),
                   help="Where to write the per-row floor .npy (host path, default calibration/...).")
    p.add_argument("--per-freq-container-path", type=str,
                   default="/workspace/holohub/applications/usrp_wideband_signal_detection/calibration/coherent_power_per_freq_floor.npy",
                   help="Container path to the per-row floor .npy, written into the config's "
                        "per_freq_threshold_path (must resolve inside the demo container).")
    return p.parse_args()


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path} (run with save_coherent_power_stats: true)")
    meta = json.loads(meta_path.read_text())
    if meta.get("artifact_contract") != _CONTRACT:
        raise ValueError(f"unexpected artifact_contract in {meta_path}: {meta.get('artifact_contract')}")
    return meta


def _frame_pairs(run_dir: Path, max_frames: int):
    corrected = sorted(glob.glob(str(run_dir / "coherent_power_stats_*_corrected_sxx_db.npy")))
    if not corrected:
        raise FileNotFoundError(f"no coherent_power_stats_*_corrected_sxx_db.npy under {run_dir}")
    if max_frames and max_frames > 0:
        corrected = corrected[:max_frames]
    for cpath in corrected:
        bpath = cpath[:-len("_corrected_sxx_db.npy")] + "_background_db.npy"
        if not Path(bpath).is_file():
            print(f"  warn: missing background for {Path(cpath).name}; skipping", file=sys.stderr)
            continue
        yield cpath, bpath


def _support_map(cpath: str, bpath: str, ignore_bins: int) -> np.ndarray:
    """Reconstruct the kernel's per-pixel support_db and return finite valid-row values."""
    corrected = np.load(cpath).astype(np.float64)
    background = np.load(bpath).astype(np.float64)
    if corrected.shape != background.shape:
        raise ValueError(f"shape mismatch {corrected.shape} vs {background.shape} for {cpath}")
    support = corrected - background
    rows = support.shape[0]
    if ignore_bins > 0 and rows > 2 * ignore_bins:
        support = support[ignore_bins:rows - ignore_bins, :]
    support = support[np.isfinite(support)]
    return support


def _load_noise_support(run_dir: Path, ignore_bins: int, max_frames: int) -> np.ndarray:
    chunks = []
    n_frames = 0
    for cpath, bpath in _frame_pairs(run_dir, max_frames):
        chunks.append(_support_map(cpath, bpath, ignore_bins).ravel())
        n_frames += 1
    if not chunks:
        raise RuntimeError(f"no usable noise frames in {run_dir}")
    print(f"  noise: pooled {n_frames} frame(s)")
    return np.concatenate(chunks)


def _load_signal_levels(run_dir: Path, ignore_bins: int, signal_q: float, max_frames: int) -> np.ndarray:
    levels = []
    for cpath, bpath in _frame_pairs(run_dir, max_frames):
        support = _support_map(cpath, bpath, ignore_bins)
        if support.size:
            levels.append(float(np.quantile(support, signal_q)))
    if not levels:
        raise RuntimeError(f"no usable signal frames in {run_dir}")
    print(f"  signal: {len(levels)} frame(s), per-frame q{signal_q:.2f} of support_db")
    return np.asarray(levels)


def _per_freq_floor(run_dir: Path, per_freq_fp: float, max_frames: int) -> np.ndarray:
    """Per-row (per-frequency) noise floor in dB: quantile(corrected_db[row], 1 - per_freq_fp)
    pooled over all noise frames' time bins. Uses absolute corrected_db (not support), so the
    detector can fire on absolute power above the floor regardless of the local box background."""
    cols = []  # each (rows, cols_frame)
    for cpath, _ in _frame_pairs(run_dir, max_frames):
        cols.append(np.load(cpath).astype(np.float32))
    if not cols:
        raise RuntimeError(f"no usable noise frames in {run_dir} for the per-frequency floor")
    rows0 = cols[0].shape[0]
    if any(c.shape[0] != rows0 for c in cols):
        raise ValueError("noise frames disagree on row count; cannot build per-frequency floor")
    stacked = np.concatenate(cols, axis=1)  # (rows, cols_frame * n_frames)
    q = float(np.clip(1.0 - per_freq_fp, 0.0, 1.0))
    floor = np.quantile(stacked, q, axis=1).astype(np.float32)  # (rows,)
    return floor


def _block_bounds(text: str, block_key: str) -> tuple[int, int]:
    """Character bounds of a top-level yaml block ``block_key:`` up to the next top-level key."""
    m = re.search(rf"^{re.escape(block_key)}:\s*$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"block '{block_key}:' not found in base config")
    start = m.start()
    tail = text[m.end():]
    nxt = re.search(r"^\S", tail, re.MULTILINE)
    end = m.end() + nxt.start() if nxt else len(text)
    return start, end


def _set_scalar_in_block(text: str, block_key: str, key: str, value, comment: str) -> str:
    start, end = _block_bounds(text, block_key)
    block = text[start:end]
    if isinstance(value, bool):
        val_str = "true" if value else "false"
    elif isinstance(value, float):
        val_str = f"{value:.6g}"
    elif isinstance(value, str):
        val_str = f'"{value}"'
    else:
        val_str = str(value)
    line_re = re.compile(rf"^(?P<indent>\s+){re.escape(key)}\s*:.*$", re.MULTILINE)

    def repl(mm):
        return f"{mm.group('indent')}{key}: {val_str}  # {comment}"

    new_block, n = line_re.subn(repl, block, count=1)
    if n == 0:
        # Key absent in base block: insert as the first entry after the block header line.
        header_end = block.find("\n") + 1
        indent = re.search(r"^(\s+)\S", block[header_end:], re.MULTILINE)
        pad = indent.group(1) if indent else "  "
        new_block = block[:header_end] + f"{pad}{key}: {val_str}  # {comment}\n" + block[header_end:]
    return text[:start] + new_block + text[end:]


def main() -> int:
    args = parse_args()
    block_key = "coherent_power_signal_detector"

    noise_meta = _load_meta(args.noise_run_dir)
    ignore_bins = int(noise_meta.get("ignore_bins_per_side", 0))
    src_rows = int(noise_meta.get("src_rows", 0))
    src_cols = int(noise_meta.get("src_cols", 0))
    print(f"grid {src_rows}x{src_cols}  ignore_bins_per_side={ignore_bins}")

    noise_support = _load_noise_support(args.noise_run_dir, ignore_bins, args.max_frames)

    # Per-frequency noise floor (absolute corrected_db per row) for the fill path.
    per_freq_floor = _per_freq_floor(args.noise_run_dir, args.per_freq_fp, args.max_frames)
    if per_freq_floor.shape[0] != src_rows:
        raise ValueError(f"per-frequency floor length {per_freq_floor.shape[0]} != src_rows {src_rows}")
    args.per_freq_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.per_freq_out, per_freq_floor)
    valid = per_freq_floor[ignore_bins:src_rows - ignore_bins] if src_rows > 2 * ignore_bins else per_freq_floor
    print(f"  per-freq floor: {src_rows} rows @ {1-args.per_freq_fp:.3g} quantile "
          f"(valid-row median {float(np.median(valid)):.2f} dB, min {float(valid.min()):.2f}, "
          f"max {float(valid.max()):.2f}) -> {args.per_freq_out}")
    print(f"  per-freq fill fires where corrected_db > floor[row] + {args.per_freq_offset_db:g} dB")

    fp = float(np.clip(args.noise_false_positive_rate, 1e-9, 0.5))
    t_db = float(np.quantile(noise_support, 1.0 - fp)) + args.threshold_margin_db
    floor_db = float(np.quantile(noise_support, float(np.clip(args.floor_quantile, 0.0, 1.0))))
    # Keep floor strictly below the detection threshold so support ramps up to it.
    if floor_db >= t_db:
        floor_db = t_db - max(0.5 * args.min_span_db, 1e-3)

    signal_level_db = None
    if args.signal_run_dir is not None:
        signal_meta = _load_meta(args.signal_run_dir)
        if int(signal_meta.get("src_rows", src_rows)) != src_rows or \
           int(signal_meta.get("src_cols", src_cols)) != src_cols:
            raise ValueError("signal run grid differs from noise run grid")
        signal_levels = _load_signal_levels(args.signal_run_dir, ignore_bins, args.signal_quantile,
                                            args.max_frames)
        signal_level_db = float(np.median(signal_levels))

    support_target = float(np.clip(args.signal_support_target, 1e-3, 1.0))
    base_text = args.base_config.read_text()

    # Build candidate span policies. threshold is ALWAYS derived to preserve the
    # FP-calibrated boundary T = floor + threshold*span, so every candidate yields the
    # SAME detection mask (fires at support_db >= T); span only reshapes the soft score.
    def _candidate(raw_span: float) -> dict:
        span = max(raw_span, args.min_span_db)
        thr = float(np.clip((t_db - floor_db) / max(span, 1e-6), 0.0, 1.0))
        sup = None
        if signal_level_db is not None:
            sup = float(np.clip((signal_level_db - floor_db) / max(span, 1e-6), 0.0, 1.0))
        return {"span_db": round(span, 4), "threshold": round(thr, 4), "expected_signal_support": sup}

    candidates: dict[str, dict] = {}
    if signal_level_db is not None and signal_level_db > floor_db:
        candidates["signal"] = _candidate((signal_level_db - floor_db) / support_target)
    candidates["fixed"] = _candidate(args.operating_range_db)

    mode = args.span_mode
    if mode == "signal" and "signal" not in candidates:
        print("  warn: no usable signal level (no signal run, or signal <= floor); "
              "writing --span-mode fixed instead", file=sys.stderr)
        mode = "fixed"
    chosen = candidates[mode]
    span_db = chosen["span_db"]
    threshold = chosen["threshold"]
    expected_signal_support = chosen["expected_signal_support"]

    # Report both candidates.
    print("\n=== calibrated fast-path thresholds ===")
    print(f"  fast_power_floor_db (noise q{args.floor_quantile:.2f})            : {floor_db:.3f} dB"
          f"  (was {noise_meta.get('fast_power_floor_db')})")
    print(f"  effective dB detection boundary T (FP={fp:.3g})     : {t_db:.3f} dB  "
          f"[identical mask for every span mode]")
    if signal_level_db is not None:
        print(f"  signal level (median per-frame q{args.signal_quantile:.2f})       : {signal_level_db:.3f} dB")
    print("  span candidates (span only shapes the soft score feeding grouping_seed_score_q):")
    for name, c in candidates.items():
        sup = f", signal_support={c['expected_signal_support']:.3f}" if c["expected_signal_support"] is not None else ""
        mark = "  <== WRITTEN" if name == mode else ""
        label = f"fixed {args.operating_range_db:g}dB" if name == "fixed" else "signal-derived"
        print(f"    {label:16s}: fast_power_span_db={c['span_db']:.3f}, "
              f"fast_score_threshold={c['threshold']:.4f}{sup}{mark}")
    if expected_signal_support is not None and expected_signal_support < threshold:
        print("  WARNING: expected signal support < threshold — signal too close to noise; "
              "loosen --noise-false-positive-rate or check the signal run.", file=sys.stderr)

    stamp = f"calibrated {datetime.datetime.now().isoformat(timespec='seconds')} span-mode={mode}"
    out_text = base_text
    out_text = _set_scalar_in_block(out_text, block_key, "fast_power_floor_db", round(floor_db, 4), stamp)
    out_text = _set_scalar_in_block(out_text, block_key, "fast_power_span_db", round(span_db, 4), stamp)
    out_text = _set_scalar_in_block(out_text, block_key, "fast_score_threshold", round(threshold, 4), stamp)
    out_text = _set_scalar_in_block(out_text, block_key, "per_freq_threshold_enable", True, stamp)
    out_text = _set_scalar_in_block(out_text, block_key, "per_freq_threshold_path",
                                    args.per_freq_container_path, stamp)
    out_text = _set_scalar_in_block(out_text, block_key, "per_freq_threshold_offset_db",
                                    round(float(args.per_freq_offset_db), 4), stamp)
    args.output_config.write_text(out_text)

    sidecar = {
        "artifact": "coherent_power_fast_threshold_calibration_v1",
        "base_config": str(args.base_config),
        "output_config": str(args.output_config),
        "src_rows": src_rows, "src_cols": src_cols, "ignore_bins_per_side": ignore_bins,
        "noise_run_dir": str(args.noise_run_dir),
        "signal_run_dir": str(args.signal_run_dir) if args.signal_run_dir else None,
        "policy": {
            "noise_false_positive_rate": fp,
            "floor_quantile": args.floor_quantile,
            "signal_quantile": args.signal_quantile,
            "signal_support_target": support_target,
            "threshold_margin_db": args.threshold_margin_db,
            "min_span_db": args.min_span_db,
            "span_mode": mode,
            "operating_range_db": args.operating_range_db,
            "per_freq_fp": args.per_freq_fp,
            "per_freq_offset_db": args.per_freq_offset_db,
        },
        "per_freq_floor": {
            "path": str(args.per_freq_out),
            "container_path": args.per_freq_container_path,
            "rows": int(per_freq_floor.shape[0]),
            "quantile": float(1.0 - args.per_freq_fp),
            "offset_db": float(args.per_freq_offset_db),
            "valid_row_median_db": float(np.median(valid)),
            "valid_row_min_db": float(valid.min()),
            "valid_row_max_db": float(valid.max()),
        },
        "span_candidates": candidates,
        "measured": {
            "noise_support_db_q": {
                "p50": float(np.quantile(noise_support, 0.50)),
                "p90": float(np.quantile(noise_support, 0.90)),
                "p99": float(np.quantile(noise_support, 0.99)),
                f"p{100*(1-fp):.4g}": t_db - args.threshold_margin_db,
            },
            "effective_threshold_db": t_db,
            "signal_level_db": signal_level_db,
            "expected_signal_support": expected_signal_support,
        },
        "calibrated": {
            "fast_power_floor_db": round(floor_db, 4),
            "fast_power_span_db": round(span_db, 4),
            "fast_score_threshold": round(threshold, 4),
        },
        "previous": {
            "fast_power_floor_db": noise_meta.get("fast_power_floor_db"),
            "fast_power_span_db": noise_meta.get("fast_power_span_db"),
            "fast_score_threshold": noise_meta.get("fast_score_threshold"),
        },
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    args.output_config.with_suffix(".calibration.json").write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {args.output_config}  (+ sidecar {args.output_config.with_suffix('.calibration.json').name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

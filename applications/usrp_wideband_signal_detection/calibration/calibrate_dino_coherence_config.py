#!/usr/bin/env python3
"""Stage 1 of the CUDA DINO coherence-gate calibration: measure, from noise regions, the
per-frequency inputs that Stage 2 will feed into the (unchanged) coherence method.

The DINO detector's coherence gate computes, per chunk, a local-box-mean residual of the
frontend-corrected power, normalizes by a fixed 40 dB span, adds a directional term, and
thresholds the result (`coherence_band_threshold`, a single static 0.05). This script
measures two per-frequency arrays from a noise-only run dumped with
`save_coherence_stats: true`:

  1. per-frequency POWER floor (dB) = quantile(noise corrected_db[freq], 1 - power_fp)
     -> Stage 2 caps the gate's box-mean background at this floor so strong broad signals
        (whose box-mean interior rides up) fill in instead of hollowing out.
  2. per-frequency GATE threshold (gate units) = quantile(noise gate[freq], 1 - gate_fp)
     -> Stage 2 replaces the static coherence_band_threshold with this per-row vector.

This stage only WRITES the two arrays + a sidecar + optional diagnostic plots for
inspection. It does not modify any config (the consuming operator params land in Stage 2).

Usage:
  python3 calibrate_dino_coherence_config.py \
      --stats-run-dir /tmp/usrp_spectrograms/dino_coherence_cal/noise \
      --power-floor-out calibration/dino_coherence_per_freq_power_floor.npy \
      --gate-threshold-out calibration/dino_coherence_per_freq_gate_threshold.npy \
      --plots-dir calibration/diagnostics
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False

_CONTRACT = "cuda_dino_coherence_stats_v1"
# Gate max is ~0.11; use 1.0 so uncovered/edge rows never fire (they are also masked out).
_GATE_SENTINEL = 1.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate DINO per-frequency coherence floor + gate threshold from noise.")
    p.add_argument("--stats-run-dir", required=True, type=Path,
                   help="Dir with coherence_stats_*.npy + meta.json from a noise run (save_coherence_stats: true).")
    p.add_argument("--power-floor-fp", type=float, default=0.02,
                   help="False-positive rate for the per-freq power floor: quantile(corrected_db[f], 1-fp) (default 0.02).")
    p.add_argument("--gate-fp", type=float, default=0.02,
                   help="False-positive rate for the per-freq gate threshold: quantile(gate[f], 1-fp) (default 0.02).")
    p.add_argument("--power-floor-out", type=Path,
                   default=Path("calibration/dino_coherence_per_freq_power_floor.npy"),
                   help="Output .npy for the per-frequency power floor (dB, length src_rows).")
    p.add_argument("--gate-threshold-out", type=Path,
                   default=Path("calibration/dino_coherence_per_freq_gate_threshold.npy"),
                   help="Output .npy for the per-frequency gate threshold (gate units, length src_rows).")
    p.add_argument("--max-frames", type=int, default=0, help="Cap frames loaded (0 = all).")
    p.add_argument("--plots-dir", type=Path, default=None, help="Optional diagnostics output dir.")
    # Optional Stage-2 config emission (enables the per-frequency consumption in the operator).
    p.add_argument("--base-config", type=Path, default=None,
                   help="If set (with --output-config), write a calibrated config enabling the per-freq gate.")
    p.add_argument("--output-config", type=Path, default=None, help="Calibrated config path to write.")
    p.add_argument("--power-floor-container-path", type=str,
                   default="/workspace/holohub/applications/usrp_wideband_signal_detection/calibration/dino_coherence_per_freq_power_floor.npy",
                   help="Container path to the power-floor .npy written into the config.")
    p.add_argument("--gate-threshold-container-path", type=str,
                   default="/workspace/holohub/applications/usrp_wideband_signal_detection/calibration/dino_coherence_per_freq_gate_threshold.npy",
                   help="Container path to the gate-threshold .npy written into the config.")
    p.add_argument("--coherence-band-threshold", type=float, default=0.0,
                   help="Static coherence_band_threshold floor written to the calibrated config; the per-row "
                        "threshold is max'd with this. Default 0.0 lets the per-frequency values govern fully.")
    p.add_argument("--floor-offset-db", type=float, default=0.0,
                   help="per_freq_floor_offset_db written to the config (dB added to the floor before capping).")
    return p.parse_args()


def _block_bounds(text: str, block_key: str) -> tuple[int, int]:
    m = re.search(rf"^{re.escape(block_key)}:\s*$", text, re.MULTILINE)
    if not m:
        raise ValueError(f"block '{block_key}:' not found in base config")
    start = m.start()
    tail = text[m.end():]
    nxt = re.search(r"^\S", tail, re.MULTILINE)
    return start, (m.end() + nxt.start() if nxt else len(text))


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
    new_block, n = line_re.subn(lambda mm: f"{mm.group('indent')}{key}: {val_str}  # {comment}", block, count=1)
    if n == 0:
        header_end = block.find("\n") + 1
        indent = re.search(r"^(\s+)\S", block[header_end:], re.MULTILINE)
        pad = indent.group(1) if indent else "  "
        new_block = block[:header_end] + f"{pad}{key}: {val_str}  # {comment}\n" + block[header_end:]
    return text[:start] + new_block + text[end:]


def _load_meta(run_dir: Path) -> dict:
    meta_path = run_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path} (run with save_coherence_stats: true)")
    meta = json.loads(meta_path.read_text())
    if meta.get("artifact_contract") != _CONTRACT:
        raise ValueError(f"unexpected artifact_contract in {meta_path}: {meta.get('artifact_contract')}")
    return meta


def _frames(run_dir: Path, suffix: str, max_frames: int) -> list[str]:
    files = sorted(glob.glob(str(run_dir / f"coherence_stats_*_{suffix}.npy")))
    if not files:
        raise FileNotFoundError(f"no coherence_stats_*_{suffix}.npy under {run_dir}")
    return files[:max_frames] if (max_frames and max_frames > 0) else files


def _per_freq_power_floor(run_dir: Path, src_rows: int, src_cols: int, fp: float, max_frames: int) -> np.ndarray:
    """quantile(corrected_db[row], 1-fp) pooled over all noise frames' time bins. Length src_rows."""
    per_row_cols = []  # each (src_rows, src_cols)
    for f in _frames(run_dir, "corrected_db", max_frames):
        arr = np.load(f).astype(np.float32)
        if arr.shape != (src_rows, src_cols):
            raise ValueError(f"{f}: shape {arr.shape} != expected {(src_rows, src_cols)}")
        per_row_cols.append(arr)
    stacked = np.concatenate(per_row_cols, axis=1)  # (src_rows, src_cols*n_frames)
    return np.quantile(stacked, float(np.clip(1.0 - fp, 0.0, 1.0)), axis=1).astype(np.float32)


def _per_freq_gate_threshold(run_dir: Path, meta: dict, fp: float, max_frames: int):
    """quantile(gate[global_freq], 1-fp) per frequency, mapping packed chunk rows to global freq
    via chunk_row_starts. Returns (threshold[src_rows], covered_mask[src_rows])."""
    src_rows = int(meta["src_rows"])
    src_cols = int(meta["src_cols"])
    chunk_count = int(meta["chunk_count"])
    uniform_chunk_rows = int(meta["uniform_chunk_rows"])
    starts = list(meta["chunk_row_starts"])
    packed_rows = chunk_count * uniform_chunk_rows

    # global freq -> packed row (chunks are contiguous, non-overlapping => unique).
    global_to_packed = np.full(src_rows, -1, dtype=np.int64)
    for c in range(chunk_count):
        for r in range(uniform_chunk_rows):
            g = starts[c] + r
            if 0 <= g < src_rows:
                global_to_packed[g] = c * uniform_chunk_rows + r
    covered = np.where(global_to_packed >= 0)[0]
    packed_idx = global_to_packed[covered]

    per_frame = []  # each (len(covered), src_cols)
    for f in _frames(run_dir, "coherence_gate", max_frames):
        arr = np.load(f).astype(np.float32)
        if arr.shape != (packed_rows, src_cols):
            raise ValueError(f"{f}: shape {arr.shape} != expected {(packed_rows, src_cols)}")
        per_frame.append(arr[packed_idx, :])
    stacked = np.concatenate(per_frame, axis=1)  # (len(covered), src_cols*n_frames)
    thr_covered = np.quantile(stacked, float(np.clip(1.0 - fp, 0.0, 1.0)), axis=1).astype(np.float32)

    threshold = np.full(src_rows, _GATE_SENTINEL, dtype=np.float32)
    threshold[covered] = thr_covered
    covered_mask = np.zeros(src_rows, dtype=bool)
    covered_mask[covered] = True
    return threshold, covered_mask


def main() -> int:
    args = parse_args()
    meta = _load_meta(args.stats_run_dir)
    src_rows = int(meta["src_rows"])
    src_cols = int(meta["src_cols"])
    print(f"grid src_rows(freq)={src_rows} src_cols(time)={src_cols} "
          f"chunk_count={meta['chunk_count']} uniform_chunk_rows={meta['uniform_chunk_rows']} "
          f"gate_normalize_span_db={meta.get('gate_normalize_span_db')} "
          f"current coherence_band_threshold={meta.get('coherence_band_threshold')}")

    power_floor = _per_freq_power_floor(args.stats_run_dir, src_rows, src_cols, args.power_floor_fp, args.max_frames)
    gate_threshold, covered = _per_freq_gate_threshold(args.stats_run_dir, meta, args.gate_fp, args.max_frames)

    args.power_floor_out.parent.mkdir(parents=True, exist_ok=True)
    args.gate_threshold_out.parent.mkdir(parents=True, exist_ok=True)
    # Saved as (src_rows, 1) so the operator's 2D .npy reader (read_npy_2d_float) accepts them.
    np.save(args.power_floor_out, power_floor.reshape(-1, 1))
    np.save(args.gate_threshold_out, gate_threshold.reshape(-1, 1))

    cov_floor = power_floor[covered]
    cov_gate = gate_threshold[covered]
    print(f"\n=== per-frequency power floor (dB, q{1-args.power_floor_fp:.3g}) ===")
    print(f"  covered rows {int(covered.sum())}/{src_rows}: median {np.median(cov_floor):.2f}  "
          f"min {cov_floor.min():.2f}  max {cov_floor.max():.2f}  (DC/band-edge spikes show as high rows)")
    print(f"=== per-frequency gate threshold (gate units, q{1-args.gate_fp:.3g}) ===")
    print(f"  covered rows: median {np.median(cov_gate):.4f}  min {cov_gate.min():.4f}  max {cov_gate.max():.4f}  "
          f"(vs current static coherence_band_threshold={meta.get('coherence_band_threshold')})")

    sidecar = {
        "artifact": "cuda_dino_coherence_calibration_v1",
        "stats_run_dir": str(args.stats_run_dir),
        "src_rows": src_rows, "src_cols": src_cols,
        "chunk_count": int(meta["chunk_count"]), "uniform_chunk_rows": int(meta["uniform_chunk_rows"]),
        "gate_normalize_span_db": meta.get("gate_normalize_span_db"),
        "current_coherence_band_threshold": meta.get("coherence_band_threshold"),
        "policy": {"power_floor_fp": args.power_floor_fp, "gate_fp": args.gate_fp},
        "power_floor": {
            "path": str(args.power_floor_out), "quantile": float(1.0 - args.power_floor_fp),
            "covered_median_db": float(np.median(cov_floor)),
            "covered_min_db": float(cov_floor.min()), "covered_max_db": float(cov_floor.max()),
        },
        "gate_threshold": {
            "path": str(args.gate_threshold_out), "quantile": float(1.0 - args.gate_fp),
            "covered_rows": int(covered.sum()), "sentinel_value": _GATE_SENTINEL,
            "covered_median": float(np.median(cov_gate)),
            "covered_min": float(cov_gate.min()), "covered_max": float(cov_gate.max()),
        },
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    sidecar_path = args.power_floor_out.with_suffix(".calibration.json")
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"\nwrote {args.power_floor_out}\n      {args.gate_threshold_out}\n      {sidecar_path}")

    if args.base_config is not None and args.output_config is not None:
        block = "cuda_dino_detector"
        stamp = f"dino coherence calibrated {datetime.datetime.now().isoformat(timespec='seconds')}"
        text = args.base_config.read_text()
        text = _set_scalar_in_block(text, block, "per_freq_floor_enable", True, stamp)
        text = _set_scalar_in_block(text, block, "per_freq_floor_path", args.power_floor_container_path, stamp)
        text = _set_scalar_in_block(text, block, "per_freq_floor_offset_db", round(float(args.floor_offset_db), 4), stamp)
        text = _set_scalar_in_block(text, block, "per_freq_gate_threshold_enable", True, stamp)
        text = _set_scalar_in_block(text, block, "per_freq_gate_threshold_path", args.gate_threshold_container_path, stamp)
        text = _set_scalar_in_block(text, block, "coherence_band_threshold", round(float(args.coherence_band_threshold), 6), stamp)
        args.output_config.write_text(text)
        print(f"      {args.output_config}  (per-freq gate enabled; coherence_band_threshold floor -> {args.coherence_band_threshold})")
    else:
        print("Stage 1 mode — no config written (pass --base-config/--output-config to emit a calibrated config).")

    if args.plots_dir is not None and _HAVE_MPL:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        freq = np.arange(src_rows)
        fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        ax[0].plot(freq[covered], power_floor[covered], lw=0.7)
        ax[0].set_ylabel("power floor (dB)")
        ax[0].set_title(f"DINO per-frequency noise calibration (q_power={1-args.power_floor_fp:.3g}, q_gate={1-args.gate_fp:.3g})")
        ax[1].plot(freq[covered], gate_threshold[covered], lw=0.7, color="tab:orange")
        ax[1].axhline(float(meta.get("coherence_band_threshold", 0.05)), color="gray", ls="--", lw=0.8,
                      label=f"current static {meta.get('coherence_band_threshold')}")
        ax[1].set_ylabel("gate threshold")
        ax[1].set_xlabel("frequency bin")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(args.plots_dir / "dino_coherence_per_freq.png", dpi=150)
        plt.close(fig)
        print(f"  [plots] wrote {args.plots_dir / 'dino_coherence_per_freq.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the non-ML baseline detectors offline over an existing batch-eval tree.

The trained detectors (``coherent_power`` / ``cuda_dino``) are evaluated by the
C++ binary ``run_offline_cuda_detector_eval``, which writes, per capture, a run
directory holding the ground truth + per-frame spectrogram tensors + a manifest:

    <batch_root>/<detector>/<file_stem>/
        frame_manifest.csv
        gt_masks/            ground_truth_mask_ch<c>_f<f>_<H>x<W>.npy
        gt_annotations/      ground_truth_ch<c>_f<f>_<H>x<W>.json
        spectrogram_tensors/ spectrogram_tensor_ch<c>_f<f>_<H>x<W>.npy   (if --save-tensors)
        mask_arrays/         mask_ch<c>_f<f>_<H>x<W>.npy

This driver reuses that ground truth and those spectrograms (or reconstructs the
spectrogram from the source SigMF on the identical FFT grid when tensors were not
saved) and produces the *same* run-directory layout for each Python baseline
detector, writing it as a sibling detector directory in the same batch root:

    <batch_root>/3dB_power/<file_stem>/ ...
    <batch_root>/blob_detection/<file_stem>/ ...

So a single ``eval_detector_masks.py --batch-root <batch_root>`` afterwards scores
the baselines next to the trained detectors, and the notebook / plots compare them
directly — the whole pipeline downstream is detector-agnostic.

Example
-------
    python3 run_baseline_offline.py \
        --config baseline_detectors_config.yaml \
        --source-batch-root ../batch_runs/<run_id>

Everything (which detectors, their parameters, the batch root, the reference
detector to read GT/tensors from) can live in the YAML; CLI flags override it.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# The shared eval helpers live one directory up (signal_detection_experiments/).
_THIS_DIR = Path(__file__).resolve().parent
_PARENT = _THIS_DIR.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import eval_viz as ev            # noqa: E402  spectrogram load/reconstruct helpers
import mask_eval_metrics as mem  # noqa: E402  manifest loader (tensor/SigMF agnostic)
import baseline_detectors as bd  # noqa: E402  the two baseline algorithms


def _load_config(path: Optional[Path]) -> dict:
    if not path:
        return {}
    import yaml
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    return cfg.get("baseline_eval", cfg)


def _link_or_copy_dir(src: Path, dst: Path, copy: bool) -> None:
    """Make ``dst`` mirror ``src`` (a shared GT/tensor dir) via symlink or copy."""
    if dst.exists() or dst.is_symlink():
        return
    if not src.exists():
        return
    if copy:
        shutil.copytree(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=True)


def _discover_source_runs(batch_root: Path, source_detector: Optional[str]) -> tuple[str, list[str]]:
    """Return (source_detector, [file_stem, ...]) from an existing batch root.

    The source detector is any completed detector directory that carries the GT +
    (optionally) spectrogram tensors; defaults to the first one found.
    """
    detector_dirs = sorted(p for p in batch_root.iterdir()
                           if p.is_dir() and p.name not in bd.DETECTORS)
    if source_detector:
        chosen = batch_root / source_detector
        if not chosen.is_dir():
            raise FileNotFoundError(f"source detector dir not found: {chosen}")
    else:
        if not detector_dirs:
            raise FileNotFoundError(
                f"no source detector directory (with GT) found under {batch_root}")
        chosen = detector_dirs[0]
    stems = sorted(p.name for p in chosen.iterdir()
                   if p.is_dir() and ((p / "frame_manifest.csv").exists()
                                      or (p / "gt_annotations").is_dir()))
    return chosen.name, stems


def _mask_rel_path(row: dict, channel: int, frame: int, rows: int, cols: int) -> str:
    """Relative mask path from the manifest, or the standard builder as a fallback.

    The C++ builder names masks ``mask_ch<c>_f<f>_<H>x<W>.npy`` with no detector in
    the name, so reusing the source manifest's ``mask_npy`` column is correct — the
    baseline writes the identically named file in its own ``mask_arrays/``.
    """
    rel = row.get("mask_npy")
    if rel:
        # strip any .packed.npz variant back to a plain .npy we will write
        rel = rel.replace(".packed.npz", ".npy")
        return rel
    return f"mask_arrays/mask_ch{channel}_f{frame}_{rows}x{cols}.npy"


def run_one_detector(detector_type: str,
                     source_run_dir: Path,
                     out_run_dir: Path,
                     params: dict,
                     capture_dirs: Optional[list[Path]],
                     file_stem: str,
                     copy_shared: bool,
                     frame_limit: Optional[int]) -> dict:
    """Produce one baseline detector's run directory for one capture."""
    out_run_dir.mkdir(parents=True, exist_ok=True)
    (out_run_dir / "mask_arrays").mkdir(exist_ok=True)

    # Reuse the GT + tensors from the source run (shared, immutable) via symlink.
    for shared in ("gt_masks", "gt_annotations", "spectrogram_tensors"):
        _link_or_copy_dir(source_run_dir / shared, out_run_dir / shared, copy_shared)
    for small in ("frame_manifest.csv", "offline_eval_summary.json"):
        src = source_run_dir / small
        if src.exists():
            shutil.copy2(src, out_run_dir / small)

    manifest = mem.load_manifest(source_run_dir)
    n_frames = 0
    n_on_pixels = 0
    rows = cols = 0
    for row in manifest:
        frame = int(row["frame_number"])
        if frame_limit is not None and frame > frame_limit:
            continue
        rows = int(row.get("fft_rows") or 0)
        cols = int(row.get("fft_cols") or 0)
        channel = int(row.get("channel") or 0)
        try:
            spec_db = ev._load_or_reconstruct_spectrogram(
                source_run_dir, row, file_stem, rows, cols, capture_dirs)
        except FileNotFoundError as exc:
            print(f"  [skip] frame {frame}: {exc}")
            continue

        mask = bd.run_detector(detector_type, spec_db, params)
        if mask.shape != (rows, cols):
            mask = mem.resize_mask_nearest(mask, rows, cols)

        rel = _mask_rel_path(row, channel, frame, mask.shape[0], mask.shape[1])
        out_path = out_run_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, mask.astype(np.uint8))
        n_frames += 1
        n_on_pixels += int(mask.sum())

    return {"frames": n_frames, "mean_on_fraction":
            (n_on_pixels / (n_frames * rows * cols)) if n_frames and rows and cols else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=_THIS_DIR / "baseline_detectors_config.yaml",
                    help="YAML with a baseline_eval: block (detectors, params, paths).")
    ap.add_argument("--source-batch-root", type=Path, default=None,
                    help="Existing batch-eval root (<root>/<detector>/<stem>/). "
                         "Overrides source_batch_root in the config.")
    ap.add_argument("--out-batch-root", type=Path, default=None,
                    help="Where baseline detector dirs are written "
                         "(default: same as the source batch root, as siblings).")
    ap.add_argument("--detectors", nargs="+", default=None,
                    help="Subset of baseline detectors to run (default: from config).")
    ap.add_argument("--source-detector", default=None,
                    help="Detector dir to read GT/tensors from (default: first found).")
    ap.add_argument("--captures-dir", action="append", default=[],
                    help="Extra dir(s) with source *.sigmf-data for spectrogram "
                         "reconstruction when tensors were not saved (repeatable).")
    ap.add_argument("--copy-shared", action="store_true",
                    help="Copy GT/tensor dirs instead of symlinking them.")
    ap.add_argument("--frame-limit", type=int, default=None, help="Debug: first N frames.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    source_batch_root = Path(args.source_batch_root or cfg.get("source_batch_root"))
    if not source_batch_root or not source_batch_root.is_dir():
        ap.error(f"source batch root not found: {source_batch_root} "
                 "(set --source-batch-root or baseline_eval.source_batch_root)")
    out_batch_root = Path(args.out_batch_root or cfg.get("out_batch_root") or source_batch_root)
    out_batch_root.mkdir(parents=True, exist_ok=True)

    detectors_cfg = cfg.get("detectors", {}) or {}
    requested = args.detectors or list(detectors_cfg.keys()) or list(bd.DETECTORS.keys())
    for det in requested:
        if det not in bd.DETECTORS:
            ap.error(f"unknown baseline detector {det!r}; choices: {sorted(bd.DETECTORS)}")

    capture_dirs = [Path(d) for d in args.captures_dir] or None
    source_detector, stems = _discover_source_runs(source_batch_root, args.source_detector)
    print(f"source batch root : {source_batch_root}")
    print(f"source detector   : {source_detector}  ({len(stems)} captures)")
    print(f"out batch root    : {out_batch_root}")
    print(f"baseline detectors: {requested}\n")

    t0 = time.time()
    summary: list[dict] = []
    for det in requested:
        params = dict(detectors_cfg.get(det, {}) or {})
        for stem in stems:
            src_run = source_batch_root / source_detector / stem
            out_run = out_batch_root / det / stem
            res = run_one_detector(det, src_run, out_run, params, capture_dirs,
                                   stem, args.copy_shared, args.frame_limit)
            summary.append({"detector": det, "file_stem": stem, **res})
            print(f"[{det}/{stem}] frames={res['frames']} "
                  f"mean_on={res['mean_on_fraction']*100:.3f}%")

    out_summary = out_batch_root / "baseline_run_summary.json"
    out_summary.write_text(json.dumps({
        "source_batch_root": str(source_batch_root),
        "source_detector": source_detector,
        "out_batch_root": str(out_batch_root),
        "detectors": requested,
        "params": {d: detectors_cfg.get(d, {}) for d in requested},
        "elapsed_s": round(time.time() - t0, 1),
        "runs": summary,
    }, indent=2))
    print(f"\nWrote {len(summary)} baseline runs in {time.time() - t0:.1f}s")
    print(f"Summary: {out_summary}")
    print("\nNext: score everything (baselines + trained detectors) together:")
    print(f"  cd {_PARENT}")
    print(f"  python3 eval_detector_masks.py --batch-root {out_batch_root} "
          f"--out-dir {out_batch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

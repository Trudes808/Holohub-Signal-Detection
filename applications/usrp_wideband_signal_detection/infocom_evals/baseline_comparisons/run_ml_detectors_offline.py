#!/usr/bin/env python3
"""Materialize the ML detectors (fine-tuned YOLO26 + fine-tuned DINOv3) into a
batch-eval-format run tree, so the SAME ``eval_detector_masks.py`` scores them
identically to the deployed detectors (``coherent_power`` / ``cuda_dino``) and the
non-ML baselines (``3dB_power`` / ``blob_detection``).

This is the ML counterpart of ``run_baseline_offline.py``. For each requested ML
detector and each capture it writes a sibling detector directory in the batch root:

    <batch_root>/yolo/<file_stem>/mask_arrays/mask_ch0_f{N}_{R}x{C}.packed.npz
    <batch_root>/dino_finetuned/<file_stem>/ ...

reusing the reference detector's ``frame_manifest.csv`` + GT (symlinked — GT is
detector-independent), reading each frame's IQ from the source SigMF, running the
model on its own native FFT geometry (nfft=1024, 256-row tiles), then max-pooling
the native mask onto the batch display/GT grid (``to_display_grid``) exactly as the
colleague's per-model ``gen_yolo_run.py`` / ``gen_finetuned_run.py`` do.

Why this exists instead of the colleague's gen_*_run.py
-------------------------------------------------------
Those scripts hardcode ``/home/bqn82/Holohub-Signal-Detection`` on ``sys.path`` for
the *shared* eval modules (``eval_viz`` / ``mask_eval_metrics``), so a run from this
branch would score the ML detectors with a *different* checkout's code than the
baselines. This driver instead resolves the shared eval + model modules from THIS
branch's checkout and pre-imports ``rfdata`` / ``model`` so the vendored classes
reuse them — one identical code path for all six detectors, which is the whole
point of a fair comparison. Only the trained artifacts (weights, dB calibration,
dinov3 backbone) still live wherever you trained them; point at them via the
config / CLI (defaults match the colleague's ``/home/bqn82`` tree).

Example
-------
    python3 run_ml_detectors_offline.py \
        --config comparison_config.yaml \
        --batch-root ../signal_detection_experiments/batch_runs/<run_id>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# --------------------------------------------------------------------------- #
# Path wiring: resolve every module from THIS branch, then pre-import rfdata /
# model so the vendored detector classes reuse the branch copies (Python caches
# modules by name, so a later ``import rfdata`` inside yolo_infer/finetuned_infer
# picks up the already-cached branch module regardless of their hardcoded inserts).
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_INFOCOM = _THIS_DIR.parent                                   # infocom_evals/
_HELPERS = _INFOCOM / "signal_detection_experiments"          # eval_viz, mask_eval_metrics
_REPO_ROOT = _THIS_DIR.parents[3]                             # holohub-dev/
_DINO_SRC = _REPO_ROOT / "dino_fine_tuning" / "src"           # rfdata, model, finetuned_infer
_YOLO_SRC = _REPO_ROOT / "yolo_training" / "src"              # yolo_infer

DEFAULT_DINOV3_REPO = "/home/bqn82/dinov3"


def _wire_syspath(dinov3_repo: Optional[str]) -> None:
    """Put the branch's shared/model dirs (and the dinov3 repo) on sys.path first."""
    for p in (str(_HELPERS), str(_DINO_SRC), str(_YOLO_SRC)):
        if p not in sys.path:
            sys.path.insert(0, p)
    repo = dinov3_repo or DEFAULT_DINOV3_REPO
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _load_config(path: Optional[Path]) -> dict:
    if not path:
        return {}
    import yaml
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    return cfg.get("comparison_eval", cfg)


def _save_packed(path: Path, mask: np.ndarray) -> None:
    """Write a packbits ``.packed.npz`` mask (same format the gen_*_run.py use)."""
    rows, cols = mask.shape
    np.savez_compressed(path, packed=np.packbits(mask.astype(np.uint8).ravel()),
                        rows=rows, cols=cols)


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink() or not src.exists():
        return
    os.symlink(src.resolve(), dst)


# --------------------------------------------------------------------------- #
# Detector construction (lazy — only imports torch/ultralytics/dinov3 when run)
# --------------------------------------------------------------------------- #
def _build_detector(name: str, spec: dict):
    """Return an object exposing ``.mask_for_iq(iq)->(rows,nfft) uint8`` and ``.nfft``.

    ``spec['kind']`` selects the family: ``yolo`` (Ultralytics box->mask) or
    ``dino_finetuned`` (fine-tuned DINOv3 segmenter). All paths come from the spec.
    """
    kind = spec.get("kind", name)
    ds_meta = json.loads(Path(spec["dataset_meta"]).read_text())

    if kind == "yolo":
        import rfdata  # noqa: F401  pre-cache the branch copy
        from yolo_infer import YoloDetector
        det = YoloDetector(spec["ckpt"], ds_meta, device=spec.get("device", "cuda"),
                           conf=float(spec.get("conf", 0.25)),
                           imgsz=int(spec.get("imgsz", 1024)), name=name)
        print(f"  [{name}] yolo ckpt={spec['ckpt']} conf={det.conf} "
              f"vmin/vmax={det.vmin:.1f}/{det.vmax:.1f}")
        return det

    if kind == "dino_finetuned":
        import yaml
        import rfdata  # noqa: F401  pre-cache the branch copy
        import model   # noqa: F401  pre-cache the branch copy
        import finetuned_infer as fi
        train_cfg = yaml.safe_load(Path(spec["train_cfg"]).read_text())
        thr = fi.load_threshold(spec["eval_meta"]) if spec.get("eval_meta") else spec.get("threshold")
        det = fi.FinetunedDetector(spec["ckpt"], train_cfg, ds_meta,
                                   device=spec.get("device", "cuda"), threshold=thr)
        det.name = name
        print(f"  [{name}] dino_finetuned ckpt={spec['ckpt']} threshold={det.threshold:.2f}")
        return det

    raise ValueError(f"unknown ML detector kind {kind!r} for {name!r} "
                     "(expected 'yolo' or 'dino_finetuned')")


# --------------------------------------------------------------------------- #
# Per-capture materialization
# --------------------------------------------------------------------------- #
def run_capture(name: str, det, to_display_grid, ev,
                src_run: Path, dst_run: Path, capture_path: Path,
                frame_limit: Optional[int]) -> dict:
    """Run one ML detector over one capture; write masks in batch-eval layout."""
    (dst_run / "mask_arrays").mkdir(parents=True, exist_ok=True)
    for shared in ("frame_manifest.csv", "gt_masks", "gt_annotations"):
        _link(src_run / shared, dst_run / shared)

    manifest = list(csv.DictReader(open(src_run / "frame_manifest.csv")))
    made = skipped = empty = 0
    for row in manifest:
        fnum = int(row["frame_number"])
        if frame_limit is not None and fnum > frame_limit:
            continue
        rows, cols = int(row["fft_rows"]), int(row["fft_cols"])
        nsamp = int(row.get("complex_samples_read") or 0)
        out_npz = dst_run / "mask_arrays" / f"mask_ch0_f{fnum}_{rows}x{cols}.packed.npz"
        if out_npz.exists():
            skipped += 1
            continue
        if nsamp < det.nfft:
            empty += 1
            continue
        iq = ev.read_frame_iq(capture_path, int(row["local_file_offset_complex"]), nsamp)
        mask = to_display_grid(det.mask_for_iq(iq), rows, cols)
        _save_packed(dst_run / "mask_arrays" / f"mask_ch0_f{fnum}_{rows}x{cols}.packed", mask)
        made += 1
    return {"frames": len(manifest), "made": made, "skipped": skipped, "too_short": empty}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=_THIS_DIR / "comparison_config.yaml",
                    help="YAML with a comparison_eval: block (ml_detectors, paths).")
    ap.add_argument("--batch-root", type=Path, default=None,
                    help="Batch-eval root <root>/<detector>/<stem>/ (overrides config).")
    ap.add_argument("--captures-dir", action="append", default=[],
                    help="Dir(s) with source *.sigmf-data (repeatable; overrides config).")
    ap.add_argument("--ref-detector", default=None,
                    help="Detector dir to mirror GT/manifest from (default from config / cuda_dino).")
    ap.add_argument("--detectors", nargs="+", default=None,
                    help="Subset of ML detectors to run (default: all in the config).")
    ap.add_argument("--dinov3-repo", default=None,
                    help="Path to the dinov3 repo for the DINO backbone (default from config).")
    ap.add_argument("--stems", nargs="+", default=None, help="Restrict to these capture stems.")
    ap.add_argument("--frame-limit", type=int, default=None, help="Debug: first N frames.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    batch_root = Path(args.batch_root or cfg.get("batch_root") or "")
    if not batch_root or not batch_root.is_dir():
        ap.error(f"batch root not found: {batch_root} (set --batch-root or comparison_eval.batch_root)")
    ref_detector = args.ref_detector or cfg.get("ref_detector") or "cuda_dino"
    capture_dirs = [Path(d) for d in args.captures_dir] or \
        [Path(d) for d in ([cfg.get("captures_dir")] if isinstance(cfg.get("captures_dir"), str)
                           else (cfg.get("captures_dir") or ["/home/bqn82/captures"]))]

    ml_specs = cfg.get("ml_detectors", {}) or {}
    requested = args.detectors or list(ml_specs.keys())
    if not requested:
        ap.error("no ML detectors configured (comparison_eval.ml_detectors) or requested (--detectors)")
    for d in requested:
        if d not in ml_specs:
            ap.error(f"unknown ML detector {d!r}; configured: {sorted(ml_specs)}")

    ref_dir = batch_root / ref_detector
    if not ref_dir.is_dir():
        ap.error(f"reference detector dir not found: {ref_dir} "
                 f"(run the trained sweep first so {ref_detector}/ carries GT + manifests)")
    stems = args.stems or sorted(p.name for p in ref_dir.iterdir()
                                 if p.is_dir() and (p / "frame_manifest.csv").exists())

    _wire_syspath(args.dinov3_repo or cfg.get("dinov3_repo"))
    import eval_viz as ev                       # noqa: E402  branch copy
    from finetuned_infer import to_display_grid  # noqa: E402  shared max-pool resampler

    print(f"batch root   : {batch_root}")
    print(f"ref detector : {ref_detector}  ({len(stems)} captures)")
    print(f"ml detectors : {requested}")
    print(f"captures dir : {', '.join(map(str, capture_dirs))}\n")

    t0 = time.time()
    summary: list[dict] = []
    for name in requested:
        print(f"=== {name} ===")
        det = _build_detector(name, ml_specs[name])
        for i, stem in enumerate(stems, 1):
            capture_path = ev.find_capture_data(stem, capture_dirs)
            res = run_capture(name, det, to_display_grid, ev,
                              ref_dir / stem, batch_root / name / stem,
                              capture_path, args.frame_limit)
            summary.append({"detector": name, "file_stem": stem, **res})
            print(f"  [{i}/{len(stems)}] {stem}: {res['made']} made, "
                  f"{res['skipped']} present, {res['too_short']} too-short "
                  f"({res['frames']} frames)")

    out_summary = batch_root / "ml_run_summary.json"
    out_summary.write_text(json.dumps({
        "batch_root": str(batch_root),
        "ref_detector": ref_detector,
        "detectors": requested,
        "specs": {d: ml_specs[d] for d in requested},
        "elapsed_s": round(time.time() - t0, 1),
        "runs": summary,
    }, indent=2))
    print(f"\nWrote {len(summary)} ML detector runs in {time.time() - t0:.1f}s")
    print(f"Summary: {out_summary}")
    print("\nNext: score everything (baselines + trained + ML) together:")
    print(f"  cd {_HELPERS}")
    print(f"  python3 eval_detector_masks.py --batch-root {batch_root} --out-dir {batch_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

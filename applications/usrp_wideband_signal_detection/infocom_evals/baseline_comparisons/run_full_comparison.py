#!/usr/bin/env python3
"""One-shot orchestrator for the 6-detector, SNR-axis offline comparison.

Runs the whole pipeline in series against a single batch-eval root so you can kick
it off over every capture / every detector, walk away, and review the result in
``baseline_eval_review.ipynb`` when it finishes. Every stage is idempotent (skips
work already present) and independently selectable, so a failed/partial run
resumes by just re-invoking.

Stages
------
  0. preflight  — verify the batch root exists and already holds the *trained*
                  detectors (coherent_power, cuda_dino). Those come from the C++
                  offline binary via run_batch_offline_eval.py, which needs the
                  demo container and is NOT run here — if they are missing this
                  prints the exact command to produce them and stops.
  1. baselines  — run_baseline_offline.py -> 3dB_power, blob_detection.
  2. ml         — run_ml_detectors_offline.py -> yolo, dino_finetuned (GPU).
  3. eval       — eval_detector_masks.py -> region_metrics.csv + frame_pixel_metrics.csv.
  4. snr        — build_snr_results.py -> snr_results.npz/.json (SNR calibration join).
  5. plots      — plot_snr_results.py -> snr_plots/ (shared -20..+40 dB axis).

All paths + detector params come from comparison_config.yaml (single source of
truth). The GPU stages (ml, and the trained sweep in preflight) run wherever the
weights + container live; the CPU stages (baselines, eval, snr, plots) run anywhere.

Example
-------
    python3 run_full_comparison.py \
        --config comparison_config.yaml \
        --batch-root ../signal_detection_experiments/batch_runs/sixway_20260716
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import yaml

_THIS_DIR = Path(__file__).resolve().parent
_INFOCOM = _THIS_DIR.parent
_HELPERS = _INFOCOM / "signal_detection_experiments"
_APP_DIR = _THIS_DIR.parents[1]

STAGES = ["preflight", "baselines", "ml", "eval", "snr", "plots"]


def _cfg(path: Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return raw.get("comparison_eval", raw)


def _run(cmd: list[str], cwd: Path, dry: bool) -> int:
    print(f"\n  $ (cd {cwd} && {' '.join(cmd)})", flush=True)
    if dry:
        return 0
    return subprocess.run(cmd, cwd=str(cwd)).returncode


def _capture_dirs(cfg: dict) -> list[str]:
    caps = cfg.get("captures_dir") or ["/home/bqn82/captures"]
    return [caps] if isinstance(caps, str) else list(caps)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def stage_preflight(cfg: dict, batch_root: Path, allow_missing: bool) -> bool:
    trained = cfg.get("trained_detectors", ["coherent_power", "cuda_dino"])
    missing = [d for d in trained if not (batch_root / d).is_dir()]
    if not batch_root.is_dir():
        missing = trained
    if missing:
        run_id = batch_root.name
        caps = _capture_dirs(cfg)[0]
        print(f"[preflight] batch root is missing the trained detector(s): {missing}")
        print("[preflight] Produce them first with the container sweep (single GPU, serialised):\n")
        print(f"    cd {_APP_DIR}")
        print(f"    python3 infocom_evals/signal_detection_experiments/run_batch_offline_eval.py \\")
        print(f"        --captures-dir {caps} \\")
        print(f"        --detectors {' '.join(trained)} \\")
        print(f"        --run-id {run_id} --no-post")
        print("\n[preflight] (--no-post: skip the attenuation-axis plots; this orchestrator does the SNR axis.)")
        if not allow_missing:
            return False
        print("[preflight] --allow-missing-trained set; continuing without them.")
    else:
        print(f"[preflight] OK — trained detectors present: {trained}")
    return True


def stage_baselines(cfg: dict, batch_root: Path, dry: bool, frame_limit: Optional[int]) -> bool:
    """Drive run_baseline_offline.py from a temp config synthesized from comparison_config."""
    baselines = cfg.get("baselines", {}) or {}
    if not baselines:
        print("[baselines] none configured — skipping.")
        return True
    tmp = Path(tempfile.mkdtemp(prefix="baseline_cfg_")) / "baseline_detectors_config.yaml"
    tmp.write_text(yaml.safe_dump({"baseline_eval": {
        "source_batch_root": str(batch_root),
        "out_batch_root": None,
        "source_detector": cfg.get("ref_detector"),
        "detectors": baselines,
    }}, sort_keys=False))
    cmd = [sys.executable, str(_THIS_DIR / "run_baseline_offline.py"),
           "--config", str(tmp), "--source-batch-root", str(batch_root),
           "--detectors", *baselines.keys()]
    for d in _capture_dirs(cfg):
        cmd += ["--captures-dir", d]
    if frame_limit is not None:
        cmd += ["--frame-limit", str(frame_limit)]
    return _run(cmd, _THIS_DIR, dry) == 0


def stage_ml(cfg: dict, batch_root: Path, dry: bool, frame_limit: Optional[int], config_path: Path) -> bool:
    ml = cfg.get("ml_detectors", {}) or {}
    if not ml:
        print("[ml] none configured — skipping.")
        return True
    cmd = [sys.executable, str(_THIS_DIR / "run_ml_detectors_offline.py"),
           "--config", str(config_path), "--batch-root", str(batch_root)]
    if frame_limit is not None:
        cmd += ["--frame-limit", str(frame_limit)]
    return _run(cmd, _THIS_DIR, dry) == 0


def stage_eval(cfg: dict, batch_root: Path, dry: bool, frame_limit: Optional[int]) -> bool:
    cmd = [sys.executable, str(_HELPERS / "eval_detector_masks.py"),
           "--batch-root", str(batch_root), "--out-dir", str(batch_root)]
    for d in _capture_dirs(cfg):
        cmd += ["--captures-dir", d]
    if frame_limit is not None:
        cmd += ["--frame-limit", str(frame_limit)]
    return _run(cmd, _HELPERS, dry) == 0


def stage_snr(cfg: dict, batch_root: Path, dry: bool) -> bool:
    snr = cfg.get("snr", {}) or {}
    cmd = [sys.executable, str(_THIS_DIR / "build_snr_results.py"),
           "--tables-dir", str(batch_root),
           "--zero-db-stem", str(snr.get("zero_db_stem", "attenuation_dB_0")),
           "--out", str(batch_root / "snr_results")]
    for d in _capture_dirs(cfg):
        cmd += ["--captures-dir", d]
    return _run(cmd, _THIS_DIR, dry) == 0


def stage_plots(cfg: dict, batch_root: Path, dry: bool) -> bool:
    snr = cfg.get("snr", {}) or {}
    rng = snr.get("snr_range", [-20.0, 40.0])
    cmd = [sys.executable, str(_THIS_DIR / "plot_snr_results.py"),
           "--results", str(batch_root / "snr_results"),
           "--out-dir", str(batch_root / "snr_plots"),
           "--snr-bin-width", str(snr.get("snr_bin_width", 5.0)),
           "--det-threshold", str(snr.get("det_threshold", 0.1)),
           "--snr-range", str(rng[0]), str(rng[1])]
    return _run(cmd, _THIS_DIR, dry) == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=_THIS_DIR / "comparison_config.yaml")
    ap.add_argument("--batch-root", type=Path, default=None,
                    help="Override comparison_eval.batch_root.")
    ap.add_argument("--stages", nargs="+", choices=STAGES, default=None,
                    help=f"Only run these stages (default: all). Order: {STAGES}")
    ap.add_argument("--skip", nargs="+", choices=STAGES, default=[],
                    help="Skip these stages.")
    ap.add_argument("--allow-missing-trained", action="store_true",
                    help="Continue even if coherent_power/cuda_dino are absent (partial compare).")
    ap.add_argument("--frame-limit", type=int, default=None, help="Debug: first N frames per run.")
    ap.add_argument("--dry-run", action="store_true", help="Print the stage commands only.")
    args = ap.parse_args()

    cfg = _cfg(args.config)
    batch_root = Path(args.batch_root or cfg.get("batch_root") or "")
    if not batch_root or str(batch_root).endswith("CHANGE_ME"):
        ap.error("set comparison_eval.batch_root (or --batch-root) to a real batch run id")
    batch_root = batch_root.resolve()

    todo = [s for s in (args.stages or STAGES) if s not in args.skip]
    print(f"batch root : {batch_root}")
    print(f"stages     : {todo}")

    ok = True
    for stage in todo:
        print(f"\n===== stage: {stage} =====")
        if stage == "preflight":
            ok = stage_preflight(cfg, batch_root, args.allow_missing_trained)
        elif stage == "baselines":
            ok = stage_baselines(cfg, batch_root, args.dry_run, args.frame_limit)
        elif stage == "ml":
            ok = stage_ml(cfg, batch_root, args.dry_run, args.frame_limit, args.config)
        elif stage == "eval":
            ok = stage_eval(cfg, batch_root, args.dry_run, args.frame_limit)
        elif stage == "snr":
            ok = stage_snr(cfg, batch_root, args.dry_run)
        elif stage == "plots":
            ok = stage_plots(cfg, batch_root, args.dry_run)
        if not ok:
            print(f"\n[stop] stage {stage!r} did not complete; fix and re-run "
                  f"(add --stages {stage} ... to resume from here).")
            return 2

    bar = "=" * 78
    print(f"\n{bar}\nDONE — review in the notebook:")
    print(f"  Open: {_THIS_DIR / 'baseline_eval_review.ipynb'}")
    print("  Set in the FIRST code cell, then Run All:")
    print(f"      BATCH_ROOT = Path('{batch_root}')")
    print(f"      RESULTS    = BATCH_ROOT / 'snr_results'")
    print(f"  Static SNR PNGs: {batch_root / 'snr_plots'}")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

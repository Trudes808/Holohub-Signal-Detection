#!/usr/bin/env python3
"""Extract the small, plot-regeneration artifacts from a batch-eval run.

The batch runs under ``signal_detection_experiments/batch_runs/<run_id>/`` are huge
(hundreds of GB), almost all of it per-frame mask arrays. But *regenerating and
tweaking the aggregate plots* only needs a handful of small files:

    region_metrics.csv        attenuation-axis plots (plot_eval_results.py)
    frame_pixel_metrics.csv    "
    snr_results.npz + .json    SNR-axis plots (plot_snr_results.py) -- reloadable
                               SnrResults, so you can re-bin / re-threshold / restyle
    eval_summary.json          provenance (thresholds, run list)
    *_run_summary.json         provenance (detector params / paths)

This copies just those into ``saved_results/<run_id>/`` (a small, reboot-persistent
sibling of the run dir), so the mask arrays in the run dir can be deleted to reclaim
disk while every plot still regenerates:

    python3 plot_snr_results.py --results saved_results/<run_id>/snr_results
    python3 ../signal_detection_experiments/plot_eval_results.py \\
        --tables-dir saved_results/<run_id>

It does NOT save per-frame masks/GT (regenerable by re-running the pipeline), and it
does not touch the source run.

Examples
--------
    python3 save_plot_data.py                 # save the config's batch_root
    python3 save_plot_data.py --all           # every run under batch_runs/ with metrics
    python3 save_plot_data.py --batch-root ../signal_detection_experiments/batch_runs/<id>
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_BATCH_RUNS = _THIS_DIR.parent / "signal_detection_experiments" / "batch_runs"

# artifacts to preserve (copied only if present)
ARTIFACTS = [
    "region_metrics.csv",
    "frame_pixel_metrics.csv",
    "region_metrics.parquet",
    "frame_pixel_metrics.parquet",
    "snr_results.npz",
    "snr_results.json",
    "eval_summary.json",
    "baseline_run_summary.json",
    "ml_run_summary.json",
]


def _config_batch_root() -> Path | None:
    cfg_path = _THIS_DIR / "comparison_config.yaml"
    if not cfg_path.exists():
        return None
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    cfg = cfg.get("comparison_eval", cfg)
    br = cfg.get("batch_root")
    return (_THIS_DIR / br).resolve() if br else None


def save_run(batch_root: Path, dest_root: Path) -> dict:
    """Copy the plot-regeneration artifacts from one run into dest_root/<run_id>/."""
    run_id = batch_root.name
    dest = dest_root / run_id
    dest.mkdir(parents=True, exist_ok=True)
    saved, total = [], 0
    for name in ARTIFACTS:
        src = batch_root / name
        if src.is_file():
            shutil.copy2(src, dest / name)
            total += src.stat().st_size
            saved.append(name)
    manifest = {
        "run_id": run_id,
        "source_batch_root": str(batch_root),
        "saved_files": saved,
        "total_bytes": total,
        "regenerate": {
            "snr_plots": f"python3 plot_snr_results.py --results {dest}/snr_results --snr-range -20 40",
            "attenuation_plots": ("python3 ../signal_detection_experiments/plot_eval_results.py "
                                  f"--tables-dir {dest} --det-threshold 0.1"),
        },
        "note": ("Aggregate plots regenerate from these files alone; per-frame mask "
                 "arrays in the source run dir are NOT needed and may be deleted."),
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    return {"run_id": run_id, "dest": dest, "files": saved, "bytes": total}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--batch-root", type=Path, default=None,
                   help="Run dir to save (default: comparison_config.yaml batch_root).")
    g.add_argument("--all", action="store_true",
                   help="Save every run under batch_runs/ that has region_metrics.csv.")
    ap.add_argument("--dest", type=Path, default=_THIS_DIR / "saved_results",
                    help="Where to write saved_results/<run_id>/ (default: ./saved_results).")
    args = ap.parse_args()

    if args.all:
        roots = sorted(p for p in _BATCH_RUNS.iterdir()
                       if p.is_dir() and (p / "region_metrics.csv").is_file())
    elif args.batch_root:
        roots = [args.batch_root.resolve()]
    else:
        br = _config_batch_root()
        if not br:
            ap.error("no --batch-root/--all and could not read batch_root from comparison_config.yaml")
        roots = [br]

    if not roots:
        print("Nothing to save (no runs with region_metrics.csv found).")
        return 1

    print(f"dest: {args.dest}\n")
    grand = 0
    for br in roots:
        if not (br / "region_metrics.csv").is_file():
            print(f"[skip] {br.name}: no region_metrics.csv")
            continue
        res = save_run(br, args.dest)
        grand += res["bytes"]
        print(f"[saved] {res['run_id']}: {len(res['files'])} files, "
              f"{res['bytes'] / 1e6:.1f} MB -> {res['dest']}")
    print(f"\nTotal saved: {grand / 1e6:.1f} MB")
    print("Plots regenerate from saved_results/<run_id>/ (see each MANIFEST.json); "
          "the source run's mask arrays can now be deleted to reclaim space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

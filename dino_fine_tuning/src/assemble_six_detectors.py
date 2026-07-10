"""Assemble a 6-detector batch-eval root and (re)generate the canonical tables.

Builds  <ft_root>/notebooks/sweep_detectors_six/  as a directory of symlinks so the
CANONICAL eval_detector_masks.py can score all six detectors identically (same
metrics, same source-SigMF buckets, all frames) -- nothing is copied or moved:

  coherent_power      -> sweep/coherent_power        (container, deployed)
  cuda_dino           -> sweep/cuda_dino             (container, zero-shot DINO)
  computer_vision     -> sweep/computer_vision       (container, NEW classical CV)
  power_detection     -> sweep/power_detection       (container, NEW power detector)
  finetuned_dino      -> sweep_detectors/finetuned_dino     (offline M1, materialized)
  finetuned_dino_m2   -> sweep_detectors/finetuned_dino_m2  (offline M2, materialized)

The two container baselines and the two fine-tuned mask dirs are REUSED verbatim
from the 3/4-detector setup, so their masks are byte-identical -> evaluations stay
consistent across the notebooks. The two NEW detectors come from the container batch
run (see the workflow doc); this script skips them with a warning if not present yet,
so it is safe to run before OR after the container run.

Then it runs eval_detector_masks.py at --coverage-threshold 0.1 (matches the original
batch_eval_review.ipynb / three-detector notebook) into  notebooks/compare_tables_six/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FT_ROOT = Path("/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning")
EVAL_DIR = Path("/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                "/infocom_evals/signal_detection_experiments")

# (run-dir name in the six-detector root, source location kind, source root)
#   kind "sweep"     -> <sweep>/<name>            (container-produced)
#   kind "materialized" -> <sweep_detectors>/<name>  (offline fine-tuned, already built)
CONTAINER_DETECTORS = ["coherent_power", "cuda_dino", "computer_vision", "power_detection"]
MATERIALIZED_DETECTORS = ["finetuned_dino", "finetuned_dino_m2"]


def log(m):
    print(f"[six] {m}", flush=True)


def link(src: Path, dst: Path) -> None:
    """Symlink dst -> src.resolve(); replace a stale/broken symlink, never a real dir."""
    src = src.resolve()
    if dst.is_symlink():
        if dst.resolve() == src:
            return
        dst.unlink()  # stale symlink -> repoint
    elif dst.exists():
        raise SystemExit(f"refusing to overwrite non-symlink {dst}")
    os.symlink(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="/tmp/usrp_spectrograms/batch_eval/sweep_20260630",
                    help="Container batch-eval root holding the deployed + new detector dirs.")
    ap.add_argument("--sweep-detectors", default=str(FT_ROOT / "notebooks/sweep_detectors"),
                    help="Existing 3/4-detector root that holds the materialized fine-tuned masks.")
    ap.add_argument("--out-root", default=str(FT_ROOT / "notebooks/sweep_detectors_six"),
                    help="Six-detector root to build (symlinks only).")
    ap.add_argument("--tables-dir", default=str(FT_ROOT / "notebooks/compare_tables_six"),
                    help="Where eval_detector_masks.py writes the combined tables.")
    ap.add_argument("--captures-dir", default="/home/bqn82/captures")
    ap.add_argument("--coverage-threshold", type=float, default=0.1,
                    help="Matches the original notebook (0.1).")
    ap.add_argument("--no-eval", action="store_true",
                    help="Only build the symlink root; skip running eval_detector_masks.py.")
    args = ap.parse_args()

    sweep = Path(args.sweep)
    sweep_dets = Path(args.sweep_detectors)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    present, missing = [], []
    for name in CONTAINER_DETECTORS:
        src = sweep / name
        if src.exists():
            link(src, out_root / name)
            present.append(name)
        else:
            missing.append((name, src))
    for name in MATERIALIZED_DETECTORS:
        src = sweep_dets / name
        if src.exists():
            link(src, out_root / name)
            present.append(name)
        else:
            missing.append((name, src))

    log(f"linked {len(present)} detector(s) into {out_root}: {present}")
    for name, src in missing:
        log(f"  MISSING (skipped): {name}  (expected at {src})")
    if any(n in dict(missing) for n in ("power_detection", "computer_vision")):
        log("  -> generate the new detectors' masks with the container batch run first:")
        log("     see dino_fine_tuning/notebooks/SIX_DETECTOR_WORKFLOW.md")
    if any(n in dict(missing) for n in MATERIALIZED_DETECTORS):
        log("  -> materialize fine-tuned masks first: python src/gen_finetuned_run.py [...]")

    if args.no_eval:
        log("(--no-eval) skipping eval_detector_masks.py")
        return 0
    if not present:
        log("no detector dirs present -> nothing to evaluate")
        return 1

    cmd = [sys.executable, str(EVAL_DIR / "eval_detector_masks.py"),
           "--batch-root", str(out_root),
           "--captures-dir", args.captures_dir,
           "--out-dir", args.tables_dir,
           "--coverage-threshold", str(args.coverage_threshold)]
    log("running: " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(EVAL_DIR)).returncode
    if rc == 0:
        log(f"DONE -> tables in {args.tables_dir} (detectors scored: {present})")
    else:
        log(f"eval_detector_masks.py FAILED (rc={rc})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

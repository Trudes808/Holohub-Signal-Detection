"""Assemble the YOLO eval: materialize YOLO26 masks into a batch-eval sweep root
alongside the DINO/deployed detectors, then run the canonical eval_detector_masks.py
into compare_tables -- so YOLO is scored identically to every other detector.

Steps:
  1. For each trained YOLO model (auto-globbed best.pt), run gen_yolo_run.py -> <sweep>/<name>/.
  2. Symlink the DINO/deployed detector run dirs (coherent_power, cuda_dino, finetuned_dino,
     finetuned_dino_m2) from the DINO sweep into the same <sweep> root.
  3. eval_detector_masks.py --batch-root <sweep> --out-dir <tables> --coverage-threshold 0.1.

Safe to run before training finishes: models with no best.pt yet are skipped with a warning.
Run with the 'yolo' env python.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

YT = Path("/home/bqn82/Holohub-Signal-Detection/yolo_training")
EVAL_DIR = Path("/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                "/infocom_evals/signal_detection_experiments")
DINO_SWEEP = Path("/home/bqn82/Holohub-Signal-Detection/notebooks/dino_fine_tuning_evals/sweeps/sweep_detectors")
DINO_DETECTORS = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2"]
YOLO_MODELS = ["yolo26s", "yolo26m"]


def log(m): print(f"[yolo-eval] {m}", flush=True)


def find_best(model: str) -> Path | None:
    hits = sorted(YT.glob(f"runs/**/{model}_signal/weights/best.pt"),
                  key=lambda q: q.stat().st_mtime, reverse=True)  # newest run wins
    return hits[0] if hits else None


def link(src: Path, dst: Path):
    src = src.resolve()
    if dst.is_symlink():
        if dst.resolve() == src:
            return
        dst.unlink()
    elif dst.exists():
        return
    os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(YT / "eval/sweeps/sweep_all"))
    ap.add_argument("--tables-dir", default=str(YT / "eval/compare_tables"))
    ap.add_argument("--dino-sweep", default=str(DINO_SWEEP))
    ap.add_argument("--captures-dir", default="/home/bqn82/captures")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--coverage-threshold", type=float, default=0.1)  # matches the DINO notebooks
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    sweep = Path(args.sweep); sweep.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    present = []

    # 1. YOLO masks
    for m in YOLO_MODELS:
        ckpt = find_best(m)
        if ckpt is None:
            log(f"SKIP {m}: no trained weights yet (runs/**/{m}_signal/weights/best.pt)")
            continue
        log(f"{m}: generating masks from {ckpt}")
        rc = subprocess.run([py, str(YT / "src/gen_yolo_run.py"),
                             "--yolo-ckpt", str(ckpt), "--detector-name", m,
                             "--out-root", str(sweep), "--captures-dir", args.captures_dir,
                             "--conf", str(args.conf)]).returncode
        if rc == 0:
            present.append(m)
        else:
            log(f"{m}: gen_yolo_run FAILED (rc={rc})")

    # 2. symlink DINO/deployed detectors
    for d in DINO_DETECTORS:
        src = Path(args.dino_sweep) / d
        if src.exists():
            link(src, sweep / d); present.append(d)
        else:
            log(f"missing DINO detector dir (skipped): {src}")

    log(f"detectors in sweep root: {present}")
    if args.no_eval:
        log("(--no-eval) skipping eval_detector_masks.py"); return 0
    if not any(m in present for m in YOLO_MODELS):
        log("no YOLO masks present yet -> not running eval (train first)"); return 1

    cmd = [py, str(EVAL_DIR / "eval_detector_masks.py"), "--batch-root", str(sweep),
           "--captures-dir", args.captures_dir, "--out-dir", args.tables_dir,
           "--coverage-threshold", str(args.coverage_threshold)]
    log("running: " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(EVAL_DIR)).returncode
    log(f"DONE (eval rc={rc}) -> tables in {args.tables_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

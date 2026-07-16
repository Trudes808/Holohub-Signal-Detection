"""Assemble the OOD-LTE YOLO eval: materialize YOLO26 masks on the LTE sweep alongside
the fine-tuned DINO detectors, then run eval_detector_masks -> compare_tables_lte.

The LTE sweep (from the DINO pipeline's build_lte_batch.py) only has finetuned_dino +
finetuned_dino_m2 (coherent_power / cuda_dino were never generated for LTE -- they'd need
a container run over ~/captures/lte). So this compares fine-tuned DINO vs YOLO26 on OOD data.
Run with the 'yolo' env python.
"""
from __future__ import annotations
import argparse, os, subprocess, sys
from pathlib import Path

YT = Path("/home/bqn82/Holohub-Signal-Detection/yolo_training")
EVAL_DIR = Path("/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection"
                "/infocom_evals/signal_detection_experiments")
DINO_LTE = Path("/home/bqn82/Holohub-Signal-Detection/notebooks/dino_fine_tuning_evals/sweeps/sweep_lte")
DINO_DETECTORS = ["finetuned_dino", "finetuned_dino_m2"]
YOLO_MODELS = ["yolo26s", "yolo26m"]


def log(m): print(f"[yolo-lte] {m}", flush=True)


def find_best(model):
    hits = sorted(YT.glob(f"runs/**/{model}_signal/weights/best.pt"),
                  key=lambda q: q.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def link(src, dst):
    src = Path(src).resolve()
    if dst.is_symlink():
        (dst.resolve() == src) or dst.unlink()
    if not dst.exists():
        os.symlink(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(YT / "eval/sweeps/sweep_lte"))
    ap.add_argument("--tables-dir", default=str(YT / "eval/compare_tables_lte"))
    ap.add_argument("--captures-dir", default="/home/bqn82/captures/lte")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--coverage-threshold", type=float, default=0.1)
    args = ap.parse_args()

    sweep = Path(args.sweep); sweep.mkdir(parents=True, exist_ok=True)
    py = sys.executable; present = []
    for m in YOLO_MODELS:
        ckpt = find_best(m)
        if ckpt is None:
            log(f"SKIP {m}: no weights"); continue
        log(f"{m}: generating LTE masks from {ckpt}")
        rc = subprocess.run([py, str(YT / "src/gen_yolo_run.py"),
                             "--sweep", str(DINO_LTE.parent.parent) if False else str(DINO_LTE),  # batch root = DINO_LTE
                             "--ref-detector", "finetuned_dino",
                             "--yolo-ckpt", str(ckpt), "--detector-name", m,
                             "--out-root", str(sweep), "--captures-dir", args.captures_dir,
                             "--conf", str(args.conf)]).returncode
        if rc == 0: present.append(m)
        else: log(f"{m}: gen FAILED rc={rc}")
    for d in DINO_DETECTORS:
        src = DINO_LTE / d
        if src.exists(): link(src, sweep / d); present.append(d)
        else: log(f"missing {src}")
    log(f"detectors in LTE sweep: {present}")
    if not any(m in present for m in YOLO_MODELS):
        log("no YOLO masks -> skip eval"); return 1
    cmd = [py, str(EVAL_DIR / "eval_detector_masks.py"), "--batch-root", str(sweep),
           "--captures-dir", args.captures_dir, "--out-dir", args.tables_dir,
           "--coverage-threshold", str(args.coverage_threshold)]
    log("running: " + " ".join(cmd))
    rc = subprocess.run(cmd, cwd=str(EVAL_DIR)).returncode
    log(f"DONE (eval rc={rc}) -> {args.tables_dir}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

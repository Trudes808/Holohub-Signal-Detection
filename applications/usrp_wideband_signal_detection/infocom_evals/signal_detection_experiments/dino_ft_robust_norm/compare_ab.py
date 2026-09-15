#!/usr/bin/env python3
"""Container A/B: M3 (491 robust) vs M2_dr (old) through the REAL operator on held-out composites.

Reads the offline-eval run dirs produced by run_validation-style runs and computes pixel recall/
precision/IoU (mean over frames) + region detection rate vs the SigMF GT, for each model x composite.
This confirms the retrain's gain holds in the deployment operator path (not just the Python front-end).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mask_eval_metrics as mem

AB = Path("/tmp/usrp_spectrograms/ab_out")
HELD = Path("/home/genesys-dgx1/Documents/Holohub-Signal-Detection/dino_fine_tuning/data/dataset_491/heldout_sigmf")
OUT = Path(__file__).resolve().parent / "results"
MODELS = ["m2", "m3"]
COMPS = ["dense_s9000", "sparse_s10000"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'model':>5} {'composite':>14}  {'frames':>6} {'pix_recall':>10} {'pix_prec':>9} "
          f"{'pix_IoU':>8} {'region_det':>10}")
    for model in MODELS:
        for c in COMPS:
            run = AB / f"{model}_{c}"
            meta = HELD / f"{c}.sigmf-meta"
            if not (run / "frame_manifest.csv").exists() and not (run / "mask_arrays").exists():
                # try the reconstruct path (evaluate_run handles missing manifest)
                pass
            fr, rg = mem.evaluate_run(run, "cuda_dino_finetuned", c, sigmf_meta_path=meta)
            rec = np.nanmean([r["recall"] for r in fr if "recall" in r]) if fr else float("nan")
            prec = np.nanmean([r["precision"] for r in fr if "precision" in r]) if fr else float("nan")
            iou = np.nanmean([r["iou"] for r in fr if "iou" in r]) if fr else float("nan")
            det = [bool(r.get("detected")) for r in rg]
            detrate = 100 * np.mean(det) if det else float("nan")
            rows.append(dict(model=model, composite=c, frames=len(fr), pix_recall=100*rec,
                             pix_precision=100*prec, pix_iou=100*iou, region_detrate=detrate,
                             regions=len(rg)))
            print(f"{model:>5} {c:>14}  {len(fr):>6} {100*rec:>9.1f}% {100*prec:>8.1f}% "
                  f"{100*iou:>7.1f}% {detrate:>9.1f}%  (n_reg={len(rg)})")
    (OUT / "ab_summary.json").write_text(json.dumps(rows, indent=2))
    print("\nwrote", OUT / "ab_summary.json")


if __name__ == "__main__":
    raise SystemExit(main())

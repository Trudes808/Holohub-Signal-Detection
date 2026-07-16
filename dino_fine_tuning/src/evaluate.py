"""Evaluate a segmenter (or the classical energy baseline) on the test split.

Produces, per predictor:
  - frame_metrics.csv : per-frame pixel IoU/precision/recall/F1/FP-area + attenuation
  - region_metrics.csv: per-GT-region coverage/detected + class/bandwidth/length/atten

Reuses the deployed detector's metric primitives (pixel confusion + bucket edges)
from infocom_evals/.../mask_eval_metrics.py so results are directly comparable.

Predictors:
  --ckpt <path>    a trained DinoSegmenter checkpoint (best.pt)
  --baseline energy  non-learned adaptive dB threshold (CFAR-ish), no training
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import RFSegDataset
from model import DinoSegmenter

# reuse the existing eval primitives / bucket edges
_MEM_DIR = Path("/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/"
                "infocom_evals/signal_detection_experiments")
sys.path.insert(0, str(_MEM_DIR))
import mask_eval_metrics as mem  # noqa: E402


def log(m):
    print(f"[eval] {m}", flush=True)


# --------------------------------------------------------------------------- #
# Predictors
# --------------------------------------------------------------------------- #
class LearnedPredictor:
    def __init__(self, ckpt_path, cfg, device):
        st = torch.load(ckpt_path, map_location=device)
        self.model = DinoSegmenter(cfg["weights_path"], feat_layers=tuple(cfg["feat_layers"]),
                                   mode=st["mode"], unfreeze_last_n=cfg["unfreeze_last_n"]).to(device)
        self.model.load_state_dict(st["model"]); self.model.eval()
        self.device = device; self.amp = cfg["amp"]
        self.threshold = 0.5

    @torch.no_grad()
    def prob(self, img):
        img = img.to(self.device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
            logits = self.model(img)
        return torch.sigmoid(logits.float()).cpu().numpy()[:, 0]  # B,H,W

    def tune_threshold(self, val_loader):
        """Pick the threshold maximizing micro-F1 on val."""
        probs, gts = [], []
        for b in val_loader:
            probs.append(self.prob(b["image"])); gts.append(b["mask"].numpy()[:, 0])
        P = np.concatenate(probs).ravel(); G = np.concatenate(gts).ravel().astype(bool)
        best_t, best_f1 = 0.5, -1
        for t in np.linspace(0.05, 0.95, 19):
            pred = P >= t
            tp = (pred & G).sum(); fp = (pred & ~G).sum(); fn = (~pred & G).sum()
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        self.threshold = best_t
        log(f"tuned threshold={best_t:.2f} (val micro-F1={best_f1:.3f})")
        return best_t


class EnergyBaseline:
    """Non-learned adaptive threshold: pixel is signal if dB > median + k*MAD."""
    def __init__(self, meta, k=4.0):
        self.vmin, self.vmax = meta["db_vmin"], meta["db_vmax"]
        self.k = k
        self.threshold = 0.5  # unused

    def prob(self, img):  # img: B,1,H,W float [0,1]
        x = img.numpy()[:, 0]
        db = x * (self.vmax - self.vmin) + self.vmin
        out = np.zeros_like(db)
        for i in range(db.shape[0]):
            f = db[i]
            med = np.median(f)
            mad = np.median(np.abs(f - med)) * 1.4826 + 1e-6
            out[i] = (f > med + self.k * mad).astype(np.float32)
        return out

    def tune_threshold(self, val_loader):
        return self.threshold


# --------------------------------------------------------------------------- #
def run(predictor, loader, regions_by_frame, thr, out_dir):
    frame_rows, region_rows = [], []
    is_learned = isinstance(predictor, LearnedPredictor)
    for b in loader:
        probs = predictor.prob(b["image"])
        gts = b["mask"].numpy()[:, 0].astype(np.uint8)
        for i in range(probs.shape[0]):
            pred = (probs[i] >= thr).astype(np.uint8) if is_learned else probs[i].astype(np.uint8)
            gt = gts[i]
            pm = mem.pixel_metrics(pred, gt)
            fid = b["frame_id"][i]
            atten = b["attenuation_db"][i]
            is_signal = int(gt.sum() > 0)
            frame_rows.append({
                "frame_id": fid, "attenuation_db": atten, "is_signal": is_signal,
                "precision": pm.precision, "recall": pm.recall, "f1": pm.f1,
                "iou": pm.iou, "fp_area_fraction": pm.fp_area_fraction,
                "gt_pixels": pm.gt_pixels, "pred_pixels": pm.pred_pixels,
            })
            # per-region coverage
            for reg in regions_by_frame.get(fid, []):
                item = {"row_start": reg["row0"], "row_stop": reg["row1"],
                        "col_start": reg["col0"], "col_stop": reg["col1"]}
                rr = mem.region_coverage(pred, item, gt.shape[0], gt.shape[1])
                attrs = {"occupied_bw_hz": reg["bandwidth_hz"],
                         "length_samples": reg["length_samples"],
                         "sample_count": reg["length_samples"]}
                region_rows.append({
                    "frame_id": fid, "attenuation_db": atten,
                    "signal_class": reg["label"], "kind": reg["kind"],
                    "bandwidth_bucket": mem.bucket_bandwidth(attrs),
                    "length_bucket": mem.bucket_length(attrs),
                    "bandwidth_hz": reg["bandwidth_hz"],
                    "length_samples": reg["length_samples"],
                    "coverage": rr.coverage, "box_pixels": rr.box_pixels,
                    "covered_pixels": rr.covered_pixels,
                })
    out_dir.mkdir(parents=True, exist_ok=True)
    _wcsv(out_dir / "frame_metrics.csv", frame_rows)
    _wcsv(out_dir / "region_metrics.csv", region_rows)
    log(f"wrote {len(frame_rows)} frame rows, {len(region_rows)} region rows -> {out_dir}")


def _wcsv(path, rows):
    if not rows:
        path.write_text(""); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def load_regions(dataset_dir):
    by_frame = {}
    for r in csv.DictReader(open(Path(dataset_dir) / "regions.csv")):
        if r["split"] != "test":
            continue
        by_frame.setdefault(r["frame_id"], []).append({
            "row0": int(r["row0"]), "row1": int(r["row1"]),
            "col0": int(r["col0"]), "col1": int(r["col1"]),
            "label": r["label"], "kind": r["kind"],
            "bandwidth_hz": float(r["bandwidth_hz"]),
            "length_samples": int(r["length_samples"]),
        })
    return by_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--baseline", choices=["energy"], default=None)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = "cuda"
    meta = json.loads((Path(args.dataset) / "dataset_meta.json").read_text())

    te = RFSegDataset(args.dataset, "test", augment=False)
    va = RFSegDataset(args.dataset, "val", augment=False)
    tl = DataLoader(te, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    vl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    if args.ckpt:
        pred = LearnedPredictor(args.ckpt, cfg, device)
        thr = pred.tune_threshold(vl)
    else:
        pred = EnergyBaseline(meta); thr = pred.threshold

    regions = load_regions(args.dataset)
    out_dir = Path(args.out) / args.name
    run(pred, tl, regions, thr, out_dir)
    (out_dir / "eval_meta.json").write_text(json.dumps(
        {"name": args.name, "threshold": thr, "ckpt": args.ckpt,
         "baseline": args.baseline, "n_test": len(te)}, indent=2))


if __name__ == "__main__":
    main()

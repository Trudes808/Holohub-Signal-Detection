#!/usr/bin/env python3
"""Evaluate DINO-FT recall vs GT across the SNR sweep (+ per-signal-type detection rate)."""
from __future__ import annotations
import sys, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mask_eval_metrics as mem

DINO = Path("/tmp/usrp_spectrograms/offline_eval/snr_dino")
VAR = Path("/tmp/usrp_spectrograms/snr_bench")
OUT = Path(__file__).resolve().parent / "results"
TAGS = [("clean", 99), ("snr20", 20), ("snr12", 12), ("snr6", 6), ("snr0", 0)]


def type_of(row: dict) -> str:
    for k in ("kind", "wfgt_kind", "label", "core_label", "signal_kind"):
        if row.get(k):
            return str(row[k])
    return "?"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    per_type = defaultdict(dict)   # type -> {snr: (det, total)}
    for tag, snr in TAGS:
        run = DINO / tag
        meta = VAR / tag / f"comprehensive_ordered_{tag}.sigmf-meta"
        if not (run / "frame_manifest.csv").exists():
            print(f"skip {tag} (no run)"); continue
        fr, rg = mem.evaluate_run(run, "cuda_dino_finetuned", f"comprehensive_ordered_{tag}", sigmf_meta_path=meta)
        rec = np.mean([r["recall"] for r in fr if "recall" in r]) if fr else float("nan")
        prec = np.mean([r["precision"] for r in fr if "precision" in r]) if fr else float("nan")
        iou = np.mean([r["iou"] for r in fr if "iou" in r]) if fr else float("nan")
        det = [bool(r.get("detected")) for r in rg]
        detrate = 100 * np.mean(det) if det else float("nan")
        rows_out.append(dict(tag=tag, snr=snr, frames=len(fr), regions=len(rg),
                             pixel_recall=100*rec, pixel_precision=100*prec, pixel_iou=100*iou, region_detrate=detrate))
        # per-type detection rate
        agg = defaultdict(lambda: [0, 0])
        for r in rg:
            t = type_of(r); agg[t][1] += 1; agg[t][0] += 1 if r.get("detected") else 0
        for t, (d, n) in agg.items():
            per_type[t][snr] = (d, n)
        print(f"{tag:>6} SNR={snr:>3}dB  frames={len(fr):>3} regions={len(rg):>4}  "
              f"pixel_recall={100*rec:5.1f}%  precision={100*prec:5.1f}%  IoU={100*iou:5.1f}%  "
              f"region_detected={detrate:5.1f}%")

    (OUT / "snr_summary.json").write_text(json.dumps(rows_out, indent=2))

    # plot 1: overall vs SNR
    xs = [r["snr"] for r in rows_out]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for key, lab in [("region_detrate", "region detection rate"), ("pixel_recall", "pixel recall"),
                     ("pixel_precision", "pixel precision"), ("pixel_iou", "pixel IoU")]:
        ax[0].plot(xs, [r[key] for r in rows_out], marker="o", label=lab)
    ax[0].set_xlabel("SNR (dB); 99=clean"); ax[0].set_ylabel("%"); ax[0].set_ylim(0, 100)
    ax[0].set_title("DINO-FT vs GT across SNR (comprehensive_ordered)"); ax[0].grid(alpha=.3); ax[0].legend()
    # plot 2: per-type region detection rate vs SNR
    types = sorted(per_type, key=lambda t: -sum(n for _, n in per_type[t].values()))
    types = [t for t in types if t not in ("metadata", "METADATA", "?")][:8]
    for t in types:
        xx = sorted(per_type[t]); yy = [100*per_type[t][s][0]/max(1, per_type[t][s][1]) for s in xx]
        ax[1].plot(xx, yy, marker="o", label=t)
    ax[1].set_xlabel("SNR (dB); 99=clean"); ax[1].set_ylabel("detection rate %"); ax[1].set_ylim(0, 105)
    ax[1].set_title("Per-signal-type detection rate vs SNR"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "snr_recall.png", dpi=110); plt.close(fig)
    print("\nwrote", OUT / "snr_recall.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

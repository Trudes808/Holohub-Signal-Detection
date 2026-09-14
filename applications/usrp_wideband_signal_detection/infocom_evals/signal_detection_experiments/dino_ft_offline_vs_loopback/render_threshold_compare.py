#!/usr/bin/env python3
"""
Show what lowering the DINO-FT decision threshold recovers for weak/small signals.

Both offline runs process the SAME capture deterministically (same frame boundaries), so frame N is the
same spectrogram in each -- only the mask differs. Overlay on the raw spectrogram:
  red  = detected at 0.95 (current)
  cyan = ADDITIONAL detections at 0.70 (recovered by the lower threshold)
Frames are ranked by recovered in-band pixels (weak-signal recovery), and aggregate occupancy is printed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mask_eval_metrics as mem

STEM = "x410_ota_2g4_gain10_20260908"
OFF95 = Path(f"/tmp/usrp_spectrograms/offline_eval/cuda_dino_finetuned_rt/{STEM}")
OFF70 = Path(f"/tmp/usrp_spectrograms/offline_eval/cuda_dino_finetuned_rt_thr070/{STEM}")
OUT = Path(__file__).resolve().parent / "results_guard" / "threshold_compare.png"
DCOLS = 2048


def binmax(a, n=DCOLS):
    w = a.shape[-1]; s = w // n
    return (a[..., :s * n].reshape(a.shape[:-1] + (n, s))).max(-1) if s > 1 else a


def binmean(a, n=DCOLS):
    w = a.shape[-1]; s = w // n
    return (a[..., :s * n].reshape(a.shape[:-1] + (n, s))).mean(-1) if s > 1 else a


def manifest(run):
    return {int(r["frame_number"]): r for r in mem.load_manifest(run)}


def main() -> int:
    m95, m70 = manifest(OFF95), manifest(OFF70)
    common = sorted(set(m95) & set(m70))
    recs = []
    occ95_all, occ70_all = [], []
    for fn in common:
        a = mem.load_mask_any(OFF95 / m95[fn]["mask_npy"]) > 0
        b = mem.load_mask_any(OFF70 / m70[fn]["mask_npy"]) > 0
        occ95_all.append(a.mean()); occ70_all.append(b.mean())
        recovered = b & ~a
        recs.append((fn, float(a.mean()), float(b.mean()), int(recovered.sum())))
    print(f"frames={len(common)}  occ mean: 0.95={100*np.mean(occ95_all):.3f}%  "
          f"0.70={100*np.mean(occ70_all):.3f}%  (x{np.mean(occ70_all)/max(1e-9,np.mean(occ95_all)):.2f})")
    recs.sort(key=lambda r: -r[3])
    picks = [r[0] for r in recs[:2]]   # two representative panels (keeps the artifact under the size cap)

    fig, axes = plt.subplots(len(picks), 1, figsize=(12, 3.0 * len(picks)), squeeze=False)
    for ax, fn in zip(axes[:, 0], picks):
        t = np.load(OFF95 / m95[fn]["spectrogram_tensor_npy"])
        db = 20 * np.log10(np.abs(t).astype(np.float64) + 1e-6)
        a = mem.load_mask_any(OFF95 / m95[fn]["mask_npy"]) > 0
        b = mem.load_mask_any(OFF70 / m70[fn]["mask_npy"]) > 0
        dbd = binmean(db)
        a_d = binmax(a.astype(np.float64)) > 0
        rec_d = binmax((b & ~a).astype(np.float64)) > 0
        vmin, vmax = np.percentile(dbd, 5), np.percentile(dbd, 99.5)
        ax.imshow(dbd, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ov = np.zeros((*a_d.shape, 4))
        ov[a_d] = [1, 0, 0, 0.55]        # current (0.95)
        ov[rec_d] = [0, 1, 1, 0.75]      # recovered by 0.70
        ax.imshow(ov, aspect="auto", origin="lower")
        occ95 = 100 * float(a.mean()); occ70 = 100 * float(b.mean())
        ax.set_title(f"frame {fn}: red=thr0.95 ({occ95:.2f}%)  cyan=recovered@0.70 (+{occ70-occ95:.2f}%)  "
                     f"— raw spectrogram")
        ax.set_xlabel("freq bin"); ax.set_ylabel("time row")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=90); plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

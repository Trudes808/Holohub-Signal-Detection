#!/usr/bin/env python3
"""
Render real-time (loopback) DINO-FT masks overlaid on the ACTUAL spectrogram the RT model saw.

Uses the loopback mask dump WITH spectrograms (debug_spectrogram_dump=true): each frame has both the
emitted mask (uint8) and the normalized [0,1] input image (float32) on the same rows x mask_width grid,
so the overlay is truthful (not a fuzzy alignment to offline frames). Run guard-OFF so a drop-corrupted
frame is captured with its spectrogram; panels annotate which frames the invalid-frame guard suppresses.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP = Path("/tmp/usrp_spectrograms/loopback_eval/dino_ft_rt/mask_arrays")
OUT = Path(__file__).resolve().parent / "results_guard" / "rt_mask_overlays.png"
GUARD_FLOOR = 0.03      # invalid_frame_min_occupancy: frames above this are suppressed by the guard
DCOLS = 2048            # display column bins


def bin_cols(a, n=DCOLS, how="mean"):
    w = a.shape[-1]; step = w // n
    if step < 1: return a
    r = a[..., :step * n].reshape(a.shape[:-1] + (n, step))
    return r.max(-1) if how == "max" else r.mean(-1)


def main() -> int:
    man = DUMP / "mask_dump_manifest.csv"
    rows = list(csv.DictReader(man.open()))
    rows = [r for r in rows if r.get("spec_npy")]  # only frames with a spectrogram
    if not rows:
        print("no frames with spec_npy — did the run use debug_spectrogram_dump=true?")
        return 1

    recs = []
    for r in rows:
        m = np.load(DUMP / r["mask_npy"]) > 0
        recs.append((r, float(m.mean())))
    occs = np.array([o for _, o in recs])
    print(f"{len(recs)} frames  occ mean={100*occs.mean():.3f}%  max={100*occs.max():.3f}%  "
          f"suppressed-by-guard(>{100*GUARD_FLOOR:.0f}%): {(occs>GUARD_FLOOR).sum()}")

    # Selection: the worst corrupt frame(s) (occ > floor) + legitimate busy frames (real signals) + a mid.
    order = sorted(range(len(recs)), key=lambda i: -recs[i][1])
    corrupt = [i for i in order if recs[i][1] > GUARD_FLOOR][:2]
    legit = [i for i in order if 0.003 < recs[i][1] <= GUARD_FLOOR][:3]
    mid = [i for i in order if 0.0 < recs[i][1] <= 0.003][:1]
    picks = corrupt + legit + mid
    if not picks:
        picks = order[:5]

    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 3.1 * len(picks)), squeeze=False)
    for ax, i in zip(axes[:, 0], picks):
        r, occ = recs[i]
        spec = np.load(DUMP / r["spec_npy"]).astype(np.float64)   # rows x cols, [0,1]
        mask = np.load(DUMP / r["mask_npy"]) > 0
        sd = bin_cols(spec, how="mean")
        md = bin_cols(mask.astype(np.float64), how="max") > 0.0
        vmin, vmax = np.percentile(sd, 2), np.percentile(sd, 99.5)
        ax.imshow(sd, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ov = np.zeros((*md.shape, 4)); ov[md] = [1, 0, 0, 0.5]
        ax.imshow(ov, aspect="auto", origin="lower")
        tag = "SUPPRESSED by invalid-frame guard" if occ > GUARD_FLOOR else "kept (legitimate)"
        ax.set_title(f"RT loopback — frame {r['frame_number']}  occ {100*occ:.2f}%  [{tag}]  "
                     f"— RT input spectrogram + DINO-FT mask (red)")
        ax.set_xlabel("freq bin"); ax.set_ylabel("time row")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    plt.close(fig)
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

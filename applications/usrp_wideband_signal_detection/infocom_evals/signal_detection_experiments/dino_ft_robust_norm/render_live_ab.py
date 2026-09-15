#!/usr/bin/env python3
"""Render M2_dr vs M3_491 masks on the REAL 2.4 GHz OTA capture (the scene that looked 'atrocious').

No GT (real capture), so this is a visual A/B: for a few frames, each model's dumped model-input
spectrogram with its own emitted mask outlined. M3 should light up the clear signals M2 misses.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DUMP = Path("/tmp/usrp_spectrograms/live_ab")
OUT = Path(__file__).resolve().parent / "results"
FRAMES = [4, 10, 16]   # representative emitted-frame seqs


def load(model, seq):
    d = DUMP / model
    man = {int(r["seq"]): r for r in csv.DictReader(open(d / "mask_dump_manifest.csv"))}
    if seq not in man:
        return None
    r = man[seq]
    spec = np.load(d / Path(r["spec_npy"]).name).astype(np.float32)
    mask = np.load(d / Path(r["mask_npy"]).name).astype(np.uint8)
    return spec, mask


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(FRAMES)
    fig, axes = plt.subplots(n, 2, figsize=(15, 3.2 * n), squeeze=False)
    for i, seq in enumerate(FRAMES):
        for j, model in enumerate(["m2", "m3"]):
            ax = axes[i][j]
            r = load(model, seq)
            if r is None:
                ax.text(0.5, 0.5, f"{model} seq{seq}: no dump", ha="center", va="center"); ax.axis("off"); continue
            spec, mask = r
            ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                      interpolation="nearest")
            if mask.any():
                ax.contour(mask, levels=[0.5], colors="cyan", linewidths=0.6)
            occ = 100 * mask.mean()
            label = "M2_dr (old, q-blend, thr0.95)" if model == "m2" else "M3_491 (new, robust, thr0.6)"
            ax.set_title(f"{label} — frame {seq}, mask occ {occ:.2f}%", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Real 2.4 GHz OTA capture: DINO-FT masks (cyan) — old M2_dr vs new M3_491", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = OUT / "live_ab_ota.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    raise SystemExit(main())

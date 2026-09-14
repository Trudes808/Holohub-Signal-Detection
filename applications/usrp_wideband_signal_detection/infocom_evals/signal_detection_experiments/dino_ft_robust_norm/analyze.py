#!/usr/bin/env python3
"""Quantify whether robust normalization collapses, from the dumped [0,1] model input + mask.

For each (scene, mode) the operator dumped spec_ch0_*.npy (the float32 [0,1] image the model actually
saw) and mask_ch0_*.npy (uint8). "Collapse" = the normalization destroyed usable contrast:
  - saturation collapse: nearly every pixel maps to ~1 (frac>=0.99 huge) -> model sees a white sheet
  - dead collapse:       nearly every pixel maps to ~0 (frac<=0.02 huge) -> signal crushed into floor
A healthy input keeps a real spread (std well above 0) with the floor near ~0.1 and signals near ~1.
We report per-mode input std / sat% / dead% / p50, and mask occupancy, across all dumped frames, per
scene, then render a scene x mode panel (representative model input + mask contour + a value histogram).
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DUMP_ROOT = Path("/tmp/usrp_spectrograms/robust_val")
OUT = Path(__file__).resolve().parent / "results"
SCENES = ["clean", "snr0", "sparse"]
MODES = ["fixed", "adapt", "robust"]
MODE_LABEL = {"fixed": "fixed clip", "adapt": "adaptive (q-blend)", "robust": "robust (p-floor+fallback)"}


def load_run(scene: str, mode: str):
    d = DUMP_ROOT / f"{scene}_{mode}"
    man = d / "mask_dump_manifest.csv"
    if not man.exists():
        return None
    specs, masks = [], []
    with man.open() as f:
        for row in csv.DictReader(f):
            sp = row.get("spec_npy") or ""
            mk = row.get("mask_npy") or ""
            spath = d / Path(sp).name if sp else None
            mpath = d / Path(mk).name if mk else None
            if spath and spath.exists():
                specs.append(np.load(spath).astype(np.float32))
            if mpath and mpath.exists():
                masks.append(np.load(mpath).astype(np.uint8))
    if not specs:
        return None
    return specs, masks


def stats(specs, masks):
    allpx = np.concatenate([s.ravel() for s in specs])
    occ = float(np.mean([m.mean() for m in masks])) if masks else float("nan")
    return dict(
        n_frames=len(specs),
        std=float(allpx.std()),
        mean=float(allpx.mean()),
        p50=float(np.percentile(allpx, 50)),
        sat_frac=float(np.mean(allpx >= 0.99)),
        dead_frac=float(np.mean(allpx <= 0.02)),
        mask_occ=occ,
    )


def verdict(st):
    if st["sat_frac"] > 0.5:
        return "COLLAPSE-sat"
    if st["dead_frac"] > 0.9:
        return "COLLAPSE-dead"
    if st["std"] < 0.05:
        return "COLLAPSE-flat"
    return "healthy"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    print(f"{'scene':>7} {'mode':>7}  {'frames':>6} {'std':>6} {'p50':>6} {'sat%':>6} {'dead%':>6} "
          f"{'mask_occ%':>9}  verdict")
    for scene in SCENES:
        summary[scene] = {}
        for mode in MODES:
            run = load_run(scene, mode)
            if run is None:
                print(f"{scene:>7} {mode:>7}  (no dump)")
                continue
            specs, masks = run
            st = stats(specs, masks)
            st["verdict"] = verdict(st)
            summary[scene][mode] = st
            print(f"{scene:>7} {mode:>7}  {st['n_frames']:>6} {st['std']:>6.3f} {st['p50']:>6.3f} "
                  f"{100*st['sat_frac']:>6.1f} {100*st['dead_frac']:>6.1f} {100*st['mask_occ']:>9.2f}  "
                  f"{st['verdict']}")
    (OUT / "robust_norm_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure: scene (rows) x mode (cols); model input + mask contour + value histogram inset ----
    fig, axes = plt.subplots(len(SCENES), len(MODES), figsize=(4.6 * len(MODES), 3.4 * len(SCENES)),
                             squeeze=False)
    for i, scene in enumerate(SCENES):
        for j, mode in enumerate(MODES):
            ax = axes[i][j]
            run = load_run(scene, mode)
            if run is None:
                ax.text(0.5, 0.5, "no dump", ha="center", va="center"); ax.axis("off"); continue
            specs, masks = run
            k = len(specs) // 2  # representative middle frame
            spec = specs[k]
            ax.imshow(spec, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                      interpolation="nearest")
            if masks and k < len(masks):
                ax.contour(masks[k], levels=[0.5], colors="cyan", linewidths=0.5)
            st = summary[scene][mode]
            ax.set_title(f"{scene} / {MODE_LABEL[mode]}\nstd={st['std']:.2f} sat={100*st['sat_frac']:.0f}% "
                         f"dead={100*st['dead_frac']:.0f}% occ={100*st['mask_occ']:.1f}% [{st['verdict']}]",
                         fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            # value histogram inset
            ins = ax.inset_axes([0.6, 0.05, 0.36, 0.28])
            ins.hist(np.concatenate([s.ravel() for s in specs]), bins=40, range=(0, 1),
                     color="white", alpha=0.8)
            ins.set_yticks([]); ins.set_xticks([0, 1]); ins.tick_params(labelsize=6)
            ins.patch.set_alpha(0.3)
    fig.suptitle("DINO-FT model input [0,1] by normalization mode x scene (cyan = emitted mask)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "robust_norm_compare.png", dpi=110)
    plt.close(fig)
    print("\nwrote", OUT / "robust_norm_compare.png")
    print("wrote", OUT / "robust_norm_summary.json")


if __name__ == "__main__":
    main()

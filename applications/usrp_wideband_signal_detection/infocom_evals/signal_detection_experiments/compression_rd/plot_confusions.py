#!/usr/bin/env python3
"""Confusion-matrix heatmaps for the codec-model eval.

Reads codec_model_confusions.csv (written by eval_codec_models.py) and renders
row-normalized 10-class confusion heatmaps for the matched bfp8 cell (codec
effect is null, so one matched pair represents them all):

    confusions_gt.png   -- ground-truth-box lane, one panel per variant
    confusions_dino.png -- DINO-FT lane (incl. NOISE-truth false boxes)

    python3 plot_confusions.py [--store bfp8 --model bfp8]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

THIS = Path(__file__).resolve().parent
CLASSES = ["BPSK", "QPSK", "16QAM", "OFDM", "5G_Downlink", "802_11ax",
           "Bluetooth", "Narrowband_FM", "Broadband_FM", "NOISE"]
SHORT = ["BPSK", "QPSK", "16QAM", "OFDM", "5G", "11ax", "BT", "NBFM", "BBFM", "NOISE"]
VARIANTS = ["clean", "30", "15", "0", "-10"]


def load(store: str, model: str):
    conf: dict[tuple, np.ndarray] = {}
    with open(THIS / "codec_model_confusions.csv") as f:
        for r in csv.DictReader(f):
            if r["storage_codec"] != store or r["model_codec"] != model:
                continue
            key = (r["lane"], r["variant"])
            m = conf.setdefault(key, np.zeros((len(CLASSES), len(CLASSES)), np.int64))
            if r["truth"] in CLASSES and r["pred"] in CLASSES:
                m[CLASSES.index(r["truth"]), CLASSES.index(r["pred"])] += int(r["n"])
    return conf


def draw(conf, lane: str, out: Path, store: str, model: str):
    panels = [(v, conf.get((lane, v))) for v in VARIANTS if (lane, v) in conf]
    if not panels:
        print(f"[plot] no data for lane {lane}")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.4))
    axes = np.atleast_1d(axes)
    for ax, (variant, m) in zip(axes, panels):
        keep = m.sum(axis=1) > 0                      # drop empty truth rows (e.g. gt-lane NOISE)
        rows = [i for i in range(len(CLASSES)) if keep[i]]
        mm = m[rows]
        norm = mm / np.maximum(mm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                if norm[i, j] >= 0.05:
                    ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center",
                            fontsize=7, color="white" if norm[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(CLASSES)), SHORT, rotation=60, fontsize=8)
        ax.set_yticks(range(len(rows)),
                      [f"{SHORT[i]} (n={m[i].sum()})" for i in rows], fontsize=8)
        ax.set_title(f"{lane} / nominal {variant}", fontsize=11)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("truth")
    fig.suptitle(f"Confusion (row-normalized), storage={store}, model={model}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", default="bfp8")
    ap.add_argument("--model", default="bfp8")
    args = ap.parse_args()
    conf = load(args.store, args.model)
    draw(conf, "gt", THIS / "confusions_gt.png", args.store, args.model)
    draw(conf, "dino", THIS / "confusions_dino.png", args.store, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

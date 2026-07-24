#!/usr/bin/env python3
"""Which detectors' masks contain the persistent 48 MHz streak, and at which attenuations?

For each detector x attenuation, over all staged mask frames: per-column occupancy = fraction of
time-rows lit, averaged over frames. The streak column (~+48 MHz -> col 7120 at 10240 cols) shows up
as a high occupancy spike. Reports max occupancy in a +/-10-col window at the streak vs the median
column occupancy (background), plus the fraction of frames where the streak column is lit in >=90%
of rows ("full-height streak frames").

Run: ~/miniforge3/envs/dinov3/bin/python streak_mask_presence.py
"""
from __future__ import annotations
import csv
import glob
from pathlib import Path

import numpy as np

SE = Path(__file__).resolve().parent
FS = 245.76e6
STREAK_HZ = 48e6
ATTENS = [40, 50, 55, 60, 65, 70]
DETS = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2",
        "3dB_power", "blob_detection", "yolo26m", "yolo26s", "ground_truth"]


def load(f):
    z = np.load(f)
    return np.unpackbits(z["packed"])[: int(z["rows"]) * int(z["cols"])].reshape(
        int(z["rows"]), int(z["cols"]))


def main():
    rows_out = []
    for det in DETS:
        for att in ATTENS:
            fl = sorted(glob.glob(str(SE / f"snip_run/{det}/attenuation_dB_{att}/mask_arrays/*.packed.npz")))
            if not fl:
                continue
            occ_sum = None
            full_height_frames = 0
            n = 0
            cov = 0.0
            for f in fl:
                m = load(f)
                if occ_sum is None:
                    R, C = m.shape
                    c0 = int(round((STREAK_HZ / FS + 0.5) * C))
                    occ_sum = np.zeros(C)
                colocc = m.mean(axis=0)
                occ_sum += colocc
                cov += m.mean()
                if colocc[max(0, c0 - 10):c0 + 11].max() >= 0.9:
                    full_height_frames += 1
                n += 1
            occ = occ_sum / n
            streak_occ = float(occ[max(0, c0 - 10):c0 + 11].max())
            streak_col = int(max(0, c0 - 10) + np.argmax(occ[max(0, c0 - 10):c0 + 11]))
            bg = float(np.median(occ))
            rows_out.append(dict(detector=det, attenuation_db=att, n_frames=n,
                                 mean_coverage_pct=round(100 * cov / n, 3),
                                 streak_col=streak_col,
                                 streak_mhz=round((streak_col / C - 0.5) * FS / 1e6, 2),
                                 streak_occupancy_pct=round(100 * streak_occ, 1),
                                 median_col_occupancy_pct=round(100 * bg, 3),
                                 full_height_streak_frames_pct=round(100 * full_height_frames / n, 1)))
            print(rows_out[-1])
    with open(SE / "streak_mask_presence.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", SE / "streak_mask_presence.csv")


if __name__ == "__main__":
    main()

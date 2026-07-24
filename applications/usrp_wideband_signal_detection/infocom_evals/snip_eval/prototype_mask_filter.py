#!/usr/bin/env python3
"""Prototype of the snipper's planned mask pre-filter: per-row run-length filtering that removes
lit runs narrower than min_mask_bandwidth_hz (auto-scaled to columns), applied BEFORE
connected-component labeling. The 48 MHz spur line (~3 cols) dies even though it touches a big
component; wide bursts keep their pixels. min-duration stays at the existing bbox gate.

For each (detector, atten, gate) prints current vs mask-filtered footprints using the same
replicated snipper math as quantify_fixes.py, plus what fraction of lit pixels the filter removes
(and how that splits between the spur column region and everything else).

Run: ~/miniforge3/envs/dinov3/bin/python prototype_mask_filter.py
"""
from __future__ import annotations
import csv
import glob
from pathlib import Path

import numpy as np

from quantify_fixes import (SE, FS, GATES, cc_boxes, merge, gate, freq_samples, time_samples,
                            lit_inside, load)

BYTES = 8
STREAK_HZ = 48e6


def mask_filter_min_bw(m: np.ndarray, min_bw_hz: float) -> np.ndarray:
    """Remove lit runs narrower than min_bw_hz along each row (frequency axis)."""
    rows, cols = m.shape
    min_cols = int(np.ceil(min_bw_hz / (FS / cols)))
    if min_cols <= 1:
        return m
    out = m.copy()
    # Vectorized run-length per row: diff of padded mask marks run starts/ends.
    padded = np.zeros((rows, cols + 2), dtype=np.int8)
    padded[:, 1:-1] = m
    d = np.diff(padded, axis=1)
    for r in range(rows):
        starts = np.flatnonzero(d[r] == 1)
        ends = np.flatnonzero(d[r] == -1)
        for s, e in zip(starts, ends):
            if e - s < min_cols:
                out[r, s:e] = 0
    return out


def footprints(m, min_bw, min_dur):
    rows, cols = m.shape
    frame_samples = rows * 10240
    bs = gate(merge(cc_boxes(m)), rows, cols, frame_samples, min_bw, min_dur)
    return (freq_samples(bs, rows, cols, frame_samples),
            time_samples(bs, rows, cols, frame_samples), bs)


def main():
    rows_out = []
    gates = {k: v for k, v in GATES.items() if k != "default"}
    for det in ["coherent_power", "finetuned_dino_m2", "ground_truth"]:
        for att in [40, 50, 60, 65, 70]:
            fl = sorted(glob.glob(str(SE / f"snip_run/{det}/attenuation_dB_{att}/mask_arrays/*.packed.npz")))
            if not fl:
                continue
            acc = {(g, s): [0, 0] for g in gates for s in ("current", "maskfilter")}
            lit_before = lit_after = spur_removed = 0
            n = 0
            for f in fl:
                m = load(f)
                rows, cols = m.shape
                c0 = int(round((STREAK_HZ / FS + 0.5) * cols))
                n += 1
                lit_before += int(m.sum())
                for g, (mbw, mdur) in gates.items():
                    fs_, ts_, _ = footprints(m, mbw, mdur)
                    acc[(g, "current")][0] += fs_
                    acc[(g, "current")][1] += ts_
                mf = mask_filter_min_bw(m, 100e3)
                lit_after += int(mf.sum())
                spur_removed += int((m[:, c0 - 10:c0 + 11].astype(int) - mf[:, c0 - 10:c0 + 11]).sum())
                for g, (mbw, mdur) in gates.items():
                    mf_g = mf if g == "minsize_100k_5ms" else mask_filter_min_bw(m, 75e3)
                    fs_, ts_, _ = footprints(mf_g, mbw, mdur)
                    acc[(g, "maskfilter")][0] += fs_
                    acc[(g, "maskfilter")][1] += ts_
            sec = n * (rows * 10240) / FS
            removed = lit_before - lit_after
            for g in gates:
                for s in ("current", "maskfilter"):
                    fsamp, tsamp = acc[(g, s)]
                    rows_out.append(dict(
                        detector=det, attenuation_db=att, gate=g, strategy=s, n_frames=n,
                        freq_TB_hr=round(fsamp * BYTES / sec * 3600 / 1e12, 4),
                        time_TB_hr=round(tsamp * BYTES / sec * 3600 / 1e12, 4),
                        lit_removed_pct=round(100 * removed / max(lit_before, 1), 2) if s == "maskfilter" else 0.0,
                        removed_at_spur_pct=round(100 * spur_removed / max(removed, 1), 1) if s == "maskfilter" else 0.0))
                    print(rows_out[-1])
    with open(SE / "prototype_mask_filter.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", SE / "prototype_mask_filter.csv")


if __name__ == "__main__":
    main()

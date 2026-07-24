#!/usr/bin/env python3
"""Quantify candidate fixes for the bounding-box fusion artifact (problem.md), offline on the staged
masks, replicating the real snipper's clustering + decimation math (signal_snip_core.cu).

Strategies compared per (detector, attenuation, gate):
  current   - CC(>=min_px) -> merge(16,80) -> size gate -> store whole bbox (what the snipper does)
  split     - pre-labeling split: columns lit in >=60% of rows ("persistent columns", i.e. the
              streak) are clustered SEPARATELY from the rest; each part is CC+merge+gate+stored.
              Streak becomes its own tall-thin box; bursts stay wide-short. Nothing detected is lost
              unless it fails the gate on its own merits.
  fill10    - current, then DROP any final box with <10% lit-pixel fill (fill-ratio gate).
  suppress  - split, then DROP the persistent-column boxes entirely (treat streak as environment).
  content   - accounting-only reference: bytes = lit mask pixels (each pixel = 10240/512 samples
              decimated ideally). Lower bound; not a snipper change.

For each: frequency-mode TB/hr (mix->lowpass 1.25x oversample->integer decimate per box, as
ddc_extract) and time_only TB/hr (union of box row-spans at full rate), plus lit-pixel retention
(fraction of detected mask pixels inside stored boxes) so the fidelity cost of dropping boxes is
visible.

Gates: default (256px only), minsize (100 kHz AND 5 ms), 75k_1ms (75 kHz AND 1 ms).

Run: ~/miniforge3/envs/dinov3/bin/python quantify_fixes.py [--dets d1,d2] [--attens 60,65,70]
Writes fix_quantification.csv and prints a summary table.
"""
from __future__ import annotations
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from scipy import ndimage

SE = Path(__file__).resolve().parent
FS = 245.76e6
MIN_PX, GAP_R, GAP_C = 256, 16, 80
OVERSAMPLE = 1.25
PERSIST_FRAC = 0.6   # column lit in >=60% of rows -> "persistent column" (streak)
FILL_MIN = 0.10
BYTES = 8
GATES = {"default": (0.0, 0.0), "minsize_100k_5ms": (100e3, 5e-3), "75k_1ms": (75e3, 1e-3)}
FOUR = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def load(f):
    z = np.load(f)
    return np.unpackbits(z["packed"])[: int(z["rows"]) * int(z["cols"])].reshape(
        int(z["rows"]), int(z["cols"]))


def cc_boxes(m):
    """4-connected components >= MIN_PX -> [r0, r1, c0, c1, lit] (mirrors label_components)."""
    lbl, n = ndimage.label(m, structure=FOUR)
    if not n:
        return []
    sz = np.bincount(lbl.ravel())
    out = []
    for i, s in enumerate(ndimage.find_objects(lbl), 1):
        if sz[i] >= MIN_PX:
            out.append([s[0].start, s[0].stop - 1, s[1].start, s[1].stop - 1, int(sz[i])])
    return out


def merge(bs):
    """Gap-merge to a fixed point (mirrors merge_boxes)."""
    changed = True
    while changed:
        changed = False
        out = []
        for b in bs:
            hit = False
            for r in out:
                if (b[0] <= r[1] + GAP_R and r[0] <= b[1] + GAP_R and
                        b[2] <= r[3] + GAP_C and r[2] <= b[3] + GAP_C):
                    r[0] = min(r[0], b[0]); r[1] = max(r[1], b[1])
                    r[2] = min(r[2], b[2]); r[3] = max(r[3], b[3])
                    r[4] += b[4]; hit = changed = True
                    break
            if not hit:
                out.append(b[:])
        bs = out
    return bs


def gate(bs, rows, cols, frame_samples, min_bw, min_dur):
    hz_col = FS / cols
    s_row = (frame_samples / rows) / FS
    keep = []
    for b in bs:
        bw = (b[3] - b[2] + 1) * hz_col
        dur = (b[1] - b[0] + 1) * s_row
        if min_bw > 0 and bw < min_bw:
            continue
        if min_dur > 0 and dur < min_dur:
            continue
        keep.append(b)
    return keep


def freq_samples(bs, rows, cols, frame_samples):
    """Stored decimated samples, frequency mode (mirrors ddc_extract's decim math)."""
    total = 0
    for b in bs:
        n_in = (int(np.ceil((b[1] + 1) / rows * frame_samples))
                - int(np.floor(b[0] / rows * frame_samples)))
        bw = max((b[3] - b[2] + 1) / cols * FS, 1.0)
        keep_bw = bw * OVERSAMPLE
        decim = max(1, int(FS // keep_bw))
        total += (n_in + decim - 1) // decim
    return total


def time_samples(bs, rows, cols, frame_samples):
    """Stored full-rate samples, time_only mode (union of row spans, mirrors merge_time_intervals)."""
    iv = sorted((int(np.floor(b[0] / rows * frame_samples)),
                 int(np.ceil((b[1] + 1) / rows * frame_samples))) for b in bs)
    total, cur_s, cur_e = 0, None, None
    for s, e in iv:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def lit_inside(m, bs):
    keep = np.zeros_like(m, dtype=bool)
    for b in bs:
        keep[b[0]:b[1] + 1, b[2]:b[3] + 1] = True
    return int((m.astype(bool) & keep).sum())


def frame_boxes(m):
    """Label/merge ONCE per frame: merged boxes of the full mask, and of the persistent-column
    split (streak columns vs rest). Gates are applied later per gate config (cheap)."""
    col_occ = m.mean(axis=0)
    persist = col_occ >= PERSIST_FRAC
    full = merge(cc_boxes(m))
    if persist.any():
        b_streak = merge(cc_boxes(np.where(persist[None, :], m, 0)))
        b_rest = merge(cc_boxes(np.where(persist[None, :], 0, m)))
    else:
        b_streak, b_rest = [], full
    return full, b_streak, b_rest


def boxes_for_strategy(pre, strat, rows, cols, frame_samples, min_bw, min_dur):
    full, b_streak, b_rest = pre
    if strat in ("split", "suppress"):
        b_s = gate(b_streak, rows, cols, frame_samples, min_bw, min_dur)
        b_r = gate(b_rest, rows, cols, frame_samples, min_bw, min_dur)
        return b_r if strat == "suppress" else (b_s + b_r)
    bs = gate(full, rows, cols, frame_samples, min_bw, min_dur)
    if strat == "fill10":
        bs = [b for b in bs if b[4] / ((b[1] - b[0] + 1) * (b[3] - b[2] + 1)) >= FILL_MIN]
    return bs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dets", default="coherent_power,finetuned_dino_m2,ground_truth")
    ap.add_argument("--attens", default="40,50,60,65,70")
    a = ap.parse_args()

    strategies = ["current", "split", "fill10", "suppress"]
    rows_out = []
    for det in a.dets.split(","):
        for att in [int(x) for x in a.attens.split(",")]:
            fl = sorted(glob.glob(str(SE / f"snip_run/{det}/attenuation_dB_{att}/mask_arrays/*.packed.npz")))
            if not fl:
                continue
            acc = {(s, g): dict(fsamp=0, tsamp=0, lit_kept=0, nbox=0)
                   for s in strategies for g in GATES}
            lit_total = 0
            content_samp = 0
            n = 0
            for f in fl:
                m = load(f)
                rows, cols = m.shape
                frame_samples = rows * 10240
                lit = int(m.sum())
                lit_total += lit
                n += 1
                pre = frame_boxes(m)
                for g, (mbw, mdur) in GATES.items():
                    for s in strategies:
                        bs = boxes_for_strategy(pre, s, rows, cols, frame_samples, mbw, mdur)
                        e = acc[(s, g)]
                        e["fsamp"] += freq_samples(bs, rows, cols, frame_samples)
                        e["tsamp"] += time_samples(bs, rows, cols, frame_samples)
                        e["lit_kept"] += lit_inside(m, bs)
                        e["nbox"] += len(bs)
            # content accounting: each lit pixel = frame_samples/(rows*cols) ideally-decimated samples
            sec = n * (rows * 10240) / FS
            content_tb = lit_total * (10240 / cols) * BYTES / sec * 3600 / 1e12
            for g in GATES:
                for s in strategies:
                    e = acc[(s, g)]
                    rows_out.append(dict(
                        detector=det, attenuation_db=att, gate=g, strategy=s, n_frames=n,
                        n_boxes=e["nbox"],
                        freq_TB_hr=round(e["fsamp"] * BYTES / sec * 3600 / 1e12, 4),
                        time_TB_hr=round(e["tsamp"] * BYTES / sec * 3600 / 1e12, 4),
                        lit_retention_pct=round(100 * e["lit_kept"] / lit_total, 2) if lit_total else 100.0,
                        content_lowerbound_TB_hr=round(content_tb, 4)))
                    print(rows_out[-1])
    out = SE / "fix_quantification.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", out)


if __name__ == "__main__":
    main()

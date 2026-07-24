#!/usr/bin/env python3
"""Offline-replicated BEFORE curve for the 75 kHz/1 ms gate (the real pre-fix 75k run was never
made): coherent_power, 'current' snipper behavior (no mask filter), all staged attenuations.
The replication is validated elsewhere to match the real pipeline to three decimals.
Writes real_snip_metrics_75k_before_replicated.csv (same key columns as verify_snip output).

Run: ~/miniforge3/envs/dinov3/bin/python replicate_75k_before.py
"""
from __future__ import annotations
import csv
import glob
import re
from pathlib import Path

from quantify_fixes import SE, FS, cc_boxes, merge, gate, freq_samples, time_samples, load

BYTES = 8
MIN_BW, MIN_DUR = 75e3, 1e-3
ATTENS = ["0", "5", "10", "15", "20", "25", "30", "30_v2", "35", "40", "45",
          "50", "55", "60", "65", "70"]


def main():
    rows_out = []
    for att in ATTENS:
        fl = sorted(glob.glob(str(SE / f"snip_run/coherent_power/attenuation_dB_{att}/mask_arrays/*.packed.npz")))
        if not fl:
            continue
        fsamp = tsamp = 0
        n = 0
        for f in fl:
            m = load(f)
            rows, cols = m.shape
            frame_samples = rows * 10240
            bs = gate(merge(cc_boxes(m)), rows, cols, frame_samples, MIN_BW, MIN_DUR)
            fsamp += freq_samples(bs, rows, cols, frame_samples)
            tsamp += time_samples(bs, rows, cols, frame_samples)
            n += 1
        sec = n * (rows * 10240) / FS
        att_num = int(re.match(r"\d+", att).group())
        for mode, samp in (("frequency", fsamp), ("time_only", tsamp)):
            rows_out.append(dict(mode=mode, detector="coherent_power",
                                 file_stem=f"attenuation_dB_{att}", attenuation_db=att_num,
                                 decimated_TB_per_hour=round(samp * BYTES / sec * 3600 / 1e12, 4),
                                 n_frames=n, source="offline_replicated_no_maskfilter"))
        print(att, rows_out[-2]["decimated_TB_per_hour"], rows_out[-1]["decimated_TB_per_hour"])
    out = SE / "real_snip_metrics_75k_before_replicated.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print("wrote", out)


if __name__ == "__main__":
    main()

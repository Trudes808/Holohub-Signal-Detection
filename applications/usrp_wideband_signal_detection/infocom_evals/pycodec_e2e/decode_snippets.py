#!/usr/bin/env python3
"""Detect -> snip -> decode validation over a pycodec composite.

Takes (a) the composite's TX-truth SigMF meta (ground-truth waveform placements
with wfgt:source_mat) and (b) the signal_snipper's SigMF pack output from the
offline detection run, then for every truth placement: finds the snippet
annotations that overlap it in time, extracts their IQ, mixes the placement's
true center to baseband (snippets are already mixed by their own snip center),
and decodes with pycodec against the tiled txBits. Reports per-placement
detection + BER and an aggregate summary.

This is the offline rehearsal of the live chain:
    detector -> signal_snipper -> sigmf_file_sink -> pycodec.decode

Usage:
    python3 decode_snippets.py \
        --truth  .../composites/comprehensive_ordered_py.sigmf-meta \
        --snips  /tmp/usrp_spectrograms/pycodec_e2e/snippets \
        [--library .../generated_waveforms_pycodec] [--pycodec-root ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

PYCODEC_ROOT = os.environ.get(
    "PYCODEC_ROOT", os.path.expanduser("~/Documents/holoscan_waveform_generation"))
sys.path.insert(0, PYCODEC_ROOT)

from scipy.io import loadmat  # noqa: E402
from pycodec.decode import spec_from_metadata  # noqa: E402
from pycodec.psk import demodulate  # noqa: E402


def load_truth(meta_path):
    meta = json.load(open(meta_path))
    fs = float(meta["global"]["core:sample_rate"])
    placements = [a for a in meta["annotations"] if a.get("wfgt:kind") == "waveform"]
    return fs, placements


def load_snips(snip_dir):
    """Return a list of (annotation, pack_data_memmap, pack_meta_global)."""
    out = []
    for mp in sorted(glob.glob(os.path.join(snip_dir, "*.sigmf-meta"))):
        meta = json.load(open(mp))
        data = np.memmap(mp.replace(".sigmf-meta", ".sigmf-data"), dtype="<c8", mode="r")
        for a in meta.get("annotations", []):
            out.append((a, data))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--snips", required=True)
    ap.add_argument("--library", default=os.path.join(PYCODEC_ROOT, "generated_waveforms_pycodec"))
    ap.add_argument("--min-bits", type=int, default=64,
                    help="skip overlaps contributing fewer bits than this")
    args = ap.parse_args()

    fs_orig, placements = load_truth(args.truth)
    snips = load_snips(args.snips)
    print(f"truth placements: {len(placements)}   snippet annotations: {len(snips)}")

    lib_cache = {}
    n_detected, n_decoded, total_err, total_bits = 0, 0, 0, 0
    worst = (0.0, None)
    for i, gt in enumerate(placements):
        gt_start = int(gt["core:sample_start"])
        gt_count = int(gt["core:sample_count"])
        gt_end = gt_start + gt_count
        gt_center = 0.5 * (float(gt["core:freq_lower_edge"]) + float(gt["core:freq_upper_edge"]))
        src = gt["wfgt:source_mat"]
        if src not in lib_cache:
            mat_path = os.path.join(args.library, src)
            m = loadmat(mat_path)
            md = json.load(open(os.path.splitext(mat_path)[0] + ".json"))
            lib_cache[src] = (spec_from_metadata(md),
                              np.asarray(m["txBits"]).ravel().astype(np.uint8))
        spec, tx_bits = lib_cache[src]

        overlaps = []
        for a, data in snips:
            o_start = int(a["wfgt:orig_sample_start"])
            o_end = o_start + int(a["wfgt:orig_sample_count"])
            lo, hi = max(gt_start, o_start), min(gt_end, o_end)
            if hi > lo:
                overlaps.append((lo, hi, a, data))
        if not overlaps:
            print(f"  [{i:2d}] MISS {gt['wfgt:variation'][:56]:56s} no overlapping snippet")
            continue
        n_detected += 1

        err, bits, covered = 0, 0, 0
        for lo, hi, a, data in overlaps:
            o_start = int(a["wfgt:orig_sample_start"])
            decim = int(a.get("wfgt:decimation_factor", 1))
            snip_fs = float(a.get("wfgt:snippet_sample_rate", fs_orig))
            snip_center = float(a.get("wfgt:center_frequency", 0.0))
            in_start = int(a["core:sample_start"]) + (lo - o_start) // decim
            in_count = (hi - lo) // decim
            iq = np.asarray(data[in_start:in_start + in_count], dtype=np.complex64)
            covered += hi - lo

            # bring the placement's true center to baseband inside the snippet
            residual = gt_center - snip_center
            if residual != 0.0:
                n = np.arange(iq.size, dtype=np.float64)
                iq = (iq * np.exp(-2j * np.pi * residual * n / snip_fs)).astype(np.complex64)

            # pre-roll the tiled reference so alignment lag stays small
            sym_offset = int(round((lo - gt_start) / fs_orig * spec.symbol_rate_hz))
            bit_offset = (sym_offset * spec.bits_per_symbol) % tx_bits.size
            nbits_needed = int(iq.size / snip_fs * spec.symbol_rate_hz + 2) * spec.bits_per_symbol
            reps = int(np.ceil((bit_offset + nbits_needed + 8192) / tx_bits.size)) + 1
            reference = np.tile(tx_bits, reps)[bit_offset:]

            if nbits_needed < args.min_bits:
                continue
            try:
                res = demodulate(iq, spec, input_fs_hz=snip_fs,
                                 reference_bits=reference, align_max_lag=4096)
            except Exception as e:  # keep going; report as failure
                print(f"       decode error on overlap: {e}")
                continue
            err += res.bit_errors
            bits += res.bits_compared

        if bits == 0:
            print(f"  [{i:2d}] DET  {gt['wfgt:variation'][:56]:56s} detected but no decodable bits")
            continue
        n_decoded += 1
        ber = err / bits
        total_err += err
        total_bits += bits
        if ber > worst[0]:
            worst = (ber, gt["wfgt:variation"])
        cov = covered / gt_count
        flag = "OK  " if ber == 0.0 else ("ok* " if ber < 1e-3 else "FAIL")
        print(f"  [{i:2d}] {flag} {gt['wfgt:variation'][:56]:56s} "
              f"cov {cov:5.1%}  BER {ber:.2e} ({err}/{bits})")

    print(f"\nplacements {len(placements)}  detected {n_detected}  decoded {n_decoded}")
    if total_bits:
        print(f"aggregate BER over decoded bits: {total_err}/{total_bits} = {total_err/total_bits:.3e}")
        print(f"worst placement BER: {worst[0]:.3e} ({worst[1]})")
    return 0 if (n_decoded == len(placements) and total_err == 0) else 1


if __name__ == "__main__":
    sys.exit(main())

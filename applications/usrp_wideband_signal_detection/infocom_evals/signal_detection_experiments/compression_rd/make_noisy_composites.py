#!/usr/bin/env python3
"""AWGN variants of the ORIGINAL MATLAB composite for the codec-model eval.

comprehensive_ordered is (near) noise-free TX ground truth. One flat noise
floor over a multi-bandwidth composite gives every signal a DIFFERENT in-band
SNR, so nominal labels are calibrated to the MEDIAN per-annotation in-band
PSD, and every annotation in the output meta gets its ACTUAL in-band SNR
(wfgt:snr_db) for honest per-signal binning.

    python3 make_noisy_composites.py [--snrs 30 15 0 -10] [--calib-anns 400]

Writes comprehensive_ordered_awgn<S>db.sigmf-{data,meta} next to the source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMP = Path.home() / ("Documents/holoscan_waveform_generation/composition/"
                      "composites/comprehensive_ordered")
FS = 245.76e6
CHUNK = 1 << 24          # complex samples per streaming chunk (128 MiB)
PSD_SLICE = 1 << 20      # samples used per annotation for in-band power


def inband_psd(data: np.memmap, ann: dict) -> float:
    """Mean signal PSD (power per Hz) inside the annotation's band, from a
    Welch periodogram over a sub-slice of its time span."""
    s, c = int(ann["core:sample_start"]), int(ann["core:sample_count"])
    n = min(c, PSD_SLICE)
    x = np.asarray(data[s + (c - n) // 2 : s + (c - n) // 2 + n])
    nfft = 1 << 14
    nseg = n // nfft
    if nseg == 0:
        return 0.0
    segs = x[: nseg * nfft].reshape(nseg, nfft) * np.hanning(nfft)[None, :]
    psd = np.mean(np.abs(np.fft.fftshift(np.fft.fft(segs, axis=1), axes=1)) ** 2, axis=0)
    win_pow = np.mean(np.hanning(nfft) ** 2)
    psd = psd / (nfft * FS * win_pow)          # power/Hz, two-sided over fs
    f = (np.arange(nfft) - nfft // 2) * (FS / nfft)
    lo, hi = float(ann["core:freq_lower_edge"]), float(ann["core:freq_upper_edge"])
    m = (f >= lo) & (f <= hi)
    return float(np.mean(psd[m])) if m.any() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snrs", nargs="+", type=float, default=[30, 15, 0, -10])
    ap.add_argument("--calib-anns", type=int, default=400)
    args = ap.parse_args()

    meta = json.load(open(f"{COMP}.sigmf-meta"))
    anns = [a for a in meta["annotations"] if a.get("wfgt:kind") == "waveform"]
    data = np.memmap(f"{COMP}.sigmf-data", dtype=np.complex64, mode="r")

    rng = np.random.default_rng(0xC0FFEE)
    calib = rng.choice(len(anns), size=min(args.calib_anns, len(anns)), replace=False)
    print(f"[awgn] calibrating in-band PSD on {len(calib)} annotations...", flush=True)
    psds = {}
    for i in calib:
        psds[i] = inband_psd(data, anns[i])
    median_psd = float(np.median([v for v in psds.values() if v > 0]))
    print(f"[awgn] median in-band PSD {median_psd:.3e} /Hz", flush=True)

    # actual per-annotation PSD for EVERY annotation (for wfgt:snr_db). Reuse
    # calibration values; compute the rest.
    print(f"[awgn] measuring all {len(anns)} annotations...", flush=True)
    for i, a in enumerate(anns):
        a["_psd"] = psds.get(i) if i in psds else inband_psd(data, a)

    for snr in args.snrs:
        n0 = median_psd / (10 ** (snr / 10.0))     # noise PSD for the nominal label
        sigma = np.sqrt(n0 * FS / 2.0)
        tag = f"{'m' if snr < 0 else ''}{abs(int(snr))}"
        out = COMP.parent / f"comprehensive_ordered_awgn{tag}db"
        print(f"[awgn] {snr:+.0f} dB nominal -> sigma {sigma:.4g} -> {out.name}", flush=True)
        wrng = np.random.default_rng(0x5EED ^ int(snr * 1000) & 0xFFFFFFFF)
        with open(f"{out}.sigmf-data", "wb") as f:
            for s0 in range(0, data.size, CHUNK):
                x = np.asarray(data[s0:s0 + CHUNK]).copy()
                x += (sigma * (wrng.standard_normal(x.size) +
                               1j * wrng.standard_normal(x.size))).astype(np.complex64)
                x.tofile(f)
        m = json.loads(json.dumps(meta))
        out_anns = []
        for a_src in meta["annotations"]:
            a = dict(a_src)
            a.pop("_psd", None)
            if a.get("wfgt:kind") == "waveform" and a_src.get("_psd", 0) > 0:
                a["wfgt:snr_db"] = round(10 * np.log10(a_src["_psd"] / n0), 2)
            out_anns.append(a)
        m["annotations"] = out_anns
        m["global"]["wfgt:awgn_nominal_snr_db"] = snr
        json.dump(m, open(f"{out}.sigmf-meta", "w"), indent=1)
    # the clean source also gets per-annotation "SNR" vs a reference floor? No:
    # clean stays unlabeled (effectively noiseless); evals bin it as "clean".
    print("[awgn] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

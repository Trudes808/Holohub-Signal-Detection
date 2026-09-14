#!/usr/bin/env python3
"""
GT benchmark: take the comprehensive_ordered TRANSMIT composite (clean IQ + SigMF annotations = ground
truth) and emit SNR-swept SigMF variants (add complex AWGN at target SNRs). Each variant carries the
SAME annotations (clipped to the subset) so the offline eval rasterizes GT masks and we can measure
DINO-FT recall/precision/IoU vs GT as a function of SNR.

SNR definition: signal reference = mean |IQ|^2 over ACTIVE samples (samples inside any annotation's time
span); noise power = sig_ref / 10^(SNR/10). "clean" = no added noise (the transmit is noiseless).
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np

SRC = Path("/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/composites/comprehensive_ordered")
OUT = Path("/tmp/usrp_spectrograms/snr_bench")          # mappable into the container (/workspace/spectrograms)
SUBSET_SECONDS = 0.75                                    # keep it tractable (~18 detector frames)
SNRS_DB = [None, 20.0, 12.0, 6.0, 0.0]                   # None = clean
SEED = 7


def main():
    meta = json.loads((SRC.with_suffix(".sigmf-meta")).read_text())
    g = meta["global"]; rate = float(g["core:sample_rate"]); dt = g["core:datatype"]
    assert dt == "cf32_le", f"expected cf32_le, got {dt}"
    nsub = int(SUBSET_SECONDS * rate)
    print(f"rate={rate/1e6:.2f} MHz  subset={SUBSET_SECONDS}s = {nsub} samples")

    iq = np.fromfile(str(SRC.with_suffix(".sigmf-data")), dtype=np.complex64, count=nsub)
    print(f"loaded {iq.size} complex samples ({iq.nbytes/1e9:.2f} GB)")

    # annotations within the subset (clip counts)
    ann = []
    for a in meta.get("annotations", []):
        s = int(a.get("core:sample_start", 0)); c = int(a.get("core:sample_count", 0))
        if s >= nsub:
            continue
        b = dict(a); b["core:sample_count"] = min(c, nsub - s); ann.append(b)
    print(f"annotations in subset: {len(ann)} (of {len(meta.get('annotations', []))})")

    # signal reference power = mean|IQ|^2 over ACTIVE samples (union of annotation time spans)
    active = np.zeros(nsub, dtype=bool)
    for a in ann:
        s = int(a["core:sample_start"]); active[s:s + int(a["core:sample_count"])] = True
    p_all = float(np.mean(np.abs(iq) ** 2))
    p_active = float(np.mean(np.abs(iq[active]) ** 2)) if active.any() else p_all
    print(f"signal ref (active-mean |IQ|^2) = {10*math.log10(p_active+1e-30):.1f} dB ; active frac={active.mean():.2f}")

    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    for snr in SNRS_DB:
        tag = "clean" if snr is None else f"snr{int(snr)}"
        d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
        stem = d / f"comprehensive_ordered_{tag}"
        if snr is None:
            out = iq.copy()
        else:
            npow = p_active / (10 ** (snr / 10.0))
            n = (rng.standard_normal(nsub) + 1j * rng.standard_normal(nsub)).astype(np.complex64)
            n *= np.sqrt(npow / 2.0)   # per-quadrature variance npow/2 -> total noise power npow
            out = (iq + n).astype(np.complex64)
        out.tofile(str(stem.with_suffix(".sigmf-data")))
        m = {"global": {"core:datatype": "cf32_le", "core:sample_rate": rate, "core:version": "1.0.0",
                        "core:num_channels": 1,
                        "core:description": f"comprehensive_ordered subset {SUBSET_SECONDS}s, SNR={tag}"},
             "captures": [{"core:sample_start": 0,
                           "core:frequency": float(meta.get("captures", [{}])[0].get("core:frequency", 0.0))}],
             "annotations": ann}
        stem.with_suffix(".sigmf-meta").write_text(json.dumps(m))
        print(f"wrote {tag}: {stem.with_suffix('.sigmf-data')} ({out.nbytes/1e9:.2f} GB)")


if __name__ == "__main__":
    main()

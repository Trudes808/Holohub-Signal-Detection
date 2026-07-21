#!/usr/bin/env python3
"""Extract a REAL noise-only frame (a gap between bursts) from a full capture into generated_inputs/.

Receiver thermal noise is the same regardless of transmit attenuation, so any capture's
annotation-free gap is a genuine noise region. We find the largest gap between annotations, take a
frame's worth of IQ from its interior (away from burst edges), and write it as a cf32_le SigMF file
with NO annotations -> empty GT, so the offline eval measures false-positive / hallucination behavior
on real noise. Verifies the slice is noise-like (non-zero, low power), not end-of-file zeros.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
FRAME = 5_242_880  # complex samples = one offline frame (512 x 10240)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture", default="/home/bqn82/captures/attenuation_dB_0.sigmf-meta",
                   help="Full-capture .sigmf-meta to pull a noise gap from.")
    p.add_argument("--out", default=str(APP_DIR / "generated_inputs" / "noise_region_real.sigmf-data"))
    p.add_argument("--n-samples", type=int, default=FRAME)
    args = p.parse_args()

    meta_path = Path(args.capture)
    m = json.loads(meta_path.read_text())
    data = Path(str(meta_path)[:-len(".sigmf-meta")] + ".sigmf-data")
    total = data.stat().st_size // 8  # cf32_le = 8 bytes/sample
    anns = sorted((int(a.get("core:sample_start", 0)), int(a.get("core:sample_count", 0)))
                  for a in m.get("annotations", []))
    covered, prev, gaps = [], 0, []
    for s, c in anns:
        covered.append((s, s + c))
    for s, e in sorted(covered):
        if s - prev > 0:
            gaps.append((prev, s))
        prev = max(prev, e)
    if total - prev > 0:
        gaps.append((prev, total))
    gaps.sort(key=lambda g: g[1] - g[0], reverse=True)
    if not gaps or (gaps[0][1] - gaps[0][0]) < args.n_samples:
        raise RuntimeError("no annotation-free gap large enough for a frame")
    g0, g1 = gaps[0]
    # take from the interior of the gap (margin from both edges)
    margin = min((g1 - g0 - args.n_samples) // 2, 4 * args.n_samples)
    start = g0 + max(0, margin)
    start = min(start, g1 - args.n_samples)

    mm = np.memmap(data, dtype=np.complex64, mode="r", shape=(total,))
    iq = np.array(mm[start:start + args.n_samples], dtype=np.complex64)
    del mm
    power = float(np.mean(iq.real.astype(np.float64) ** 2 + iq.imag.astype(np.float64) ** 2))
    print(f"[make_noise] {meta_path.name} gap=({g0},{g1}) len={g1-g0} -> take [{start},{start+args.n_samples}]")
    print(f"[make_noise] mean |z|^2 = {power:.4e} ({10*np.log10(power+1e-12):.1f} dB); "
          f"nonzero frac = {np.mean(iq != 0):.3f}")
    if power <= 0 or np.mean(iq != 0) < 0.5:
        raise RuntimeError("extracted region looks like zeros/EOF padding, not noise -- pick another gap/capture")

    out = Path(args.out)
    iq.tofile(out)
    g = m["global"]
    meta = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": g.get("core:sample_rate", 245760000.0),
            "core:version": g.get("core:version", "1.0.0"),
            "core:num_channels": 1,
            "core:description": f"REAL noise-only region from {meta_path.name} gap [{start},{start+args.n_samples}]",
        },
        "captures": [{"core:sample_start": 0, "core:frequency": g.get("core:frequency", 2.4e9)}],
        "annotations": [],
    }
    out.with_suffix(".sigmf-meta").write_text(json.dumps(meta, indent=2))
    print(f"[make_noise] wrote {out} ({args.n_samples} samples) + meta (0 annotations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

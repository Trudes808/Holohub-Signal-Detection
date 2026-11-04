"""GT sanity check: for every placed burst, measure the realised PSD SNR in the wide spectrogram (median dB
inside the GT box minus the scene floor) and compare it with the intended wfgt:snr_psd_db. Lists emitters
whose rendered level falls short (missing content, wrong scaling, aliasing)."""
from __future__ import annotations
import sys, math, numpy as np, torch
import gen_scenes as g, rates as R, frontend as fe
from collections import defaultdict


def check(scenario, rate, seed, verbose=True):
    sc = g.make_scene(scenario, rate, seed)
    F = sc.fft_size; FS = sc.frame_samples; dev = "cuda"
    w = fe._hann_rms(F, "hann", dev)
    res = defaultdict(list)
    for fidx in range(sc.n_frames):
        x = torch.from_numpy(sc.canvas[fidx * FS:(fidx + 1) * FS]).to(dev).reshape(R.ROWS_WIDE, F) * w
        db = (10 * torch.log10((torch.fft.fftshift(torch.fft.fft(x, dim=-1), dim=-1).abs() ** 2) + 1e-12)).cpu().numpy()
        floor = float(np.percentile(db, 20)) + 1.3   # p20 of chi2 dB -> median approx (+1.3 dB)
        f0 = fidx * FS
        for b in sc.bursts:
            s, c = b.sample_start, b.sample_count
            if s >= f0 + FS or s + c <= f0:
                continue
            r0 = max(0, (s - f0) // F); r1 = min(R.ROWS_WIDE, -(-(s + c - f0) // F))
            lo = (b.fc_hz - b.bw_hz / 2 + rate / 2) / rate * F; hi = (b.fc_hz + b.bw_hz / 2 + rate / 2) / rate * F
            c0 = max(0, int(math.floor(lo))); c1 = min(F, int(math.ceil(hi)))
            # inner 60 % of the box to avoid ramps / skirts
            cc0 = int(c0 + 0.2 * (c1 - c0)); cc1 = max(cc0 + 1, int(c1 - 0.2 * (c1 - c0)))
            if r1 <= r0 or cc1 <= cc0 or c0 >= F:
                continue
            box = db[r0:r1, cc0:cc1]
            meas = float(np.median(box)) - floor
            res[b.emitter].append((b.snr_psd_db, meas, b.label, b.bw_hz, b.dur_s, b.fc_hz))
    if verbose:
        print(f"== {scenario} @ {rate/1e6:.2f} MSps seed {seed}: {len(sc.bursts)} bursts")
        for em, L in res.items():
            d = np.array([[a, m] for a, m, *_ in L])
            short = sum(1 for a, m, *_ in L if a >= 6 and m < a - 6)
            print(f"  {em:16s} n={len(L):4d}  intended med {np.median(d[:,0]):6.1f} dB  measured med {np.median(d[:,1]):6.1f} dB  "
                  f"| >6 dB short: {short}  e.g. {[(round(a,1), round(m,1), lab, round(bw/1e6,2)) for a,m,lab,bw,du,fc in L[:3]]}")
    return res


if __name__ == "__main__":
    for sc_, r_ in [("ism_collisions", 17.857e6), ("ism_collisions", 491.52e6), ("cellular_adjacent", 40e6), ("drone_ism", 62.5e6), ("hoppers_focus", 245.76e6), ("band_edge_focus", 30.72e6)]:
        check(sc_, r_, 5)

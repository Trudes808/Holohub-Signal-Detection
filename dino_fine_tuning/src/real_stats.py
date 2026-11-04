"""Measure the real-data statistics the scenario generator (gen_scenes.py) is calibrated from.

Two groups, written to data/real_stats/:

1. receiver floor templates (floor_profiles.json + .png): per reference capture, the per-frequency noise
   floor shape (p10 over time of the wide-FFT dB image, smoothed, median-removed, resampled to 1024 bins),
   the DC spur height/width, the band-edge roll-off and the dB-pixel spread.  References: the X410 at
   491.52 (2.4 GHz OTA), the X410 at 245.76 (ARGUS off-air), the X410 at 30.72 (HCRO noise chunks), a
   B210 at 40 MSps and an X310 at 15.36 MSps (the hard test cases).  The generator applies one of these
   shapes (with jitter) to each synthetic scene so the model never sees a perfectly flat floor.

2. real OTA scene statistics (ota_tile_stats.json + .png) from the coherent-teacher-labelled X410 tiles in
   data/dataset_ota (2.4 GHz + 1 GHz, 491.52 MSps): occupancy, per-component duration / bandwidth /
   level-above-floor, components per tile, noise-pixel level distribution.  These set the duty cycles,
   packet lengths and SNR ranges the ISM / drone scenarios draw from.

Run:  PYTHONPATH=src .venv-ml/bin/python src/real_stats.py [--out data/real_stats]
"""
from __future__ import annotations
import argparse, csv, json, math, os
from pathlib import Path
import numpy as np
import torch

import frontend as fe
import rates as R

HERE = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-genesys-dgx1-Documents-Holohub-Signal-Detection/34efe362-daf9-4094-b44f-3c2b71212f7c/scratchpad/real_refs")
HOME = Path.home()

REFS = [
    # name, path, dtype, rate, n_frames, note
    ("x410_491p52_2g4_ota", SCRATCH / "x410_ota_2400M_g10_demo_look_20260922_015810.sigmf-data", "ci16", 491.52e6, 12,
     "X410 CG_400 decim 1, 2.4 GHz busy band, gain 10"),
    ("x410_245p76_2g45_ota", HOME / "Datasets/ARGUS/OOD_captures/OTA/test_1.sigmf-data", "cf32", 245.76e6, 6,
     "X410 at 245.76 (decim 1 of the 245.76 MCR), 2.45 GHz off-air"),
    ("x410_30p72_915_noise", HOME / "Datasets/x410_recordings/CleanedData/noise/6.2.s2.1830.Txpkg3.noise.43.804-47.131.bin", "cf32", 30.72e6, 20,
     "X410 at 30.72 (decim 8), 915 MHz, signal-free chunk (HCRO)"),
    ("x410_30p72_915_noise_b", HOME / "Datasets/x410_recordings/CleanedData/noise/6.2.gate.CUTx.1238.noise.255.051-258.928.bin", "cf32", 30.72e6, 20,
     "X410 at 30.72 (decim 8), 915 MHz, signal-free chunk (HCRO gate)"),
    ("b210_40_880_cellular", HERE / "data/hard_test_cases/cellular_downlink_880MHz_small_blocks.sigmf-data", "ci16", 40e6, 20,
     "B210 at 40 MSps, 880 MHz cellular/LMR (hard test case; band nearly full -> NOT a clean floor template)"),
    ("x310_15p36_3g6_lte", HERE / "data/hard_test_cases/rx-waveform-td-rec-0-2023_02_23-17_03_59_287.sigmf-data", "cf32", 15.36e6, 1,
     "X310 UBX at 15.36 MSps, 10 MHz LTE at centre (floor only visible at the edges)"),
]


def load_iq(path: Path, dtype: str, offset_samples: int, n: int) -> np.ndarray:
    if dtype == "ci16":
        raw = np.fromfile(str(path), dtype="<i2", count=2 * n, offset=4 * offset_samples)
        m = (raw.size // 2) * 2
        iq = raw[0:m:2].astype(np.float32) + 1j * raw[1:m:2].astype(np.float32)
        return (iq / 32767.0).astype(np.complex64)
    return np.fromfile(str(path), dtype=np.complex64, count=n, offset=8 * offset_samples)


def wide_db(iq: np.ndarray, fft_size: int, device) -> torch.Tensor:
    rows = iq.size // fft_size
    x = torch.from_numpy(iq[: rows * fft_size]).to(device).reshape(rows, fft_size)
    w = fe._hann_rms(fft_size, "hann", device)
    spec = torch.fft.fftshift(torch.fft.fft(x * w, dim=-1), dim=-1)
    return 10.0 * torch.log10(spec.real ** 2 + spec.imag ** 2 + 1e-12)


def smooth1d(v: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(math.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    vp = np.pad(v, r, mode="edge")
    return np.convolve(vp, k, mode="valid")


def to_1024(v: np.ndarray) -> np.ndarray:
    n = v.size
    if n == 1024:
        return v.copy()
    if n % 1024 == 0:
        return v.reshape(1024, n // 1024).mean(axis=1)
    x = np.linspace(0, 1, n); xi = (np.arange(1024) + 0.5) / 1024
    return np.interp(xi, x, v)


def floor_profiles(out: Path, device):
    res = {}
    for name, path, dtype, rate, nfr, note in REFS:
        if not path.exists():
            print(f"[floor] MISSING {name}: {path}"); continue
        fft = R.app_fft_size(rate); frame = R.frame_samples(rate)
        bps = 4 if dtype == "ci16" else 8
        total = path.stat().st_size // bps
        if total < frame:                      # short recording: use every whole row that exists
            frame = (total // fft) * fft
        nfr = max(1, min(nfr, total // frame))
        p10_rows, p50_rows, p90_rows = [], [], []
        pix_std = []
        for f in range(nfr):
            iq = load_iq(path, dtype, f * frame, frame)
            if iq.size < fft * 64:
                break
            db = wide_db(iq, fft, device)
            q = torch.quantile(db, torch.tensor([0.10, 0.50, 0.90], device=device), dim=0).cpu().numpy()
            p10_rows.append(q[0]); p50_rows.append(q[1]); p90_rows.append(q[2])
            # per-pixel spread in the quietest 20% of columns (noise texture check: complex-Gaussian -> ~5.6 dB)
            quiet = np.argsort(q[2] - q[0])[: max(8, fft // 5)]
            pix_std.append(float(db[:, torch.from_numpy(quiet).to(device)].std().item()))
        p10 = np.median(np.stack(p10_rows), axis=0)        # median over frames of the per-frame p10 profile
        p50 = np.median(np.stack(p50_rows), axis=0)
        sigma = max(2.0, 0.005 * fft)
        sm = smooth1d(p10, sigma)
        med = float(np.median(sm))
        prof = sm - med
        # DC spur: raw p10 at the centre bins vs the smoothed local floor
        c = fft // 2
        local = np.median(sm[max(0, c - 64): c + 64])
        raw_c = p10[c - 32: c + 33] - local
        above = raw_c > 3.0
        # contiguous width around the centre
        w = 0
        if above[32]:
            l = 32
            while l > 0 and above[l - 1]: l -= 1
            r_ = 32
            while r_ < 64 and above[r_ + 1]: r_ += 1
            w = r_ - l + 1
        dc_db = float(raw_c[32])
        edge = {}
        for pct in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
            i = int(pct * fft)
            edge[f"{pct:.2f}"] = [float(prof[min(i, fft - 1)]), float(prof[max(0, fft - 1 - i)])]
        res[name] = {
            "path": str(path), "note": note, "rate_hz": rate, "fft_size": fft, "frames_used": len(p10_rows),
            "profile_db_1024": [round(float(x), 3) for x in to_1024(prof)],
            "profile_median_db": med,
            "p50_minus_p10_median_db": float(np.median(p50 - p10)),
            "dc_spur_db": dc_db, "dc_spur_bins_wide": int(w),
            "dc_spur_bins_1024": float(w * 1024 / fft),
            "edge_rolloff_db_lo_hi": edge,
            "quiet_pixel_db_std": float(np.mean(pix_std)),
            "wide_bin_hz": rate / fft,
        }
        print(f"[floor] {name}: fft {fft} frames {len(p10_rows)} | DC spur {dc_db:+.1f} dB x{w} bins | "
              f"edges lo/hi @0% {edge['0.00'][0]:+.1f}/{edge['0.00'][1]:+.1f} dB, @5% {edge['0.05'][0]:+.1f}/{edge['0.05'][1]:+.1f} | "
              f"quiet px std {np.mean(pix_std):.2f} dB | p50-p10 {np.median(p50-p10):.2f} dB")
    (out / "floor_profiles.json").write_text(json.dumps(res, indent=1))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5))
        xf = np.linspace(-0.5, 0.5, 1024)
        for name, r in res.items():
            ax.plot(xf, r["profile_db_1024"], label=f"{name} ({r['rate_hz']/1e6:.2f} MSps, fft {r['fft_size']})", lw=1.2)
        ax.set_xlabel("normalized frequency (fraction of sample rate)"); ax.set_ylabel("floor p10 - median (dB)")
        ax.set_title("Receiver noise-floor shape templates (smoothed per-frequency p10 over time)")
        ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_ylim(-20, 10)
        fig.tight_layout(); fig.savefig(out / "floor_profiles.png", dpi=110)
    except Exception as e:
        print("plot failed", e)
    return res


def ota_tile_stats(out: Path):
    from scipy.ndimage import label as cc_label, find_objects
    root = HERE / "data/dataset_ota"
    rows = list(csv.DictReader(open(root / "frames.csv")))
    stats = {}
    span, floor_frac = 34.0, 0.12   # robust norm: value v -> dB above floor = v*span - floor_frac*span
    ROW_US, BIN_MHZ = 20480 / 491.52e6 * 1e6, 0.48
    for band in ("2400M", "1000M"):
        durs, bws, means, maxs, occs, ncomp, noise_vals, levels_per_col = [], [], [], [], [], [], [], []
        per_split = {}
        for split in ("train", "val"):
            F = np.load(root / f"frames_{split}.npy", mmap_mode="r"); M = np.load(root / f"masks_{split}.npy", mmap_mode="r")
            sel = [r for r in rows if r["split"] == split and band in r["stem"] and "_on_" in r["stem"]]
            per_split[split] = len(sel)
            for r in sel:
                pos = int(r["mem_pos"]); m = np.asarray(M[pos]).astype(bool); f = np.asarray(F[pos]).astype(np.float32) / 255.0
                occs.append(float(m.mean()))
                lab, n = cc_label(m, structure=np.ones((3, 3)))
                ncomp.append(n)
                if n:
                    for i, sl in enumerate(find_objects(lab), start=1):
                        comp = (lab[sl] == i)
                        durs.append((sl[0].stop - sl[0].start) * ROW_US)
                        bws.append((sl[1].stop - sl[1].start) * BIN_MHZ)
                        vals = f[sl][comp]
                        means.append(float(vals.mean() * span - floor_frac * span)); maxs.append(float(vals.max() * span - floor_frac * span))
                nv = f[~m]
                if nv.size:
                    noise_vals.append(np.percentile(nv, [50, 90, 99, 99.9]) * span - floor_frac * span)
        def pct(a, q=(5, 25, 50, 75, 95)):
            a = np.asarray(a, dtype=np.float64)
            return {str(p): float(np.percentile(a, p)) for p in q} if a.size else {}
        nvm = np.mean(np.stack(noise_vals), axis=0) if noise_vals else np.zeros(4)
        stats[band] = {
            "tiles": per_split, "occupancy": pct(occs), "frac_tiles_occ_gt_0": float(np.mean(np.asarray(occs) > 0)),
            "components_per_tile": pct(ncomp), "component_duration_us": pct(durs), "component_bandwidth_mhz": pct(bws),
            "component_mean_db_above_floor": pct(means), "component_max_db_above_floor": pct(maxs),
            "noise_pixel_db_above_floor_p50_p90_p99_p999": [float(x) for x in nvm],
            "n_components": len(durs),
        }
        print(f"[ota {band}] tiles {per_split} occ p50 {stats[band]['occupancy'].get('50',0)*100:.2f}% | comps/tile p50 {stats[band]['components_per_tile'].get('50',0):.0f} | "
              f"dur us p5/50/95 {stats[band]['component_duration_us'].get('5',0):.0f}/{stats[band]['component_duration_us'].get('50',0):.0f}/{stats[band]['component_duration_us'].get('95',0):.0f} | "
              f"bw MHz p5/50/95 {stats[band]['component_bandwidth_mhz'].get('5',0):.1f}/{stats[band]['component_bandwidth_mhz'].get('50',0):.1f}/{stats[band]['component_bandwidth_mhz'].get('95',0):.1f} | "
              f"level dB above floor mean p5/50/95 {stats[band]['component_mean_db_above_floor'].get('5',0):.1f}/{stats[band]['component_mean_db_above_floor'].get('50',0):.1f}/{stats[band]['component_mean_db_above_floor'].get('95',0):.1f} | noise px p50/p99 {nvm[0]:.1f}/{nvm[2]:.1f}")
        stats[band]["_durs"] = durs; stats[band]["_bws"] = bws; stats[band]["_means"] = means
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for band in stats:
            ax[0].hist(np.log10(np.maximum(stats[band]["_durs"], 1)), bins=40, alpha=.5, label=band)
            ax[1].hist(np.log10(np.maximum(stats[band]["_bws"], 0.1)), bins=40, alpha=.5, label=band)
            ax[2].hist(stats[band]["_means"], bins=40, alpha=.5, label=band)
        ax[0].set_xlabel("log10 component duration (us)"); ax[1].set_xlabel("log10 component bandwidth (MHz)"); ax[2].set_xlabel("component mean level above floor (dB)")
        for a in ax: a.legend(); a.grid(alpha=.3)
        fig.suptitle("Real OTA (X410 @491.52, coherent-teacher components)"); fig.tight_layout(); fig.savefig(out / "ota_tile_stats.png", dpi=110)
    except Exception as e:
        print("plot failed", e)
    for band in stats:
        for k in ("_durs", "_bws", "_means"):
            stats[band].pop(k, None)
    (out / "ota_tile_stats.json").write_text(json.dumps(stats, indent=1))
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "data/real_stats"))
    ap.add_argument("--skip-floor", action="store_true"); ap.add_argument("--skip-ota", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not a.skip_floor: floor_profiles(out, dev)
    if not a.skip_ota: ota_tile_stats(out)
    print("DONE ->", out)

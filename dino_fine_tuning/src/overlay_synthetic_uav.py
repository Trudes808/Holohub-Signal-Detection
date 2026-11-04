"""Mix a synthetic UAV link into a REAL capture (mixed synth + real stress case).

Reads a real OTA SigMF (ci16/cf32), estimates its noise floor from the wide-FFT spectrum (p20 per-bin power
-> per-sample noise variance), scales the recording to unit noise power, and places the gen_scenes drone
emitters (OcuSync-like 10/20/40 MHz OFDM video bursts at 80-95 % duty or analog FM video, plus the hopping
control link) at the requested PSD SNR above the real floor. Writes <out>.sigmf-{data,meta} (cf32) with
annotations for the SYNTHETIC bursts only (the real background stays unlabelled) so eval_scenes.py scores
"was the UAV found" in a real ISM background.

  PYTHONPATH=src .venv-ml/bin/python src/overlay_synthetic_uav.py --capture <stem>.sigmf-meta --out <dir> \
      --snr 25 [--video-bw 20e6] [--fc-offset-hz 7e6] [--mode ofdm|fm] [--seed 3] [--max-seconds 0.5]
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

import frontend as fe
import rates as R
import synth_mods as sm
import gen_scenes as g
from eval_scenes import load_iq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--snr", type=float, default=25.0, help="video link PSD SNR above the real floor (dB)")
    ap.add_argument("--ctrl-snr", type=float, default=None, help="control hopper PSD SNR (default snr-8)")
    ap.add_argument("--video-bw", type=float, default=20e6); ap.add_argument("--mode", choices=["ofdm", "fm"], default="ofdm")
    ap.add_argument("--fc-offset-hz", type=float, default=None, help="video centre vs capture centre (default random in +-0.35 R)")
    ap.add_argument("--duty", type=float, default=None); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=0.5); ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    meta_p = Path(a.capture); meta = json.loads(meta_p.read_text()); gl = meta["global"]
    rate = float(gl["core:sample_rate"]); dtype = gl.get("core:datatype", "cf32_le")
    data_p = meta_p.with_suffix(".sigmf-data")
    bps = 4 if dtype.startswith("ci16") else (16 if dtype.startswith("cf64") else 8)
    n_total = data_p.stat().st_size // bps
    F = R.app_fft_size(rate); FS = R.ROWS_WIDE * F
    n_frames = max(1, min(n_total // FS, int(a.max_seconds * rate // FS)))
    n = n_frames * FS
    iq = load_iq(data_p, dtype, 0, n).astype(np.complex64)
    # noise floor: p20 of per-bin power of the first frame, hann-rms window -> per-sample variance = floor / F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(iq[:FS]).to(dev).reshape(R.ROWS_WIDE, F) * fe._hann_rms(F, "hann", dev)
    p = (torch.fft.fft(x, dim=-1).abs() ** 2)
    floor_bin = float(torch.quantile(p.flatten()[:: max(1, p.numel() // 2_000_000)], 0.20).item()) * 1.26  # p20 -> mean of chi2 ~ +1 dB
    sigma2 = floor_bin / F
    scale = np.float32(1.0 / math.sqrt(sigma2))
    canvas = iq * scale                                  # real recording with unit noise power
    T = n / rate
    P = sm.Placer(canvas, rate, rng)
    wide_bin = rate / F
    fc = float(a.fc_offset_hz) if a.fc_offset_hz is not None else float(rng.uniform(-0.35, 0.35)) * rate
    placed = []
    tag = a.tag or f"synthUAV_{a.mode}_{a.video_bw/1e6:g}M_snr{a.snr:g}"
    # video link
    if a.mode == "fm":
        e = g.lib_pick(rng, "Broadband_FM", bw_max=25e6, bw_min=15e6)
        xw, r0, bw = g.lib_wave(e), g.FS_LIB, e.bw_hz
        bursts = [sm.Burst("Broadband_FM", "drone_video_fm", 1, fc, -1e-3, T + 2e-3, bw, a.snr, tag)]
        placed += P.place_train(bursts, xw, r0, wide_bin, 5e-6, (2.0, 30e-3), (logu(rng, 50e-9, 1e-6),), 4.0)
    else:
        nfft = 1024 if a.video_bw <= 10e6 else (2048 if a.video_bw <= 20e6 else 4096); nu = int(nfft * 0.6); scs = a.video_bw / nu
        xw, r0, bw = sm.mod_ofdm(np.random.default_rng(a.seed + 1), nfft, nu, scs, 0.07, 30e-3, 16)
        duty = a.duty if a.duty is not None else float(rng.uniform(0.8, 0.95))
        bursts, t = [], -float(rng.uniform(0, 1e-3))
        while t < T:
            dur = float(rng.uniform(0.7e-3, 2.2e-3)); bursts.append(sm.Burst("DroneVideo", "drone_video", 1, fc, t, dur, bw, a.snr + float(rng.uniform(-1, 1)), tag))
            t += dur + dur * (1 - duty) / max(duty, 1e-3)
        placed += P.place_train(bursts, xw, r0, wide_bin, 3e-6, (2.0, 30e-3), (float(rng.uniform(50e-9, 1e-6)),), 4.0)
    # control / telemetry hopper around the video link
    cbw = float(rng.choice([1e6, 2e6])); span = min(rate, float(rng.uniform(20e6, 80e6)))
    csnr = a.ctrl_snr if a.ctrl_snr is not None else a.snr - 8.0
    xc, rc, _ = sm.mod_gfsk(np.random.default_rng(a.seed + 2), cbw / 1.5, 50e-3, 0.5, 0.5, 2)
    n_ch = int(rng.integers(8, 60)); chans = fc + np.linspace(-span / 2, span / 2, n_ch)
    hops, t = [], -float(rng.uniform(0, 5e-3)); period = float(np.exp(rng.uniform(np.log(2e-3), np.log(20e-3))))
    while t < T:
        dur = float(rng.uniform(100e-6, 500e-6)); hops.append(sm.Burst("DroneCtrl", "drone_ctrl", 2, float(chans[rng.integers(n_ch)]), t, dur, cbw, csnr, tag)); t += period
    placed += P.place_hops(hops, xc, rc, wide_bin, 2e-6)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{meta_p.stem}__{tag}"
    (canvas / scale).astype(np.complex64).tofile(str(stem) + ".sigmf-data")   # back to the recording's own scale
    anns = []
    for b in placed:
        if b.sample_count <= 0: continue
        anns.append({"core:sample_start": int(b.sample_start), "core:sample_count": int(b.sample_count),
                     "core:freq_lower_edge": float(b.fc_hz - b.bw_hz / 2), "core:freq_upper_edge": float(b.fc_hz + b.bw_hz / 2),
                     "core:label": b.label, "wfgt:kind": "waveform", "wfgt:occupied_bw_hz": float(b.bw_hz), "wfgt:power_db": float(b.total_snr_db),
                     "wfgt:snr_psd_db": float(b.snr_psd_db), "wfgt:emitter": b.emitter, "wfgt:emitter_id": int(b.emitter_id), "wfgt:scenario": "real_plus_synth_uav"})
    m2 = {"global": {**{k: v for k, v in gl.items() if k != "core:sha512"}, "core:datatype": "cf32_le",
                     "core:description": f"{gl.get('core:description','')} || first {n_frames} frames ({T:.3f} s) with a SYNTHETIC UAV link added: "
                                         f"{a.mode} video {a.video_bw/1e6:g} MHz at {a.snr:g} dB PSD SNR above the measured floor, centre offset {fc/1e6:+.2f} MHz, "
                                         f"control hopper {cbw/1e6:g} MHz at {csnr:g} dB; annotations cover the synthetic bursts only (real background unlabelled)",
                     "wfgt:scenario": "real_plus_synth_uav", "argus:synthetic_overlay": {"mode": a.mode, "video_bw_hz": a.video_bw, "snr_psd_db": a.snr,
                     "ctrl_snr_psd_db": csnr, "fc_offset_hz": fc, "noise_sigma2_est": sigma2, "seed": a.seed, "source": meta_p.name}},
          "captures": [{"core:sample_start": 0, "core:frequency": meta.get("captures", [{}])[0].get("core:frequency", 0)}],
          "annotations": anns}
    Path(str(stem) + ".sigmf-meta").write_text(json.dumps(m2, indent=1))
    print(f"wrote {stem}.sigmf-* : {n_frames} frames, {len(anns)} synthetic bursts, video fc {fc/1e6:+.2f} MHz @ {a.snr} dB, floor sigma2 {sigma2:.3e}")


def logu(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


if __name__ == "__main__":
    main()

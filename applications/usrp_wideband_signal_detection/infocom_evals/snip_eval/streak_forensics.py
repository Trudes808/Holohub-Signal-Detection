#!/usr/bin/env python3
"""Forensics on the persistent ~48 MHz narrowband streak: real transmitted signal or RX artifact?

Decisive physics test: every transmitted signal passes through the programmable attenuator, so its
received power must fall ~1 dB per dB of attenuation. Anything generated inside the receiver
(LO/clock spur, ADC artifact) is untouched by the attenuator and stays at CONSTANT absolute power
across the whole attenuation sweep, while the noise floor also stays constant.

For each capture attenuation_dB_<A>.sigmf-data this script computes a mean power spectrum over
NFRAMES frames (same 512x10240 FFT grid the detectors use), then tracks:
  - streak_db:   peak power within +/-8 bins of +48 MHz (the streak)
  - ref_db:      peak power within +/-8 bins of +60 MHz (real narrowband BPSK per ground truth)
  - floor_db:    median of the spectrum (noise floor)
Slope of (streak_db - floor_db) vs attenuation ~ 0   -> receiver artifact
Slope of (ref_db    - floor_db) vs attenuation ~ -1  -> real signal (until it hits the floor)

Also renders: power-vs-attenuation figure, and a zoomed spectrogram montage of the 48 MHz
neighborhood across attenuations. Outputs -> figs_minsize/streak_*.png + streak_forensics.csv.

Run: ~/miniforge3/envs/dinov3/bin/python streak_forensics.py
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SE = Path(__file__).resolve().parent
OUT = SE / "figs_minsize"
CAPS = Path("/home/bqn82/captures")
FS = 245.76e6
ROWS, COLS = 512, 10240
FRAME = ROWS * COLS
STREAK_HZ = 48e6          # baseband offset of the streak (abs 2.048 GHz at cf 2 GHz)
REF_HZ = 60e6             # persistent real narrowband BPSK per ground truth
ATTENS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
NFRAMES = 12

col_of = lambda hz: int(round((hz / FS + 0.5) * COLS))
hz_of = lambda c: (c / COLS - 0.5) * FS


def mean_psd_db(path: Path, frames: list[int]) -> np.ndarray:
    acc = np.zeros(COLS)
    used = 0
    for fr in frames:
        iq = np.fromfile(path, dtype=np.complex64, count=FRAME, offset=fr * FRAME * 8)
        if iq.size < FRAME:
            continue
        spec = np.abs(np.fft.fftshift(np.fft.fft(iq.reshape(ROWS, COLS), axis=1), axes=1)) ** 2
        acc += spec.mean(axis=0)
        used += 1
    if used == 0:
        return np.full(COLS, np.nan)
    return 10 * np.log10(acc / used + 1e-20)


def peak_near(psd: np.ndarray, hz: float, halfwin_bins: int = 8):
    c = col_of(hz)
    lo, hi = c - halfwin_bins, c + halfwin_bins + 1
    k = lo + int(np.argmax(psd[lo:hi]))
    return psd[k], hz_of(k)


def zoom_spectrogram(path: Path, frame: int, hz: float, halfwin_bins: int = 60) -> np.ndarray:
    iq = np.fromfile(path, dtype=np.complex64, count=FRAME, offset=frame * FRAME * 8)
    if iq.size < FRAME:
        iq = np.pad(iq, (0, FRAME - iq.size))
    s = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq.reshape(ROWS, COLS), axis=1), axes=1)) + 1e-10)
    c = col_of(hz)
    return s[:, c - halfwin_bins:c + halfwin_bins]


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    psds = {}
    for att in ATTENS:
        f = CAPS / f"attenuation_dB_{att}.sigmf-data"
        if not f.exists():
            continue
        n_frames_total = f.stat().st_size // (FRAME * 8)
        frames = list(np.linspace(20, n_frames_total - 2, NFRAMES).astype(int))
        psd = mean_psd_db(f, frames)
        psds[att] = psd
        floor = float(np.median(psd))
        streak_db, streak_hz = peak_near(psd, STREAK_HZ)
        ref_db, ref_hz = peak_near(psd, REF_HZ)
        rows.append(dict(attenuation_db=att, floor_db=round(floor, 2),
                         streak_abs_db=round(float(streak_db), 2),
                         streak_above_floor_db=round(float(streak_db) - floor, 2),
                         streak_peak_mhz=round(streak_hz / 1e6, 3),
                         ref60_abs_db=round(float(ref_db), 2),
                         ref60_above_floor_db=round(float(ref_db) - floor, 2),
                         ref60_peak_mhz=round(ref_hz / 1e6, 3)))
        print(rows[-1])

    with open(SE / "streak_forensics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    att = np.array([r["attenuation_db"] for r in rows], float)
    streak = np.array([r["streak_abs_db"] for r in rows], float)
    ref = np.array([r["ref60_abs_db"] for r in rows], float)
    floor = np.array([r["floor_db"] for r in rows], float)

    # Fit slopes over clean ranges: the ref while clearly above the floor and not floor-limited
    # (atten <= 60); the streak only at atten >= 20, where real wideband signals overlapping
    # 48 MHz have faded and no longer contaminate the peak measurement.
    vis = ((ref - floor) > 6) & (att <= 60)
    ref_slope = np.polyfit(att[vis], ref[vis], 1)[0] if vis.sum() >= 3 else float("nan")
    hi = att >= 20
    streak_slope = np.polyfit(att[hi], streak[hi], 1)[0]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(att, streak, "o-", color="#d62728", lw=2,
            label=f"streak @ +48 MHz (abs 2048 MHz)  slope={streak_slope:+.2f} dB/dB (atten>=20)")
    ax.plot(att, ref, "s-", color="#1f77b4", lw=2,
            label=f"real BPSK @ +60 MHz (ground truth)  slope={ref_slope:+.2f} dB/dB (while visible)")
    ax.plot(att, floor, "k--", lw=1.5, label="noise floor (median PSD)")
    ax.set_xlabel("programmable attenuation (dB)")
    ax.set_ylabel("mean PSD peak (dB, uncalibrated)")
    ax.set_title("Attenuation sweep: a REAL signal must fall 1 dB per dB of attenuation.\n"
                 "The 48 MHz streak does not -> generated inside the receiver, not transmitted.")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "streak_power_vs_attenuation.png", dpi=150)
    print("wrote", OUT / "streak_power_vs_attenuation.png")

    # Zoomed spectrogram montage across attenuations at the streak.
    show = [a for a in [0, 20, 40, 60, 70, 85] if a in psds]
    fig, axs = plt.subplots(1, len(show), figsize=(3.0 * len(show), 5), sharey=True)
    for ax, a in zip(np.atleast_1d(axs), show):
        z = zoom_spectrogram(CAPS / f"attenuation_dB_{a}.sigmf-data", 100, STREAK_HZ)
        ext = [hz_of(col_of(STREAK_HZ) - 60) / 1e6, hz_of(col_of(STREAK_HZ) + 60) / 1e6,
               ROWS * COLS / FS * 1e3, 0]
        ax.imshow(z, aspect="auto", cmap="magma", extent=ext,
                  vmin=np.percentile(z, 30), vmax=np.percentile(z, 99.9))
        ax.set_title(f"atten {a} dB\n(snr {54 - a:+d} dB)", fontsize=9)
        ax.set_xlabel("MHz")
    np.atleast_1d(axs)[0].set_ylabel("time (ms)")
    fig.suptitle("48 MHz neighborhood, frame 100, all attenuations: the thin line persists unchanged "
                 "while every real signal fades with attenuation", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "streak_zoom_across_attens.png", dpi=150)
    print("wrote", OUT / "streak_zoom_across_attens.png")

    # Full-band mean PSD overlay (a few attens) to spot ALL persistent spurs.
    fig, ax = plt.subplots(figsize=(12, 5))
    for a, colr in [(0, "#999999"), (40, "#1f77b4"), (70, "#2ca02c"), (85, "#d62728")]:
        if a in psds:
            ax.plot(np.linspace(-FS / 2, FS / 2, COLS) / 1e6, psds[a], lw=0.6, alpha=0.8,
                    color=colr, label=f"atten {a} dB")
    ax.axvline(48, color="#d62728", ls=":", lw=1)
    ax.annotate("48 MHz streak", (48, ax.get_ylim()[1]), color="#d62728", fontsize=9,
                ha="left", va="top")
    ax.set_xlabel("baseband offset (MHz)  [absolute = 2000 MHz + x]")
    ax.set_ylabel("mean PSD (dB)")
    ax.set_title("Mean PSD across attenuations: transmitted signals collapse into the floor; "
                 "receiver spurs stay put")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "streak_fullband_psd.png", dpi=150)
    print("wrote", OUT / "streak_fullband_psd.png")


if __name__ == "__main__":
    main()

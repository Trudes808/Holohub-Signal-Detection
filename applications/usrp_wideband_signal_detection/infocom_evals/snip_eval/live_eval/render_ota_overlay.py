#!/usr/bin/env python3
"""Spectrograms of the OTA 500 MSps captures with the coherent-power snipper's bounding boxes
overlaid. The spectrogram is computed directly from the raw cf32 IQ (FFT each 20480-sample row of a
frame -> 512 x 20480 dB image, matching the detector's grid). Boxes are the ACTUAL snipper output
read from the 75 kHz/1 ms soft-label meta (core:freq_*_edge = absolute RF, core:sample_* = absolute
samples, wfgt:frame_number 1-indexed) -- not re-derived, so they are exactly what the pipeline kept.

Run: ~/miniforge3/envs/dinov3/bin/python render_ota_overlay.py   -> ota_overlay_cf{2400,1000}MHz.png
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
CAP_DIR = Path("/tmp/usrp_spectrograms")
SOFTLABEL_DIR = Path("/tmp/usrp_spectrograms/ota_snip_pipeline_75k1ms/soft_labels")

FS = 500e6
ROWS, COLS = 512, 20480
PER_ROW = COLS                      # 20480 samples per FFT row
FRAME = ROWS * PER_ROW              # 10,485,760 samples/frame
S_ROW = PER_ROW / FS               # 40.96 us per row
FRAME_MS = FRAME / FS * 1e3        # 20.97 ms per frame
N_FRAMES_TO_SHOW = 2

CAPS = [("ota_x410_cf2400MHz_500Msps_cf32_10s", 2400e6, "2.4 GHz"),
        ("ota_x410_cf1000MHz_500Msps_cf32_10s", 1000e6, "1.0 GHz")]

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def spectrogram(data_path: Path, frame_1idx: int) -> np.ndarray:
    """FFT each 20480-sample row of frame `frame_1idx` (1-indexed) -> 512x20480 dB, fftshifted."""
    mm = np.memmap(data_path, dtype=np.complex64, mode="r")
    base = (frame_1idx - 1) * FRAME
    block = np.asarray(mm[base:base + FRAME]).reshape(ROWS, PER_ROW)
    spec = np.fft.fftshift(np.fft.fft(block, axis=1), axes=1)
    return 20.0 * np.log10(np.abs(spec) + 1e-6)


def boxes_for_frame(anns: list[dict], frame_1idx: int) -> list[dict]:
    return [a for a in anns if a.get("wfgt:frame_number") == frame_1idx]


for stem, center, label in CAPS:
    data_path = CAP_DIR / f"{stem}.sigmf-data"
    meta = json.loads((SOFTLABEL_DIR / f"{stem}.sigmf-meta").read_text())
    anns = [a for a in meta["annotations"] if a.get("wfgt:soft_label")]
    # pick the busiest frames (most kept boxes) so there is something to show
    from collections import Counter
    busiest = [f for f, _ in Counter(a["wfgt:frame_number"] for a in anns).most_common(N_FRAMES_TO_SHOW)]

    f_lo_mhz, f_hi_mhz = (center - FS / 2) / 1e6, (center + FS / 2) / 1e6
    extent = [f_lo_mhz, f_hi_mhz, FRAME_MS, 0.0]      # x=freq MHz, y=time ms (0 at top)

    fig, axes = plt.subplots(len(busiest), 1, figsize=(14, 4.3 * len(busiest)), squeeze=False)
    for ax, fr in zip(axes[:, 0], busiest):
        spec = spectrogram(data_path, fr)
        vmin, vmax = np.percentile(spec, 30), np.percentile(spec, 99.7)
        ax.imshow(spec, aspect="auto", cmap="magma", extent=extent, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        bxs = boxes_for_frame(anns, fr)
        base = (fr - 1) * FRAME
        for a in bxs:
            t0 = (a["core:sample_start"] - base) / PER_ROW * S_ROW * 1e3
            t1 = t0 + a["core:sample_count"] / PER_ROW * S_ROW * 1e3
            flo, fhi = a["core:freq_lower_edge"] / 1e6, a["core:freq_upper_edge"] / 1e6
            ax.add_patch(Rectangle((flo, t0), fhi - flo, t1 - t0, fill=False,
                                   edgecolor="#39ff14", lw=1.3))
        ax.set_title(f"{label} OTA — frame {fr}: {len(bxs)} snipped boxes "
                     f"(75 kHz / 1 ms gate, green)", fontsize=10)
        ax.set_xlabel("frequency (MHz, absolute RF)")
        ax.set_ylabel("time in frame (ms)")
    fig.suptitle(f"{label} · 500 MSps over-the-air spectrogram + coherent-power snipper boxes",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = HERE / f"ota_overlay_cf{int(center/1e6)}MHz.png"
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out, "| frames", busiest)

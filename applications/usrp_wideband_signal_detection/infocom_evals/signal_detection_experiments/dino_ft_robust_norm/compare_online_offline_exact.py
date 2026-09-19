#!/usr/bin/env python3
"""Honest online(loopback RT)-vs-offline DINO-FT M3 comparison on the SAME capture, same config.

Offline: run_cuda_dino_offline_file output (mask_arrays + spectrogram_tensors, keyed by frame_number).
Online:  loopback debug dump (mask_ch0_* + spec_ch0_*), frame_number = chdr_batch_index, in stream order.

IMPORTANT METHODOLOGY NOTE (this replaces an earlier, overstated comparison):
  The capture is 491,520,000 samples = exactly 46.875 detector frames (10,485,760 samples/frame), i.e.
  NOT a whole number of frames. `tcpreplay --loop K` therefore mis-aligns the detector's frame boundaries
  after the first pass: loop 1 is a clean pass over the 46 whole frames, but loops 2..K start mid-frame and
  every frame is a shifted splice of two capture frames. Those shifted inputs legitimately produce different
  masks. An earlier "best-IoU over all online frames" metric hid this, because loop 1 always supplies a clean
  copy of every offline frame so a max() returns ~1.0 regardless of the looped copies. That was overstated.

  The correct metric is a DIRECT 1:1 alignment, per loop segment: online frame j -> offline frame (j mod N).
  Loop 1 is the real online-vs-offline test (RT reproducing offline on one clean pass); loops 2..K measure
  only the tcpreplay loop-seam artifact and are reported separately, not folded into the headline number.

  Takeaway: on one clean pass the RT pipeline reproduces the offline masks bit-for-bit; the looped-copy
  mismatch is a test-rig artifact of a non-frame-aligned pcap, NOT an RT bug. The continuous live radio
  stream has no loop seam and is unaffected. For a fully clean 1:1 across all frames, use --loop 1 or trim
  the pcap to a whole number of frames.
"""
from __future__ import annotations
import csv, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OFF = "/tmp/usrp_spectrograms/status_offline/m3"
ON = "/tmp/usrp_spectrograms/status_online/dino"
OUT = os.path.join(os.path.dirname(__file__), "results")


def load_offline_ordered():
    """Return offline masks + dB in frame_number order (index 0..N-1)."""
    man = {int(r["frame_number"]): r for r in csv.DictReader(open(f"{OFF}/frame_manifest.csv")) if r.get("mask_npy")}
    masks, dBs = [], []
    for fn in sorted(man):
        r = man[fn]
        masks.append(np.load(f"{OFF}/{r['mask_npy']}").astype(bool))
        t = np.load(f"{OFF}/{r['spectrogram_tensor_npy']}")
        dBs.append(10*np.log10(t.real**2+t.imag**2+1e-12) if np.iscomplexobj(t) else t.astype(np.float32))
    return masks, dBs


def load_online_ordered():
    """Return online masks in stream order (index 0.. across all loops)."""
    return [np.load(p).astype(bool) for p in sorted(glob.glob(f"{ON}/mask_ch0_*.npy"))]


def iou(a, b):
    u = (a | b).sum()
    return 1.0 if u == 0 else float((a & b).sum() / u)


def main():
    os.makedirs(OUT, exist_ok=True)
    off_m, off_dB = load_offline_ordered()
    on_m = load_online_ordered()
    if not off_m or not on_m:
        print(f"missing data (offline {len(off_m)}, online {len(on_m)})"); return
    N = len(off_m)
    print(f"offline {N} frames, online {len(on_m)} frames (~{len(on_m)/N:.2f} loops)")

    # --- honest direct 1:1 per loop segment: online j -> offline (j mod N) ---
    seg_names = ["loop 1 (clean pass)"] + [f"loop {k+1} (after seam)" for k in range(1, (len(on_m)+N-1)//N)]
    per_seg = []
    for k, name in enumerate(seg_names):
        idx = [j for j in range(k*N, min((k+1)*N, len(on_m)))]
        v = np.array([iou(on_m[j], off_m[j % N]) for j in idx])
        if len(v):
            per_seg.append((name, v))
            print(f"  {name:22s}: mean {v.mean():.3f} median {np.median(v):.3f} "
                  f">=0.95 {100*(v>=0.95).mean():3.0f}%  ({len(v)} frames)")
    # headline = loop 1 only
    loop1 = per_seg[0][1]
    ident = sum(np.array_equal(on_m[j], off_m[j]) for j in range(N))
    print(f"HEADLINE (loop 1, the real RT-vs-offline test): mean IoU {loop1.mean():.3f}, "
          f"{ident}/{N} pixel-identical, {100*(loop1>=0.95).mean():.0f}% >=0.95")

    # --- figure: per-frame IoU vs stream index, colored by loop, seam marked ---
    fig, ax = plt.subplots(figsize=(13, 4))
    colors = ["#2fb46b", "#c8442e", "#e8791f", "#7a5cd0"]
    for j in range(len(on_m)):
        seg = j // N
        ax.scatter(j, iou(on_m[j], off_m[j % N]), s=14, c=colors[seg % len(colors)])
    for k in range(1, (len(on_m)+N-1)//N):
        ax.axvline(k*N, color="#888", ls="--", lw=0.8)
    ax.axhline(0.95, color="#888", ls=":", lw=0.7)
    ax.set_xlabel("online frame index (green = loop 1 clean pass | others = looped copies; "
                  "dashed = pcap loop seam)")
    ax.set_ylabel("direct 1:1 IoU vs offline")
    ax.set_ylim(-0.02, 1.03); ax.grid(alpha=0.3)
    ax.set_title("DINO-FT M3 online vs offline (direct 1:1): loop 1 near-exact; looped copies mis-align "
                 "at the seam\n(capture = 46.875 frames -> tcpreplay loop shifts frame boundaries; "
                 "a test artifact, not an RT bug)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(OUT, "online_vs_offline_perloop.png")
    fig.savefig(p, dpi=110); print("wrote", p)


if __name__ == "__main__":
    main()

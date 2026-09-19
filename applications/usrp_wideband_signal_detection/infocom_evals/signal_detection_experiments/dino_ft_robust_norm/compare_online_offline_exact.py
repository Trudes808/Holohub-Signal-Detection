#!/usr/bin/env python3
"""Exact online(loopback)-vs-offline DINO-FT M3 comparison on the SAME capture, same config.

Offline: run_cuda_dino_offline_file output (mask_arrays + spectrogram_tensors, keyed by frame_number).
Online:  loopback debug dump (mask_ch0_* + spec_ch0_* + manifest, frame_number = chdr_batch_index).
The loopback loops the pcap, so online frame_number k maps to offline frame ((k-1) mod N). We search the
best constant offset (frame-phase between DPDK batching and file chunking), report per-frame IoU + exact
match %, and render low/medium/high-density frames with both masks on the spectrogram.
"""
from __future__ import annotations
import csv, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OFF = "/tmp/usrp_spectrograms/status_offline/m3"
ON = "/tmp/usrp_spectrograms/status_online/dino"
OUT = os.path.join(os.path.dirname(__file__), "results")


def load_offline():
    man = {int(r["frame_number"]): r for r in csv.DictReader(open(f"{OFF}/frame_manifest.csv")) if r.get("mask_npy")}
    frames = {}
    for fn, r in man.items():
        mk = np.load(f"{OFF}/{r['mask_npy']}").astype(np.uint8)
        t = np.load(f"{OFF}/{r['spectrogram_tensor_npy']}")
        dB = 10*np.log10(t.real**2+t.imag**2+1e-12) if np.iscomplexobj(t) else t.astype(np.float32)
        frames[fn] = (mk, dB)
    return frames


def load_online():
    man = f"{ON}/mask_dump_manifest.csv"
    frames = {}
    if os.path.exists(man):
        for r in csv.DictReader(open(man)):
            fn = int(r["frame_number"]); p = os.path.join(ON, os.path.basename(r["mask_npy"]))
            if os.path.exists(p):
                frames[fn] = np.load(p).astype(np.uint8)
    else:
        for p in sorted(glob.glob(f"{ON}/mask_ch0_*.npy")):
            fn = int(os.path.basename(p).split("_")[-1].split(".")[0]); frames[fn] = np.load(p).astype(np.uint8)
    return frames


def iou(a, b):
    a = a > 0; b = b > 0; u = (a | b).sum()
    return 1.0 if u == 0 else (a & b).sum()/u


def main():
    os.makedirs(OUT, exist_ok=True)
    off = load_offline(); on = load_online()
    if not off or not on:
        print(f"missing data (offline {len(off)}, online {len(on)})"); return
    N = max(off) - min(off) + 1
    off0 = min(off)
    # map online fn -> offline index (0-based within one loop), search best constant offset
    on_items = sorted(on.items())
    best = (0, -1, [])
    for off_shift in range(N):
        ious = []
        for fn, m_on in on_items:
            oidx = ((fn - 1 + off_shift) % N) + off0
            if oidx in off and off[oidx][0].shape == m_on.shape:
                ious.append(iou(m_on, off[oidx][0]))
        if ious and np.mean(ious) > best[1]:
            best = (off_shift, float(np.mean(ious)), ious)
    shift, miou, ious = best
    ious = np.array(ious)
    print(f"offline {len(off)} frames, online {len(on)} frames, loop N={N}")
    print(f"best frame-offset {shift}: mean IoU {miou:.3f}  median {np.median(ious):.3f}  "
          f"min {ious.min():.3f}  exact(IoU=1.0) {100*(ious>=0.999).mean():.0f}%  IoU>=0.95 {100*(ious>=0.95).mean():.0f}%")

    # per-offline-frame occupancy -> pick low/med/high density frames
    occ = {fn: off[fn][0].mean() for fn in off}
    order = sorted(off, key=lambda f: occ[f])
    lo = order[len(order)//6]; md = order[len(order)//2]; hi = order[-2]
    picks = [("low density", lo), ("medium density", md), ("high density", hi)]
    # for each pick, find the online frame mapping to it
    def online_for(oidx):
        for fn, m in on_items:
            if ((fn - 1 + shift) % N) + off0 == oidx:
                return m
        return None
    def dsc(a, f=20, agg=np.mean):
        c = (a.shape[1]//f)*f; return agg(a[:, :c].reshape(a.shape[0], a.shape[1]//f, f), axis=2)
    fig, axes = plt.subplots(len(picks), 2, figsize=(17, 3.0*len(picks)), squeeze=False)
    for i, (label, oidx) in enumerate(picks):
        mk_off, dB = off[oidx]; mk_on = online_for(oidx)
        dBd = dsc(dB); vmin, vmax = np.percentile(dBd, 5), np.percentile(dBd, 99.5)
        for j, (name, mk) in enumerate([("OFFLINE", mk_off), ("ONLINE (loopback RT)", mk_on)]):
            ax = axes[i][j]
            ax.imshow(dBd, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax, interpolation="nearest")
            if mk is not None:
                mkd = dsc(mk.astype(np.float32), agg=np.max)
                if mkd.any(): ax.contour(mkd, levels=[0.5], colors=(0.24,1.0,0.35), linewidths=0.7)
            ij = iou(mk_on, mk_off) if mk is not None else float("nan")
            ax.set_title(f"{name} — {label} (frame {oidx}, occ {occ[oidx]*100:.2f}%"
                         + (f", IoU {ij:.3f}" if j==1 else "") + ")", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"DINO-FT M3: online (loopback RT) vs offline on the SAME capture — mean IoU {miou:.3f} (offset {shift})",
                 fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.98])
    p = os.path.join(OUT, "online_vs_offline_exact.png")
    fig.savefig(p, dpi=110); print("wrote", p)


if __name__ == "__main__":
    main()

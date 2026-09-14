#!/usr/bin/env python3
"""
DINO-FT offline-vs-loopback A/B analysis.

Both legs run the IDENTICAL finetuned_dino_detector operator (real_time_downsample=true, wide-FFT ->
bilinear resize -> circular edge -> flatten). The ONLY difference is how IQ reaches the operator:
  - offline : run_cuda_dino_offline_file.py reads the SigMF file directly (clean, deterministic)
  - loopback: the same capture, packetized to a CHDR pcap, replayed over DPDK at the LIVE 491.52 rate
So any divergence in the emitted masks localizes to the real-time ingest, not the model.

Masks from both legs are 512 x 20480 uint8 on the SAME wide-FFT grid (same frame geometry:
20 pkts/fft * 1024 samp/pkt * 512 ffts/batch = 10,485,760 complex samples/frame). Because the two
legs are NOT frame-aligned (loopback loops the 1 s capture and may drop frames under load), the
comparison is on ingest-robust AGGREGATE statistics, plus per-leg visual rasters:
  - occupancy spectrum  : per-frequency fraction-of-time detected  (missing signals / spurious bars)
  - occupancy raster    : frame x freq heatmap                     (temporal structure)
  - edge occupancy      : outer 12.5% cols vs interior             (edge/border false positives)
  - full-width bar rate : fraction of mask rows lit across >50% BW (spurious full-width bars)
  - continuity          : for persistent "signal" columns, on-fraction + flicker transitions
                                                                   (broken-up continuous signals)

Usage:
  python3 compare_offline_vs_loopback.py \
      --offline /tmp/usrp_spectrograms/offline_eval/cuda_dino_finetuned_rt/<stem> \
      [--loopback /tmp/usrp_spectrograms/loopback_eval/dino_ft_rt/mask_arrays] \
      --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reuse the shared mask loader/resizer from the sibling metrics lib.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mask_eval_metrics as mem  # noqa: E402

RASTER_COLS = 1024          # column bins for rasters / occupancy plots (down from 20480)
EDGE_FRAC = 0.125           # outer 12.5% of columns per side = "edge"
BAR_ROW_FRAC = 0.50         # a mask row lit across > this fraction of BW = a full-width bar
SIGNAL_COL_PRESENT = 0.30   # a column present in > this fraction of frames = a "continuous" candidate


def bin_cols(vec_or_mat: np.ndarray, nbins: int) -> np.ndarray:
    """Mean-pool the last axis into nbins (handles non-divisible widths by trimming the remainder)."""
    a = np.asarray(vec_or_mat, dtype=np.float64)
    w = a.shape[-1]
    step = w // nbins
    if step < 1:
        return a
    trimmed = a[..., : step * nbins]
    new_shape = trimmed.shape[:-1] + (nbins, step)
    return trimmed.reshape(new_shape).mean(axis=-1)


def offline_mask_paths(run_dir: Path) -> list[tuple[int, Path]]:
    """(frame_number, mask_npy_path) from the offline frame_manifest.csv."""
    out = []
    for r in mem.load_manifest(run_dir):
        rel = r.get("mask_npy")
        if rel:
            out.append((int(r["frame_number"]), run_dir / rel))
    return sorted(out, key=lambda x: x[0])


def loopback_mask_paths(dump_dir: Path) -> list[tuple[int, Path]]:
    """(frame_number, mask_npy_path) from the dump's mask_dump_manifest.csv (or dir glob fallback)."""
    man = dump_dir / "mask_dump_manifest.csv"
    out = []
    if man.is_file():
        with man.open() as f:
            for row in csv.DictReader(f):
                out.append((int(row["frame_number"]), dump_dir / row["mask_npy"]))
    else:
        for p in sorted(dump_dir.glob("mask_*.npy")):
            out.append((0, p))
    return out


def analyse(paths: list[tuple[int, Path]], label: str) -> dict:
    """Stream masks (one at a time -- the loopback set can be many GB) and accumulate aggregate stats."""
    col_on = None          # sum of on-pixels per full-res column
    total_rows = 0
    frame_occ = []         # per-frame overall occupancy
    edge_on = interior_on = 0.0
    edge_px = interior_px = 0
    bar_rows = 0
    raster = []            # per-frame binned occupancy  [nframes, RASTER_COLS]
    col_active = []        # per-frame per-binned-col "any on"  [nframes, RASTER_COLS] bool
    rows0 = cols0 = None

    n = 0
    for fn, p in paths:
        m = mem.load_mask_any(p)
        if m is None:
            continue
        mb = (m > 0)
        if col_on is None:
            col_on = np.zeros(mb.shape[1], dtype=np.float64)
            rows0, cols0 = mb.shape
        if mb.shape[1] != col_on.shape[0]:
            mb = mem.resize_mask_nearest(mb.astype(np.uint8), mb.shape[0], col_on.shape[0]) > 0
        col_on += mb.sum(axis=0)
        total_rows += mb.shape[0]
        frame_occ.append(float(mb.mean()))

        ecols = max(1, int(EDGE_FRAC * mb.shape[1]))
        edge_on += float(mb[:, :ecols].sum() + mb[:, -ecols:].sum())
        edge_px += 2 * ecols * mb.shape[0]
        interior_on += float(mb[:, ecols:-ecols].sum())
        interior_px += (mb.shape[1] - 2 * ecols) * mb.shape[0]

        rowfrac = mb.mean(axis=1)
        bar_rows += int((rowfrac > BAR_ROW_FRAC).sum())

        raster.append(bin_cols(mb.mean(axis=0), RASTER_COLS))                # per-frame occupancy per col-bin
        col_active.append(bin_cols(mb.any(axis=0).astype(np.float64), RASTER_COLS) > 0)
        n += 1

    if n == 0:
        return {"label": label, "n_frames": 0}

    # raster currently holds bin_cols over the 2D mask (mean over the col-block only, rows preserved).
    # Recompute cleanly: mean over rows first, then bin.
    # (Re-derive from col_on/total for the spectrum; raster stored per-frame below.)
    occ_spectrum_full = col_on / max(1, total_rows)                          # [cols0]
    occ_spectrum = bin_cols(occ_spectrum_full, RASTER_COLS)                  # [RASTER_COLS]

    raster = np.array(raster)                                                # [n, RASTER_COLS]
    col_active = np.array(col_active)                                        # [n, RASTER_COLS] bool

    present_rate = col_active.mean(axis=0)                                   # [RASTER_COLS]
    transitions = np.abs(np.diff(col_active.astype(np.int8), axis=0)).sum(axis=0)  # per col
    signal_cols = present_rate > SIGNAL_COL_PRESENT
    n_signal = int(signal_cols.sum())
    cont_on_fraction = float(present_rate[signal_cols].mean()) if n_signal else float("nan")
    cont_transitions = float(transitions[signal_cols].mean()) if n_signal else float("nan")

    return {
        "label": label,
        "n_frames": n,
        "mask_rows": rows0,
        "mask_cols": cols0,
        "global_occ_pct": 100.0 * float(np.mean(frame_occ)),
        "frame_occ_pct_min": 100.0 * float(np.min(frame_occ)),
        "frame_occ_pct_max": 100.0 * float(np.max(frame_occ)),
        "edge_occ_pct": 100.0 * (edge_on / max(1, edge_px)),
        "interior_occ_pct": 100.0 * (interior_on / max(1, interior_px)),
        "edge_to_interior_ratio": (edge_on / max(1, edge_px)) / max(1e-9, interior_on / max(1, interior_px)),
        "full_width_bar_row_pct": 100.0 * bar_rows / max(1, total_rows),
        "n_signal_cols": n_signal,
        "continuity_on_fraction": cont_on_fraction,
        "continuity_mean_transitions": cont_transitions,
        "_occ_spectrum": occ_spectrum,
        "_raster": raster,
        "_frame_occ": np.array(frame_occ),
    }


def spec_db(tensor_path: Path) -> np.ndarray:
    t = np.load(tensor_path)
    return 20.0 * np.log10(np.abs(t).astype(np.float64) + 1e-6)


def render_spectrum(offline: dict, loopback: dict | None, center_hz: float, rate_hz: float, out: Path):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = (np.linspace(-0.5, 0.5, RASTER_COLS) * rate_hz + center_hz) / 1e9
    ax.plot(x, 100 * offline["_occ_spectrum"], color="#1f6feb", lw=1.4, label=f"offline (file)  n={offline['n_frames']}")
    if loopback and loopback.get("n_frames"):
        ax.plot(x, 100 * loopback["_occ_spectrum"], color="#e8590c", lw=1.4, alpha=0.85,
                label=f"loopback (DPDK)  n={loopback['n_frames']}")
    ax.set_xlabel("frequency (GHz)")
    ax.set_ylabel("occupancy (% of time detected)")
    ax.set_title("DINO-FT occupancy spectrum — offline vs loopback (same capture, same operator)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def render_rasters(offline: dict, loopback: dict | None, out: Path):
    n = 2 if (loopback and loopback.get("n_frames")) else 1
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4.6), squeeze=False)
    for ax, d, name in zip(axes[0], [offline, loopback][:n], ["offline (file)", "loopback (DPDK)"][:n]):
        im = ax.imshow(100 * d["_raster"], aspect="auto", origin="lower", cmap="magma",
                       vmin=0, vmax=max(1.0, np.percentile(100 * d["_raster"], 99.5)))
        ax.set_title(f"{name}: occupancy raster (frame x freq)")
        ax.set_xlabel("freq bin"); ax.set_ylabel("frame")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="occ %")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def render_frames(run_dir: Path, out: Path, n_show: int = 4):
    """Offline sample frames: raw spectrogram (dB) with the DINO-FT mask overlaid."""
    man = {int(r["frame_number"]): r for r in mem.load_manifest(run_dir)}
    rows = []
    for fn, r in man.items():
        if r.get("mask_npy") and r.get("spectrogram_tensor_npy"):
            rows.append((fn, r))
    # pick the frames with the most detections (most informative) + spread
    def occ(r):
        m = mem.load_mask_any(run_dir / r[1]["mask_npy"])
        return 0.0 if m is None else float((m > 0).mean())
    rows.sort(key=occ, reverse=True)
    picks = rows[:n_show]
    if not picks:
        return
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 3.0 * len(picks)), squeeze=False)
    for ax, (fn, r) in zip(axes[:, 0], picks):
        db = spec_db(run_dir / r["spectrogram_tensor_npy"])
        m = mem.load_mask_any(run_dir / r["mask_npy"]) > 0
        # downsample cols for display
        dbd = bin_cols(db, 2048)
        md = bin_cols(m.astype(np.float64), 2048) > 0.0
        vmin, vmax = np.percentile(dbd, 5), np.percentile(dbd, 99.5)
        ax.imshow(dbd, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        overlay = np.zeros((*md.shape, 4))
        overlay[md] = [1, 0, 0, 0.5]
        ax.imshow(overlay, aspect="auto", origin="lower")
        ax.set_title(f"offline RT — frame {fn}  (occ {100*float(m.mean()):.2f}%)  raw spectrogram + mask (red)")
        ax.set_xlabel("freq bin"); ax.set_ylabel("time row")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", required=True, help="offline run dir (has frame_manifest.csv)")
    ap.add_argument("--loopback", default=None, help="loopback mask dump dir (has mask_dump_manifest.csv)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--center-hz", type=float, default=2.4e9)
    ap.add_argument("--rate-hz", type=float, default=491.52e6)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    off = analyse(offline_mask_paths(Path(args.offline)), "offline")
    lb = None
    if args.loopback and Path(args.loopback).exists():
        lb = analyse(loopback_mask_paths(Path(args.loopback)), "loopback")

    # scalar comparison
    summary = {"offline": {k: v for k, v in off.items() if not k.startswith("_")}}
    if lb:
        summary["loopback"] = {k: v for k, v in lb.items() if not k.startswith("_")}
        a, b = off["_occ_spectrum"], lb["_occ_spectrum"]
        if a.std() > 0 and b.std() > 0:
            summary["occ_spectrum_pearson_r"] = float(np.corrcoef(a, b)[0, 1])
        summary["occ_spectrum_mae_pct"] = float(np.mean(np.abs(a - b)) * 100)
        summary["global_occ_ratio_loopback_over_offline"] = (
            lb["global_occ_pct"] / off["global_occ_pct"] if off["global_occ_pct"] else float("nan"))

    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # per-column occupancy CSV
    with (out / "occupancy_spectrum.csv").open("w", newline="") as f:
        w = csv.writer(f)
        hdr = ["col_bin", "freq_ghz", "offline_occ_pct"] + (["loopback_occ_pct"] if lb else [])
        w.writerow(hdr)
        xf = (np.linspace(-0.5, 0.5, RASTER_COLS) * args.rate_hz + args.center_hz) / 1e9
        for i in range(RASTER_COLS):
            row = [i, f"{xf[i]:.6f}", f"{100*off['_occ_spectrum'][i]:.5f}"]
            if lb:
                row.append(f"{100*lb['_occ_spectrum'][i]:.5f}")
            w.writerow(row)

    render_spectrum(off, lb, args.center_hz, args.rate_hz, out / "occupancy_spectrum.png")
    render_rasters(off, lb, out / "occupancy_raster.png")
    render_frames(Path(args.offline), out / "offline_sample_frames.png")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

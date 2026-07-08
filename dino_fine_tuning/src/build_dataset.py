"""Build the DINOv3 signal/noise fine-tuning dataset from SigMF captures.

Two passes:
  A. PLAN  - scan candidate frames per capture (mask-only, no FFT), classify
             signal/noise, balance & cap, assign a temporal train/val/test split.
  B. WRITE - materialize spectrogram images (GPU FFT) + GT masks into per-split
             memmapped .npy stacks, and write frames.csv + regions.csv indices.

Global dB->uint8 mapping is calibrated once over all captures so that a strongly
attenuated (low-SNR) frame really does look fainter than a 0 dB frame -- critical
for the low-SNR study.

Usage:
  python build_dataset.py --config configs/dataset.yaml            # full build
  python build_dataset.py --config configs/dataset.yaml \
        --limit-captures attenuation_dB_0,attenuation_dB_45 \
        --max-frames 300 --out-suffix _smoke                       # smoke subset
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import rfdata as rf

FRAME_H, FRAME_W = None, None  # set from config at runtime (frame_rows, nfft)


def log(msg: str):
    print(f"[build] {msg}", flush=True)


def discover_captures(captures_dir: Path, limit: list[str] | None) -> list[rf.Capture]:
    metas = sorted(captures_dir.glob("*.sigmf-meta"))
    caps = []
    for m in metas:
        if limit and m.stem not in limit:
            continue
        caps.append(rf.load_capture(m))
    return caps


# --------------------------------------------------------------------------- #
# Calibration: global dB -> uint8 range
# --------------------------------------------------------------------------- #
def calibrate_db_range(caps, nfft, rows, per_file, seed, device):
    rng = np.random.default_rng(seed)
    fsamp = nfft * rows
    pooled = []
    for cap in caps:
        n_frames = cap.n_samples // fsamp
        if n_frames <= 0:
            continue
        picks = rng.choice(n_frames, size=min(per_file, n_frames), replace=False)
        mm = cap.memmap()
        batch = []
        for fi in picks:
            s = int(fi) * fsamp
            batch.append(np.asarray(mm[s:s + fsamp], dtype=np.complex64))
        iq = torch.from_numpy(np.stack(batch)).to(device)
        db = rf.frames_to_db(iq, nfft, rows).cpu().numpy().ravel()
        # subsample pixels to keep memory bounded
        pooled.append(rng.choice(db, size=min(db.size, 200_000), replace=False))
    allpx = np.concatenate(pooled)
    vmin = float(np.percentile(allpx, 1.0))
    vmax = float(np.percentile(allpx, 99.9))
    log(f"calibrated global dB range: vmin={vmin:.2f} vmax={vmax:.2f} "
        f"(from {len(caps)} captures, {allpx.size} pooled px)")
    return vmin, vmax


# --------------------------------------------------------------------------- #
# Pass A: plan frames
# --------------------------------------------------------------------------- #
def plan_frames(cap, cfg, rng):
    """Return list of dicts: {frame_index, abs_start, is_signal, n_signal_px, boxes}."""
    nfft, rows = cfg["nfft"], cfg["frame_rows"]
    fsamp = nfft * rows
    stride = cfg.get("frame_stride", 1)
    n_frames = cap.n_samples // fsamp
    sig, noise = [], []
    for fi in range(0, n_frames, stride):
        s = fi * fsamp
        mask, boxes = rf.build_frame_mask(cap, s, nfft, rows)
        n_px = int(mask.sum())
        rec = {"frame_index": fi, "abs_start": s, "n_signal_px": n_px, "boxes": boxes}
        if n_px >= cfg["min_signal_pixels"]:
            rec["is_signal"] = True
            sig.append(rec)
        else:
            rec["is_signal"] = False
            noise.append(rec)
    # balance noise relative to signal, then cap total, preserving time order
    n_noise_keep = int(round(len(sig) * cfg["noise_to_signal_ratio"]))
    if noise and n_noise_keep < len(noise):
        keep_idx = np.sort(rng.choice(len(noise), size=n_noise_keep, replace=False))
        noise = [noise[i] for i in keep_idx]
    chosen = sorted(sig + noise, key=lambda r: r["frame_index"])
    cap_cap = cfg["max_frames_per_capture"]
    if len(chosen) > cap_cap:
        # uniform temporal subsample to the cap
        idx = np.linspace(0, len(chosen) - 1, cap_cap).round().astype(int)
        chosen = [chosen[i] for i in sorted(set(idx.tolist()))]
    return chosen, n_frames


def assign_splits(chosen, split_cfg):
    """Temporal split with guard gaps; every capture contributes to all 3 splits."""
    n = len(chosen)
    guard = split_cfg["guard_frames"]
    n_train = int(n * split_cfg["train"])
    n_val = int(n * split_cfg["val"])
    for i, rec in enumerate(chosen):
        if i < n_train - guard:
            rec["split"] = "train"
        elif i < n_train:
            rec["split"] = "drop"
        elif i < n_train + n_val - guard:
            rec["split"] = "val"
        elif i < n_train + n_val:
            rec["split"] = "drop"
        else:
            rec["split"] = "test"
    return [r for r in chosen if r["split"] != "drop"]


# --------------------------------------------------------------------------- #
# Pass B: materialize
# --------------------------------------------------------------------------- #
def materialize(caps, plan_by_cap, cfg, vmin, vmax, out_dir, device):
    nfft, rows = cfg["nfft"], cfg["frame_rows"]
    fsamp = nfft * rows
    # count per split
    counts = {"train": 0, "val": 0, "test": 0}
    for recs in plan_by_cap.values():
        for r in recs:
            counts[r["split"]] += 1
    log(f"materializing splits: {counts}")

    out_dir.mkdir(parents=True, exist_ok=True)
    mm_frames, mm_masks, cursors = {}, {}, {}
    for sp, n in counts.items():
        if n == 0:
            continue
        mm_frames[sp] = np.lib.format.open_memmap(
            out_dir / f"frames_{sp}.npy", mode="w+", dtype=np.uint8, shape=(n, rows, nfft))
        mm_masks[sp] = np.lib.format.open_memmap(
            out_dir / f"masks_{sp}.npy", mode="w+", dtype=np.uint8, shape=(n, rows, nfft))
        cursors[sp] = 0

    frame_rows_csv, region_rows_csv = [], []
    batch_size = 16
    t0 = time.time()
    for cap in caps:
        recs = plan_by_cap[cap.stem]
        if not recs:
            continue
        mm = cap.memmap()
        for bstart in range(0, len(recs), batch_size):
            batch = recs[bstart:bstart + batch_size]
            iq = np.stack([np.asarray(mm[r["abs_start"]:r["abs_start"] + fsamp],
                                      dtype=np.complex64) for r in batch])
            db = rf.frames_to_db(torch.from_numpy(iq).to(device), nfft, rows).cpu().numpy()
            imgs = rf.db_to_uint8(db, vmin, vmax)
            for k, r in enumerate(batch):
                sp = r["split"]
                pos = cursors[sp]
                cursors[sp] += 1
                mm_frames[sp][pos] = imgs[k]
                mask, boxes = rf.build_frame_mask(cap, r["abs_start"], nfft, rows)
                mm_masks[sp][pos] = mask
                frame_id = f"{cap.stem}#{r['frame_index']}"
                frame_rows_csv.append({
                    "frame_id": frame_id, "stem": cap.stem, "split": sp,
                    "attenuation_db": cap.attenuation_db, "frame_index": r["frame_index"],
                    "abs_start": r["abs_start"], "is_signal": int(r["is_signal"]),
                    "n_signal_px": r["n_signal_px"], "mem_pos": pos,
                })
                for b in boxes:
                    region_rows_csv.append({
                        "frame_id": frame_id, "split": sp, "stem": cap.stem,
                        "attenuation_db": cap.attenuation_db, "ann_idx": b.ann_idx,
                        "label": b.label, "kind": b.kind,
                        "bandwidth_hz": b.bandwidth_hz, "length_samples": b.length_samples,
                        "time_group": b.time_group,
                        "row0": b.row0, "row1": b.row1, "col0": b.col0, "col1": b.col1,
                    })
        log(f"  {cap.stem}: wrote {len(recs)} frames ({time.time()-t0:.1f}s elapsed)")
    for sp in mm_frames:
        mm_frames[sp].flush(); mm_masks[sp].flush()

    _write_csv(out_dir / "frames.csv", frame_rows_csv)
    _write_csv(out_dir / "regions.csv", region_rows_csv)
    return counts, len(region_rows_csv)


def _write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit-captures", default=None,
                    help="comma-separated stems for a subset build (smoke test)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="override max_frames_per_capture")
    ap.add_argument("--out-suffix", default="", help="suffix appended to out_dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.max_frames is not None:
        cfg["max_frames_per_capture"] = args.max_frames
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(cfg["seed"])
    limit = args.limit_captures.split(",") if args.limit_captures else None

    caps = discover_captures(Path(cfg["captures_dir"]), limit)
    log(f"captures: {[c.stem for c in caps]}")

    out_dir = Path(cfg["out_dir"] + args.out_suffix)

    # calibration
    vmin, vmax = cfg.get("db_vmin"), cfg.get("db_vmax")
    if vmin is None or vmax is None:
        vmin, vmax = calibrate_db_range(caps, cfg["nfft"], cfg["frame_rows"],
                                        cfg["calib_frames_per_file"], cfg["seed"], device)

    # pass A
    plan_by_cap = {}
    for cap in caps:
        chosen, n_total = plan_frames(cap, cfg, rng)
        chosen = assign_splits(chosen, cfg["split"])
        plan_by_cap[cap.stem] = chosen
        n_sig = sum(r["is_signal"] for r in chosen)
        log(f"plan {cap.stem} ({cap.attenuation_db}dB): {len(chosen)}/{n_total} frames "
            f"kept ({n_sig} signal, {len(chosen)-n_sig} noise)")

    # pass B
    counts, n_regions = materialize(caps, plan_by_cap, cfg, vmin, vmax, out_dir, device)

    meta = {
        "nfft": cfg["nfft"], "frame_rows": cfg["frame_rows"],
        "db_vmin": vmin, "db_vmax": vmax, "counts": counts, "n_regions": n_regions,
        "low_atten_max_db": cfg["low_atten_max_db"],
        "captures": [c.stem for c in caps],
        "attenuations": {c.stem: c.attenuation_db for c in caps},
        "sample_rate": caps[0].sample_rate if caps else None,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE -> {out_dir}  counts={counts} regions={n_regions}")


if __name__ == "__main__":
    main()

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
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import rfdata as rf
import rate_augment as rate_aug

FRAME_H, FRAME_W = None, None  # set from config at runtime (frame_rows, nfft)


def log(msg: str):
    print(f"[build] {msg}", flush=True)


try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


def pbar(total, desc):
    """tqdm progress bar with ETA; falls back to periodic prints if tqdm isn't installed."""
    if _tqdm is not None:
        return _tqdm(total=total, desc=desc, unit="frame", smoothing=0.05, dynamic_ncols=True)

    class _Fallback:
        def __init__(s):
            s.n = 0; s.t0 = time.time()
        def update(s, k=1):
            s.n += k
            if s.n % 500 == 0 or s.n == total:
                el = time.time() - s.t0
                rate = s.n / el if el > 0 else 0
                eta = (total - s.n) / rate if rate > 0 else 0
                log(f"{desc}: {s.n}/{total} ({100*s.n/max(total,1):.1f}%) {rate:.0f}/s ETA {eta/60:.1f} min")
        def close(s):
            pass
    return _Fallback()


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
def calibrate_db_range(caps, nfft, rows, per_file, seed, device, window="hann"):
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
        db = rf.frames_to_db(iq, nfft, rows, window=window).cpu().numpy().ravel()
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
    frame_iter = range(0, n_frames, stride)
    if _tqdm is not None:
        frame_iter = _tqdm(frame_iter, desc=f"plan {cap.stem}", unit="fr", leave=False, dynamic_ncols=True)
    for fi in frame_iter:
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
            db = rf.frames_to_db(torch.from_numpy(iq).to(device), nfft, rows,
                                 window=cfg.get("fft_window", "hann")).cpu().numpy()
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
# Pass B (domain-randomized): per-rate capture-chain emulation -> float16 dB stacks
# --------------------------------------------------------------------------- #
def load_envelopes(sweep_stats_dir):
    """Load per-rate envelope templates {rate_hz: [nfft] zero-mean dB} from sweep_stats envelopes.npz."""
    if not sweep_stats_dir:
        return {}
    p = Path(sweep_stats_dir) / "envelopes.npz"
    if not p.exists():
        log(f"WARN: no envelopes at {p} -> emulation runs without envelope reshaping.")
        return {}
    z = np.load(p)
    env = {}
    for k in z.files:
        if k.startswith("rate_"):
            env[float(k[len("rate_"):])] = z[k].astype(np.float32)
    log(f"loaded {len(env)} envelope templates from {p}")
    return env


def load_backgrounds(sweep_stats_dir):
    """{rate_hz: [abs iq paths]} from sweep_stats backgrounds.json (antenna IQ, for the upsample+paste
    path at rates > the source rate). Empty if not present."""
    if not sweep_stats_dir:
        return {}
    p = Path(sweep_stats_dir) / "backgrounds.json"
    if not p.exists():
        return {}
    by_rate = json.loads(p.read_text()).get("by_rate", {})
    bank = {}
    for rk, items in by_rate.items():
        paths = [it["iq_file"] for it in items if Path(it["iq_file"]).exists()]
        if paths:
            bank[float(rk)] = paths
    return bank


def materialize_dr(caps, plan_by_cap, cfg, out_dir, device, envelopes):
    """Domain-randomized build: emulate each planned source frame at every rate in `dr_rates_hz`
    (<= source rate) and `dr_centers_per_frame` random center offsets, storing float16 dB frames
    (pre-clip) so dataset.py can do dB-domain level/envelope/paste augmentation then clip. Rates above
    the source rate are skipped here (handled by the upsample+paste path once the wideband sweep exists)."""
    nfft, rows = cfg["nfft"], cfg["frame_rows"]
    fsamp = nfft * rows
    dr_rates = [float(r) for r in cfg["dr_rates_hz"]]
    fft_window = cfg.get("fft_window", "hann")
    n_centers = int(cfg.get("dr_centers_per_frame", 2))
    max_frac = float(cfg.get("dr_center_max_frac", 0.4))  # keep sub-band within +-max_frac of source band
    rng = np.random.default_rng(cfg["seed"] + 1)
    src_rate = caps[0].sample_rate if caps else 245.76e6

    usable = [r for r in dr_rates if r <= src_rate + 1]                 # decimation-emulation path
    paste_rates = [r for r in dr_rates if r > src_rate + 1]            # upsample+paste path (needs bg)
    bank = load_backgrounds(cfg.get("sweep_stats_dir"))
    paste_ok = bool(bank) and bool(paste_rates)
    if paste_rates and not paste_ok:
        log(f"NOTE: rates > source {src_rate/1e6:.2f} MS/s ({[r/1e6 for r in paste_rates]}) need the "
            f"upsample+paste path with sweep backgrounds; none found -> skipped. Run the wideband-image "
            f"sweep + sweep_stats first, then rebuild.")

    # Count exactly (skip-aware): a (frame,rate) combo needs fsamp*D source samples; lower rates add
    # n_centers, the source rate adds 1; paste rates add n_centers if a background exists. Mirrors the
    # write loop so the memmap is sized precisely.
    def combos_for(cap, rec):
        c = 0
        for rate in usable:
            D = max(1, int(round(src_rate / rate)))
            if rec["abs_start"] + fsamp * D <= cap.n_samples:
                c += 1 if abs(rate - src_rate) <= 1 else n_centers
        if paste_ok:
            for rate in paste_rates:
                U = max(1, int(round(rate / src_rate)))
                if rec["abs_start"] + fsamp // U <= cap.n_samples:
                    c += n_centers
        return c
    counts = {"train": 0, "val": 0, "test": 0}
    for cap in caps:
        for rec in plan_by_cap.get(cap.stem, []):
            counts[rec["split"]] += combos_for(cap, rec)
    log(f"DR materialize: {len(usable)} rates x ~{n_centers} centers; splits {counts}")

    # ---- disk estimate + guard (float16 frame + uint8 mask per frame) --------------------------
    total_frames = sum(counts.values())
    bytes_per = rows * nfft * (2 + 1)                      # float16 frames + uint8 masks
    est = total_frames * bytes_per
    out_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out_dir).free
    min_free_gb = float(cfg.get("min_free_gb", 10.0))
    log(f"disk estimate: {total_frames} frames x {bytes_per/1e3:.0f} KB = {est/1e9:.2f} GB "
        f"(frames {total_frames*rows*nfft*2/1e9:.2f} GB + masks {total_frames*rows*nfft/1e9:.2f} GB); "
        f"free on '{out_dir}': {free/1e9:.2f} GB")
    if est > free - min_free_gb * 1e9:
        raise RuntimeError(
            f"ABORT: estimated {est/1e9:.2f} GB would leave < {min_free_gb} GB free on {out_dir}. "
            f"Lower max_frames_per_capture / dr_centers_per_frame / #dr_rates_hz, or free disk / set a "
            f"different out_dir. (min_free_gb overrides the {min_free_gb} GB margin.)")

    mm_frames, mm_masks, cursors = {}, {}, {}
    for sp, n in counts.items():
        if n == 0:
            continue
        mm_frames[sp] = np.lib.format.open_memmap(
            out_dir / f"frames_{sp}.npy", mode="w+", dtype=np.float16, shape=(n, rows, nfft))
        mm_masks[sp] = np.lib.format.open_memmap(
            out_dir / f"masks_{sp}.npy", mode="w+", dtype=np.uint8, shape=(n, rows, nfft))
        cursors[sp] = 0

    frame_rows_csv = []
    t0 = time.time()
    bar = pbar(sum(counts.values()), "DR emulate")
    for cap in caps:
        recs = plan_by_cap[cap.stem]
        if not recs:
            continue
        mm = cap.memmap()
        for r in recs:
            sp = r["split"]
            for rate in usable:
                D = max(1, int(round(src_rate / rate)))
                need = fsamp * D
                if r["abs_start"] + need > cap.n_samples:
                    continue  # not enough source samples for this decimation
                chunk = np.array(mm[r["abs_start"]:r["abs_start"] + need], dtype=np.complex64)  # writable copy
                # rate==src -> single centered frame; lower rates -> n_centers random offsets.
                if abs(rate - src_rate) <= 1:
                    fcs = [0.0]
                else:
                    lim = max_frac * src_rate - rate / 2.0     # keep [f_c-R/2, f_c+R/2] within +-max_frac*src
                    lim = max(0.0, lim)
                    fcs = (rng.uniform(-lim, lim, size=n_centers) if lim > 0 else np.zeros(n_centers))
                for f_c in fcs:
                    db, mask, boxes = rate_aug.emulate_frame(
                        chunk, r["abs_start"], cap.annotations, src_rate, rate, float(f_c),
                        nfft, rows, envelopes=envelopes, device=device, window=fft_window)
                    pos = cursors[sp]; cursors[sp] += 1
                    mm_frames[sp][pos] = db.astype(np.float16)
                    mm_masks[sp][pos] = mask
                    frame_rows_csv.append({
                        "frame_id": f"{cap.stem}#{r['frame_index']}@{rate/1e6:.3f}MHz@fc{f_c/1e6:.2f}",
                        "stem": cap.stem, "split": sp, "attenuation_db": cap.attenuation_db,
                        "frame_index": r["frame_index"], "abs_start": r["abs_start"],
                        "emulated_rate_hz": rate, "center_offset_hz": float(f_c),
                        "n_signal_px": int(mask.sum()), "is_signal": int(mask.sum() > 0), "mem_pos": pos})
                    bar.update(1)
            # upsample+paste path for rates wider than the source (real bg supplies the wide noise/env).
            if paste_ok:
                for rate in paste_rates:
                    U = max(1, int(round(rate / src_rate)))
                    src_need = fsamp // U
                    if r["abs_start"] + src_need > cap.n_samples:
                        continue
                    sig_chunk = np.array(mm[r["abs_start"]:r["abs_start"] + src_need], dtype=np.complex64)
                    nearest = min(bank, key=lambda k: abs(k - rate))
                    for _ in range(n_centers):
                        bg_iq = np.load(bank[nearest][int(rng.integers(len(bank[nearest])))])
                        db, mask, boxes = rate_aug.emulate_frame_upsample_paste(
                            sig_chunk, r["abs_start"], cap.annotations, src_rate, rate, bg_iq,
                            nfft, rows, sig_gain=1.0, envelopes=envelopes, device=device, window=fft_window)
                        pos = cursors[sp]; cursors[sp] += 1
                        mm_frames[sp][pos] = db.astype(np.float16)
                        mm_masks[sp][pos] = mask
                        frame_rows_csv.append({
                            "frame_id": f"{cap.stem}#{r['frame_index']}@{rate/1e6:.3f}MHz@paste",
                            "stem": cap.stem, "split": sp, "attenuation_db": cap.attenuation_db,
                            "frame_index": r["frame_index"], "abs_start": r["abs_start"],
                            "emulated_rate_hz": rate, "center_offset_hz": 0.0,
                            "n_signal_px": int(mask.sum()), "is_signal": int(mask.sum() > 0), "mem_pos": pos})
                        bar.update(1)
        log(f"  {cap.stem}: emulated {len([f for f in frame_rows_csv if f['stem']==cap.stem])} frames "
            f"({time.time()-t0:.1f}s)")
    bar.close()
    for sp in list(mm_frames):
        mm_frames[sp].flush(); mm_masks[sp].flush()
    _write_csv(out_dir / "frames.csv", frame_rows_csv)
    _write_csv(out_dir / "regions.csv", [])
    # record true written counts
    counts = {sp: cursors.get(sp, 0) for sp in counts}
    return counts


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
                                        cfg["calib_frames_per_file"], cfg["seed"], device,
                                        window=cfg.get("fft_window", "hann"))

    # pass A
    plan_by_cap = {}
    for cap in caps:
        chosen, n_total = plan_frames(cap, cfg, rng)
        chosen = assign_splits(chosen, cfg["split"])
        plan_by_cap[cap.stem] = chosen
        n_sig = sum(r["is_signal"] for r in chosen)
        log(f"plan {cap.stem} ({cap.attenuation_db}dB): {len(chosen)}/{n_total} frames "
            f"kept ({n_sig} signal, {len(chosen)-n_sig} noise)")

    # pass B -- domain-randomized (multi-rate float-dB emulation) or the original single-grid build.
    domain_randomize = bool(cfg.get("domain_randomize", False))
    if domain_randomize:
        envelopes = load_envelopes(cfg.get("sweep_stats_dir"))
        counts = materialize_dr(caps, plan_by_cap, cfg, out_dir, device, envelopes)
        n_regions = 0
    else:
        counts, n_regions = materialize(caps, plan_by_cap, cfg, vmin, vmax, out_dir, device)

    meta = {
        "nfft": cfg["nfft"], "frame_rows": cfg["frame_rows"],
        "db_vmin": vmin, "db_vmax": vmax, "counts": counts, "n_regions": n_regions,
        "low_atten_max_db": cfg["low_atten_max_db"],
        "captures": [c.stem for c in caps],
        "attenuations": {c.stem: c.attenuation_db for c in caps},
        "sample_rate": caps[0].sample_rate if caps else None,
        # DR frames are float16 dB (PRE-clip); the plain build stores uint8 [0,1] (POST-clip).
        "domain_randomize": domain_randomize,
        "frame_storage": "float16_db" if domain_randomize else "uint8_norm",
        "fft_window": cfg.get("fft_window", "hann"),   # must match the deployed detector's FFT window
        "dr_rates_hz": [float(r) for r in cfg.get("dr_rates_hz", [])] if domain_randomize else None,
        "sweep_stats_dir": cfg.get("sweep_stats_dir") if domain_randomize else None,
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"DONE -> {out_dir}  counts={counts} regions={n_regions}")


if __name__ == "__main__":
    main()

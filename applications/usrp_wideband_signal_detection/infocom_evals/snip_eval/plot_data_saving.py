#!/usr/bin/env python3
"""Data-saving evaluation from masks (no snipper/GPU/container needed for the NUMBERS).

Per (detector, capture, frame): cluster the mask into boxes (signal_snipper's min_box_pixels/merge_gap
rule) and compute the frequency-mode stored bytes with the snipper's exact formula
(keep_bw=(bw+margin)*(1+oversample%); decim=floor(fs/keep_bw); n_out=ceil(n_in/decim); bytes=n_out*8).
Aggregate to bytes/hr vs SNR per detector, vs the naive save-all baseline, plus signal retention.
These are byte-exact to what the C++ snipper's SigMF output sums to (validate once against a real run).

Usage: python3 plot_data_saving.py --batch-root /tmp/ds_batch --captures-dir /home/bqn82/captures
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snip_annotations import boxes_from_mask, load_mask

BYTES, SEC_HR = 8, 3600
GAP_R, GAP_C = 2, 8                          # signal_snipper merge gaps, scaled to pooled grid (16/2, 80/10)
OVERSAMPLE, BW_MARGIN, DOWNSAMPLE = 25.0, 0.0, True
# Cluster on a pooled grid for speed (byte formula uses col/row FRACTIONS, so pooling preserves it;
# this is a projection -- the exact bytes come from the real C++ snipper). 512x10240 -> 256x1024.
RPOOL = int(os.environ.get("DS_RPOOL", "2")); CPOOL = int(os.environ.get("DS_CPOOL", "10"))
MIN_BOX = max(1, round(256 / (RPOOL * CPOOL)))   # min_box_pixels scaled to the pooled grid
TS_MIN_PIX = int(os.environ.get("DS_TS_MIN_PIXELS", "256"))  # time-slice: keep a time-row only if it has >= this many on-pixels (full grid)

def pool(m):
    r, c = m.shape; rr, cc = r // RPOOL, c // CPOOL
    return m[:rr*RPOOL, :cc*CPOOL].reshape(rr, RPOOL, cc, CPOOL).any(axis=(1, 3))

def frame_bytes(mask, fsc, fs):
    rows, cols = mask.shape; total = 0
    for r0, r1, c0, c1 in boxes_from_mask(mask, MIN_BOX, GAP_R, GAP_C):
        n_in = max(0, min(int(math.ceil(((r1+1)/rows)*fsc)), fsc) - int(math.floor((r0/rows)*fsc)))
        bw = ((c1+1-c0)/cols)*fs
        keep = (max(bw, 1.0)+BW_MARGIN)*(1+OVERSAMPLE/100)
        decim = max(1, int(math.floor(fs/keep))) if DOWNSAMPLE else 1
        total += (int(math.ceil(n_in/decim)) if n_in > 0 else 0)*BYTES
    # cap at the frame's raw IQ size: snipping can never sensibly store MORE than saving the
    # whole frame (broad detectors tile the band and the per-region oversample would otherwise
    # sum past naive). So worst case = keep the raw frame -> reduction >= 1x.
    return min(total, fsc * BYTES)

def atten(stem):
    m = re.search(r"dB_(\d+)", stem); return int(m.group(1)) if m else None

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-root", required=True, type=Path)
    ap.add_argument("--captures-dir", type=Path, default=Path("/home/bqn82/captures"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--nframes", type=int, default=int(os.environ.get("DS_NFRAMES", "0")))
    a = ap.parse_args()
    out = a.out_dir or (a.batch_root / "data_saving_figs"); out.mkdir(parents=True, exist_ok=True)
    RAW = out / "_raw_rows.csv"
    FIELDS = ["detector", "stem", "attenuation_db", "stored_TB_hr", "naive_TB_hr", "reduction_x",
              "timeslice_frac", "timeslice_TB_hr", "timeslice_reduction_x", "retention"]

    def process_run(det, run):
        stem = run.name; a_db = atten(stem)
        man = {int(r["frame_number"]): r for r in csv.DictReader(open(run/"frame_manifest.csv"))}
        fs = float(json.load(open(a.captures_dir/f"{stem}.sigmf-meta")).get("global", {}).get("core:sample_rate"))
        mfs = sorted((run/"mask_arrays").glob("mask_ch0_f*.*"))
        if a.nframes: mfs = mfs[:a.nframes]
        tot, cap_sec, ret, ts_rows, tot_rows = 0, 0.0, [], 0, 0
        for mf in mfs:
            num = int(mf.name.split("_f")[1].split("_")[0]); r = man.get(num)
            if r is None: continue
            fsc = int(float(r["complex_samples_read"])); mfull = load_mask(mf)
            kept = (mfull.sum(axis=1) >= TS_MIN_PIX)               # time-slice keep-rule (full grid)
            ts_rows += int(kept.sum()); tot_rows += mfull.shape[0]
            tot += frame_bytes(pool(mfull), fsc, fs); cap_sec += fsc/fs
            gt = list((run/"gt_masks").glob(f"*_f{num}_*.*"))
            if gt:
                gt_rows = load_mask(gt[0]).any(axis=1)             # GT signal-rows (full grid)
                if gt_rows.any():
                    ret.append(float((gt_rows & kept).sum()/gt_rows.sum()))   # retention uses the SAME >=TS_MIN_PIX keep-rule as time-slice
        if cap_sec <= 0:
            return None
        stored = tot/cap_sec*SEC_HR; naive = fs*BYTES*SEC_HR
        tsf = (ts_rows/tot_rows) if tot_rows else 0.0
        return dict(detector=det, stem=stem, attenuation_db=a_db, stored_TB_hr=stored/1e12,
                    naive_TB_hr=naive/1e12, reduction_x=(naive/stored if stored > 0 else np.inf),
                    timeslice_frac=tsf, timeslice_TB_hr=naive*tsf/1e12,
                    timeslice_reduction_x=(1.0/tsf if tsf > 0 else np.inf),
                    retention=float(np.mean(ret)) if ret else np.nan)

    # resumable: per-(detector,stem) rows appended incrementally; skip already-done so an overnight
    # run that gets killed just resumes where it left off, and picks up new attenuations (dB_65/70).
    done = set()
    if RAW.exists():
        done = {(r["detector"], r["stem"]) for r in csv.DictReader(open(RAW))}
    expected = []
    for det_dir in sorted(pp for pp in a.batch_root.iterdir() if pp.is_dir()):
        for run in sorted(pp for pp in det_dir.iterdir() if pp.is_dir()):
            if (atten(run.name) is not None and (run/"frame_manifest.csv").exists()
                    and (a.captures_dir/f"{run.name}.sigmf-meta").exists() and (run/"mask_arrays").exists()):
                expected.append((det_dir.name, run))
    todo = [(d, r) for d, r in expected if (d, r.name) not in done]
    print(f"resume: {len(done)} done, {len(todo)} to do, {len(expected)} expected")
    raw_f = open(RAW, "a", newline=""); w = csv.DictWriter(raw_f, fieldnames=FIELDS)
    if raw_f.tell() == 0:
        w.writeheader()
    for det, run in todo:
        row = process_run(det, run)
        if row:
            w.writerow(row); raw_f.flush()
        print(f"  done {det}/{run.name}", flush=True)
    raw_f.close()

    raw_rows = list(csv.DictReader(open(RAW)))
    if len({(r["detector"], r["stem"]) for r in raw_rows}) < len(expected):
        print(f"PARTIAL: {len({(r['detector'],r['stem']) for r in raw_rows})}/{len(expected)} runs done -- re-run to resume")
        return 0
    df = pd.DataFrame(raw_rows)
    for c in FIELDS:
        if c not in ("detector", "stem"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.groupby(["detector", "attenuation_db"], as_index=False).mean(numeric_only=True)
    df.to_csv(out/"data_saving_table.csv", index=False)
    print(df.to_string(index=False))
    if df.empty: print("no data"); return 0
    naive = df.naive_TB_hr.iloc[0]
    # Fig 1: stored bytes/hr vs SNR per detector (log), naive baseline
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(naive, color="k", lw=2, label=f"naive save-all ({naive:.2f} TB/hr)")
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det].sort_values("attenuation_db")
        ax.plot(d.attenuation_db, d.stored_TB_hr, "-o", label=f"{det} (freq)", ms=4)
        ax.plot(d.attenuation_db, d.timeslice_TB_hr, "--", alpha=.5)   # dashed = time-slice strategy
    ax.set_yscale("log"); ax.set_xlabel("attenuation (dB)"); ax.set_ylabel("stored TB/hr (freq-mode snip, log)")
    ax.set_title(f"Data stored/hr vs SNR — solid=freq-mode snip, dashed=time-slice (>= {TS_MIN_PIX}px/row)")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(out/"stored_vs_snr.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    # Fig 2: reduction vs retention
    fig, ax = plt.subplots(figsize=(9, 6))
    for det in sorted(df.detector.unique()):
        d = df[df.detector == det].sort_values("attenuation_db")
        ax.plot(d.reduction_x, 100*d.retention, "-o", label=det, ms=4)
    ax.set_xlabel("data-reduction factor (x vs save-all)"); ax.set_ylabel("signal-time retention (%)")
    ax.set_title("Reduction vs retention (each path = one detector across SNR)")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out/"reduction_vs_retention.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"\nwrote {out}/data_saving_table.csv + stored_vs_snr.png + reduction_vs_retention.png")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

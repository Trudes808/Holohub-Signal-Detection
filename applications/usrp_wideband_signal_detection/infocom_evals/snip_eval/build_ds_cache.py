#!/usr/bin/env python3
"""Precompute the data-reduction table ONCE and cache it persistently (in-repo, not /tmp) so the
notebook loads instantly and any figure/table can be restyled without the ~30-min mask crunch.

Per (detector, attenuation) for the 8 detectors + a `ground_truth` pseudo-detector:
  timeslice_frac, timeslice_frac_raw, tf_coverage, retention   (denoised; N_FRAMES sample)
  resample_meas_TB_hr                                           (snip/resample+filter footprint)

Detector snip footprint is reused from data_saving_metrics.csv (all-frames, already computed);
ground_truth's timeslice + snip are computed here (GT masks are clean/fast). Writes ds_cache.csv.

Env: DS_SWEEP, DS_CAPTURES_DIR, DS_METRICS, DS_NFRAMES(=120), DS_MIN_BOX_PIXELS(=256).
"""
import os, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage

HSD = Path.home() / "Holohub-Signal-Detection"
SNIP = HSD / "applications/usrp_wideband_signal_detection/infocom_evals/snip_eval"
sys.path.insert(0, str(SNIP))
from snip_annotations import load_mask, boxes_from_mask  # exact snipper clustering rule

SWEEP    = Path(os.environ.get("DS_SWEEP", str(HSD / "notebooks/yolo_evals/sweeps/sweep_all")))
CAPTURES = Path(os.environ.get("DS_CAPTURES_DIR", "/home/bqn82/captures"))
METRICS  = Path(os.environ.get("DS_METRICS", str(SNIP / "detected/data_saving_metrics.csv")))
N_FRAMES = int(os.environ.get("DS_NFRAMES", "120"))
MIN_BOX  = int(os.environ.get("DS_MIN_BOX_PIXELS", "256"))
RATE, BPS, SEC_HR = 245.76e6, 8, 3600.0
DETS = ["coherent_power", "cuda_dino", "finetuned_dino", "finetuned_dino_m2",
        "yolo26s", "yolo26m", "3dB_power", "blob_detection"]
GT_REF = "cuda_dino"                       # GT masks are detector-independent; read them from here
OUT = HSD / "applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/ds_cache.csv"


def denoise(m):
    if MIN_BOX <= 1 or not m.any():
        return m
    lab, nl = ndimage.label(m)
    if nl == 0:
        return np.zeros_like(m)
    sizes = np.bincount(lab.ravel()); keep = sizes >= MIN_BOX; keep[0] = False
    return keep[lab].astype(m.dtype)


def timeslice_frac(mask, block_rows=1):
    r = mask.shape[0]; nb = r // block_rows
    return float(mask[:nb * block_rows].reshape(nb, block_rows, -1).any(axis=(1, 2)).mean())


def atten(s):
    m = re.search(r"dB_(\d+)", s); return int(m.group(1)) if m else None


def cap_sec(stem):
    f = CAPTURES / f"{stem}.sigmf-data"
    return f.stat().st_size / (BPS * RATE) if f.exists() else float("nan")


def det_mask_files(sd):
    return (sorted(sd.glob("mask_arrays/mask_ch0_f*.packed.npz"))
            or sorted(sd.glob("mask_arrays/mask_ch0_f*.npy")))


def gt_mask_files(sd):
    return (sorted(sd.glob("gt_masks/*_f*_*.packed.npz"))
            or sorted(sd.glob("gt_masks/*_f*_*.npy")))


def gt_file_for_frame(sd, n):
    g = (list(sd.glob(f"gt_masks/*_f{n}_*.packed.npz"))
         or list(sd.glob(f"gt_masks/*_f{n}_*.npy")))
    return g[0] if g else None


def gt_snip_TB_hr(sd, stem):
    """GT resample+filter footprint (bytes/hr), per-box sum == snipper, over ALL frames."""
    import csv as _csv
    man_path = sd / "frame_manifest.csv"
    if not man_path.exists():
        return float("nan")
    man = {int(r["frame_number"]): r for r in _csv.DictReader(open(man_path))}
    retained = 0.0
    for mf in gt_mask_files(sd):
        n = int(re.search(r"_f(\d+)_", mf.name).group(1))
        row = man.get(n)
        if row is None:
            continue
        m = load_mask(mf).astype(bool); rows_, cols_ = m.shape
        fsc = int(float(row["complex_samples_read"]))
        for r0, r1, c0, c1 in boxes_from_mask(m, MIN_BOX, 16, 80):
            samp = max(1, int(np.ceil(((r1 + 1) / rows_) * fsc)) - int(np.floor((r0 / rows_) * fsc)))
            retained += BPS * ((c1 + 1 - c0) / cols_) * samp
    sec = cap_sec(stem)
    return retained / sec * SEC_HR / 1e12 if sec > 0 else float("nan")


def frac_metrics(mfiles, sd, is_gt):
    """timeslice/tf_coverage/retention over first N_FRAMES frames."""
    ts, tsr, cov, ret = [], [], [], []
    for mf in mfiles[:N_FRAMES]:
        n = re.search(r"_f(\d+)_", mf.name).group(1)
        m = load_mask(mf).astype(bool); md = denoise(m)
        ts.append(timeslice_frac(md)); tsr.append(timeslice_frac(m)); cov.append(float((md != 0).mean()))
        if is_gt:
            ret.append(1.0)
        else:
            gf = gt_file_for_frame(sd, n)
            if gf is not None:
                g = load_mask(gf).astype(bool); gr = g.any(axis=1)
                if gr.any():
                    ret.append(float((gr & md.any(axis=1)).sum()) / float(gr.sum()))
    return (float(np.mean(ts)), float(np.mean(tsr)), float(np.mean(cov)),
            float(np.mean(ret)) if ret else float("nan"))


def load_det_snip():
    """detector -> {attenuation: resample_meas_TB_hr} from the metrics CSV (all-frames)."""
    if not METRICS.exists():
        print(f"  (no metrics CSV at {METRICS}; detector snip left NaN)")
        return {}
    mdf = pd.read_csv(METRICS)
    out = {}
    for (d, a), g in mdf.groupby(["detector", "attenuation_db"]):
        out[(d, int(a))] = float(g.retained_TB_per_hour.mean())
    return out


def main():
    det_snip = load_det_snip()
    rows = []
    for det in DETS + ["ground_truth"]:
        src = GT_REF if det == "ground_truth" else det
        base = SWEEP / src
        if not base.exists():
            print(f"SKIP {det}: {base} missing"); continue
        for sd in sorted(base.glob("*/")):
            a = atten(sd.name)
            if a is None:
                continue
            is_gt = det == "ground_truth"
            mfiles = gt_mask_files(sd) if is_gt else det_mask_files(sd)
            if not mfiles:
                continue
            tsf, tsr, cov, ret = frac_metrics(mfiles, sd, is_gt)
            snip = (gt_snip_TB_hr(sd, sd.name) if is_gt else det_snip.get((det, a), float("nan")))
            rows.append(dict(detector=det, file_stem=sd.name, attenuation_db=a,
                             timeslice_frac=tsf, timeslice_frac_raw=tsr, tf_coverage=cov,
                             retention=ret, resample_meas_TB_hr=snip))
            print(f"  [{det}/{sd.name}] ts={tsf:.3f} cov={cov:.3f} ret={ret:.3f} snip={snip:.3f}", flush=True)
    df = pd.DataFrame(rows)
    # average 30 & 30_v2 into a single attenuation=30 row per detector
    agg = (df.groupby(["detector", "attenuation_db"], as_index=False)
             .agg({"timeslice_frac": "mean", "timeslice_frac_raw": "mean", "tf_coverage": "mean",
                   "retention": "mean", "resample_meas_TB_hr": "mean"}))
    agg.to_csv(OUT, index=False)
    print(f"\nwrote {len(agg)} (detector, attenuation) rows -> {OUT}")
    print(f"detectors: {sorted(agg.detector.unique())}")


if __name__ == "__main__":
    main()

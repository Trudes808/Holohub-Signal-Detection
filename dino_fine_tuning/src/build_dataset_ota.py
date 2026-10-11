"""Build a DINO-FT domain-adaptation dataset from REAL X410 OTA captures.

Same tile format as build_dataset_rt.py (frames_{split}.npy [N,256,1024] uint8, masks_{split}.npy, frames.csv,
dataset_meta.json) so train.py/RFSegDataset consume it unchanged. The ONLY differences vs the synthetic
builder are the input source and the label source:

  * INPUT  : real OTA IQ (ci16_le SigMF) -> int16/32767 -> the SAME deployment front-end (frontend.py),
             so the model trains on exactly what it sees live.
  * LABELS : antenna-ON captures  -> the COHERENT detector's mask (dumped by run_cuda_dino_offline_file.py
             --detector coherent_power), resized 20480->1024 (block-max) and despeckled (min connected-
             component size) -> a tight, high-precision teacher label.
             antenna-OFF captures -> all-zero masks (guaranteed negatives: "real receiver noise = no signal").

Capture-level train/val split (no frame leakage). Labels come from a labels-dir laid out as
<labels_dir>/<capture_stem>/mask_arrays/mask_ch0_f<frame>_512x20480.npy (produced by label_ota.sh).
"""
from __future__ import annotations
import argparse, csv, glob, json, os, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

import frontend as fe

try:
    from scipy.ndimage import label as _cc_label
except Exception:
    _cc_label = None

FFT = 20480
ROWS = 512
TILE = 256
FSAMP = ROWS * FFT           # 10,485,760 complex samples per frame
FS = 491_520_000.0


def load_iq_ci16(path: str) -> np.ndarray:
    """SigMF ci16_le -> complex64 scaled to [-1,1] (matches the deployed operator's int16/0x7FFF)."""
    raw = np.fromfile(path, dtype="<i2")
    n = (raw.size // 2) * 2
    iq = raw[0:n:2].astype(np.float32) + 1j * raw[1:n:2].astype(np.float32)
    return (iq / 32767.0).astype(np.complex64)


def frames_of(iq: np.ndarray):
    n = iq.size // FSAMP
    for f in range(n):
        yield f, iq[f * FSAMP:(f + 1) * FSAMP]


def resize_mask_freq_blockmax(mask: np.ndarray, nfft: int = 1024) -> np.ndarray:
    """[512,20480] binary -> [512,1024] by OR-reducing each 20-bin block (preserves thin signals)."""
    m = mask.astype(bool)
    rows, wide = m.shape
    assert wide % nfft == 0, f"{wide} not divisible by {nfft}"
    return m.reshape(rows, nfft, wide // nfft).any(axis=2)


def despeckle(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Remove connected components smaller than min_px (the coherent teacher's small-noise blips)."""
    if min_px <= 0 or _cc_label is None:
        return mask
    lab, n = _cc_label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False  # background
    return keep[lab]


def coherent_masks_for(stem: str, labels_dir: str) -> dict:
    d = os.path.join(labels_dir, stem, "mask_arrays")
    out = {}
    for p in glob.glob(os.path.join(d, "mask_ch0_f*.npy")):
        m = re.search(r"_f(\d+)_", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def cap_band(stem: str) -> str:
    return "2400M" if "2400M" in stem else ("1000M" if "1000M" in stem else "other")


def cap_type(stem: str) -> str:
    return "noise" if "noise" in stem else "on"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures-dir", default=os.path.expanduser("~/Documents/captures"))
    ap.add_argument("--labels-dir", required=True, help="coherent-teacher mask dirs (from label_ota.sh)")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "data" / "dataset_ota"))
    ap.add_argument("--glob", default="x410_ota_*.sigmf-data",
                    help="capture glob; only files with _on_ or _noise_ in the name are used")
    ap.add_argument("--min-sig-px", type=int, default=8, help="tile is 'signal' if mask sum >= this")
    ap.add_argument("--min-comp-px", type=int, default=24, help="despeckle: drop mask components < this (at 512x1024)")
    ap.add_argument("--noise-ratio", type=float, default=0.6, help="noise tiles kept <= ratio * signal tiles")
    ap.add_argument("--debug-viz", action="store_true", help="save one input+coherent-mask overlay and exit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = fe.FrontEndCfg()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    caps = sorted(glob.glob(os.path.join(args.captures_dir, args.glob)))
    caps = [c for c in caps if c.endswith(".sigmf-data")
            and ("_on_" in Path(c).name or "_noise_" in Path(c).name)]
    if not caps:
        raise SystemExit(f"no _on_/_noise_ captures match {args.captures_dir}/{args.glob}")
    # capture-level split: first capture of each (band,type) group -> val, rest -> train.
    groups = defaultdict(list)
    for c in caps:
        stem = Path(c).name[:-len(".sigmf-data")]
        groups[(cap_band(stem), cap_type(stem))].append(c)
    val_caps = {sorted(v)[0] for v in groups.values()}
    print(f"{len(caps)} captures; val={sorted(Path(c).name for c in val_caps)}", flush=True)

    if args.debug_viz:
        # overlay the first ON capture's middle frame: front-end input + resized coherent mask (alignment check)
        on = sorted(c for c in caps if cap_type(Path(c).name[:-11]) == "on")[0]
        stem = Path(on).name[:-len(".sigmf-data")]
        iq = load_iq_ci16(on)
        cm = coherent_masks_for(stem, args.labels_dir)
        fidx = (iq.size // FSAMP) // 2
        fr = iq[fidx * FSAMP:(fidx + 1) * FSAMP]
        o = fe.compute_front_end(torch.from_numpy(fr).to(device).unsqueeze(0), FS, cfg, fft_size=FFT)
        img = o.resized[0].clamp(0, 1).cpu().numpy()
        raw = np.load(cm[fidx + 1]); m = despeckle(resize_mask_freq_blockmax(raw), args.min_comp_px)
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(14, 6))
        ax[0].imshow(img, aspect="auto", origin="lower", cmap="viridis"); ax[0].set_title(f"{stem} f{fidx+1}: front-end input")
        ax[1].imshow(img, aspect="auto", origin="lower", cmap="gray")
        ax[1].imshow(np.ma.masked_where(~m, m), aspect="auto", origin="lower", cmap="autumn", alpha=0.6)
        ax[1].set_title(f"coherent teacher mask (despeckled, occ {100*m.mean():.2f}%) over input -- should sit ON signals")
        p = str(out / "ota_label_alignment.png"); plt.tight_layout(); plt.savefig(p, dpi=90)
        print(f"saved {p}"); return

    meta = {"nfft": cfg.nfft, "tile_rows": TILE, "fft_size": FFT, "rows_wide": ROWS,
            "sample_rate": FS, "front_end": "robust", "source": "ota_coherent_teacher",
            "counts": {}, "sig_noise": {}}
    split_data = {"train": ([], []), "val": ([], [])}
    split_csv = {"train": [], "val": []}
    split_cnt = {"train": [0, 0], "val": [0, 0]}  # [n_sig, n_noise]
    rng = np.random.default_rng(7)

    for c in caps:
        stem = Path(c).name[:-len(".sigmf-data")]
        split = "val" if c in val_caps else "train"
        is_noise_cap = cap_type(stem) == "noise"
        iq = load_iq_ci16(c)
        cm = {} if is_noise_cap else coherent_masks_for(stem, args.labels_dir)
        if not is_noise_cap and not cm:
            print(f"  !! {stem}: NO coherent masks in labels-dir -- skipping (run label_ota.sh first)", flush=True)
            continue
        t0 = time.time(); kept = 0
        F_l, M_l = split_data[split]
        n_sig, n_noise = split_cnt[split]
        for fidx, fr in frames_of(iq):
            o = fe.compute_front_end(torch.from_numpy(fr).to(device).unsqueeze(0), FS, cfg, fft_size=FFT)
            resized = (o.resized[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()  # [512,1024]
            if is_noise_cap:
                gt = np.zeros((ROWS, cfg.nfft), dtype=np.uint8)
            else:
                p = cm.get(fidx + 1)   # offline masks are 1-based (f1 == fidx 0)
                if p is None:
                    continue
                gt = despeckle(resize_mask_freq_blockmax(np.load(p), cfg.nfft), args.min_comp_px).astype(np.uint8)
            for ti in range(ROWS // TILE):
                r0 = ti * TILE
                ftile = resized[r0:r0 + TILE]; mtile = gt[r0:r0 + TILE]
                is_sig = int(mtile.sum()) >= args.min_sig_px
                if is_sig:
                    keep = True
                elif is_noise_cap:
                    keep = True   # dedicated antenna-off negatives: keep every tile (that is the point)
                else:
                    keep = (n_noise < args.noise_ratio * max(1, n_sig)) and (rng.random() < 0.9)
                if not keep:
                    continue
                pos = len(F_l)
                F_l.append(ftile); M_l.append(mtile); kept += 1
                n_sig += int(is_sig); n_noise += int(not is_sig)
                split_csv[split].append({"frame_id": f"{stem}#f{fidx}t{ti}", "stem": stem, "split": split,
                                         "attenuation_db": "", "frame_index": fidx,
                                         "abs_start": fidx * FSAMP, "is_signal": int(is_sig),
                                         "n_signal_px": int(mtile.sum()), "mem_pos": pos})
        split_cnt[split] = [n_sig, n_noise]
        print(f"  [{split}] {stem}: {kept} tiles ({time.time()-t0:.1f}s)  running sig={n_sig} noise={n_noise}", flush=True)

    all_csv = []
    for sp in ("train", "val"):
        F_l, M_l = split_data[sp]
        if not F_l:
            print(f"[{sp}] EMPTY"); continue
        F = np.stack(F_l).astype(np.uint8); M = np.stack(M_l).astype(np.uint8)
        np.save(out / f"frames_{sp}.npy", F); np.save(out / f"masks_{sp}.npy", M)
        all_csv.extend(split_csv[sp])
        meta["counts"][sp] = int(F.shape[0])
        meta["sig_noise"][sp] = {"signal": split_cnt[sp][0], "noise": split_cnt[sp][1]}
        print(f"[{sp}] wrote {F.shape}  ({split_cnt[sp][0]} sig, {split_cnt[sp][1]} noise)  mask occ {100*M.mean():.2f}%", flush=True)
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_csv[0].keys())); w.writeheader(); w.writerows(all_csv)
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", meta["counts"], flush=True)


if __name__ == "__main__":
    main()

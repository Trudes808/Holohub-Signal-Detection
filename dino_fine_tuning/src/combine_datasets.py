"""Merge several RFSegDataset dirs (synthetic + OTA) into one, so train.py trains on the union
(domain-adaptation without catastrophic forgetting). Concatenates frames_{split}.npy / masks_{split}.npy
and remaps each source's frames.csv mem_pos into the concatenated array.

  python combine_datasets.py --inputs data/dataset_491 data/dataset_ota --out data/dataset_491_plus_ota
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

SPLITS = ("train", "val", "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    inputs = [Path(p) for p in args.inputs]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    all_csv = []
    meta = {"source": "combined", "inputs": [str(p) for p in inputs], "counts": {}, "sig_noise": {}}
    for base in inputs:
        m = json.loads((base / "dataset_meta.json").read_text())
        for k in ("nfft", "tile_rows", "fft_size", "rows_wide", "sample_rate"):
            meta.setdefault(k, m.get(k))

    for sp in SPLITS:
        frames, masks, rows_out, offset = [], [], [], 0
        for base in inputs:
            fp = base / f"frames_{sp}.npy"
            if not fp.exists():
                continue
            F = np.load(fp, mmap_mode="r"); M = np.load(base / f"masks_{sp}.npy", mmap_mode="r")
            rows = [r for r in csv.DictReader(open(base / "frames.csv")) if r["split"] == sp]
            # sanity: rows for this split index 0..N-1 of this source's array
            for r in rows:
                r = dict(r); r["mem_pos"] = str(int(r["mem_pos"]) + offset); rows_out.append(r)
            frames.append(F); masks.append(M); offset += F.shape[0]
        if not frames:
            continue
        Fc = np.concatenate([np.asarray(f) for f in frames], axis=0)
        Mc = np.concatenate([np.asarray(m) for m in masks], axis=0)
        np.save(out / f"frames_{sp}.npy", Fc); np.save(out / f"masks_{sp}.npy", Mc)
        all_csv.extend(rows_out)
        meta["counts"][sp] = int(Fc.shape[0])
        meta["sig_noise"][sp] = {"signal": sum(int(r["is_signal"]) for r in rows_out),
                                 "noise": sum(1 - int(r["is_signal"]) for r in rows_out)}
        print(f"[{sp}] combined {Fc.shape}  occ {100*np.asarray(Mc).mean():.2f}%", flush=True)

    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_csv[0].keys())); w.writeheader(); w.writerows(all_csv)
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print("DONE", meta["counts"], flush=True)


if __name__ == "__main__":
    main()

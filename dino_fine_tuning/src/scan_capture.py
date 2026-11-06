"""Per-frame / per-tile scan of one SigMF capture with several models + front-end statistics, to localise
failure modes (whole-tile fills, misses). Writes <out>/scan.csv and <out>/scan.json; optional panels for the
worst frames (--panels N) with: fixed-clip dB image, robust-normalised model input, each model's mask, and the
probability map of --prob-model.

  PYTHONPATH=src .venv-ml/bin/python src/scan_capture.py --capture <x.sigmf-meta> --models M5S=... M7S=... --out <dir> [--panels 6]
"""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

import frontend as fe
import rates as R
from eval_scenes import Model, load_iq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--max-frames", type=int, default=100000); ap.add_argument("--panels", type=int, default=6)
    ap.add_argument("--panel-model", default=None, help="model whose occupancy picks the panel frames (default: last)")
    ap.add_argument("--panel-frames", nargs="*", type=int, default=None)
    ap.add_argument("--train-yaml", default="configs/train.yaml")
    ap.add_argument("--fill-guard", action="store_true", help="also report occupancy after postfilter.fill_guard (columns occ_<model>_fg)")
    a = ap.parse_args()
    dev = "cuda"; cfg = fe.FrontEndCfg()
    models = {}
    for spec in a.models:
        n, p = spec.split("=", 1); models[n] = Model(p, a.train_yaml, dev)
    names = list(models); pm = a.panel_model or names[-1]
    meta_p = Path(a.capture); meta = json.loads(meta_p.read_text()); g = meta["global"]
    rate = float(g["core:sample_rate"]); dtype = g.get("core:datatype", "cf32_le")
    data_p = meta_p.with_suffix(".sigmf-data")
    bps = 4 if dtype.startswith("ci16") else (16 if dtype.startswith("cf64") else 8)
    n_total = data_p.stat().st_size // bps
    F = R.app_fft_size(rate); rows = R.ROWS_WIDE; FS = rows * F
    nfr = min(n_total // FS, a.max_frames)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    print(f"{meta_p.stem}: {nfr} frames @ {rate/1e6:.2f} MSps, fft {F}", flush=True)
    for fidx in range(nfr):
        iq = load_iq(data_p, dtype, fidx * FS, FS)
        fr = torch.from_numpy(iq).to(dev).unsqueeze(0)
        o = fe.compute_front_end(fr, rate, cfg, fft_size=F)
        img = o.resized[0]                                              # [512,1024] robust-normalised
        # front-end statistics on the (flattened) dB image: floor, spread, fraction of pixels well above floor
        wide = o.wide_norm[0]                                           # [512, F] normalised (robust) values
        v = img.flatten()
        q = torch.quantile(v[:: max(1, v.numel() // 500_000)], torch.tensor([0.05, 0.2, 0.5, 0.8, 0.95, 0.99], device=dev)).tolist()
        energy_occ = float((img > 0.45).float().mean().item())        # ~> floor + 11 dB
        rec = {"frame": fidx, "used_robust": int(bool(o.used_robust[0])) if o.used_robust else -1,
               "floor_db": float(o.floor_db[0]) if o.floor_db else float("nan"),
               "p05": q[0], "p20": q[1], "p50": q[2], "p80": q[3], "p95": q[4], "p99": q[5], "energy_occ": energy_occ,
               "iq_rms": float(np.sqrt(np.mean(np.abs(iq[:2_000_000]) ** 2))), "iq_peak": float(np.max(np.abs(iq[:2_000_000].real)))}
        for n, m in models.items():
            prob = m.prob(o.tiles)                                      # [2,256,1024]
            mk = (prob >= a.thr)
            rec[f"occ_{n}"] = float(mk.float().mean().item())
            rec[f"occ_{n}_t0"] = float(mk[0].float().mean().item()); rec[f"occ_{n}_t1"] = float(mk[1].float().mean().item())
            rec[f"pmean_{n}"] = float(prob.mean().item())
            if a.fill_guard:
                from postfilter import fill_guard as _fg
                fgm, gtiles = _fg(mk.to(torch.uint8).cpu().numpy().reshape(-1, cfg.nfft)[:rows], img.clamp(0, 1).cpu().numpy())
                rec[f"occ_{n}_fg"] = float(fgm.mean()); rec[f"fg_tiles_{n}"] = len(gtiles)
        recs.append(rec)
        if fidx % 20 == 0:
            print(f"  f{fidx}: " + " ".join(f"{n} {100*rec[f'occ_{n}']:.1f}%" for n in names) + f" | floor {rec['floor_db']:.1f} dB, energy_occ {100*energy_occ:.1f}%, robust {rec['used_robust']}", flush=True)
    with open(out / "scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    # summary
    summ = {"capture": meta_p.stem, "rate_hz": rate, "frames": nfr, "threshold": a.thr}
    for n in names:
        occ = np.array([r[f"occ_{n}"] for r in recs]); t = np.array([[r[f"occ_{n}_t0"], r[f"occ_{n}_t1"]] for r in recs])
        summ[n] = {"occ_mean": float(occ.mean()), "occ_median": float(np.median(occ)), "occ_p95": float(np.percentile(occ, 95)),
                   "tiles_over_50pct": int((t > 0.5).sum()), "tiles_over_80pct": int((t > 0.8).sum()), "tiles_total": int(t.size),
                   "frames_over_50pct": int((occ > 0.5).sum())}
    eo = np.array([r["energy_occ"] for r in recs]); summ["energy_occ_mean"] = float(eo.mean()); summ["energy_occ_p95"] = float(np.percentile(eo, 95))
    (out / "scan.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    # panels for the worst frames of the panel model (or chosen frames)
    if a.panel_frames:
        sel = a.panel_frames
    else:
        occ = np.array([r[f"occ_{pm}"] for r in recs]); sel = list(np.argsort(-occ)[: a.panels])
    for fidx in sel:
        iq = load_iq(data_p, dtype, int(fidx) * FS, FS); fr = torch.from_numpy(iq).to(dev).unsqueeze(0)
        o = fe.compute_front_end(fr, rate, cfg, fft_size=F); img = o.resized[0].clamp(0, 1).cpu().numpy()
        # fixed-clip image (absolute level) for comparison
        cfg_fixed = fe.FrontEndCfg(adaptive=False); of = fe.compute_front_end(fr, rate, cfg_fixed, fft_size=F); img_fixed = of.resized[0].clamp(0, 1).cpu().numpy()
        ext = [-rate / 2e6, rate / 2e6, 0, rows * F / rate * 1e3]
        nr = 2 + len(names) + 1
        fig, ax = plt.subplots(nr, 1, figsize=(14, 3.1 * nr), sharex=True)
        ax[0].imshow(img_fixed, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext)
        ax[0].set_title(f"{meta_p.stem} frame {fidx}: FIXED-clip dB image (absolute level; vmin {cfg.db_vmin:.1f} vmax {cfg.db_vmax:.1f})", fontsize=10)
        r = recs[int(fidx)]
        ax[1].imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext)
        ax[1].set_title(f"robust-normalised model input: floor {r['floor_db']:.1f} dB, robust={r['used_robust']}, p20/p50/p95/p99 = {r['p20']:.2f}/{r['p50']:.2f}/{r['p95']:.2f}/{r['p99']:.2f}, energy occ (>0.45) {100*r['energy_occ']:.1f}%", fontsize=10)
        for i, (n, m) in enumerate(models.items(), start=2):
            prob = m.prob(o.tiles).cpu().numpy().reshape(-1, cfg.nfft)[:rows]
            mk = prob >= a.thr
            ax[i].imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
            ax[i].imshow(np.ma.masked_where(~mk, mk), aspect="auto", origin="lower", cmap="autumn", alpha=0.55, extent=ext, vmin=0, vmax=1)
            ax[i].set_title(f"{n} mask: occupancy {100*mk.mean():.1f}% (tiles {100*mk[:256].mean():.0f}% / {100*mk[256:].mean():.0f}%)", fontsize=10)
            if n == pm:
                pim = ax[-1].imshow(prob, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1, extent=ext)
                ax[-1].set_title(f"{n} probability map (mean {prob.mean():.2f})", fontsize=10)
        for x in ax: x.set_ylabel("ms")
        ax[-1].set_xlabel("MHz"); fig.tight_layout(); fig.savefig(out / f"frame_{int(fidx):04d}.png", dpi=80); plt.close(fig)
        print("panel", fidx, flush=True)


if __name__ == "__main__":
    main()

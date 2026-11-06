"""Held-out evaluation of a DINO-FT model on SigMF scenes at ANY sample rate (synthetic gen_scenes output or
real recordings), through the deployment front-end at each file's own rate.

Reports, per scenario / rate bucket / class / PSD-SNR / band-edge visibility / size:
  * region detection: fraction of GT boxes with mask coverage >= 0.1 and >= 0.5 (the eval harness convention)
  * pixel precision / recall / IoU vs the rasterized GT
  * false alarms on annotation-free scenes (noise_only, off-air): predicted occupancy

Models: a training checkpoint (best.pt, via model.DinoSegmenter) or a deployed TorchScript export (.ts).

  PYTHONPATH=src .venv-ml/bin/python src/eval_scenes.py --model checkpoints/M5S_491_ota/best.pt --name M5S \
      --sigmf data/dataset_scenes/heldout_sigmf --out eval_out/scenes_M5S [--thr 0.6] [--max-frames 6]
Real recordings: --sigmf data/hard_test_cases --real  (absolute-RF annotation edges are re-based to baseband;
NI RF-recording annotations that span the whole capture are tightened to 0.9 x signal:detail.bandwidth).
"""
from __future__ import annotations
import argparse, csv, json, math, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch

import frontend as fe
import rates as R

HERE = Path(__file__).resolve().parents[1]
SNR_BUCKETS = [(-99, 0, "<0"), (0, 6, "0-6"), (6, 12, "6-12"), (12, 20, "12-20"), (20, 99, ">=20")]


# --------------------------------------------------------------------------- model
class Model:
    def __init__(self, path: str, train_yaml: str, device="cuda"):
        self.device = device
        p = Path(path)
        if p.suffix == ".ts":
            self.m = torch.jit.load(str(p), map_location=device).eval(); self.kind = "ts"
        else:
            from eval_heldout import load_model
            self.m = load_model(str(p), train_yaml, device); self.kind = "pt"

    @torch.no_grad()
    def prob(self, tiles: torch.Tensor) -> torch.Tensor:
        out = []
        for i in range(0, tiles.shape[0], 16):
            x = tiles[i:i + 16].to(self.device).float()
            if self.kind == "pt":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = self.m(x)
            else:
                lg = self.m(x)
            out.append(torch.sigmoid(lg.float()).squeeze(1))
        return torch.cat(out)


# --------------------------------------------------------------------------- SigMF
def load_iq(path: Path, dtype: str, offset: int, n: int) -> np.ndarray:
    if dtype.startswith("ci16"):
        raw = np.fromfile(str(path), dtype="<i2", count=2 * n, offset=4 * offset)
        m = (raw.size // 2) * 2
        return ((raw[0:m:2].astype(np.float32) + 1j * raw[1:m:2].astype(np.float32)) / 32767.0).astype(np.complex64)
    if dtype.startswith("ci8"):
        raw = np.fromfile(str(path), dtype=np.int8, count=2 * n, offset=2 * offset)
        m = (raw.size // 2) * 2
        return ((raw[0:m:2].astype(np.float32) + 1j * raw[1:m:2].astype(np.float32)) / 127.0).astype(np.complex64)
    if dtype.startswith("cf64"):
        return np.fromfile(str(path), dtype=np.complex128, count=n, offset=16 * offset).astype(np.complex64)
    return np.fromfile(str(path), dtype=np.complex64, count=n, offset=8 * offset)


def parse_annotations(meta: dict, rate: float, real: bool, n_total: int = 0, continuous: bool = False) -> list[dict]:
    center = float(meta.get("captures", [{}])[0].get("core:frequency", 0.0) or 0.0)
    out = []
    for a in meta.get("annotations", []):
        if continuous and n_total:               # carriers are on for the whole recording; labels only mark a window
            a = dict(a); a["core:sample_start"] = 0; a["core:sample_count"] = n_total
        lo = float(a.get("core:freq_lower_edge", -rate / 2)); hi = float(a.get("core:freq_upper_edge", rate / 2))
        tx = a.get("system_components:transmitter")
        if real and tx:
            # NI RF data recording API: the annotation spans the acquisition; tighten to the emitter
            det = list(tx.values())[0]
            bw = float(det.get("signal:detail", {}).get("bandwidth", hi - lo))
            f_tx = float(det.get("signal:emitter", {}).get("frequency", center) or center)
            fc = f_tx - center
            lo, hi = fc - 0.9 * bw / 2, fc + 0.9 * bw / 2
        elif max(abs(lo), abs(hi)) > rate:          # absolute RF edges -> baseband
            lo, hi = lo - center, hi - center
        out.append({"core:sample_start": int(a.get("core:sample_start", 0)), "core:sample_count": int(a.get("core:sample_count", 0)),
                    "core:freq_lower_edge": lo, "core:freq_upper_edge": hi,
                    "label": str(a.get("core:label", a.get("core:description", "signal"))),
                    "emitter": str(a.get("wfgt:emitter", a.get("wfgt:kind", ""))),
                    "snr": a.get("wfgt:snr_psd_db", None), "bw": hi - lo})
    return out


def snr_bucket(v):
    if v is None:
        return "n/a"
    for lo, hi, name in SNR_BUCKETS:
        if lo <= v < hi:
            return name
    return "n/a"


def vis_bucket(frac):
    return "<0.5 (mostly cut)" if frac < 0.5 else ("0.5-0.9 (cut)" if frac < 0.9 else ">=0.9 (in band)")


def bw_bucket(bins):
    return "<2 bins" if bins < 2 else ("2-8 bins" if bins < 8 else ("8-32 bins" if bins < 32 else (">=32 bins" if bins < 256 else ">=256 bins")))


def dur_bucket(rows):
    return "<3 rows" if rows < 3 else ("3-16 rows" if rows < 16 else (">=16 rows" if rows < 128 else ">=128 rows"))


# --------------------------------------------------------------------------- eval
def eval_scene(model: Model, meta_p: Path, cfg, thr: float, max_frames: int, real: bool, device, png: Path | None,
               continuous: bool = False, fill_guard: bool = False):
    meta = json.loads(meta_p.read_text()); g = meta["global"]
    rate = float(g["core:sample_rate"]); dtype = g.get("core:datatype", "cf32_le")
    data_p = meta_p.with_suffix(".sigmf-data")
    bps = 4 if dtype.startswith("ci16") else (2 if dtype.startswith("ci8") else (16 if dtype.startswith("cf64") else 8))
    n_total = data_p.stat().st_size // bps
    F = R.app_fft_size(rate); FS = R.ROWS_WIDE * F
    nfr = n_total // FS
    if nfr == 0:   # short recording: evaluate the rows that exist (>= 1 tile)
        rows = max(256, (n_total // F) // 256 * 256); FS_eff = rows * F; nfr_eff = 1 if n_total >= FS_eff else 0
    else:
        rows = R.ROWS_WIDE; FS_eff = FS; nfr_eff = min(nfr, max_frames)
    anns = parse_annotations(meta, rate, real, n_total, continuous)
    scenario = g.get("wfgt:scenario", "real")
    tp = fp = fn = 0; pred_px = 0; tot_px = 0
    regions = []
    overlay = None
    stride = max(1, nfr // max(1, nfr_eff)) if nfr else 1
    for k in range(nfr_eff):
        fidx = k * stride
        iq = load_iq(data_p, dtype, fidx * FS_eff, FS_eff)
        if iq.size < FS_eff:
            break
        fr = torch.from_numpy(iq).to(device).unsqueeze(0)
        out = fe.compute_front_end(fr, rate, cfg, fft_size=F)
        prob = model.prob(out.tiles)                                  # [n_tiles,256,1024]
        mask = (prob >= thr).to(torch.uint8).cpu().numpy().reshape(-1, cfg.nfft)[:rows]
        if fill_guard:
            from postfilter import fill_guard as _fg
            mask, _ = _fg(mask, out.resized[0].clamp(0, 1).cpu().numpy()[:rows])
        gt = fe.rasterize_gt(anns, fidx * FS_eff, F, rows, cfg.nfft, rate).numpy()
        m = mask.astype(bool); gtb = gt.astype(bool)
        tp += int((m & gtb).sum()); fp += int((m & ~gtb).sum()); fn += int((~m & gtb).sum())
        pred_px += int(m.sum()); tot_px += m.size
        f0 = fidx * FS_eff; f1 = f0 + rows * F
        for a in anns:
            s, c = a["core:sample_start"], a["core:sample_count"]
            if s >= f1 or s + c <= f0:
                continue
            r0 = max(0, (s - f0) // F); r1 = min(rows, -(-(s + c - f0) // F))
            lo, hi = a["core:freq_lower_edge"], a["core:freq_upper_edge"]
            c0 = max(0, int(math.floor((lo + rate / 2) / rate * cfg.nfft))); c1 = min(cfg.nfft, int(math.ceil((hi + rate / 2) / rate * cfg.nfft)))
            r1 = max(r1, r0 + 1); c1 = max(c1, c0 + 1)
            if r1 <= r0 or c1 <= c0 or c0 >= cfg.nfft:
                continue
            box = m[r0:r1, c0:c1]
            cov = float(box.mean()) if box.size else float("nan")
            vis = (min(hi, rate / 2) - max(lo, -rate / 2)) / max(hi - lo, 1e-9)
            regions.append({"scene": meta_p.stem, "scenario": scenario, "rate_bucket": R.rate_bucket(rate), "rate_mhz": rate / 1e6,
                            "label": a["label"], "emitter": a["emitter"], "snr": a["snr"], "snr_bucket": snr_bucket(a["snr"]),
                            "vis_frac": vis, "vis_bucket": vis_bucket(vis), "bw_bins": (hi - lo) / (rate / cfg.nfft),
                            "bw_bucket": bw_bucket((hi - lo) / (rate / cfg.nfft)), "rows": c / F, "dur_bucket": dur_bucket(c / F),
                            "coverage": cov, "det01": int(cov >= 0.1), "det05": int(cov >= 0.5), "frame": fidx})
        if png is not None and overlay is None:
            overlay = (out.resized[0].clamp(0, 1).cpu().numpy(), gt, mask, fidx)
    if png is not None and overlay is not None:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        img, gt, mask, fidx = overlay
        fig, ax = plt.subplots(2, 1, figsize=(14, 7.5))
        ext = [-rate / 2e6, rate / 2e6, 0, rows * F / rate * 1e3]
        ax[0].imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1, extent=ext)
        xs = np.linspace(ext[0], ext[1], cfg.nfft); ys = np.linspace(0, ext[3], rows)
        if gt.any(): ax[0].contour(xs, ys, gt, levels=[0.5], colors="red", linewidths=0.6)
        ax[0].set_title(f"{meta_p.stem} @ {rate/1e6:.2f} MSps frame {fidx}: input + GT (red)")
        ax[1].imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=0, vmax=1, extent=ext)
        ax[1].imshow(np.ma.masked_where(mask == 0, mask), aspect="auto", origin="lower", cmap="autumn", alpha=0.55, extent=ext)
        if gt.any(): ax[1].contour(xs, ys, gt, levels=[0.5], colors="cyan", linewidths=0.5)
        ax[1].set_title(f"prediction (orange, thr {thr}) vs GT (cyan); pred occ {100*mask.mean():.2f}%  GT occ {100*gt.mean():.2f}%")
        for a_ in ax: a_.set_xlabel("MHz"); a_.set_ylabel("ms")
        fig.tight_layout(); png.parent.mkdir(parents=True, exist_ok=True); fig.savefig(png, dpi=85); plt.close(fig)
    return {"scene": meta_p.stem, "scenario": scenario, "rate_hz": rate, "rate_bucket": R.rate_bucket(rate), "frames": nfr_eff,
            "tp": tp, "fp": fp, "fn": fn, "pred_px": pred_px, "tot_px": tot_px, "n_ann": len(anns)}, regions


def table(rows, key, title):
    agg = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        a = agg[r[key]]; a[0] += r["det01"]; a[1] += r["det05"]; a[2] += 1
    lines = [f"**{title}**", "", "| group | regions | det@0.1 | det@0.5 |", "|---|---|---|---|"]
    for k in sorted(agg, key=lambda x: str(x)):
        d1, d5, n = agg[k]
        lines.append(f"| {k} | {n} | {100*d1/n:.1f}% | {100*d5/n:.1f}% |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--name", default=None)
    ap.add_argument("--train-yaml", default="configs/train.yaml")
    ap.add_argument("--sigmf", nargs="+", required=True, help="dirs (recursive) or .sigmf-meta files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--max-frames", type=int, default=6, help="frames evaluated per scene (spread over the file)")
    ap.add_argument("--real", action="store_true", help="real recordings: re-base RF annotation edges, tighten NI annotations")
    ap.add_argument("--png-per-scenario", type=int, default=2)
    ap.add_argument("--assume-continuous", action="store_true", help="real: extend every annotation over the whole recording")
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--fill-guard", action="store_true", help="apply postfilter.fill_guard to every mask")
    a = ap.parse_args()
    device = "cuda"
    cfg = fe.FrontEndCfg()
    name = a.name or Path(a.model).stem
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = Model(a.model, a.train_yaml, device)
    metas = []
    for s in a.sigmf:
        p = Path(s)
        metas += sorted(p.rglob("*.sigmf-meta")) if p.is_dir() else [p]
    if a.max_scenes:
        metas = metas[: a.max_scenes]
    print(f"[{name}] {len(metas)} scenes, thr {a.thr}", flush=True)
    scene_rows, regions = [], []
    png_count = defaultdict(int); t0 = time.time()
    for i, mp in enumerate(metas):
        scen = json.loads(mp.read_text())["global"].get("wfgt:scenario", "real")
        png = None
        if png_count[scen] < a.png_per_scenario:
            png = out / "overlays" / f"{scen}_{mp.stem}.png"; png_count[scen] += 1
        try:
            srow, regs = eval_scene(model, mp, cfg, a.thr, a.max_frames, a.real, device, png, a.assume_continuous, a.fill_guard)
        except Exception as e:
            print(f"  !! {mp.name}: {e}", flush=True); continue
        scene_rows.append(srow); regions += regs
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(metas)} ({time.time()-t0:.0f}s)", flush=True)
    # pixel metrics by scenario and rate bucket
    def pix(rows_):
        tp = sum(r["tp"] for r in rows_); fp = sum(r["fp"] for r in rows_); fn = sum(r["fn"] for r in rows_)
        p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1); iou = tp / max(tp + fp + fn, 1)
        return p, rc, iou
    md = [f"# Held-out scene evaluation: {name} (thr {a.thr})", "", f"scenes {len(scene_rows)}, GT regions {len(regions)}, "
          f"frames/scene <= {a.max_frames}", ""]
    P, Rc, I = pix(scene_rows)
    md += [f"**Overall pixel**: precision {P:.3f} recall {Rc:.3f} IoU {I:.3f}", ""]
    md += ["**Pixel metrics by scenario**", "", "| scenario | scenes | precision | recall | IoU | pred occ |", "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in scene_rows: by[r["scenario"]].append(r)
    for k in sorted(by):
        p_, r_, i_ = pix(by[k]); occ = sum(x["pred_px"] for x in by[k]) / max(1, sum(x["tot_px"] for x in by[k]))
        md.append(f"| {k} | {len(by[k])} | {p_:.3f} | {r_:.3f} | {i_:.3f} | {100*occ:.2f}% |")
    md.append("")
    noise = [r for r in scene_rows if r["n_ann"] == 0]
    if noise:
        occ = sum(x["pred_px"] for x in noise) / max(1, sum(x["tot_px"] for x in noise))
        md += [f"**False alarms on {len(noise)} annotation-free scenes**: predicted occupancy {100*occ:.3f}%", ""]
    if regions:
        md.append(table(regions, "scenario", "Region detection by scenario (all GT boxes)"))
        # headline breakdowns exclude sub-pixel boxes (< 2 rows AND < 2 bins: radar ticks the grid cannot resolve)
        main = [r for r in regions if not (r["rows"] < 2 and r["bw_bins"] < 2) and r["scenario"] != "radar_pulses"]
        sub = len(regions) - len(main)
        if main:
            per_sc = defaultdict(lambda: [0, 0, 0])
            for r in main:
                a = per_sc[r["scenario"]]; a[0] += r["det01"]; a[1] += r["det05"]; a[2] += 1
            macro1 = np.mean([a[0] / a[2] for a in per_sc.values()]); macro5 = np.mean([a[1] / a[2] for a in per_sc.values()])
            md += [f"**Headline (excluding {sub} sub-pixel boxes and the radar_pulses scenario)**: {len(main)} regions, det@0.1 {100*np.mean([r['det01'] for r in main]):.1f}% "
                   f"det@0.5 {100*np.mean([r['det05'] for r in main]):.1f}%; scenario macro-average det@0.1 {100*macro1:.1f}% det@0.5 {100*macro5:.1f}%", ""]
            for key, title in [("scenario", "by scenario (excl. sub-pixel)"), ("rate_bucket", "by sample-rate bucket"),
                               ("snr_bucket", "by PSD SNR (dB above floor)"), ("vis_bucket", "by in-band fraction (band-edge cut)"),
                               ("bw_bucket", "by bandwidth (model bins)"), ("dur_bucket", "by duration (wide rows)"), ("emitter", "by emitter behaviour"), ("label", "by class label")]:
                md.append(table(main, key, title))
    (out / "summary.md").write_text("\n".join(md))
    (out / "scenes.json").write_text(json.dumps(scene_rows, indent=1))
    with open(out / "regions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(regions[0].keys()) if regions else ["scene"]); w.writeheader(); w.writerows(regions)
    print("\n".join(md[:40]))
    print("wrote", out / "summary.md")


if __name__ == "__main__":
    main()

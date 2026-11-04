"""Build the multi-rate, multi-scenario DINO-FT training dataset from gen_scenes.py scenes.

Scenes are synthesized in parallel worker processes (CPU) and streamed through the deployment front-end
(frontend.py, GPU) in the main process at EACH scene's own X410 rate: frame = 512 rows x app_fft_size(rate)
samples, wide FFT -> flatten -> robust norm -> bilinear resize to 1024 -> two 256-row tiles per frame; GT
rasterized on the same wide-row grid (frontend.rasterize_gt).  Output has the RFSegDataset layout
(frames_{split}.npy / masks_{split}.npy uint8 + frames.csv + dataset_meta.json) so train.py / combine_datasets.py
consume it unchanged.  Held-out test scenes are also written as SigMF (heldout_sigmf/<scenario>/) for
eval_scenes.py.

  PYTHONPATH=src .venv-ml/bin/python src/build_dataset_scenes.py --out-dir data/dataset_scenes \
      --train-scenes 3000 --val-scenes 300 --test-per-cell 2 --workers 12
  ... --smoke   (tiny build to validate the pipeline)
"""
from __future__ import annotations
import argparse, csv, json, os, time, traceback
from collections import Counter, defaultdict
from multiprocessing import get_context
from pathlib import Path
import numpy as np

import rates as R
import gen_scenes as g

HERE = Path(__file__).resolve().parents[1]
TMP = Path(os.environ.get("SCENES_TMP", "/tmp/claude-1000/-home-genesys-dgx1-Documents-Holohub-Signal-Detection/34efe362-daf9-4094-b44f-3c2b71212f7c/scratchpad/scene_tmp"))
TILE, NFFT, ROWS = 256, 1024, R.ROWS_WIDE


def worker(spec):
    """(split, idx, scenario, rate, seed, sigmf_stem|None) -> dict with canvas path + annotations + meta."""
    split, idx, scenario, rate, seed, sigmf_stem = spec
    try:
        sc = g.make_scene(scenario, rate, seed)
        TMP.mkdir(parents=True, exist_ok=True)
        p = TMP / f"{split}_{idx}_{os.getpid()}.npy"
        np.save(p, sc.canvas)
        if sigmf_stem:
            g.write_sigmf(sc, Path(sigmf_stem))
        return {"ok": True, "split": split, "idx": idx, "scenario": scenario, "rate": rate, "seed": seed,
                "path": str(p), "anns": g.annotations_of(sc), "meta": sc.meta, "n_frames": sc.n_frames}
    except Exception as e:
        return {"ok": False, "split": split, "idx": idx, "scenario": scenario, "rate": rate, "seed": seed,
                "err": f"{e}\n{traceback.format_exc()}"}


def make_specs(n: int, split: str, seed0: int, rng: np.random.Generator, p_deploy: float):
    specs = []
    scen = list(g.SCENARIO_WEIGHTS.keys()); w = np.asarray([g.SCENARIO_WEIGHTS[s] for s in scen]); w /= w.sum()
    for i in range(n):
        s = scen[rng.choice(len(scen), p=w)]
        r = R.draw_rate(rng, p_deploy=p_deploy)
        specs.append((split, i, s, r, seed0 + i, None))
    return specs


def make_test_specs(per_cell: int, seed0: int, rng: np.random.Generator, sigmf_root: Path):
    specs, i = [], 0
    rates_all = R.x410_rates()
    buckets = defaultdict(list)
    for r in rates_all:
        buckets[R.rate_bucket(r)].append(r)
    for s in g.SCENARIOS:
        for b, rs in buckets.items():
            for k in range(per_cell):
                if b == "400-500M":
                    r = float(rs[k % len(rs)])              # 491.52 and 500 both covered
                else:
                    r = float(rs[rng.integers(len(rs))])
                stem = sigmf_root / s / f"{s}_{b}_{r/1e6:.2f}M_s{seed0 + i}"
                specs.append(("test", i, s, r, seed0 + i, str(stem))); i += 1
    return specs


class Accum:
    def __init__(self, split, noise_ratio, min_sig_px, rng):
        self.split, self.noise_ratio, self.min_sig_px, self.rng = split, noise_ratio, min_sig_px, rng
        self.frames, self.masks, self.rows = [], [], []
        self.n_sig = self.n_noise = 0

    def add(self, ftile, mtile, info):
        is_sig = int(mtile.sum()) >= self.min_sig_px
        if not is_sig:
            if info["scenario"] != "noise_only" and not (self.n_noise < self.noise_ratio * max(1, self.n_sig) and self.rng.random() < 0.85):
                return
            if info["scenario"] == "noise_only" and not (self.n_noise < (self.noise_ratio + 0.15) * max(1, self.n_sig)):
                return
        pos = len(self.frames)
        self.frames.append(ftile); self.masks.append(mtile)
        self.n_sig += int(is_sig); self.n_noise += int(not is_sig)
        self.rows.append({"frame_id": info["frame_id"], "stem": info["stem"], "split": self.split, "attenuation_db": "",
                          "frame_index": info["fidx"], "abs_start": info["abs_start"], "is_signal": int(is_sig),
                          "n_signal_px": int(mtile.sum()), "mem_pos": pos, "scenario": info["scenario"],
                          "rate_hz": info["rate"], "fft_size": info["fft"], "rate_bucket": R.rate_bucket(info["rate"])})


def process_scene(res, acc: Accum, cfg, device, gallery: dict | None):
    import torch, frontend as fe
    rate = res["rate"]; F = R.app_fft_size(rate); FS = ROWS * F
    canvas = np.load(res["path"], mmap_mode="r")
    n = canvas.size // FS
    stem = f"{res['scenario']}_{rate/1e6:.2f}M_s{res['seed']}"
    for fidx in range(n):
        fr = torch.from_numpy(np.ascontiguousarray(canvas[fidx * FS:(fidx + 1) * FS])).to(device).unsqueeze(0)
        out = fe.compute_front_end(fr, rate, cfg, fft_size=F)
        resized = (out.resized[0].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
        gt = fe.rasterize_gt(res["anns"], fidx * FS, F, ROWS, NFFT, rate).numpy()
        for ti in range(ROWS // TILE):
            r0 = ti * TILE
            info = {"frame_id": f"{stem}#f{fidx}t{ti}", "stem": stem, "fidx": fidx, "abs_start": fidx * FS,
                    "scenario": res["scenario"], "rate": rate, "fft": F}
            acc.add(resized[r0:r0 + TILE], gt[r0:r0 + TILE], info)
            if gallery is not None and len(gallery[res["scenario"]]) < 6 and gt[r0:r0 + TILE].sum() > 0:
                gallery[res["scenario"]].append((resized[r0:r0 + TILE].copy(), gt[r0:r0 + TILE].copy(), f"{rate/1e6:.2f}M s{res['seed']}"))
    del canvas
    try:
        os.remove(res["path"])
    except OSError:
        pass


def save_gallery(gallery: dict, out: Path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for sc, items in gallery.items():
        if not items:
            continue
        fig, axes = plt.subplots(len(items), 1, figsize=(12, 2.6 * len(items)), squeeze=False)
        for ax, (img, m, tag) in zip(axes[:, 0], items):
            ax.imshow(img, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=255)
            ax.contour(m, levels=[0.5], colors="red", linewidths=0.5)
            ax.set_title(f"{sc} {tag}  occ {100*m.mean():.1f}%", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout(); fig.savefig(out / f"gallery_{sc}.png", dpi=80); plt.close(fig)


def run_split(split, specs, acc, cfg, device, workers, gallery, log_every=25):
    t0 = time.time(); done = fail = 0
    ctx = get_context("fork")
    with ctx.Pool(workers, maxtasksperchild=40) as pool:
        for res in pool.imap_unordered(worker, specs, chunksize=1):
            if not res["ok"]:
                fail += 1
                print(f"  !! {res['scenario']} @ {res['rate']/1e6:.2f}M seed {res['seed']} FAILED: {res['err'].splitlines()[-1]}", flush=True)
                continue
            process_scene(res, acc, cfg, device, gallery)
            done += 1
            if done % log_every == 0:
                el = time.time() - t0
                print(f"  [{split}] {done}/{len(specs)} scenes, tiles sig={acc.n_sig} noise={acc.n_noise}, "
                      f"{el/60:.1f} min, eta {el/done*(len(specs)-done)/60:.1f} min", flush=True)
    print(f"[{split}] done {done} ok / {fail} failed in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(HERE / "data/dataset_scenes"))
    ap.add_argument("--train-scenes", type=int, default=3000)
    ap.add_argument("--val-scenes", type=int, default=300)
    ap.add_argument("--test-per-cell", type=int, default=2, help="held-out SigMF scenes per (scenario x rate bucket)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--p-deploy", type=float, default=0.30, help="prob. a train/val scene is at 491.52 or 500 MSps")
    ap.add_argument("--noise-ratio", type=float, default=0.35)
    ap.add_argument("--min-sig-px", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--heldout-only", action="store_true", help="regenerate only the held-out SigMF scenes (same seeds), no tiles")
    a = ap.parse_args()
    if a.heldout_only:
        out = Path(a.out_dir); rng = np.random.default_rng(a.seed)
        _ = make_specs(a.train_scenes, "train", 1, rng, a.p_deploy); _ = make_specs(a.val_scenes, "val", 100000, rng, a.p_deploy)
        specs = make_test_specs(a.test_per_cell, 200000, rng, out / "heldout_sigmf")
        import shutil; shutil.rmtree(out / "heldout_sigmf", ignore_errors=True)
        ctx = get_context("fork"); n_ok = 0
        with ctx.Pool(a.workers, maxtasksperchild=40) as pool:
            for res in pool.imap_unordered(worker, specs, chunksize=1):
                if res["ok"]:
                    n_ok += 1; os.remove(res["path"])
                else:
                    print("  !!", res["err"].splitlines()[-1], flush=True)
        print(f"held-out SigMF regenerated: {n_ok}/{len(specs)} scenes ->", out / "heldout_sigmf", flush=True); return
    if a.smoke:
        a.train_scenes, a.val_scenes, a.test_per_cell, a.workers = 24, 6, 1, 6
        a.out_dir = a.out_dir.rstrip("/") + "_smoke"
    import torch, frontend as fe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = fe.FrontEndCfg()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    print(f"library {len(g.lib_index())} waveforms | floor templates {len(g.floor_templates())} | device {device} | out {out}", flush=True)

    splits = {"train": make_specs(a.train_scenes, "train", 1, rng, a.p_deploy),
              "val": make_specs(a.val_scenes, "val", 100000, rng, a.p_deploy)}
    if not a.skip_test:
        splits["test"] = make_test_specs(a.test_per_cell, 200000, rng, out / "heldout_sigmf")
        if a.smoke:
            splits["test"] = splits["test"][::7]
    meta = {"nfft": NFFT, "tile_rows": TILE, "fft_size": "per scene: 512 rows x app_fft_size(rate) (see frames.csv)",
            "rows_wide": ROWS, "sample_rate": "multi (X410 rate set 5-500 MSps)", "front_end": "robust",
            "generator": "gen_scenes.py", "scenario_weights": g.SCENARIO_WEIGHTS, "p_deploy": a.p_deploy,
            "counts": {}, "sig_noise": {}, "scenes": {}, "by_scenario": {}, "by_rate_bucket": {}}
    all_rows = []
    gallery = defaultdict(list)
    for split, specs in splits.items():
        print(f"[{split}] {len(specs)} scenes", flush=True)
        acc = Accum(split, a.noise_ratio, a.min_sig_px, np.random.default_rng(a.seed + len(split)))
        run_split(split, specs, acc, cfg, device, a.workers, gallery if split == "train" else None)
        if not acc.frames:
            print(f"[{split}] EMPTY"); continue
        Fa = np.stack(acc.frames).astype(np.uint8); Ma = np.stack(acc.masks).astype(np.uint8)
        np.save(out / f"frames_{split}.npy", Fa); np.save(out / f"masks_{split}.npy", Ma)
        all_rows.extend(acc.rows)
        meta["counts"][split] = int(Fa.shape[0]); meta["sig_noise"][split] = {"signal": acc.n_sig, "noise": acc.n_noise}
        meta["scenes"][split] = len(specs)
        meta["by_scenario"][split] = dict(Counter(r["scenario"] for r in acc.rows))
        meta["by_rate_bucket"][split] = dict(Counter(r["rate_bucket"] for r in acc.rows))
        print(f"[{split}] wrote {Fa.shape} ({acc.n_sig} signal, {acc.n_noise} noise) mask occ {100*Ma.mean():.2f}%", flush=True)
        del Fa, Ma, acc
    with open(out / "frames.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    (out / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    save_gallery(gallery, out)
    print("DONE", meta["counts"], flush=True)


if __name__ == "__main__":
    main()

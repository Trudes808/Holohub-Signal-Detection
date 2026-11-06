"""Regression probe for the quiet-band fill mode: noise-only scenes with a dense irregular spur comb (the pattern
that made M7-S/M8-S paint whole tiles). Reports, per model, the fraction of tiles painted > 50 % and the mean
occupancy vs ground truth, at several rates, on fixed seeds (deterministic).

  PYTHONPATH=src .venv-ml/bin/python src/probe_comb_fill.py --models M7S=checkpoints/M7S_scenes_v2/best.pt M10S=checkpoints/M10S_v5/best.pt
"""
from __future__ import annotations
import argparse, os
import numpy as np, torch
os.environ.setdefault("REALBG_P", "0.5")
import gen_scenes as g, frontend as fe
from eval_scenes import Model


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--rates", nargs="*", type=float, default=[10e6, 30.72e6, 61.44e6, 245.76e6]); ap.add_argument("--per-rate", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.6); ap.add_argument("--min-lines", type=int, default=35)
    a = ap.parse_args()
    cfg = fe.FrontEndCfg(); models = {n: Model(p, "configs/train.yaml", "cuda") for n, p in (s.split("=", 1) for s in a.models)}
    res = {n: [] for n in models}; gts = []
    for rate in a.rates:
        n, seed = 0, 1000
        while n < a.per_rate and seed < 1400:
            rng_probe = np.random.default_rng(seed); seed += 1
            sc = g.make_scene("noise_only", rate, int(rng_probe.integers(0, 2**31)))
            if sc.meta["receiver"].get("spur_comb", 0) < a.min_lines:
                continue
            n += 1; F = sc.fft_size; FS = sc.frame_samples
            o = fe.compute_front_end(torch.from_numpy(sc.canvas[:FS]).cuda().unsqueeze(0), rate, cfg, fft_size=F)
            gt = fe.rasterize_gt(g.annotations_of(sc), 0, F, 512, 1024, rate).numpy()
            gts.append(gt.mean())
            for name, m in models.items():
                mk = (m.prob(o.tiles) >= a.thr).cpu().numpy()
                res[name].append((rate, float(mk.mean()), float((mk.reshape(2, -1).mean(axis=1) > 0.5).mean()), float(((mk.reshape(-1, 1024) > 0) & (gt > 0)).sum() / max(gt.sum(), 1))))
    print(f"comb-fill probe: {len(gts)} scenes, GT occupancy mean {100*np.mean(gts):.1f}%  (thr {a.thr}, >= {a.min_lines} lines)")
    for name, rows in res.items():
        r = np.array(rows)
        line = f"  {name:14s} tiles>50%: {100*r[:,2].mean():5.1f}%  mean occ {100*r[:,1].mean():5.1f}%  line recall {100*r[:,3].mean():5.1f}%  | per rate: "
        line += "  ".join(f"{rate/1e6:.0f}M fill {100*r[r[:,0]==rate,2].mean():.0f}%" for rate in a.rates)
        print(line)


if __name__ == "__main__":
    main()

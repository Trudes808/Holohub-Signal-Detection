"""Side-by-side comparison of two (or more) eval_scenes.py runs -> markdown tables for the experiment record.

  PYTHONPATH=src .venv-ml/bin/python src/compare_evals.py --names M5S M6S_multirate --out <results.md>
reads eval_out/{scenes,hard,argus_lte,unlabelled}_<name>/{scenes.json,regions.csv} for each name.
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]


def load(name, kind):
    d = HERE / "eval_out" / f"{kind}_{name}"
    if not (d / "scenes.json").exists():
        return None, None
    scenes = json.loads((d / "scenes.json").read_text())
    regs = list(csv.DictReader(open(d / "regions.csv"))) if (d / "regions.csv").exists() else []
    for r in regs:
        for k in ("det01", "det05"):
            r[k] = int(r[k])
        for k in ("rows", "bw_bins", "vis_frac", "coverage"):
            r[k] = float(r[k]) if r.get(k) not in (None, "", "nan") else float("nan")
    return scenes, regs


def pix(scenes):
    tp = sum(s["tp"] for s in scenes); fp = sum(s["fp"] for s in scenes); fn = sum(s["fn"] for s in scenes)
    return tp / max(tp + fp, 1), tp / max(tp + fn, 1), tp / max(tp + fp + fn, 1)


def headline(regs):
    main = [r for r in regs if not (r["rows"] < 2 and r["bw_bins"] < 2) and r["scenario"] != "radar_pulses"]
    return main


def grouped(regs, key):
    g = defaultdict(lambda: [0, 0, 0])
    for r in regs:
        a = g[r[key]]; a[0] += r["det01"]; a[1] += r["det05"]; a[2] += 1
    return g


def table(names, data, key, title, min_n=10):
    groups = sorted(set(k for d in data.values() for k in d[key]))
    lines = [f"**{title}** (det@0.1 / det@0.5, %)", "", "| group | n | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for gname in groups:
        n = max(data[nm][key].get(gname, [0, 0, 0])[2] for nm in names)
        if n < min_n:
            continue
        cells = []
        for nm in names:
            d1, d5, nn = data[nm][key].get(gname, [0, 0, 0])
            cells.append(f"{100*d1/nn:.0f} / {100*d5/nn:.0f}" if nn else "–")
        lines.append(f"| {gname} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes-kind", default="scenes", help="eval_out/<kind>_<name> dirs of the synthetic held-out")
    a = ap.parse_args()
    md = [f"# Detector comparison: {' vs '.join(a.names)}", ""]
    # ---- synthetic held-out scenes
    S = {nm: load(nm, a.scenes_kind) for nm in a.names}
    if all(v[0] for v in S.values()):
        md += ["## Synthetic held-out scenes (120 scenes, 10 scenarios x 6 rate buckets, thr 0.6)", ""]
        md += ["| metric | " + " | ".join(a.names) + " |", "|---|" + "---|" * len(a.names)]
        rows = []
        for nm in a.names:
            sc, rg = S[nm]; p, r, i = pix(sc); m = headline(rg)
            noise = [s for s in sc if s["n_ann"] == 0]
            fa = sum(x["pred_px"] for x in noise) / max(1, sum(x["tot_px"] for x in noise))
            per_sc = grouped(m, "scenario")
            rows.append({"pixel precision": f"{p:.3f}", "pixel recall": f"{r:.3f}", "pixel IoU": f"{i:.3f}",
                         "region det@0.1 (headline)": f"{100*np.mean([x['det01'] for x in m]):.1f}%",
                         "region det@0.5 (headline)": f"{100*np.mean([x['det05'] for x in m]):.1f}%",
                         "scenario macro det@0.1": f"{100*np.mean([v[0]/v[2] for v in per_sc.values()]):.1f}%",
                         "scenario macro det@0.5": f"{100*np.mean([v[1]/v[2] for v in per_sc.values()]):.1f}%",
                         "false-alarm occupancy (noise_only)": f"{100*fa:.3f}%"})
        for k in rows[0]:
            md.append(f"| {k} | " + " | ".join(r[k] for r in rows) + " |")
        md.append("")
        data = {nm: {key: grouped(headline(S[nm][1]), key) for key in ("scenario", "rate_bucket", "snr_bucket", "vis_bucket", "bw_bucket", "dur_bucket", "emitter", "label")} for nm in a.names}
        for key, title in [("scenario", "by scenario"), ("rate_bucket", "by sample-rate bucket"), ("snr_bucket", "by PSD SNR (dB above floor)"),
                           ("vis_bucket", "by in-band fraction (band-edge cut)"), ("bw_bucket", "by bandwidth (model bins)"),
                           ("dur_bucket", "by duration (wide rows)"), ("emitter", "by emitter behaviour"), ("label", "by class label")]:
            md.append(table(a.names, data, key, title))
        # pixel by scenario
        md += ["**Pixel recall by scenario** (precision in brackets)", "", "| scenario | " + " | ".join(a.names) + " |", "|---|" + "---|" * len(a.names)]
        scen = sorted(set(s["scenario"] for s in S[a.names[0]][0]))
        for sname in scen:
            cells = []
            for nm in a.names:
                ss = [s for s in S[nm][0] if s["scenario"] == sname]; p, r, i = pix(ss)
                cells.append(f"{r:.2f} ({p:.2f})" if sname != "noise_only" else f"occ {100*sum(x['pred_px'] for x in ss)/max(1,sum(x['tot_px'] for x in ss)):.3f}%")
            md.append(f"| {sname} | " + " | ".join(cells) + " |")
        md.append("")
    # ---- real labelled sets
    for kind, title in [("hard", "Real hard test cases (B210 880 MHz cellular/LMR @40 MSps, X310 LTE-FDD + NR 10 MHz @15.36 MSps; continuous labels)"),
                        ("argus_lte", "ARGUS LTE-only OOD captures (245.76 MSps, +46 / +26 / +6 / -4 dB)")]:
        D = {nm: load(nm, kind) for nm in a.names}
        if not all(v[0] for v in D.values()):
            continue
        md += [f"## {title}", "", "| metric | " + " | ".join(a.names) + " |", "|---|" + "---|" * len(a.names)]
        vals = []
        for nm in a.names:
            sc, rg = D[nm]; p, r, i = pix(sc)
            vals.append({"pixel precision": f"{p:.3f}", "pixel recall": f"{r:.3f}", "pixel IoU": f"{i:.3f}",
                         "region det@0.1": f"{100*np.mean([x['det01'] for x in rg]):.1f}%", "region det@0.5": f"{100*np.mean([x['det05'] for x in rg]):.1f}%"})
        for k in vals[0]:
            md.append(f"| {k} | " + " | ".join(v[k] for v in vals) + " |")
        md.append("")
        keyname = "label" if kind == "hard" else "scene"
        data = {nm: {keyname: grouped(D[nm][1], keyname)} for nm in a.names}
        md.append(table(a.names, data, keyname, "by " + keyname, min_n=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(md))
    print("\n".join(md[:30])); print("wrote", a.out)


if __name__ == "__main__":
    main()

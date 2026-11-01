#!/usr/bin/env python3
"""Roll every scored leg under the results root into one markdown + CSV summary.

  python3 summarize.py [--root /tmp/usrp_spectrograms/rt_stress] [--out results/summary.md]
Reads <root>/<run>/metrics.json (collect_metrics.py) and <root>/<run>/eval/summary.json
(score_leg.py, when present) and writes a speed table + an accuracy table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LEG_LABEL = {
    "coherent": "coherent_power (dynamic floor)",
    "dinoft_native_m2dr": "DINO-FT native (M2_dr, ViT-B)",
    "dinoft_rt_m5s": "DINO-FT RT (M5-S, ViT-S) -- deployed",
    "dinoft_native_m5s": "DINO-FT native (M5-S, ViT-S)",
    "dinoft_rt_m2dr": "DINO-FT RT (M2_dr, ViT-B)",
    "dinoft_rt_m5s_nosnip": "DINO-FT RT (M5-S), no snipper",
    "dinoft_rt_m5s_nosnip_noviz": "DINO-FT RT (M5-S), lean: no snipper, no dashboard",
    "dinoft_native_m2dr_nosnip_noviz": "DINO-FT native (M2_dr), lean",
    "dinoft_rt_m5s_noviz": "DINO-FT RT (M5-S), snipper, no dashboard",
}


def f(v, nd=2, pct=False):
    if v is None:
        return "-"
    if pct:
        return f"{100*v:.{nd}f}%"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/usrp_spectrograms/rt_stress")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "summary.md"))
    ap.add_argument("--runs", nargs="*", default=None, help="run names in display order (default: all, sorted)")
    args = ap.parse_args()
    root = Path(args.root)
    runs = args.runs or sorted(p.name for p in root.iterdir() if (p / "metrics.json").exists())
    speed_rows, acc_rows = [], []
    for run in runs:
        m = json.loads((root / run / "metrics.json").read_text())
        h = m["headline"]
        man = m.get("manifest", {})
        speed_rows.append({
            "run": run, "leg": m.get("leg"), "rate_msps": m["rate_hz"] / 1e6, "loops": m["loops"],
            "frame_period_ms": h["frame_period_ms"],
            "frames_emitted": h["frames_emitted"], "frames_expected": h["frames_expected"],
            "frames_frac": h["frames_emitted_frac"],
            "det_p50": h["detector_ms_p50"], "det_p95": h["detector_ms_p95"], "det_max": h["detector_ms_max"],
            "cadence_p50": (man.get("cadence_ms") or {}).get("p50"),
            "fft_lat_mean": h["fft_ingress_ms_mean"], "fft_lat_max": h["fft_ingress_ms_max"],
            "pkt_loss": h["packet_loss_frac"], "missed": h["dpdk_missed_packets"],
            "partial": h["chdr_partial_batch_drops"], "backlog": h["chdr_backlog_events"],
            "fell_behind": h["fell_behind"], "gpu_mean": h["gpu_util_mean"], "gpu_max": h["gpu_util_max"],
            "achieved_pps": h["achieved_pps"],
            "pool_drain": h.get("pool_drain_frac"), "pool_exhausted": h.get("pool_exhausted"),
            "sustained": h.get("sustained_realtime"),
        })
        sj = root / run / "eval" / "summary.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            for side in ("loopback", "offline"):
                a = s[side]
                acc_rows.append({
                    "run": run, "leg": s["leg"], "side": side,
                    "frames": f"{a['frames_with_mask']}/{a['frames_total']}",
                    "det01": a["region_det_rate@0.1"], "det05": a["region_det_rate@0.5"],
                    "det05_eff": a["region_det_rate_effective@0.5"],
                    "prec": a["pixel_precision"], "rec": a["pixel_recall"], "f1": a["pixel_f1"],
                    "iou": a["pixel_iou"], "fp_area": a["fp_area_fraction"],
                    "nbfm": (a["per_class"].get("Narrowband_FM") or {}).get("det@0.5"),
                    "bt": (a["per_class"].get("Bluetooth") or {}).get("det@0.5"),
                    "zc": (a["per_class"].get("ZC") or {}).get("det@0.5"),
                    "lt2mhz": (a["per_bandwidth"].get("<2MHz") or {}).get("det@0.5"),
                    "agree_iou": (s.get("online_vs_offline_iou") or {}).get("median"),
                    "align_k": s.get("alignment_k"),
                    "valid": s.get("gt_scoring_valid", True),
                })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    L = []
    L.append("## Speed / real-time health (loopback legs)")
    L.append("")
    L.append("Sustained = no packet lost AND the NIC buffer pool not draining (drain < 1% of arrivals); a finite "
             "replay can end loss-free while the pool drains, i.e. while the pipeline is slower than real time.")
    L.append("")
    L.append("| run | leg | rate MSps | loops | budget ms/frame | frames emitted/expected | det ms p50 / p95 / max | emit cadence p50 ms | CHDR->FFT lat mean / max ms | pkt loss | NIC pool drain (% of arrivals) | sustained | backlog / fell-behind | GPU mean / max |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in speed_rows:
        L.append(f"| {r['run']} | {LEG_LABEL.get(r['leg'], r['leg'])} | {r['rate_msps']:.2f} | {r['loops']} | {f(r['frame_period_ms'],1)} | "
                 f"{f(r['frames_emitted'])}/{f(r['frames_expected'])} ({f(r['frames_frac'],1,True)}) | "
                 f"{f(r['det_p50'],1)} / {f(r['det_p95'],1)} / {f(r['det_max'],1)} | {f(r['cadence_p50'],1)} | "
                 f"{f(r['fft_lat_mean'],1)} / {f(r['fft_lat_max'],1)} | {f(r['pkt_loss'],3,True)} | "
                 f"{'exhausted' if r['pool_exhausted'] else f(r['pool_drain'],1,True)} | {'yes' if r['sustained'] else 'NO'} | "
                 f"{f(r['backlog'])} / {f(r['fell_behind'])} | {f(r['gpu_mean'],0)}% / {f(r['gpu_max'],0)}% |")
    if acc_rows:
        L += ["", "## Accuracy vs SigMF ground truth (region = annotation box; det@t = box coverage >= t; "
              "effective = unprocessed frames count as misses; pixel metrics on processed frames)", "",
              "| run | leg | side | frames w/ mask | det@0.1 | det@0.5 | effective det@0.5 | px prec | px recall | px F1 | px IoU | FP area | NBFM det | Bluetooth det | ZC det | <2 MHz det | online==offline IoU (k) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in acc_rows:
            if r["side"] == "loopback" and not r["valid"]:
                L.append(f"| {r['run']} | {LEG_LABEL.get(r['leg'], r['leg'])} | loopback | {r['frames']} | "
                         f"n/a: frame alignment lost after packet drops (online-vs-offline IoU {f(r['agree_iou'],3)}) | | | | | | | | | | | | - |")
                continue
            L.append(f"| {r['run']} | {LEG_LABEL.get(r['leg'], r['leg'])} | {r['side']} | {r['frames']} | {f(r['det01'],3)} | {f(r['det05'],3)} | "
                     f"{f(r['det05_eff'],3)} | {f(r['prec'],3)} | {f(r['rec'],3)} | {f(r['f1'],3)} | {f(r['iou'],3)} | {f(r['fp_area'],4)} | "
                     f"{f(r['nbfm'],2)} | {f(r['bt'],2)} | {f(r['zc'],2)} | {f(r['lt2mhz'],2)} | "
                     f"{f(r['agree_iou'],3)} ({r['align_k']:+d}) |" if r['side'] == 'loopback' and r['align_k'] is not None else
                     f"| {r['run']} | {LEG_LABEL.get(r['leg'], r['leg'])} | {r['side']} | {r['frames']} | {f(r['det01'],3)} | {f(r['det05'],3)} | "
                     f"{f(r['det05_eff'],3)} | {f(r['prec'],3)} | {f(r['rec'],3)} | {f(r['f1'],3)} | {f(r['iou'],3)} | {f(r['fp_area'],4)} | "
                     f"{f(r['nbfm'],2)} | {f(r['bt'],2)} | {f(r['zc'],2)} | {f(r['lt2mhz'],2)} | - |")
    out.write_text("\n".join(L) + "\n")
    with open(out.with_suffix(".speed.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(speed_rows[0].keys())); w.writeheader(); w.writerows(speed_rows)
    if acc_rows:
        with open(out.with_suffix(".accuracy.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(acc_rows[0].keys())); w.writeheader(); w.writerows(acc_rows)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

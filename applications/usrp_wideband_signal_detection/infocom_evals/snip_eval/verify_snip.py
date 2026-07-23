#!/usr/bin/env python3
"""Verify + summarize the REAL signal_snipper output (meta-only, from run_snip_all.sh with
write_iq=false). Reads every <SNIP_OUT>/<mode>/<detector>/<stem>/snippets/*.sigmf-meta, computes the
true stored footprint from the metas (no .sigmf-data needed), checks that decimation actually
happened, and compares the frequency-mode footprint to the analytic ds_cache proxy.

Footprint per snippet = sum(annotation core:sample_count) * BYTES_PER_SAMPLE (the decimated, stored
sample count). Rate check uses wfgt:snippet_sample_rate.

Usage:
  python3 verify_snip.py [--snip-out /tmp/usrp_spectrograms/snipped]
                         [--captures-dir /home/bqn82/captures]
                         [--cache <repo>/applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/ds_cache.csv]
                         [--out real_snip_metrics.csv]
"""
from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path

RATE_HZ, BYTES_PER_SAMPLE, SEC_PER_HR = 245.76e6, 8, 3600.0
SAVE_ALL_TB_HR = RATE_HZ * BYTES_PER_SAMPLE * SEC_PER_HR / 1e12   # 7.078


def atten(stem):
    m = re.search(r"dB_(\d+)", stem)
    return int(m.group(1)) if m else None


def capture_sec(captures_dir: Path, stem: str) -> float:
    f = captures_dir / f"{stem}.sigmf-data"
    return f.stat().st_size / (BYTES_PER_SAMPLE * RATE_HZ) if f.exists() else float("nan")


def summarize_run(snip_dir: Path):
    """snip_dir = <mode>/<det>/<stem>; read snippets/*.sigmf-meta -> footprint + rate stats."""
    metas = sorted((snip_dir / "snippets").glob("*.sigmf-meta"))
    stored_samples = 0      # decimated (what the snipper actually stores)
    orig_samples = 0        # full-rate span (what a full-band time-slice would store)
    rates, decims = [], []
    n_ann = 0
    for mp in metas:
        try:
            d = json.load(open(mp))
        except Exception:
            continue
        anns = d.get("annotations", [])
        n_ann += len(anns)
        # Recording payload = extent of its annotations. In per_signal every annotation shares
        # core:sample_count (= whole file) so DON'T sum; in pack/container they tile the file, so
        # max(start+count) is the total either way = the (deleted) .sigmf-data sample count.
        payload = max((int(a.get("core:sample_start", 0)) + int(a.get("core:sample_count", 0))
                       for a in anns), default=0)
        orig = max((int(a.get("wfgt:orig_sample_count", 0)) for a in anns), default=0)
        stored_samples += payload
        orig_samples += orig
        if anns:
            r = anns[0].get("wfgt:snippet_sample_rate")
            if r:
                rates.append(float(r))
            decims.append(float(anns[0].get("wfgt:decimation_factor",
                                            (orig / payload) if payload else 1.0)))
    return dict(n_snippets=len(metas), n_annotations=n_ann,
                stored_bytes=stored_samples * BYTES_PER_SAMPLE,
                orig_bytes=orig_samples * BYTES_PER_SAMPLE, rates=rates, decims=decims)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snip-out", type=Path, default=Path("/tmp/usrp_spectrograms/snipped"))
    ap.add_argument("--captures-dir", type=Path, default=Path("/home/bqn82/captures"))
    ap.add_argument("--cache", type=Path,
                    default=Path("/home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/ds_cache.csv"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    # analytic proxy (per detector, attenuation) for comparison, if present
    proxy = {}
    if a.cache.exists():
        for r in csv.DictReader(open(a.cache)):
            try:
                proxy[(r["detector"], int(float(r["attenuation_db"])))] = float(r["resample_meas_TB_hr"])
            except Exception:
                pass

    modes = sorted(p.name for p in a.snip_out.iterdir() if p.is_dir() and not p.name.startswith("_")
                   and p.name not in ("annotations",)) if a.snip_out.exists() else []
    if not modes:
        print(f"no mode dirs under {a.snip_out} (run run_snip_all.sh first)")
        return 1

    rows = []
    for mode in modes:
        print(f"\n================  MODE = {mode}  ================")
        print(f"{'detector':18s} {'stem':18s} {'snips':>6s} {'decim TB/hr':>11s} {'full(ovlp)':>10s} "
              f"{'decim×':>7s} {'reduction':>9s} {'%full':>6s} {'proxy TB/hr':>11s}")
        for detdir in sorted((a.snip_out / mode).glob("*/")):
            det = detdir.name
            for stemdir in sorted(detdir.glob("*/")):
                stem = stemdir.name
                if not (stemdir / "snippets").is_dir():
                    continue
                s = summarize_run(stemdir)
                sec = capture_sec(a.captures_dir, stem)
                _ok = sec == sec and sec > 0
                tbhr = s["stored_bytes"] / sec * SEC_PER_HR / 1e12 if _ok else float("nan")           # decimated (stored)
                full_tbhr = s["orig_bytes"] / sec * SEC_PER_HR / 1e12 if _ok else float("nan")         # full-rate (no decim)
                red = SAVE_ALL_TB_HR / tbhr if tbhr and tbhr == tbhr and tbhr > 0 else float("nan")
                dec = s["decims"]; decim_mean = sum(dec) / len(dec) if dec else float("nan")
                rts = s["rates"]
                if rts:
                    rmin, rmax = min(rts) / 1e6, max(rts) / 1e6
                    rmean = sum(rts) / len(rts) / 1e6
                    pct_full = 100.0 * sum(1 for r in rts if abs(r - RATE_HZ) < 1) / len(rts)
                else:
                    rmin = rmean = rmax = pct_full = float("nan")
                px = proxy.get((det, atten(stem)))
                print(f"{det:18s} {stem:18s} {s['n_snippets']:6d} {tbhr:11.3f} {full_tbhr:10.3f} "
                      f"{decim_mean:6.1f}x {red:8.1f}x {pct_full:5.0f} "
                      f"{(f'{px:.3f}' if px is not None else '—'):>11s}")
                rows.append(dict(mode=mode, detector=det, file_stem=stem, attenuation_db=atten(stem),
                                 n_snippets=s["n_snippets"],
                                 decimated_TB_per_hour=round(tbhr, 4),
                                 fullrate_overlapping_TB_per_hour=round(full_tbhr, 4),  # sum of per-snippet full-rate spans; snippets overlap in time so this over-counts (can exceed save-all)
                                 mean_decimation_factor=round(decim_mean, 3),
                                 reduction_x=round(red, 2),
                                 rate_min_MHz=round(rmin, 2), rate_mean_MHz=round(rmean, 2),
                                 rate_max_MHz=round(rmax, 2), pct_full_rate=round(pct_full, 1),
                                 proxy_TB_per_hour=(round(px, 4) if px is not None else "")))
        # decimation sanity per mode
        fr = [r for r in rows if r["mode"] == mode and r["pct_full_rate"] == r["pct_full_rate"]]
        if fr:
            avg_full = sum(r["pct_full_rate"] for r in fr) / len(fr)
            note = ("DECIMATION ACTIVE (rates < full fs)" if mode == "frequency" and avg_full < 90
                    else "all full-rate (expected for time_only)" if mode == "time_only"
                    else "WARNING: frequency mode but most snippets are full-rate -> decimation NOT firing")
            print(f"  [{mode}] mean %full-rate = {avg_full:.0f}%  -> {note}")

    if rows:
        # Default to writing next to this script (repo, always writable) rather than SNIP_OUT, which
        # the sudo snip run creates as root.
        out = a.out or (Path(__file__).resolve().parent / "real_snip_metrics.csv")
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out}")
        print(f"save-all reference = {SAVE_ALL_TB_HR:.3f} TB/hr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

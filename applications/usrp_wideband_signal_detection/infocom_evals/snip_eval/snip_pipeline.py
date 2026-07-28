#!/usr/bin/env python3
"""End-to-end configurable snip-eval pipeline.

One YAML config selects the detector, the set of waveform captures, and the snipper selectivity
(frequency / time / area); the pipeline then runs, per capture, in sequence:

  1. MASKS   — run the chosen detector offline over the capture (real container binary) to
               produce per-frame detection masks (mask_arrays/*.npy).
  2. SNIP    — replay those masks through mask_replay_detector -> signal_snipper ->
               sigmf_file_sink with the configured selectivity, writing one SigMF recording
               (.sigmf-data + .sigmf-meta) per snipped signal (write_iq: true).
  3. SOFT-LABEL META — a copy of the ORIGINAL capture's .sigmf-meta with new annotations
               appended, one per snipped detection (absolute RF edges per the SigMF spec, plus
               wfgt:*_offset_hz baseband-offset copies and provenance/soft-label fields).
  4. EVAL    — per-capture metrics row (snippets, stored bytes -> GB/hr, reduction vs save-all,
               mask coverage) -> pipeline_metrics.csv + pipeline_summary.png.

Stages are resumable per capture (marker files); re-running the pipeline skips finished work.

Usage:
  ~/miniforge3/envs/dinov3/bin/python snip_pipeline.py <pipeline_config.yaml>

See snip_pipeline_demo.yaml for a complete example config.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

SE = Path(__file__).resolve().parent
APP_DIR = SE.parents[1]
BASE_SNIP_CONFIG = APP_DIR / "config_mask_replay_snip_single_channel.yaml"
DRIVER = APP_DIR / "run_cuda_dino_offline_file.py"
BYTES_PER_SAMPLE = 8


# ----------------------------------------------------------------------------------------------
def load_pipeline_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    caps = [Path(c).expanduser() for c in cfg.get("captures", [])]
    if cfg.get("captures_dir"):
        caps += sorted(Path(cfg["captures_dir"]).expanduser().glob(cfg.get("captures_glob", "*.sigmf-data")))
    caps = [c for c in caps if c.suffix == ".sigmf-data" or c.name.endswith(".sigmf-data")]
    if not caps:
        sys.exit("pipeline config selects no captures (set `captures:` and/or `captures_dir:`)")
    cfg["_captures"] = caps
    cfg.setdefault("detector", "coherent_power")
    cfg.setdefault("write_iq", True)
    cfg.setdefault("snipper", {})
    cfg.setdefault("output_root", "/tmp/usrp_spectrograms/snip_pipeline")
    return cfg


def make_snip_config(cfg: dict, out_dir: Path) -> Path:
    """Patch the base mask-replay config's signal_snipper / sigmf_file_sink blocks with the
    pipeline's selectivity settings (text-level, block-aware; preserves everything else)."""
    sn = cfg["snipper"]
    overrides = {
        "signal_snipper": {
            "mode": f'"{sn.get("mode", "frequency")}"',
            "min_mask_bandwidth_hz": float(sn.get("min_mask_bandwidth_hz", 0.0)),
            "min_bandwidth_hz": float(sn.get("min_bandwidth_hz", 0.0)),
            "min_duration_s": float(sn.get("min_duration_s", 0.0)),
            "min_box_pixels": int(sn.get("min_box_pixels", 256)),
        },
        "sigmf_file_sink": {"write_iq": "true" if cfg["write_iq"] else "false"},
    }
    lines = BASE_SNIP_CONFIG.read_text().splitlines(keepends=True)
    block = None
    out = []
    for line in lines:
        m = re.match(r"^(\w[\w_]*):", line)
        if m:
            block = m.group(1)
        if block in overrides:
            km = re.match(r"^(\s+)([\w_]+):", line)
            if km and km.group(2) in overrides[block]:
                line = f"{km.group(1)}{km.group(2)}: {overrides[block][km.group(2)]}\n"
        out.append(line)
    gen = out_dir / "config_snip_pipeline_generated.yaml"
    gen.write_text("".join(out))
    return gen


def run_driver(cfg: dict, args: list[str]) -> int:
    env = dict(os.environ)
    if cfg.get("container_name"):
        env["CONTAINER_NAME"] = cfg["container_name"]
    if cfg.get("captures_root"):
        env["HOST_CAPTURES_ROOT"] = str(Path(cfg["captures_root"]).expanduser())
    cmd = [sys.executable, str(DRIVER)] + args + ["--captures-mounted", "--no-tensors"]
    print("  $", " ".join(cmd[-8:]))
    return subprocess.run(cmd, env=env).returncode


# ----------------------------------------------------------------------------------------------
def stage_masks(cfg: dict, cap: Path, mask_root: Path) -> Path:
    mask_dir = mask_root / "mask_arrays"
    marker = mask_root / ".masks_complete"
    if marker.exists() and any(mask_dir.glob("*.npy")):
        print(f"  [masks] done ({len(list(mask_dir.glob('*.npy')))} frames) — skip")
        return mask_dir
    extra = ["--config", str(Path(cfg["detector_config"]).expanduser())] if cfg.get("detector_config") else []
    rc = run_driver(cfg, [str(cap), "--detector", cfg["detector"], "--output-root", str(mask_root)] + extra)
    if rc != 0 or not any(mask_dir.glob("*.npy")):
        sys.exit(f"mask generation failed for {cap.name} (rc={rc}, masks at {mask_dir})")
    marker.touch()
    return mask_dir


def stage_snip(cfg: dict, cap: Path, mask_dir: Path, snip_root: Path, snip_cfg: Path) -> Path:
    marker = snip_root / ".snip_complete"
    if marker.exists():
        print("  [snip] done — skip")
        return snip_root / "snippets"
    rc = run_driver(cfg, [str(cap), "--detector", "mask_replay", "--config", str(snip_cfg),
                          "--mask-dir", str(mask_dir), "--output-root", str(snip_root),
                          "--snippets-only"])
    if rc != 0:
        sys.exit(f"mask-replay snip failed for {cap.name} (rc={rc})")
    marker.touch()
    return snip_root / "snippets"


def stage_soft_label(cfg: dict, cap: Path, snippets_dir: Path, out_dir: Path) -> tuple[Path, list[dict]]:
    """Original capture meta + one appended annotation per snipped detection."""
    orig_meta_path = Path(str(cap)[: -len(".sigmf-data")] + ".sigmf-meta")
    meta = json.loads(orig_meta_path.read_text())
    center = float(meta.get("captures", [{}])[0].get("core:frequency", 0.0))
    det_anns = []
    for mp in sorted(snippets_dir.glob("*.sigmf-meta")):
        snip = json.loads(mp.read_text())
        for a in snip.get("annotations", []):
            lo, hi = float(a.get("core:freq_lower_edge", 0)), float(a.get("core:freq_upper_edge", 0))
            det_anns.append({
                "core:sample_start": int(a.get("wfgt:orig_sample_start", 0)),
                "core:sample_count": int(a.get("wfgt:orig_sample_count", 0)),
                "core:freq_lower_edge": lo,
                "core:freq_upper_edge": hi,
                "core:label": f"{cfg['detector']}_detection",
                "core:generator": "snip_pipeline.py",
                "wfgt:soft_label": True,
                "wfgt:detector": cfg["detector"],
                "wfgt:freq_lower_offset_hz": lo - center,
                "wfgt:freq_upper_offset_hz": hi - center,
                "wfgt:frame_number": int(a.get("wfgt:frame_number", 0)),
                "wfgt:snippet_recording": mp.stem,
                "wfgt:snippet_sample_rate": a.get("wfgt:snippet_sample_rate"),
            })
    det_anns.sort(key=lambda a: (a["core:sample_start"], a["core:freq_lower_edge"]))
    out = dict(meta)
    out["annotations"] = list(meta.get("annotations", [])) + det_anns
    g = out.setdefault("global", {})
    g["wfgt:soft_label_source"] = (f"snip_pipeline detector={cfg['detector']} "
                                   f"snipper={cfg['snipper']} (appended to original annotations)")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / (cap.name[: -len(".sigmf-data")] + ".sigmf-meta")
    dst.write_text(json.dumps(out, indent=2))
    return dst, det_anns


def stage_eval(cap: Path, mask_dir: Path, snippets_dir: Path, det_anns: list[dict]) -> dict:
    sample_rate = 245.76e6
    try:
        m = json.loads(Path(str(cap)[: -len(".sigmf-data")] + ".sigmf-meta").read_text())
        sample_rate = float(m["global"].get("core:sample_rate", sample_rate))
    except Exception:
        pass
    cap_sec = cap.stat().st_size / (BYTES_PER_SAMPLE * sample_rate)
    stored = 0
    n_snips = 0
    for mp in snippets_dir.glob("*.sigmf-meta"):
        d = json.loads(mp.read_text())
        anns = d.get("annotations", [])
        stored += max((int(a.get("core:sample_start", 0)) + int(a.get("core:sample_count", 0))
                       for a in anns), default=0)
        n_snips += 1
    data_bytes = sum(p.stat().st_size for p in snippets_dir.glob("*.sigmf-data"))
    cov_sum, cov_n = 0.0, 0
    for f in sorted(mask_dir.glob("*.npy")):
        try:
            cov_sum += float(np.load(f, mmap_mode="r").mean())
            cov_n += 1
        except Exception:
            pass
    save_all_gb_hr = sample_rate * BYTES_PER_SAMPLE * 3600 / 1e9
    gb_hr = stored * BYTES_PER_SAMPLE / cap_sec * 3600 / 1e9 if cap_sec > 0 else float("nan")
    return dict(capture=cap.name, seconds=round(cap_sec, 2), n_frames=cov_n,
                mask_coverage_pct=round(100 * cov_sum / max(cov_n, 1), 3),
                n_snippets=n_snips, n_detections=len(det_anns),
                stored_samples=stored, stored_iq_bytes_on_disk=data_bytes,
                stored_GB_per_hour=round(gb_hr, 3),
                save_all_GB_per_hour=round(save_all_gb_hr, 1),
                reduction_x=round(save_all_gb_hr / gb_hr, 1) if gb_hr and gb_hr > 0 else float("inf"))


def summary_figure(rows: list[dict], out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "savefig.bbox": "tight",
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    names = [r["capture"].replace(".sigmf-data", "") for r in rows]
    stored = [max(r["stored_GB_per_hour"], 1e-2) for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(1.2 + 2.2 * len(rows), 5.0))
    ax.bar(x, stored, 0.55, color="#4a3aa7", label="stored (snipped)")
    ax.axhline(rows[0]["save_all_GB_per_hour"], color="#0b0b0b", lw=2.6,
               label=f"naive save-all ({rows[0]['save_all_GB_per_hour']:g} GB/hr)")
    for i, r in enumerate(rows):
        ax.text(i, stored[i] * 1.15,
                f"{r['stored_GB_per_hour']:g} GB/hr\n×{r['reduction_x']:g} less\n"
                f"{r['n_snippets']} snips · {r['mask_coverage_pct']:g}% mask",
                ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel("stored data (GB / hour, log)")
    ax.set_title("snip_pipeline: real snipped footprint per capture")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    print("wrote", out_png)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path, help="pipeline YAML (see snip_pipeline_demo.yaml)")
    a = ap.parse_args()
    cfg = load_pipeline_config(a.config)
    root = Path(cfg["output_root"]).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    snip_cfg = make_snip_config(cfg, root)
    print(f"pipeline: detector={cfg['detector']}  captures={len(cfg['_captures'])}  "
          f"snipper={cfg['snipper']}  write_iq={cfg['write_iq']}\n  output_root={root}")

    rows = []
    for cap in cfg["_captures"]:
        stem = cap.name[: -len(".sigmf-data")]
        print(f"\n=== {stem} ===")
        mask_dir = stage_masks(cfg, cap, root / "masks" / stem)
        snippets = stage_snip(cfg, cap, mask_dir, root / "snip" / stem, snip_cfg)
        soft_meta, det_anns = stage_soft_label(cfg, cap, snippets, root / "soft_labels")
        row = stage_eval(cap, mask_dir, snippets, det_anns)
        row["soft_label_meta"] = str(soft_meta)
        rows.append(row)
        print(f"  [eval] {row['n_snippets']} snippets, {row['stored_GB_per_hour']} GB/hr "
              f"(×{row['reduction_x']} vs save-all), mask coverage {row['mask_coverage_pct']}%, "
              f"soft-label meta -> {soft_meta}")

    csv_path = root / "pipeline_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("\nwrote", csv_path)
    summary_figure(rows, root / "pipeline_summary.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

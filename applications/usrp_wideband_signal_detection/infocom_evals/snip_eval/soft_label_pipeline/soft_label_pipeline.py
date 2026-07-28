#!/usr/bin/env python3
"""Soft-labelling pipeline: a folder of SigMF waveforms + a detector -> masks -> detections.

For every ``*.sigmf-data`` in a chosen folder, run a chosen detector, generate its per-frame masks,
and emit soft labels:
  * ``<stem>_snipped.sigmf-meta``  -- SigMF annotations = the detector's ``detected_waveform`` boxes on
                                      the ORIGINAL capture timeline (via snip_annotations.py).  [always]
  * ``snippets/<...>.sigmf-{data,meta}`` -- the actual frequency+time-decimated IQ per detection
                                      (mix->lowpass->decimate), one recording each.              [--outputs iq|both]

Detector tiers (all 8 of the Holohub USRP comparison set):
  trained   : coherent_power, cuda_dino        -> container binary (run_cuda_dino_offline_file.py)
  baselines : 3dB_power, blob_detection        -> run_baseline_offline.py (mirror a trained run's frames)
  ml        : yolo, dino_finetuned,            -> run_ml_detectors_offline.py (GPU + weights; mirror ref)
              dino_finetuned_m1, yolo26s
Baselines/ML do NOT process raw IQ alone -- they reuse a trained detector's frame grid + GT +
frame_manifest.csv, so this pipeline first seeds that with the ref detector (default cuda_dino) when
the chosen detector is a baseline/ML one. See README.md for setup + prerequisites.

Usage:
  python3 soft_label_pipeline.py --waveform-dir DIR --detector coherent_power --outputs both
See README.md for the full option list and the container setup.
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent.parent.parent                       # .../usrp_wideband_signal_detection
DRIVER = APP_DIR / "run_cuda_dino_offline_file.py"
CMP_DIR = APP_DIR / "infocom_evals" / "baseline_comparisons"
BASELINE_RUNNER = CMP_DIR / "run_baseline_offline.py"
ML_RUNNER = CMP_DIR / "run_ml_detectors_offline.py"
CMP_CONFIG = CMP_DIR / "comparison_config.yaml"
BASE_SNIP_CONFIG = APP_DIR / "config_mask_replay_snip_single_channel.yaml"
SNIP_ANNOTATIONS = HERE / "snip_annotations.py"
MATERIALIZE = HERE / "materialize_npy.py"

TRAINED = ["coherent_power", "cuda_dino"]
BASELINES = ["3dB_power", "blob_detection"]
ML_FALLBACK = ["yolo", "dino_finetuned", "dino_finetuned_m1", "yolo26s"]
_ML: list = []          # ML detector names, loaded from --config in main() (new models auto-enable)


def ml_detectors(config_path=CMP_CONFIG) -> list:
    """ML detector names come from comparison_config.yaml's `ml_detectors:` block, so adding a new
    model there (a `kind` + checkpoint) auto-enables it here with no edit to this script."""
    try:
        import yaml
        block = yaml.safe_load(Path(config_path).read_text())
        block = block.get("comparison_eval", block)
        return list((block.get("ml_detectors") or {}).keys()) or ML_FALLBACK
    except Exception:
        return ML_FALLBACK


def tier(det: str) -> str:
    return ("trained" if det in TRAINED else "baseline" if det in BASELINES
            else "ml" if det in (_ML or ML_FALLBACK) else "unknown")


def run(cmd, env=None, cwd=None) -> int:
    print("  $", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], env=env, cwd=str(cwd) if cwd else None).returncode


def driver_env(args) -> dict:
    env = dict(os.environ)
    env["CONTAINER_NAME"] = args.container_name
    env["HOST_CAPTURES_ROOT"] = str(Path(args.captures_root).expanduser().resolve())
    return env


# ---- mask generation --------------------------------------------------------------------------
def gen_trained(det, cap, run_dir, args) -> None:
    """Container run: masks + frame_manifest + GT into run_dir. Idempotent."""
    if (run_dir / "frame_manifest.csv").exists() and any((run_dir / "mask_arrays").glob("*")):
        print(f"  [masks] {det}/{cap.stem} present -> skip"); return
    extra = ["--config", str(Path(args.detector_config).expanduser())] if args.detector_config else []
    rc = run([sys.executable, DRIVER, str(cap), "--detector", det, "--output-root", str(run_dir),
              "--captures-mounted", "--no-tensors"] + extra, env=driver_env(args))
    if rc != 0 or not any((run_dir / "mask_arrays").glob("*")):
        sys.exit(f"mask generation failed for {det}/{cap.name} (rc={rc})")


def ensure_foundation(captures, batch_root, args) -> None:
    """Baselines/ML mirror the ref detector's frames+GT+manifest -> seed it first."""
    ref = args.ref_detector
    print(f"[foundation] seeding ref detector '{ref}' (frames+GT+manifest) for the {tier(args.detector)} detector")
    for cap in captures:
        gen_trained(ref, cap, batch_root / ref / cap.stem, args)


def gen_baseline(det, captures, batch_root, args) -> None:
    ensure_foundation(captures, batch_root, args)
    # run_baseline_offline reads its `detectors:` params from a config with source_batch_root/source_detector.
    import tempfile, yaml
    cmp = yaml.safe_load(CMP_CONFIG.read_text())["comparison_eval"]
    params = {det: cmp["baselines"][det]}
    tmp = Path(tempfile.mkdtemp(prefix="soft_label_baseline_")) / "cfg.yaml"
    tmp.write_text(yaml.safe_dump({"source_batch_root": str(batch_root), "out_batch_root": None,
                                   "source_detector": args.ref_detector, "detectors": params}))
    rc = run([sys.executable, BASELINE_RUNNER, "--config", tmp, "--source-batch-root", batch_root,
              "--detectors", det, "--captures-dir", args.waveform_dir])
    if rc != 0:
        sys.exit(f"baseline detector {det} failed (rc={rc})")


def gen_ml(det, captures, batch_root, args) -> None:
    ensure_foundation(captures, batch_root, args)
    cmd = [sys.executable, ML_RUNNER, "--config", args.config, "--batch-root", batch_root,
           "--ref-detector", args.ref_detector, "--detectors", det, "--captures-dir", args.waveform_dir]
    if args.dinov3_repo:
        cmd += ["--dinov3-repo", args.dinov3_repo]
    if args.stems_only:
        cmd += ["--stems", *[c.stem for c in captures]]
    rc = run(cmd)
    if rc != 0:
        sys.exit(f"ML detector {det} failed (rc={rc}). Check GPU + model weights (see README).")


# ---- snipped-meta (annotations) + IQ ----------------------------------------------------------
def make_snip_config(args, out_dir: Path) -> Path:
    """Patch config_mask_replay_snip_single_channel.yaml's signal_snipper / sigmf_file_sink blocks."""
    overrides = {
        "signal_snipper": {"mode": f'"{args.mode}"',
                           "min_mask_bandwidth_hz": float(args.min_mask_bandwidth_hz),
                           "min_bandwidth_hz": float(args.min_bandwidth_hz),
                           "min_duration_s": float(args.min_duration_s),
                           "min_box_pixels": int(args.min_box_pixels)},
        "sigmf_file_sink": {"write_iq": "true"},          # this path always wants the IQ
    }
    out, block = [], None
    for line in BASE_SNIP_CONFIG.read_text().splitlines(keepends=True):
        m = re.match(r"^(\w[\w_]*):", line)
        if m:
            block = m.group(1)
        if block in overrides:
            km = re.match(r"^(\s+)([\w_]+):", line)
            if km and km.group(2) in overrides[block]:
                line = f"{km.group(1)}{km.group(2)}: {overrides[block][km.group(2)]}\n"
        out.append(line)
    gen = out_dir / "config_snip_generated.yaml"
    gen.write_text("".join(out))
    return gen


def snipped_meta(det, cap, run_dir, out_dir, args) -> None:
    rc = run([sys.executable, SNIP_ANNOTATIONS, "--run-dir", run_dir, "--captures-dir", args.waveform_dir,
              "--out-dir", out_dir, "--min-box-pixels", args.min_box_pixels,
              "--merge-gap-rows", args.merge_gap_rows, "--merge-gap-cols", args.merge_gap_cols])
    if rc != 0:
        sys.exit(f"snip_annotations failed for {det}/{cap.name} (rc={rc})")


def snip_iq(det, cap, run_dir, iq_root, snip_cfg, args) -> None:
    mask_dir = run_dir / "mask_arrays"
    if any(mask_dir.glob("*.packed.npz")):                # mask_replay needs .npy
        run([sys.executable, MATERIALIZE, str(run_dir)])
    rc = run([sys.executable, DRIVER, str(cap), "--detector", "mask_replay", "--config", str(snip_cfg),
              "--mask-dir", str(mask_dir), "--output-root", str(iq_root), "--captures-mounted", "--no-tensors"],
             env=driver_env(args))
    if rc != 0:
        sys.exit(f"IQ snip (mask_replay) failed for {det}/{cap.name} (rc={rc})")


# ---- main -------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waveform-dir", required=True, help="folder of *.sigmf-data (+ .sigmf-meta) to process")
    ap.add_argument("--detector", required=True,
                    help="detector to soft-label with. trained: " + ", ".join(TRAINED)
                         + " | baselines: " + ", ".join(BASELINES)
                         + " | ml (from --config ml_detectors): " + ", ".join(ml_detectors()))
    ap.add_argument("--outputs", choices=["snipped_meta", "iq", "both"], default="snipped_meta",
                    help="snipped_meta = <stem>_snipped.sigmf-meta only; iq = decimated IQ pairs only; both")
    ap.add_argument("--glob", default="*.sigmf-data", help="capture filter within --waveform-dir")
    ap.add_argument("--output-root", default="/tmp/usrp_spectrograms/soft_label_pipeline",
                    help="everything is written under here (masks/, snipped/, iq/); keep it off /home")
    ap.add_argument("--mask-root", default=None,
                    help="reuse pre-computed masks at <mask-root>/<detector>/<stem>/ (skip generation)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N captures")
    # snipper / snip_annotations selectivity
    ap.add_argument("--mode", choices=["frequency", "time_only"], default="frequency")
    ap.add_argument("--min-box-pixels", type=int, default=256)
    ap.add_argument("--min-bandwidth-hz", type=float, default=0.0)
    ap.add_argument("--min-duration-s", type=float, default=0.0)
    ap.add_argument("--min-mask-bandwidth-hz", type=float, default=0.0)
    ap.add_argument("--merge-gap-rows", type=int, default=16)
    ap.add_argument("--merge-gap-cols", type=int, default=80)
    # container / detector wiring
    ap.add_argument("--container-name", default=os.environ.get("CONTAINER_NAME", "usrp_x410_sig_det_bqn82"))
    ap.add_argument("--captures-root", default=None, help="host dir mapped into the container (default: --waveform-dir)")
    ap.add_argument("--detector-config", default=None, help="override the trained detector's base config")
    ap.add_argument("--ref-detector", default="cuda_dino", choices=TRAINED, help="frame/GT source for baselines+ML")
    ap.add_argument("--config", default=str(CMP_CONFIG), help="comparison_config.yaml (baseline/ML params + weights)")
    ap.add_argument("--dinov3-repo", default=None, help="dinov3 repo path for the fine-tuned DINO models")
    ap.add_argument("--stems-only", action="store_true", help="restrict the ML runner to just these captures")
    args = ap.parse_args()

    global _ML
    _ML = ml_detectors(args.config)
    valid = TRAINED + BASELINES + _ML
    if args.detector not in valid:
        sys.exit(f"unknown --detector '{args.detector}'. Valid: {', '.join(valid)}")

    wdir = Path(args.waveform_dir).expanduser().resolve()
    args.waveform_dir = str(wdir)
    if args.captures_root is None:
        args.captures_root = str(wdir)
    captures = sorted(p for p in wdir.glob(args.glob) if p.name.endswith(".sigmf-data"))
    captures = [c for c in captures if (Path(str(c)[:-len(".sigmf-data")] + ".sigmf-meta")).exists()]
    if args.limit:
        captures = captures[: args.limit]
    if not captures:
        sys.exit(f"no *.sigmf-data (+ .sigmf-meta) under {wdir} matching {args.glob}")

    root = Path(args.output_root).expanduser()
    batch_root = Path(args.mask_root).expanduser() if args.mask_root else (root / "masks")
    snipped_dir = root / "snipped"
    (root).mkdir(parents=True, exist_ok=True); snipped_dir.mkdir(parents=True, exist_ok=True)
    det = args.detector

    print(f"soft_label_pipeline: detector={det} ({tier(det)})  captures={len(captures)}  outputs={args.outputs}\n"
          f"  waveform_dir={wdir}\n  output_root={root}\n  masks={'reuse ' if args.mask_root else ''}{batch_root}")

    # 1. masks (unless reusing a mask-root)
    if not args.mask_root:
        if tier(det) == "trained":
            for cap in captures:
                gen_trained(det, cap, batch_root / det / cap.stem, args)
        elif tier(det) == "baseline":
            gen_baseline(det, captures, batch_root, args)
        else:
            gen_ml(det, captures, batch_root, args)

    # 2. per-capture soft labels
    snip_cfg = make_snip_config(args, root) if args.outputs in ("iq", "both") else None
    for cap in captures:
        run_dir = batch_root / det / cap.stem
        if not (run_dir / "mask_arrays").exists():
            sys.exit(f"masks missing for {det}/{cap.stem} at {run_dir} (check --mask-root layout)")
        print(f"\n=== {cap.stem} ===")
        if args.outputs in ("snipped_meta", "both"):
            snipped_meta(det, cap, run_dir, snipped_dir, args)
        if args.outputs in ("iq", "both"):
            snip_iq(det, cap, run_dir, root / "iq" / cap.stem, snip_cfg, args)

    print(f"\nDONE. snipped metas -> {snipped_dir}/<stem>_snipped.sigmf-meta"
          + (f"\n      decimated IQ  -> {root}/iq/<stem>/snippets/*.sigmf-{{data,meta}}" if args.outputs in ("iq", "both") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

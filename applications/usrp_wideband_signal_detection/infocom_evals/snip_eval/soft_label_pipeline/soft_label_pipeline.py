#!/usr/bin/env python3
"""Soft-labelling pipeline: recordings + a detector -> masks -> detections as SigMF annotations.

For every recording, run a chosen detector, generate its per-frame masks, and emit soft labels:
  * ``<stem>_softlabel.sigmf-meta`` (+ ``.sigmf-data``, a symlink to the samples or their converted copy) --
        SigMF annotations = the detector's ``detected_waveform`` boxes on the recording's own timeline
        (via snip_annotations.py), written under --out-dir mirroring the input tree.        [always]
  * ``snippets/<...>.sigmf-{data,meta}`` -- the actual frequency+time-decimated IQ per detection
        (mix->lowpass->decimate), one recording each; container detectors only.          [--outputs iq|both]

Detector tiers:
  torchscript : dino_finetuned_m12s, dino_finetuned_m13s,  -> dino_ts_masks.py: the released fine-tuned DINO
                dino_finetuned_m7s, dino_finetuned_ts         exports in plain PyTorch, NO container; any input
                (+ --ts-model <file>.ts)                      format (to_sigmf.py), files or whole directory trees
  trained     : coherent_power, cuda_dino        -> container binary (run_cuda_dino_offline_file.py)
  baselines   : 3dB_power, blob_detection        -> run_baseline_offline.py (mirror a trained run's frames)
  ml          : yolo, dino_finetuned,            -> run_ml_detectors_offline.py (GPU + weights; mirror ref)
                dino_finetuned_m1, yolo26s
Baselines/ML do NOT process raw IQ alone -- they reuse a trained detector's frame grid + GT +
frame_manifest.csv, so this pipeline first seeds that with the ref detector (default cuda_dino) when
the chosen detector is a baseline/ML one. Container tiers take a flat folder of cf32_le SigMF
(to_sigmf.py --cf32 converts anything else). See README.md for setup + prerequisites.

Usage:
  python3 soft_label_pipeline.py --input ~/captures --recursive --detector dino_finetuned_m12s --out-dir ~/labels
  python3 soft_label_pipeline.py --waveform-dir DIR --detector coherent_power --outputs both
See README.md for the full option list and the container setup.
"""
from __future__ import annotations
import argparse, csv, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import to_sigmf as tsg  # noqa: E402
import dino_ts_masks as dtm  # noqa: E402
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
    return ("torchscript" if dtm.is_ts_detector(det) else "trained" if det in TRAINED
            else "baseline" if det in BASELINES else "ml" if det in (_ML or ML_FALLBACK) else "unknown")


def run_key(f: Path, root: Path) -> str:
    """Mask run-dir name: the stem, prefixed by the sub-folders below the input root (unique for trees)."""
    rel = tsg.rel_dir(f, root)
    stem = tsg._stem_of(f)
    return "__".join(list(rel.parts) + [stem]) if rel.parts else stem


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


# ---- soft-label meta (annotations) + IQ ----------------------------------------------------------
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


def softlabel_meta(label_meta: Path, run_dir: Path, args) -> int:
    """Fill <stem>_softlabel.sigmf-meta (written by to_sigmf.py) with the run's detections. Returns their count."""
    import snip_annotations
    return snip_annotations.annotate(run_dir, label_meta, label_meta, args.min_box_pixels, args.merge_gap_rows,
                                     args.merge_gap_cols, quiet=True)


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
    io = ap.add_argument_group("inputs and outputs")
    io.add_argument("--input", nargs="+", default=None, metavar="PATH",
                    help="recordings to label: files and/or directories (torchscript detectors read any format "
                         "to_sigmf.py knows; container detectors a flat folder of cf32_le SigMF)")
    io.add_argument("--waveform-dir", default=None, help="a folder of recordings (same as --input DIR)")
    io.add_argument("--recursive", action="store_true", help="also label recordings in sub-directories")
    io.add_argument("--glob", default=None,
                    help="only file names matching this pattern (default: every supported file; container "
                         "detectors: *.sigmf-data)")
    io.add_argument("--limit", type=int, default=None, help="process only the first N recordings")
    io.add_argument("--out-dir", default=None,
                    help="where the <stem>_softlabel.sigmf-{meta,data} pairs go, mirroring the input tree "
                         "(default: <output-root>/softlabels)")
    io.add_argument("--output-root", default="/tmp/usrp_spectrograms/soft_label_pipeline",
                    help="working files: masks/, iq/, generated configs (keep it off /home)")
    io.add_argument("--mask-root", default=None,
                    help="reuse pre-computed masks at <mask-root>/<detector>/<stem>/ (skip generation)")
    ap.add_argument("--detector", required=True,
                    help="detector to soft-label with. torchscript (no container): "
                         + ", ".join(list(dtm.TS_DETECTORS) + [dtm.GENERIC])
                         + " | trained: " + ", ".join(TRAINED)
                         + " | baselines: " + ", ".join(BASELINES)
                         + " | ml (from --config ml_detectors): " + ", ".join(ml_detectors()))
    ap.add_argument("--outputs", choices=["softlabel_meta", "snipped_meta", "iq", "both"], default="softlabel_meta",
                    help="softlabel_meta = <stem>_softlabel.sigmf-meta only (snipped_meta: old name); "
                         "iq = decimated IQ pairs only; both (iq: container detectors)")
    # snipper / snip_annotations selectivity (defaults: the torchscript model's live snipper gates, else 256/16/80)
    ap.add_argument("--mode", choices=["frequency", "time_only"], default="frequency")
    ap.add_argument("--min-box-pixels", type=int, default=None)
    ap.add_argument("--min-bandwidth-hz", type=float, default=0.0)
    ap.add_argument("--min-duration-s", type=float, default=0.0)
    ap.add_argument("--min-mask-bandwidth-hz", type=float, default=0.0)
    ap.add_argument("--merge-gap-rows", type=int, default=None)
    ap.add_argument("--merge-gap-cols", type=int, default=None)
    # torchscript detectors
    t = ap.add_argument_group("torchscript detectors (no container)")
    t.add_argument("--ts-model", default=None, help="a dino_fine_tuning .ts export (required for dino_finetuned_ts; "
                                                    "overrides the model file of the named ones)")
    t.add_argument("--threshold", type=float, default=None, help="override the model's decision threshold")
    t.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    t.add_argument("--max-seconds", type=float, default=None, help="label at most this many seconds per recording")
    t.add_argument("--force", action="store_true", help="regenerate masks that already exist for this model")
    t.add_argument("--offline", action="store_true", help="never download missing release weights")
    t.add_argument("--no-frame-guard", action="store_true",
                   help="keep frames the live invalid-frame guard would empty (occupancy > max(min, k x baseline)); "
                        "the guard targets corrupted live-ingest frames, a clean recording can be legitimately dense")
    tsg.add_override_args(ap)
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
    det = args.detector
    valid = list(dtm.TS_DETECTORS) + [dtm.GENERIC] + TRAINED + BASELINES + _ML
    if det not in valid:
        sys.exit(f"unknown --detector '{det}'. Valid: {', '.join(valid)}")
    tr = tier(det)
    if args.outputs == "snipped_meta":
        args.outputs = "softlabel_meta"
    inputs = args.input or ([args.waveform_dir] if args.waveform_dir else None)
    if not inputs:
        ap.error("give --input (files and/or directories) or --waveform-dir")
    paths = [Path(p).expanduser().resolve() for p in inputs]

    ts_settings = None
    if tr == "torchscript":
        if args.outputs != "softlabel_meta":
            sys.exit("--outputs iq|both needs the container's IQ snipper, which is not wired for the torchscript "
                     "detectors; use --outputs softlabel_meta")
        ts_settings = dtm.load_settings(det, args.ts_model, args.threshold, args.offline)
        if args.no_frame_guard:
            ts_settings.guard = False
    gates = ts_settings.snipper if ts_settings else {}
    if args.min_box_pixels is None:
        args.min_box_pixels = int(gates.get("min_box_pixels", 256))
    if args.merge_gap_rows is None:
        args.merge_gap_rows = int(gates.get("merge_gap_rows", 16))
    if args.merge_gap_cols is None:
        args.merge_gap_cols = int(gates.get("merge_gap_cols", 80))

    if tr == "torchscript":
        found = tsg.find_inputs(paths, args.recursive, args.glob)
    else:
        if args.recursive or len(paths) != 1 or not paths[0].is_dir():
            sys.exit(f"the {tr} detectors run in the container and take ONE flat folder of cf32_le SigMF "
                     "(--waveform-dir DIR); for trees or other formats use a torchscript detector, or convert "
                     "first with to_sigmf.py --cf32")
        args.waveform_dir = str(paths[0])
        found = [(f, r) for f, r in tsg.find_inputs(paths, False, args.glob or "*.sigmf-data")
                 if f.name.endswith(".sigmf-meta")]
        if args.captures_root is None:
            args.captures_root = args.waveform_dir
    if args.limit:
        found = found[: args.limit]
    if not found:
        sys.exit(f"no recordings found under {', '.join(map(str, paths))}"
                 + (f" matching {args.glob}" if args.glob else ""))

    root = Path(args.output_root).expanduser()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "softlabels"
    batch_root = Path(args.mask_root).expanduser() if args.mask_root else (root / "masks")
    root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"soft_label_pipeline: detector={det} ({tr})  recordings={len(found)}  outputs={args.outputs}\n"
          f"  inputs={', '.join(map(str, paths))}{' (recursive)' if args.recursive else ''}\n"
          f"  out_dir={out_dir}\n  masks={'reuse ' if args.mask_root else ''}{batch_root}\n"
          f"  gates: min_box_pixels={args.min_box_pixels} merge_gap_rows={args.merge_gap_rows} "
          f"merge_gap_cols={args.merge_gap_cols}", flush=True)

    # 1. the <stem>_softlabel.sigmf-{meta,data} pair of every recording in out_dir (samples symlinked or converted)
    ov = tsg.overrides_from(args)
    items, index = [], []
    for f, r in found:
        stem = tsg._stem_of(f)
        try:
            conv = tsg.convert(f, out_dir / tsg.rel_dir(f, r), ov, out_stem=f"{stem}_softlabel")
        except (tsg.ConvertError, OSError, ValueError) as e:
            print(f"  SKIP {f}: {e}", flush=True)
            index.append({"source": str(f), "skipped": str(e)})
            continue
        if tr != "torchscript" and conv.source_dtype != "cf32_le":
            msg = f"{conv.source_dtype} samples: the container detectors read cf32_le (convert with to_sigmf.py --cf32)"
            print(f"  SKIP {f}: {msg}", flush=True)
            index.append({"source": str(f), "skipped": msg})
            continue
        items.append({"src": f, "key": run_key(f, r) if tr == "torchscript" else stem, "conv": conv})

    # 2. masks (unless reusing a mask-root)
    if tr == "torchscript":
        detector = dtm.TsDetector(ts_settings, args.device)
        print(f"  model {ts_settings.ts_path.name}, threshold {ts_settings.threshold}, device {detector.device}, "
              f"config {ts_settings.config.name}", flush=True)
        for it in items:
            run_dir = batch_root / det / it["key"]
            if args.mask_root and (run_dir / "frame_manifest.csv").exists():
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            it["info"] = detector.run(it["conv"].meta_path, run_dir, args.max_seconds, args.force)
    elif not args.mask_root:
        captures = [it["src"].with_name(it["key"] + ".sigmf-data") for it in items]
        if tr == "trained":
            for cap in captures:
                gen_trained(det, cap, batch_root / det / cap.stem, args)
        elif tr == "baseline":
            gen_baseline(det, captures, batch_root, args)
        else:
            gen_ml(det, captures, batch_root, args)

    # 3. per-recording soft labels
    snip_cfg = make_snip_config(args, root) if args.outputs in ("iq", "both") else None
    for it in items:
        run_dir = batch_root / det / it["key"]
        if not (run_dir / "mask_arrays").exists():
            sys.exit(f"masks missing for {det}/{it['key']} at {run_dir} (check --mask-root layout)")
        conv, info = it["conv"], it.get("info", {})
        n = softlabel_meta(conv.meta_path, run_dir, args) if args.outputs in ("softlabel_meta", "both") else ""
        if args.outputs in ("iq", "both"):
            cap = it["src"].with_name(it["key"] + ".sigmf-data")
            snip_iq(det, cap, run_dir, root / "iq" / cap.stem, snip_cfg, args)
        notes = conv.notes + info.get("notes", [])
        print(f"  {it['src'].name}: {conv.rate / 1e6:.4g} MS/s @ {conv.fc / 1e6:.6g} MHz, "
              f"{info.get('frames', '?')} frames, {n} detections -> {conv.meta_path}"
              + (" [masks reused]" if info.get("skipped") else ""), flush=True)
        for note in notes:
            print(f"      note: {note}")
        index.append({"source": str(it["src"]), "softlabel_meta": str(conv.meta_path), "detector": det,
                      "rate_hz": conv.rate, "fc_hz": conv.fc, "frames": info.get("frames", ""),
                      "guard_suppressed": info.get("guard_suppressed", ""), "detections": n,
                      "notes": " | ".join(notes)})

    fields = ["source", "softlabel_meta", "detector", "rate_hz", "fc_hz", "frames", "guard_suppressed", "detections",
              "notes", "skipped"]
    with open(out_dir / "softlabel_index.csv", "w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fields)
        w.writeheader()
        w.writerows(index)
    n_ok = len(items)
    print(f"\nDONE. {n_ok} recording(s) labelled, {len(index) - n_ok} skipped.\n"
          f"      soft labels   -> {out_dir}/<path>/<stem>_softlabel.sigmf-meta (index: {out_dir / 'softlabel_index.csv'})"
          + (f"\n      decimated IQ  -> {root}/iq/<stem>/snippets/*.sigmf-{{data,meta}}" if args.outputs in ("iq", "both") else ""))
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

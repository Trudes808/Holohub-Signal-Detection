#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 National Instruments Corporation
# SPDX-License-Identifier: Apache-2.0
"""Preprocess a directory of SigMF captures with one detector: find the regions of interest, cut
them out of the IQ with the signal snipper, and write a sibling dataset ``<dir>_preprocessed``.

    sudo python3 preprocess_dataset.py <dataset_dir> --detector coherent_power
    sudo python3 preprocess_dataset.py <dataset_dir> --detector cuda_dino_finetuned --mode time_only

The detector, snipper and SigMF sink are the SAME compiled operators the live app runs; this script
only orchestrates ``run_cuda_dino_offline_file.py`` (one container run per capture) and lays the
results out as a dataset:

    <dir>_preprocessed/
      manifest.csv                 one row per emitted region (all captures)
      preprocess_summary.json      detector/config/mode + per-capture counts and byte totals
      <capture_stem>/
        <capture_stem>.source.sigmf-meta   copy of the input metadata (provenance)
        snippets/*.sigmf-{data,meta}       one standard SigMF recording per region (default)
        manifest.csv                       this capture's rows
        frame_manifest.csv, offline_eval_summary.json   per-frame detector bookkeeping
        [mask_arrays/*.packed.npz]         the detector masks, with --save-masks

Why one recording per region: in ``--mode frequency`` every region is mixed to baseband, low-pass
filtered and decimated to its own rate, so the output is intrinsically ragged (different lengths,
different sample rates). SigMF handles that natively when each region is its own recording: the
file's ``core:sample_rate`` is that region's rate, ``captures[0].core:frequency`` its RF centre, and
the annotation carries the detected band edges plus provenance into the source stream
(``wfgt:orig_sample_start`` / ``wfgt:orig_sample_rate`` / ``wfgt:frame_number``). The flat
``manifest.csv`` is the table a loader iterates. ``--sink pack`` instead writes one variable-rate
container file per ``pack_frames`` frames (``wfgt:container: true``; see the app README) which is
the compact form the live app uses, at the cost of a custom reader.

Requirements: the demo container exists and the app has been built
(``bash_scripts/build_demo_container.sh`` once, ``bash_scripts/rebuild_demo_container_app.sh``
after changes). Needs docker/sudo like every container-touching wrapper.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
import run_cuda_dino_offline_file as offline  # noqa: E402  (shared config/staging helpers)

OFFLINE_DRIVER = APP_DIR / "run_cuda_dino_offline_file.py"
REPACK_SCRIPT = APP_DIR / "repack_offline_masks.py"

# Base config per detector. Any config works as long as it carries the detector's own block; the
# snipper / sink blocks are injected (or overridden) below, so a plain detector config is fine.
DETECTOR_BASE_CONFIGS = {
    "coherent_power": APP_DIR / "config_signal_snipper_single_channel.yaml",
    "cuda_dino": APP_DIR / "config_cuda_dino_performance_single_channel.yaml",
    "cuda_dino_finetuned": APP_DIR / "config_live_v3_dino_ft_m5s.yaml",
}

# Defaults used only when the base config has no signal_snipper / sigmf_file_sink block
# (mirrors config_signal_snipper_single_channel.yaml). merge_gap_cols is in detector-mask pixels:
# 80 on the 10240/20480-column coherent / zero-shot masks, 8 on the 1024-column fine-tuned mask.
SNIPPER_DEFAULTS = {
    "mode": "frequency",
    "oversample_percent": 25.0,
    "enable_downsample": True,
    "bandwidth_margin_hz": 0.0,
    "min_box_pixels": 256,
    "merge_gap_rows": 16,
    "merge_gap_cols": 80,
    "fir_num_taps": 129,
    "ring_depth": 32,
    "center_frequency_hz": 0.0,
    "sample_rate_hz": 0.0,
}
SINK_DEFAULTS = {
    "mode": "per_signal",
    "pack_frames": 16,
    "output_dir": "/workspace/spectrograms/snippets",
    "filename_prefix": "snip",
}


# --------------------------------------------------------------------------------------------- #
# config generation
# --------------------------------------------------------------------------------------------- #
def ensure_block(config_text: str, block: str, defaults: dict[str, object]) -> str:
    if offline.has_top_level_key(config_text, block):
        return config_text
    lines = ["", f"{block}:"] + [f"  {k}: {offline._format_yaml_scalar(v)}" for k, v in defaults.items()] + [""]
    return config_text.rstrip() + "\n" + "\n".join(lines)


def sanitize_prefix(stem: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in stem)[:80] or "snip"


def build_preprocess_config(base_config: Path, detector: str, stem: str, mode: str,
                            downsample: bool, sink_mode: str, pack_frames: int | None) -> Path:
    text = base_config.read_text(encoding="utf-8")
    snipper_defaults = dict(SNIPPER_DEFAULTS)
    if detector == "cuda_dino_finetuned":
        snipper_defaults["merge_gap_cols"] = 8
    text = ensure_block(text, "signal_snipper", snipper_defaults)
    text = ensure_block(text, "sigmf_file_sink", SINK_DEFAULTS)
    text = offline.set_block_scalar_values(text, "pipeline", {
        "enable_detector": True,
        "enable_signal_snipper": True,
        "enable_snippet_compression": False,   # a dataset wants plain cf32, not the demo codec
    })
    text = offline.set_block_scalar_values(text, "signal_snipper", {
        "mode": mode,
        "enable_downsample": bool(downsample),
    })
    sink_values: dict[str, object] = {
        "mode": sink_mode,
        "write_iq": True,            # never footprint-only here
        "write_budget_mb_s": 0.0,    # unlimited: offline runs must not skip packs
        "filename_prefix": sanitize_prefix(stem),
    }
    if pack_frames is not None:
        sink_values["pack_frames"] = int(pack_frames)
    text = offline.set_block_scalar_values(text, "sigmf_file_sink", sink_values)

    offline.GENERATED_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    out = offline.GENERATED_CONFIG_ROOT / f"preprocess_{detector}_{base_config.stem}_{stem}.yaml"
    try:
        out.write_text(text, encoding="utf-8")
    except PermissionError:   # root-owned leftover from an earlier sudo run
        out.unlink()
        out.write_text(text, encoding="utf-8")
    return out


# --------------------------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------------------------- #
def sh(cmd: list[str], check: bool = True, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, **kw)


def sudo_free_first(cmd: list[str]) -> None:
    """Run `cmd` plain first, then under sudo (container outputs under /tmp are root-owned)."""
    if subprocess.run(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).returncode == 0:
        return
    sh(["sudo", *cmd])


def move_tree(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        sudo_free_first(["rm", "-rf", str(dst)])
    sudo_free_first(["mv", str(src), str(dst)])


def dir_bytes(root: Path, suffix: str | None = None) -> int:
    total = 0
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        if p.is_file() and (suffix is None or p.name.endswith(suffix)):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def stem_of(data_path: Path) -> str:
    n = data_path.name
    return n[: -len(".sigmf-data")] if n.endswith(".sigmf-data") else data_path.stem


def discover(dataset_dir: Path, recursive: bool) -> list[Path]:
    it = dataset_dir.rglob("*.sigmf-data") if recursive else dataset_dir.glob("*.sigmf-data")
    return sorted(p for p in it if p.is_file())


def container_ready(binary: Path) -> tuple[bool, str]:
    name = offline.CONTAINER_NAME
    r = subprocess.run(["sudo", "docker", "inspect", "-f", "{{.State.Running}}", name],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, (f"container '{name}' not found -- create it with "
                       f"bash_scripts/build_demo_container.sh (or set CONTAINER_NAME)")
    if r.stdout.strip() != "true":
        print(f"==> starting stopped container {name}")
        if subprocess.run(["sudo", "docker", "start", name], capture_output=True).returncode != 0:
            return False, f"could not start container '{name}'"
    r = subprocess.run(["sudo", "docker", "exec", name, "test", "-x", str(binary)])
    if r.returncode != 0:
        return False, (f"offline binary missing in the container ({binary}) -- run "
                       f"sudo ./bash_scripts/rebuild_demo_container_app.sh first")
    return True, name


# --------------------------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------------------------- #
MANIFEST_FIELDS = [
    "capture", "file", "sink_layout", "sample_rate_hz", "center_frequency_hz",
    "sample_start", "sample_count", "duration_s",
    "freq_lower_edge_hz", "freq_upper_edge_hz", "bandwidth_hz", "label",
    "snippet_sample_rate_hz", "decimation_factor",
    "orig_sample_rate_hz", "orig_sample_start", "orig_sample_count", "frame_number",
]


def manifest_rows_for_capture(stem: str, capture_out: Path, out_root: Path) -> list[dict]:
    rows: list[dict] = []
    snip_dir = capture_out / "snippets"
    for meta_path in sorted(snip_dir.glob("*.sigmf-meta")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"   ! unreadable meta {meta_path.name}: {e}", file=sys.stderr)
            continue
        g = meta.get("global", {})
        caps = meta.get("captures") or [{}]
        rate = float(g.get("core:sample_rate", 0.0) or 0.0)
        layout = "container" if g.get("wfgt:container") else ("pack" if "_pack" in meta_path.stem else "recording")
        rel_file = os.path.relpath(meta_path.with_name(meta_path.name[: -len(".sigmf-meta")] + ".sigmf-data"), out_root)
        for a in meta.get("annotations", []):
            snip_rate = float(a.get("wfgt:snippet_sample_rate", rate) or rate)
            count = int(a.get("core:sample_count", 0) or 0)
            lo = a.get("core:freq_lower_edge")
            hi = a.get("core:freq_upper_edge")
            rows.append({
                "capture": stem,
                "file": rel_file,
                "sink_layout": layout,
                "sample_rate_hz": rate,
                "center_frequency_hz": a.get("wfgt:center_frequency", caps[0].get("core:frequency")),
                "sample_start": a.get("core:sample_start", 0),
                "sample_count": count,
                "duration_s": (count / snip_rate) if snip_rate else "",
                "freq_lower_edge_hz": lo,
                "freq_upper_edge_hz": hi,
                "bandwidth_hz": (float(hi) - float(lo)) if (lo is not None and hi is not None) else "",
                "label": a.get("core:label", ""),
                "snippet_sample_rate_hz": snip_rate,
                "decimation_factor": a.get("wfgt:decimation_factor", ""),
                "orig_sample_rate_hz": g.get("wfgt:orig_sample_rate", a.get("wfgt:orig_sample_rate")),
                "orig_sample_start": a.get("wfgt:orig_sample_start", ""),
                "orig_sample_count": a.get("wfgt:orig_sample_count", ""),
                "frame_number": a.get("wfgt:frame_number", ""),
            })
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset_dir", help="directory of *.sigmf-data + *.sigmf-meta captures")
    p.add_argument("--detector", default="coherent_power", choices=sorted(DETECTOR_BASE_CONFIGS),
                   help="which detector produces the regions of interest (default: coherent_power)")
    p.add_argument("--config", default=None,
                   help="base app config (default: the detector's config, see DETECTOR_BASE_CONFIGS)")
    p.add_argument("--out", default=None, help="output dataset dir (default: <dataset_dir>_preprocessed)")
    p.add_argument("--mode", default="frequency", choices=["frequency", "time_only"],
                   help="frequency = isolate each signal (mix+filter+decimate, default); "
                        "time_only = keep full-band time slices that contain any signal")
    p.add_argument("--no-downsample", action="store_true",
                   help="frequency mode: mix + low-pass but keep the original sample rate")
    p.add_argument("--sink", default="per_signal", choices=["per_signal", "pack"],
                   help="per_signal = one SigMF recording per region (default, dataset friendly); "
                        "pack = one variable-rate container file per pack_frames frames")
    p.add_argument("--pack-frames", type=int, default=None, help="pack mode: frames per output file")
    p.add_argument("--save-masks", action="store_true",
                   help="also keep the detector masks (mask_arrays/ packed .npz) + frame manifest")
    p.add_argument("--recursive", action="store_true", help="also descend into sub-directories")
    p.add_argument("--force", action="store_true", help="re-run captures that already have an output")
    p.add_argument("--keep-staged", action="store_true",
                   help="keep the staged input copy under /tmp/usrp_spectrograms/offline_inputs")
    p.add_argument("--captures-mounted", action="store_true",
                   help="inputs live under the container's read-only captures mount; skip staging")
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="print the plan and commands only")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        print(f"dataset dir not found: {dataset_dir}", file=sys.stderr)
        return 2
    captures = discover(dataset_dir, args.recursive)
    if not captures:
        print(f"no *.sigmf-data files under {dataset_dir}", file=sys.stderr)
        return 2
    missing_meta = [c for c in captures if not offline.sigmf_meta_for_data_path(c).is_file()]
    if missing_meta:
        print("captures without a .sigmf-meta sidecar (skipped):", file=sys.stderr)
        for c in missing_meta:
            print(f"  {c}", file=sys.stderr)
        captures = [c for c in captures if c not in missing_meta]

    out_root = Path(args.out).expanduser().resolve() if args.out else dataset_dir.with_name(dataset_dir.name + "_preprocessed")
    if out_root == dataset_dir or dataset_dir in out_root.parents:
        print("output dir must not be inside the dataset dir", file=sys.stderr)
        return 2
    base_config = Path(args.config).expanduser().resolve() if args.config else DETECTOR_BASE_CONFIGS[args.detector]
    if not base_config.is_file():
        print(f"config not found: {base_config}", file=sys.stderr)
        return 2
    if not args.config and args.detector == "cuda_dino_finetuned":
        # the fine-tuned detector loads a TorchScript export that is NOT in git
        import re
        m = re.search(r'model_script_path:\s*"([^"]+)"', base_config.read_text())
        if m:
            host_path = Path(m.group(1).replace("/workspace/holohub", str(offline.REPO_ROOT)))
            if not host_path.is_file():
                print(f"fine-tuned weights missing: {host_path}\n"
                      f"(see dino_fine_tuning/README.md -> 'Weights'; or pass --config)", file=sys.stderr)
                return 2

    run_id = time.strftime("%Y%m%d_%H%M%S")
    scratch_root = offline.HOST_SCRATCH_ROOT / "preprocess" / run_id
    binary = offline.expected_binary_path(None)

    print(f"dataset:   {dataset_dir}  ({len(captures)} capture(s))")
    print(f"output:    {out_root}")
    print(f"detector:  {args.detector}   base config: {base_config.relative_to(APP_DIR) if base_config.is_relative_to(APP_DIR) else base_config}")
    print(f"snipper:   mode={args.mode} downsample={not args.no_downsample}   sink={args.sink}"
          + (f" pack_frames={args.pack_frames}" if args.pack_frames else ""))
    print(f"container: {offline.CONTAINER_NAME}   scratch: {scratch_root}")

    if not args.dry_run:
        ok, msg = container_ready(binary)
        if not ok:
            print(msg, file=sys.stderr)
            return 1
        out_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_dir": str(dataset_dir), "output_dir": str(out_root), "detector": args.detector,
        "base_config": str(base_config), "snipper_mode": args.mode, "downsample": not args.no_downsample,
        "sink_mode": args.sink, "run_id": run_id, "captures": {},
    }
    try:
        summary["git_commit"] = subprocess.run(["git", "-C", str(APP_DIR), "rev-parse", "--short", "HEAD"],
                                               capture_output=True, text=True).stdout.strip()
    except OSError:
        pass

    failures = 0
    for cap in captures:
        stem = stem_of(cap)
        rel_parent = cap.parent.relative_to(dataset_dir)
        capture_out = out_root / rel_parent / stem
        if capture_out.joinpath("manifest.csv").exists() and not args.force:
            print(f"=== {stem}: already preprocessed, skipping (--force to redo)")
            summary["captures"][stem] = {"status": "skipped"}
            continue

        gen_cfg = build_preprocess_config(base_config, args.detector, stem, args.mode,
                                          not args.no_downsample, args.sink, args.pack_frames)
        run_root = scratch_root / stem
        cmd = [sys.executable, str(OFFLINE_DRIVER), str(cap),
               "--detector", args.detector, "--config", str(gen_cfg),
               "--output-root", str(run_root), "--no-tensors",
               "--progress-every", str(args.progress_every)]
        if not args.save_masks:
            cmd.append("--snippets-only")
        if args.captures_mounted:
            cmd.append("--captures-mounted")
        if args.dry_run:
            cmd.append("--dry-run")

        print(f"=== {stem}  ->  {capture_out}")
        print("    " + " ".join(shlex.quote(c) for c in cmd))
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(APP_DIR)).returncode
        if args.dry_run:
            continue
        if rc != 0:
            print(f"   ! detector run failed (exit {rc}) for {stem}", file=sys.stderr)
            summary["captures"][stem] = {"status": "failed", "exit_code": rc}
            failures += 1
            continue

        # ---- lay the results out as a dataset -------------------------------------------------
        snip_src = run_root / "snippets"
        if not snip_src.exists():
            print(f"   ! no snippets/ produced for {stem} (detector found nothing?) -- recording empty capture")
            snip_src.mkdir(parents=True, exist_ok=True)
        capture_out.mkdir(parents=True, exist_ok=True)
        move_tree(snip_src, capture_out / "snippets")
        for extra in ("offline_eval_summary.json", "frame_manifest.csv"):
            if (run_root / extra).exists():
                move_tree(run_root / extra, capture_out / extra)
        if args.save_masks and (run_root / "mask_arrays").exists():
            subprocess.run([sys.executable, str(REPACK_SCRIPT), str(run_root)], check=False)
            move_tree(run_root / "mask_arrays", capture_out / "mask_arrays")
        meta_src = offline.sigmf_meta_for_data_path(cap)
        sudo_free_first(["cp", "-f", str(meta_src), str(capture_out / f"{stem}.source.sigmf-meta")])
        # make the output readable/writable by the invoking user (container writes as root)
        uid = os.environ.get("SUDO_UID") or str(os.getuid())
        gid = os.environ.get("SUDO_GID") or str(os.getgid())
        sudo_free_first(["chown", "-R", f"{uid}:{gid}", str(capture_out)])

        rows = manifest_rows_for_capture(stem, capture_out, out_root)
        write_manifest(capture_out / "manifest.csv", rows)
        in_bytes = cap.stat().st_size
        out_bytes = dir_bytes(capture_out / "snippets", ".sigmf-data")
        n_files = len(list((capture_out / "snippets").glob("*.sigmf-data")))
        summary["captures"][stem] = {
            "status": "ok", "regions": len(rows), "files": n_files,
            "input_bytes": in_bytes, "output_iq_bytes": out_bytes,
            "kept_fraction": (out_bytes / in_bytes) if in_bytes else None,
            "seconds": round(time.time() - t0, 1),
        }
        print(f"   ok: {len(rows)} region(s) in {n_files} file(s); kept {100.0 * out_bytes / max(1, in_bytes):.2f}% "
              f"of the input IQ bytes in {time.time() - t0:.0f}s")

        # ---- free scratch ----------------------------------------------------------------------
        sudo_free_first(["rm", "-rf", str(run_root)])
        if not args.keep_staged and not args.captures_mounted:
            sudo_free_first(["rm", "-rf", str(offline.HOST_SCRATCH_ROOT / "offline_inputs" / stem)])

    if args.dry_run:
        return 0

    # dataset-level manifest + summary (rebuild from every per-capture manifest so resumed runs merge)
    all_rows: list[dict] = []
    for m in sorted(out_root.rglob("manifest.csv")):
        if m.parent != out_root:
            all_rows.extend(read_manifest(m))
    write_manifest(out_root / "manifest.csv", all_rows)
    summary["total_regions"] = len(all_rows)
    (out_root / "preprocess_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    uid = os.environ.get("SUDO_UID") or str(os.getuid())
    gid = os.environ.get("SUDO_GID") or str(os.getgid())
    sudo_free_first(["chown", "-R", f"{uid}:{gid}", str(out_root)])
    sudo_free_first(["rm", "-rf", str(scratch_root)])
    print(f"DONE -> {out_root}  ({len(all_rows)} regions; {failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

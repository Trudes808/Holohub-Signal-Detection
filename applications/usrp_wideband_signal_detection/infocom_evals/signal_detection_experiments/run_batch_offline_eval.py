#!/usr/bin/env python3
"""Batch orchestrator: run every capture through every detector, offline.

Drives ``run_cuda_dino_offline_file.py`` (the docker-exec offline wrapper) over the
cross-product of {captures} x {detectors}, one GPU job at a time (single GPU, so the
detectors serialise), with resume/skip-completed, lazy per-file staging cleanup, and
optional per-job metrics + mask repacking.

Layout produced::

    <output-root>/<detector>/<file_stem>/   (frame_manifest.csv, mask_arrays/, gt_masks/, ...)
    <state-dir>/batch_state.json            (resume state)

Typical run (masks only, both detectors, all 15 captures)::

    python3 run_batch_offline_eval.py \
        --captures-dir /home/bqn82/captures \
        --run-id sweep_2026_06_30 \
        --progress-every 25

Because each GPU job is a ``sudo docker exec`` into the demo container, this script
must run where ``sudo docker`` works (i.e. the user's shell, not a sandbox).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

APP_DIR = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = Path(__file__).resolve().parent
OFFLINE_WRAPPER = APP_DIR / "run_cuda_dino_offline_file.py"
DEFAULT_CAPTURES_DIR = Path("/home/bqn82/captures")
HOST_SCRATCH_ROOT = Path("/tmp/usrp_spectrograms")
# Outputs (masks/GT/manifests + state + metrics) live under the in-repo, git-ignored
# batch_runs/ dir. The repo is bind-mounted read-write at /workspace/holohub in the demo
# container, so the binary can write here; the wrapper maps the path into the container.
DEFAULT_OUTPUT_ROOT = EXPERIMENTS_DIR / "batch_runs"
DEFAULT_DETECTORS = ["cuda_dino", "coherent_power"]


def discover_captures(captures_dir: Path, only: Optional[list[str]]) -> list[Path]:
    """Return sorted *.sigmf-data files, optionally filtered to a list of stems."""
    files = sorted(captures_dir.glob("*.sigmf-data"))
    if only:
        wanted = set(only)
        files = [f for f in files if _stem(f) in wanted]
    return files


def _stem(data_path: Path) -> str:
    name = data_path.name
    return name[: -len(".sigmf-data")] if name.endswith(".sigmf-data") else data_path.stem


def run_dir_complete(run_dir: Path) -> bool:
    """A run is complete if its summary says manifest_complete and the manifest exists."""
    summary = run_dir / "offline_eval_summary.json"
    manifest = run_dir / "frame_manifest.csv"
    if not summary.exists() or not manifest.exists():
        return False
    try:
        return bool(json.loads(summary.read_text()).get("manifest_complete", False))
    except Exception:
        return False


def staged_input_dir(file_stem: str) -> Path:
    return HOST_SCRATCH_ROOT / "offline_inputs" / file_stem


def cleanup_staged_input(file_stem: str) -> None:
    staged = staged_input_dir(file_stem)
    if staged.exists():
        # staged via `sudo cp`, so removal also needs sudo
        subprocess.run(["sudo", "rm", "-rf", str(staged)], check=False)


def repack_masks(run_dir: Path) -> None:
    """Compress mask_arrays/ and gt_masks/ .npy into packbits .npz, removing raw .npy.

    Saves ~8x on binary masks. The metrics loader transparently reads either form.
    """
    import numpy as np

    for subdir in ("mask_arrays", "gt_masks"):
        d = run_dir / subdir
        if not d.exists():
            continue
        for npy in d.glob("*.npy"):
            arr = (np.load(npy) != 0)
            packed = np.packbits(arr.reshape(-1))
            np.savez_compressed(
                npy.with_suffix(".packed.npz"),
                packed=packed, rows=arr.shape[0], cols=arr.shape[1],
            )
            npy.unlink()


def run_one(
    data_path: Path,
    detector: str,
    output_root: Path,
    progress_every: int,
    save_tensors: bool,
    trace_frames: bool,
    dry_run: bool,
    config: Optional[str] = None,
) -> int:
    file_stem = _stem(data_path)
    run_dir = output_root / detector / file_stem
    cmd = [
        sys.executable, str(OFFLINE_WRAPPER), str(data_path),
        "--detector", detector,
        "--output-root", str(run_dir),
        "--progress-every", str(progress_every),
    ]
    if config:
        cmd.extend(["--config", config])
    if not save_tensors:
        cmd.append("--no-tensors")
    if trace_frames:
        cmd.append("--trace-frames")
    if dry_run:
        cmd.append("--dry-run")
    print(f"  $ {' '.join(cmd)}", flush=True)
    completed = subprocess.run(cmd, cwd=str(APP_DIR))
    return completed.returncode


def run_post_pipeline(output_root: Path, captures_dir: Path, tables_dir: Path,
                      det_threshold: float) -> bool:
    """Run metrics (eval_detector_masks) then plots (plot_eval_results) in series.

    Uses this run's own resolved paths so it stays consistent regardless of --output-root.
    Returns True if both succeeded.
    """
    py = sys.executable
    eval_cmd = [py, str(EXPERIMENTS_DIR / "eval_detector_masks.py"),
                "--batch-root", str(output_root), "--captures-dir", str(captures_dir),
                "--out-dir", str(tables_dir)]
    print(f"\n[post] metrics: {' '.join(eval_cmd)}", flush=True)
    if subprocess.run(eval_cmd, cwd=str(EXPERIMENTS_DIR)).returncode != 0:
        print("[post] eval_detector_masks.py FAILED; skipping plots.")
        return False
    plot_cmd = [py, str(EXPERIMENTS_DIR / "plot_eval_results.py"),
                "--tables-dir", str(tables_dir), "--det-threshold", str(det_threshold)]
    print(f"[post] plots: {' '.join(plot_cmd)}", flush=True)
    if subprocess.run(plot_cmd, cwd=str(EXPERIMENTS_DIR)).returncode != 0:
        print("[post] plot_eval_results.py FAILED.")
        return False
    return True


def print_notebook_setup(output_root: Path, tables_dir: Path, det_threshold: float,
                         file_stem: str, post_ok: bool) -> None:
    """Tell the user exactly which notebook cells to edit so all viz lives in the notebook."""
    nb = EXPERIMENTS_DIR / "batch_eval_review.ipynb"
    bar = "=" * 78
    print(f"\n{bar}")
    print("VISUALIZE IN THE NOTEBOOK")
    print(f"  Open: {nb}")
    print("  Restart the kernel, edit the two cells below, then Run All:")
    print()
    print("  [Parameters cell — the FIRST code cell]:")
    print(f"      BATCH_ROOT = Path('{output_root}')")
    print(f"      FILE_STEM  = '{file_stem}'")
    print()
    print("  ['Aggregate performance plots' cell — near the BOTTOM]:")
    print(f"      TABLES_DIR    = Path('{tables_dir}')")
    print(f"      DET_THRESHOLD = {det_threshold}")
    print()
    if post_ok:
        print(f"  Static PNGs already written to: {tables_dir / 'plots'}")
    else:
        print("  (Post pipeline did not complete — run eval_detector_masks.py + plot_eval_results.py manually.)")
    print(bar)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures-dir", default=str(DEFAULT_CAPTURES_DIR),
                        help="Directory of *.sigmf-data + *.sigmf-meta captures.")
    parser.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS,
                        help="Detectors to run (default: cuda_dino coherent_power).")
    parser.add_argument("--config", default=None,
                        help="Override the per-detector base config for ALL jobs (passed through "
                             "to run_cuda_dino_offline_file.py --config). Use with a single detector.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Restrict to these capture stems (e.g. attenuation_dB_0).")
    parser.add_argument("--run-id", required=True, help="Identifier for this batch run.")
    parser.add_argument("--output-root", default=None,
                        help="Mask/GT output root (default ./batch_runs/<run_id>, git-ignored).")
    parser.add_argument("--state-dir", default=None,
                        help="Where batch_state.json + metrics live (default ./batch_runs/<run_id>).")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--save-tensors", action="store_true",
                        help="Also save spectrogram tensors/previews (huge; off by default).")
    parser.add_argument("--trace-frames", action="store_true")
    parser.add_argument("--repack-masks", action="store_true",
                        help="After each job, packbits-compress masks and delete raw .npy.")
    parser.add_argument("--keep-staged", action="store_true",
                        help="Do not delete the staged input copy after each job.")
    parser.add_argument("--force", action="store_true", help="Re-run even if a run looks complete.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.add_argument("--no-post", action="store_true",
                        help="Skip the post-sweep metrics + plots pipeline (just run the detectors).")
    parser.add_argument("--det-threshold", type=float, default=0.1,
                        help="Coverage threshold passed to plot_eval_results.py (default 0.1).")
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir).expanduser().resolve()
    output_root = Path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT / args.run_id
    state_dir = Path(args.state_dir) if args.state_dir else DEFAULT_OUTPUT_ROOT / args.run_id
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "batch_state.json"

    captures = discover_captures(captures_dir, args.only)
    if not captures:
        print(f"No captures found under {captures_dir}")
        return 1

    jobs = [(det, cap) for det in args.detectors for cap in captures]
    print(f"Batch '{args.run_id}': {len(captures)} captures x {len(args.detectors)} detectors "
          f"= {len(jobs)} jobs -> {output_root}")

    state = {"run_id": args.run_id, "jobs": {}}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            state.setdefault("jobs", {})
        except Exception:
            pass

    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    completed = skipped = failed = misaligned = 0
    for index, (detector, data_path) in enumerate(jobs, start=1):
        file_stem = _stem(data_path)
        run_dir = output_root / detector / file_stem
        job_key = f"{detector}/{file_stem}"
        print(f"\n[{index}/{len(jobs)}] {job_key}")

        if not args.force and run_dir_complete(run_dir):
            print("  already complete -> skip")
            state["jobs"][job_key] = {"status": "complete", "run_dir": str(run_dir)}
            skipped += 1
            save_state()
            continue

        meta = data_path.with_name(file_stem + ".sigmf-meta")
        if not meta.exists():
            print(f"  MISSING meta {meta} -> skip")
            state["jobs"][job_key] = {"status": "missing_meta"}
            failed += 1
            save_state()
            continue

        state["jobs"][job_key] = {"status": "running", "run_dir": str(run_dir),
                                  "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        save_state()

        start = time.time()
        rc = run_one(data_path, detector, output_root, args.progress_every,
                     args.save_tensors, args.trace_frames, args.dry_run, args.config)
        elapsed = time.time() - start

        if args.dry_run:
            state["jobs"][job_key] = {"status": "dry_run"}
            save_state()
            continue

        ok = rc == 0 and run_dir_complete(run_dir)

        # Self-check for the ring-aliasing frame<->mask desync: masks should align at
        # frame offset k=0. A non-zero offset (e.g. +ring_size) means a stale/buggy binary.
        alignment_k = None
        if ok:
            try:
                from check_mask_alignment import alignment_verdict
                verdict = alignment_verdict(run_dir)
                alignment_k = verdict["best_k"]
                if not verdict["aligned"]:
                    misaligned += 1
                    print(f"  !! ALIGNMENT WARNING: masks lead GT by {alignment_k} frames "
                          f"(margin {verdict['margin']:.3f}; ring-aliasing / stale binary) — REBUILD and re-run.")
            except Exception as exc:
                print(f"  alignment check warning: {exc}")

        if ok and args.repack_masks:
            try:
                repack_masks(run_dir)
            except Exception as exc:  # repack failure shouldn't fail the job
                print(f"  repack warning: {exc}")
        if not args.keep_staged:
            cleanup_staged_input(file_stem)

        state["jobs"][job_key] = {
            "status": "complete" if ok else "failed",
            "run_dir": str(run_dir),
            "returncode": rc,
            "elapsed_sec": round(elapsed, 1),
            "alignment_frame_offset": alignment_k,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        save_state()
        if ok:
            completed += 1
            print(f"  done in {elapsed:.0f}s")
        else:
            failed += 1
            print(f"  FAILED rc={rc} (see logs above)")

    print(f"\nBatch '{args.run_id}' summary: {completed} done, {skipped} skipped, {failed} failed.")
    if misaligned:
        print(f"  !! {misaligned} run(s) had FRAME-MISALIGNED masks (ring-aliasing / stale binary). "
              f"Rebuild the container and re-run those before trusting results.")
    print(f"State: {state_path}")

    # Post-sweep pipeline: metrics -> plots, in series, using this run's own paths.
    post_ok = False
    have_runs = (completed + skipped) > 0
    if args.dry_run:
        print("(dry-run: skipping metrics + plots)")
    elif args.no_post:
        print("(--no-post: skipping metrics + plots)")
        print(f"Run manually: python3 eval_detector_masks.py --batch-root {output_root} "
              f"--captures-dir {captures_dir} --out-dir {state_dir}")
    elif not have_runs:
        print("(no completed runs -> skipping metrics + plots)")
    else:
        post_ok = run_post_pipeline(output_root, captures_dir, state_dir, args.det_threshold)

    # representative capture stem for the notebook's frame-review cell
    stems = [_stem(c) for c in captures]
    file_stem = "attenuation_dB_25" if "attenuation_dB_25" in stems else stems[len(stems) // 2]
    print_notebook_setup(output_root, state_dir, args.det_threshold, file_stem, post_ok)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

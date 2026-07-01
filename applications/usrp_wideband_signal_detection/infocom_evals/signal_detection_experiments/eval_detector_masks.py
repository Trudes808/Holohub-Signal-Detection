#!/usr/bin/env python3
"""Driver: evaluate detector-mask runs against SigMF ground truth.

Walks a batch output tree laid out as ``<root>/<detector>/<file_stem>/`` (each leaf
containing ``frame_manifest.csv`` + ``mask_arrays/`` + ``gt_masks/`` +
``gt_annotations/``), evaluates every run with :mod:`mask_eval_metrics`, joins the
breakdown attributes from the source ``.sigmf-meta``, and writes two combined tidy
fact tables (``frame_pixel_metrics`` and ``region_metrics``) plus a small JSON
summary.

Examples
--------
Evaluate a whole batch run::

    python3 eval_detector_masks.py \
        --batch-root /tmp/usrp_spectrograms/batch_eval/<run_id> \
        --captures-dir /home/bqn82/captures \
        --out-dir batch_runs/<run_id>

Evaluate a single run dir::

    python3 eval_detector_masks.py \
        --run-dir /tmp/usrp_spectrograms/offline_cuda_dino/attenuation_dB_0_... \
        --detector cuda_dino --captures-dir /home/bqn82/captures \
        --out-dir /tmp/scratch_eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import mask_eval_metrics as mem


def find_source_meta(file_stem: str, search_dirs: list[Path]) -> Optional[Path]:
    """Locate the source ``.sigmf-meta`` for a capture stem.

    Handles both the full-capture stems (``attenuation_dB_25``) and the
    generated slice stems (``attenuation_dB_25_samples_<a>_<b>``) by trying the
    exact stem first, then the leading ``attenuation_dB_<n>`` prefix.
    """
    candidates = [file_stem]
    import re

    match = re.match(r"(attenuation_dB_\d+(?:_v\d+)?)", file_stem)
    if match and match.group(1) != file_stem:
        candidates.append(match.group(1))
    for directory in search_dirs:
        for cand in candidates:
            meta = directory / f"{cand}.sigmf-meta"
            if meta.exists():
                return meta
    return None


def _is_run_dir(p: Path) -> bool:
    # completed runs have a manifest; coverage-rejected runs still have gt_annotations
    # (the metrics layer reconstructs the manifest from those).
    return (p / "frame_manifest.csv").exists() or (p / "gt_annotations").is_dir()


def discover_runs(batch_root: Path) -> list[tuple[str, str, Path]]:
    """Return ``(detector, file_stem, run_dir)`` for each leaf under a batch root."""
    runs: list[tuple[str, str, Path]] = []
    for detector_dir in sorted(p for p in batch_root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in detector_dir.iterdir() if p.is_dir()):
            if _is_run_dir(run_dir):
                runs.append((detector_dir.name, run_dir.name, run_dir))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--batch-root", help="Root of a batch run: <root>/<detector>/<file_stem>/")
    src.add_argument("--run-dir", help="A single run directory (requires --detector).")
    parser.add_argument("--detector", help="Detector label for --run-dir mode.")
    parser.add_argument("--captures-dir", action="append", default=[],
                        help="Directory holding source *.sigmf-meta (repeatable). "
                             "Defaults include the app generated_inputs and /home/bqn82/captures.")
    parser.add_argument("--out-dir", required=True, help="Where to write combined fact tables + summary.")
    parser.add_argument("--coverage-threshold", type=float, default=0.5,
                        help="Region 'detected' if box coverage >= this (default 0.5).")
    parser.add_argument("--frame-limit", type=int, default=None,
                        help="Only evaluate the first N frames per run (debug).")
    args = parser.parse_args()

    app_dir = Path(__file__).resolve().parents[2]
    search_dirs = [Path(d) for d in args.captures_dir]
    search_dirs += [app_dir / "generated_inputs", Path("/home/bqn82/captures")]
    search_dirs = [d for d in search_dirs if d.exists()]

    if args.run_dir:
        if not args.detector:
            parser.error("--run-dir requires --detector")
        run_dir = Path(args.run_dir)
        runs = [(args.detector, run_dir.name, run_dir)]
    else:
        runs = discover_runs(Path(args.batch_root))

    if not runs:
        print("No runs with frame_manifest.csv found.")
        return 1

    config = mem.EvalConfig(region_coverage_threshold=args.coverage_threshold)
    all_frame_rows: list[dict] = []
    all_region_rows: list[dict] = []
    per_run_summary: list[dict] = []

    for detector, file_stem, run_dir in runs:
        meta = find_source_meta(file_stem, search_dirs)
        frame_rows, region_rows = mem.evaluate_run(
            run_dir, detector=detector, file_stem=file_stem,
            sigmf_meta_path=meta, config=config, frame_limit=args.frame_limit,
        )
        all_frame_rows.extend(frame_rows)
        all_region_rows.extend(region_rows)
        matched = sum(1 for r in region_rows if r.get("matched_source"))
        per_run_summary.append({
            "detector": detector,
            "file_stem": file_stem,
            "run_dir": str(run_dir),
            "source_meta": str(meta) if meta else None,
            "frames": len(frame_rows),
            "regions": len(region_rows),
            "regions_matched_source": matched,
        })
        print(f"[{detector}/{file_stem}] frames={len(frame_rows)} "
              f"regions={len(region_rows)} matched_source={matched} "
              f"meta={'yes' if meta else 'MISSING'}")

    out_dir = Path(args.out_dir)
    paths = mem.write_tables(all_frame_rows, all_region_rows, out_dir)
    summary = {
        "n_runs": len(runs),
        "n_frame_rows": len(all_frame_rows),
        "n_region_rows": len(all_region_rows),
        "coverage_threshold": args.coverage_threshold,
        "tables": {k: str(v) for k, v in paths.items()},
        "runs": per_run_summary,
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(all_frame_rows)} frame rows + {len(all_region_rows)} region rows to {out_dir}")
    print(f"Summary: {out_dir / 'eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build an empirical per-position (RoPE) noise template for the CUDA DINO detector.

The detector, when run on noise-only input with `save_raw_dino_patch_prenorm: true`
and the template disabled, dumps the pre-qnorm per-patch residual RMS for every
frame under `<output>/chunk_debug/patch_prenorm/patch_prenorm_f*.npy` plus a
`meta.json`. This script averages those into a mean-1 `sigma_template[patch_rows,
patch_cols]` that the detector divides the residual RMS by at runtime to remove the
fixed edge/corner bias RoPE bakes into DINOv3 at eval.

Usage:
  python3 calibrate_dino_positional_template.py \
      --run-dir /tmp/usrp_spectrograms/offline_cuda_dino/noise_55dB \
      --run-dir /tmp/usrp_spectrograms/offline_cuda_dino/noise_45dB \
      --output calibration/dino_vitb16_noise_sigma_64x64.npy \
      --expect-deweight 0.75 --plots-dir calibration/diagnostics
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
from pathlib import Path
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:  # pragma: no cover - plots are optional
    _HAVE_MPL = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate the DINO positional noise template.")
    p.add_argument("--run-dir", action="append", required=True, type=Path,
                   help="Offline run output dir (repeatable). Expects chunk_debug/patch_prenorm/.")
    p.add_argument("--output", required=True, type=Path, help="Output template .npy path.")
    p.add_argument("--expect-deweight", type=float, default=0.75,
                   help="Assert each run's raw_dino_positional_deweight matches this.")
    p.add_argument("--min-samples", type=int, default=8,
                   help="Minimum total (frame*chunk) samples required.")
    p.add_argument("--min-correlation", type=float, default=0.98,
                   help="Hard-fail if any pair of per-run templates correlate below this.")
    p.add_argument("--plots-dir", type=Path, default=None, help="Optional diagnostics output dir.")
    p.add_argument("--mu-output", type=Path, default=None,
                   help="Optional Stage-D mean-embedding .npy from patch_features_f*.npy dumps.")
    p.add_argument("--interior-depth", type=int, default=None,
                   help="Edge-distance (patches) defining the interior for flatness metrics. "
                        "Default: min(patch_rows,patch_cols)//5.")
    p.add_argument("--reduce", choices=["median", "mean", "quantile"], default="median",
                   help="Per-position statistic across frames*chunks. 'median' (default) is robust to "
                        "signal contamination (signal spikes a minority of frames per patch, so the "
                        "median tracks the noise floor). 'quantile' with a low --quantile isolates the "
                        "noise floor even when a position is signal-covered in up to (1-q) of frames. "
                        "'mean' only for pristine noise-only input.")
    p.add_argument("--quantile", type=float, default=0.30,
                   help="Per-position quantile in [0,1] when --reduce quantile (default 0.30). Lower = "
                        "more aggressive noise-floor isolation, at the cost of fewer effective samples.")
    return p.parse_args()


def reduce_samples(samples: np.ndarray, how: str, quantile: float = 0.30) -> np.ndarray:
    """Per-position reduce over axis 0 (frames*chunks) -> (patch_rows, patch_cols).

    Signal only raises a patch's residual RMS, so a low per-position quantile (or the
    median) recovers the noise floor without reading GT annotations."""
    if how == "median":
        return np.median(samples, axis=0)
    if how == "quantile":
        return np.quantile(samples, float(np.clip(quantile, 0.0, 1.0)), axis=0)
    return samples.mean(axis=0)


def _prenorm_dir(run_dir: Path) -> Path:
    return run_dir / "chunk_debug" / "patch_prenorm"


def load_run(run_dir: Path) -> dict:
    pdir = _prenorm_dir(run_dir)
    meta_path = pdir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing {meta_path} (run with save_raw_dino_patch_prenorm: true)")
    meta = json.loads(meta_path.read_text())
    if meta.get("artifact_contract") != "raw_dino_patch_prenorm_v1":
        raise ValueError(f"unexpected artifact_contract in {meta_path}: {meta.get('artifact_contract')}")
    if meta.get("raw_dino_positional_template_path", "") != "":
        raise ValueError(
            f"{meta_path}: dumps were produced WITH a template applied "
            f"('{meta['raw_dino_positional_template_path']}') — recapture with the template disabled.")
    patch_rows = int(meta["patch_rows"])
    patch_cols = int(meta["patch_cols"])
    chunk_count = int(meta["chunk_count"])
    files = sorted(glob.glob(str(pdir / "patch_prenorm_f*.npy")))
    if not files:
        raise FileNotFoundError(f"no patch_prenorm_f*.npy under {pdir}")

    samples = []  # each (patch_rows, patch_cols)
    for f in files:
        arr = np.load(f).astype(np.float64)  # shape (chunk_count*patch_rows, patch_cols)
        expected_rows = chunk_count * patch_rows
        if arr.shape != (expected_rows, patch_cols):
            raise ValueError(f"{f}: shape {arr.shape} != expected {(expected_rows, patch_cols)}")
        per_chunk = arr.reshape(chunk_count, patch_rows, patch_cols)
        for c in range(chunk_count):
            samples.append(per_chunk[c])
    stacked = np.stack(samples, axis=0)  # (n_samples, patch_rows, patch_cols)
    return {
        "run_dir": run_dir,
        "meta": meta,
        "patch_rows": patch_rows,
        "patch_cols": patch_cols,
        "chunk_count": chunk_count,
        "samples": stacked,
        "template": stacked.mean(axis=0),  # per-run mean (patch_rows, patch_cols)
        "deweight": float(meta.get("raw_dino_positional_deweight", float("nan"))),
    }


def edge_distance_map(rows: int, cols: int) -> np.ndarray:
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return np.minimum.reduce([rr, rows - 1 - rr, cc, cols - 1 - cc])


def ring_profile(template: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    max_d = int(dist.max())
    rings = np.arange(max_d + 1)
    means = np.array([template[dist == d].mean() for d in rings])
    return rings, means


def flatness_metrics(template: np.ndarray, dist: np.ndarray, interior_depth: int) -> dict:
    edge = template[dist == 0]
    interior = template[dist >= interior_depth]
    interior_mean = float(interior.mean()) if interior.size else float(template.mean())
    edge_interior_ratio = float(edge.mean()) / max(interior_mean, 1e-9)
    ring_cv = float(template.std()) / max(float(template.mean()), 1e-9)
    return {"edge_interior_ratio": edge_interior_ratio, "ring_cv": ring_cv,
            "interior_depth": int(interior_depth)}


def maybe_write_mu(run_dirs: list[Path], mu_output: Path) -> None:
    feats = []
    for rd in run_dirs:
        for f in sorted(glob.glob(str(_prenorm_dir(rd) / "patch_features_f*.npy"))):
            feats.append(np.load(f).astype(np.float64))  # (chunk_count*patch_count, feature_dim)
    if not feats:
        print("  [mu] no patch_features_f*.npy found; skipping mean-embedding output", file=sys.stderr)
        return
    # All frames share (patch_count, feature_dim); average the per-position rows.
    # Each file is (chunk_count*patch_count, feature_dim); we cannot separate chunk vs
    # patch here without patch_count, so infer it from the template grid via the caller.
    stacked = np.concatenate(feats, axis=0)
    mu_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(mu_output, stacked.mean(axis=0, keepdims=True).astype(np.float32))
    print(f"  [mu] wrote pooled mean embedding {stacked.shape} -> {mu_output}")


def main() -> int:
    args = parse_args()
    runs = [load_run(rd) for rd in args.run_dir]

    # Consistency across runs.
    r0 = runs[0]
    for r in runs[1:]:
        if (r["patch_rows"], r["patch_cols"]) != (r0["patch_rows"], r0["patch_cols"]):
            raise ValueError(f"grid mismatch: {r['run_dir']} {r['patch_rows']}x{r['patch_cols']} "
                             f"vs {r0['patch_rows']}x{r0['patch_cols']}")
    for r in runs:
        if not np.isnan(args.expect_deweight) and abs(r["deweight"] - args.expect_deweight) > 1e-6:
            raise ValueError(f"{r['run_dir']}: deweight {r['deweight']} != --expect-deweight {args.expect_deweight}")

    total_samples = sum(r["samples"].shape[0] for r in runs)
    if total_samples < args.min_samples:
        raise ValueError(f"only {total_samples} samples (< --min-samples {args.min_samples}); capture more frames")

    # Cross-run correlation gate (per-run templates use the chosen robust reduction).
    templates = [reduce_samples(r["samples"], args.reduce, args.quantile) for r in runs]
    for i in range(len(templates)):
        for j in range(i + 1, len(templates)):
            corr = float(np.corrcoef(templates[i].ravel(), templates[j].ravel())[0, 1])
            print(f"cross-run template correlation {runs[i]['run_dir'].name} vs {runs[j]['run_dir'].name}: {corr:.4f}")
            if corr < args.min_correlation:
                raise SystemExit(f"FAIL: correlation {corr:.4f} < {args.min_correlation} — bias not deterministic "
                                 f"or input contaminated (signal present?).")

    # Global template = per-position reduce over all samples, normalized to mean 1.0, clamped.
    all_samples = np.concatenate([r["samples"] for r in runs], axis=0)
    template = reduce_samples(all_samples, args.reduce, args.quantile)
    template = template / max(float(template.mean()), 1e-9)
    template = np.clip(template, 0.05, 20.0).astype(np.float32)

    rows, cols = template.shape
    dist = edge_distance_map(rows, cols)
    interior_depth = args.interior_depth if args.interior_depth is not None else max(1, min(rows, cols) // 5)
    metrics = flatness_metrics(template.astype(np.float64), dist, interior_depth)
    print(f"template {rows}x{cols}  samples={total_samples}  "
          f"edge_interior_ratio={metrics['edge_interior_ratio']:.3f}  ring_cv={metrics['ring_cv']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, template)
    sidecar = {
        "artifact": "dino_positional_sigma_template_v1",
        "patch_rows": rows,
        "patch_cols": cols,
        "expect_deweight": args.expect_deweight,
        "model_script_path": r0["meta"].get("model_script_path", ""),
        "aligned_rows": r0["meta"].get("aligned_rows"),
        "aligned_cols": r0["meta"].get("aligned_cols"),
        "source_runs": [str(r["run_dir"]) for r in runs],
        "total_samples": int(total_samples),
        "edge_interior_ratio": metrics["edge_interior_ratio"],
        "ring_cv": metrics["ring_cv"],
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote template -> {args.output}  (+ sidecar {args.output.with_suffix('.json').name})")

    # Per-chunk-index deviation report (decides whether a per-chunk-index template is
    # ever needed). Samples were appended chunk-major per frame, so index % chunk_count
    # recovers the originating chunk index.
    if r0["chunk_count"] > 1:
        cc = r0["chunk_count"]
        worst = 0.0
        for r in runs:
            s = r["samples"]
            for c in range(cc):
                idx = np.arange(c, s.shape[0], cc)
                if idx.size == 0:
                    continue
                chunk_template = s[idx].mean(axis=0)
                chunk_template = chunk_template / max(float(chunk_template.mean()), 1e-9)
                worst = max(worst, float(np.max(np.abs(chunk_template - template))))
        print(f"  [per-chunk] max |per-chunk-index template - global| = {worst:.3f} "
              f"(per-chunk-index template deferred unless this is large, e.g. > ~0.1).")

    if args.mu_output is not None:
        maybe_write_mu(args.run_dir, args.mu_output)

    # Diagnostics.
    if args.plots_dir is not None and _HAVE_MPL:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(template, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"positional sigma template {rows}x{cols}\n"
                     f"edge/interior={metrics['edge_interior_ratio']:.2f} ring_cv={metrics['ring_cv']:.3f}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(args.plots_dir / "template_heatmap.png", dpi=160)
        plt.close(fig)

        rings, before = ring_profile(template.astype(np.float64), dist)
        _, after = ring_profile((template.astype(np.float64) / template.astype(np.float64)), dist)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rings, before, "o-", label="template (before division)")
        ax.plot(rings, after, "s-", label="after simulated division")
        ax.axhline(1.0, color="gray", lw=0.7, ls="--")
        ax.set_xlabel("distance to edge (patches)")
        ax.set_ylabel("mean sigma")
        ax.set_title("ring profile")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.plots_dir / "ring_profile.png", dpi=160)
        plt.close(fig)
        print(f"  [plots] wrote diagnostics to {args.plots_dir}")
    elif args.plots_dir is not None:
        print("  [plots] matplotlib unavailable; skipping diagnostics", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

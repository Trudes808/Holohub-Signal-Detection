#!/usr/bin/env python3
"""Export a fine-tuned DINOv3 segmenter (DinoSegmenter) to TorchScript for the live
`cuda_dino_finetuned` Holoscan operator.

Unlike `export_dinov3_torchscript.py` (which exports the *backbone* only, for the
zero-shot `cuda_dino` path), this traces the **full segmenter** (backbone + trained
SegHead) so the operator can run `mask = sigmoid(model(tile)) >= threshold` with no
fusion/post-processing.

The model input geometry is CONFIG-DRIVEN. A checkpoint is trained at a fixed grid
(tile_rows x nfft); this script traces at that exact grid and writes a sidecar
`<out>.meta.json` carrying the "geometry contract" the operator/config must match:

    { sample_rate_hz, nfft, tile_rows, db_vmin, db_vmax, threshold,
      bin_hz = sample_rate_hz/nfft, row_seconds = nfft/sample_rate_hz }

Deployment note: the shipped checkpoints are 256 x 1024 (M1_ft thr 0.45, M2_ft thr
0.85). A different receive setup that cannot match the trained per-pixel physics
(240 kHz/bin x 4.17 us/row @ 245.76 MSps) needs a *retrain* at the new geometry
(edit dino_fine_tuning/configs/dataset.yaml, retrain, then export with the matching
--tile-rows / --nfft / --sample-rate-hz). See the app README fine-tuning section.

Run in the `dinov3` conda env (needs the dinov3 repo + backbone weights + the
`dino_fine_tuning/src` modules importable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# --- documented defaults for the shipped 256x1024 captures (245.76 MSps) --------
DEFAULT_SAMPLE_RATE_HZ = 245_760_000.0
# Global fixed dB->[0,1] clip used by the shipped M-series dataset (build_dataset
# --calibrate). Override for a differently calibrated dataset via --db-vmin/--db-vmax
# or --dataset-meta.
DEFAULT_DB_VMIN = -46.934
DEFAULT_DB_VMAX = 19.557


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True,
                   help="Fine-tuned checkpoint best.pt (bundles mode + state_dict + name).")
    p.add_argument("--output", required=True, help="Output TorchScript .ts path.")
    p.add_argument("--train-yaml", default=None,
                   help="dino_fine_tuning/configs/train.yaml (for weights_path, feat_layers, "
                        "unfreeze_last_n). Defaults to the repo copy next to this script's "
                        "dino_fine_tuning tree.")
    p.add_argument("--src-dir", default=None,
                   help="dino_fine_tuning/src dir to import model.py from (auto-detected if omitted).")
    p.add_argument("--dinov3-repo", default="/home/bqn82/dinov3",
                   help="Path to the DINOv3 repo (prepended to sys.path so `import dinov3` works). "
                        "Set empty if dinov3 is already installed/importable.")
    # geometry (the config contract) --------------------------------------------
    p.add_argument("--tile-rows", type=int, default=256, help="Model input time rows (mult of 16).")
    p.add_argument("--nfft", type=int, default=1024, help="Model input freq bins (mult of 16).")
    p.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ,
                   help="Receive sample rate the checkpoint's physics assumes (single-rate models).")
    p.add_argument("--rate-range-hz", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                   help="Deployment sample-rate range the model was domain-randomized over. Defaults to "
                        "[min,max] of the dataset meta's dr_rates_hz if present.")
    p.add_argument("--fft-window", default=None,
                   help="FFT analysis window the model was trained with (hann|hamming|blackman|none). "
                        "Defaults to the dataset meta's fft_window. Recorded so the operator matches.")
    # dB clip + threshold: from --dataset-meta / --eval-meta if given, else these --
    p.add_argument("--db-vmin", type=float, default=None)
    p.add_argument("--db-vmax", type=float, default=None)
    p.add_argument("--threshold", type=float, default=None,
                   help="Decision threshold. Falls back to --eval-meta, then checkpoint, then 0.5.")
    p.add_argument("--dataset-meta", default=None,
                   help="JSON with db_vmin/db_vmax (e.g. built dataset meta).")
    p.add_argument("--eval-meta", default=None,
                   help="eval_meta.json with the val-tuned 'threshold' (e.g. eval_out/<model>/).")
    # overrides / trace opts ------------------------------------------------------
    p.add_argument("--weights-path", default=None, help="Override backbone weights_path.")
    p.add_argument("--trace-batch", type=int, default=1, help="Example batch for tracing.")
    p.add_argument("--autocast", default="none", choices=["none", "bf16", "fp16"],
                   help="Bake mixed precision into the trace (bf16 recommended: ~3.4x faster, "
                        "~unchanged accuracy, real-time). 'none' = fp32 trace.")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--no-verify", action="store_true",
                   help="Skip the eager-vs-traced IoU parity check.")
    return p.parse_args()


def _load_json(path) -> dict:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}


def resolve_device(name: str) -> torch.device:
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA export requested but torch.cuda.is_available() is False.")
    return dev


def main() -> int:
    args = parse_args()
    here = Path(__file__).resolve()
    repo_root = here.parents[2]  # .../holohub-dev
    dft = repo_root / "dino_fine_tuning"

    src_dir = Path(args.src_dir) if args.src_dir else (dft / "src")
    if not (src_dir / "model.py").exists():
        raise FileNotFoundError(f"model.py not found under --src-dir {src_dir}")
    # model.py does `import dinov3.hub.backbones` at import time, so the dinov3 repo must be on
    # sys.path first (it is typically not pip-installed in the env).
    if args.dinov3_repo:
        dinov3_repo = Path(args.dinov3_repo)
        if not (dinov3_repo / "dinov3").exists():
            raise FileNotFoundError(
                f"--dinov3-repo {dinov3_repo} does not contain a 'dinov3/' package")
        sys.path.insert(0, str(dinov3_repo))
    sys.path.insert(0, str(src_dir))
    from model import DinoSegmenter  # noqa: E402

    train_yaml = Path(args.train_yaml) if args.train_yaml else (dft / "configs" / "train.yaml")
    train_cfg = yaml.safe_load(Path(train_yaml).read_text())

    if args.tile_rows % 16 or args.nfft % 16:
        raise ValueError(f"tile_rows ({args.tile_rows}) and nfft ({args.nfft}) must be multiples of 16.")

    device = resolve_device(args.device)

    ckpt = torch.load(args.ckpt, map_location=device)
    weights_path = args.weights_path or train_cfg["weights_path"]
    model = DinoSegmenter(
        weights_path,
        feat_layers=tuple(train_cfg["feat_layers"]),
        mode=ckpt.get("mode", "ft_lastN"),
        unfreeze_last_n=int(train_cfg.get("unfreeze_last_n", 4)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # dB clip + threshold resolution (explicit > meta files > checkpoint > defaults)
    ds_meta = _load_json(args.dataset_meta)
    ev_meta = _load_json(args.eval_meta)
    db_vmin = args.db_vmin if args.db_vmin is not None else ds_meta.get("db_vmin", DEFAULT_DB_VMIN)
    db_vmax = args.db_vmax if args.db_vmax is not None else ds_meta.get("db_vmax", DEFAULT_DB_VMAX)
    threshold = (args.threshold if args.threshold is not None
                 else ev_meta.get("threshold", ckpt.get("threshold", 0.5)))
    db_vmin, db_vmax, threshold = float(db_vmin), float(db_vmax), float(threshold)

    example = torch.randn(args.trace_batch, 1, args.tile_rows, args.nfft,
                          device=device, dtype=torch.float32)
    # Mixed precision is baked into the trace (not toggled at runtime): tracing under autocast records
    # the reduced-precision casts into the graph, so the operator runs bf16/fp16 compute on an fp32
    # input with no runtime dtype handling. bf16 gives ~3.4x speedup at ~unchanged accuracy and is what
    # brings the real-time downsample path under the frame budget.
    autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.autocast)
    with torch.no_grad():
        if autocast_dtype is not None:
            with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=autocast_dtype):
                traced = torch.jit.trace(model, example, strict=False, check_trace=False)
        else:
            traced = torch.jit.trace(model, example, strict=False)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out))

    meta = {
        "model_name": ckpt.get("name", "finetuned_dino"),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "sample_rate_hz": float(args.sample_rate_hz),
        "nfft": int(args.nfft),
        "tile_rows": int(args.tile_rows),
        "autocast": args.autocast,
        "bin_hz": float(args.sample_rate_hz) / int(args.nfft),
        "row_seconds": int(args.nfft) / float(args.sample_rate_hz),
        # Deployment rate range the model is valid over: explicit > dataset dr_rates_hz > null
        # (single-rate). The operator can warn if the live rate falls outside this.
        "trained_rate_range_hz": (list(args.rate_range_hz) if args.rate_range_hz else
                                  ([float(min(ds_meta["dr_rates_hz"])), float(max(ds_meta["dr_rates_hz"]))]
                                   if ds_meta.get("dr_rates_hz") else None)),
        "fft_window": args.fft_window or ds_meta.get("fft_window", "hann"),
        "db_vmin": db_vmin,
        "db_vmax": db_vmax,
        "threshold": threshold,
        "output_is_logits": True,
        "input_layout": "B,1,tile_rows,nfft (float[0,1]); channel repeat + imagenet-norm are inside the model",
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Saved TorchScript -> {out}")
    print(f"Saved geometry contract -> {meta_path}")
    print(f"  geometry: {args.tile_rows}x{args.nfft} @ {args.sample_rate_hz/1e6:.2f} MSps "
          f"= {meta['bin_hz']/1e3:.1f} kHz/bin, {meta['row_seconds']*1e6:.3f} us/row; "
          f"vmin={db_vmin}, vmax={db_vmax}, thr={threshold}")

    if not args.no_verify:
        with torch.no_grad():
            if autocast_dtype is not None:  # compare like-for-like (eager under the same autocast)
                with torch.autocast("cuda" if device.type == "cuda" else "cpu", dtype=autocast_dtype):
                    le = model(example)
            else:
                le = model(example)
            lt = traced(example)
        me = (torch.sigmoid(le) >= threshold)
        mt = (torch.sigmoid(lt) >= threshold)
        inter = (me & mt).sum().item()
        union = (me | mt).sum().item()
        iou = 1.0 if union == 0 else inter / union
        max_abs = (le - lt).abs().max().item()
        print(f"  verify: eager-vs-traced mask IoU={iou:.6f}, max|dlogit|={max_abs:.3e}")
        if iou < 0.999:
            print("  WARNING: traced/eager masks diverge; investigate before deploying.",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Export a fine-tuned DinoSegmenter checkpoint to TorchScript for the native C++
finetuned_dino_detector operator, and verify TorchScript == eager numerically.

The traced module IS the model only (backbone + SegHead): input [B,1,256,1024] float
in [0,1], output logits [B,1,256,1024]. The spectrogram front-end (frames_to_db),
[0,1] normalization, tiling, sigmoid+threshold, and stitch live in the C++ operator --
exactly mirroring how cuda_dino_detector keeps its front-end in C++ and torchscript = model.

Usage:
  python export_finetuned_torchscript.py --ckpt ../checkpoints/M1_ft/best.pt \
      --out ../weights/finetuned_dino_m1.ts --eval-meta ../eval_out/M1_ft/eval_meta.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

FT_ROOT = Path(__file__).resolve().parents[1]
for p in ("/home/bqn82/dinov3", str(FT_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import yaml
from model import DinoSegmenter
import finetuned_infer as fi


def build_model(ckpt_path, train_cfg, device):
    st = torch.load(ckpt_path, map_location=device)
    m = DinoSegmenter(train_cfg["weights_path"], feat_layers=tuple(train_cfg["feat_layers"]),
                      mode=st["mode"], unfreeze_last_n=train_cfg["unfreeze_last_n"]).to(device)
    m.load_state_dict(st["model"]); m.eval()
    return m, st.get("name", Path(ckpt_path).parent.name), st["mode"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-meta", default=None, help="eval_meta.json for the tuned threshold (recorded in sidecar)")
    ap.add_argument("--tile-rows", type=int, default=256)
    ap.add_argument("--nfft", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4, help="example batch for tracing/parity")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_cfg = yaml.safe_load(open(FT_ROOT / "configs/train.yaml"))
    ds_meta = json.loads((FT_ROOT / "data/dataset/dataset_meta.json").read_text())
    thr = fi.load_threshold(a.eval_meta) if a.eval_meta else 0.5
    model, name, mode = build_model(a.ckpt, train_cfg, device)
    print(f"loaded {name} mode={mode} threshold={thr:.3f} device={device}")

    ex = torch.rand(a.batch, 1, a.tile_rows, a.nfft, device=device, dtype=torch.float32)
    with torch.no_grad():
        ts = torch.jit.trace(model, ex, check_trace=False)
        ts = torch.jit.freeze(ts.eval())

    # parity: TorchScript vs eager (fp32) on fresh random inputs of a different batch size
    with torch.no_grad():
        x = torch.rand(a.batch + 3, 1, a.tile_rows, a.nfft, device=device, dtype=torch.float32)
        le = model(x).float(); lt = ts(x).float()
        max_abs = (le - lt).abs().max().item()
        me = (torch.sigmoid(le) >= thr); mt = (torch.sigmoid(lt) >= thr)
        agree = (me == mt).float().mean().item()
    print(f"parity: max|logit_eager - logit_ts| = {max_abs:.3e}   mask-agreement = {agree*100:.4f}%")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    ts.save(str(out))
    sidecar = out.with_suffix(".meta.json")
    sidecar.write_text(json.dumps({
        "name": name, "mode": mode, "threshold": thr, "tile_rows": a.tile_rows, "nfft": a.nfft,
        "db_vmin": ds_meta["db_vmin"], "db_vmax": ds_meta["db_vmax"],
        "input": "float[B,1,tile_rows,nfft] in [0,1]", "output": "logits[B,1,tile_rows,nfft]",
        "post": "sigmoid(logits) >= threshold", "ckpt": str(a.ckpt),
    }, indent=2))
    print(f"saved {out}  ({out.stat().st_size/1e6:.1f} MB)  + {sidecar.name}")
    print("PARITY_OK" if max_abs < 1e-3 and agree > 0.9999 else "PARITY_WARN")


if __name__ == "__main__":
    main()

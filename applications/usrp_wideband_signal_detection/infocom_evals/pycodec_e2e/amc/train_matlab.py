"""Train compressed-domain T-PRIME on the ORIGINAL MATLAB waveform library.

    .venv-ml/bin/python -m amc.train_matlab [--codecs none sc16 bfp12 bfp8]
        [--n-train-per-class 4500] [--n-val-per-class 600]
        [--out amc/weights_matlab]

PAIRED design: ONE seeded window set (9 protocol classes + noise, AWGN drawn
U(-10, 30) dB in-band, stored-snip-like decimation) is generated once and
cached; each codec model trains on the SAME windows under its own quantization,
so the codec comparison carries no sampling variance. Saves
tprime_<codec>.pt (fp16) + matlab_amc_config.json (classes, val acc, confusion).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .matlab_ds import CLASSES10, featurize, make_windows, quantize_windows, scan_library
from .models import TPrimeC

HP = dict(lr=2e-4, epochs=30, batch=128)


def evaluate(net, loader, device):
    net.eval()
    conf = np.zeros((len(CLASSES10), len(CLASSES10)), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            pred = net(xb.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(yb.numpy(), pred):
                conf[t, p] += 1
    return conf.trace() / conf.sum(), conf


def featurize_split(x: np.ndarray, codec: str, chunk: int = 2048) -> torch.Tensor:
    outs = []
    for i in range(0, len(x), chunk):
        mant, e_rel = quantize_windows(x[i:i + chunk], codec)
        outs.append(featurize(mant, e_rel))
    return torch.from_numpy(np.concatenate(outs))


def train_codec(codec, xtr, ytr, xva, yva, device, out_dir):
    print(f"[{codec}] featurizing...", flush=True)
    tr = DataLoader(TensorDataset(featurize_split(xtr, codec),
                                  torch.from_numpy(ytr)),
                    batch_size=HP["batch"], shuffle=True, drop_last=True)
    va = DataLoader(TensorDataset(featurize_split(xva, codec),
                                  torch.from_numpy(yva)),
                    batch_size=512)
    net = TPrimeC(classes=len(CLASSES10)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=HP["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=HP["epochs"])
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc, best_state, best_conf = 0.0, None, None
    for ep in range(HP["epochs"]):
        net.train()
        tot = 0.0
        t0 = time.time()
        for xb, yb in tr:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        acc, conf = evaluate(net, va, device)
        if acc > best_acc:
            best_acc, best_conf = acc, conf
            best_state = {k: v.detach().cpu().half() for k, v in net.state_dict().items()}
        print(f"[{codec}] epoch {ep + 1}/{HP['epochs']} loss {tot / len(tr):.4f} "
              f"val acc {acc:.4f} (best {best_acc:.4f}) [{time.time() - t0:.0f}s]",
              flush=True)
    torch.save(best_state, os.path.join(out_dir, f"tprime_{codec}.pt"))
    return best_acc, best_conf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--codecs", nargs="+", default=["none", "sc16", "bfp12", "bfp8"])
    ap.add_argument("--n-train-per-class", type=int, default=4500)
    ap.add_argument("--n-val-per-class", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "weights_matlab"))
    ap.add_argument("--cache", default="/tmp/usrp_spectrograms/matlab_ds_cache.npz")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if os.path.exists(args.cache):
        print(f"[ds] cache hit {args.cache}", flush=True)
        z = np.load(args.cache)
        xtr, ytr, xva, yva = z["xtr"], z["ytr"], z["xva"], z["yva"]
    else:
        print("[ds] scanning MATLAB library...", flush=True)
        lib = scan_library()
        for cls, recs in lib.items():
            print(f"[ds]   {cls}: {len(recs)} records, snip rate "
                  f"{recs[0]['rate'] / 1e6:.3f} MHz", flush=True)
        xtr, ytr, _ = make_windows(lib, args.n_train_per_class, seed=0xA11CE)
        xva, yva, _ = make_windows(lib, args.n_val_per_class, seed=0xB0B)
        np.savez(args.cache, xtr=xtr, ytr=ytr, xva=xva, yva=yva)
        print(f"[ds] cached {args.cache}", flush=True)
    print(f"[ds] train {xtr.shape} val {xva.shape}", flush=True)

    results = {}
    for codec in args.codecs:
        acc, conf = train_codec(codec, xtr, ytr, xva, yva, device, args.out)
        results[codec] = dict(val_acc=round(float(acc), 4), confusion=conf.tolist())
        print(f"[{codec}] DONE best val acc {acc:.4f}", flush=True)

    cfg = dict(classes=CLASSES10, window=8192, block=64, feats_per_token=258,
               snr_train_db=[-10, 30], results={k: v["val_acc"] for k, v in results.items()})
    with open(os.path.join(args.out, "matlab_amc_config.json"), "w") as f:
        json.dump(dict(cfg, confusions={k: v["confusion"] for k, v in results.items()}), f)
    print("[done]", json.dumps(cfg["results"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

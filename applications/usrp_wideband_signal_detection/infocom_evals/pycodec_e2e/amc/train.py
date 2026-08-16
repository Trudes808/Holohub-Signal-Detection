"""Train the three AMC models on synthetic channelized windows.

    .venv-ml/bin/python -m amc.train [--n-train 45000] [--n-val 6000]
        [--out amc/weights]

Saves fp16 state dicts + amc_config.json (classes, window sizes, val
accuracy, confusion matrices). Reproducible: dataset is seeded synthesis.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import CLASSES, WINDOW, build, rms_normalize, to_tensor
from .synth import WIN, make_dataset

HP = {  # lr per the source papers/repos; epochs sized for synthetic data
    "vtcnn2": dict(lr=1e-3, epochs=15, batch=256),
    "resnet1d": dict(lr=1e-3, epochs=20, batch=256),
    "tprime": dict(lr=2e-4, epochs=30, batch=128),
}


def tensorize(x: np.ndarray, model: str) -> torch.Tensor:
    n = WINDOW[model]
    out = np.stack([to_tensor(rms_normalize(w[:n]), model) for w in x])
    return torch.from_numpy(out)


def evaluate(net, loader, device):
    net.eval()
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in loader:
            pred = net(xb.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(yb.numpy(), pred):
                conf[t, p] += 1
    return conf.trace() / conf.sum(), conf


def train_one(name, xtr, ytr, xva, yva, device):
    hp = HP[name]
    net = build(name).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    tr = DataLoader(TensorDataset(tensorize(xtr, name), torch.from_numpy(ytr.astype(np.int64))),
                    batch_size=hp["batch"], shuffle=True, drop_last=True)
    va = DataLoader(TensorDataset(tensorize(xva, name), torch.from_numpy(yva.astype(np.int64))),
                    batch_size=512)
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    loss_fn = torch.nn.CrossEntropyLoss()
    best_acc, best_state, best_conf, since = 0.0, None, None, 0
    for ep in range(hp["epochs"]):
        net.train()
        t0, tot = time.time(), 0.0
        for xb, yb in tr:
            opt.zero_grad()
            loss = loss_fn(net(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()
            tot += loss.item()
        sched.step()
        acc, conf = evaluate(net, va, device)
        print(f"[{name}] epoch {ep + 1}/{hp['epochs']} loss {tot / len(tr):.4f} "
              f"val_acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if acc > best_acc:
            best_acc, since = acc, 0
            best_state = {k: v.detach().cpu().half() for k, v in net.state_dict().items()}
            best_conf = conf
        else:
            since += 1
            if since >= 6:
                print(f"[{name}] early stop", flush=True)
                break
    return best_state, best_acc, best_conf, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=45000)
    ap.add_argument("--n-val", type=int, default=6000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "weights"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}; synthesizing {args.n_train}+{args.n_val} windows "
          f"of {WIN} samples", flush=True)
    t0 = time.time()
    xtr, ytr = make_dataset(args.n_train, seed=1)
    xva, yva = make_dataset(args.n_val, seed=2)
    print(f"dataset ready in {time.time() - t0:.0f}s", flush=True)

    os.makedirs(args.out, exist_ok=True)
    report = {"classes": CLASSES, "window": WINDOW, "trained": time.strftime("%Y-%m-%d %H:%M"),
              "n_train": args.n_train, "n_val": args.n_val, "models": {}}
    for name in ("vtcnn2", "resnet1d", "tprime"):
        state, acc, conf, n_params = train_one(name, xtr, ytr, xva, yva, device)
        torch.save(state, os.path.join(args.out, f"{name}.pt"))
        report["models"][name] = {"val_acc": round(float(acc), 4),
                                  "params": int(n_params),
                                  "confusion": conf.tolist()}
        print(f"[{name}] BEST val_acc {acc:.4f} ({n_params / 1e6:.2f}M params)", flush=True)
    with open(os.path.join(args.out, "amc_config.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v["val_acc"] for k, v in report["models"].items()}, indent=1))


if __name__ == "__main__":
    main()

"""Train a DINOv3 segmenter for binary signal/noise detection.

Examples:
  # frozen-head, all attenuations (M2 backbone-frozen baseline head)
  python train.py --config configs/train.yaml --dataset data/dataset \
      --mode frozen --name M2_frozen --out checkpoints/M2_frozen

  # backbone-adapted, low attenuation only (M1)
  python train.py --config configs/train.yaml --dataset data/dataset \
      --mode ft_lastN --atten-max 30 --name M1_ft --out checkpoints/M1_ft
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import RFSegDataset
from model import DinoSegmenter, DiceBCELoss


def log(msg):
    print(f"[train] {msg}", flush=True)


@torch.no_grad()
def evaluate(model, loader, device, thr, amp):
    model.eval()
    tp = fp = fn = 0
    tot_pos = tot = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = model(img)
        pred = (torch.sigmoid(logits.float()) >= thr)
        gt = mask.bool()
        tp += int((pred & gt).sum()); fp += int((pred & ~gt).sum()); fn += int((~pred & gt).sum())
        tot_pos += int(gt.sum()); tot += int(gt.numel())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec,
            "pos_frac": tot_pos / tot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["frozen", "ft_lastN"], default="frozen")
    ap.add_argument("--atten-max", type=int, default=None,
                    help="train only on captures with attenuation_db <= this (M1)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.epochs:
        cfg["epochs"] = args.epochs
    device = "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    tr = RFSegDataset(args.dataset, "train", atten_max_db=args.atten_max,
                      augment=cfg["augment"], seed=cfg["seed"])
    va = RFSegDataset(args.dataset, "val", atten_max_db=None, augment=False)
    log(f"{args.name}: train={len(tr)} (atten_max={args.atten_max}) val={len(va)}")
    tl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True,
                    num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    vl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=cfg["num_workers"], pin_memory=True)

    model = DinoSegmenter(cfg["weights_path"], feat_layers=tuple(cfg["feat_layers"]),
                          mode=args.mode, unfreeze_last_n=cfg["unfreeze_last_n"]).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log(f"trainable params: {n_train:.2f}M  (mode={args.mode})")
    loss_fn = DiceBCELoss(pos_weight=cfg["pos_weight"], dice_w=cfg["dice_w"],
                          bce_w=cfg["bce_w"]).to(device)
    opt = torch.optim.AdamW(model.param_groups(cfg["lr_head"], cfg["lr_backbone"]),
                            weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * max(1, len(tl))
    warmup = int(cfg["warmup_frac"] * steps)

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    start_epoch, best_iou, gstep = 0, -1.0, 0
    history = []
    ckpt_last = out / "last.pt"
    if args.resume and ckpt_last.exists():
        st = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        start_epoch = st["epoch"] + 1; best_iou = st["best_iou"]; gstep = st["gstep"]
        history = st.get("history", [])
        log(f"resumed from epoch {start_epoch}, best_iou={best_iou:.4f}")

    base_lrs = [g["lr"] for g in opt.param_groups]
    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        t0 = time.time(); run = 0.0
        for it, batch in enumerate(tl):
            for g, blr in zip(opt.param_groups, base_lrs):
                g["lr"] = blr * lr_at(gstep)
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg["amp"]):
                logits = model(img)
                loss = loss_fn(logits, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            run += loss.item(); gstep += 1
        val = evaluate(model, vl, device, cfg["threshold"], cfg["amp"])
        rec = {"epoch": epoch, "train_loss": run / max(1, len(tl)),
               "lr": opt.param_groups[0]["lr"], "time_s": time.time() - t0, **val}
        history.append(rec)
        log(f"ep{epoch:02d} loss={rec['train_loss']:.4f} "
            f"val_IoU={val['iou']:.4f} F1={val['f1']:.4f} "
            f"P={val['precision']:.3f} R={val['recall']:.3f} ({rec['time_s']:.0f}s)")
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": epoch, "best_iou": best_iou, "gstep": gstep,
                 "history": history, "cfg": cfg, "mode": args.mode,
                 "atten_max": args.atten_max, "name": args.name}
        torch.save(state, ckpt_last)
        if val["iou"] > best_iou:
            best_iou = val["iou"]; state["best_iou"] = best_iou
            torch.save(state, out / "best.pt")
            log(f"  new best val_IoU={best_iou:.4f} -> best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "DONE").write_text(f"best_val_iou={best_iou:.4f}\nepochs={cfg['epochs']}\n")
    log(f"DONE {args.name}: best val_IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()

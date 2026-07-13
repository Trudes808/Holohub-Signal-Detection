"""Torch Dataset over the materialized spectrogram/mask stacks."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class RFSegDataset(Dataset):
    def __init__(self, root, split, atten_max_db=None, augment=False, seed=0):
        self.root = Path(root)
        self.split = split
        self.meta = json.loads((self.root / "dataset_meta.json").read_text())
        self.frames = np.load(self.root / f"frames_{split}.npy", mmap_mode="r")
        self.masks = np.load(self.root / f"masks_{split}.npy", mmap_mode="r")
        rows = [r for r in csv.DictReader(open(self.root / "frames.csv")) if r["split"] == split]
        if atten_max_db is not None:
            rows = [r for r in rows if r["attenuation_db"] != "" and int(r["attenuation_db"]) <= atten_max_db]
        self.rows = rows
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        pos = int(r["mem_pos"])
        img = np.asarray(self.frames[pos], dtype=np.float32) / 255.0
        mask = np.asarray(self.masks[pos], dtype=np.float32)
        if self.augment:
            img, mask = self._augment(img, mask)
        img = torch.from_numpy(np.ascontiguousarray(img))[None]   # 1,H,W
        mask = torch.from_numpy(np.ascontiguousarray(mask))[None]  # 1,H,W
        return {"image": img, "mask": mask, "frame_id": r["frame_id"],
                "attenuation_db": r["attenuation_db"], "mem_pos": pos}

    def _augment(self, img, mask):
        # random frequency flip (spectrum mirrored about DC)
        if self.rng.random() < 0.5:
            img = img[:, ::-1]; mask = mask[:, ::-1]
        # random circular time roll
        if self.rng.random() < 0.5:
            s = int(self.rng.integers(0, img.shape[0]))
            img = np.roll(img, s, axis=0); mask = np.roll(mask, s, axis=0)
        # light additive noise on the image (SNR jitter)
        if self.rng.random() < 0.3:
            img = np.clip(img + self.rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
        return img, mask

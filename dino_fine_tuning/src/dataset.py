"""Torch Dataset over the materialized spectrogram/mask stacks.

Two storage modes (from dataset_meta.json "frame_storage"):
  - "uint8_norm"  : legacy single-grid build; frames are uint8 [0,1] (post dB->clip). Light image-domain
                    augmentation only.
  - "float16_db"  : domain-randomized build; frames are float16 dB (PRE-clip), already emulated across
                    rates/centers with the per-rate envelope baked in. Augment in the dB DOMAIN (random
                    level offset -> gain invariance; small envelope tilt jitter) BEFORE clipping to [0,1]
                    with the global db_vmin/db_vmax. This is what teaches level/envelope invariance.
"""
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
            rows = [r for r in rows if r.get("attenuation_db", "") not in ("", None)
                    and int(r["attenuation_db"]) <= atten_max_db]
        self.rows = rows
        self.augment = augment
        self.rng = np.random.default_rng(seed)

        self.storage = self.meta.get("frame_storage", "uint8_norm")
        self.vmin = float(self.meta.get("db_vmin", 0.0) or 0.0)
        self.vmax = float(self.meta.get("db_vmax", 1.0) or 1.0)
        self.db_span = max(self.vmax - self.vmin, 1e-6)
        # dB-domain level-offset augmentation range (gain invariance). Informed by the measured floor
        # spread from the sweep (floor_stats.json level_offset_range_db), but CLAMPED: a per-frame dB
        # offset larger than ~level_cap pushes the noise floor past the fixed db_vmax and saturates the
        # image, which teaches the model that a bright floor is "signal" (precision collapse). The raw
        # sweep range can be ~60 dB wide (a 0-60 dB rx-gain sweep), so cap the per-frame jitter here.
        # level_cap_db is tunable via dataset_meta / dataset.yaml; default keeps the floor well inside [0,1].
        level_cap = float(self.meta.get("level_cap_db", 12.0))
        self.level_lo, self.level_hi = -min(8.0, level_cap), min(8.0, level_cap)
        self.floor_pct = 20.0
        # Floor-ANCHORED level augmentation: re-level each frame's noise floor to a value sampled from
        # the sweep's observed floor distribution (per-(role,rate) p10..p90). A scalar dB shift is
        # SNR-preserving, matches the real captured power range exactly, and can never produce an
        # out-of-sweep floor (unlike a blind offset, which double-counts the per-rate spread already
        # baked into the emulated frames). Falls back to the clamped ±offset if no sweep stats.
        self.floor_targets = []
        ssd = self.meta.get("sweep_stats_dir")
        fs_path = Path(ssd) / "floor_stats.json" if ssd else None
        if fs_path and fs_path.exists():
            fs = json.loads(fs_path.read_text())
            for role, rate_map in fs.get("floor_stats", {}).items():
                for st in rate_map.values():
                    self.floor_targets.append((float(st["p10"]), float(st["p90"])))
            rng_db = fs.get("level_offset_range_db")
            if rng_db and len(rng_db) == 2:
                self.level_lo = float(np.clip(rng_db[0], -level_cap, -6.0))
                self.level_hi = float(np.clip(rng_db[1], 6.0, level_cap))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        pos = int(r["mem_pos"])
        raw = np.asarray(self.frames[pos], dtype=np.float32)
        mask = np.asarray(self.masks[pos], dtype=np.float32)

        if self.storage == "float16_db":
            db = raw
            if self.augment:
                db, mask = self._augment_db(db, mask)
            img = np.clip((db - self.vmin) / self.db_span, 0.0, 1.0)
            if self.augment and self.rng.random() < 0.3:
                img = np.clip(img + self.rng.normal(0, 0.02, img.shape).astype(np.float32), 0.0, 1.0)
        else:
            img = raw / 255.0
            if self.augment:
                img, mask = self._augment(img, mask)

        img = torch.from_numpy(np.ascontiguousarray(img))[None]    # 1,H,W
        mask = torch.from_numpy(np.ascontiguousarray(mask))[None]
        return {"image": img, "mask": mask, "frame_id": r["frame_id"],
                "attenuation_db": r.get("attenuation_db", ""), "mem_pos": pos}

    # ---- augmentation ------------------------------------------------------------------------------
    def _augment(self, img, mask):
        """Legacy [0,1] image-domain aug (uint8_norm frames)."""
        if self.rng.random() < 0.5:
            img = img[:, ::-1]; mask = mask[:, ::-1]
        if self.rng.random() < 0.5:
            s = int(self.rng.integers(0, img.shape[0]))
            img = np.roll(img, s, axis=0); mask = np.roll(mask, s, axis=0)
        if self.rng.random() < 0.3:
            img = np.clip(img + self.rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
        return img, mask

    def _augment_db(self, db, mask):
        """dB-domain aug for float16_db frames (applied BEFORE the clip)."""
        # geometry: frequency flip + circular time roll
        if self.rng.random() < 0.5:
            db = db[:, ::-1]; mask = mask[:, ::-1]
        if self.rng.random() < 0.5:
            s = int(self.rng.integers(0, db.shape[0]))
            db = np.roll(db, s, axis=0); mask = np.roll(mask, s, axis=0)
        # LEVEL (gain / antenna / band invariance -- the key one): re-level the frame's noise floor to a
        # floor sampled from the sweep's real distribution (SNR-preserving), so the model sees the varied
        # absolute levels the radio actually produces -- and never an out-of-sweep power.
        if self.floor_targets:
            p10, p90 = self.floor_targets[int(self.rng.integers(len(self.floor_targets)))]
            target = self.rng.uniform(p10, p90)
            cur = np.percentile(db, self.floor_pct)
            db = db + np.float32(target - cur)
        else:
            db = db + np.float32(self.rng.uniform(self.level_lo, self.level_hi))
        # envelope JITTER: a small random smooth tilt on top of the baked per-rate envelope, so residual
        # per-band envelope differences don't trip the model.
        if self.rng.random() < 0.5:
            tilt = float(self.rng.uniform(-3.0, 3.0))  # dB across the band
            db = db + np.linspace(-tilt / 2, tilt / 2, db.shape[1], dtype=np.float32)[None, :]
        return db, mask

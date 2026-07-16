"""Run the fine-tuned DINOv3 segmenter on a batch-eval frame's IQ.

The batch eval frames are 512 x 10240 (nfft=10240). The fine-tuned model has its
own front-end (nfft=1024, 256-row tiles), so we reconstruct the spectrogram from
the frame's raw IQ at the model's native geometry, run the model tile-by-tile,
stitch the per-tile masks, and hand back a native (rows, 1024) mask. The caller
resizes it onto the common display/GT grid with mem.resize_mask_nearest -- exactly
how the other detectors' masks are resampled for comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model import DinoSegmenter
import rfdata as rf


class FinetunedDetector:
    def __init__(self, ckpt_path, train_cfg, dataset_meta, device="cuda",
                 threshold=None, nfft=1024, tile_rows=256, amp=True):
        st = torch.load(ckpt_path, map_location=device)
        self.model = DinoSegmenter(
            train_cfg["weights_path"], feat_layers=tuple(train_cfg["feat_layers"]),
            mode=st["mode"], unfreeze_last_n=train_cfg["unfreeze_last_n"]).to(device)
        self.model.load_state_dict(st["model"])
        self.model.eval()
        self.vmin = float(dataset_meta["db_vmin"])
        self.vmax = float(dataset_meta["db_vmax"])
        self.nfft = nfft
        self.tile = tile_rows          # multiple of patch size (16)
        self.device = device
        self.amp = amp
        self.threshold = 0.5 if threshold is None else float(threshold)
        self.name = st.get("name", "finetuned_dino")

    @torch.no_grad()
    def mask_for_iq(self, iq: np.ndarray) -> np.ndarray:
        """iq (complex) -> binary (rows, nfft) uint8 mask at the model's geometry."""
        n = (len(iq) // self.nfft) * self.nfft
        rows = n // self.nfft
        iqt = torch.from_numpy(np.ascontiguousarray(iq[:n].astype(np.complex64))).to(self.device)
        db = rf.frames_to_db(iqt[None], self.nfft, rows)[0]                 # rows, nfft
        img = torch.clamp((db - self.vmin) / max(self.vmax - self.vmin, 1e-6), 0, 1)
        # split into tiles of `tile` rows (pad the last tile)
        spans = [(r0, min(rows, r0 + self.tile)) for r0 in range(0, rows, self.tile)]
        batch = []
        for r0, r1 in spans:
            t = img[r0:r1]
            if t.shape[0] < self.tile:
                t = torch.nn.functional.pad(t, (0, 0, 0, self.tile - t.shape[0]))
            batch.append(t)
        x = torch.stack(batch)[:, None]                                     # B,1,tile,nfft
        out = []
        for i in range(0, x.shape[0], 16):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                logits = self.model(x[i:i + 16])
            out.append((torch.sigmoid(logits.float()) >= self.threshold)[:, 0].to(torch.uint8))
        pm = torch.cat(out)                                                 # B,tile,nfft
        mask = torch.zeros((rows, self.nfft), dtype=torch.uint8, device=self.device)
        for k, (r0, r1) in enumerate(spans):
            mask[r0:r1] = pm[k, :r1 - r0]
        return mask.cpu().numpy()

    def mask_for_frame(self, data_path, local_offset_complex, n_complex,
                       read_frame_iq) -> np.ndarray:
        """Convenience: read the frame's IQ (via eval_viz.read_frame_iq) then infer."""
        iq = read_frame_iq(Path(data_path), int(local_offset_complex), int(n_complex))
        return self.mask_for_iq(iq)


def to_display_grid(mask_native: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Map a native binary mask onto the (out_rows, out_cols) display/GT grid.

    Uses MAX pooling when shrinking a dimension (a coarse cell is ON if ANY fine
    cell under it is ON) and nearest-neighbour when growing. This preserves thin,
    short-time detections (e.g. Zadoff-Chu pulses) that plain nearest-downsampling
    would drop -- important because the batch grid (512 rows) is 10x coarser in time
    than the model's native grid (~5120 rows).
    """
    t = torch.from_numpy(mask_native.astype(np.float32))[None, None]
    h, w = t.shape[-2], t.shape[-1]
    # rows
    if out_rows < h:
        t = torch.nn.functional.adaptive_max_pool2d(t, (out_rows, w))
    elif out_rows > h:
        t = torch.nn.functional.interpolate(t, size=(out_rows, w), mode="nearest")
    # cols
    h2 = t.shape[-2]
    if out_cols < w:
        t = torch.nn.functional.adaptive_max_pool2d(t, (h2, out_cols))
    elif out_cols > w:
        t = torch.nn.functional.interpolate(t, size=(h2, out_cols), mode="nearest")
    return (t[0, 0].numpy() > 0.5).astype(np.uint8)


def load_threshold(eval_meta_path) -> float:
    p = Path(eval_meta_path)
    if p.exists():
        return float(json.loads(p.read_text()).get("threshold", 0.5))
    return 0.5

"""Run a fine-tuned YOLO26 detector on a batch-eval frame's IQ and emit a binary mask
in the SAME geometry as the DINO detectors, so eval_detector_masks.py scores it identically.

Mirrors dino_fine_tuning/src/finetuned_infer.py: reconstruct the spectrogram from the
frame's raw IQ at the model's native geometry (nfft=1024, 256-row tiles, the global
dB->uint8 calibration the model trained on), run YOLO detection per tile, fill each
predicted box into the tile mask, stitch tiles -> (rows, nfft), and reuse
finetuned_infer.to_display_grid to map onto the common display/GT grid (max-pool).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

FT_SRC = Path("/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/src")
sys.path.insert(0, str(FT_SRC))
import rfdata as rf                              # noqa: E402
from ultralytics import YOLO                     # noqa: E402


class YoloDetector:
    def __init__(self, ckpt_path, dataset_meta, device="cuda", conf=0.25, imgsz=1024,
                 nfft=1024, tile_rows=256, name=None):
        self.model = YOLO(str(ckpt_path))
        self.vmin = float(dataset_meta["db_vmin"]); self.vmax = float(dataset_meta["db_vmax"])
        self.nfft = nfft; self.tile = tile_rows
        self.device = device; self.conf = float(conf); self.imgsz = int(imgsz)
        self.name = name or Path(ckpt_path).resolve().parent.parent.name

    @torch.no_grad()
    def mask_for_iq(self, iq: np.ndarray) -> np.ndarray:
        """iq (complex) -> binary (rows, nfft) uint8 mask at the model's native geometry."""
        n = (len(iq) // self.nfft) * self.nfft
        if n == 0:
            return np.zeros((self.tile, self.nfft), np.uint8)
        rows = n // self.nfft
        iqt = torch.from_numpy(np.ascontiguousarray(iq[:n].astype(np.complex64))).to(self.device)
        db = rf.frames_to_db(iqt[None], self.nfft, rows)[0].cpu().numpy()      # rows, nfft
        img = rf.db_to_uint8(db, self.vmin, self.vmax)                         # rows, nfft uint8
        spans = [(r0, min(rows, r0 + self.tile)) for r0 in range(0, rows, self.tile)]
        tiles = []
        for r0, r1 in spans:
            t = img[r0:r1]
            if t.shape[0] < self.tile:                                         # pad last tile
                t = np.pad(t, ((0, self.tile - t.shape[0]), (0, 0)))
            tiles.append(np.repeat(t[:, :, None], 3, axis=2))                  # HxWx3 (replicate gray)
        results = self.model.predict(tiles, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)
        mask = np.zeros((rows, self.nfft), np.uint8)
        for k, (r0, r1) in enumerate(spans):
            tm = np.zeros((self.tile, self.nfft), np.uint8)
            for x0, y0, x1, y1 in results[k].boxes.xyxy.cpu().numpy():
                xi0, xi1 = max(0, int(np.floor(x0))), min(self.nfft, int(np.ceil(x1)))
                yi0, yi1 = max(0, int(np.floor(y0))), min(self.tile, int(np.ceil(y1)))
                if xi1 > xi0 and yi1 > yi0:
                    tm[yi0:yi1, xi0:xi1] = 1
            mask[r0:r1] = tm[:r1 - r0]
        return mask


def to_display_grid(mask_native: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """Map a native binary mask onto the (out_rows,out_cols) display/GT grid: MAX-pool when
    shrinking (a coarse cell is ON if ANY fine cell is), nearest when growing. Verbatim copy of
    dino_fine_tuning/src/finetuned_infer.to_display_grid so YOLO masks resample identically."""
    t = torch.from_numpy(mask_native.astype(np.float32))[None, None]
    h, w = t.shape[-2], t.shape[-1]
    if out_rows < h:
        t = torch.nn.functional.adaptive_max_pool2d(t, (out_rows, w))
    elif out_rows > h:
        t = torch.nn.functional.interpolate(t, size=(out_rows, w), mode="nearest")
    h2 = t.shape[-2]
    if out_cols < w:
        t = torch.nn.functional.adaptive_max_pool2d(t, (h2, out_cols))
    elif out_cols > w:
        t = torch.nn.functional.interpolate(t, size=(h2, out_cols), mode="nearest")
    return (t[0, 0].numpy() > 0.5).astype(np.uint8)

"""Real-time (downsample) fine-tuned DINOv3 detector — the Python twin of the deployed
`finetuned_dino_detector` operator's `real_time_downsample` path, for the 6-detector eval harness.

Mirrors the C++ operator exactly so the eval measures the SAME optimization:
  wide FFT (downsample_fft, = the app's dynamic size) -> dB
  -> gain correction  -10*log10(downsample_fft / model_nfft)   (FFT-length processing-gain, OTA-derivable)
  -> normalize (dataset db_vmin/db_vmax) -> bilinear resize freq to the model width (1024)
  -> tile rows into 256-row tiles -> segmenter forward under autocast bf16 -> sigmoid >= threshold
  -> stitch -> nearest-upsample freq back to the wide grid.

Exposes the same `.mask_for_iq(iq)->(rows, wide) uint8` + `.nfft` contract as FinetunedDetector, so
run_ml_detectors_offline drives it and `to_display_grid` max-pools it onto the shared eval grid
identically to the native `dino_finetuned`. The only difference vs `dino_finetuned` is this front-end
— which is the whole point: it quantifies the accuracy cost of the real-time optimization.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from finetuned_infer import FinetunedDetector

_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16}


class RealtimeFinetunedDetector:
    def __init__(self, ckpt_path, train_cfg, dataset_meta, device="cuda", threshold=None,
                 downsample_fft=10240, model_nfft=1024, tile_rows=256, amp_dtype="bf16"):
        # Reuse FinetunedDetector purely to load the model + dB calibration + threshold identically.
        base = FinetunedDetector(ckpt_path, train_cfg, dataset_meta, device=device,
                                 threshold=threshold, nfft=model_nfft, tile_rows=tile_rows)
        self.model = base.model
        self.vmin, self.vmax = base.vmin, base.vmax
        self.threshold = base.threshold
        self.device = device
        self.model_nfft = int(model_nfft)
        self.tile = int(tile_rows)
        self.fft_size = int(downsample_fft)
        self.gain_offset_db = 10.0 * math.log10(self.fft_size / self.model_nfft)
        self.amp_dtype = _AMP.get(amp_dtype)  # None => fp32
        self.nfft = self.fft_size             # frames need >= this many samples (too-short check)
        self.name = "dino_finetuned_rt"

    @torch.no_grad()
    def mask_for_iq(self, iq: np.ndarray) -> np.ndarray:
        """iq (complex) -> binary (rows, fft_size) uint8 mask on the wide grid."""
        n = (len(iq) // self.fft_size) * self.fft_size
        rows = n // self.fft_size
        iqt = torch.from_numpy(np.ascontiguousarray(iq[:n].astype(np.complex64))).to(self.device)
        blk = iqt.reshape(rows, self.fft_size)
        spec = torch.fft.fftshift(torch.fft.fft(blk, dim=-1), dim=-1)
        db = 10.0 * torch.log10(spec.real ** 2 + spec.imag ** 2 + 1e-12) - self.gain_offset_db
        img = torch.clamp((db - self.vmin) / max(self.vmax - self.vmin, 1e-6), 0.0, 1.0)  # rows,fft_size
        # bilinear resize freq -> model width
        img = F.interpolate(img[None, None], size=(rows, self.model_nfft),
                            mode="bilinear", align_corners=False)[0, 0]                    # rows,1024
        # tile rows (pad to whole tiles)
        B = math.ceil(rows / self.tile)
        pad = B * self.tile - rows
        if pad > 0:
            img = F.pad(img, (0, 0, 0, pad))
        tiles = img.view(B, 1, self.tile, self.model_nfft)
        out = []
        for i in range(0, B, 16):
            if self.amp_dtype is not None:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    logits = self.model(tiles[i:i + 16])
            else:
                logits = self.model(tiles[i:i + 16])
            out.append((torch.sigmoid(logits.float()) >= self.threshold)[:, 0].to(torch.uint8))
        m = torch.cat(out).reshape(B * self.tile, self.model_nfft)[:rows]                  # rows,1024
        # nearest-upsample freq back to the wide grid (preserves thin detections)
        up = F.interpolate(m[None, None].float(), size=(rows, self.fft_size), mode="nearest")[0, 0]
        return (up >= 0.5).to(torch.uint8).cpu().numpy()

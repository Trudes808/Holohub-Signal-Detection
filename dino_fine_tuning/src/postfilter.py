"""Model-independent post-filters for the deployment mask (candidates for the operator).

fill_guard: per 256-row tile, if the mask covers >= occ_thr of the tile while only < energy_frac of the tile's
input pixels sit above `energy_thr` (robust-normalised value; 0.45 ~ +11 dB above the p20 floor), the tile is a
"paint-over" of a quiet band: keep only the mask pixels whose input exceeds `keep_thr` (~ +6 dB). Legitimate dense
frames (a full cellular band) keep their mask because their energy fraction is high.
"""
from __future__ import annotations
import numpy as np


def fill_guard(mask: np.ndarray, img: np.ndarray, tile_rows: int = 256, occ_thr: float = 0.5, mask_median_thr: float = 0.33,
               keep_thr: float = 0.30) -> tuple[np.ndarray, list]:
    """mask/img: [rows, cols] (uint8 / float in [0,1]). A tile is a paint-over when the mask covers >= occ_thr of it
    AND the median input value under the mask is below mask_median_thr (~ +7 dB above the p20 floor, i.e. the mask
    sits on noise-level pixels). A real dense carrier keeps its mask because its masked pixels are bright.
    Returns (filtered mask, list of guarded tile indices)."""
    out = mask.copy(); guarded = []
    for t in range(mask.shape[0] // tile_rows):
        sl = slice(t * tile_rows, (t + 1) * tile_rows)
        m = mask[sl] > 0; x = img[sl]
        if m.mean() >= occ_thr and float(np.median(x[m])) < mask_median_thr:
            out[sl] = m & (x > keep_thr)
            guarded.append(t)
    return out, guarded

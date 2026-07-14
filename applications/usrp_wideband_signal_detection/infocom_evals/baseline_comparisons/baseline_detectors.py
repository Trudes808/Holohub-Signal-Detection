#!/usr/bin/env python3
"""Non-ML baseline signal detectors for the offline mask evaluation.

Two classic (no-machine-learning) detectors that consume the same (time, freq)
spectrogram the trained detectors see and emit a binary ``uint8`` mask on the
identical FFT grid, so they drop straight into the existing evaluation
(``eval_detector_masks.py`` + ``eval_viz`` + the notebook) as extra
``detector_type`` values alongside ``coherent_power`` / ``cuda_dino``:

  * ``3dB_power``       — a pure moving-average power detector. For each time row
    it slides a moving average along frequency and flags every bin that rises a
    few dB above that local average (crosses ON above ``threshold_db``, OFF when
    it falls back below). No calibration, no learning — just a local mean.
  * ``blob_detection``  — classic image-processing blob detection: estimate a
    smooth background, threshold the residual, clean up with morphological
    opening/closing, label connected components and keep blobs above a minimum
    area. The kept blobs are the mask.

Both operate on the spectrogram in **dB power** (``10*log10(|X|^2)``), which is
what ``eval_viz`` produces from either a saved ``spectrogram_tensor`` (``<c8``)
or a reconstruction from the source SigMF. Pure numpy + ``scipy.ndimage`` — no
GPU, no Holoscan, no skimage/opencv.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def power_db_from_tensor(tensor: np.ndarray) -> np.ndarray:
    """Complex (or already-real-dB) spectrogram tensor -> dB power (float32).

    Accepts a complex ``<c8`` tensor (the saved ``spectrogram_tensor_*.npy``) and
    returns ``10*log10(|x|^2)``. If handed a real array it is assumed to already
    be dB power and returned unchanged (float32).
    """
    arr = np.asarray(tensor)
    if np.iscomplexobj(arr):
        power = np.abs(arr) ** 2 + 1e-12
        return (10.0 * np.log10(power)).astype(np.float32)
    return arr.astype(np.float32)


# --------------------------------------------------------------------------- #
# Detector 1: static 3 dB power-threshold detector
# --------------------------------------------------------------------------- #
@dataclass
class StaticThresholdParams:
    threshold_db: float = 3.0        # trigger this many dB above the reference floor
    noise_percentile: float = 50.0   # single scalar floor = this percentile of the whole frame (dB)
    floor_db: float | None = None    # if set, use this ABSOLUTE floor (truly static across frames)


def static_3db_threshold(spectrogram_db: np.ndarray,
                         params: StaticThresholdParams | None = None,
                         **kwargs) -> np.ndarray:
    """Static single-threshold power detector.

    Uses ONE threshold value for the whole frame (not per frequency bin): a single
    scalar noise-floor reference plus ``threshold_db``. A bin is ON when its power
    exceeds that one value. The floor is either an absolute ``floor_db`` (truly
    static across frames) or, by default, a percentile of the whole frame's dB
    power (a single per-frame scalar that tracks the overall level). Returns a
    ``uint8`` mask the same shape as the input.
    """
    p = params or StaticThresholdParams(**kwargs)
    db = power_db_from_tensor(spectrogram_db)
    floor = float(p.floor_db) if p.floor_db is not None else float(np.percentile(db, p.noise_percentile))
    mask = db > (floor + float(p.threshold_db))
    return mask.astype(np.uint8)


# --------------------------------------------------------------------------- #
# Detector 2: traditional edge-detection-based blob detection
# --------------------------------------------------------------------------- #
@dataclass
class BlobDetectionParams:
    smooth_sigma: float = 1.0        # Gaussian pre-smoothing before the edge operator
    edge_percentile: float = 90.0    # keep gradient magnitudes above this percentile
    close_iters: int = 2             # binary-closing iterations to link edge fragments
    fill_holes: bool = True          # fill regions enclosed by edges -> solid blobs
    min_blob_area: int = 64          # discard connected components smaller than this (px)


def blob_detection(spectrogram_db: np.ndarray,
                   params: BlobDetectionParams | None = None,
                   **kwargs) -> np.ndarray:
    """Traditional (non-ML) edge-detection-based blob detector.

    A generic textbook computer-vision pipeline — deliberately *not* tuned to these
    spectrograms — meant as a plain baseline:

      1. Gaussian smooth the image.
      2. Sobel gradient magnitude -> edge map (edges bound signal regions).
      3. Threshold edges at a percentile of the gradient magnitude.
      4. Morphological closing to link edge fragments into closed contours.
      5. Fill the enclosed regions -> solid blobs.
      6. Connected-component label; keep blobs above ``min_blob_area``.

    Returns a ``uint8`` mask the same shape as the input.
    """
    p = params or BlobDetectionParams(**kwargs)
    db = power_db_from_tensor(spectrogram_db)

    smoothed = ndimage.gaussian_filter(db, sigma=float(p.smooth_sigma))
    gx = ndimage.sobel(smoothed, axis=0, mode="reflect")
    gy = ndimage.sobel(smoothed, axis=1, mode="reflect")
    grad = np.hypot(gx, gy)

    edges = grad > np.percentile(grad, float(p.edge_percentile))

    cross = ndimage.generate_binary_structure(2, 1)
    if p.close_iters and p.close_iters > 0:
        edges = ndimage.binary_closing(edges, structure=cross, iterations=int(p.close_iters))
    regions = ndimage.binary_fill_holes(edges) if p.fill_holes else edges

    labels, n = ndimage.label(regions, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(db, dtype=np.uint8)

    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep_ids = np.nonzero(sizes >= p.min_blob_area)[0] + 1
    if keep_ids.size == 0:
        return np.zeros_like(db, dtype=np.uint8)
    return np.isin(labels, keep_ids).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Registry: detector_type -> callable(spectrogram_db, **params) -> uint8 mask
# --------------------------------------------------------------------------- #
DETECTORS: dict[str, Callable[..., np.ndarray]] = {
    "3dB_power": static_3db_threshold,
    "blob_detection": blob_detection,
}

PARAM_TYPES = {
    "3dB_power": StaticThresholdParams,
    "blob_detection": BlobDetectionParams,
}


def run_detector(detector_type: str, spectrogram_db: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Dispatch to a baseline detector by name with a plain param dict."""
    if detector_type not in DETECTORS:
        raise KeyError(f"unknown baseline detector_type {detector_type!r}; "
                       f"choices: {sorted(DETECTORS)}")
    param_obj = PARAM_TYPES[detector_type](**(params or {}))
    return DETECTORS[detector_type](spectrogram_db, param_obj)

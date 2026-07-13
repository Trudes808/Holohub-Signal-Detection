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


def _moving_average_1d(row_stack: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average along the last axis (reflect-padded edges).

    ``row_stack`` is (time, freq); the average is taken along frequency so each
    time row gets its own local baseline. ``uniform_filter1d`` with
    ``mode='reflect'`` keeps the window length honest at the band edges.
    """
    window = max(1, int(window))
    if window <= 1:
        return row_stack.astype(np.float32, copy=True)
    return ndimage.uniform_filter1d(
        row_stack.astype(np.float32), size=window, axis=-1, mode="reflect"
    )


# --------------------------------------------------------------------------- #
# Detector 1: 3 dB moving-average power detector
# --------------------------------------------------------------------------- #
@dataclass
class MovingAveragePowerParams:
    window_bins: int = 129        # sliding-average length along frequency (odd)
    threshold_db: float = 3.0     # trigger ON this many dB above the local average
    axis: str = "freq"            # 'freq' (per-time-row) or 'time' (per-freq-column)
    min_run_bins: int = 1         # drop ON runs shorter than this (speckle guard)


def moving_average_3db(spectrogram_db: np.ndarray,
                       params: MovingAveragePowerParams | None = None,
                       **kwargs) -> np.ndarray:
    """Pure moving-average power detector.

    For each frequency row (a spectrum at one time step) compute a moving average
    along frequency and turn a bin ON when its power crosses ``threshold_db`` above
    that moving average, OFF when it drops back below. This yields contiguous ON
    runs wherever a signal sits above its local noise baseline. Returns a
    ``uint8`` mask the same shape as the input.
    """
    p = params or MovingAveragePowerParams(**kwargs)
    db = power_db_from_tensor(spectrogram_db)
    if p.axis == "time":
        # per-frequency-column baseline: average down the time axis instead
        baseline = _moving_average_1d(db.T, p.window_bins).T
    else:
        baseline = _moving_average_1d(db, p.window_bins)
    mask = (db > baseline + float(p.threshold_db))

    if p.min_run_bins > 1:
        mask = _drop_short_runs(mask, p.min_run_bins, axis=(-1 if p.axis == "freq" else 0))
    return mask.astype(np.uint8)


def _drop_short_runs(mask: np.ndarray, min_run: int, axis: int) -> np.ndarray:
    """Zero out ON runs shorter than ``min_run`` along ``axis`` (speckle guard)."""
    structure = np.zeros((3, 3), dtype=bool)
    if axis in (-1, 1):
        structure[1, :] = True     # connect along frequency
    else:
        structure[:, 1] = True     # connect along time
    labels, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_run
    return keep[labels]


# --------------------------------------------------------------------------- #
# Detector 2: classic image-processing blob detection
# --------------------------------------------------------------------------- #
@dataclass
class BlobDetectionParams:
    background_mode: str = "per_freq_median"  # 'per_freq_median' | 'uniform' | 'global'
    background_window: int = 257     # window for the 'uniform' background estimate
    threshold_k: float = 3.5         # threshold = background + k * robust_std(residual)
    threshold_db: float | None = None  # if set, absolute dB-over-background threshold
    open_size: int = 2               # morphological opening structuring element (px)
    close_size: int = 3              # morphological closing structuring element (px)
    min_blob_area: int = 64          # discard connected components smaller than this
    fill_bboxes: bool = False        # fill each kept blob's bounding box (region mask)


def blob_detection(spectrogram_db: np.ndarray,
                   params: BlobDetectionParams | None = None,
                   **kwargs) -> np.ndarray:
    """Classic (non-ML) image-processing blob detector.

    Pipeline: estimate a smooth background, threshold the residual, morphological
    open (remove speckle) then close (bridge gaps), connected-component label, and
    keep blobs whose area exceeds ``min_blob_area``. The union of kept blobs is the
    mask. Returns a ``uint8`` mask the same shape as the input.
    """
    p = params or BlobDetectionParams(**kwargs)
    db = power_db_from_tensor(spectrogram_db)

    background = _estimate_background(db, p)
    residual = db - background

    if p.threshold_db is not None:
        thresh = float(p.threshold_db)
    else:
        # robust spread of the residual (MAD -> ~sigma) so a few strong signals do
        # not inflate the threshold the way a plain std would.
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med))) + 1e-6
        thresh = med + p.threshold_k * (1.4826 * mad)
    binary = residual > thresh

    if p.open_size and p.open_size > 0:
        binary = ndimage.binary_opening(binary, structure=_disk(p.open_size))
    if p.close_size and p.close_size > 0:
        binary = ndimage.binary_closing(binary, structure=_disk(p.close_size))

    labels, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(db, dtype=np.uint8)

    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1))
    keep_ids = np.nonzero(sizes >= p.min_blob_area)[0] + 1
    if keep_ids.size == 0:
        return np.zeros_like(db, dtype=np.uint8)

    mask = np.isin(labels, keep_ids)
    if p.fill_bboxes:
        mask = _fill_blob_bboxes(labels, keep_ids, mask.shape)
    return mask.astype(np.uint8)


def _estimate_background(db: np.ndarray, p: BlobDetectionParams) -> np.ndarray:
    """Smooth background estimate the signals sit on top of."""
    if p.background_mode == "global":
        return np.full_like(db, float(np.median(db)))
    if p.background_mode == "uniform":
        # large 2D moving average; signals are small relative to the window so it
        # tracks the noise floor rather than the signals.
        win = max(3, int(p.background_window))
        return ndimage.uniform_filter(db, size=(3, win), mode="reflect").astype(np.float32)
    # default: per-frequency-bin median over time (a per-column noise floor).
    col_floor = np.median(db, axis=0, keepdims=True)
    return np.broadcast_to(col_floor, db.shape).astype(np.float32)


def _disk(radius: int) -> np.ndarray:
    """A small disk structuring element for morphology."""
    r = max(1, int(radius))
    return ndimage.generate_binary_structure(2, 1) if r == 1 else _circular(r)


def _circular(radius: int) -> np.ndarray:
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def _fill_blob_bboxes(labels: np.ndarray, keep_ids: np.ndarray, shape) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    slices = ndimage.find_objects(labels)
    for blob_id in keep_ids:
        sl = slices[blob_id - 1]
        if sl is not None:
            mask[sl] = True
    return mask


# --------------------------------------------------------------------------- #
# Registry: detector_type -> callable(spectrogram_db, **params) -> uint8 mask
# --------------------------------------------------------------------------- #
DETECTORS: dict[str, Callable[..., np.ndarray]] = {
    "3dB_power": moving_average_3db,
    "blob_detection": blob_detection,
}

PARAM_TYPES = {
    "3dB_power": MovingAveragePowerParams,
    "blob_detection": BlobDetectionParams,
}


def run_detector(detector_type: str, spectrogram_db: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Dispatch to a baseline detector by name with a plain param dict."""
    if detector_type not in DETECTORS:
        raise KeyError(f"unknown baseline detector_type {detector_type!r}; "
                       f"choices: {sorted(DETECTORS)}")
    param_obj = PARAM_TYPES[detector_type](**(params or {}))
    return DETECTORS[detector_type](spectrogram_db, param_obj)

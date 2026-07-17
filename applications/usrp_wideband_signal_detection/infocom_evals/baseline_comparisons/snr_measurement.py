#!/usr/bin/env python3
"""Per-signal SNR calibration for the offline detector evaluation.

The batch-eval fact tables index detector performance by *attenuation* (parsed
from the capture stem, e.g. ``attenuation_dB_25`` -> 25 dB). Attenuation is a knob,
not a physical quantity that transfers across signal types: a 20 dB-attenuated
wideband 5G burst and a 20 dB-attenuated narrowband FM tone sit at very different
signal-to-noise ratios. This module converts the attenuation axis into a physically
meaningful **SNR (dB)** axis.

Method (see the eval notes / the design conversation):

  * SNR is measured **once, on the 0 dB capture**, for each signal instance:

      - *peak*  = mean of the top ``peak_top_fraction`` (default 2%) of the linear
        FFT power **inside the signal's bounding box** (its row/time span x its
        freq_lower..freq_upper column band), expressed in dB.
      - *noise* = mean linear FFT power in the **same frequency band**, in a quiet
        time window a few ms **before that burst's Zadoff-Chu (ZC) preamble**
        (default 3 ms -> 1 ms before ``wfgt:zc_sample``), expressed in dB.
      - ``snr0_db = peak_db - noise_db``.

    Per-instance values are aggregated (median) to a calibration keyed by
    ``(signal_class, occupied_bw_hz)``.

  * Every other capture is a **physical attenuator step** on the same emitter, so
    the signal drops by exactly the attenuation while the receiver noise floor is
    unchanged. Hence for any capture at attenuation ``A``::

        snr_db(class, bw, A) = snr0_db(class, bw) - A

    which is far more robust than trying to measure a peak that is buried in noise
    at high attenuation.

The FFT grid here matches the offline eval grid: row r is the ``fft_cols``-point
``fftshift(fft)`` of samples ``[r*fft_cols:(r+1)*fft_cols]`` (== ``samples_per_row``
in the batch manifest), power = ``|.|**2``. Frequencies are baseband, spanning
``[-sr/2, +sr/2)`` across the ``fft_cols`` columns.

Pure numpy; no GPU / Holoscan / pandas. Also defines :class:`SnrResults`, the
serialized (NPZ + JSON) results object the plotting layer reloads so figures can be
re-tweaked without recomputing anything.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

_BYTES_PER_COMPLEX = {"cf32_le": 8, "ci16_le": 4}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class SnrConfig:
    """Knobs for the SNR measurement. Defaults match the agreed method."""

    fft_cols: int = 10240            # freq bins == samples_per_row of the eval grid
    peak_top_fraction: float = 0.02  # peak = mean of the top 2% of in-box power
    noise_pre_zc_start_ms: float = 3.0  # noise window begins this many ms before ZC
    noise_pre_zc_stop_ms: float = 1.0   # noise window ends this many ms before ZC
    max_peak_rows: int = 128         # cap on signal rows read for the peak estimate
    max_instances_per_key: int = 25  # cap on measured instances per (class, bw)
    attenuation_step_db: float = 5.0  # physical step between consecutive captures (doc only)


# --------------------------------------------------------------------------- #
# IQ + spectrum primitives (self-contained; mirror eval_viz's FFT grid)
# --------------------------------------------------------------------------- #
def _read_iq(data_path: Path, start_complex: int, n_complex: int, datatype: str) -> np.ndarray:
    """Read ``n_complex`` complex samples from ``start_complex`` (memmap slice)."""
    if n_complex <= 0 or start_complex < 0:
        return np.zeros(0, dtype=np.complex64)
    if datatype == "cf32_le":
        mm = np.memmap(data_path, dtype=np.complex64, mode="r")
        return np.asarray(mm[start_complex:start_complex + n_complex], dtype=np.complex64)
    if datatype == "ci16_le":
        mm = np.memmap(data_path, dtype=np.int16, mode="r")
        s = start_complex * 2
        flat = np.asarray(mm[s:s + n_complex * 2], dtype=np.float32) / 32767.0
        return (flat[0::2] + 1j * flat[1::2]).astype(np.complex64)
    raise ValueError(f"unsupported SigMF datatype: {datatype}")


def _linear_power(iq: np.ndarray, cols: int) -> np.ndarray:
    """(rows, cols) linear FFT power on the eval grid; drops any partial final row."""
    rows = iq.size // cols
    if rows < 1:
        return np.zeros((0, cols), dtype=np.float64)
    block = iq[:rows * cols].reshape(rows, cols)
    spectrum = np.fft.fftshift(np.fft.fft(block, axis=1), axes=1)
    return (np.abs(spectrum) ** 2).astype(np.float64)


def _band_cols(freq_lower_hz: float, freq_upper_hz: float, span_hz: float, cols: int) -> tuple[int, int]:
    """Column range [c0, c1) for a baseband frequency band on the fftshifted grid."""
    def to_col(f: float) -> int:
        return int(round((f + 0.5 * span_hz) / span_hz * cols))
    c0, c1 = sorted((to_col(freq_lower_hz), to_col(freq_upper_hz)))
    c0 = max(0, min(c0, cols - 1))
    c1 = max(c0 + 1, min(c1, cols))
    return c0, c1


def _ms_to_samples(ms: float, sample_rate_hz: float) -> int:
    return int(round(ms * 1e-3 * sample_rate_hz))


def _to_db(linear_power: float) -> float:
    return float(10.0 * np.log10(max(linear_power, 1e-12)))


# --------------------------------------------------------------------------- #
# Per-signal measurement
# --------------------------------------------------------------------------- #
def measure_annotation_snr(data_path: Path, ann: dict, sample_rate_hz: float,
                           datatype: str, cfg: SnrConfig) -> Optional[dict]:
    """Measure peak / noise / SNR (dB) for one waveform annotation on the 0 dB file.

    Returns ``None`` when the annotation lacks the geometry needed (no ZC sample, or
    a degenerate band), so the caller can skip it.
    """
    zc_sample = ann.get("wfgt:zc_sample")
    if zc_sample is None:
        return None
    cols = cfg.fft_cols
    span_hz = sample_rate_hz  # baseband spans the full sample rate
    c0, c1 = _band_cols(float(ann["core:freq_lower_edge"]),
                        float(ann["core:freq_upper_edge"]), span_hz, cols)

    # --- peak: top-fraction of linear power inside the box ------------------- #
    sample_start = int(ann["core:sample_start"])
    sample_count = int(ann.get("core:sample_count", 0))
    peak_rows = max(1, min(cfg.max_peak_rows, sample_count // cols))
    sig_pow = _linear_power(_read_iq(data_path, sample_start, peak_rows * cols, datatype), cols)
    if sig_pow.shape[0] == 0:
        return None
    box = sig_pow[:, c0:c1].ravel()
    k = max(1, int(round(cfg.peak_top_fraction * box.size)))
    peak_db = _to_db(float(np.mean(np.sort(box)[-k:])))

    # --- noise: same band, quiet window a few ms before the ZC preamble ------ #
    n_start = int(zc_sample) - _ms_to_samples(cfg.noise_pre_zc_start_ms, sample_rate_hz)
    n_stop = int(zc_sample) - _ms_to_samples(cfg.noise_pre_zc_stop_ms, sample_rate_hz)
    n_start = max(0, n_start)
    n_rows = max(1, (n_stop - n_start) // cols)
    noise_pow = _linear_power(_read_iq(data_path, n_start, n_rows * cols, datatype), cols)
    if noise_pow.shape[0] == 0:
        return None
    noise_db = _to_db(float(np.mean(noise_pow[:, c0:c1])))

    return {
        "wfgt_class": str(ann.get("wfgt:class", ann.get("core:label", "unknown"))),
        "occupied_bw_hz": _round_bw(ann.get("wfgt:occupied_bw_hz")),
        "time_group": ann.get("wfgt:time_group"),
        "freq_lower_hz": float(ann["core:freq_lower_edge"]),
        "freq_upper_hz": float(ann["core:freq_upper_edge"]),
        "peak_db": peak_db,
        "noise_db": noise_db,
        "snr0_db": peak_db - noise_db,
    }


def _round_bw(bw) -> Optional[int]:
    try:
        return int(round(float(bw)))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Whole-file calibration
# --------------------------------------------------------------------------- #
def _select_instances(annotations: list[dict], cap: int) -> list[dict]:
    """Evenly-spaced subset of up to ``cap`` annotations (deterministic)."""
    if cap <= 0 or len(annotations) <= cap:
        return annotations
    idx = [round(i * (len(annotations) - 1) / (cap - 1)) for i in range(cap)]
    return [annotations[i] for i in sorted(set(idx))]


def calibrate_from_capture(data_path: Path, meta_path: Path, cfg: SnrConfig = SnrConfig()) -> dict:
    """Build the ``(class, bw) -> snr0_db`` calibration from a 0 dB capture.

    Returns ``{"per_signal": [...], "calibration": [...], "sample_rate_hz": ...,
    "datatype": ...}``. ``calibration`` rows carry the aggregated (median) snr0_db /
    peak_db / noise_db and the instance count per ``(wfgt_class, occupied_bw_hz)``.
    """
    meta = json.loads(Path(meta_path).read_text())
    g = meta.get("global", {})
    sample_rate_hz = float(g.get("core:sample_rate"))
    datatype = str(g.get("core:datatype", "cf32_le"))

    waveforms = [a for a in meta.get("annotations", []) if a.get("wfgt:kind") == "waveform"]

    # group by (class, bw) first so the instance cap is applied per key
    grouped: dict[tuple, list[dict]] = {}
    for a in waveforms:
        key = (str(a.get("wfgt:class", a.get("core:label", "unknown"))), _round_bw(a.get("wfgt:occupied_bw_hz")))
        grouped.setdefault(key, []).append(a)

    per_signal: list[dict] = []
    for key, anns in grouped.items():
        for a in _select_instances(anns, cfg.max_instances_per_key):
            m = measure_annotation_snr(data_path, a, sample_rate_hz, datatype, cfg)
            if m is not None:
                per_signal.append(m)

    calibration: list[dict] = []
    by_key: dict[tuple, list[dict]] = {}
    for m in per_signal:
        by_key.setdefault((m["wfgt_class"], m["occupied_bw_hz"]), []).append(m)
    for (cls, bw), rows in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        calibration.append({
            "wfgt_class": cls,
            "occupied_bw_hz": bw,
            "snr0_db": float(np.median([r["snr0_db"] for r in rows])),
            "peak_db": float(np.median([r["peak_db"] for r in rows])),
            "noise_db": float(np.median([r["noise_db"] for r in rows])),
            "n_instances": len(rows),
        })

    return {
        "per_signal": per_signal,
        "calibration": calibration,
        "sample_rate_hz": sample_rate_hz,
        "datatype": datatype,
    }


def calibration_lookup(calibration: list[dict]) -> dict[tuple, float]:
    """``(wfgt_class, occupied_bw_hz) -> snr0_db`` dict from a calibration list."""
    return {(c["wfgt_class"], c["occupied_bw_hz"]): c["snr0_db"] for c in calibration}


def snr_at_attenuation(snr0_db: float, attenuation_db: float) -> float:
    """Physical attenuator identity: SNR falls 1:1 with added attenuation."""
    return snr0_db - attenuation_db


# --------------------------------------------------------------------------- #
# Serialized results object (NPZ arrays + JSON sidecar)
# --------------------------------------------------------------------------- #
@dataclass
class SnrResults:
    """Everything the SNR plots need, reloadable so figures can be re-tweaked
    without recomputing the calibration or re-reading the fact tables.

    ``region`` / ``frame`` are column-oriented dicts (name -> 1-D np.ndarray), a
    superset of the eval fact tables joined with an ``snr_db`` column (and
    ``frame_snr_db`` for frames). ``calibration`` is the per-(class, bw) table.
    ``params`` / ``provenance`` record how it was built.
    """

    region: dict = field(default_factory=dict)
    frame: dict = field(default_factory=dict)
    calibration: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def save(self, base_path) -> dict:
        """Write ``<base>.npz`` (region+frame columns) + ``<base>.json`` (sidecar).

        ``base_path`` may end in ``.npz``/``.json`` or be a bare stem; both files are
        written next to it. Returns the two paths.
        """
        base = Path(base_path)
        if base.suffix in (".npz", ".json"):
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for col, arr in self.region.items():
            arrays[f"region::{col}"] = np.asarray(arr)
        for col, arr in self.frame.items():
            arrays[f"frame::{col}"] = np.asarray(arr)
        npz_path = base.with_suffix(".npz")
        json_path = base.with_suffix(".json")
        np.savez(npz_path, **arrays)
        json_path.write_text(json.dumps({
            "region_columns": list(self.region.keys()),
            "frame_columns": list(self.frame.keys()),
            "calibration": self.calibration,
            "params": self.params,
            "provenance": self.provenance,
        }, indent=2))
        return {"npz": npz_path, "json": json_path}

    @classmethod
    def load(cls, base_path) -> "SnrResults":
        base = Path(base_path)
        if base.suffix in (".npz", ".json"):
            base = base.with_suffix("")
        sidecar = json.loads(base.with_suffix(".json").read_text())
        npz = np.load(base.with_suffix(".npz"), allow_pickle=False)
        region = {col: npz[f"region::{col}"] for col in sidecar["region_columns"]}
        frame = {col: npz[f"frame::{col}"] for col in sidecar["frame_columns"]}
        return cls(region=region, frame=frame,
                   calibration=sidecar.get("calibration", []),
                   params=sidecar.get("params", {}),
                   provenance=sidecar.get("provenance", {}))

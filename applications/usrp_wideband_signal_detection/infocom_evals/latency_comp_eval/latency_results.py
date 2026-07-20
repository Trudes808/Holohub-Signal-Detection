#!/usr/bin/env python3
"""Reloadable results object for the per-frame latency + compute-load eval.

Mirrors ``baseline_comparisons/snr_measurement.py``'s ``SnrResults`` serialization
contract (flat ``.npz`` with prefixed keys + a ``.json`` sidecar of column names /
geometry / provenance) so re-styling or re-plotting needs no recompute.

Two tables:
  * ``cells``   -- one row per (detector, sample_rate, device): the scalar per-cell
                   summary (mean/median/pXX latency, GFLOPs, peak GPU MB, geometry).
  * ``samples`` -- flat long form of every timed latency sample: ``cell_index`` (into
                   the cells table) + ``latency_ms``. This is what the histograms bin.

Plus ``geometry`` (per-rate FFT size / samples-per-frame / real-time budget) and
``provenance`` (capture, params, host).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _to_npz_arrays(prefix: str, columns: dict) -> dict:
    out = {}
    for k, v in columns.items():
        arr = np.asarray(v)
        if arr.dtype == object or arr.dtype.kind in ("U", "S"):
            arr = arr.astype(str)
        out[f"{prefix}::{k}"] = arr
    return out


def _from_npz(npz, prefix: str) -> dict:
    cols = {}
    plen = len(prefix) + 2
    for key in npz.files:
        if key.startswith(prefix + "::"):
            cols[key[plen:]] = npz[key]
    return cols


@dataclass
class LatencyResults:
    cells: dict = field(default_factory=dict)       # column-oriented, len = n_cells
    samples: dict = field(default_factory=dict)     # {cell_index[], latency_ms[]}
    geometry: dict = field(default_factory=dict)    # {sample_rate_hz: {...}} (str keys in json)
    params: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    # ---- convenience accessors --------------------------------------------- #
    @property
    def cell_columns(self) -> list[str]:
        return list(self.cells.keys())

    @property
    def n_cells(self) -> int:
        return len(next(iter(self.cells.values()))) if self.cells else 0

    def detectors(self) -> list[str]:
        return sorted(set(np.asarray(self.cells["detector"]).tolist()))

    def sample_rates(self) -> list[float]:
        return sorted(set(np.asarray(self.cells["sample_rate_hz"], dtype=float).tolist()))

    def latency_samples(self, detector: str, sample_rate_hz: float, device: str) -> np.ndarray:
        """All raw latency_ms samples for one (detector, rate, device) cell."""
        det = np.asarray(self.cells["detector"]).astype(str)
        rate = np.asarray(self.cells["sample_rate_hz"], dtype=float)
        dev = np.asarray(self.cells["device"]).astype(str)
        hit = np.nonzero((det == detector) &
                         (np.isclose(rate, float(sample_rate_hz))) &
                         (dev == device))[0]
        if hit.size == 0:
            return np.zeros(0)
        ci = int(hit[0])
        sidx = np.asarray(self.samples["cell_index"], dtype=int)
        lat = np.asarray(self.samples["latency_ms"], dtype=float)
        return lat[sidx == ci]

    def budget_ms(self, sample_rate_hz: float) -> float:
        return float(self.geometry[str(int(sample_rate_hz))]["frame_budget_ms"])

    # ---- serialization ----------------------------------------------------- #
    def save(self, base) -> tuple[Path, Path]:
        base = Path(base)
        npz_path = base.with_suffix(".npz")
        json_path = base.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        arrays.update(_to_npz_arrays("cell", self.cells))
        arrays.update(_to_npz_arrays("sample", self.samples))
        np.savez(npz_path, **arrays)
        json_path.write_text(json.dumps({
            "cell_columns": self.cell_columns,
            "sample_columns": list(self.samples.keys()),
            "geometry": self.geometry,
            "params": self.params,
            "provenance": self.provenance,
        }, indent=2, default=str))
        return npz_path, json_path

    @classmethod
    def load(cls, base) -> "LatencyResults":
        base = Path(base)
        npz_path = base if base.suffix == ".npz" else base.with_suffix(".npz")
        json_path = base.with_suffix(".json") if base.suffix in (".npz", ".json") \
            else base.with_suffix(".json")
        meta = json.loads(json_path.read_text())
        with np.load(npz_path, allow_pickle=False) as npz:
            cells = _from_npz(npz, "cell")
            samples = _from_npz(npz, "sample")
        return cls(cells=cells, samples=samples,
                   geometry=meta.get("geometry", {}),
                   params=meta.get("params", {}),
                   provenance=meta.get("provenance", {}))

#!/usr/bin/env python3
"""Turn IQ recordings in (almost) any container into SigMF pairs (the soft-label pipeline's input stage).

    python3 to_sigmf.py <file-or-dir> [...] --out <dir> [--recursive] [--rate 25e6 --fc 2.44e9 --dtype cf32_le ...]

Inputs:

  *.sigmf-meta          SigMF 1.x, and pre-1.0 files wrapped in "_metadata" (ORACLE); string numbers, centre
                        frequencies written in MHz (LoRaIQ) and an RF centre kept in wfgt:rf_center_freq_hz (ARGUS)
                        are fixed up
  raw binaries          .bin .dat .raw .iq .cfile .cf32 .cf64 .cs16 .sc16 .cs8 .cu8 ...; the sample format comes from
                        --dtype, a known-dataset preset, a JSON sidecar (<stem>.json, or a metadata.json listing the
                        file by name) or the extension, and is guessed from the bytes (with a note) as a last resort
  *.npy / *.npz         complex arrays, or real arrays holding I/Q in a last axis of 2 or interleaved along the last axis
  *.mat                 MATLAB v5 (scipy) or v7.3 (h5py); the rate is read from a scalar Fs / fs / sample_rate
  *.h5 / *.hdf5         any dataset (h5py); the rate from its attributes when present
  *.pkl / *.pickle      dicts / lists of numpy arrays only; a restricted unpickler refuses anything else
  *.wav                 2-channel IQ WAV (SDR#, SDR++, GQRX ...); the centre from a "<n>Hz" token in the file name

Arrays with leading dimensions (N clips x L samples, channels x pulses x samples, ...) become one recording with one
SigMF capture segment per clip.

Output: <out>/<path relative to the input>/<stem>.sigmf-meta + .sigmf-data. When the original bytes already are a SigMF
sample format (cf32_le, cf64_le, ci16_le, ci8) the .sigmf-data is a symlink to the original file (no copy; --copy to
copy, --cf32 to always write real cf32_le files, e.g. for the container detectors); otherwise the samples are written
out as cf32_le (ci16_le / ci8 for integer I/Q). The source's own annotations are not carried over (their count and the
source path are recorded in the global block). The source files are never modified.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

TOOL = "to_sigmf.py 1.0 (Holohub soft_label_pipeline)"
SIGMF_VERSION = "1.2.0"

# SigMF sample formats readable in place: numpy dtype of one component, components per sample
DIRECT = {"cf32_le": ("<f4", 2), "cf64_le": ("<f8", 2), "ci16_le": ("<i2", 2), "ci8": ("i1", 2)}
# everything this tool can read from a raw file (cu8 = RTL-SDR offset binary, *_be = big endian)
READABLE = dict(DIRECT, **{"cu8": ("u1", 2), "cf32_be": (">f4", 2), "cf64_be": (">f8", 2), "ci16_be": (">i2", 2),
                           "ci32_le": ("<i4", 2)})
DTYPE_ALIASES = {
    "cf32": "cf32_le", "fc32": "cf32_le", "complex64": "cf32_le", "c8": "cf32_le", "float32": "cf32_le", "cfile": "cf32_le",
    "cf64": "cf64_le", "fc64": "cf64_le", "complex128": "cf64_le", "c16": "cf64_le", "float64": "cf64_le",
    "ci16": "ci16_le", "cs16": "ci16_le", "sc16": "ci16_le", "int16": "ci16_le", "s16": "ci16_le",
    "ci8_le": "ci8", "cs8": "ci8", "sc8": "ci8", "int8": "ci8", "s8": "ci8",
    "cu8_le": "cu8", "uint8": "cu8", "u8": "cu8", "ci32": "ci32_le", "int32": "ci32_le",
}
RAW_EXT_DTYPE = {".cfile": "cf32_le", ".cf32": "cf32_le", ".fc32": "cf32_le", ".complex64": "cf32_le",
                 ".cf64": "cf64_le", ".fc64": "cf64_le", ".complex128": "cf64_le",
                 ".cs16": "ci16_le", ".sc16": "ci16_le", ".ci16": "ci16_le", ".s16": "ci16_le", ".int16": "ci16_le",
                 ".cs8": "ci8", ".sc8": "ci8", ".ci8": "ci8", ".s8": "ci8", ".int8": "ci8", ".cu8": "cu8", ".u8": "cu8"}
RAW_EXTS = set(RAW_EXT_DTYPE) | {".bin", ".dat", ".raw", ".iq", ".data", ".sigmf-data"}
ARRAY_EXTS = {".npy", ".npz", ".mat", ".h5", ".hdf5", ".he5", ".pkl", ".pickle", ".wav"}

# Known datasets (matched as a substring of the resolved input path, first match wins). Values here are used only
# when the file itself does not say; command-line options override everything. None = undocumented upstream.
PRESETS = [
    dict(name="DroneDetect_V2", match="DroneDetect_V2", rate=60e6, fc=2.4375e9, dtype="cf32_le",
         note="IEEE DataPort 69350: BladeRF 60 MS/s at 2.4375 GHz (dataset description; files carry no metadata)"),
    dict(name="T-PRIME SR20MHZ", match="T-PRIME/DATASET3_3/SR20MHZ", rate=20e6, fc=2.442e9, dtype="cf64_le",
         note="T-PRIME: raw complex128; receiver centre assumed at Tx1 (2.442 GHz)"),
    dict(name="T-PRIME SR625MHZ", match="T-PRIME/DATASET3_3/SR625MHZ", rate=62.5e6, fc=2.442e9, dtype="cf64_le",
         note="T-PRIME: raw complex128; receiver centre assumed at Tx1 (2.442 GHz)"),
    dict(name="T-PRIME", match="T-PRIME", rate=31.25e6, fc=2.442e9, dtype="cf64_le",
         note="T-PRIME DATASET3_0: raw complex128, receiver 31.25 MS/s at 2.442 GHz"),
    dict(name="ORACLE", match="ORACLE", dtype="cf64_le",
         note="ORACLE: the metadata says cf32 but the samples are complex128 (16 bytes each)"),
    dict(name="GENESYS_CBRS", match="GENESYS_CBRS", fc_if_zero=3.6e9,
         note="GENESYS_CBRS: metadata centre is 0 (baseband); the recording is at the 3.6 GHz CBRS TX frequency"),
    dict(name="x410_recordings", match="x410_recordings", fc=915e6,
         note="HCRO X410 captures: 900 MHz ISM, centred on 915 MHz; rate/format from the class metadata.json"),
    dict(name="WiSig", match="WiSig", rate=25e6, fc=2.462e9, iq_layout="pairs",
         note="WiSig: 256-sample WiFi packet preambles (25 MS/s at 2.462 GHz per the paper)"),
    dict(name="RadarML", match="RadarML", iq_layout="interleaved",
         note="RadarML: int16 (channels, pulses, samples); I/Q interleaving and sample rate are undocumented upstream"),
    dict(name="RadSeg", match="RadSeg", var="signals", note="RadSeg: complex128 clips; sample rate undocumented"),
    dict(name="RFChallenge", match="RFChallenge", fc=0.0, note="RFChallenge: baseband mixtures; sample rate unspecified"),
]

RATE_KEYS = ("core:sample_rate", "sample_rate_hz", "sample_rate", "samp_rate", "sampleRate", "outputSampleRateHz",
             "SampleRate", "fs", "Fs", "FS", "rate")
FC_KEYS = ("core:frequency", "core:center_frequency", "center_frequency_hz", "center_frequency", "centerFrequencyHz",
           "center_freq", "centre_frequency", "rf_center_freq_hz", "frequency", "fc", "Fc", "freq")
DTYPE_KEYS = ("core:datatype", "datatype", "sample_format", "data_type", "dtype")
LABEL_KEYS = ("signal_type", "core:protocol", "class", "label", "core:label")
# macOS junk, side-car arrays, derived masks and this pipeline's own outputs
SKIP_NAMES = re.compile(r"(^\._)|(^\.DS_Store$)|(_metadata\.npy$)|(\.masks\.npz$)|(\.packed\.npz$)|"
                        r"(_softlabel\.sigmf-(meta|data)$)|(_snipped\.sigmf-meta$)")
# folders of derived artefacts next to recordings (detector masks, figures, caches)
SKIP_DIRS = {"masks", "mask_arrays", "gt_masks", "overlays", "panels", "figures", "__pycache__", ".git",
             ".download_logs", "checkpoints"}


class ConvertError(Exception):
    pass


@dataclass
class Overrides:
    rate: float | None = None
    fc: float | None = None
    dtype: str | None = None
    iq_layout: str = "auto"          # auto | pairs | interleaved | planar | real
    var: str | None = None
    index: tuple | None = None       # select arr[index] on the leading axes before splitting into clips
    max_segments: int | None = None
    copy: bool = False
    force_cf32: bool = False         # always write a real cf32_le .sigmf-data (container detectors need it)


@dataclass
class Converted:
    meta_path: Path
    source: Path
    rate: float
    fc: float
    dtype: str
    n_samples: int
    n_segments: int
    source_dtype: str = ""
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------------------------------------- helpers
def norm_dtype(s: str | None) -> str | None:
    if not s:
        return None
    s = str(s).strip().lower()
    return DTYPE_ALIASES.get(s, s)


def _num(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    if isinstance(x, np.ndarray) and x.size == 1:
        return float(np.real(x.reshape(-1)[0]))
    return None


def _find_key(obj, keys, depth=0):
    """First value for any of `keys` in a nested dict/list."""
    if depth > 4:
        return None
    if isinstance(obj, dict):
        for k in keys:
            if k in obj and not isinstance(obj[k], (dict, list)):
                return obj[k]
        for v in obj.values():
            if isinstance(v, (dict, list)):
                r = _find_key(v, keys, depth + 1)
                if r is not None:
                    return r
    elif isinstance(obj, list):
        for v in obj[:4]:
            r = _find_key(v, keys, depth + 1)
            if r is not None:
                return r
    return None


def preset_for(path: Path) -> dict:
    s = str(path.resolve())
    for p in PRESETS:
        if p["match"] in s:
            return p
    return {}


def fix_fc(fc: float | None, notes: list) -> float | None:
    """Centre frequencies below 10 kHz that are not 0 are almost certainly MHz written as Hz (LoRaIQ: 862.5)."""
    if fc is not None and 0 < abs(fc) < 1e4:
        notes.append(f"centre frequency {fc} read as MHz -> {fc * 1e6:.0f} Hz")
        return fc * 1e6
    return fc


def parse_hz_from_name(name: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*([kKmMgG]?)Hz", name)
    if not m:
        return None
    return float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6, "g": 1e9}[m.group(2).lower()]


def sidecar_for(path: Path) -> tuple[dict | None, Path | None]:
    """JSON sidecar describing a raw file: <stem>.json / <name>.json, or a metadata.json that lists it by file name.
    For a symlink, the folder of the file it points to is searched too."""
    if path.is_symlink() and path.resolve() != path:
        side = _sidecar_in(path)
        return side if side[0] is not None else _sidecar_in(path.resolve())
    return _sidecar_in(path)


def _sidecar_in(path: Path) -> tuple[dict | None, Path | None]:
    for cand in (path.with_suffix(".json"), path.with_name(path.name + ".json")):
        if cand.is_file():
            try:
                return json.loads(cand.read_text()), cand
            except (OSError, json.JSONDecodeError):
                pass
    for name in ("metadata.json", "meta.json", "index.json"):
        cand = path.parent / name
        if cand.is_file():
            try:
                d = json.loads(cand.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            recs = d if isinstance(d, list) else (d.get("files") or d.get("records") or []) if isinstance(d, dict) else []
            for r in recs:
                if isinstance(r, dict) and Path(str(r.get("file_name", r.get("file", r.get("filename", ""))))).name == path.name:
                    return r, cand
    return None, None


def guess_raw_dtype(path: Path) -> str:
    """Pick cf32 / cf64 / ci16 from the bytes: a float layout read with the wrong width gives values with wild exponents."""
    size = path.stat().st_size
    chunk = 1 << 16
    offsets = [int(size * f) // 16 * 16 for f in (0.25, 0.5, 0.75)] if size > 4 * chunk else [0]
    for name, np_t in (("cf32_le", "<f4"), ("cf64_le", "<f8")):
        ok = True
        for off in offsets:
            v = np.fromfile(path, dtype=np_t, count=chunk // np.dtype(np_t).itemsize, offset=off)
            v = v[np.isfinite(v)] if v.size else v
            nz = np.abs(v[v != 0])
            if nz.size < 16:
                continue
            if np.mean((nz > 1e6) | (nz < 1e-12)) > 0.01:
                ok = False
                break
        if ok:
            return name
    return "ci16_le"


def read_raw(path: Path, dtype: str, start: int = 0, count: int = -1) -> np.ndarray:
    """Samples [start, start+count) of a raw interleaved-I/Q file as complex64."""
    comp_t, per = READABLE[dtype]
    it = np.dtype(comp_t).itemsize
    raw = np.fromfile(path, dtype=comp_t, count=-1 if count < 0 else count * per, offset=start * per * it)
    raw = raw[: (raw.size // 2) * 2]
    if dtype == "cu8":
        x = (raw.astype(np.float32) - 127.5) / 127.5
    elif dtype.startswith("ci"):
        # same expression (complex first, then the division) as the fine-tuning tools' load_iq: the detector is
        # sensitive enough near its threshold that a last-bit difference in the samples flips mask pixels
        return ((raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32))
                / float(np.iinfo(raw.dtype).max)).astype(np.complex64)
    else:
        x = raw.astype(np.float32)
    return (x[0::2] + 1j * x[1::2]).astype(np.complex64)


def bytes_per_sample(dtype: str) -> int:
    comp_t, per = READABLE[dtype]
    return np.dtype(comp_t).itemsize * per


# ----------------------------------------------------------------------------------------------------------- arrays
class _SafeUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"), ("numpy", "ndarray"),
               ("numpy", "dtype"), ("collections", "OrderedDict"), ("builtins", "complex")}

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} (only plain numpy arrays are allowed)")


def _arrays_in(obj, prefix="", depth=0):
    """(name, array) for every numeric array in a nested dict/list/tuple; lists of same-tail-shape arrays are joined."""
    if depth > 4:
        return
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in "biufc" and obj.size > 1:
            yield prefix or "data", obj
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _arrays_in(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
    elif isinstance(obj, (list, tuple)) and obj:
        arrs = [a for a in obj if isinstance(a, np.ndarray) and a.dtype.kind in "biufc"]
        if len(arrs) == len(obj) and len({a.shape[1:] for a in arrs}) == 1 and len({a.dtype for a in arrs}) == 1:
            yield prefix or "data", ("concat", arrs)
        else:
            for i, v in enumerate(obj[:64]):
                yield from _arrays_in(v, f"{prefix}[{i}]", depth + 1)


def _shape_of(a):
    if isinstance(a, tuple) and a[0] == "concat":
        return (sum(x.shape[0] for x in a[1]),) + a[1][0].shape[1:]
    if isinstance(a, tuple):
        return tuple(a[1].shape)
    return tuple(a.shape)


def _dtype_of(a):
    if isinstance(a, tuple) and a[0] == "concat":
        return a[1][0].dtype
    if isinstance(a, tuple):
        return a[1].dtype
    return a.dtype


def _pick_array(cands: list, var: str | None):
    if not cands:
        raise ConvertError("no numeric array found")
    if var:
        for n, a in cands:
            if n == var or n.split(".")[-1] == var or n.endswith("/" + var):
                return n, a
        raise ConvertError(f"variable {var!r} not found; arrays: {[n for n, _ in cands]}")
    cplx = [(n, a) for n, a in cands if _dtype_of(a).kind == "c"]
    return max(cplx or cands, key=lambda na: int(np.prod(_shape_of(na[1]))))


def load_container(path: Path, ov: Overrides) -> tuple[str, object, float | None, float | None, list]:
    """-> (array name, array-like, rate or None, fc or None, notes) for npy/npz/mat/h5/pkl/wav."""
    ext = path.suffix.lower()
    notes: list = []
    rate = fc = None
    if ext == ".npy":
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        if a.dtype.kind not in "biufc":
            raise ConvertError(f"array dtype {a.dtype} is not numeric (side-car metadata?)")
        return path.stem, a, None, None, notes
    if ext == ".npz":
        z = np.load(path, allow_pickle=False)
        cands = [(k, z[k]) for k in z.files if z[k].dtype.kind in "biufc" and z[k].size > 1]
        for k in z.files:
            if z[k].size == 1 and k in RATE_KEYS:
                rate = _num(z[k])
            if z[k].size == 1 and k in FC_KEYS:
                fc = _num(z[k])
        n, a = _pick_array(cands, ov.var)
        return n, a, rate, fc, notes
    if ext == ".wav":
        from scipy.io import wavfile
        sr, a = wavfile.read(path, mmap=True)
        if a.ndim != 2 or a.shape[1] != 2:
            raise ConvertError("WAV is not 2-channel I/Q")
        notes.append("2-channel WAV read as I (left) / Q (right)")
        return "wav", ("wav", a), float(sr), parse_hz_from_name(path.name), notes
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = _SafeUnpickler(f).load()
        n, a = _pick_array(list(_arrays_in(obj)), ov.var)
        if isinstance(obj, dict):
            rate = _num(_find_key(obj, RATE_KEYS))
            fc = _num(_find_key(obj, FC_KEYS))
        return n, a, rate, fc, notes
    if ext == ".mat":
        try:
            from scipy.io import loadmat
            d = loadmat(path, squeeze_me=False)
        except NotImplementedError:      # v7.3 = HDF5
            return _load_h5(path, ov, notes, mat73=True)
        cands = [(k, v) for k, v in d.items() if not k.startswith("__") and isinstance(v, np.ndarray)
                 and v.dtype.kind in "biufc" and v.size > 1]
        scal = {k: v for k, v in d.items() if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "iuf"}
        rate = next((_num(scal[k]) for k in RATE_KEYS if k in scal), None)
        fc = next((_num(scal[k]) for k in FC_KEYS if k in scal), None)
        n, a = _pick_array(cands, ov.var)
        return n, a, rate, fc, notes
    if ext in (".h5", ".hdf5", ".he5"):
        return _load_h5(path, ov, notes)
    raise ConvertError(f"unsupported container {ext}")


def _load_h5(path: Path, ov: Overrides, notes: list, mat73: bool = False):
    try:
        import h5py
    except ImportError as e:
        raise ConvertError("reading HDF5 / MATLAB v7.3 needs h5py (pip install h5py)") from e
    f = h5py.File(path, "r")
    cands, rate, fc = [], None, None
    for k, v in f.attrs.items():
        if k in RATE_KEYS:
            rate = _num(v)
        if k in FC_KEYS:
            fc = _num(v)

    def visit(name, obj):
        nonlocal rate, fc
        if not isinstance(obj, h5py.Dataset):
            return
        dt = obj.dtype
        if dt.names and set(dt.names) >= {"real", "imag"}:      # MATLAB v7.3 complex
            cands.append((name, ("h5cplx", obj)))
        elif dt.kind in "biufc" and obj.size > 1:
            cands.append((name, obj))
        elif obj.size == 1 and dt.kind in "iuf":
            if name.split("/")[-1] in RATE_KEYS:
                rate = _num(obj[()])
            if name.split("/")[-1] in FC_KEYS:
                fc = _num(obj[()])
        for k, v in obj.attrs.items():
            if k in RATE_KEYS and rate is None:
                rate = _num(v)
            if k in FC_KEYS and fc is None:
                fc = _num(v)
    f.visititems(visit)
    if mat73:
        notes.append("MATLAB v7.3 (HDF5): arrays are stored transposed")
    if not cands:
        raise ConvertError("no numeric dataset in the HDF5 file")
    n, a = _pick_array(cands, ov.var)
    return n, a, rate, fc, notes


class ArrayView:
    """Uniform clip access to an array-like: clip_native(i) over the leading axes flattened."""

    def __init__(self, a, layout: str, index: tuple | None, notes: list):
        self.kind = "plain"
        if isinstance(a, tuple) and a[0] == "concat":
            self.parts, self.kind = a[1], "concat"
            shape, dt = _shape_of(a), _dtype_of(a)
        elif isinstance(a, tuple) and a[0] == "wav":
            self.a, self.kind = a[1], "wav"
            shape, dt = tuple(a[1].shape), a[1].dtype
        elif isinstance(a, tuple) and a[0] == "h5cplx":
            self.a, self.kind = a[1], "h5cplx"
            shape, dt = tuple(a[1].shape), np.dtype(np.complex64)
        else:
            if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[-1] == 1:
                a = a[..., 0]            # MATLAB column vector (N, 1)
            self.a = a
            shape, dt = tuple(a.shape), a.dtype
        if self.kind == "wav":
            self.layout, self.lead, self.L, self.n_clips, self.index = "wav", (), shape[0], 1, ()
            self.out_dtype = "ci16_le" if dt == np.int16 else "cf32_le"
            return
        self.index = tuple(index) if index else ()
        shape = shape[len(self.index):]
        if dt.kind == "c":
            self.layout = "complex"
            lead, L = shape[:-1], shape[-1]
            self.out_dtype = "cf32_le"
        else:
            lay = layout
            if lay == "auto":
                if shape[-1] == 2:
                    lay = "pairs"
                elif len(shape) >= 2 and shape[-2] == 2:
                    lay = "planar"
                elif dt.kind == "i" and shape[-1] % 2 == 0:
                    lay = "interleaved"
                    notes.append("integer array: I/Q assumed interleaved along the last axis (--iq-layout to change)")
                else:
                    raise ConvertError(f"real array {shape} {dt}: say how I/Q is stored with --iq-layout "
                                       "pairs|interleaved|planar|real")
            self.layout = lay
            if lay == "pairs":
                lead, L = shape[:-2], shape[-2]
            elif lay == "planar":
                lead, L = shape[:-2], shape[-1]
            elif lay == "interleaved":
                lead, L = shape[:-1], shape[-1] // 2
            elif lay == "real":
                lead, L = shape[:-1], shape[-1]
                notes.append("real-valued samples: the spectrum is mirrored around 0 Hz")
            else:
                raise ConvertError(f"unknown --iq-layout {lay}")
            ints = lay != "real" and dt in (np.int16, np.int8)
            self.out_dtype = ("ci16_le" if dt == np.int16 else "ci8") if ints else "cf32_le"
        self.lead = tuple(lead)
        self.L = int(L)
        self.n_clips = int(np.prod(self.lead)) if self.lead else 1

    def clip_index(self, i: int) -> tuple:
        return tuple(int(v) for v in np.unravel_index(i, self.lead)) if self.lead else ()

    def _raw(self, i: int) -> np.ndarray:
        idx = self.index + self.clip_index(i)
        if self.kind == "concat":
            k = idx[0]
            for p in self.parts:
                if k < p.shape[0]:
                    return np.asarray(p[(k,) + idx[1:]])
                k -= p.shape[0]
            raise IndexError(i)
        if self.kind == "h5cplx":
            r = self.a[idx] if idx else self.a[()]
            return (r["real"] + 1j * r["imag"]).astype(np.complex64)
        return np.asarray(self.a[idx] if idx else self.a[()])

    def clip_native(self, i: int) -> np.ndarray:
        """The clip in the output sample format (interleaved components for ci16/ci8, complex64 otherwise)."""
        if self.kind == "wav":
            a = np.asarray(self.a)
            if self.out_dtype == "ci16_le":
                return np.ascontiguousarray(a, dtype="<i2").reshape(-1)
            sc = 1.0 / float(np.iinfo(a.dtype).max) if a.dtype.kind in "iu" else 1.0
            return (a[:, 0].astype(np.float32) * sc + 1j * a[:, 1].astype(np.float32) * sc).astype(np.complex64)
        r = self._raw(i)
        if self.layout == "complex":
            return r.astype(np.complex64).reshape(-1)
        if self.layout == "pairs":
            c = (r[..., 0], r[..., 1])
        elif self.layout == "planar":
            c = (r[..., 0, :], r[..., 1, :])
        elif self.layout == "interleaved":
            c = (r[..., 0::2], r[..., 1::2])
        else:
            return r.astype(np.float32).astype(np.complex64).reshape(-1)
        if self.out_dtype in ("ci16_le", "ci8"):
            out = np.empty(c[0].size * 2, dtype="<i2" if self.out_dtype == "ci16_le" else "i1")
            out[0::2] = c[0].reshape(-1)
            out[1::2] = c[1].reshape(-1)
            return out
        return (c[0].astype(np.float32) + 1j * c[1].astype(np.float32)).astype(np.complex64).reshape(-1)


# ----------------------------------------------------------------------------------------------------------- writers
def _write_meta(path: Path, global_: dict, captures: list):
    path.write_text(json.dumps({"global": global_, "captures": captures, "annotations": []}, indent=2) + "\n")


def _unlink(p: Path):
    if p.exists() or p.is_symlink():
        p.unlink()


def _place_data(src: Path, dtype: str, dst: Path, ov: Overrides) -> str:
    """Put the samples at dst: symlink (default), copy, or convert to cf32_le. Returns the dtype of dst."""
    _unlink(dst)
    if ov.force_cf32 and dtype != "cf32_le":
        _stream_convert(src, dtype, dst)
        return "cf32_le"
    if dtype in DIRECT:
        if ov.copy or ov.force_cf32:
            shutil.copyfile(src, dst)
        else:
            os.symlink(src.resolve(), dst)
        return dtype
    _stream_convert(src, dtype, dst)
    return "cf32_le"


def _stream_convert(src: Path, dtype: str, dst: Path, chunk: int = 1 << 22):
    _unlink(dst)
    n = src.stat().st_size // bytes_per_sample(dtype)
    with open(dst, "wb") as f:
        for s in range(0, n, chunk):
            read_raw(src, dtype, s, min(chunk, n - s)).tofile(f)


def _global(rate: float, dtype: str, source: Path, fmt: str, notes: list, extra: dict | None = None) -> dict:
    g = {"core:datatype": dtype, "core:sample_rate": float(rate), "core:version": SIGMF_VERSION, "core:num_channels": 1,
         "argus:source_file": str(source.resolve()), "argus:source_format": fmt, "argus:converted_by": TOOL}
    if notes:
        g["argus:conversion_notes"] = list(notes)
    if extra:
        g.update({k: v for k, v in extra.items() if v is not None})
    return g


def _require_rate(rate, path: Path, preset: dict):
    if rate is None or not (rate > 0):
        hint = f" ({preset['note']})" if preset.get("note") else ""
        raise ConvertError(f"sample rate unknown for {path.name}{hint}: pass --rate")


def _stem_of(path: Path) -> str:
    n = path.name
    for suf in (".sigmf-meta", ".sigmf-data"):
        if n.endswith(suf):
            return n[: -len(suf)]
    return path.stem


def convert_sigmf(meta_path: Path, out_dir: Path, ov: Overrides, out_stem: str | None = None) -> Converted:
    d = json.loads(meta_path.read_text())
    notes: list = []
    fmt = "SigMF"
    if "_metadata" in d and "global" not in d:
        d = d["_metadata"]
        fmt = "SigMF pre-1.0 (_metadata)"
        notes.append("pre-1.0 SigMF metadata (wrapped in _metadata) normalised")
    g = d.get("global", {})
    caps = d.get("captures") or [{}]
    if isinstance(caps, dict):
        caps = [caps]
    preset = preset_for(meta_path)
    stem = _stem_of(meta_path)
    data = meta_path.with_name(stem + ".sigmf-data")
    if not data.is_file():
        raise ConvertError(f"missing {data.name}")
    rate = ov.rate or _num(g.get("core:sample_rate"))
    _require_rate(rate, meta_path, preset)
    declared = norm_dtype(g.get("core:datatype"))
    dtype = norm_dtype(ov.dtype) or norm_dtype(preset.get("dtype")) or declared or "cf32_le"
    if declared and dtype != declared:
        notes.append(f"datatype {g.get('core:datatype')} in the metadata corrected to {dtype}"
                     + (f" ({preset['note']})" if preset.get("note") and not ov.dtype else ""))
    if dtype not in READABLE:
        raise ConvertError(f"unsupported SigMF datatype {dtype}")
    if any(int(_num(c.get("core:header_bytes")) or 0) for c in caps):
        raise ConvertError("captures with core:header_bytes are not supported")
    segs = []
    for i, c in enumerate(caps):
        s0 = int(_num(c.get("core:sample_start")) or 0)
        f = _num(c.get("wfgt:rf_center_freq_hz"))
        if f is not None:
            if i == 0:
                notes.append("RF centre taken from wfgt:rf_center_freq_hz (core:frequency is the baseband 0)")
        else:
            f = _num(c.get("core:frequency", c.get("frequency", c.get("core:center_frequency"))))
        if f is None:
            f = _num(g.get("core:frequency"))
        segs.append([s0, f, c.get("core:datetime", c.get("core:time"))])
    fc0 = segs[0][1]
    for k, s in enumerate(segs):
        if ov.fc is not None:
            s[1] = ov.fc
            continue
        s[1] = fix_fc(s[1], notes if k == 0 else [])
        if s[1] in (None, 0.0) and preset.get("fc_if_zero"):
            if k == 0:
                notes.append(preset["note"])
            s[1] = preset["fc_if_zero"]
        if s[1] is None and preset.get("fc") is not None:
            s[1] = preset["fc"]
    out_dir.mkdir(parents=True, exist_ok=True)
    o = out_stem or stem
    out_meta, out_data = out_dir / f"{o}.sigmf-meta", out_dir / f"{o}.sigmf-data"
    out_dtype = _place_data(data, dtype, out_data, ov)
    n_total = out_data.stat().st_size // bytes_per_sample(out_dtype)
    extra = {k: g[k] for k in ("core:author", "core:description", "core:hw", "core:recorder", "core:license") if k in g}
    anns = d.get("annotations") or []
    extra["argus:source_annotations"] = len(anns) if isinstance(anns, list) else 1
    extra["argus:preset"] = preset.get("name")
    if fc0 is not None and segs[0][1] != fc0:
        extra["argus:source_frequency"] = fc0
    captures = []
    for s0, f, dt in segs:
        c = {"core:sample_start": s0, "core:frequency": float(f or 0.0)}
        if dt:
            c["core:datetime"] = dt
        captures.append(c)
    _write_meta(out_meta, _global(rate, out_dtype, meta_path, fmt, notes, extra), captures)
    return Converted(out_meta, meta_path, rate, float(segs[0][1] or 0.0), out_dtype, n_total, len(segs), dtype, notes)


def convert_raw(path: Path, out_dir: Path, ov: Overrides, out_stem: str | None = None) -> Converted:
    notes: list = []
    preset = preset_for(path)
    side, side_path = sidecar_for(path)
    rate = ov.rate or (_num(_find_key(side, RATE_KEYS)) if side else None) or preset.get("rate")
    _require_rate(rate, path, preset)
    fc = ov.fc if ov.fc is not None else (_num(_find_key(side, FC_KEYS)) if side else None)
    if fc is None:
        fc = preset.get("fc")
    if fc is None:
        fc = parse_hz_from_name(path.name)
    fc = fix_fc(fc, notes)
    dtype = norm_dtype(ov.dtype) or (norm_dtype(_find_key(side, DTYPE_KEYS)) if side else None) \
        or norm_dtype(preset.get("dtype")) or RAW_EXT_DTYPE.get(path.suffix.lower())
    if dtype is None:
        dtype = guess_raw_dtype(path)
        notes.append(f"sample format guessed from the bytes: {dtype} (pass --dtype to be sure)")
    if dtype not in READABLE:
        raise ConvertError(f"unsupported sample format {dtype}")
    if preset.get("note"):
        notes.append(preset["note"])
    if fc is None:
        notes.append("RF centre frequency unknown (pass --fc): annotation frequencies are baseband offsets")
    out_dir.mkdir(parents=True, exist_ok=True)
    o = out_stem or _stem_of(path)
    out_meta, out_data = out_dir / f"{o}.sigmf-meta", out_dir / f"{o}.sigmf-data"
    out_dtype = _place_data(path, dtype, out_data, ov)
    n_total = out_data.stat().st_size // bytes_per_sample(out_dtype)
    extra = {"argus:source_sidecar": str(side_path) if side_path else None,
             "argus:source_label": _find_key(side, LABEL_KEYS) if side else None, "argus:preset": preset.get("name")}
    _write_meta(out_meta, _global(rate, out_dtype, path, "raw " + dtype, notes, extra),
                [{"core:sample_start": 0, "core:frequency": float(fc or 0.0)}])
    return Converted(out_meta, path, rate, float(fc or 0.0), out_dtype, n_total, 1, dtype, notes)


def convert_array(path: Path, out_dir: Path, ov: Overrides, out_stem: str | None = None) -> Converted:
    preset = preset_for(path)
    ov2 = Overrides(**{**ov.__dict__})
    if ov2.var is None and preset.get("var"):
        ov2.var = preset["var"]
    if ov2.iq_layout == "auto" and preset.get("iq_layout"):
        ov2.iq_layout = preset["iq_layout"]
    name, arr, rate_f, fc_f, notes = load_container(path, ov2)
    side, side_path = sidecar_for(path)
    rate = ov.rate or rate_f or (_num(_find_key(side, RATE_KEYS)) if side else None) or preset.get("rate")
    _require_rate(rate, path, preset)
    fc = ov.fc if ov.fc is not None else (fc_f if fc_f is not None else
                                          (_num(_find_key(side, FC_KEYS)) if side else None))
    if fc is None:
        fc = preset.get("fc")
    fc = fix_fc(fc, notes)
    if preset.get("note"):
        notes.append(preset["note"])
    if fc is None:
        notes.append("RF centre frequency unknown (pass --fc): annotation frequencies are baseband offsets")
    view = ArrayView(arr, ov2.iq_layout, ov2.index, notes)
    n_clips = min(view.n_clips, ov.max_segments) if ov.max_segments else view.n_clips
    out_dtype = "cf32_le" if ov.force_cf32 else view.out_dtype
    out_dir.mkdir(parents=True, exist_ok=True)
    base = path.stem if not name or name in (path.stem, "data", "wav") else \
        f"{path.stem}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}"
    o = out_stem or base
    out_meta, out_data = out_dir / f"{o}.sigmf-meta", out_dir / f"{o}.sigmf-data"
    _unlink(out_data)
    captures, pos = [], 0
    with open(out_data, "wb") as f:
        for i in range(n_clips):
            x = view.clip_native(i)
            if ov.force_cf32 and x.dtype != np.complex64:     # integer I/Q -> cf32, scaled like read_raw
                x = ((x[0::2].astype(np.float32) + 1j * x[1::2].astype(np.float32))
                     / float(np.iinfo(x.dtype).max)).astype(np.complex64)
            n = x.size if x.dtype == np.complex64 else x.size // 2
            c = {"core:sample_start": pos, "core:frequency": float(fc or 0.0)}
            if n_clips > 1 or view.lead:
                c["argus:segment"] = list(tuple(ov2.index or ()) + view.clip_index(i))
            captures.append(c)
            x.tofile(f)
            pos += n
    if n_clips < view.n_clips:
        notes.append(f"only the first {n_clips} of {view.n_clips} clips converted (--max-segments)")
    extra = {"argus:source_variable": name, "argus:source_shape": list(_shape_of(arr)) if view.kind != "wav" else None,
             "argus:iq_layout": view.layout, "argus:segments_total": view.n_clips,
             "argus:source_sidecar": str(side_path) if side_path else None, "argus:preset": preset.get("name")}
    _write_meta(out_meta, _global(rate, out_dtype, path, path.suffix.lower().lstrip(".") + " array", notes, extra),
                captures)
    return Converted(out_meta, path, rate, float(fc or 0.0), out_dtype, pos, len(captures),
                     str(_dtype_of(arr)) if view.kind != "wav" else "wav", notes)


def convert(path: Path, out_dir: Path, ov: Overrides, out_stem: str | None = None) -> Converted:
    if path.name.lower().endswith(".sigmf-meta"):
        return convert_sigmf(path, out_dir, ov, out_stem)
    if path.suffix.lower() in ARRAY_EXTS:
        return convert_array(path, out_dir, ov, out_stem)
    return convert_raw(path, out_dir, ov, out_stem)


def find_inputs(paths: list[Path], recursive: bool = True, glob: str | None = None) -> list[tuple[Path, Path]]:
    """-> [(file, root it was found under)]; a .sigmf-data whose .sigmf-meta exists is handled through the meta."""
    exts = RAW_EXTS | ARRAY_EXTS
    out = []
    for p in paths:
        p = Path(p).expanduser()
        if p.is_file():
            if p.name.endswith(".sigmf-data") and p.with_name(_stem_of(p) + ".sigmf-meta").is_file():
                p = p.with_name(_stem_of(p) + ".sigmf-meta")
            out.append((p, p.parent))
            continue
        if not p.is_dir():
            raise ConvertError(f"no such file or directory: {p}")
        for f in sorted(p.rglob("*") if recursive else p.glob("*")):
            if not f.is_file() or SKIP_NAMES.search(f.name):
                continue
            if any(d in SKIP_DIRS for d in f.relative_to(p).parts[:-1]):
                continue
            n = f.name.lower()
            if n.endswith(".sigmf-meta"):
                pass
            elif n.endswith(".sigmf-data"):
                if f.with_name(_stem_of(f) + ".sigmf-meta").exists():
                    continue
            elif f.suffix.lower() not in exts:
                continue
            if glob and not (fnmatch.fnmatch(f.name, glob) or
                             (n.endswith(".sigmf-meta") and fnmatch.fnmatch(_stem_of(f) + ".sigmf-data", glob))):
                continue
            out.append((f, p))
    return out


def rel_dir(f: Path, root: Path) -> Path:
    try:
        return f.parent.resolve().relative_to(root.resolve())
    except ValueError:
        return Path()


def parse_index(s: str | None) -> tuple | None:
    return tuple(int(v) for v in s.split(",")) if s else None


def add_override_args(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("how to read inputs that are not self-describing (to_sigmf.py)")
    g.add_argument("--rate", type=float, default=None, help="sample rate in S/s, e.g. 25e6")
    g.add_argument("--fc", type=float, default=None, help="RF centre frequency in Hz")
    g.add_argument("--dtype", default=None, help="raw sample format: cf32_le cf64_le ci16_le ci8 cu8 cf32_be ...")
    g.add_argument("--iq-layout", default="auto", choices=["auto", "pairs", "interleaved", "planar", "real"],
                   help="real arrays: I/Q in a last axis of 2 (pairs), alternating along the last axis (interleaved), "
                        "in an axis of 2 before the samples (planar), or real-valued samples")
    g.add_argument("--var", default=None, help="array / variable / dataset name inside npz, mat, h5 or pickle files")
    g.add_argument("--index", default=None, help="select along the leading axes first, e.g. '0' = channel 0 of a "
                                                 "(channels, clips, samples) array")
    g.add_argument("--max-segments", type=int, default=None, help="convert only the first N clips of an array")
    g.add_argument("--copy", action="store_true", help="copy raw samples instead of symlinking them")


def overrides_from(a) -> Overrides:
    return Overrides(rate=a.rate, fc=a.fc, dtype=a.dtype, iq_layout=a.iq_layout, var=a.var, index=parse_index(a.index),
                     max_segments=a.max_segments, copy=a.copy, force_cf32=getattr(a, "cf32", False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="files or directories")
    ap.add_argument("--out", default=None, help="output directory (mirrors the input tree)")
    ap.add_argument("--recursive", action="store_true", help="also descend into sub-directories")
    ap.add_argument("--glob", default=None, help="only file names matching this pattern")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--cf32", action="store_true", help="always write real cf32_le data files (no symlinks)")
    ap.add_argument("--list-presets", action="store_true", help="print the known-dataset presets and exit")
    add_override_args(ap)
    a = ap.parse_args(argv)
    if a.list_presets:
        for p in PRESETS:
            print(f"{p['name']:<18} {p['match']:<30} {p.get('note', '')}")
        return 0
    if not a.inputs or not a.out:
        ap.error("give input files/directories and --out")
    ov = overrides_from(a)
    files = find_inputs([Path(p) for p in a.inputs], a.recursive, a.glob)[: a.max_files]
    ok = bad = 0
    for f, root in files:
        try:
            c = convert(f, Path(a.out).expanduser() / rel_dir(f, root), ov)
            ok += 1
            print(f"{f.name}: {c.n_samples} samples in {c.n_segments} segment(s), {c.rate / 1e6:.4g} MS/s, "
                  f"fc {c.fc / 1e6:.6g} MHz, {c.dtype} -> {c.meta_path}")
            for n in c.notes:
                print(f"    note: {n}")
        except (ConvertError, OSError, ValueError) as e:
            bad += 1
            print(f"{f.name}: SKIPPED ({e})", file=sys.stderr)
    print(f"{ok} converted, {bad} skipped")
    return 0 if ok or not files else 1


if __name__ == "__main__":
    sys.exit(main())

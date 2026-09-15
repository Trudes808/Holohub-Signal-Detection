"""491.52 MSps composite generator for the DINO-FT domain-match fine-tune.

Places REAL library waveforms (~/Documents/holoscan_waveform_generation/generated_waveforms_24576/,
9 classes, all at 245.76 MSps) onto a 491.52 MSps canvas: each waveform is resampled 2x (245.76->491.52,
occupied BW preserved in Hz), tiled/cropped to a chosen duration, scaled to a per-signal SNR (vs the
complex-AWGN floor), and frequency-shifted anywhere in the +/-~230 MHz band. Emits in-memory IQ +
SigMF-style annotations (consumed by rfdata for GT rasterization); optionally writes SigMF to disk.

Two densities:
  - DENSE : many overlapping signals filling the band (comprehensive-style)
  - SPARSE: few signals with large time/freq gaps (matches the failing live OTA scene)

Per-signal SNR is drawn to span field distance (near/far). Noise power is fixed at 1.0, so SNR_dB fully
sets each signal's amplitude.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

FS = 491_520_000.0          # deployment rate
FS_LIB = 245_760_000.0      # library waveform rate
LIB = Path("/home/genesys-dgx1/Documents/holoscan_waveform_generation/generated_waveforms_24576")
CLASSES = ["BPSK", "QPSK", "16QAM", "OFDM", "802_11ax", "5G_Downlink",
           "Bluetooth", "Broadband_FM", "Narrowband_FM"]


@dataclass
class LibEntry:
    path: Path
    cls: str
    occ_bw_hz: float
    n: int


def index_library(rng: np.random.Generator | None = None) -> list[LibEntry]:
    entries = []
    for cls in CLASSES:
        d = LIB / cls
        if not d.is_dir():
            continue
        for j in sorted(d.glob("*.json")):
            try:
                md = json.loads(j.read_text())
            except Exception:
                continue
            bw = md.get("designedOccupiedBandwidthHz")
            if not bw:
                continue
            mat = j.with_suffix(".mat")
            if mat.exists():
                entries.append(LibEntry(path=mat, cls=cls, occ_bw_hz=float(bw), n=0))
    return entries


_CACHE: dict[Path, np.ndarray] = {}


def load_wave_491(path: Path) -> np.ndarray:
    """Load a 245.76 library waveform and resample 2x -> 491.52 (occupied BW preserved)."""
    if path in _CACHE:
        return _CACHE[path]
    iq = np.asarray(loadmat(str(path))["f_sig"]).ravel().astype(np.complex64)
    up = resample_poly(iq, 2, 1).astype(np.complex64)   # 245.76 -> 491.52
    # peak-normalize so power scaling below is well-defined
    p = np.sqrt(np.mean(np.abs(up) ** 2)) + 1e-12
    up = (up / p).astype(np.complex64)
    _CACHE[path] = up
    return up


@dataclass
class Annotation:
    sample_start: int
    sample_count: int
    freq_lower_hz: float
    freq_upper_hz: float
    label: str
    occ_bw_hz: float
    snr_db: float


def _place_signal(canvas, e: LibEntry, t0, dur, fc, snr_db, rng):
    """Add one waveform to the canvas at [t0,t0+dur), center fc, at the given SNR (noise power=1)."""
    w = load_wave_491(e.path)
    if w.size == 0:
        return None
    # tile/crop to dur (random phase offset into the waveform)
    if w.size < dur:
        reps = int(np.ceil(dur / w.size))
        w = np.tile(w, reps)
    s0 = int(rng.integers(0, max(1, w.size - dur)))
    seg = w[s0:s0 + dur].copy()
    amp = 10.0 ** (snr_db / 20.0)          # noise power 1 -> signal amplitude = 10^(snr/20)
    n = np.arange(dur, dtype=np.float64)
    seg = (seg * amp * np.exp(2j * np.pi * fc / FS * n)).astype(np.complex64)
    canvas[t0:t0 + dur] += seg
    lo = fc - e.occ_bw_hz / 2.0
    hi = fc + e.occ_bw_hz / 2.0
    return Annotation(sample_start=t0, sample_count=dur, freq_lower_hz=lo, freq_upper_hz=hi,
                      label=e.cls, occ_bw_hz=e.occ_bw_hz, snr_db=snr_db)


# SNR buckets spanning field distance (near -> far), drawn per signal.
SNR_BUCKETS = [(20.0, 32.0), (10.0, 20.0), (3.0, 10.0), (-3.0, 3.0)]


def _draw_snr(rng):
    lo, hi = SNR_BUCKETS[rng.integers(len(SNR_BUCKETS))]
    return float(rng.uniform(lo, hi))


def _draw_dur(rng, T):
    """Log-uniform duration in samples spanning short bursts (~80us) to frame-spanning carriers (~20ms),
    so the detector sees both transient and persistent signals within a 256-row (~10.7ms) tile."""
    d = float(np.exp(rng.uniform(np.log(80e-6), np.log(20e-3))) * FS)
    return int(min(max(d, 1024), T - 1))


def make_composite(mode: str, dur_s: float, lib: list[LibEntry], seed: int):
    """Return (iq complex64, [Annotation]) for one composite. mode in {dense, sparse}."""
    rng = np.random.default_rng(seed)
    T = int(dur_s * FS)
    canvas = (rng.standard_normal(T) + 1j * rng.standard_normal(T)).astype(np.complex64)
    canvas *= np.float32(1.0 / np.sqrt(2.0))   # unit noise power
    anns: list[Annotation] = []
    band = 0.92 * FS / 2.0                       # keep signals off the very edges

    n_sig = int(rng.integers(60, 100)) if mode == "dense" else int(rng.integers(3, 10))
    if mode not in ("dense", "sparse"):
        raise ValueError(mode)
    for _ in range(n_sig):
        e = lib[rng.integers(len(lib))]
        dur = _draw_dur(rng, T)
        t0 = int(rng.integers(0, max(1, T - dur)))
        fc_max = band - e.occ_bw_hz / 2.0
        if fc_max <= 0:
            continue
        fc = float(rng.uniform(-fc_max, fc_max))
        a = _place_signal(canvas, e, t0, dur, fc, _draw_snr(rng), rng)
        if a:
            anns.append(a)
    return canvas, anns


def write_sigmf(stem: Path, iq: np.ndarray, anns: list[Annotation], center_hz=2.4e9):
    stem.parent.mkdir(parents=True, exist_ok=True)
    iq.astype(np.complex64).tofile(str(stem.with_suffix(".sigmf-data")))
    meta = {
        "global": {"core:datatype": "cf32_le", "core:sample_rate": FS, "core:version": "1.0.0",
                   "core:num_channels": 1, "core:description": f"491.52 synthetic composite {stem.name}"},
        "captures": [{"core:sample_start": 0, "core:frequency": center_hz}],
        "annotations": [
            {"core:sample_start": a.sample_start, "core:sample_count": a.sample_count,
             "core:freq_lower_edge": a.freq_lower_hz, "core:freq_upper_edge": a.freq_upper_hz,
             "core:label": a.label, "wfgt:kind": "waveform", "wfgt:occupied_bw_hz": a.occ_bw_hz,
             "wfgt:power_db": a.snr_db}
            for a in anns
        ],
    }
    stem.with_suffix(".sigmf-meta").write_text(json.dumps(meta))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dense", "sparse"], default="sparse")
    ap.add_argument("--dur-s", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output stem (no extension)")
    args = ap.parse_args()
    lib = index_library()
    print(f"library: {len(lib)} waveforms across {len(set(e.cls for e in lib))} classes")
    iq, anns = make_composite(args.mode, args.dur_s, lib, args.seed)
    write_sigmf(Path(args.out), iq, anns)
    print(f"wrote {args.out} ({iq.size} samples = {iq.size/FS:.3f}s, {len(anns)} signals)")

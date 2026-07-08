"""SigMF I/O, spectrogram construction, and GT-mask rasterization.

The spectrogram convention matches the deployed detector and the existing eval
(`eval_viz.spectrogram_db_from_iq`):

    row r  = fftshift(fft( iq[r*nfft : (r+1)*nfft] ))   -> time increases downward
    col c  = frequency, c=0 -> -Fs/2, c=nfft-1 -> +Fs/2 - binwidth, DC at c=nfft/2

so a frame is a (frame_rows, nfft) image of 10*log10(|.|^2). All geometry is
kept on a single grid: the DINO input, the GT mask, and the predicted mask all
live on (frame_rows, nfft), so no resampling is needed between train and eval.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch

CF32 = np.dtype("<c8")  # SigMF cf32_le == interleaved float32 I/Q == little-endian complex64


# --------------------------------------------------------------------------- #
# Capture metadata
# --------------------------------------------------------------------------- #
_ATTEN_RE = re.compile(r"attenuation_dB_(\d+)")


def parse_attenuation_db(stem: str) -> Optional[int]:
    """attenuation_dB_30, attenuation_dB_30_v2 -> 30 ; None if not parseable."""
    m = _ATTEN_RE.search(stem)
    return int(m.group(1)) if m else None


@dataclass
class Annotation:
    sample_start: int
    sample_count: int
    freq_lower_hz: float
    freq_upper_hz: float
    label: str
    kind: str
    time_group: Optional[int]
    bandwidth_hz: float  # occupied bw; from wfgt if present else (upper-lower)

    @property
    def sample_stop(self) -> int:
        return self.sample_start + self.sample_count


@dataclass
class Capture:
    stem: str
    meta_path: Path
    data_path: Path
    sample_rate: float
    center_freq_hz: float
    attenuation_db: Optional[int]
    annotations: list[Annotation]
    n_samples: int
    # annotation start samples sorted, for fast overlap lookup
    _ann_starts: np.ndarray = field(default=None, repr=False)
    _ann_stops: np.ndarray = field(default=None, repr=False)

    def memmap(self) -> np.memmap:
        return np.memmap(self.data_path, dtype=CF32, mode="r", shape=(self.n_samples,))


def load_capture(meta_path: Path) -> Capture:
    meta_path = Path(meta_path)
    meta = json.loads(meta_path.read_text())
    g = meta["global"]
    sr = float(g["core:sample_rate"])
    caps = meta.get("captures", [{}])
    center = float(caps[0].get("core:frequency", 0.0)) if caps else 0.0
    data_path = meta_path.with_suffix(".sigmf-data")
    n_samples = data_path.stat().st_size // CF32.itemsize

    anns: list[Annotation] = []
    for a in meta.get("annotations", []):
        lo = float(a.get("core:freq_lower_edge", 0.0))
        hi = float(a.get("core:freq_upper_edge", 0.0))
        bw = a.get("wfgt:occupied_bw_hz")
        bw = float(bw) if bw is not None else abs(hi - lo)
        anns.append(
            Annotation(
                sample_start=int(a.get("core:sample_start", 0)),
                sample_count=int(a.get("core:sample_count", 0)),
                freq_lower_hz=lo,
                freq_upper_hz=hi,
                label=str(a.get("core:label", "UNLABELED")),
                kind=str(a.get("wfgt:kind", "annotation")),
                time_group=a.get("wfgt:time_group"),
                bandwidth_hz=bw,
            )
        )
    stem = meta_path.stem
    cap = Capture(
        stem=stem,
        meta_path=meta_path,
        data_path=data_path,
        sample_rate=sr,
        center_freq_hz=center,
        attenuation_db=parse_attenuation_db(stem),
        annotations=anns,
        n_samples=n_samples,
    )
    if anns:
        order = np.argsort([a.sample_start for a in anns])
        cap.annotations = [anns[i] for i in order]
        cap._ann_starts = np.array([a.sample_start for a in cap.annotations], dtype=np.int64)
        cap._ann_stops = np.array([a.sample_stop for a in cap.annotations], dtype=np.int64)
    else:
        cap._ann_starts = np.zeros(0, dtype=np.int64)
        cap._ann_stops = np.zeros(0, dtype=np.int64)
    return cap


# --------------------------------------------------------------------------- #
# Spectrogram (GPU-batched)
# --------------------------------------------------------------------------- #
def frames_to_db(
    iq_frames: torch.Tensor, nfft: int, frame_rows: int
) -> torch.Tensor:
    """(B, frame_rows*nfft) complex -> (B, frame_rows, nfft) dB spectrogram.

    Matches eval_viz.spectrogram_db_from_iq: per-row fftshift(fft) magnitude in dB.
    """
    B = iq_frames.shape[0]
    block = iq_frames.reshape(B, frame_rows, nfft)
    spec = torch.fft.fftshift(torch.fft.fft(block, dim=-1), dim=-1)
    power = spec.real**2 + spec.imag**2 + 1e-12
    return 10.0 * torch.log10(power)


def db_to_uint8(db: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Global fixed dB->uint8 mapping so cross-attenuation SNR is preserved."""
    x = (db - vmin) / max(vmax - vmin, 1e-6)
    return (np.clip(x, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------- #
# GT mask rasterization
# --------------------------------------------------------------------------- #
def freq_to_col(f_hz: float, sample_rate: float, nfft: int) -> float:
    """Map baseband frequency (Hz, DC=0) to fractional column index [0, nfft]."""
    return (f_hz + sample_rate / 2.0) / sample_rate * nfft


@dataclass
class RegionBox:
    """A GT annotation clipped to one frame, in (row,col) pixel coords."""
    ann_idx: int
    row0: int
    row1: int
    col0: int
    col1: int
    label: str
    kind: str
    bandwidth_hz: float
    length_samples: int
    time_group: Optional[int]


def frame_annotations(cap: Capture, frame_abs_start: int, frame_samples: int):
    """Indices of annotations overlapping [frame_abs_start, frame_abs_start+frame_samples)."""
    fs, fe = frame_abs_start, frame_abs_start + frame_samples
    # overlap iff ann_start < fe and ann_stop > fs
    idx = np.nonzero((cap._ann_starts < fe) & (cap._ann_stops > fs))[0]
    return idx


def build_frame_mask(
    cap: Capture,
    frame_abs_start: int,
    nfft: int,
    frame_rows: int,
) -> tuple[np.ndarray, list[RegionBox]]:
    """Binary (frame_rows, nfft) signal mask + per-region boxes for this frame.

    All annotation kinds (ZC, METADATA, waveforms) count as signal.
    """
    frame_samples = nfft * frame_rows
    sr = cap.sample_rate
    mask = np.zeros((frame_rows, nfft), dtype=np.uint8)
    boxes: list[RegionBox] = []
    for ai in frame_annotations(cap, frame_abs_start, frame_samples):
        a = cap.annotations[ai]
        rel_start = a.sample_start - frame_abs_start
        rel_stop = a.sample_stop - frame_abs_start
        r0 = max(0, int(np.floor(rel_start / nfft)))
        r1 = min(frame_rows, int(np.ceil(rel_stop / nfft)))
        c0 = max(0, int(np.floor(freq_to_col(a.freq_lower_hz, sr, nfft))))
        c1 = min(nfft, int(np.ceil(freq_to_col(a.freq_upper_hz, sr, nfft))))
        # guarantee >=1px so a present annotation is never silently empty
        r1 = max(r1, r0 + 1) if r1 > r0 or rel_stop > 0 else r1
        c1 = max(c1, c0 + 1) if c1 > c0 or a.freq_upper_hz > a.freq_lower_hz else c1
        if r1 <= r0 or c1 <= c0:
            continue
        mask[r0:r1, c0:c1] = 1
        boxes.append(
            RegionBox(
                ann_idx=int(ai), row0=r0, row1=r1, col0=c0, col1=c1,
                label=a.label, kind=a.kind, bandwidth_hz=a.bandwidth_hz,
                length_samples=a.sample_count, time_group=a.time_group,
            )
        )
    return mask, boxes

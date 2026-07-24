"""Capture-chain emulation for band/rate-invariant fine-tuning.

Emulate what the wideband labeled IQ (245.76 MS/s) WOULD have looked like if captured at a lower rate
`R` and a random center offset `f_c` inside the band: frequency-shift `f_c` to DC -> anti-alias
low-pass -> decimate to `R` -> spectrogram. Optionally re-shape the per-frequency envelope to the real
per-rate receiver envelope (measured by the sweep). Annotations are remapped to the new grid (freq
shifted by -f_c and clipped to the emulated band; time rescaled by the decimation), so labels stay
aligned. See notes/retrain_band_rate_invariant_plan.md sec 2a.

Target rates ABOVE the source (e.g. 491.52) cannot be produced by decimation; those are handled by the
cut-paste path in build_dataset (upsample the labeled signal onto a real wide sweep background), not
here.
"""
from __future__ import annotations

import types
from typing import Optional

import numpy as np
import torch

import rfdata as rf

try:
    from scipy.signal import resample_poly
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


def emulate_iq_at_rate(src_iq, src_rate: float, target_rate: float, f_c_hz: float, device: str = "cpu"):
    """Freq-shift src_iq so f_c->DC, then decimate to ~target_rate via **GPU FFT band-extraction**
    (exact brick-wall LPF; the real per-rate rolloff is applied later by apply_envelope_reshape). Runs
    on `device` (cuda -> the decimation FFT is on the GPU, ~100x faster than a CPU polyphase resample at
    high D). src_iq: numpy complex64 or torch complex tensor. Returns (emu torch complex64 on device,
    actual_rate, D)."""
    x = torch.as_tensor(src_iq, dtype=torch.complex64, device=device)
    N = int(x.shape[0])
    if abs(f_c_hz) > 0.0:
        n = torch.arange(N, device=device, dtype=torch.float32)
        x = x * torch.polar(torch.ones_like(n), (-2.0 * np.pi * f_c_hz / src_rate) * n)
    D = max(1, int(round(src_rate / target_rate)))
    if D == 1:
        return x, src_rate, 1
    M = N // D
    X = torch.fft.fftshift(torch.fft.fft(x))
    lo = (N - M) // 2
    emu = torch.fft.ifft(torch.fft.ifftshift(X[lo:lo + M])) * (M / N)
    return emu, src_rate / D, D


def remap_annotations(anns, src_rate: float, actual_rate: float, D: int, f_c_hz: float,
                      chunk_abs_start: int):
    """Shift ann freq by -f_c and clip to [-actual_rate/2, +actual_rate/2]; rescale sample positions
    (relative to the chunk start) by 1/D. Drops annotations that fall outside the emulated band."""
    out = []
    half = actual_rate / 2.0
    for a in anns:
        lo = a.freq_lower_hz - f_c_hz
        hi = a.freq_upper_hz - f_c_hz
        lo_c, hi_c = max(lo, -half), min(hi, half)
        if hi_c <= lo_c:
            continue  # signal outside the emulated band
        out.append(rf.Annotation(
            sample_start=int(round((a.sample_start - chunk_abs_start) / D)),
            sample_count=max(1, int(round(a.sample_count / D))),
            freq_lower_hz=lo_c, freq_upper_hz=hi_c, label=a.label, kind=a.kind,
            time_group=a.time_group, bandwidth_hz=min(a.bandwidth_hz, hi_c - lo_c)))
    return out


def _emulated_capture(anns, actual_rate: float):
    """Duck-typed object exposing exactly what rf.build_frame_mask needs."""
    cap = types.SimpleNamespace(sample_rate=actual_rate, annotations=anns)
    cap._ann_starts = np.array([a.sample_start for a in anns], dtype=np.int64) if anns else np.zeros(0, np.int64)
    cap._ann_stops = np.array([a.sample_stop for a in anns], dtype=np.int64) if anns else np.zeros(0, np.int64)
    return cap


def _nearest_env(envelopes: dict, rate_hz: float) -> Optional[np.ndarray]:
    if not envelopes:
        return None
    rates = np.array(list(envelopes.keys()), dtype=float)
    return envelopes[float(rates[np.argmin(np.abs(rates - rate_hz))])]


def apply_envelope_reshape(db_frame: np.ndarray, target_rate: float, src_rate: float, f_c_hz: float,
                           envelopes: Optional[dict]) -> np.ndarray:
    """Re-shape the per-frequency envelope to the measured target-rate receiver envelope: remove the
    source envelope over the extracted sub-band and add the target-rate envelope (both zero-mean dB
    shapes, per column). No-op if templates are missing. envelopes: {rate_hz: [nfft] zero-mean dB}."""
    env_R = _nearest_env(envelopes or {}, target_rate)
    if env_R is None:
        return db_frame
    nfft = db_frame.shape[1]
    if env_R.shape[0] != nfft:
        env_R = np.interp(np.linspace(0, 1, nfft), np.linspace(0, 1, env_R.shape[0]), env_R)
    add = env_R.astype(np.float32).copy()
    env_src = _nearest_env(envelopes or {}, src_rate)
    if env_src is not None:
        f_axis = np.linspace(-src_rate / 2, src_rate / 2, env_src.shape[0])
        sub_f = np.linspace(f_c_hz - target_rate / 2, f_c_hz + target_rate / 2, nfft)
        add = add - np.interp(sub_f, f_axis, env_src).astype(np.float32)
    return db_frame + add[None, :]


def emulate_frame(src_iq_chunk: np.ndarray, chunk_abs_start: int, anns, src_rate: float,
                  target_rate: float, f_c_hz: float, nfft: int, frame_rows: int,
                  envelopes: Optional[dict] = None, device: str = "cpu", window: str = "hann"):
    """One emulated (db_frame float32 [frame_rows,nfft], mask uint8, boxes) at target_rate/f_c.

    src_iq_chunk must be >= frame_rows*nfft*D samples (D=round(src/target)); the first frame after
    decimation is used. `window` = the FFT analysis window (must match the deployed detector)."""
    emu_iq, actual_rate, D = emulate_iq_at_rate(src_iq_chunk, src_rate, target_rate, f_c_hz, device=device)
    need = frame_rows * nfft
    if emu_iq.shape[0] < need:
        emu_iq = torch.cat([emu_iq, torch.zeros(need - emu_iq.shape[0], dtype=torch.complex64,
                                                device=emu_iq.device)])
    frame_iq = emu_iq[:need].reshape(1, need)  # torch complex on device
    db = rf.frames_to_db(frame_iq, nfft, frame_rows, window=window)[0].cpu().numpy().astype(np.float32)
    db = apply_envelope_reshape(db, target_rate, src_rate, f_c_hz, envelopes)
    emu_cap = _emulated_capture(remap_annotations(anns, src_rate, actual_rate, D, f_c_hz, chunk_abs_start),
                                actual_rate)
    mask, boxes = rf.build_frame_mask(emu_cap, 0, nfft, frame_rows)
    return db, mask, boxes


def upsample_iq(src_iq: np.ndarray, src_rate: float, target_rate: float):
    """Upsample bandlimited src_iq (@ src_rate) to ~target_rate (target > src). Lossless: the signal
    stays within +-src_rate/2 of the wider band. Returns (up_iq, actual_rate, U)."""
    U = max(1, int(round(target_rate / src_rate)))
    if U == 1:
        return src_iq.astype(np.complex64), src_rate, 1
    if _HAVE_SCIPY:
        up = resample_poly(src_iq, up=U, down=1)
    else:
        N = src_iq.shape[0]
        X = np.fft.fftshift(np.fft.fft(src_iq))
        pad = (U * N - N) // 2
        Xp = np.pad(X, (pad, U * N - N - pad))
        up = np.fft.ifft(np.fft.ifftshift(Xp)) * U
    return up.astype(np.complex64), src_rate * U, U


def emulate_frame_upsample_paste(signal_chunk, chunk_abs_start, anns, src_rate, target_rate,
                                 bg_iq, nfft, frame_rows, sig_gain=1.0, envelopes=None, device="cpu",
                                 window="hann"):
    """Phase-2 path for target_rate > src_rate (can't decimate): upsample the labeled signal to the
    wide rate and ADD it (IQ domain) onto a REAL wideband background captured at that rate (bg_iq, from
    the sweep). The background supplies the true wide noise/envelope/spurs; labels come from the signal.
    sig_gain scales the signal (SNR augmentation). Annotations: freq unchanged (signal centered in the
    wider band), time rescaled by the upsample factor U."""
    up_iq, actual_rate, U = upsample_iq(signal_chunk, src_rate, target_rate)
    need = frame_rows * nfft
    up = up_iq[:need]
    if up.shape[0] < need:
        up = np.concatenate([up, np.zeros(need - up.shape[0], np.complex64)])
    bg = np.asarray(bg_iq[:need], np.complex64)
    if bg.shape[0] < need:
        bg = np.concatenate([bg, np.zeros(need - bg.shape[0], np.complex64)])
    emu = (np.complex64(sig_gain) * up + bg).astype(np.complex64)
    db = rf.frames_to_db(torch.from_numpy(emu[None, :]).to(device), nfft, frame_rows, window=window)[0].cpu().numpy().astype(np.float32)
    db = apply_envelope_reshape(db, target_rate, actual_rate, 0.0, envelopes)  # env_src==target -> ~no-op
    half = actual_rate / 2.0
    remapped = []
    for a in anns:
        lo, hi = max(a.freq_lower_hz, -half), min(a.freq_upper_hz, half)
        if hi <= lo:
            continue
        remapped.append(rf.Annotation(
            sample_start=int(round((a.sample_start - chunk_abs_start) * U)),
            sample_count=max(1, int(round(a.sample_count * U))),
            freq_lower_hz=lo, freq_upper_hz=hi, label=a.label, kind=a.kind,
            time_group=a.time_group, bandwidth_hz=min(a.bandwidth_hz, hi - lo)))
    mask, boxes = rf.build_frame_mask(_emulated_capture(remapped, actual_rate), 0, nfft, frame_rows)
    return db, mask, boxes


# --------------------------------------------------------------------------- #
# Self-test: synthetic tone lands in the expected bin + a label remaps correctly.
# --------------------------------------------------------------------------- #
def _selftest():
    src_rate, nfft, frame_rows = 245.76e6, 1024, 64
    D = 4
    target_rate = src_rate / D                       # 61.44 MS/s
    f_c = 30e6                                        # random center offset inside the band
    tone_hz = 40e6                                    # absolute baseband tone; after -f_c -> 10 MHz
    N = frame_rows * nfft * D + 4096
    n = np.arange(N)
    iq = np.exp(2j * np.pi * tone_hz / src_rate * n).astype(np.complex64)
    iq += 0.01 * (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex64)
    ann = rf.Annotation(sample_start=0, sample_count=frame_rows * nfft * D,
                        freq_lower_hz=tone_hz - 1e6, freq_upper_hz=tone_hz + 1e6,
                        label="TEST", kind="tone", time_group=0, bandwidth_hz=2e6)
    db, mask, boxes = emulate_frame(iq, 0, [ann], src_rate, target_rate, f_c, nfft, frame_rows)
    # expected column of the tone at (tone-f_c)=10 MHz in the 61.44 MS/s band:
    exp_col = int(rf.freq_to_col(tone_hz - f_c, target_rate, nfft))
    peak_col = int(np.argmax(db.mean(axis=0)))
    print(f"emulate: D={D} target={target_rate/1e6:.2f} MS/s  tone expected col~{exp_col}, "
          f"spectrogram peak col={peak_col}  (|diff|={abs(exp_col-peak_col)})")
    print(f"mask: {int(mask.sum())} px, box cols {boxes[0].col0}..{boxes[0].col1} "
          f"(expected ~{exp_col})" if boxes else "mask: EMPTY (FAIL)")
    assert boxes and abs(peak_col - exp_col) <= 3 and boxes[0].col0 <= peak_col <= boxes[0].col1, "FAIL"
    print("SELFTEST PASS")


if __name__ == "__main__":
    _selftest()

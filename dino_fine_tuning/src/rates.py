"""USRP X410 sample-rate set and the app's per-rate frame geometry.

Mirrors `applications/usrp_wideband_signal_detection/fft_runtime_config.hpp` (how the live app derives the
wide-FFT size / packets-per-row from the delivered rate) and `operators/finetuned_dino_detector/
finetuned_dino_detector.cu::auto_fft_size` (identical formula), so the training geometry at ANY rate is the
deployment geometry at that rate:

    span_ratio   = rate / 500 MHz
    snapped      = 2 ** round(log2(span_ratio))            (C++ std::round: half away from zero)
    packets      = max(1, llround(20480 * snapped / 1024))   (C++ llround: half away from zero)
    fft_size     = max(1024, packets * 1024)
    frame        = 512 rows x fft_size samples   (chdr_converter.num_ffts_per_batch = 512)
    model image  = frame -> wide FFT -> ... -> bilinear resize freq to 1024 -> 2 tiles of 256 rows

NOTE Python's round() is banker's rounding; at 2.5 packets (rates 44.2-88.4 MHz: 49.152/50/61.44/62.5/
81.92/83.33 MHz) it gives 2048 where the app uses 3072. Use app_fft_size(), not round().

Rate set: X410 master clock 491.52 MHz or 500 MHz, decimation 1 or any even factor (odd CIC decimations
are deprecated by UHD for passband flatness), kept within [5, 500] MHz -> 101 distinct rates.
"""
from __future__ import annotations
import math
import numpy as np

MCRS = (491.52e6, 500.0e6)
REF_SPAN_HZ = 500.0e6
REF_FFT = 20480
PACKET = 1024
ROWS_WIDE = 512          # chdr_converter.num_ffts_per_batch -> rows per frame at every rate
NFFT_MODEL = 1024
TILE_ROWS = 256


def _cpp_round(x: float) -> float:
    """C++ std::round / llround: round half away from zero (Python round() is half-to-even)."""
    return math.copysign(math.floor(abs(x) + 0.5), x)


def app_fft_size(rate_hz: float, ref_span_hz: float = REF_SPAN_HZ, ref_fft: int = REF_FFT,
                 packet: int = PACKET) -> int:
    span_ratio = rate_hz / ref_span_hz
    snapped = 2.0 ** _cpp_round(math.log2(span_ratio)) if span_ratio > 0 else 1.0
    packets = max(1, int(_cpp_round(ref_fft * snapped / packet)))
    return max(packet, packets * packet)


def frame_samples(rate_hz: float, rows_wide: int = ROWS_WIDE) -> int:
    return rows_wide * app_fft_size(rate_hz)


def geometry(rate_hz: float) -> dict:
    f = app_fft_size(rate_hz)
    return {"rate_hz": float(rate_hz), "fft_size": f, "rows_wide": ROWS_WIDE, "frame_samples": ROWS_WIDE * f,
            "row_s": f / rate_hz, "tile_s": TILE_ROWS * f / rate_hz, "frame_s": ROWS_WIDE * f / rate_hz,
            "bin_hz": rate_hz / NFFT_MODEL, "wide_bin_hz": rate_hz / f}


def decimations() -> list[int]:
    return [1] + list(range(2, 101, 2))


def x410_rates(rate_min: float = 5e6, rate_max: float = 500e6, mcrs=MCRS) -> list[float]:
    rates = set()
    for mcr in mcrs:
        for d in decimations():
            r = mcr / d
            if rate_min - 1 <= r <= rate_max + 1:
                rates.add(round(r, 3))
    return sorted(rates, reverse=True)


DEPLOY_RATES = (491.52e6, 500.0e6)


def draw_rate(rng: np.random.Generator, p_deploy: float = 0.30, rates: list[float] | None = None) -> float:
    """Sample a rate: with prob p_deploy one of the full-rate deployment rates, else log-uniform over
    [5, 500] MHz snapped to the nearest valid X410 rate (log-uniform keeps every octave equally represented
    instead of piling onto the dense low-rate end of the decimation ladder)."""
    rates = rates or x410_rates()
    if rng.random() < p_deploy:
        return float(DEPLOY_RATES[rng.integers(len(DEPLOY_RATES))])
    target = float(np.exp(rng.uniform(np.log(min(rates)), np.log(max(rates)))))
    arr = np.asarray(rates)
    return float(arr[np.argmin(np.abs(np.log(arr) - np.log(target)))])


def rate_bucket(rate_hz: float) -> str:
    """Coarse bucket for reporting."""
    m = rate_hz / 1e6
    if m >= 400: return "400-500M"
    if m >= 150: return "150-400M"
    if m >= 60:  return "60-150M"
    if m >= 25:  return "25-60M"
    if m >= 10:  return "10-25M"
    return "5-10M"


if __name__ == "__main__":
    rs = x410_rates()
    print(f"{len(rs)} rates")
    for r in rs:
        g = geometry(r)
        print(f"{r/1e6:9.3f} MHz  fft {g['fft_size']:6d}  row {g['row_s']*1e6:7.2f} us  tile {g['tile_s']*1e3:6.2f} ms  bin {g['bin_hz']/1e3:7.1f} kHz")

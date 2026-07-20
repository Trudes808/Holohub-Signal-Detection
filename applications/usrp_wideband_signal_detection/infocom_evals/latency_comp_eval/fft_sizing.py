#!/usr/bin/env python3
"""Auto FFT-size selection + real-time frame budget, ported from the deployed C++.

This is a faithful, self-contained Python port of ``resolve_fft_runtime_config`` in
``applications/usrp_wideband_signal_detection/fft_runtime_config.hpp`` (the logic the
live pipeline uses to pick an FFT size for a given sample rate). We reproduce it here
so the offline latency/compute eval runs every detector on the *exact* per-frame
geometry the deployed system would use at 20 / 100 / 250 / 500 MHz -- no container or
C++ needed.

Deployed formula (fft_runtime_config.hpp:145-166), no bin-size override:

    span_ratio    = active_span_hz / reference_span_hz          # reference = 500 MHz
    snapped_ratio = 2 ** round(log2(span_ratio))                # snap to power-of-two
    requested_fft = reference_fft_size * snapped_ratio          # reference = 20480
    packets_per_fft = round(requested_fft / packet_samples)     # packet_samples = 1024
    actual_fft_size = max(packet_samples, packets_per_fft * packet_samples)

A detector "frame" is one batch = ``num_ffts_per_batch`` FFT rows (default 512, the
spectrogram time dimension; see chdr_converter.num_ffts_per_batch in the shipped
configs). So:

    samples_per_frame = num_ffts_per_batch * actual_fft_size
    frame_budget_s    = samples_per_frame / sample_rate      # real-time deadline

If a detector's per-frame latency exceeds ``frame_budget_s`` it cannot keep up with
the live stream at that rate -- this is the horizontal "real-time threshold" drawn on
the latency histograms.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

# Deployed reference constants (fft_runtime_config.hpp:17-18, and the shipped
# config_*_single_channel.yaml fft: blocks). Keep in sync with those.
DEFAULT_REFERENCE_SPAN_HZ = 500.0e6
DEFAULT_REFERENCE_FFT_SIZE = 20480
DEFAULT_PACKET_SAMPLES = 1024
DEFAULT_NUM_FFTS_PER_BATCH = 512


@dataclass(frozen=True)
class FrameGeometry:
    """Per-rate frame geometry + real-time budget for one sample rate."""
    sample_rate_hz: float
    actual_fft_size: int          # freq bins per FFT row (== spectrogram width)
    num_ffts_per_batch: int       # FFT rows per frame (spectrogram height)
    samples_per_frame: int        # complex samples consumed per frame
    resolution_hz: float          # FFT bin width
    frame_budget_s: float         # real-time deadline for one frame (s)

    @property
    def frame_budget_ms(self) -> float:
        return self.frame_budget_s * 1e3

    @property
    def spectrogram_shape(self) -> tuple[int, int]:
        """(rows, cols) of the full-frame spectrogram this rate produces."""
        return (self.num_ffts_per_batch, self.actual_fft_size)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["frame_budget_ms"] = self.frame_budget_ms
        return d


def auto_fft_size(sample_rate_hz: float,
                  reference_span_hz: float = DEFAULT_REFERENCE_SPAN_HZ,
                  reference_fft_size: int = DEFAULT_REFERENCE_FFT_SIZE,
                  packet_samples: int = DEFAULT_PACKET_SAMPLES) -> int:
    """Return the FFT size the deployed pipeline would auto-select for ``sample_rate_hz``.

    Mirrors fft_runtime_config.hpp:145-166 exactly (power-of-two span snap, then
    quantize to a whole number of ``packet_samples``-sized packets, floored to one
    packet).
    """
    span_ratio = sample_rate_hz / reference_span_hz
    if not math.isfinite(span_ratio) or span_ratio <= 0.0:
        snapped_ratio = 1.0
    else:
        snapped_ratio = 2.0 ** round(math.log2(span_ratio))
    requested_fft = reference_fft_size * snapped_ratio
    packets_per_fft = max(1, round(requested_fft / packet_samples))
    return max(packet_samples, packets_per_fft * packet_samples)


def frame_geometry(sample_rate_hz: float,
                   num_ffts_per_batch: int = DEFAULT_NUM_FFTS_PER_BATCH,
                   reference_span_hz: float = DEFAULT_REFERENCE_SPAN_HZ,
                   reference_fft_size: int = DEFAULT_REFERENCE_FFT_SIZE,
                   packet_samples: int = DEFAULT_PACKET_SAMPLES) -> FrameGeometry:
    """Full per-rate geometry + real-time budget for one sample rate."""
    nfft = auto_fft_size(sample_rate_hz, reference_span_hz, reference_fft_size, packet_samples)
    samples_per_frame = num_ffts_per_batch * nfft
    return FrameGeometry(
        sample_rate_hz=float(sample_rate_hz),
        actual_fft_size=int(nfft),
        num_ffts_per_batch=int(num_ffts_per_batch),
        samples_per_frame=int(samples_per_frame),
        resolution_hz=float(sample_rate_hz) / float(nfft),
        frame_budget_s=float(samples_per_frame) / float(sample_rate_hz),
    )


def budget_for_bin_size(bin_size_hz: float,
                        num_ffts_per_batch: int = DEFAULT_NUM_FFTS_PER_BATCH) -> float:
    """Real-time frame budget (s) for a target FFT bin size (frequency resolution).

    Independent of sample rate: a frame is a fixed ``num_ffts_per_batch`` FFT rows, and each
    row spans ``1/bin_size`` seconds (fft_size samples = sample_rate/bin_size, /sample_rate =
    1/bin_size s), so

        budget = num_ffts_per_batch * (1 / bin_size) = num_ffts_per_batch / bin_size.

    A finer bin (smaller Hz -> longer FFT -> longer frame) gives MORE budget; a coarser bin
    less. This is the appropriate real-time deadline to compare against when the detector's
    frequency resolution -- not the raw sample rate -- is what's fixed.
    """
    return float(num_ffts_per_batch) / float(bin_size_hz)


def fft_flops(nfft: int, n_rows: int) -> float:
    """Approx real FLOPs for ``n_rows`` length-``nfft`` complex FFTs (radix-2 model).

    5 * N * log2(N) flops per length-N complex FFT is the standard count; this is the
    dominant arithmetic in the spectrogram front-end shared by the power-based
    detectors, and is added on top of the aten conv/matmul flops (torch's flop counter
    does not model aten::_fft_c2c)."""
    if nfft <= 1:
        return 0.0
    return 5.0 * nfft * math.log2(nfft) * n_rows


if __name__ == "__main__":
    # Quick self-check: print the table for the four eval rates.
    print(f"{'rate (MHz)':>12} {'fft_size':>9} {'samples/frame':>15} "
          f"{'res (kHz)':>10} {'budget (ms)':>12}")
    for r in (20e6, 100e6, 250e6, 500e6):
        g = frame_geometry(r)
        print(f"{r/1e6:>12.0f} {g.actual_fft_size:>9d} {g.samples_per_frame:>15d} "
              f"{g.resolution_hz/1e3:>10.2f} {g.frame_budget_ms:>12.3f}")

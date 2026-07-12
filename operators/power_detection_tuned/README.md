# Power Detection (Tuned) Operator — stronger classical power detector

A GPU-resident, **still fully classical** energy detector that keeps the same
concept as `power_detection` (energy on the power spectrogram of raw IQ) but is
tuned to this task and fixes the DSP weaknesses that made the naive baseline
artificially weak. It is intended as a *fair* traditional baseline against the
DINO detectors — strong enough that any remaining low-SNR / wideband losses are
attributable to classical DSP itself, not to a poor operating point.

## What changed vs. `power_detection`

| Concern in the naive baseline | Fix here |
|---|---|
| Rectangular window → spectral leakage from strong emitters | **Blackman-Harris window** before the FFT (`window_type`). |
| "N-sigma in dB" threshold — not Gaussian, not CFAR; z=6 needed ~31 dB SNR | **Proper CFAR on linear power**: `alpha = N·(Pfa^(-1/N) − 1)` for exponential noise → constant false-alarm rate at a chosen `pfa` (~8–9 dB above the floor at Pfa 1e-3). |
| 1-D frequency-only reference | **2-D reference window** (frequency × time) for a lower-variance noise estimate. |
| Cell-averaging only (masking on sloping floors / extended targets) | **CA / GO / SO** variants (`cfar_variant`): GO for sloping floors, SO to bite into extended-target edges. |
| Baseline frozen from first N frames; corrupted if a signal is present at start | **Adaptive per-bin temporal floor** (sigma-clipped EMA) that keeps learning and tracks drift. |
| No DC/LO-leakage handling | **DC notch** at band center (`dc_notch_bins`). |

## Signal flow

```
raw IQ -> window -> cuFFT -> |X|^2 (linear, fftshifted) + dB
       -> 2-D Pfa-CFAR (CA/GO/SO)         ─┐
       -> adaptive per-bin temporal floor ─┴─(OR in "combined")-> DetectorMaskMessage
```

Input: `("in")` raw IQ, `num_bursts x burst_size` complex.
Output: `("mask_out")` `holoscan::ops::DetectorMaskMessage` (time × frequency, `uint8` 0/255).

## Modes (`mode`)

- **`cfar`** — 2-D Pfa CFAR across frequency/time. Best for narrowband tones and
  bursts sitting on a locally estimable floor.
- **`temporal`** — exceedance vs. the adaptive per-bin temporal floor. Detects
  wideband/extended signals a frequency-CFAR self-masks on, and tracks drift.
- **`combined`** (default) — logical OR of both; the strongest classical config.

All thresholds are statistical (`pfa`, sigma), so no per-file dB calibration is
required. All computation stays on the GPU.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `burst_size` / `num_bursts` | — | Raw-IQ geometry (injected by the app). |
| `num_channels` / `channel_filter` | 1 / 0 | Routing. |
| `mode` | `combined` | `cfar` \| `temporal` \| `combined`. |
| `window_type` | `blackman_harris` | `blackman_harris` \| `hann` \| `none`. |
| `cfar_variant` | `go` | `ca` \| `go` \| `so`. |
| `pfa` | 1e-3 | Target CFAR false-alarm probability. |
| `guard_freq` / `train_freq` | 4 / 24 | CFAR guard/training half-widths in frequency (bins). |
| `guard_time` / `train_time` | 1 / 4 | CFAR guard/training half-widths in time (rows). |
| `temporal_zscore` | 5.0 | Temporal-floor detection threshold (sigma). |
| `temporal_alpha` | 0.05 | Temporal-floor EMA update rate. |
| `temporal_clip_z` | 3.0 | Sigma-clip guard on the floor update. |
| `min_std_db` | 0.5 | Floor on the std estimate. |
| `warmup_frames` | 2 | Frames to learn the temporal floor before it detects. |
| `dc_notch_bins` | 4 | Bins each side of band center forced to no-detect. |
| `emit_stride` | 1 | Emit one mask every N frames. |

## Notes / future work

The 2-D CFAR uses a direct windowed reference sum (O(window) per cell). For live
real-time at the largest windows, replace it with a separable integral-image
(summed-area) estimate for O(1) per cell. The per-emit `cudaMalloc` of the owned
mask buffer could be pooled if it shows up in profiling.

## Build

Registered via `add_holohub_operator(power_detection_tuned)` in
`operators/CMakeLists.txt`. Built as part of the
`usrp_wideband_signal_detection` container app.

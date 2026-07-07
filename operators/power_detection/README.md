# Power Detection Operator (traditional power-detector baseline)

A GPU-resident, system-agnostic **energy detector** intended as a traditional
baseline to compare against the DINO image-segmentation detectors. It is
deliberately simple so that its failure modes (weak/low-SNR signals, signals
buried in a sloping noise floor, wideband signals wider than the CFAR window,
non-stationary interference) are easy to demonstrate.

## Signal flow

The operator consumes **raw IQ** — the same `std::tuple<matx::tensor_t<complex,2>,
cudaStream_t>` the FFT operator consumes — and performs its own FFT internally,
so it does **not** depend on the pipeline `FFT` operator. Everything stays on the
GPU: `raw IQ -> cuFFT -> |X|^2 in dB (fftshifted) -> statistical threshold ->
DetectorMaskMessage`. The emitted mask matches the DINO detectors' output
contract, so downstream visualization and offline comparison tooling treat it as
a drop-in peer.

Input: `("in")` raw IQ, `num_bursts x burst_size` complex.
Output: `("mask_out")` `holoscan::ops::DetectorMaskMessage` (time x frequency, `uint8` 0/255).

## Thresholding modes (no absolute dB thresholds — see `threshold_mode`)

- **`moving_average`** (default): a cell-averaging CFAR-style rule. For each
  frequency bin the local mean/std are estimated from training bins on both
  sides (skipping `guard_bins`), and the bin is flagged when its power exceeds
  the local mean by more than `zscore_threshold` standard deviations. Stateless
  per frame.
- **`baseline`**: accumulates a per-bin noise-floor baseline (running mean/var)
  over the first `baseline_frames` frames, then flags exceedance against that
  baseline in z-score units. Adapts to the system's own noise floor; requires a
  short warm-up during which no mask is emitted.

Because detection is expressed in standard deviations relative to a locally- or
temporally-estimated noise floor, no per-file calibration is required.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `burst_size` | — | Complex samples per burst / FFT length (match the raw-IQ producer). |
| `num_bursts` | — | Time rows per frame (match the raw-IQ producer). |
| `num_channels` | 1 | Pipeline channel count (routing validation). |
| `channel_filter` | 0 | Channel index this instance handles. |
| `threshold_mode` | `moving_average` | `moving_average` or `baseline`. |
| `zscore_threshold` | 6.0 | Detection threshold in standard deviations (N-sigma). |
| `moving_average_window` | 64 | CFAR training half-width in bins (`moving_average`). |
| `guard_bins` | 4 | Guard bins each side of the cell under test (`moving_average`). |
| `baseline_frames` | 16 | Warm-up frames folded into the baseline (`baseline`). |
| `min_std_db` | 0.5 | Floor on the std estimate to avoid divide-by-noise. |
| `emit_stride` | 1 | Emit one mask every N frames. |

## Build

Registered via `add_holohub_operator(power_detection)` in `operators/CMakeLists.txt`.
Built as part of the `usrp_wideband_signal_detection` container app; see that
app's README and the repo build scripts.

// SPDX-FileCopyrightText: 2026 The University of Texas at Austin
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cufft.h>
#include <matx.h>

#include "holoscan/holoscan.hpp"

using namespace matx;

namespace holoscan::ops {

using complex = cuda::std::complex<float>;

// Raw-IQ input: (num_bursts x burst_size) complex samples, plus the owning CUDA stream.
using power_detection_tuned_in_t = std::tuple<matx::tensor_t<complex, 2>, cudaStream_t>;

// Tuned traditional power-detector baseline (still 100% classical DSP -- no ML).
//
// This is a stronger, task-aware version of the `power_detection` baseline. It
// keeps the same concept (energy detection on the power spectrogram of raw IQ)
// but fixes the DSP weaknesses that made the naive baseline artificially poor:
//
//   1. Window before the FFT (Blackman-Harris by default) so strong emitters do
//      not leak across bins and corrupt the noise estimate.
//   2. A *proper* constant-false-alarm-rate (CFAR) detector on LINEAR power with
//      the textbook threshold multiplier alpha = N*(Pfa^(-1/N) - 1) for
//      exponentially distributed noise power, instead of a fixed "N-sigma in dB"
//      rule (which is neither Gaussian nor CFAR and forced ~31 dB SNR at z=6).
//   3. A 2-D CFAR reference window (frequency x time) with CA / GO / SO variants:
//      GO suppresses false alarms on a sloping noise floor; SO reduces
//      self-masking so the detector can bite into the edges of extended signals.
//   4. An adaptive per-bin temporal noise floor (sigma-clipped EMA) that keeps
//      learning and tracks drift, instead of a floor frozen from the first N
//      frames. Its per-bin, over-time reference detects wideband/extended signals
//      that a frequency-only CFAR self-masks on.
//   5. A DC/LO-leakage notch at band center.
//
// `mode` selects `cfar`, `temporal`, or `combined` (default = logical OR of both,
// which is the strongest classical configuration: frequency-CFAR for narrowband
// bursts + temporal floor for wideband extended signals). All thresholds are
// statistical (Pfa / sigma) so no per-file dB calibration is required, and all
// computation stays on the GPU. Emits the shared DetectorMaskMessage.
class PowerDetectionTuned : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PowerDetectionTuned)

  PowerDetectionTuned() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
  void stop() override;

 private:
  struct DeviceState {
    float* window = nullptr;       // burst_size (analysis window taps)
    complex* windowed = nullptr;   // num_bursts x burst_size (windowed IQ)
    complex* fft_out = nullptr;    // num_bursts x burst_size (post-FFT, unshifted)
    float* power_lin = nullptr;    // num_bursts x burst_size (fftshifted linear power)
    float* power_db = nullptr;     // num_bursts x burst_size (fftshifted dB)
    float* ema_mean = nullptr;     // burst_size (per-bin temporal mean, dB)
    float* ema_var = nullptr;      // burst_size (per-bin temporal variance, dB^2)
    cufftHandle fft_plan = 0;
    bool fft_plan_initialized = false;
    bool ema_initialized = false;
    bool allocated = false;
  };

  void ensure_allocated(cudaStream_t stream);
  void free_device_state();

  DeviceState state_;
  uint64_t frame_number_ = 0;
  uint64_t detections_emitted_ = 0;

  // Geometry (must match the raw-IQ producer).
  Parameter<int> burst_size_;
  Parameter<int> num_bursts_;
  Parameter<int> num_channels_;
  Parameter<int> channel_filter_;

  // Detection controls (all statistical / system-agnostic).
  Parameter<std::string> mode_;          // "cfar" | "temporal" | "combined"
  Parameter<std::string> window_type_;   // "blackman_harris" | "hann" | "none"
  Parameter<std::string> cfar_variant_;  // "ca" | "go" | "so"
  Parameter<float> pfa_;                 // target CFAR false-alarm probability
  Parameter<int> guard_freq_;            // guard bins each side (frequency)
  Parameter<int> train_freq_;            // training bins each side (frequency)
  Parameter<int> guard_time_;            // guard rows each side (time)
  Parameter<int> train_time_;            // training rows each side (time)
  Parameter<float> temporal_zscore_;     // temporal-floor detection threshold (sigma)
  Parameter<float> temporal_alpha_;      // EMA update rate for the temporal floor
  Parameter<float> temporal_clip_z_;     // sigma-clip guard on the floor update
  Parameter<float> min_std_db_;          // floor on the std estimate (dB)
  Parameter<int> warmup_frames_;         // frames to learn the temporal floor before it detects
  Parameter<int> dc_notch_bins_;         // bins each side of band center forced to no-detect
  Parameter<int> emit_stride_;           // emit one mask every N frames
};

}  // namespace holoscan::ops

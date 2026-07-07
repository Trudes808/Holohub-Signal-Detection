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
using power_detection_in_t = std::tuple<matx::tensor_t<complex, 2>, cudaStream_t>;

// Traditional power-detector baseline.
//
// Consumes raw IQ (the same source the FFT operator consumes), performs its own
// FFT internally, forms a power spectrogram in dB, and flags energy using a
// system-agnostic statistical rule (no fixed dB thresholds). Two thresholding
// modes are selectable at config time:
//   - "moving_average": per-frequency CA-CFAR-style z-score against a local
//     moving-average/standard-deviation window across frequency (stateless).
//   - "baseline": accumulate a per-bin noise-floor baseline from the first N
//     frames, then flag exceedance in z-score units against that baseline.
//
// The operator emits a DetectorMaskMessage identical in shape to the DINO
// detectors so downstream visualization and offline comparison tooling can treat
// it as a drop-in baseline. All computation stays on the GPU.
class PowerDetection : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PowerDetection)

  PowerDetection() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
  void stop() override;

 private:
  // Per-frame device working set. One instance per operator (one operator per channel).
  struct DeviceState {
    complex* fft_out = nullptr;        // num_bursts x burst_size (post-FFT, unshifted)
    float* power_db = nullptr;         // num_bursts x burst_size (fftshifted, dB)
    double* baseline_sum = nullptr;    // burst_size (running sum of dB per bin)
    double* baseline_sumsq = nullptr;  // burst_size (running sum of dB^2 per bin)
    uint64_t baseline_samples = 0;     // number of (frame x burst) rows folded into baseline
    cufftHandle fft_plan = 0;
    bool fft_plan_initialized = false;
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
  Parameter<std::string> threshold_mode_;      // "moving_average" | "baseline"
  Parameter<float> zscore_threshold_;          // detect when (x - mean) / std > this (N sigma)
  Parameter<int> moving_average_window_;        // half-width of the CFAR training window (bins)
  Parameter<int> guard_bins_;                   // guard bins excluded around the cell under test
  Parameter<int> baseline_frames_;              // frames to accumulate before baseline detection
  Parameter<float> min_std_db_;                 // floor on the std estimate to avoid divide-by-noise
  Parameter<int> emit_stride_;                  // emit one mask every N frames
};

}  // namespace holoscan::ops

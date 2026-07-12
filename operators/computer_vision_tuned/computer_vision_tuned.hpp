// SPDX-FileCopyrightText: 2026 The University of Texas at Austin
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <matx.h>

#include "holoscan/holoscan.hpp"

using namespace matx;

namespace holoscan::ops {

using complex = cuda::std::complex<float>;

// Spectrogram input: (num_bursts x burst_size) complex FFT tensor (time x
// frequency), plus the owning CUDA stream. Same tuple the DINO detectors consume.
using computer_vision_tuned_in_t = std::tuple<matx::tensor_t<complex, 2>, cudaStream_t>;

// Tuned classical computer-vision baseline (still fully pre-ML CV).
//
// Stronger, task-aware version of `computer_vision_baseline`. Same concept
// (threshold -> morphology -> connected components on the spectrogram image) but
// with the textbook-correct choices for this kind of image:
//
//   1. Local, per-frequency-column adaptive threshold (sigma-clipped background
//      per bin) instead of a single GLOBAL threshold. Fixes the non-flat noise
//      floor and the "one strong signal desensitizes the whole frame" failure.
//   2. Hysteresis (dual) thresholding: a high threshold seeds detections and a
//      low threshold grows them through connected components -- recovers faint
//      signal pixels attached to a confident core (better low-SNR completeness).
//   3. Direction-aware morphological opening (union of a horizontal-line opening
//      and a vertical-line opening) so thin wideband bursts AND thin narrowband
//      carriers survive despeckling, instead of being erased by a square element.
//   4. DC/LO-leakage notch, and blob-only output by default (edges kept as an
//      optional diagnostic rather than polluting the occupancy mask).
//
// All thresholds are statistical (sigma relative to a local background), so no
// per-file dB calibration is required. All computation stays on the GPU. Emits
// the shared DetectorMaskMessage.
class ComputerVisionTuned : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ComputerVisionTuned)

  ComputerVisionTuned() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
  void stop() override;

 private:
  struct DeviceState {
    float* db_image = nullptr;   // rows x cols (fftshifted dB)
    float* scratch_f = nullptr;  // rows x cols (Sobel gradient magnitude)
    float* col_mean = nullptr;   // cols (per-bin sigma-clipped background mean)
    float* col_std = nullptr;    // cols (per-bin sigma-clipped background std)
    uint8_t* high = nullptr;     // rows x cols (high-threshold seeds)
    uint8_t* low = nullptr;      // rows x cols (low-threshold candidates)
    uint8_t* m1 = nullptr;       // rows x cols (morphology scratch)
    uint8_t* m2 = nullptr;       // rows x cols (morphology scratch)
    uint8_t* m3 = nullptr;       // rows x cols (morphology scratch)
    uint8_t* edges = nullptr;    // rows x cols (optional edge mask)
    int32_t* labels = nullptr;   // rows x cols (connected-component labels)
    int32_t* areas = nullptr;    // rows x cols (per-root pixel counts)
    int32_t* seeds = nullptr;    // rows x cols (per-root: contains a high seed?)
    int32_t* changed = nullptr;  // CCL convergence flag
    double* stats = nullptr;     // [sum, sumsq] reduction target (edge gradient)
    bool allocated = false;
  };

  void free_device_state();

  DeviceState state_;
  uint64_t frame_number_ = 0;
  uint64_t detections_emitted_ = 0;

  Parameter<int> num_channels_;
  Parameter<int> channel_filter_;

  Parameter<float> z_high_;              // seed threshold (sigma above the local column floor)
  Parameter<float> z_low_;               // grow threshold (sigma) for hysteresis
  Parameter<float> clip_z_;              // sigma-clip for the per-column background estimate
  Parameter<int> morph_radius_;          // line structuring-element radius (bins/rows)
  Parameter<int> close_iterations_;      // morphological closing passes (fill gaps)
  Parameter<float> edge_zscore_;         // Sobel gradient-magnitude threshold (sigma), optional
  Parameter<int> min_blob_area_;         // drop connected components smaller than this (pixels)
  Parameter<int> ccl_max_iterations_;    // safety cap on label-propagation sweeps
  Parameter<float> min_std_db_;          // floor on the background std estimate
  Parameter<int> dc_notch_bins_;         // bins each side of band center forced to no-detect
  Parameter<std::string> combine_mode_;  // "blob" | "blob_or_edge" | "edge"
  Parameter<int> emit_stride_;
};

}  // namespace holoscan::ops

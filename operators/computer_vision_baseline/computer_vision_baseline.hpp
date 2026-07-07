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
// frequency), plus the owning CUDA stream. This is the same tuple the DINO
// detectors consume from the FFT/Spectrogram operators.
using computer_vision_in_t = std::tuple<matx::tensor_t<complex, 2>, cudaStream_t>;

// Traditional computer-vision baseline (classical, pre-ML).
//
// Consumes the complex spectrogram tensor, forms a dB magnitude image on the
// GPU, and detects signal regions with classical image processing only:
//   adaptive (z-score) thresholding -> morphological opening/closing ->
//   Sobel edge detection -> connected-component blob area filtering.
// It emits a DetectorMaskMessage identical in shape to the DINO detectors so it
// is a drop-in comparison baseline. Its failure modes (faint signals lost by the
// global threshold, blobs merged/split by morphology, boundaries missed by a
// fixed structuring element) are the point of the comparison. All computation
// stays on the GPU.
class ComputerVisionBaseline : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ComputerVisionBaseline)

  ComputerVisionBaseline() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
  void stop() override;

 private:
  struct DeviceState {
    float* db_image = nullptr;   // rows x cols (fftshifted dB)
    float* scratch_f = nullptr;  // rows x cols (Sobel gradient magnitude)
    uint8_t* binary = nullptr;   // rows x cols (thresholded foreground)
    uint8_t* morph_a = nullptr;  // rows x cols (morphology ping)
    uint8_t* morph_b = nullptr;  // rows x cols (morphology pong)
    uint8_t* edges = nullptr;    // rows x cols (edge mask)
    int32_t* labels = nullptr;   // rows x cols (connected-component labels)
    int32_t* areas = nullptr;    // rows x cols (per-root-label pixel counts)
    double* stats = nullptr;     // [sum, sumsq] reduction target
    int32_t* changed = nullptr;  // CCL convergence flag
    bool allocated = false;
  };

  void free_device_state();

  DeviceState state_;
  uint64_t frame_number_ = 0;
  uint64_t detections_emitted_ = 0;

  Parameter<int> num_channels_;
  Parameter<int> channel_filter_;

  Parameter<float> threshold_zscore_;   // foreground when dB > mean + z*std over the image
  Parameter<int> morph_radius_;          // structuring-element radius (bins); 1 => 3x3
  Parameter<int> open_iterations_;       // opening passes (remove speckle)
  Parameter<int> close_iterations_;      // closing passes (fill gaps)
  Parameter<float> edge_zscore_;         // Sobel gradient magnitude threshold in z-score units
  Parameter<int> min_blob_area_;         // drop connected components smaller than this (pixels)
  Parameter<int> ccl_max_iterations_;    // safety cap on label-propagation sweeps
  Parameter<std::string> combine_mode_;  // "blob" | "blob_or_edge" | "edge"
  Parameter<int> emit_stride_;
};

}  // namespace holoscan::ops

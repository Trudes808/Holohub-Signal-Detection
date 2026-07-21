// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

// Opaque TorchScript runtime handle (defined in finetuned_dino_torch_helpers.cpp so libtorch
// headers never reach nvcc -- same split cuda_dino_detector uses).
class FinetunedDinoTorchRuntime;

// Native detector operator for the fine-tuned DINOv3 segmenter (DinoSegmenter): backbone + trained
// SegHead. Emits holoscan::ops::DetectorMaskMessage on "mask_out" so signal_snipper can snip it.
//
// GEOMETRY-MATCHED FRONT-END: the model is trained at a fixed per-pixel physics
// (bin_hz = sample_rate/nfft, row_seconds = nfft/sample_rate). To reproduce that live, this operator
// taps the RAW time-domain IQ (upstream of the app's wide analysis FFT, exactly like signal_snipper)
// and computes its OWN dedicated nfft-point FFT -> dB spectrogram, matching dino_fine_tuning/src/
// finetuned_infer.mask_for_iq. The model input shape (tile_rows x nfft) is CONFIG-DRIVEN and must
// match the checkpoint's training geometry (see the exported *.meta.json geometry contract).
//
// TorchScript module contract:
//   input  float[B,1,tile_rows,nfft] in [0,1]   (channel-repeat + imagenet-norm are inside the model)
//   output logits[B,1,tile_rows,nfft]
//   post   sigmoid(logits) >= threshold  -> binary mask, tiles stitched back to (rows x nfft).
class FinetunedDinoDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FinetunedDinoDetector)

  FinetunedDinoDetector() = default;
  ~FinetunedDinoDetector() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  struct ChannelBuffers {
    void*    spec_device        = nullptr;   // rows x nfft  complex (cuda::std::complex<float>)
    float*   normalized_device  = nullptr;   // rows x nfft  ([0,1])
    float*   tile_batch_device  = nullptr;   // B x 1 x tile_rows x nfft (model input)
    float*   logits_device      = nullptr;   // B x 1 x tile_rows x nfft (model output)
    uint8_t* tile_mask_device   = nullptr;   // B x tile_rows x nfft (thresholded)
    size_t rows = 0;
    size_t nfft = 0;
    size_t tile_rows = 0;
    size_t batch = 0;
    void ensure(size_t new_rows, size_t new_nfft, size_t new_tile_rows, size_t new_batch);
    void release();
  };

  holoscan::Parameter<std::string> model_script_path_;   // path to <name>.ts (container path)
  holoscan::Parameter<double>      threshold_;           // sigmoid(logits) >= threshold
  holoscan::Parameter<int>         tile_rows_;           // model input time rows (mult of 16)
  holoscan::Parameter<int>         nfft_;                // model input freq bins (mult of 16)
  holoscan::Parameter<double>      db_vmin_;             // lower dB clip for [0,1] normalization
  holoscan::Parameter<double>      db_vmax_;             // upper dB clip
  holoscan::Parameter<int>         num_channels_;
  holoscan::Parameter<int>         channel_filter_;
  holoscan::Parameter<int>         emit_stride_;
  holoscan::Parameter<std::string> torch_dtype_;         // "fp32" | "fp16"
  // Real-time "downsample" mode: run a wide FFT (downsample_fft_size) to reproduce the app's dynamic
  // spectrogram, then bilinear-resize the freq axis down to the model width before inference (bounded,
  // rate-independent cost). false = validated native path (dedicated nfft-point FFT, tile natively).
  holoscan::Parameter<bool>        real_time_downsample_;
  holoscan::Parameter<int>         downsample_fft_size_;

  uint64_t compute_count_ = 0;
  bool startup_log_emitted_ = false;
  double inference_ms_ewma_ = 0.0;   // rolling mean of downsample inference time (ms)
  uint64_t inference_samples_ = 0;
  std::vector<uint64_t> frame_count_;
  std::vector<ChannelBuffers> channel_buffers_;
  std::shared_ptr<FinetunedDinoTorchRuntime> runtime_;

  void release_channel_buffers();
};

}  // namespace holoscan::ops

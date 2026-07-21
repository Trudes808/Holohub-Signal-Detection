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
// TorchScript module contract (see dino_fine_tuning/weights/finetuned_dino_m*.meta.json):
//   input  float[B,1,tile_rows,nfft] in [0,1]   (dB spectrogram, 256-row tiles)
//   output logits[B,1,tile_rows,nfft]
//   post   sigmoid(logits) >= threshold  -> binary mask, stitch tiles.
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
    float*   spectrogram_db_device = nullptr;   // rows x nfft  (dB)
    float*   normalized_device     = nullptr;   // rows x nfft  ([0,1])
    float*   tile_batch_device     = nullptr;   // B x 1 x tile_rows x nfft (model input)
    float*   logits_device         = nullptr;   // B x 1 x tile_rows x nfft (model output)
    uint8_t* tile_mask_device      = nullptr;   // B x tile_rows x nfft (thresholded)
    uint8_t* stitched_mask_device  = nullptr;   // rows x nfft (native)
    cudaStream_t processing_stream = nullptr;
    size_t rows = 0;
    size_t nfft = 0;
    size_t batch = 0;
  };

  holoscan::Parameter<std::string> model_script_path_;   // /workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m{1,2}.ts
  holoscan::Parameter<double>      threshold_;           // M1=0.45, M2=0.85
  holoscan::Parameter<int>         tile_rows_;           // 256
  holoscan::Parameter<int>         nfft_;                // 1024
  holoscan::Parameter<double>      db_vmin_;             // from dataset_meta
  holoscan::Parameter<double>      db_vmax_;
  holoscan::Parameter<int>         num_channels_;
  holoscan::Parameter<int>         channel_filter_;
  holoscan::Parameter<int>         emit_stride_;
  holoscan::Parameter<std::string> torch_dtype_;         // "fp32"

  uint64_t compute_count_ = 0;
  bool startup_log_emitted_ = false;
  std::vector<uint64_t> frame_count_;
  std::vector<ChannelBuffers> channel_buffers_;
  std::shared_ptr<FinetunedDinoTorchRuntime> runtime_;

  void release_channel_buffers();
};

}  // namespace holoscan::ops

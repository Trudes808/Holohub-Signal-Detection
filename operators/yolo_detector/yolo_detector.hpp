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

class YoloTorchRuntime;  // defined in yolo_torch_helpers.cpp (libtorch kept out of nvcc)

// Native detector for the fine-tuned Ultralytics YOLO26 s/m detectors. Emits DetectorMaskMessage by
// filling predicted boxes into the mask grid (box->mask convention from yolo_training/src/yolo_infer.py).
// TorchScript module (yolo_training/weights/yolo26{s,m}.torchscript): input float[1,3,imgsz,imgsz] in
// [0,1] (letterboxed RGB tile) -> Ultralytics detection head raw preds; decode + NMS in the operator.
class YoloDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(YoloDetector)

  YoloDetector() = default;
  ~YoloDetector() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  struct ChannelBuffers {
    float*   spectrogram_db_device = nullptr;   // rows x nfft dB
    uint8_t* u8_device             = nullptr;   // rows x nfft (db_to_uint8)
    float*   letterbox_batch_device= nullptr;   // B x 3 x imgsz x imgsz ([0,1])
    uint8_t* stitched_mask_device  = nullptr;   // rows x nfft
    cudaStream_t processing_stream = nullptr;
    size_t rows = 0, nfft = 0, batch = 0;
  };

  holoscan::Parameter<std::string> model_script_path_;   // .../yolo_training/weights/yolo26{s,m}.torchscript
  holoscan::Parameter<int>         imgsz_;               // 1024
  holoscan::Parameter<double>      conf_;                // 0.25
  holoscan::Parameter<double>      iou_;                 // 0.45
  holoscan::Parameter<int>         tile_rows_;           // 256
  holoscan::Parameter<int>         nfft_;                // 1024
  holoscan::Parameter<double>      db_vmin_;
  holoscan::Parameter<double>      db_vmax_;
  holoscan::Parameter<int>         num_channels_;
  holoscan::Parameter<int>         channel_filter_;
  holoscan::Parameter<int>         emit_stride_;
  holoscan::Parameter<std::string> torch_dtype_;

  uint64_t compute_count_ = 0;
  bool startup_log_emitted_ = false;
  std::vector<uint64_t> frame_count_;
  std::vector<ChannelBuffers> channel_buffers_;
  std::shared_ptr<YoloTorchRuntime> runtime_;

  void release_channel_buffers();
};

}  // namespace holoscan::ops

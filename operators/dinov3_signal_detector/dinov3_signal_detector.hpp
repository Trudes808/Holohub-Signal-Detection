// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/std/complex>
#include <holoscan/holoscan.hpp>
#include <matx.h>

#ifdef HOLOHUB_HAS_TORCH
#include <torch/script.h>
#endif

namespace holoscan::ops {

using dino_complex = cuda::std::complex<float>;
using dino_in_t = std::tuple<matx::tensor_t<dino_complex, 2>, cudaStream_t>;
using dino_out_t = std::tuple<matx::tensor_t<float, 2>, cudaStream_t>;

class DinoV3SignalDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DinoV3SignalDetector)

  DinoV3SignalDetector() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<int> input_height_;
  holoscan::Parameter<int> input_width_;
  holoscan::Parameter<int> emit_stride_;
  holoscan::Parameter<float> mask_threshold_db_;
  holoscan::Parameter<bool> log_detections_;
    holoscan::Parameter<bool> use_pytorch_backend_;
    holoscan::Parameter<std::string> inference_backend_;
    holoscan::Parameter<std::string> model_name_;
    holoscan::Parameter<std::string> model_repo_path_;
    holoscan::Parameter<std::string> weights_path_;
    holoscan::Parameter<std::string> model_script_path_;
    holoscan::Parameter<bool> strict_model_forward_;

  std::vector<uint64_t> frame_count_;
  matx::tensor_t<float, 3> detection_masks_;
    bool pytorch_runtime_ready_ = false;
    bool pytorch_warning_emitted_ = false;
    bool torchscript_model_loaded_ = false;

  #ifdef HOLOHUB_HAS_TORCH
    torch::jit::script::Module torchscript_module_;
  #endif
};

}  // namespace holoscan::ops

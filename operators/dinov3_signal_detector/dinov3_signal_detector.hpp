// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/std/complex>
#include <holoscan/holoscan.hpp>
#include <matx.h>

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

  std::vector<uint64_t> frame_count_;
  matx::tensor_t<float, 3> detection_masks_;
};

}  // namespace holoscan::ops

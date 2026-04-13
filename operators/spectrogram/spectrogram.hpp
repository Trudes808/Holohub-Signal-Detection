// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/std/complex>
#include <holoscan/holoscan.hpp>
#include <matx.h>

namespace holoscan::ops {

using complex = cuda::std::complex<float>;
using in_t = std::tuple<matx::tensor_t<complex, 2>, cudaStream_t>;
using out_t = in_t;

class Spectrogram : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(Spectrogram)

  Spectrogram() = default;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<bool> enable_save_;
    holoscan::Parameter<bool> enable_tensor_save_;
  holoscan::Parameter<int> save_every_n_frames_;
  holoscan::Parameter<int> max_images_per_channel_;
  holoscan::Parameter<int> output_height_;
  holoscan::Parameter<int> output_width_;
  holoscan::Parameter<std::string> output_dir_;
    holoscan::Parameter<std::string> tensor_output_dir_;

  std::vector<uint64_t> frame_count_;
  std::vector<int> images_saved_;
};

}  // namespace holoscan::ops

// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include <matx.h>
#include "holoscan/holoscan.hpp"

using namespace matx;

using complex = cuda::std::complex<float>;

namespace holoscan::ops {
class FFT : public Operator {
 public:
     HOLOSCAN_OPERATOR_FORWARD_ARGS(FFT)

     FFT() = default;

     void initialize() override;
     void setup(OperatorSpec& spec) override;
     void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
     void stop() override;

 private:
     struct ChannelIngressStats {
         uint64_t samples = 0;
         uint64_t emitted_frames = 0;
         uint64_t window_samples = 0;
         uint64_t window_emitted_frames = 0;
         double total_chdr_to_fft_ms = 0.0;
         double max_chdr_to_fft_ms = 0.0;
         double window_total_chdr_to_fft_ms = 0.0;
         double window_max_chdr_to_fft_ms = 0.0;
     };

     tensor_t<complex, 3> outputs;
     std::vector<ChannelIngressStats> ingress_stats;
     std::vector<uint64_t> output_frame_count;
     Parameter<int> burst_size;
     Parameter<int> emit_stride;
     Parameter<int> num_bursts;
     Parameter<uint16_t> num_channels;
     Parameter<uint8_t> spectrum_type;
     Parameter<uint8_t> averaging_type;
     Parameter<uint8_t> window_time;
     Parameter<uint8_t> window_type;
     Parameter<uint32_t> transform_points;
     Parameter<uint32_t> window_points;
     Parameter<uint64_t> resolution;
     Parameter<uint64_t> span;
     Parameter<float> weighting_factor;
     Parameter<int32_t> f1_index;
     Parameter<int32_t> f2_index;
     Parameter<uint32_t> window_time_delta;
    Parameter<int> timing_summary_every_n;
};

}  // namespace holoscan::ops

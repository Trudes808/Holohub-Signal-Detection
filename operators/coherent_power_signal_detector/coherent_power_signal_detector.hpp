// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <chrono>
#include <cuda/std/complex>
#include <holoscan/holoscan.hpp>
#include <matx.h>
#include <string>
#include <vector>

namespace holoscan::ops {

using coherent_power_complex = cuda::std::complex<float>;
using coherent_power_in_t = std::tuple<matx::tensor_t<coherent_power_complex, 2>, cudaStream_t>;

class CoherentPowerSignalDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CoherentPowerSignalDetector)

  static constexpr size_t kTimingStageCount = 6;

  CoherentPowerSignalDetector() = default;
  ~CoherentPowerSignalDetector() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext& context) override;

 private:
  struct ChannelBuffers {
    float* power_db_device = nullptr;
    float* corrected_db_device = nullptr;
    float* time_mean_device = nullptr;
    float* freq_mean_device = nullptr;
    float* background_device = nullptr;
    float* box_filter_scratch_device = nullptr;
    float* score_device = nullptr;
    float* row_stat_device = nullptr;
    float* row_smooth_device = nullptr;
    float* frontend_reference_device = nullptr;
    float* power_db_host = nullptr;
    uint8_t* mask_device = nullptr;
    uint8_t* scratch_mask_device = nullptr;
    uint8_t* mask_host = nullptr;
    size_t frame_elements = 0;
    size_t row_elements = 0;
    size_t mask_elements = 0;
  };

  struct ChannelTimingStats {
    uint64_t window_frames = 0;
    std::array<double, kTimingStageCount> total_ms {};
    std::array<double, kTimingStageCount> max_ms {};
  };

  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<int> input_height_;
  holoscan::Parameter<int> input_width_;
  holoscan::Parameter<int> emit_stride_;
  holoscan::Parameter<int> channel_filter_;
  holoscan::Parameter<bool> log_detections_;
  holoscan::Parameter<std::string> backend_mode_;
  holoscan::Parameter<bool> enable_mask_save_;
  holoscan::Parameter<int> save_every_n_frames_;
  holoscan::Parameter<int> max_masks_per_channel_;
  holoscan::Parameter<std::string> output_dir_;
  holoscan::Parameter<double> chunk_bandwidth_hz_;
  holoscan::Parameter<double> chunk_overlap_hz_;
  holoscan::Parameter<double> uncalibrated_chunk_fraction_;
  holoscan::Parameter<double> uncalibrated_overlap_fraction_;
  holoscan::Parameter<double> ignore_sideband_percent_;
  holoscan::Parameter<double> ignore_sideband_hz_;
  holoscan::Parameter<double> frontend_row_q_;
  holoscan::Parameter<double> frontend_reference_q_;
  holoscan::Parameter<double> frontend_smooth_sigma_;
  holoscan::Parameter<double> frontend_max_boost_db_;
  holoscan::Parameter<double> coherence_weight_;
  holoscan::Parameter<double> power_weight_;
  holoscan::Parameter<double> coherence_power_support_q_;
  holoscan::Parameter<double> coherence_power_q_;
  holoscan::Parameter<int> min_component_size_;
  holoscan::Parameter<double> fast_power_floor_db_;
  holoscan::Parameter<double> fast_power_span_db_;
  holoscan::Parameter<double> fast_coherence_floor_db_;
  holoscan::Parameter<double> fast_coherence_span_db_;
  holoscan::Parameter<double> fast_score_threshold_;
  holoscan::Parameter<int> fast_time_smooth_radius_;
  holoscan::Parameter<int> fast_freq_smooth_radius_;
  holoscan::Parameter<int> fast_background_freq_radius_;
  holoscan::Parameter<int> fast_background_time_radius_;
  holoscan::Parameter<int> fast_mask_smooth_iterations_;
  holoscan::Parameter<double> grouping_seed_score_q_;
  holoscan::Parameter<int> grouping_bridge_freq_px_;
  holoscan::Parameter<int> grouping_bridge_time_px_;
  holoscan::Parameter<int> grouping_min_component_size_;
  holoscan::Parameter<int> grouping_min_freq_span_px_;
  holoscan::Parameter<int> grouping_min_time_span_px_;
  holoscan::Parameter<double> grouping_min_density_;
  holoscan::Parameter<bool> timing_summary_enable_;
  holoscan::Parameter<int> timing_summary_every_n_;
  holoscan::Parameter<int> timing_summary_window_;

  std::vector<uint64_t> frame_count_;
  std::vector<int> masks_saved_;
  std::vector<ChannelTimingStats> timing_stats_;
  std::vector<ChannelBuffers> channel_buffers_;
};

}  // namespace holoscan::ops
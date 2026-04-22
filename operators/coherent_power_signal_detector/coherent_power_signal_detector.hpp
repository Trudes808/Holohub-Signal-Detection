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

struct CoherentPowerReferenceConfig {
  int input_height = 256;
  int input_width = 512;
  double chunk_bandwidth_hz = 25.0e6;
  double chunk_overlap_hz = 6.25e6;
  double uncalibrated_chunk_fraction = 0.40;
  double uncalibrated_overlap_fraction = 0.20;
  double ignore_sideband_percent = 0.0;
  double ignore_sideband_hz = 7.0e6;
  double frontend_row_q = 25.0;
  double frontend_reference_q = 75.0;
  double frontend_smooth_sigma = 12.0;
  double frontend_max_boost_db = 12.0;
  double coherence_weight = 0.55;
  double power_weight = 0.45;
  double coherence_power_support_q = 0.82;
  double coherence_power_q = 0.92;
  int min_component_size = 6;
  bool filter_detection_mask = true;
  double grouping_seed_score_q = 0.72;
  int grouping_bridge_freq_px = 33;
  int grouping_bridge_time_px = 5;
  int grouping_min_component_size = 24;
  int grouping_min_freq_span_px = 18;
  int grouping_min_time_span_px = 2;
  double grouping_min_density = 0.06;
  double grouping_time_continuity_ratio = 0.85;
};

struct CoherentPowerReferenceResult {
  int src_rows = 0;
  int src_cols = 0;
  int dst_rows = 0;
  int dst_cols = 0;
  double sample_rate_hz = 0.0;
  double span_hz = 0.0;
  bool frequency_axis_calibrated = false;
  int ignore_bins_per_side = 0;
  int grouped_box_count = 0;
  float merged_threshold = 0.0f;
  float seed_threshold = 0.0f;
  std::vector<float> power_db;
  std::vector<float> corrected_sxx_db;
  std::vector<float> final_mask;
};

CoherentPowerReferenceResult run_coherent_power_reference_validation(
    const std::vector<coherent_power_complex>& input_tensor,
    int src_rows,
    int src_cols,
    double resolution_hz,
    const CoherentPowerReferenceConfig& config);

class CoherentPowerSignalDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CoherentPowerSignalDetector)

  static constexpr size_t kTimingStageCount = 6;
  static constexpr size_t kReferenceTimingStageCount = 6;
  static constexpr size_t kChunkTimingStageCount = 5;

  CoherentPowerSignalDetector() = default;
  ~CoherentPowerSignalDetector() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext& context) override;

 private:
  struct ChannelBuffers {
    coherent_power_complex* input_tensor_host = nullptr;
    coherent_power_complex* analysis_tensor_device = nullptr;
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
    std::array<double, kReferenceTimingStageCount> reference_total_ms {};
    std::array<double, kReferenceTimingStageCount> reference_max_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_sum_total_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_sum_max_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_peak_total_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_peak_max_ms {};
  };

  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<int> input_height_;
  holoscan::Parameter<int> input_width_;
  holoscan::Parameter<int> emit_stride_;
  holoscan::Parameter<int> channel_filter_;
  holoscan::Parameter<bool> log_detections_;
  holoscan::Parameter<std::string> backend_mode_;
  holoscan::Parameter<bool> enable_mask_save_;
  holoscan::Parameter<bool> enable_tensor_snapshot_save_;
  holoscan::Parameter<int> save_every_n_frames_;
  holoscan::Parameter<int> max_masks_per_channel_;
  holoscan::Parameter<int> max_snapshots_per_channel_;
  holoscan::Parameter<std::string> output_dir_;
  holoscan::Parameter<std::string> tensor_snapshot_dir_;
  holoscan::Parameter<bool> save_power_db_snapshot_;
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
  holoscan::Parameter<bool> filter_detection_mask_;
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
  holoscan::Parameter<double> grouping_time_continuity_ratio_;
  holoscan::Parameter<bool> timing_summary_enable_;
  holoscan::Parameter<int> timing_summary_every_n_;
  holoscan::Parameter<int> timing_summary_window_;

  std::vector<uint64_t> frame_count_;
  std::vector<int> masks_saved_;
  std::vector<int> snapshots_saved_;
  std::vector<ChannelTimingStats> timing_stats_;
  std::vector<ChannelBuffers> channel_buffers_;
};

}  // namespace holoscan::ops
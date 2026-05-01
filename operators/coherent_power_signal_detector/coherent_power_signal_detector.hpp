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
  double frontend_signal_cap_db = 6.0;
  double coherence_weight = 0.55;
  double power_weight = 0.45;
  std::string power_assist_mode = "absolute_direct";
  double power_floor_time_q = 25.0;
  double power_floor_global_q = 30.0;
  double power_excess_start_db = 3.0;
  double power_excess_full_db = 15.0;
  double power_local_blend = 0.25;
  std::string coherence_source_mode = "power_assist";
  double coherence_gate_start = 0.15;
  double coherence_gate_full = 0.45;
  double coherence_bridge_bias = 0.05;
  double coherence_power_joint_weight = 0.70;
  std::string score_threshold_mode = "quantile";
  double fixed_score_threshold = 0.58;
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
  struct DetectionBoxRecord {
    int freq_start = 0;
    int freq_stop = 0;
    int time_start = 0;
    int time_stop = 0;
    int freq_span = 0;
    int time_span = 0;
    int filled_area = 0;
    float density = 0.0f;
    float bbox_density = 0.0f;
    float envelope_density = 0.0f;
    float score_mean = 0.0f;
    float score_peak = 0.0f;
    std::string split_role = "unsplit";
    bool split_applied = false;
    int parent_component_id = -1;
    std::vector<int> source_chunk_indices;
  };

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
  std::vector<float> merged_coherence;
  std::vector<float> merged_power;
  std::vector<float> merged_score;
  std::vector<float> raw_projected_mask;
  std::vector<float> final_mask;
  std::vector<DetectionBoxRecord> grouped_boxes;
};

CoherentPowerReferenceResult run_coherent_power_reference_validation(
    const std::vector<coherent_power_complex>& input_tensor,
    int src_rows,
    int src_cols,
    double resolution_hz,
    const CoherentPowerReferenceConfig& config);

CoherentPowerReferenceResult run_coherent_power_live_validation(
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
  static constexpr size_t kPowerSupportTimingStageCount = 4;

  CoherentPowerSignalDetector() = default;
  ~CoherentPowerSignalDetector() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;

  void compute(holoscan::InputContext& op_input,
             holoscan::OutputContext& op_output,
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
    uint8_t* always_on_stripe_flags_device = nullptr;
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
    uint64_t grouped_box_count_total = 0;
    uint64_t grouped_box_count_max = 0;
    uint64_t emitted_mask_nonzero_total = 0;
    uint64_t emitted_mask_nonzero_max = 0;
    uint64_t final_mask_nonzero_total = 0;
    uint64_t final_mask_nonzero_max = 0;
    uint64_t final_mask_component_count_total = 0;
    uint64_t final_mask_component_count_max = 0;
    uint64_t final_mask_grouped_box_count_total = 0;
    uint64_t final_mask_grouped_box_count_max = 0;
    double fft_to_detector_enter_total_ms = 0.0;
    double fft_to_detector_enter_max_ms = 0.0;
    double fft_to_detector_done_total_ms = 0.0;
    double fft_to_detector_done_max_ms = 0.0;
    std::array<double, kTimingStageCount> total_ms {};
    std::array<double, kTimingStageCount> max_ms {};
    std::array<double, kReferenceTimingStageCount> reference_total_ms {};
    std::array<double, kReferenceTimingStageCount> reference_max_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_sum_total_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_sum_max_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_peak_total_ms {};
    std::array<double, kChunkTimingStageCount> chunk_stage_peak_max_ms {};
    std::array<double, kPowerSupportTimingStageCount> power_support_stage_total_ms {};
    std::array<double, kPowerSupportTimingStageCount> power_support_stage_max_ms {};
  };

  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<int> input_height_;
  holoscan::Parameter<int> input_width_;
  holoscan::Parameter<int> emit_stride_;
  holoscan::Parameter<int> channel_filter_;
  holoscan::Parameter<bool> log_detections_;
  holoscan::Parameter<bool> fast_performance_;
  holoscan::Parameter<bool> save_performance_path_artifacts_;
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
  holoscan::Parameter<double> frontend_signal_cap_db_;
  holoscan::Parameter<double> coherence_weight_;
  holoscan::Parameter<double> power_weight_;
  holoscan::Parameter<double> power_floor_time_q_;
  holoscan::Parameter<double> power_floor_global_q_;
  holoscan::Parameter<double> power_excess_start_db_;
  holoscan::Parameter<double> power_excess_full_db_;
  holoscan::Parameter<double> power_local_blend_;
  holoscan::Parameter<double> coherence_gate_start_;
  holoscan::Parameter<double> coherence_gate_full_;
  holoscan::Parameter<double> coherence_bridge_bias_;
  holoscan::Parameter<double> coherence_power_joint_weight_;
  holoscan::Parameter<std::string> score_threshold_mode_;
  holoscan::Parameter<double> fixed_score_threshold_;
  holoscan::Parameter<double> coherence_power_support_q_;
  holoscan::Parameter<double> coherence_power_q_;
  holoscan::Parameter<int> min_component_size_;
  holoscan::Parameter<bool> filter_detection_mask_;
  holoscan::Parameter<double> fast_power_floor_db_;
  holoscan::Parameter<double> fast_power_span_db_;
  holoscan::Parameter<double> fast_coherence_floor_db_;
  holoscan::Parameter<double> fast_coherence_span_db_;
  holoscan::Parameter<double> fast_score_threshold_;
  holoscan::Parameter<int> live_emit_mask_rows_;
  holoscan::Parameter<int> live_emit_mask_cols_;
  holoscan::Parameter<double> live_emit_mask_min_coverage_;
  holoscan::Parameter<int> live_emit_freq_persistence_window_;
  holoscan::Parameter<int> live_emit_freq_persistence_min_hits_;
  holoscan::Parameter<bool> live_emit_always_on_enable_;
  holoscan::Parameter<int> live_emit_always_on_row_mean_stride_;
  holoscan::Parameter<double> live_emit_always_on_low_quantile_;
  holoscan::Parameter<double> live_emit_always_on_excess_db_;
  holoscan::Parameter<double> live_emit_always_on_min_time_coverage_;
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
  std::vector<int> path_artifacts_saved_;
  std::vector<ChannelTimingStats> timing_stats_;
  std::vector<ChannelBuffers> channel_buffers_;
  std::atomic<bool> stop_requested_ {false};
};

}  // namespace holoscan::ops
// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

class DinoTorchRuntime;

class CudaDinoDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(CudaDinoDetector)

  static constexpr size_t kTimingStageCount = 8;

  CudaDinoDetector() = default;
  ~CudaDinoDetector() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void stop() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override;

 private:
  struct ChannelBuffers {
    void* analysis_tensor_device = nullptr;
    float* power_db_device = nullptr;
    float* corrected_db_device = nullptr;
    float* corrected_batch_device = nullptr;
    float* dino_enhanced_batch_device = nullptr;  // matched-integration contrast input for DINO only
    float* coherence_gate_batch_device = nullptr;
    float* raw_dino_score_batch_device = nullptr;
    float* hybrid_combined_score_batch_device = nullptr;
    float* hybrid_final_mask_batch_device = nullptr;
    uint8_t* hybrid_filled_mask_batch_device = nullptr;
    uint8_t* hybrid_component_filtered_mask_batch_device = nullptr;
    // Coherence-primary fusion (Method 2) + DINO scout (Method 3) buffers.
    uint8_t* coherence_band_mask_batch_device = nullptr;   // batch_elements
    uint8_t* dino_structure_mask_batch_device = nullptr;   // batch_elements
    float* patch_qnorm_batch_device = nullptr;             // lazy: chunk_count * patch_rows * patch_cols
    float* patch_prenorm_batch_device = nullptr;           // lazy: pre-qnorm patch RMS for calibration dumps
    float* hybrid_initial_product_batch_device = nullptr;  // debug-lazy: batch_elements
    float* corrected_scout_batch_device = nullptr;         // scout: subset_count * chunk_rows * cols
    float* raw_dino_score_scout_device = nullptr;          // scout: subset_count * chunk_rows * cols
    float* patch_qnorm_scout_device = nullptr;             // scout: subset_count * patch_rows * patch_cols
    int* scout_row_starts_device = nullptr;                // scout: subset row starts (<= chunk_count)
    int* scout_plane_indices_device = nullptr;             // scout: compact->full plane map
    float* row_stat_device = nullptr;
    float* row_smooth_device = nullptr;
    float* frontend_reference_device = nullptr;
    int* chunk_row_starts_device = nullptr;
    float* per_freq_floor_packed_device = nullptr;          // chunk_count*uniform_chunk_rows (dB + offset)
    float* per_freq_gate_threshold_packed_device = nullptr; // chunk_count*uniform_chunk_rows (gate units)
    size_t per_freq_packed_capacity = 0;
    cudaStream_t processing_stream = nullptr;
    cudaEvent_t copy_complete_event = nullptr;
    cudaEvent_t coherence_start_event = nullptr;
    cudaEvent_t coherence_end_event = nullptr;
    size_t frame_elements = 0;
    size_t batch_elements = 0;
    size_t chunk_row_start_capacity = 0;
    size_t patch_qnorm_capacity = 0;
    size_t patch_prenorm_capacity = 0;
    size_t patch_qnorm_scout_capacity = 0;
    size_t scout_batch_capacity = 0;
    size_t scout_row_start_capacity = 0;
    size_t row_elements = 0;
  };

  struct ChannelTimingStats {
    uint64_t window_frames = 0;
    std::array<double, kTimingStageCount> total_ms {};
    std::array<double, kTimingStageCount> max_ms {};
  };

  holoscan::Parameter<int> num_channels_;
  holoscan::Parameter<int> input_height_;
  holoscan::Parameter<int> input_width_;
  holoscan::Parameter<int> patch_size_;
  holoscan::Parameter<int> emit_stride_;
  holoscan::Parameter<int> channel_filter_;
  holoscan::Parameter<bool> debug_mode_;
  holoscan::Parameter<bool> enable_debug_artifact_host_copy_;
  holoscan::Parameter<int> debug_chunk_index_;
  holoscan::Parameter<std::string> debug_artifact_output_dir_;
  holoscan::Parameter<bool> save_coherence_stats_;
  holoscan::Parameter<std::string> coherence_stats_dir_;
  holoscan::Parameter<bool> per_freq_floor_enable_;
  holoscan::Parameter<std::string> per_freq_floor_path_;
  holoscan::Parameter<double> per_freq_floor_offset_db_;
  holoscan::Parameter<bool> per_freq_gate_threshold_enable_;
  holoscan::Parameter<std::string> per_freq_gate_threshold_path_;
  holoscan::Parameter<bool> save_aligned_spectrogram_preview_;
  holoscan::Parameter<bool> save_aligned_spectrogram_tensor_;
  holoscan::Parameter<int> aligned_spectrogram_output_height_;
  holoscan::Parameter<int> aligned_spectrogram_output_width_;
  holoscan::Parameter<std::string> aligned_spectrogram_output_dir_;
  holoscan::Parameter<std::string> execution_strategy_;
  holoscan::Parameter<int> max_tokens_per_inference_;
  holoscan::Parameter<double> chunk_bandwidth_hz_;
  holoscan::Parameter<double> chunk_overlap_hz_;
  holoscan::Parameter<double> uncalibrated_chunk_fraction_;
  holoscan::Parameter<double> uncalibrated_overlap_fraction_;
  holoscan::Parameter<double> ignore_sideband_hz_;
  holoscan::Parameter<bool> frontend_correction_enable_;
  holoscan::Parameter<double> frontend_correction_row_q_;
  holoscan::Parameter<double> frontend_correction_smooth_sigma_;
  holoscan::Parameter<double> frontend_correction_reference_q_;
  holoscan::Parameter<double> frontend_correction_max_boost_db_;
  holoscan::Parameter<double> frontend_correction_soft_knee_db_;
  holoscan::Parameter<double> frontend_correction_edge_taper_fraction_;
  holoscan::Parameter<double> frontend_correction_edge_taper_sigma_;
  holoscan::Parameter<double> frontend_correction_edge_target_drop_db_;
  holoscan::Parameter<bool> dino_input_enhance_;
  holoscan::Parameter<int> dino_input_enhance_freq_bins_;
  holoscan::Parameter<int> dino_input_enhance_time_bins_;
  holoscan::Parameter<double> dino_gray_local_resid_weight_;
  holoscan::Parameter<bool> dino_colormap_enable_;
  holoscan::Parameter<double> dino_coherence_gate_floor_;
  holoscan::Parameter<double> dino_coherence_gate_span_db_;
  holoscan::Parameter<bool> dino_coherence_rescue_enable_;
  holoscan::Parameter<int> dino_coherence_rescue_min_area_px_;
  holoscan::Parameter<double> dino_coherence_rescue_threshold_;
  holoscan::Parameter<double> dino_coherence_rescue_floor_strength_;
  holoscan::Parameter<std::string> hybrid_fusion_mode_;
  holoscan::Parameter<double> coherence_band_threshold_;
  holoscan::Parameter<double> coherence_band_threshold_quantile_;
  holoscan::Parameter<int> coherence_band_open_time_px_;
  holoscan::Parameter<int> coherence_band_close_freq_px_;
  holoscan::Parameter<int> coherence_band_close_time_px_;
  holoscan::Parameter<int> coherence_band_min_area_px_;
  holoscan::Parameter<double> dino_structure_threshold_quantile_;
  holoscan::Parameter<int> dino_structure_open_len_;
  holoscan::Parameter<bool> coherence_primary_include_legacy_mask_;
  holoscan::Parameter<std::string> coherence_primary_legacy_score_;
  holoscan::Parameter<double> dino_contribution_strength_;
  holoscan::Parameter<bool> dino_scout_enable_;
  holoscan::Parameter<double> dino_scout_coverage_threshold_;
  holoscan::Parameter<float> raw_dino_positional_deweight_;
  holoscan::Parameter<bool> save_raw_dino_patch_prenorm_;
  holoscan::Parameter<bool> save_raw_dino_patch_features_;
  holoscan::Parameter<std::string> raw_dino_positional_template_path_;
  holoscan::Parameter<double> raw_dino_positional_template_strength_;
  holoscan::Parameter<std::string> raw_dino_positional_mu_path_;
  holoscan::Parameter<double> power_q_;
  holoscan::Parameter<std::string> hybrid_torch_dtype_;
  holoscan::Parameter<bool> enable_mask_post_processing_;
  holoscan::Parameter<int> hybrid_component_min_size_;
  holoscan::Parameter<int> grouping_bridge_freq_px_;
  holoscan::Parameter<int> grouping_bridge_time_px_;
  holoscan::Parameter<int> grouping_min_component_size_;
  holoscan::Parameter<int> grouping_min_freq_span_px_;
  holoscan::Parameter<int> grouping_min_time_span_px_;
  holoscan::Parameter<double> grouping_min_density_;
  holoscan::Parameter<bool> filter_detection_mask_;
  holoscan::Parameter<bool> emit_grouped_merged_mask_;
  holoscan::Parameter<double> grouping_time_continuity_ratio_;
  holoscan::Parameter<std::string> backend_mode_;
  holoscan::Parameter<std::string> inference_backend_;
  holoscan::Parameter<std::string> model_script_path_;
  holoscan::Parameter<std::string> torchscript_init_mode_;
  holoscan::Parameter<std::string> torch_dtype_;
  holoscan::Parameter<std::vector<double>> imagenet_mean_;
  holoscan::Parameter<std::vector<double>> imagenet_std_;
  holoscan::Parameter<bool> timing_summary_enable_;
  holoscan::Parameter<int> timing_summary_every_n_;
  holoscan::Parameter<int> timing_summary_window_;

  uint64_t compute_count_ = 0;
  bool startup_log_emitted_ = false;
  uint64_t artifact_dump_count_ = 0;
  uint64_t coherence_stats_dump_count_ = 0;
  // Calibrated per-frequency (global, length src_rows) arrays for the coherence gate.
  std::vector<float> per_freq_floor_global_;          // dB; box-mean background capped at this
  std::vector<float> per_freq_gate_threshold_global_; // gate units; per-row band threshold
  bool per_freq_floor_loaded_ = false;
  bool per_freq_floor_failed_ = false;
  bool per_freq_gate_threshold_loaded_ = false;
  bool per_freq_gate_threshold_failed_ = false;
  std::vector<uint64_t> frame_count_;
  std::vector<uint64_t> skipped_partial_batches_;
  std::vector<uint64_t> skipped_stride_frames_;
  std::vector<ChannelBuffers> channel_buffers_;
  std::vector<ChannelTimingStats> timing_stats_;
  std::shared_ptr<DinoTorchRuntime> runtime_;

  void release_channel_buffers();
};

}  // namespace holoscan::ops
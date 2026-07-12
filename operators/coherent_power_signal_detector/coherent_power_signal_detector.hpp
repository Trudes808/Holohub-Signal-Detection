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
    float* power_db_host = nullptr;
    uint8_t* mask_device = nullptr;
    uint8_t* scratch_mask_device = nullptr;
    uint8_t* strong_mask_device = nullptr;
    uint8_t* strong_scratch_device = nullptr;
    float* strong_row_floor_device = nullptr;
    // Dynamic per-frequency floor (dB), one value per frequency row. In "dynamic" mode this holds the
    // published floor = minimum over a small ring of sub-window minima, so each bin tracks its noise
    // floor while stale lows age out (bounded creep); re-seeded to a high bar on reset / retune.
    float* dynamic_floor_device = nullptr;
    // Ring of per-bin sub-window minima, laid out row-major as [row * window_slots + slot]. Each slot
    // accumulates the running min of the per-row statistic over dynamic_floor_slot_frames frames; the
    // published floor is the min across all slots. Rotating slots gives a sliding window of
    // window_slots * slot_frames frames without storing every frame.
    float* dynamic_floor_ring_device = nullptr;
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
  holoscan::Parameter<int> max_snapshots_per_channel_;
  holoscan::Parameter<std::string> output_dir_;
  holoscan::Parameter<std::string> tensor_snapshot_dir_;
  holoscan::Parameter<bool> save_power_db_snapshot_;
  holoscan::Parameter<bool> save_coherent_power_stats_;
  holoscan::Parameter<std::string> coherent_power_stats_dir_;
  holoscan::Parameter<std::string> per_freq_threshold_path_;
  holoscan::Parameter<double> per_freq_threshold_offset_db_;
  // Per-frequency floor source: "calibrated" (load .npy), "dynamic" (learn a monotone running-min
  // floor live), or "static"/empty (disable the per-frequency fill). Dynamic-mode tuning below.
  holoscan::Parameter<std::string> per_freq_threshold_mode_;
  holoscan::Parameter<double> dynamic_floor_init_db_;
  holoscan::Parameter<double> dynamic_floor_std_k_;
  holoscan::Parameter<int> dynamic_floor_warmup_frames_;
  holoscan::Parameter<int> dynamic_floor_window_slots_;
  holoscan::Parameter<int> dynamic_floor_slot_frames_;
  holoscan::Parameter<double> ignore_sideband_percent_;
  holoscan::Parameter<double> ignore_sideband_hz_;
  holoscan::Parameter<double> frontend_reference_q_;
  holoscan::Parameter<double> frontend_smooth_sigma_;
  holoscan::Parameter<double> frontend_max_boost_db_;
  holoscan::Parameter<double> frontend_signal_cap_db_;
  holoscan::Parameter<bool> filter_detection_mask_;
  holoscan::Parameter<double> fast_power_floor_db_;
  holoscan::Parameter<double> fast_power_span_db_;
  holoscan::Parameter<double> fast_score_threshold_;
  // Strong-signal rescue: OR pixels whose absolute corrected power exceeds the per-row
  // (per-frequency) noise floor by a large margin back into the emitted mask AFTER the emit
  // morphology + frequency-persistence pass, so a strong but frequency-narrow signal that the
  // width filters would otherwise erase still survives. Requires >min_time_bins strong time
  // bins so isolated impulsive spikes are not rescued.
  holoscan::Parameter<bool> fast_strong_rescue_enable_;
  holoscan::Parameter<double> fast_strong_rescue_excess_db_;
  holoscan::Parameter<int> fast_strong_rescue_min_time_bins_;
  holoscan::Parameter<int> live_emit_mask_rows_;
  holoscan::Parameter<int> live_emit_mask_cols_;
  holoscan::Parameter<int> live_emit_freq_persistence_window_;
  holoscan::Parameter<int> live_emit_freq_persistence_min_hits_;
  holoscan::Parameter<int> fast_time_smooth_radius_;
  holoscan::Parameter<int> fast_freq_smooth_radius_;
  holoscan::Parameter<int> fast_background_freq_radius_;
  holoscan::Parameter<int> fast_background_time_radius_;
  holoscan::Parameter<int> fast_mask_smooth_iterations_;
  holoscan::Parameter<bool> timing_summary_enable_;
  holoscan::Parameter<int> timing_summary_every_n_;
  holoscan::Parameter<int> timing_summary_window_;

  std::vector<uint64_t> frame_count_;
  std::vector<int> masks_saved_;
  std::vector<int> snapshots_saved_;
  std::vector<int> path_artifacts_saved_;
  std::vector<ChannelTimingStats> timing_stats_;
  std::vector<ChannelBuffers> channel_buffers_;
  std::vector<uint8_t> reset_detector_state_on_next_full_batch_;
  std::vector<uint64_t> last_seen_chdr_soft_resync_epoch_;
  // Dynamic per-frequency floor bookkeeping: last-seen tuning center frequency (a change re-seeds
  // the floor) and a per-channel "re-seed the floor to the high bar on the next dynamic frame" flag.
  std::vector<uint64_t> last_seen_center_frequency_;
  std::vector<uint8_t> dynamic_floor_seed_pending_;
  // Per-channel ring cursor: which sub-window slot is currently accumulating, and how many frames
  // have been folded into it so far (rotates to the next slot after dynamic_floor_slot_frames).
  std::vector<int> dynamic_floor_slot_;
  std::vector<int> dynamic_floor_slot_frame_;
  std::atomic<bool> stop_requested_ {false};

  // Calibrated per-frequency (per-row) noise floor in dB, shared across channels. A pixel
  // is OR-ed into the mask when corrected_db > per_freq_floor[row] + offset, letting strong
  // signals broader than the local box (whose interior the box-mean hollows out) fill in.
  float* per_freq_threshold_device_ = nullptr;
  int per_freq_threshold_len_ = 0;
  bool per_freq_threshold_ready_ = false;
  bool per_freq_threshold_failed_ = false;

  void reset_channel_state(uint16_t channel_number,
                           size_t row_elements,
                           size_t frame_elements,
                           cudaStream_t stream);
};

}  // namespace holoscan::ops
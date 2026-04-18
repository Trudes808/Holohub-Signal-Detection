// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "dinov3_torch_runtime.hpp"

#include <array>
#include <cuda/std/complex>
#include <holoscan/holoscan.hpp>
#include <matx.h>
#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

using dino_complex = cuda::std::complex<float>;
using dino_in_t = std::tuple<matx::tensor_t<dino_complex, 2>, cudaStream_t>;

class DinoV3SignalDetector : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(DinoV3SignalDetector)

    static constexpr size_t kTimingStageCount = 8;

  DinoV3SignalDetector() = default;
    ~DinoV3SignalDetector() override;

  void setup(holoscan::OperatorSpec& spec) override;
  void initialize() override;
  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext& context) override;

 private:
    struct ChannelBuffers {
      dino_complex* analysis_tensor_device = nullptr;
      float* power_db_device = nullptr;
      float* corrected_db_device = nullptr;
      float* row_stat_device = nullptr;
      float* row_smooth_device = nullptr;
      float* frontend_reference_device = nullptr;
      float* time_mean_device = nullptr;
      float* freq_mean_device = nullptr;
      float* background_device = nullptr;
      float* box_filter_scratch_device = nullptr;
      float* coherence_gate_device = nullptr;
      float* coherence_gate_host = nullptr;
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
  holoscan::Parameter<int> patch_size_;
  holoscan::Parameter<int> emit_stride_;
  holoscan::Parameter<float> mask_threshold_db_;
  holoscan::Parameter<bool> log_detections_;
  holoscan::Parameter<bool> enable_mask_save_;
  holoscan::Parameter<int> save_every_n_frames_;
  holoscan::Parameter<int> max_masks_per_channel_;
  holoscan::Parameter<std::string> output_dir_;
  holoscan::Parameter<bool> use_pytorch_backend_;
  holoscan::Parameter<std::string> inference_backend_;
  holoscan::Parameter<std::string> model_name_;
  holoscan::Parameter<std::string> model_repo_path_;
  holoscan::Parameter<std::string> weights_path_;
  holoscan::Parameter<std::string> model_script_path_;
  holoscan::Parameter<std::string> torchscript_init_mode_;
  holoscan::Parameter<bool> strict_model_forward_;
  holoscan::Parameter<std::vector<double>> imagenet_mean_;
  holoscan::Parameter<std::vector<double>> imagenet_std_;
  holoscan::Parameter<int> fft_size_;
  holoscan::Parameter<int> noverlap_;
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
  holoscan::Parameter<double> frontend_edge_guard_floor_;
  holoscan::Parameter<double> dino_coherence_gate_floor_;
  holoscan::Parameter<double> dino_coherence_gate_span_db_;
  holoscan::Parameter<double> texture_q_;
  holoscan::Parameter<int> texture_k_;
  holoscan::Parameter<double> power_q_;
  holoscan::Parameter<int> dino_group_k_;
  holoscan::Parameter<double> dino_group_spatial_weight_;
  holoscan::Parameter<double> dino_group_score_q_;
  holoscan::Parameter<double> pipeline_final_threshold_;
  holoscan::Parameter<double> pipeline_final_threshold_no_speckle_;
  holoscan::Parameter<double> pipeline_gap_floor_;
  holoscan::Parameter<int> pipeline_component_min_size_;
  holoscan::Parameter<int> pipeline_component_min_size_no_speckle_;
  holoscan::Parameter<double> pipeline_power_rescue_floor_;
  holoscan::Parameter<double> pipeline_power_rescue_gain_;
  holoscan::Parameter<int> pipeline_strong_speckle_min_component_;
  holoscan::Parameter<double> pipeline_texture_speckle_clean_threshold_;
  holoscan::Parameter<double> pipeline_texture_speckle_strong_threshold_;
  holoscan::Parameter<bool> timing_summary_enable_;
  holoscan::Parameter<int> timing_summary_every_n_;
  holoscan::Parameter<int> timing_summary_window_;

  std::vector<uint64_t> frame_count_;
  std::vector<int> masks_saved_;
  std::vector<ChannelTimingStats> timing_stats_;
  std::vector<ChannelBuffers> channel_buffers_;
  bool pytorch_runtime_ready_ = false;
  bool pytorch_warning_emitted_ = false;

  std::unique_ptr<DinoTorchRuntime> torch_runtime_;
};

}  // namespace holoscan::ops

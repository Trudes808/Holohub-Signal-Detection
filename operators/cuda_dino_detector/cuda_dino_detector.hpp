// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <dinov3_torch_runtime.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/operator.hpp>
#include <matx.h>

#include <array>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace holoscan::ops {

using cuda_dino_complex = cuda::std::complex<float>;
using cuda_dino_in_t = std::tuple<matx::tensor_t<cuda_dino_complex, 2>, cudaStream_t>;

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
    cuda_dino_complex* analysis_tensor_device = nullptr;
    float* power_db_device = nullptr;
    float* corrected_db_device = nullptr;
    float* corrected_batch_device = nullptr;
    float* coherence_gate_batch_device = nullptr;
    float* raw_dino_score_batch_device = nullptr;
    float* hybrid_combined_score_batch_device = nullptr;
    float* hybrid_final_mask_batch_device = nullptr;
    float* row_stat_device = nullptr;
    float* row_smooth_device = nullptr;
    float* frontend_reference_device = nullptr;
    int* chunk_row_starts_device = nullptr;
    cudaStream_t processing_stream = nullptr;
    size_t frame_elements = 0;
    size_t batch_elements = 0;
    size_t chunk_row_start_capacity = 0;
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
  holoscan::Parameter<double> dino_coherence_gate_floor_;
  holoscan::Parameter<double> dino_coherence_gate_span_db_;
  holoscan::Parameter<float> raw_dino_positional_deweight_;
  holoscan::Parameter<double> power_q_;
  holoscan::Parameter<std::string> hybrid_torch_dtype_;
  holoscan::Parameter<int> hybrid_component_min_size_;
  holoscan::Parameter<int> grouping_bridge_freq_px_;
  holoscan::Parameter<int> grouping_bridge_time_px_;
  holoscan::Parameter<int> grouping_min_component_size_;
  holoscan::Parameter<int> grouping_min_freq_span_px_;
  holoscan::Parameter<int> grouping_min_time_span_px_;
  holoscan::Parameter<double> grouping_min_density_;
  holoscan::Parameter<bool> filter_detection_mask_;
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
  std::vector<ChannelBuffers> channel_buffers_;
  std::vector<ChannelTimingStats> timing_stats_;
  DinoTorchRuntime runtime_;

  void release_channel_buffers();
};

}  // namespace holoscan::ops
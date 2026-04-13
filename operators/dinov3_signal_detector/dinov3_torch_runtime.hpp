// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>

namespace holoscan::ops {

struct DinoTorchRuntimeConfig {
  std::string inference_backend;
  std::string model_script_path;
  std::string torchscript_init_mode;
  std::vector<double> imagenet_mean;
  std::vector<double> imagenet_std;
  bool return_final_mask = true;
  double ignore_sideband_hz = 0.0;
  bool frontend_correction_enable = true;
  double frontend_correction_row_q = 25.0;
  double frontend_correction_smooth_sigma = 12.0;
  double frontend_correction_reference_q = 75.0;
  double frontend_correction_max_boost_db = 12.0;
  double frontend_correction_soft_knee_db = 4.0;
  double frontend_correction_edge_taper_fraction = 0.10;
  double frontend_correction_edge_taper_sigma = 6.0;
  double frontend_correction_edge_target_drop_db = 2.5;
  double power_q = 0.90;
  double dino_group_score_q = 0.60;
  double pipeline_final_threshold = 0.20;
  double pipeline_gap_floor = 0.10;
  double pipeline_power_rescue_floor = 0.10;
  double pipeline_power_rescue_gain = 2.0;
};

struct DinoTorchRuntimeInput {
  uint16_t channel_number = 0;
  uint64_t frame_number = 0;
  int src_rows = 0;
  int src_cols = 0;
  int dst_rows = 0;
  int dst_cols = 0;
  int patch_size = 16;
  cudaStream_t cuda_stream = nullptr;
  double resolution_hz = 0.0;
  double span_hz = 0.0;
  const std::vector<float>* power_db = nullptr;
  const float* power_db_device = nullptr;
};

struct DinoTorchRuntimeTiming {
  double frontend_correction_ms = 0.0;
  double crop_align_ms = 0.0;
  double resize_ms = 0.0;
  double model_prep_ms = 0.0;
  double torch_forward_ms = 0.0;
  double dino_score_ms = 0.0;
  double power_score_ms = 0.0;
  double fusion_ms = 0.0;
};

struct DinoTorchRuntimeResult {
  bool success = false;
  std::string error_stage;
  std::string error_message;
  std::string error_detail;
  std::string backend_used = "pytorch_placeholder";
  bool torchscript_forward_ready = false;
  int ignore_bins_per_side = 0;
  double freq_bin_hz = 0.0;
  int aligned_rows = 0;
  int aligned_cols = 0;
  double dino_threshold = 0.0;
  double power_threshold = 0.0;
  double final_threshold = 0.0;
  DinoTorchRuntimeTiming timing;
  std::vector<float> final_mask;
};

class DinoTorchRuntime {
 public:
  DinoTorchRuntime();
  ~DinoTorchRuntime();

  DinoTorchRuntime(const DinoTorchRuntime&) = delete;
  DinoTorchRuntime& operator=(const DinoTorchRuntime&) = delete;
  DinoTorchRuntime(DinoTorchRuntime&&) noexcept;
  DinoTorchRuntime& operator=(DinoTorchRuntime&&) noexcept;

  DinoTorchRuntimeResult run(const DinoTorchRuntimeConfig& config, const DinoTorchRuntimeInput& input);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::ops
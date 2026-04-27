// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace holoscan::ops {

struct CudaHybridStageTiming {
    double normalization_ms = 0.0;
    double residual_stack_ms = 0.0;
    double threshold_extract_ms = 0.0;
    double closing_ms = 0.0;
    double fill_holes_ms = 0.0;
    double component_filter_ms = 0.0;
    double output_copy_ms = 0.0;
};

bool compute_fast_directional_coherence_gate_gpu_batch_to_device(const float* corrected_batch_device,
                                                                 int batch_size,
                                                                 int rows,
                                                                 int cols,
                                                                 const std::vector<uint8_t>& valid_row_mask_batch,
                                                                 float* output_gate_device,
                                                                 cudaStream_t cuda_stream);

bool compute_structure_tensor_gate_gpu_batch_to_device(const float* corrected_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       const std::vector<uint8_t>& valid_row_mask_batch,
                                                       float* output_gate_device,
                                                       cudaStream_t cuda_stream);

bool compute_deweighted_raw_dino_score_gpu_batch_to_device(const float* patch_features_batch_device,
                                                           int batch_size,
                                                           int patch_rows,
                                                           int patch_cols,
                                                           int feature_dim,
                                                           int aligned_rows,
                                                           int aligned_cols,
                                                           int output_rows,
                                                           int output_cols,
                                                           float positional_suppression,
                                                           bool resized_full_chunk,
                                                           float* output_score_device,
                                                           cudaStream_t cuda_stream);

bool project_runtime_score_batch_to_device(const float* score_maps_batch_device,
                                           int batch_size,
                                           int runtime_rows,
                                           int runtime_cols,
                                           int aligned_rows,
                                           int aligned_cols,
                                           int output_rows,
                                           int output_cols,
                                           bool resized_full_chunk,
                                           float* output_score_device,
                                           cudaStream_t cuda_stream);

bool binary_fill_holes_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                            int batch_size,
                                            int rows,
                                            int cols,
                                            uint8_t* output_mask_batch_device,
                                            cudaStream_t cuda_stream);

bool keep_large_components_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                                int batch_size,
                                                int rows,
                                                int cols,
                                                int min_size,
                                                uint8_t* output_mask_batch_device,
                                                cudaStream_t cuda_stream);

bool extract_masked_quantile_thresholds_cuda_batch_to_device(const float* input_a_batch_device,
                                                             const float* input_b_batch_device,
                                                             const float* input_c_batch_device,
                                                             const bool* valid_mask_batch_device,
                                                             int batch_size,
                                                             int rows,
                                                             int cols,
                                                             int bin_count,
                                                             float quantile_a,
                                                             float quantile_b,
                                                             float quantile_c,
                                                             float fallback,
                                                             float* output_threshold_a_device,
                                                             float* output_threshold_b_device,
                                                             float* output_threshold_c_device,
                                                             cudaStream_t cuda_stream);

bool compute_residual_veto_hybrid_gpu_batch_to_device(const float* dino_score_batch_device,
                                                      const float* coherence_batch_device,
                                                      int batch_size,
                                                      int rows,
                                                      int cols,
                                                      const std::vector<uint8_t>& valid_row_mask_batch,
                                                      bool use_fp16,
                                                      int min_component_size,
                                                      float* output_combined_score_device,
                                                      float* output_final_mask_device,
                                                      uint8_t* output_filled_mask_batch_device,
                                                      uint8_t* output_component_filtered_mask_batch_device,
                                                      cudaStream_t cuda_stream,
                                                      CudaHybridStageTiming* stage_timing = nullptr);

}  // namespace holoscan::ops
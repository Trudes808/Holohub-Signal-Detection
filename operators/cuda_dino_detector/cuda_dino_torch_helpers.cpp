// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_dino_torch_helpers.hpp"

#if __has_include(<torch/torch.h>) && __has_include(<c10/cuda/CUDAGuard.h>)

#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>

namespace holoscan::ops {

bool compute_residual_veto_native_cuda_batch_to_device(const float* dino_score_batch_device,
                                                       const float* coherence_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       const std::vector<uint8_t>& valid_row_mask_batch,
                                                       bool use_fp16,
                                                       bool enable_mask_post_processing,
                                                       int min_component_size,
                                                       float* output_combined_score_device,
                                                       float* output_final_mask_device,
                                                       uint8_t* output_filled_mask_batch_device,
                                                       uint8_t* output_component_filtered_mask_batch_device,
                                                       cudaStream_t cuda_stream,
                                                       CudaHybridStageTiming* stage_timing,
                                                       float* debug_initial_product_device,
                                                       bool combine_scores_with_max,
                                                       float dino_contribution_strength);

bool compute_structure_tensor_gate_gpu_batch_to_device(const float* corrected_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       const std::vector<uint8_t>& valid_row_mask_batch,
                                                       const float* per_row_floor_device,
                                                       float* output_gate_device,
                                                       cudaStream_t cuda_stream) {
  return compute_fast_directional_coherence_gate_gpu_batch_to_device(corrected_batch_device,
                                                                     batch_size,
                                                                     rows,
                                                                     cols,
                                                                     valid_row_mask_batch,
                                                                     per_row_floor_device,
                                                                     output_gate_device,
                                                                     cuda_stream);
}

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
                                                           cudaStream_t cuda_stream,
                                                           float* output_patch_qnorm_device,
                                                           float* output_patch_prenorm_device,
                                                           const float* positional_template_device,
                                                           float positional_template_strength) {
  return compute_deweighted_raw_dino_score_native_cuda_batch_to_device(patch_features_batch_device,
                                                                        batch_size,
                                                                        patch_rows,
                                                                        patch_cols,
                                                                        feature_dim,
                                                                        aligned_rows,
                                                                        aligned_cols,
                                                                        output_rows,
                                                                        output_cols,
                                                                        positional_suppression,
                                                                        resized_full_chunk,
                                                                        output_score_device,
                                                                        cuda_stream,
                                                                        output_patch_qnorm_device,
                                                                        output_patch_prenorm_device,
                                                                        positional_template_device,
                                                                        positional_template_strength);
}

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
                                           cudaStream_t cuda_stream) {
  return project_runtime_score_native_cuda_batch_to_device(score_maps_batch_device,
                                                           batch_size,
                                                           runtime_rows,
                                                           runtime_cols,
                                                           aligned_rows,
                                                           aligned_cols,
                                                           output_rows,
                                                           output_cols,
                                                           resized_full_chunk,
                                                           output_score_device,
                                                           cuda_stream);
}

bool compute_residual_veto_hybrid_gpu_batch_to_device(const float* dino_score_batch_device,
                                                      const float* coherence_batch_device,
                                                      int batch_size,
                                                      int rows,
                                                      int cols,
                                                      const std::vector<uint8_t>& valid_row_mask_batch,
                                                      bool use_fp16,
                                                      bool enable_mask_post_processing,
                                                      int min_component_size,
                                                      float* output_combined_score_device,
                                                      float* output_final_mask_device,
                                                      uint8_t* output_filled_mask_batch_device,
                                                      uint8_t* output_component_filtered_mask_batch_device,
                                                      cudaStream_t cuda_stream,
                                                      CudaHybridStageTiming* stage_timing,
                                                      float* debug_initial_product_device,
                                                      bool combine_scores_with_max,
                                                      float dino_contribution_strength) {
  return compute_residual_veto_native_cuda_batch_to_device(dino_score_batch_device,
                                                           coherence_batch_device,
                                                           batch_size,
                                                           rows,
                                                           cols,
                                                           valid_row_mask_batch,
                                                           use_fp16,
                                                           enable_mask_post_processing,
                                                           min_component_size,
                                                           output_combined_score_device,
                                                           output_final_mask_device,
                                                           output_filled_mask_batch_device,
                                                           output_component_filtered_mask_batch_device,
                                                           cuda_stream,
                                                           stage_timing,
                                                           debug_initial_product_device,
                                                           combine_scores_with_max,
                                                           dino_contribution_strength);
}

}  // namespace holoscan::ops

#else

namespace holoscan::ops {

bool compute_residual_veto_native_cuda_batch_to_device(const float* dino_score_batch_device,
                                                       const float* coherence_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       const std::vector<uint8_t>& valid_row_mask_batch,
                                                       bool use_fp16,
                                                       bool enable_mask_post_processing,
                                                       int min_component_size,
                                                       float* output_combined_score_device,
                                                       float* output_final_mask_device,
                                                       uint8_t* output_filled_mask_batch_device,
                                                       uint8_t* output_component_filtered_mask_batch_device,
                                                       cudaStream_t cuda_stream,
                                                       CudaHybridStageTiming* stage_timing,
                                                       float* debug_initial_product_device,
                                                       bool combine_scores_with_max,
                                                       float dino_contribution_strength);

bool compute_structure_tensor_gate_gpu_batch_to_device(const float* corrected_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       const std::vector<uint8_t>& valid_row_mask_batch,
                                                       const float* per_row_floor_device,
                                                       float* output_gate_device,
                                                       cudaStream_t cuda_stream) {
  return compute_fast_directional_coherence_gate_gpu_batch_to_device(corrected_batch_device,
                                                                     batch_size,
                                                                     rows,
                                                                     cols,
                                                                     valid_row_mask_batch,
                                                                     per_row_floor_device,
                                                                     output_gate_device,
                                                                     cuda_stream);
}

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
                                                           cudaStream_t cuda_stream,
                                                           float* output_patch_qnorm_device,
                                                           float* output_patch_prenorm_device,
                                                           const float* positional_template_device,
                                                           float positional_template_strength) {
  return compute_deweighted_raw_dino_score_native_cuda_batch_to_device(patch_features_batch_device,
                                                                        batch_size,
                                                                        patch_rows,
                                                                        patch_cols,
                                                                        feature_dim,
                                                                        aligned_rows,
                                                                        aligned_cols,
                                                                        output_rows,
                                                                        output_cols,
                                                                        positional_suppression,
                                                                        resized_full_chunk,
                                                                        output_score_device,
                                                                        cuda_stream,
                                                                        output_patch_qnorm_device,
                                                                        output_patch_prenorm_device,
                                                                        positional_template_device,
                                                                        positional_template_strength);
}

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
                                           cudaStream_t cuda_stream) {
  return project_runtime_score_native_cuda_batch_to_device(score_maps_batch_device,
                                                           batch_size,
                                                           runtime_rows,
                                                           runtime_cols,
                                                           aligned_rows,
                                                           aligned_cols,
                                                           output_rows,
                                                           output_cols,
                                                           resized_full_chunk,
                                                           output_score_device,
                                                           cuda_stream);
}

bool compute_residual_veto_hybrid_gpu_batch_to_device(const float* dino_score_batch_device,
                                                      const float* coherence_batch_device,
                                                      int batch_size,
                                                      int rows,
                                                      int cols,
                                                      const std::vector<uint8_t>& valid_row_mask_batch,
                                                      bool use_fp16,
                                                      bool enable_mask_post_processing,
                                                      int min_component_size,
                                                      float* output_combined_score_device,
                                                      float* output_final_mask_device,
                                                      uint8_t* output_filled_mask_batch_device,
                                                      uint8_t* output_component_filtered_mask_batch_device,
                                                      cudaStream_t cuda_stream,
                                                      CudaHybridStageTiming* stage_timing,
                                                      float* debug_initial_product_device,
                                                      bool combine_scores_with_max,
                                                      float dino_contribution_strength) {
  return compute_residual_veto_native_cuda_batch_to_device(dino_score_batch_device,
                                                           coherence_batch_device,
                                                           batch_size,
                                                           rows,
                                                           cols,
                                                           valid_row_mask_batch,
                                                           use_fp16,
                                                           enable_mask_post_processing,
                                                           min_component_size,
                                                           output_combined_score_device,
                                                           output_final_mask_device,
                                                           output_filled_mask_batch_device,
                                                           output_component_filtered_mask_batch_device,
                                                           cuda_stream,
                                                           stage_timing,
                                                           debug_initial_product_device,
                                                           combine_scores_with_max,
                                                           dino_contribution_strength);
}

}  // namespace holoscan::ops

#endif
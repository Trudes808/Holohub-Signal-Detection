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

// Coherence-rescue floor: a monotonic OR path that can only ADD detections in
// regions where the coherence gate is strong AND spatially extended. It never
// weakens the DINO path (disabled => byte-identical behavior), so it cannot
// regress narrowband detection. See compute_residual_veto_native_cuda_batch_to_device.
struct CoherenceRescueConfig {
    bool enable = false;            // master switch; false => current behavior
    float coherence_threshold = 0.35f;  // resolved [0,1] threshold on the raw coherence gate
    int min_area_px = 256;          // min contiguous strong-coherence component area to rescue
    float floor_strength = 0.0f;    // optional combined-score floor inside rescued pixels (0 => mask-only)
};

// Coherence-primary fusion (hybrid_fusion_mode: "coherence_primary"): the final
// mask is a union of independently derived decisions instead of the legacy
// product+residual-veto pipeline. Banded/time-continuous coherence is trusted
// outright; DINO contributes only through a structure gate (runs of >=N patches
// in a row). Thresholds on the RAW coherence gate must be anchored to its true
// ~0.1 dynamic range (measured max ~0.11).
struct CoherencePrimaryConfig {
    bool enable = false;                 // false => legacy fusion path, byte-identical
    float coherence_threshold = 0.05f;   // absolute threshold on the raw coherence gate
    float coherence_quantile = -1.0f;    // >=0 => per-chunk threshold = max(absolute, quantile)
    int open_time_px = 5;                // 1xk opening along time (cols) enforcing banded structure
    int close_freq_px = 0;               // optional closing after opening (<=1 => skip)
    int close_time_px = 0;
    int min_area_px = 256;               // keep_large_components on the band mask
    bool include_legacy_mask = true;     // OR the legacy residual-veto final mask into the union
};

// DINO structure gate: applied on the native patch grid (patch_rows x patch_cols,
// per-chunk quantile-normalized scores). A per-chunk quantile threshold keeps the
// gate self-normalizing to noise; the run-length opening (1xN time OR Nx1 freq)
// encodes the prior that >=N contiguous patch responses nearly always mean signal.
struct DinoStructureGateConfig {
    float threshold_quantile = 0.90f;    // per-chunk quantile on patch scores
    int open_len = 3;                    // run length in PATCH units (odd; realized as radius (len-1)/2)
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

bool compute_deweighted_raw_dino_score_native_cuda_batch_to_device(const float* patch_features_batch_device,
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
                                                                   float* output_patch_qnorm_device = nullptr,
                                                                   float* output_patch_prenorm_device = nullptr,
                                                                   const float* positional_template_device = nullptr,
                                                                   float positional_template_strength = 1.0f);

bool project_runtime_score_native_cuda_batch_to_device(const float* score_maps_batch_device,
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
                                                           float* output_patch_qnorm_device = nullptr,
                                                           float* output_patch_prenorm_device = nullptr,
                                                           const float* positional_template_device = nullptr,
                                                           float positional_template_strength = 1.0f);

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

bool project_aligned_maps_cuda_batch_to_device(const float* aligned_maps_batch_device,
                                               int batch_size,
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

bool binary_closing_rect_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                              int batch_size,
                                              int rows,
                                              int cols,
                                              int kernel_rows,
                                              int kernel_cols,
                                              uint8_t* output_mask_batch_device,
                                              cudaStream_t cuda_stream);

bool keep_large_components_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                                int batch_size,
                                                int rows,
                                                int cols,
                                                int min_size,
                                                uint8_t* output_mask_batch_device,
                                                cudaStream_t cuda_stream);

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
                                                      CudaHybridStageTiming* stage_timing = nullptr,
                                                      const CoherenceRescueConfig* rescue = nullptr,
                                                      float* debug_initial_product_device = nullptr,
                                                      bool combine_scores_with_max = false,
                                                      float dino_contribution_strength = 1.0f);

// Coherence band mask: threshold the RAW coherence gate (absolute + optional
// per-chunk quantile), enforce time-continuity via a 1xk opening along cols,
// optional closing, then drop small components. The result is the trusted
// coherence-primary decision (and the DINO-scout coverage source).
bool compute_coherence_band_mask_gpu_batch_to_device(const float* coherence_gate_batch_device,
                                                     int batch_size,
                                                     int rows,
                                                     int cols,
                                                     const std::vector<uint8_t>& valid_row_mask_batch,
                                                     const CoherencePrimaryConfig& config,
                                                     uint8_t* output_band_mask_batch_device,
                                                     cudaStream_t cuda_stream);

// DINO structure mask: per-chunk quantile threshold on the patch-grid scores,
// then union of a 1xN (time) and Nx1 (freq) opening — "runs of >=N patches" —
// projected nearest to the chunk grid. output_plane_indices_device maps compact
// batch index -> output plane (nullptr => identity, for full-batch operation).
bool compute_dino_structure_mask_gpu_batch_to_device(const float* patch_qnorm_batch_device,
                                                     int batch_size,
                                                     int patch_rows,
                                                     int patch_cols,
                                                     const DinoStructureGateConfig& config,
                                                     int aligned_rows,
                                                     int aligned_cols,
                                                     int output_rows,
                                                     int output_cols,
                                                     bool resized_full_chunk,
                                                     const int* output_plane_indices_device,
                                                     uint8_t* output_mask_full_batch_device,
                                                     cudaStream_t cuda_stream);

// Coherence-primary fusion: final mask = band_mask ∪ structure_mask ∪
// threshold(legacy_final_mask, 0.5) [legacy input nullable]; combined score =
// max(qnorm(coherence, 0.05-0.99), qnorm(dino, 0.05-0.95)) for grouping/viz.
// The legacy input may alias output_final_mask_device — it is consumed into
// scratch before any output is written. debug_initial_product_device (nullable)
// receives qnorm(dino)*qnorm(coherence) for diagnostics.
bool compute_coherence_primary_fusion_gpu_batch_to_device(const float* dino_score_batch_device,
                                                          const float* coherence_batch_device,
                                                          const uint8_t* coherence_band_mask_batch_device,
                                                          const uint8_t* dino_structure_mask_batch_device,
                                                          const float* legacy_final_mask_batch_device,
                                                          int batch_size,
                                                          int rows,
                                                          int cols,
                                                          const std::vector<uint8_t>& valid_row_mask_batch,
                                                          float* output_combined_score_device,
                                                          float* output_final_mask_device,
                                                          float* debug_initial_product_device,
                                                          cudaStream_t cuda_stream,
                                                          float dino_contribution_strength = 1.0f);

}  // namespace holoscan::ops
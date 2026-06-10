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
                                                       CudaHybridStageTiming* stage_timing);

namespace {

torch::Tensor select_quantile_flat_batch_torch(const torch::Tensor& input, double q) {
  if (input.dim() != 3 || input.size(0) <= 0) {
    return torch::zeros({0, 1, 1}, input.options());
  }
  auto flat = input.reshape({input.size(0), -1});
  const auto size = flat.size(1);
  if (size <= 1) {
    return flat.select(1, 0).view({input.size(0), 1, 1});
  }
  const double clamped = std::clamp(q, 0.0, 1.0);
  const auto rank = static_cast<int64_t>(std::llround(clamped * static_cast<double>(size - 1)));
  return std::get<0>(torch::kthvalue(flat, rank + 1, 1, false)).view({input.size(0), 1, 1});
}

torch::Tensor normalize_map01_quantile_torch_batch(const torch::Tensor& input, double low_q, double high_q) {
  if (input.dim() != 3) {
    return torch::zeros_like(input);
  }
  auto lo = select_quantile_flat_batch_torch(input, low_q);
  auto hi = select_quantile_flat_batch_torch(input, high_q);
  auto scale = torch::clamp_min(hi - lo, 1e-6);
  return torch::clamp((input - lo) / scale, 0.0, 1.0);
}

torch::Tensor upload_valid_row_mask_torch_batch(const std::vector<uint8_t>& valid_row_mask_batch,
                                                int batch_size,
                                                int rows,
                                                const c10::Device& device,
                                                cudaStream_t cuda_stream) {
  const int64_t total = static_cast<int64_t>(batch_size) * static_cast<int64_t>(rows);
  if (batch_size <= 0 || rows <= 0 || total <= 0 || valid_row_mask_batch.size() != static_cast<size_t>(total)) {
    return torch::Tensor();
  }

  struct ValidRowMaskCache {
    torch::Tensor values_u8;
  };
  thread_local ValidRowMaskCache cache;

  if (!cache.values_u8.defined() || cache.values_u8.numel() != total || cache.values_u8.device() != device) {
    cache.values_u8 = torch::empty({total}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
  }

  if (cudaMemcpyAsync(cache.values_u8.data_ptr<uint8_t>(),
                      valid_row_mask_batch.data(),
                      static_cast<size_t>(total) * sizeof(uint8_t),
                      cudaMemcpyHostToDevice,
                      cuda_stream) != cudaSuccess) {
    return torch::Tensor();
  }

  return cache.values_u8.view({static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), 1});
}

torch::Tensor masked_quantile_histogram_torch_batch(const torch::Tensor& input,
                                                    const torch::Tensor& mask,
                                                    double q,
                                                    int64_t bin_count = 256,
                                                    float fallback = 1.0f) {
  if (input.dim() != 3 || mask.sizes() != input.sizes()) {
    return torch::full({std::max<int64_t>(input.size(0), 0), 1, 1}, fallback, input.options());
  }

  const auto batch_size = input.size(0);
  if (batch_size <= 0) {
    return torch::zeros({0, 1, 1}, input.options());
  }

  bin_count = std::max<int64_t>(bin_count, 2);
  auto values = torch::clamp(input.to(torch::kFloat32), 0.0, 1.0).reshape({batch_size, -1}).contiguous();
  auto flat_mask = mask.to(torch::kBool).reshape({batch_size, -1}).contiguous();
  auto counts = flat_mask.sum(1, false, torch::kLong);

  auto thresholds = torch::full({batch_size}, fallback, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  if (!counts.gt(0).any().item<bool>()) {
    return thresholds.view({batch_size, 1, 1}).to(input.scalar_type());
  }

  auto bins = torch::clamp((values * static_cast<double>(bin_count - 1)).round().to(torch::kLong), 0, bin_count - 1);
  auto masked_bins = torch::where(flat_mask, bins, torch::zeros_like(bins));
  auto histogram = torch::zeros({batch_size, bin_count}, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  histogram.scatter_add_(1, masked_bins, flat_mask.to(torch::kFloat32));

  const double clamped_q = std::clamp(q, 0.0, 1.0);
  auto target = torch::clamp((counts.to(torch::kFloat32) - 1.0f) * clamped_q, 0.0, std::numeric_limits<float>::max()).round().to(torch::kLong) + 1;
  auto cumulative = histogram.cumsum(1);
  auto reached = cumulative >= target.unsqueeze(1).to(torch::kFloat32);
  auto quantile_bins = std::get<1>(reached.to(torch::kInt64).max(1, false));
  auto quantile_values = quantile_bins.to(torch::kFloat32) / static_cast<float>(bin_count - 1);
  thresholds = torch::where(counts.gt(0), quantile_values, thresholds);
  return thresholds.view({batch_size, 1, 1}).to(input.scalar_type());
}

torch::Tensor positional_design_matrix_torch(int patch_rows,
                                             int patch_cols,
                                             const c10::Device& device,
                                             c10::ScalarType dtype) {
  constexpr float kPi = 3.14159265358979323846f;
  const int patch_count = patch_rows * patch_cols;
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  if (patch_rows <= 0 || patch_cols <= 0) {
    return torch::zeros({0, 16}, options);
  }

  std::vector<float> design(static_cast<size_t>(patch_count) * 16U, 0.0f);
  for (int row = 0; row < patch_rows; ++row) {
    const float row_coord = patch_rows > 1 ? -1.0f + 2.0f * static_cast<float>(row) / static_cast<float>(patch_rows - 1) : 0.0f;
    for (int col = 0; col < patch_cols; ++col) {
      const float col_coord = patch_cols > 1 ? -1.0f + 2.0f * static_cast<float>(col) / static_cast<float>(patch_cols - 1) : 0.0f;
      const size_t base = static_cast<size_t>(row * patch_cols + col) * 16U;
      design[base + 0] = 1.0f;
      design[base + 1] = row_coord;
      design[base + 2] = col_coord;
      design[base + 3] = row_coord * row_coord;
      design[base + 4] = col_coord * col_coord;
      design[base + 5] = row_coord * col_coord;
      design[base + 6] = std::sin(kPi * row_coord);
      design[base + 7] = std::sin(kPi * col_coord);
      design[base + 8] = std::cos(kPi * row_coord);
      design[base + 9] = std::cos(kPi * col_coord);
      design[base + 10] = std::sin(2.0f * kPi * row_coord);
      design[base + 11] = std::sin(2.0f * kPi * col_coord);
      design[base + 12] = std::cos(2.0f * kPi * row_coord);
      design[base + 13] = std::cos(2.0f * kPi * col_coord);
      design[base + 14] = std::sin(kPi * row_coord) * std::cos(kPi * col_coord);
      design[base + 15] = std::cos(kPi * row_coord) * std::sin(kPi * col_coord);
    }
  }
  auto cpu = torch::from_blob(design.data(), {patch_count, 16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).clone();
  return cpu.to(device, dtype).contiguous();
}

struct PositionalSuppressionCache {
  int patch_rows = 0;
  int patch_cols = 0;
  c10::Device device = c10::Device(torch::kCPU);
  c10::ScalarType dtype = torch::kFloat32;
  torch::Tensor design;
  torch::Tensor projection_left;
};

const PositionalSuppressionCache& positional_suppression_cache_torch(int patch_rows,
                                                                     int patch_cols,
                                                                     const c10::Device& device,
                                                                     c10::ScalarType dtype) {
  thread_local PositionalSuppressionCache cache;

  const bool cache_valid = cache.design.defined() && cache.projection_left.defined() &&
                           cache.patch_rows == patch_rows && cache.patch_cols == patch_cols &&
                           cache.device == device && cache.dtype == dtype;
  if (cache_valid) {
    return cache;
  }

  cache.patch_rows = patch_rows;
  cache.patch_cols = patch_cols;
  cache.device = device;
  cache.dtype = dtype;
  cache.design = positional_design_matrix_torch(patch_rows, patch_cols, device, dtype);

  if (!cache.design.defined() || cache.design.numel() == 0) {
    cache.projection_left = torch::Tensor();
    return cache;
  }

  auto design_t = cache.design.transpose(0, 1).contiguous();
  auto ridge = 1.0e-3f * torch::eye(cache.design.size(1),
                                    torch::TensorOptions().dtype(dtype).device(device));
  cache.projection_left = torch::linalg_solve(torch::matmul(design_t, cache.design) + ridge, design_t).contiguous();
  return cache;
}

torch::Tensor suppress_raw_dino_positional_features_torch_batch(const torch::Tensor& patch_features,
                                                                int patch_rows,
                                                                int patch_cols,
                                                                float suppression) {
  const float clamped = std::clamp(suppression, 0.0f, 1.0f);
  if (clamped <= 0.0f || patch_features.dim() != 3 || patch_features.size(1) <= 0 || patch_features.size(2) <= 0) {
    return patch_features;
  }

  const auto& cache = positional_suppression_cache_torch(
      patch_rows, patch_cols, patch_features.device(), patch_features.scalar_type());
  if (!cache.design.defined() || !cache.projection_left.defined() || cache.design.size(0) != patch_features.size(1)) {
    return patch_features;
  }

  const auto batch_size = patch_features.size(0);
  auto design_batch = cache.design.unsqueeze(0).expand({batch_size, cache.design.size(0), cache.design.size(1)});
  auto projection_left_batch =
      cache.projection_left.unsqueeze(0).expand({batch_size, cache.projection_left.size(0), cache.projection_left.size(1)});
  auto beta = torch::matmul(projection_left_batch, patch_features);
  auto suppressed = patch_features - torch::matmul(design_batch, beta);
  if (clamped >= 1.0f) {
    return suppressed.contiguous();
  }
  return ((1.0f - clamped) * patch_features + clamped * suppressed).contiguous();
}

torch::Tensor project_aligned_maps_torch_batch(const torch::Tensor& aligned_maps,
                                               int output_rows,
                                               int output_cols,
                                               bool resized_full_chunk) {
  if (aligned_maps.dim() != 3 || output_rows <= 0 || output_cols <= 0) {
    return torch::zeros({0, std::max(output_rows, 0), std::max(output_cols, 0)}, aligned_maps.options());
  }
  auto aligned4d = aligned_maps.unsqueeze(1);
  if (resized_full_chunk) {
    return torch::nn::functional::interpolate(
               aligned4d,
               torch::nn::functional::InterpolateFuncOptions()
                   .size(std::vector<int64_t>{static_cast<int64_t>(output_rows), static_cast<int64_t>(output_cols)})
                   .mode(torch::kBilinear)
                   .align_corners(false))
        .squeeze(1)
        .contiguous();
  }

  auto canvas = torch::zeros({aligned_maps.size(0), output_rows, output_cols}, aligned_maps.options());
  const int copy_rows = std::min(output_rows, static_cast<int>(aligned_maps.size(1)));
  const int copy_cols = std::min(output_cols, static_cast<int>(aligned_maps.size(2)));
  if (copy_rows > 0 && copy_cols > 0) {
    canvas.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, copy_rows), torch::indexing::Slice(0, copy_cols)},
                      aligned_maps.index({torch::indexing::Slice(), torch::indexing::Slice(0, copy_rows), torch::indexing::Slice(0, copy_cols)}));
  }
  return canvas.contiguous();
}

torch::Tensor normalize_map01_masked_minmax_torch_batch(const torch::Tensor& input, const torch::Tensor& valid_mask) {
  if (input.dim() != 3 || valid_mask.dim() != 3) {
    return torch::zeros_like(input);
  }
  auto input_float = input.to(torch::kFloat32);
  auto flat_input = input_float.reshape({input.size(0), -1});
  auto flat_valid = valid_mask.reshape({valid_mask.size(0), -1}).to(torch::kBool);

  auto pos_inf = torch::full_like(flat_input, std::numeric_limits<float>::infinity());
  auto neg_inf = torch::full_like(flat_input, -std::numeric_limits<float>::infinity());
  auto masked_min_input = torch::where(flat_valid, flat_input, pos_inf);
  auto masked_max_input = torch::where(flat_valid, flat_input, neg_inf);

  auto lo = std::get<0>(masked_min_input.min(1, true)).view({input.size(0), 1, 1});
  auto hi = std::get<0>(masked_max_input.max(1, true)).view({input.size(0), 1, 1});
  auto has_valid = flat_valid.any(1, true).view({input.size(0), 1, 1});
  auto safe_lo = torch::where(has_valid, lo, torch::zeros_like(lo));
  auto safe_hi = torch::where(has_valid, hi, torch::ones_like(hi));
  auto scale = torch::clamp_min(safe_hi - safe_lo, 1.0e-6);
  auto normalized = torch::clamp((input_float - safe_lo) / scale, 0.0, 1.0);
  auto zeros = torch::zeros_like(normalized);
  return torch::where(valid_mask.to(torch::kBool), torch::where(has_valid, normalized, zeros), zeros).contiguous();
}

torch::Tensor gaussian_kernel_tensor_torch(double sigma, const c10::Device& device, c10::ScalarType dtype) {
  if (sigma <= 0.0) {
    return torch::ones({1}, torch::TensorOptions().dtype(dtype).device(device));
  }
  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(dtype).device(device));
  auto kernel = torch::exp(-(x * x) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();
  return kernel.contiguous();
}

torch::Tensor gaussian_first_derivative_kernel_tensor_torch(double sigma, const c10::Device& device, c10::ScalarType dtype) {
  if (sigma <= 0.0) {
    return torch::zeros({1}, torch::TensorOptions().dtype(dtype).device(device));
  }
  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(dtype).device(device));
  const double sigma2 = sigma * sigma;
  auto kernel = (-x / sigma2) * torch::exp(-(x * x) / (2.0 * sigma2));
  return kernel.contiguous();
}

torch::Tensor convolve_rows_2d_torch_batch(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(1), {0, 0, radius, radius});
  return torch::conv2d(padded, kernel.view({1, 1, kernel.size(0), 1})).squeeze(1).contiguous();
}

torch::Tensor convolve_cols_2d_torch_batch(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(1), {radius, radius, 0, 0});
  return torch::conv2d(padded, kernel.view({1, 1, 1, kernel.size(0)})).squeeze(1).contiguous();
}

torch::Tensor gaussian_blur_2d_torch_batch(const torch::Tensor& input, double sigma_rows, double sigma_cols) {
  auto row_kernel = gaussian_kernel_tensor_torch(sigma_rows, input.device(), input.scalar_type());
  auto col_kernel = gaussian_kernel_tensor_torch(sigma_cols, input.device(), input.scalar_type());
  return convolve_cols_2d_torch_batch(convolve_rows_2d_torch_batch(input, row_kernel), col_kernel).contiguous();
}

torch::Tensor gaussian_first_derivative_rows_2d_torch_batch(const torch::Tensor& input, double sigma) {
  auto smooth_kernel = gaussian_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  auto deriv_kernel = gaussian_first_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_rows_2d_torch_batch(convolve_cols_2d_torch_batch(input, smooth_kernel), deriv_kernel).contiguous();
}

torch::Tensor gaussian_first_derivative_cols_2d_torch_batch(const torch::Tensor& input, double sigma) {
  auto smooth_kernel = gaussian_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  auto deriv_kernel = gaussian_first_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_cols_2d_torch_batch(convolve_rows_2d_torch_batch(input, smooth_kernel), deriv_kernel).contiguous();
}

torch::Tensor uniform_filter_2d_nearest_torch_batch(const torch::Tensor& input, int kernel_rows, int kernel_cols) {
  const int row_radius = std::max(0, kernel_rows / 2);
  const int col_radius = std::max(0, kernel_cols / 2);
  auto padded = torch::replication_pad2d(input.unsqueeze(1), {col_radius, col_radius, row_radius, row_radius});
  return torch::avg_pool2d(padded,
                           {std::max(1, kernel_rows), std::max(1, kernel_cols)},
                           {1, 1},
                           {0, 0},
                           false,
                           true)
      .squeeze(1)
      .contiguous();
}

torch::Tensor gaussian_second_derivative_kernel_tensor_torch(double sigma, const c10::Device& device, c10::ScalarType dtype) {
  if (sigma <= 0.0) {
    return torch::zeros({1}, torch::TensorOptions().dtype(dtype).device(device));
  }
  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(dtype).device(device));
  const double sigma2 = sigma * sigma;
  auto kernel = ((x * x - sigma2) / (sigma2 * sigma2)) * torch::exp(-(x * x) / (2.0 * sigma2));
  return kernel.contiguous();
}

torch::Tensor gaussian_second_derivative_rows_2d_torch_batch(const torch::Tensor& input, double sigma) {
  auto kernel = gaussian_second_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_rows_2d_torch_batch(input, kernel).contiguous();
}

torch::Tensor binary_dilate_rect_torch_batch(const torch::Tensor& mask, int kernel_rows, int kernel_cols) {
  auto mask_float = mask.to(torch::kFloat32);
  const int row_radius = std::max(0, kernel_rows / 2);
  const int col_radius = std::max(0, kernel_cols / 2);
  auto padded = torch::replication_pad2d(mask_float.unsqueeze(1), {col_radius, col_radius, row_radius, row_radius});
  return torch::max_pool2d(padded, {kernel_rows, kernel_cols}, {1, 1}, {0, 0}).squeeze(1) > 0.5;
}

torch::Tensor binary_erode_rect_torch_batch(const torch::Tensor& mask, int kernel_rows, int kernel_cols) {
  auto inverted = torch::logical_not(mask);
  return torch::logical_not(binary_dilate_rect_torch_batch(inverted, kernel_rows, kernel_cols));
}

torch::Tensor binary_closing_rect_torch_batch(const torch::Tensor& mask, int kernel_rows, int kernel_cols) {
  return binary_erode_rect_torch_batch(binary_dilate_rect_torch_batch(mask, kernel_rows, kernel_cols), kernel_rows, kernel_cols);
}

torch::Tensor binary_fill_holes_torch_batch(const torch::Tensor& mask) {
  auto mask_bool = mask.to(torch::kBool).contiguous();
  if (mask_bool.dim() != 3 || mask_bool.size(0) <= 0 || mask_bool.size(1) <= 0 || mask_bool.size(2) <= 0) {
    return mask_bool;
  }

  auto background = torch::logical_not(mask_bool);
  auto exterior = torch::zeros_like(background);
  exterior.index_put_({torch::indexing::Slice(), 0, torch::indexing::Slice()},
                      background.index({torch::indexing::Slice(), 0, torch::indexing::Slice()}));
  exterior.index_put_({torch::indexing::Slice(), background.size(1) - 1, torch::indexing::Slice()},
                      torch::logical_or(exterior.index({torch::indexing::Slice(), background.size(1) - 1, torch::indexing::Slice()}),
                                        background.index({torch::indexing::Slice(), background.size(1) - 1, torch::indexing::Slice()})));
  exterior.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0},
                      torch::logical_or(exterior.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}),
                                        background.index({torch::indexing::Slice(), torch::indexing::Slice(), 0})));
  exterior.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), background.size(2) - 1},
                      torch::logical_or(exterior.index({torch::indexing::Slice(), torch::indexing::Slice(), background.size(2) - 1}),
                                        background.index({torch::indexing::Slice(), torch::indexing::Slice(), background.size(2) - 1})));

  auto grown = torch::logical_and(exterior, background).contiguous();
  const int max_iterations = std::max<int>(1, static_cast<int>(mask_bool.size(1) + mask_bool.size(2)));
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    auto expanded = torch::logical_and(binary_dilate_rect_torch_batch(grown, 3, 3), background).contiguous();
    if (torch::equal(expanded, grown)) {
      break;
    }
    grown = std::move(expanded);
  }

  auto holes = torch::logical_and(background, torch::logical_not(grown));
  return torch::logical_or(mask_bool, holes).contiguous();
}

torch::Tensor keep_large_components_torch_batch(const torch::Tensor& mask, int min_size) {
  auto mask_bool = mask.to(torch::kBool).contiguous();
  if (mask_bool.dim() != 3 || min_size <= 1) {
    return mask_bool;
  }

  const auto batch_size = mask_bool.size(0);
  const auto rows = mask_bool.size(1);
  const auto cols = mask_bool.size(2);
  if (batch_size <= 0 || rows <= 0 || cols <= 0) {
    return mask_bool;
  }

  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(mask_bool.device());
  auto ids = torch::arange(1,
                           rows * cols + 1,
                           torch::TensorOptions().dtype(torch::kFloat32).device(mask_bool.device()))
                 .view({1, rows, cols})
                 .expand({batch_size, rows, cols});
  auto labels = torch::where(mask_bool, ids, torch::zeros_like(ids)).contiguous();
  const int max_iterations = std::max<int>(1, static_cast<int>(rows + cols));
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    auto pooled = torch::max_pool2d(labels.unsqueeze(1), {3, 3}, {1, 1}, {1, 1}).squeeze(1);
    auto next = torch::where(mask_bool, pooled, torch::zeros_like(pooled)).contiguous();
    if (torch::equal(next, labels)) {
      labels = std::move(next);
      break;
    }
    labels = std::move(next);
  }

  auto flat_labels = labels.to(torch::kLong).reshape({batch_size, rows * cols}).contiguous();
  std::vector<torch::Tensor> filtered_masks;
  filtered_masks.reserve(static_cast<size_t>(batch_size));
  for (int64_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    auto sample_labels = flat_labels[batch_index];
    auto active_labels = sample_labels.masked_select(sample_labels > 0);
    if (active_labels.numel() == 0) {
      filtered_masks.push_back(torch::zeros({rows, cols}, float_options.dtype(torch::kBool)));
      continue;
    }

    auto counts = torch::bincount(active_labels, {}, rows * cols + 1);
    auto keep_lookup = counts >= static_cast<int64_t>(std::max(1, min_size));
    auto kept = keep_lookup.index({sample_labels}).view({rows, cols});
    filtered_masks.push_back(torch::logical_and(mask_bool[batch_index], kept));
  }
  return torch::stack(filtered_masks, 0).to(torch::kBool).contiguous();
}

}  // namespace

bool compute_structure_tensor_gate_gpu_batch_to_device(const float* corrected_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       const std::vector<uint8_t>& valid_row_mask_batch,
                                                       float* output_gate_device,
                                                       cudaStream_t cuda_stream) {
  return compute_fast_directional_coherence_gate_gpu_batch_to_device(corrected_batch_device,
                                                                     batch_size,
                                                                     rows,
                                                                     cols,
                                                                     valid_row_mask_batch,
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
                                                           cudaStream_t cuda_stream) {
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
                                                                        cuda_stream);
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
                                                      CudaHybridStageTiming* stage_timing) {
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
                                                           stage_timing);
}

}  // namespace holoscan::ops

#else

namespace holoscan::ops {

bool compute_structure_tensor_gate_gpu_batch_to_device(const float* corrected_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       const std::vector<uint8_t>& valid_row_mask_batch,
                                                       float* output_gate_device,
                                                       cudaStream_t cuda_stream) {
  return compute_fast_directional_coherence_gate_gpu_batch_to_device(corrected_batch_device,
                                                                     batch_size,
                                                                     rows,
                                                                     cols,
                                                                     valid_row_mask_batch,
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
                                                           cudaStream_t cuda_stream) {
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
                                                                        cuda_stream);
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
                                                      CudaHybridStageTiming* stage_timing) {
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
                                                           stage_timing);
}

}  // namespace holoscan::ops

#endif
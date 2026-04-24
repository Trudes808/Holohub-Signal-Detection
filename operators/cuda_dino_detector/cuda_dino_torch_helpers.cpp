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
#include <cmath>
#include <limits>

namespace holoscan::ops {

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

torch::Tensor suppress_raw_dino_positional_features_torch_batch(const torch::Tensor& patch_features,
                                                                int patch_rows,
                                                                int patch_cols,
                                                                float suppression) {
  const float clamped = std::clamp(suppression, 0.0f, 1.0f);
  if (clamped <= 0.0f || patch_features.dim() != 3 || patch_features.size(1) <= 0 || patch_features.size(2) <= 0) {
    return patch_features;
  }

  const auto design = positional_design_matrix_torch(patch_rows, patch_cols, patch_features.device(), patch_features.scalar_type());
  if (design.size(0) != patch_features.size(1)) {
    return patch_features;
  }
  const auto batch_size = patch_features.size(0);
  auto design_batch = design.unsqueeze(0).expand({batch_size, design.size(0), design.size(1)});
  auto design_t = design_batch.transpose(1, 2);
  auto xtx = torch::matmul(design_t, design_batch);
  auto ridge = 1.0e-3f * torch::eye(design.size(1), torch::TensorOptions().dtype(patch_features.scalar_type()).device(patch_features.device()))
                             .unsqueeze(0)
                             .expand({batch_size, design.size(1), design.size(1)});
  auto xty = torch::matmul(design_t, patch_features);
  auto beta = torch::linalg_solve(xtx + ridge, xty);
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
  if (batch_size <= 0 || rows <= 0 || cols <= 0 || corrected_batch_device == nullptr || output_gate_device == nullptr ||
      valid_row_mask_batch.size() != static_cast<size_t>(batch_size) * static_cast<size_t>(rows)) {
    return false;
  }

  try {
    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto torch_stream = cuda_stream
                                  ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                  : c10::cuda::getDefaultCUDAStream(compute_device.index());
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
    auto corrected = torch::from_blob(const_cast<float*>(corrected_batch_device),
                                      {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                      float_options)
                         .contiguous();
    auto valid_mask_gpu = torch::from_blob(const_cast<uint8_t*>(valid_row_mask_batch.data()),
                                           {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), 1},
                                           torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                              .to(compute_device, torch::kBool)
                              .expand({static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

    const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
    const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
    auto background = uniform_filter_2d_nearest_torch_batch(corrected,
                                                            std::max(1, bg_freq),
                                                            std::max(1, bg_time));
    auto residual_db = torch::clamp_min(corrected - background, 0.0);
    auto residual_n = normalize_map01_quantile_torch_batch(residual_db, 0.05, 0.99);

    const std::array<double, 3> scales = {0.8, 1.6, 3.2};
    auto gate_max = torch::zeros_like(corrected, float_options);
    for (double grad_sigma : scales) {
      const double integ_sigma = std::max(1.0, 1.8 * grad_sigma);
      auto grad_f = gaussian_first_derivative_rows_2d_torch_batch(residual_n, grad_sigma);
      auto grad_t = gaussian_first_derivative_cols_2d_torch_batch(residual_n, grad_sigma);
      auto j_ff = gaussian_blur_2d_torch_batch(grad_f * grad_f, integ_sigma, integ_sigma);
      auto j_ft = gaussian_blur_2d_torch_batch(grad_f * grad_t, integ_sigma, integ_sigma);
      auto j_tt = gaussian_blur_2d_torch_batch(grad_t * grad_t, integ_sigma, integ_sigma);

      auto delta = torch::sqrt(torch::clamp_min((j_ff - j_tt) * (j_ff - j_tt) + 4.0f * (j_ft * j_ft), 0.0));
      auto lambda1 = 0.5f * (j_ff + j_tt + delta);
      auto lambda2 = 0.5f * (j_ff + j_tt - delta);
      auto coherence = (lambda1 - lambda2) / torch::clamp_min(lambda1 + lambda2, 1.0e-6);
      auto energy = lambda1 + lambda2;

      auto coherence_n = normalize_map01_quantile_torch_batch(coherence, 0.05, 0.99);
      auto energy_n = normalize_map01_quantile_torch_batch(energy, 0.05, 0.99);
      auto gate_value = coherence_n * torch::sqrt(torch::clamp_min(energy_n, 0.0));
      gate_max = torch::maximum(gate_max, gate_value);
    }

    auto gate = normalize_map01_quantile_torch_batch(gate_max, 0.05, 0.99).to(torch::kFloat32);
    gate = torch::where(valid_mask_gpu, gate, torch::zeros_like(gate)).contiguous();
    return cudaMemcpyAsync(output_gate_device,
                           gate.data_ptr<float>(),
                           static_cast<size_t>(gate.numel()) * sizeof(float),
                           cudaMemcpyDeviceToDevice,
                           cuda_stream) == cudaSuccess;
  } catch (...) {
    return false;
  }
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
  const int patch_count = patch_rows * patch_cols;
  if (batch_size <= 0 || patch_rows <= 0 || patch_cols <= 0 || feature_dim <= 0 || aligned_rows <= 0 || aligned_cols <= 0 ||
      output_rows <= 0 || output_cols <= 0 || patch_features_batch_device == nullptr || output_score_device == nullptr || patch_count <= 0) {
    return false;
  }

  try {
    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto torch_stream = cuda_stream
                                  ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                  : c10::cuda::getDefaultCUDAStream(compute_device.index());
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

    auto patch_features = torch::from_blob(const_cast<float*>(patch_features_batch_device),
                                           {static_cast<int64_t>(batch_size), static_cast<int64_t>(patch_count), static_cast<int64_t>(feature_dim)},
                                           torch::TensorOptions().dtype(torch::kFloat32).device(compute_device))
                              .contiguous();
    auto energy_features = suppress_raw_dino_positional_features_torch_batch(patch_features,
                                                                             patch_rows,
                                                                             patch_cols,
                                                                             positional_suppression);
    auto raw_patch = torch::sqrt(torch::clamp_min(torch::mean(energy_features * energy_features, 2), 1.0e-6f));
    auto raw_patch_n = normalize_map01_quantile_torch_batch(raw_patch.view({batch_size, patch_rows, patch_cols}), 0.05, 0.95);
    auto aligned_maps = torch::nn::functional::interpolate(
                           raw_patch_n.unsqueeze(1),
                           torch::nn::functional::InterpolateFuncOptions()
                               .size(std::vector<int64_t>{static_cast<int64_t>(aligned_rows), static_cast<int64_t>(aligned_cols)})
                               .mode(torch::kBilinear)
                               .align_corners(false))
                           .squeeze(1)
                           .contiguous();
    auto projected = project_aligned_maps_torch_batch(aligned_maps, output_rows, output_cols, resized_full_chunk);
    return cudaMemcpyAsync(output_score_device,
                           projected.data_ptr<float>(),
                           static_cast<size_t>(projected.numel()) * sizeof(float),
                           cudaMemcpyDeviceToDevice,
                           cuda_stream) == cudaSuccess;
  } catch (...) {
    return false;
  }
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
  if (batch_size <= 0 || runtime_rows <= 0 || runtime_cols <= 0 || aligned_rows <= 0 || aligned_cols <= 0 ||
      output_rows <= 0 || output_cols <= 0 || score_maps_batch_device == nullptr || output_score_device == nullptr) {
    return false;
  }

  try {
    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto torch_stream = cuda_stream
                                  ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                  : c10::cuda::getDefaultCUDAStream(compute_device.index());
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

    auto score_maps = torch::from_blob(const_cast<float*>(score_maps_batch_device),
                                       {static_cast<int64_t>(batch_size), static_cast<int64_t>(runtime_rows), static_cast<int64_t>(runtime_cols)},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(compute_device))
                          .contiguous();
    auto aligned_maps = torch::nn::functional::interpolate(
                           score_maps.unsqueeze(1),
                           torch::nn::functional::InterpolateFuncOptions()
                               .size(std::vector<int64_t>{static_cast<int64_t>(aligned_rows), static_cast<int64_t>(aligned_cols)})
                               .mode(torch::kBilinear)
                               .align_corners(false))
                           .squeeze(1)
                           .contiguous();
    auto projected = project_aligned_maps_torch_batch(aligned_maps, output_rows, output_cols, resized_full_chunk);
    return cudaMemcpyAsync(output_score_device,
                           projected.data_ptr<float>(),
                           static_cast<size_t>(projected.numel()) * sizeof(float),
                           cudaMemcpyDeviceToDevice,
                           cuda_stream) == cudaSuccess;
  } catch (...) {
    return false;
  }
}

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
                                                      cudaStream_t cuda_stream) {
  if (batch_size <= 0 || rows <= 0 || cols <= 0 || dino_score_batch_device == nullptr || coherence_batch_device == nullptr ||
      output_combined_score_device == nullptr || output_final_mask_device == nullptr ||
      valid_row_mask_batch.size() != static_cast<size_t>(batch_size) * static_cast<size_t>(rows)) {
    return false;
  }

  try {
    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto torch_stream = cuda_stream
                                  ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                  : c10::cuda::getDefaultCUDAStream(compute_device.index());
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

    const auto contrib_dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
    auto dino_score = torch::from_blob(const_cast<float*>(dino_score_batch_device),
                                       {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                       float_options)
                .to(compute_device, contrib_dtype)
                .contiguous();
    auto coherence = torch::from_blob(const_cast<float*>(coherence_batch_device),
                                      {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                      float_options)
               .to(compute_device, contrib_dtype)
               .contiguous();

    auto valid_rows = torch::from_blob(const_cast<uint8_t*>(valid_row_mask_batch.data()),
                                       {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), 1},
                                       torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                          .clone()
                          .to(compute_device, torch::kBool);
    auto valid_mask = valid_rows.expand({static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)}).contiguous();

    auto dino_norm = normalize_map01_quantile_torch_batch(dino_score, 0.05, 0.95);
    auto coherence_norm = normalize_map01_quantile_torch_batch(coherence, 0.05, 0.99);
    auto contrib = (dino_norm * coherence_norm).contiguous();

    auto base_norm = normalize_map01_masked_minmax_torch_batch(contrib, valid_mask);
    auto envelope_map = normalize_map01_masked_minmax_torch_batch(gaussian_blur_2d_torch_batch(base_norm, 6.0, 1.4), valid_mask);
    auto base_blur = gaussian_blur_2d_torch_batch(base_norm, 4.0, 1.0);
    auto residual_penalty = normalize_map01_masked_minmax_torch_batch(gaussian_blur_2d_torch_batch(torch::abs(base_norm - base_blur), 2.0, 0.8), valid_mask);
    auto freq_curvature_penalty = normalize_map01_masked_minmax_torch_batch(torch::abs(gaussian_second_derivative_rows_2d_torch_batch(base_norm, 0.8)), valid_mask);

    auto keep_freq = normalize_map01_masked_minmax_torch_batch(envelope_map - 0.90 * freq_curvature_penalty, valid_mask);
    auto keep_res = normalize_map01_masked_minmax_torch_batch(envelope_map - 1.00 * residual_penalty, valid_mask);
    auto residual_veto_gate = torch::clamp((keep_res - 0.30) / 0.70, 0.0, 1.0);
    auto combined_input = keep_freq * (0.35 + 0.65 * residual_veto_gate);
    auto combined_score = normalize_map01_masked_minmax_torch_batch(combined_input, valid_mask);

    std::vector<torch::Tensor> final_masks;
    final_masks.reserve(static_cast<size_t>(batch_size));
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      auto sample_valid = valid_mask[sample_index];
      auto active_freq = keep_freq[sample_index].masked_select(sample_valid);
      auto active_res = keep_res[sample_index].masked_select(sample_valid);
      auto active_combined = combined_score[sample_index].masked_select(sample_valid);
      const float seed_freq_threshold = active_freq.numel() > 0 ? static_cast<float>(select_quantile_flat_batch_torch(active_freq.view({1, -1, 1}), 0.90).item<float>()) : 1.0f;
      const float seed_res_threshold = active_res.numel() > 0 ? static_cast<float>(select_quantile_flat_batch_torch(active_res.view({1, -1, 1}), 0.82).item<float>()) : 1.0f;
      const float combined_threshold = active_combined.numel() > 0 ? static_cast<float>(select_quantile_flat_batch_torch(active_combined.view({1, -1, 1}), 0.78).item<float>()) : 1.0f;
      auto seed_mask = torch::logical_and(sample_valid,
                                          torch::logical_and(keep_freq[sample_index] >= seed_freq_threshold,
                                                             keep_res[sample_index] >= seed_res_threshold));
      auto final_mask = torch::logical_and(seed_mask,
                   torch::logical_and(sample_valid,
                          combined_score[sample_index] >= static_cast<double>(combined_threshold) * 0.85));
      auto closed_mask = binary_closing_rect_torch_batch(final_mask.unsqueeze(0), 7, 3);
      auto filled_mask = binary_fill_holes_torch_batch(closed_mask);
      auto filtered_mask = keep_large_components_torch_batch(filled_mask, min_component_size);
      final_masks.push_back(filtered_mask.squeeze(0).to(torch::kFloat32));
    }

    auto final_mask_batch = torch::stack(final_masks, 0).contiguous();
    final_mask_batch = torch::where(valid_mask, final_mask_batch, torch::zeros_like(final_mask_batch));

    const bool combined_ok = cudaMemcpyAsync(output_combined_score_device,
                                             combined_score.data_ptr<float>(),
                                             static_cast<size_t>(combined_score.numel()) * sizeof(float),
                                             cudaMemcpyDeviceToDevice,
                                             cuda_stream) == cudaSuccess;
    const bool mask_ok = cudaMemcpyAsync(output_final_mask_device,
                                         final_mask_batch.data_ptr<float>(),
                                         static_cast<size_t>(final_mask_batch.numel()) * sizeof(float),
                                         cudaMemcpyDeviceToDevice,
                                         cuda_stream) == cudaSuccess;
    return combined_ok && mask_ok;
  } catch (...) {
    return false;
  }
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
  (void)corrected_batch_device;
  (void)batch_size;
  (void)rows;
  (void)cols;
  (void)valid_row_mask_batch;
  (void)output_gate_device;
  (void)cuda_stream;
  return false;
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
  (void)patch_features_batch_device;
  (void)batch_size;
  (void)patch_rows;
  (void)patch_cols;
  (void)feature_dim;
  (void)aligned_rows;
  (void)aligned_cols;
  (void)output_rows;
  (void)output_cols;
  (void)positional_suppression;
  (void)resized_full_chunk;
  (void)output_score_device;
  (void)cuda_stream;
  return false;
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
  (void)score_maps_batch_device;
  (void)batch_size;
  (void)runtime_rows;
  (void)runtime_cols;
  (void)aligned_rows;
  (void)aligned_cols;
  (void)output_rows;
  (void)output_cols;
  (void)resized_full_chunk;
  (void)output_score_device;
  (void)cuda_stream;
  return false;
}

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
                                                      cudaStream_t cuda_stream) {
  (void)dino_score_batch_device;
  (void)coherence_batch_device;
  (void)batch_size;
  (void)rows;
  (void)cols;
  (void)valid_row_mask_batch;
  (void)use_fp16;
  (void)min_component_size;
  (void)output_combined_score_device;
  (void)output_final_mask_device;
  (void)cuda_stream;
  return false;
}

}  // namespace holoscan::ops

#endif
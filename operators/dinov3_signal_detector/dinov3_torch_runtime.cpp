// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "dinov3_torch_runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/nn/functional.h>
#include <torch/script.h>
#include <torch/torch.h>

namespace {

double resolve_stat_value(const std::vector<double>& values, size_t index, double fallback) {
  if (index < values.size() && std::isfinite(values[index])) {
    return values[index];
  }
  return fallback;
}

bool is_valid_torchscript_init_mode(const std::string& init_mode) {
  return init_mode == "load_only" ||
         init_mode == "load_cpu_eval" ||
         init_mode == "load_cuda_no_eval" ||
         init_mode == "load_cuda_eval";
}

std::string normalize_torchscript_init_mode(const std::string& init_mode) {
  if (is_valid_torchscript_init_mode(init_mode)) {
    return init_mode;
  }
  return "load_cuda_eval";
}

bool torchscript_init_moves_to_cuda(const std::string& init_mode) {
  return init_mode == "load_cuda_no_eval" || init_mode == "load_cuda_eval";
}

bool torchscript_init_runs_eval(const std::string& init_mode) {
  return init_mode == "load_cpu_eval" || init_mode == "load_cuda_eval";
}

bool use_fp16_torch_dtype(const std::string& torch_dtype) {
  return torch_dtype == "fp16" || torch_dtype == "half";
}

torch::Tensor select_quantile_along_dim(const torch::Tensor& input, double q, int64_t dim) {
  const auto size = input.size(dim);
  if (size <= 1) {
    return input.select(dim, 0);
  }

  const double clamped = std::clamp(q, 0.0, 1.0);
  const auto rank = static_cast<int64_t>(std::llround(clamped * static_cast<double>(size - 1)));
  return std::get<0>(torch::kthvalue(input, rank + 1, dim, false));
}

torch::Tensor scalar_quantile_tensor(const torch::Tensor& input, double q) {
  auto flat = input.reshape({-1});
  const auto size = flat.size(0);
  if (size <= 1) {
    return flat[0];
  }

  const double clamped = std::clamp(q, 0.0, 1.0);
  const auto rank = static_cast<int64_t>(std::llround(clamped * static_cast<double>(size - 1)));
  return std::get<0>(torch::kthvalue(flat, rank + 1, 0, false));
}

double scalar_quantile(const torch::Tensor& input, double q) {
  return scalar_quantile_tensor(input, q).item<double>();
}

torch::Tensor gaussian_filter1d(const torch::Tensor& input, double sigma) {
  if (sigma <= 0.0) {
    return input.clone();
  }

  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  const auto kernel_size = 2 * radius + 1;
  auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  auto x = torch::arange(-radius, radius + 1, options);
  auto kernel = torch::exp(-(x * x) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();

  auto padded = torch::constant_pad_nd(input.view({1, 1, -1}), {radius, radius}, 0.0);
  auto filtered = torch::conv1d(padded, kernel.view({1, 1, kernel_size}));
  return filtered.view({-1});
}

torch::Tensor gaussian_filter1d_nearest(const torch::Tensor& input, double sigma) {
  if (sigma <= 0.0) {
    return input.clone();
  }

  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  const auto kernel_size = 2 * radius + 1;
  auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  auto x = torch::arange(-radius, radius + 1, options);
  auto kernel = torch::exp(-(x * x) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();

  auto padded = torch::replication_pad1d(input.view({1, 1, -1}), {radius, radius});
  auto filtered = torch::conv1d(padded, kernel.view({1, 1, kernel_size}));
  return filtered.view({-1});
}

torch::Tensor gaussian_filter1d_nearest_batch(const torch::Tensor& input, double sigma) {
  if (sigma <= 0.0) {
    return input.clone();
  }

  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  const auto kernel_size = 2 * radius + 1;
  auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
  auto x = torch::arange(-radius, radius + 1, options);
  auto kernel = torch::exp(-(x * x) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();

  auto padded = torch::replication_pad1d(input.unsqueeze(1), {radius, radius});
  auto filtered = torch::conv1d(padded, kernel.view({1, 1, kernel_size}));
  return filtered.squeeze(1).contiguous();
}

torch::Tensor normalize_map01(const torch::Tensor& input, double low_q, double high_q) {
  auto lo = scalar_quantile_tensor(input, low_q);
  auto hi = scalar_quantile_tensor(input, high_q);
  auto scale = torch::clamp_min(hi - lo, 1e-6);
  return torch::clamp((input - lo) / scale, 0.0, 1.0);
}

torch::Tensor normalize_map01_batch(const torch::Tensor& input, double low_q, double high_q) {
  auto flat = input.reshape({input.size(0), -1});
  auto lo = select_quantile_along_dim(flat, low_q, 1).view({input.size(0), 1, 1});
  auto hi = select_quantile_along_dim(flat, high_q, 1).view({input.size(0), 1, 1});
  auto scale = torch::clamp_min(hi - lo, 1e-6);
  return torch::clamp((input - lo) / scale, 0.0, 1.0);
}

torch::Tensor normalize_map01_masked_minmax(const torch::Tensor& input, const torch::Tensor& valid_mask) {
  auto output = torch::zeros_like(input, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
  auto active = input.masked_select(valid_mask);
  if (active.numel() == 0) {
    return output;
  }

  const double lo = active.min().item<double>();
  const double hi = active.max().item<double>();
  const double scale = std::max(hi - lo, 1e-6);
  auto normalized = torch::clamp((input - lo) / scale, 0.0, 1.0).to(torch::kFloat32);
  return torch::where(valid_mask, normalized, output);
}

torch::Tensor gaussian_kernel_tensor(double sigma, const c10::Device& device) {
  if (sigma <= 0.0) {
    return torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }

  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  auto kernel = torch::exp(-(x * x) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();
  return kernel.contiguous();
}

torch::Tensor gaussian_second_derivative_kernel_tensor(double sigma, const c10::Device& device) {
  if (sigma <= 0.0) {
    return torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  }

  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  const double sigma2 = sigma * sigma;
  auto kernel = ((x * x - sigma2) / (sigma2 * sigma2)) * torch::exp(-(x * x) / (2.0 * sigma2));
  return kernel.contiguous();
}

torch::Tensor convolve_rows_2d(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(0).unsqueeze(0), {0, 0, radius, radius});
  return torch::conv2d(padded, kernel.view({1, 1, kernel.size(0), 1})).squeeze(0).squeeze(0);
}

torch::Tensor convolve_cols_2d(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(0).unsqueeze(0), {radius, radius, 0, 0});
  return torch::conv2d(padded, kernel.view({1, 1, 1, kernel.size(0)})).squeeze(0).squeeze(0);
}

torch::Tensor gaussian_blur_2d(const torch::Tensor& input, double sigma_rows, double sigma_cols) {
  auto row_kernel = gaussian_kernel_tensor(sigma_rows, input.device());
  auto col_kernel = gaussian_kernel_tensor(sigma_cols, input.device());
  return convolve_cols_2d(convolve_rows_2d(input, row_kernel), col_kernel).contiguous();
}

torch::Tensor uniform_filter_2d_nearest(const torch::Tensor& input, int kernel_rows, int kernel_cols) {
  const int row_radius = std::max(0, kernel_rows / 2);
  const int col_radius = std::max(0, kernel_cols / 2);
  auto padded = torch::replication_pad2d(input.unsqueeze(0).unsqueeze(0), {col_radius, col_radius, row_radius, row_radius});
  return torch::avg_pool2d(padded,
                           {std::max(1, kernel_rows), std::max(1, kernel_cols)},
                           {1, 1},
                           {0, 0},
                           false,
                           true)
      .squeeze(0)
      .squeeze(0)
      .contiguous();
}

    torch::Tensor uniform_filter_2d_nearest_batch(const torch::Tensor& input, int kernel_rows, int kernel_cols) {
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

torch::Tensor gaussian_second_derivative_rows_2d(const torch::Tensor& input, double sigma) {
  auto kernel = gaussian_second_derivative_kernel_tensor(sigma, input.device());
  return convolve_rows_2d(input, kernel).contiguous();
}

torch::Tensor make_valid_mask_tensor(int src_rows, int dst_rows, int dst_cols, int ignore_bins_per_side, const c10::Device& device) {
  auto dst_indices = torch::arange(dst_rows, torch::TensorOptions().dtype(torch::kInt64).device(device));
  auto src_indices = torch::div(dst_indices * src_rows, std::max(dst_rows, 1), "floor");
  auto valid_rows = torch::logical_and(src_indices >= ignore_bins_per_side,
                                       src_indices < (src_rows - ignore_bins_per_side));
  return valid_rows.unsqueeze(1).expand({dst_rows, dst_cols}).contiguous();
}

torch::Tensor binary_dilate_rect_tensor(const torch::Tensor& mask, int kernel_rows, int kernel_cols) {
  auto mask_float = mask.to(torch::kFloat32);
  const int row_radius = std::max(0, kernel_rows / 2);
  const int col_radius = std::max(0, kernel_cols / 2);
  auto padded = torch::replication_pad2d(mask_float.unsqueeze(0).unsqueeze(0), {col_radius, col_radius, row_radius, row_radius});
  return torch::max_pool2d(padded, {kernel_rows, kernel_cols}, {1, 1}, {0, 0}).squeeze(0).squeeze(0) > 0.5;
}

torch::Tensor binary_erode_rect_tensor(const torch::Tensor& mask, int kernel_rows, int kernel_cols) {
  auto inverted = torch::logical_not(mask);
  return torch::logical_not(binary_dilate_rect_tensor(inverted, kernel_rows, kernel_cols));
}

torch::Tensor binary_closing_rect_tensor(const torch::Tensor& mask, int kernel_rows, int kernel_cols) {
  return binary_erode_rect_tensor(binary_dilate_rect_tensor(mask, kernel_rows, kernel_cols), kernel_rows, kernel_cols);
}

torch::Tensor signal_agnostic_dino_gray(const torch::Tensor& sxx_db_local) {
  auto x_db = sxx_db_local.to(torch::kFloat32).contiguous();

  const double row_sigma = std::max(1.0, static_cast<double>(x_db.size(0)) / 32.0);
  const double col_sigma = std::max(1.0, static_cast<double>(x_db.size(1)) / 32.0);
  auto row_trend = gaussian_filter1d_nearest(x_db.mean(1), row_sigma).unsqueeze(1);
  auto col_trend = gaussian_filter1d_nearest(x_db.mean(0), col_sigma).unsqueeze(0);
  auto trend = (row_trend + col_trend - x_db.mean()).contiguous();

  auto detrended = (x_db - trend).contiguous();
  auto local_mean = uniform_filter_2d_nearest(detrended, 7, 7);
  auto local_resid = (detrended - local_mean).contiguous();
  auto local_scale = torch::sqrt(uniform_filter_2d_nearest(local_resid * local_resid, 9, 9) + 1e-6);
  auto local_z = (local_resid / torch::clamp_min(local_scale, 1e-4)).contiguous();

  auto abs_detrended = normalize_map01(detrended, 0.02, 0.98);
  auto local_abs = torch::abs(local_z).reshape({-1});
  auto scale = scalar_quantile_tensor(local_abs, 0.95);
  scale = torch::clamp_min(scale, 1e-6);
  auto local_resid_n = torch::clamp(0.5 + 0.5 * (local_z / scale), 0.0, 1.0);

  auto combined = (0.70 * local_resid_n + 0.30 * abs_detrended).to(torch::kFloat32).contiguous();
  const double combined_std = combined.std(/*unbiased=*/false).item<double>();
  if (combined_std < 0.02) {
    combined = normalize_map01(detrended, 0.01, 0.99).to(torch::kFloat32).contiguous();
  }

  // Match the Python path, which converts through uint8 RGB before ToTensor().
  return torch::round(torch::clamp(combined, 0.0, 1.0) * 255.0).div(255.0).contiguous();
}

torch::Tensor signal_agnostic_dino_gray_batch(const torch::Tensor& sxx_db_batch) {
  auto x_db = sxx_db_batch.to(torch::kFloat32).contiguous();

  const double row_sigma = std::max(1.0, static_cast<double>(x_db.size(1)) / 32.0);
  const double col_sigma = std::max(1.0, static_cast<double>(x_db.size(2)) / 32.0);
  auto row_trend = gaussian_filter1d_nearest_batch(x_db.mean(2), row_sigma).unsqueeze(2);
  auto col_trend = gaussian_filter1d_nearest_batch(x_db.mean(1), col_sigma).unsqueeze(1);
  auto trend = (row_trend + col_trend - x_db.mean(2, true).mean(1, true)).contiguous();

  auto detrended = (x_db - trend).contiguous();
  auto local_mean = uniform_filter_2d_nearest_batch(detrended, 7, 7);
  auto local_resid = (detrended - local_mean).contiguous();
  auto local_scale = torch::sqrt(uniform_filter_2d_nearest_batch(local_resid * local_resid, 9, 9) + 1e-6);
  auto local_z = (local_resid / torch::clamp_min(local_scale, 1e-4)).contiguous();

  auto abs_detrended = normalize_map01_batch(detrended, 0.02, 0.98);
  auto local_abs = torch::abs(local_z).reshape({x_db.size(0), -1});
  auto scale = select_quantile_along_dim(local_abs, 0.95, 1).view({x_db.size(0), 1, 1});
  scale = torch::clamp_min(scale, 1e-6);
  auto local_resid_n = torch::clamp(0.5 + 0.5 * (local_z / scale), 0.0, 1.0);

  auto combined = (0.70 * local_resid_n + 0.30 * abs_detrended).to(torch::kFloat32).contiguous();
  auto combined_flat = combined.reshape({combined.size(0), -1});
  auto combined_centered = combined_flat - combined_flat.mean(1, true);
  auto combined_std = torch::sqrt(torch::mean(combined_centered * combined_centered, 1));
  auto fallback = normalize_map01_batch(detrended, 0.01, 0.99).to(torch::kFloat32).contiguous();
  combined = torch::where(combined_std.view({combined.size(0), 1, 1}).lt(0.02), fallback, combined);

  return torch::round(torch::clamp(combined, 0.0, 1.0) * 255.0).div(255.0).contiguous();
}

torch::Tensor legacy_fast_dino_gray(const torch::Tensor& sxx_db_local) {
  return normalize_map01(sxx_db_local, 0.01, 0.99).to(torch::kFloat32).contiguous();
}

torch::Tensor legacy_fast_dino_gray_batch(const torch::Tensor& sxx_db_batch) {
  return normalize_map01_batch(sxx_db_batch, 0.01, 0.99).to(torch::kFloat32).contiguous();
}

torch::Tensor derive_dino_score_map(torch::Tensor model_output,
                                    int aligned_rows,
                                    int aligned_cols,
                                    int patch_size,
                                    int dst_rows,
                                    int dst_cols) {
  const int patch_rows = std::max(1, aligned_rows / std::max(1, patch_size));
  const int patch_cols = std::max(1, aligned_cols / std::max(1, patch_size));
  const int64_t patch_count = static_cast<int64_t>(patch_rows) * static_cast<int64_t>(patch_cols);

  if (model_output.dim() == 4 && model_output.size(0) == 1) {
    model_output = model_output.squeeze(0);
  }

  torch::Tensor base_map;
  if (model_output.dim() == 3) {
    if (model_output.size(0) == 1) {
      model_output = model_output.squeeze(0);
    } else if ((model_output.size(1) == patch_rows && model_output.size(2) == patch_cols) ||
               (model_output.size(1) == dst_rows && model_output.size(2) == dst_cols) ||
               (model_output.size(1) == aligned_rows && model_output.size(2) == aligned_cols)) {
      base_map = torch::sqrt(torch::mean(model_output * model_output, 0) + 1e-6);
    } else if ((model_output.size(0) == patch_rows && model_output.size(1) == patch_cols) ||
               (model_output.size(0) == dst_rows && model_output.size(1) == dst_cols) ||
               (model_output.size(0) == aligned_rows && model_output.size(1) == aligned_cols)) {
      base_map = torch::sqrt(torch::mean(model_output * model_output, -1) + 1e-6);
    }
  }

  if (!base_map.defined() && model_output.dim() == 2) {
    if (model_output.size(0) == patch_count) {
      base_map = torch::sqrt(torch::mean(model_output * model_output, 1) + 1e-6).view({patch_rows, patch_cols});
    } else if (model_output.size(1) == patch_count) {
      auto transposed = model_output.transpose(0, 1);
      base_map = torch::sqrt(torch::mean(transposed * transposed, 1) + 1e-6).view({patch_rows, patch_cols});
    } else if ((model_output.size(0) == patch_rows && model_output.size(1) == patch_cols) ||
               (model_output.size(0) == dst_rows && model_output.size(1) == dst_cols) ||
               (model_output.size(0) == aligned_rows && model_output.size(1) == aligned_cols)) {
      base_map = model_output;
    }
  } else if (!base_map.defined() && model_output.dim() == 1 && model_output.numel() == patch_count) {
    base_map = model_output.view({patch_rows, patch_cols});
  }

  if (!base_map.defined()) {
    std::ostringstream error;
    error << "TorchScript forward returned an unsupported tensor shape for DINO scoring: "
          << model_output.sizes();
    throw std::runtime_error(error.str());
  }
  if (base_map.dim() == 1) {
    base_map = base_map.view({1, -1});
  }
  if (base_map.dim() != 2) {
    throw std::runtime_error("Derived DINO score map is not 2D");
  }

  auto normalized = normalize_map01(base_map.to(torch::kFloat32), 0.05, 0.95);
  auto resized = torch::nn::functional::interpolate(
      normalized.unsqueeze(0).unsqueeze(0),
      torch::nn::functional::InterpolateFuncOptions()
          .size(std::vector<int64_t>{static_cast<int64_t>(dst_rows), static_cast<int64_t>(dst_cols)})
          .mode(torch::kBilinear)
          .align_corners(false));
  return resized.squeeze(0).squeeze(0).contiguous();
}

torch::Tensor extract_patch_feature_matrix(torch::Tensor model_output,
                                          int aligned_rows,
                                          int aligned_cols,
                                          int patch_size) {
  const int patch_rows = std::max(1, aligned_rows / std::max(1, patch_size));
  const int patch_cols = std::max(1, aligned_cols / std::max(1, patch_size));
  const int64_t patch_count = static_cast<int64_t>(patch_rows) * static_cast<int64_t>(patch_cols);

  if (model_output.dim() == 4 && model_output.size(0) == 1) {
    model_output = model_output.squeeze(0);
  }

  torch::Tensor features;
  if (model_output.dim() == 3) {
    if (model_output.size(0) == 1) {
      model_output = model_output.squeeze(0);
    } else if (model_output.size(1) == patch_rows && model_output.size(2) == patch_cols) {
      features = model_output.permute({1, 2, 0}).contiguous().view({patch_count, model_output.size(0)});
    } else if (model_output.size(0) == patch_rows && model_output.size(1) == patch_cols) {
      features = model_output.contiguous().view({patch_count, model_output.size(2)});
    }
  }

  if (!features.defined() && model_output.dim() == 2) {
    if (model_output.size(0) == patch_count) {
      features = model_output.contiguous();
    } else if (model_output.size(1) == patch_count) {
      features = model_output.transpose(0, 1).contiguous();
    }
  }

  if (!features.defined()) {
    std::ostringstream error;
    error << "TorchScript forward returned an unsupported tensor shape for patch features: "
          << model_output.sizes();
    throw std::runtime_error(error.str());
  }
  if (features.dim() != 2 || features.size(0) != patch_count) {
    throw std::runtime_error("Derived patch feature matrix has unexpected shape");
  }
  return features.to(torch::kFloat32).contiguous();
}

template <typename Fn>
double measure_ms(Fn&& fn) {
  const auto start = std::chrono::steady_clock::now();
  fn();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

namespace holoscan::ops {

class DinoTorchRuntime::Impl {
 public:
  DinoTorchRuntimeResult run(const DinoTorchRuntimeConfig& config, const DinoTorchRuntimeInput& input) {
    DinoTorchRuntimeResult result;
    result.backend_used = "pytorch_placeholder";
    result.aligned_rows = std::max(1, input.patch_size);
    result.aligned_cols = std::max(1, input.patch_size);
    result.final_threshold = config.pipeline_final_threshold;
    std::string failure_stage = "input_validation";
    std::string failure_detail;

    try {
      torch::InferenceMode inference_mode_guard(true);
      const size_t expected_bins = static_cast<size_t>(input.src_rows) * static_cast<size_t>(input.src_cols);

      failure_detail = std::string("channel=") + std::to_string(input.channel_number) +
                       " frame=" + std::to_string(input.frame_number) +
                       " src=" + std::to_string(input.src_rows) + "x" + std::to_string(input.src_cols) +
                       " dst=" + std::to_string(input.dst_rows) + "x" + std::to_string(input.dst_cols) +
                       " patch=" + std::to_string(input.patch_size);

      const auto init_mode = normalize_torchscript_init_mode(config.torchscript_init_mode);
      const bool use_cuda_torch = (config.inference_backend != "torchscript") || torchscript_init_moves_to_cuda(init_mode);
      const bool use_fp16 = use_fp16_torch_dtype(config.torch_dtype) && use_cuda_torch;
      const bool needs_power_db = config.compute_power_score || (!use_cuda_torch) || (input.corrected_db_device == nullptr);
      c10::Device compute_device = use_cuda_torch ? c10::Device(torch::kCUDA, 0) : c10::Device(torch::kCPU);
      std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;
      if (use_cuda_torch) {
        const auto torch_stream = input.cuda_stream
                                      ? c10::cuda::getStreamFromExternal(input.cuda_stream, compute_device.index())
                                      : c10::cuda::getDefaultCUDAStream(compute_device.index());
        stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(torch_stream);
      }

      failure_stage = "power_db_tensor_create";
      torch::Tensor power_db;
      if (use_cuda_torch && input.power_db_device && needs_power_db) {
        auto device_float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
        power_db = torch::from_blob(const_cast<float*>(input.power_db_device),
                                    {static_cast<int64_t>(input.src_rows), static_cast<int64_t>(input.src_cols)},
                                    device_float_options);
      } else if (needs_power_db) {
        if (!input.power_db || input.power_db->size() != expected_bins) {
          throw std::runtime_error("Invalid power_db input buffer");
        }
        auto cpu_float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        power_db = torch::from_blob(const_cast<float*>(input.power_db->data()),
                                    {static_cast<int64_t>(input.src_rows), static_cast<int64_t>(input.src_cols)},
                                    cpu_float_options)
                       .clone();
        if (use_cuda_torch) {
          failure_stage = "power_db_to_device";
          power_db = power_db.to(compute_device);
        }
      }

      auto backend = config.inference_backend;
      if (backend == "torchscript") {
        failure_stage = "torchscript_load";
        ensure_loaded(config, compute_device);
        if (torchscript_load_failed_ || !torchscript_forward_ready_) {
          throw std::runtime_error("TorchScript backend requested but is not ready");
        }
      }
      torch::Tensor corrected_db;
      if (use_cuda_torch && input.corrected_db_device) {
        auto device_float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
        corrected_db = torch::from_blob(const_cast<float*>(input.corrected_db_device),
                                        {static_cast<int64_t>(input.src_rows), static_cast<int64_t>(input.src_cols)},
                                        device_float_options)
                           .contiguous();
        result.timing.frontend_correction_ms = 0.0;
      } else {
        failure_stage = "frontend_correction";
        result.timing.frontend_correction_ms = measure_ms([&] {
          corrected_db = power_db.contiguous();
          if (!config.frontend_correction_enable) {
            return;
          }
          auto col_activity = select_quantile_along_dim(corrected_db, 0.85, 0);
          const double quiet_threshold = scalar_quantile(col_activity, 0.70);
          auto quiet_mask = col_activity.le(quiet_threshold);
          auto quiet_indices = torch::nonzero(quiet_mask).reshape({-1});
          const auto min_quiet_cols = std::max<int64_t>(16, corrected_db.size(1) / 8);
          torch::Tensor quiet_view = corrected_db;
          if (quiet_indices.numel() >= min_quiet_cols) {
            quiet_view = corrected_db.index({torch::indexing::Slice(), quiet_indices});
          }

          auto row_floor = select_quantile_along_dim(quiet_view, config.frontend_correction_row_q / 100.0, 1);
          auto response = gaussian_filter1d(row_floor, config.frontend_correction_smooth_sigma);
          const int64_t num_rows = response.size(0);
          const int64_t inner_span = std::min<int64_t>(num_rows, std::max<int64_t>(8, static_cast<int64_t>(std::llround(0.65 * static_cast<double>(num_rows)))));
          const int64_t inner_start = std::max<int64_t>(0, (num_rows - inner_span) / 2);
          const int64_t inner_stop = std::min<int64_t>(num_rows, inner_start + inner_span);
          auto inner_response = response.index({torch::indexing::Slice(inner_start, inner_stop)});
          if (inner_response.numel() == 0) {
            inner_response = response;
          }

          const double reference_level = scalar_quantile(inner_response, config.frontend_correction_reference_q / 100.0);
          auto edge_profile = torch::zeros({num_rows}, torch::TensorOptions().dtype(torch::kFloat32).device(corrected_db.device()));
          const int64_t edge_rows = std::min<int64_t>(std::max<int64_t>(1, num_rows / 2),
                                                      std::max<int64_t>(8, static_cast<int64_t>(std::llround(config.frontend_correction_edge_taper_fraction * static_cast<double>(num_rows)))));
          if (edge_rows > 0) {
            auto ramp = torch::linspace(1.0, 0.0, edge_rows, edge_profile.options());
            edge_profile.index_put_({torch::indexing::Slice(0, edge_rows)}, ramp);
            edge_profile.index_put_({torch::indexing::Slice(num_rows - edge_rows, num_rows)},
                                    torch::maximum(edge_profile.index({torch::indexing::Slice(num_rows - edge_rows, num_rows)}),
                                                   torch::linspace(0.0, 1.0, edge_rows, edge_profile.options())));
          }
          edge_profile = gaussian_filter1d(edge_profile, config.frontend_correction_edge_taper_sigma);
          edge_profile = edge_profile / torch::clamp(edge_profile.max(), 1e-6);
          auto target_response = reference_level - config.frontend_correction_edge_target_drop_db * edge_profile;
          auto target_deficit = torch::clamp_min(target_response - response, 0.0);
          const double soft_knee = std::max(config.frontend_correction_soft_knee_db, 1e-3);
          auto soft_boost = config.frontend_correction_max_boost_db * (1.0 - torch::exp(-target_deficit / soft_knee));
          soft_boost = torch::minimum(soft_boost, target_deficit);
          auto boost_db = gaussian_filter1d(soft_boost, std::max(1.0, config.frontend_correction_smooth_sigma / 2.0));
          boost_db = torch::minimum(boost_db, target_deficit);
          corrected_db = corrected_db + boost_db.unsqueeze(1);
        });
      }

      failure_stage = "crop_align";
      result.timing.crop_align_ms = measure_ms([&] {
        result.freq_bin_hz = input.resolution_hz > 0.0 ? input.resolution_hz : (input.span_hz > 0.0 ? input.span_hz / static_cast<double>(input.src_rows) : 0.0);
        if (config.ignore_sideband_hz > 0.0 && result.freq_bin_hz > 0.0) {
          const int requested_bins = static_cast<int>(std::ceil(config.ignore_sideband_hz / result.freq_bin_hz));
          const int max_ignore_bins = std::max(0, (input.src_rows - input.patch_size) / 2);
          result.ignore_bins_per_side = std::min(requested_bins, max_ignore_bins);
        }
        if (result.ignore_bins_per_side > 0 && (2 * result.ignore_bins_per_side) < corrected_db.size(0)) {
          corrected_db = corrected_db.index({torch::indexing::Slice(result.ignore_bins_per_side, corrected_db.size(0) - result.ignore_bins_per_side), torch::indexing::Slice()});
        }

        const int safe_patch = std::max(1, input.patch_size);
        const int source_aligned_rows = static_cast<int>((corrected_db.size(0) / safe_patch) * safe_patch);
        const int source_aligned_cols = static_cast<int>((corrected_db.size(1) / safe_patch) * safe_patch);
        if (source_aligned_rows < safe_patch || source_aligned_cols < safe_patch) {
          throw std::runtime_error("Corrected chunk is too small for DINO patch-aligned input");
        }
        const int target_aligned_rows = std::max(safe_patch, (std::max(1, input.dst_rows) / safe_patch) * safe_patch);
        const int target_aligned_cols = std::max(safe_patch, (std::max(1, input.dst_cols) / safe_patch) * safe_patch);
        result.input_resized_to_target = target_aligned_rows < source_aligned_rows || target_aligned_cols < source_aligned_cols;
        result.aligned_rows = result.input_resized_to_target ? target_aligned_rows : source_aligned_rows;
        result.aligned_cols = result.input_resized_to_target ? target_aligned_cols : source_aligned_cols;
      });

      torch::Tensor resized_db;
      failure_stage = "resize";
      result.timing.resize_ms = measure_ms([&] {
        if (result.input_resized_to_target) {
          resized_db = torch::nn::functional::interpolate(
                           corrected_db.unsqueeze(0).unsqueeze(0),
                           torch::nn::functional::InterpolateFuncOptions()
                               .size(std::vector<int64_t>{static_cast<int64_t>(result.aligned_rows), static_cast<int64_t>(result.aligned_cols)})
                               .mode(torch::kBilinear)
                               .align_corners(false))
                           .squeeze(0)
                           .squeeze(0)
                           .contiguous();
        } else if (corrected_db.size(0) == result.aligned_rows && corrected_db.size(1) == result.aligned_cols) {
          resized_db = corrected_db.contiguous();
        } else {
          resized_db = corrected_db.index({
              torch::indexing::Slice(0, result.aligned_rows),
              torch::indexing::Slice(0, result.aligned_cols),
          }).contiguous();
        }
      });

      torch::Tensor model_input;
      failure_stage = "model_prep";
      result.timing.model_prep_ms = measure_ms([&] {
        auto grayscale_2d = (config.legacy_fast_gray_preprocess
                                 ? legacy_fast_dino_gray(resized_db)
                                 : signal_agnostic_dino_gray(resized_db))
                                .contiguous();
        const auto target_dtype = use_fp16 ? torch::kHalf : torch::kFloat32;
        auto grayscale = grayscale_2d.unsqueeze(0).unsqueeze(0).to(target_dtype);
        auto rgb = grayscale.expand({1, 3, grayscale.size(2), grayscale.size(3)}).contiguous(torch::MemoryFormat::ChannelsLast);
        const auto [mean, std] = get_normalization_tensors(config, compute_device, target_dtype);
        model_input = ((rgb - mean) / std).contiguous(torch::MemoryFormat::ChannelsLast);

        if (config.return_pre_model_gray) {
          auto pre_model_gray_cpu = grayscale_2d.device().is_cuda() ? grayscale_2d.to(torch::kCPU) : grayscale_2d;
          result.pre_model_gray.resize(static_cast<size_t>(result.aligned_rows) * static_cast<size_t>(result.aligned_cols));
          std::memcpy(result.pre_model_gray.data(), pre_model_gray_cpu.data_ptr<float>(), result.pre_model_gray.size() * sizeof(float));
        }
      });

      torch::Tensor dino_score;
      if (backend == "torchscript") {
        torch::Tensor model_output;
        torch::Tensor patch_features;
        failure_stage = "torch_forward";
        result.timing.torch_forward_ms = measure_ms([&] {
          auto raw_output = torchscript_module_->forward({model_input});
          if (raw_output.isTensor()) {
            model_output = raw_output.toTensor();
          } else if (raw_output.isTuple()) {
            auto tuple_ptr = raw_output.toTuple();
            if (tuple_ptr && !tuple_ptr->elements().empty() && tuple_ptr->elements()[0].isTensor()) {
              model_output = tuple_ptr->elements()[0].toTensor();
            }
          }
          if (!model_output.defined()) {
            throw std::runtime_error("TorchScript forward returned non-tensor output");
          }
        });

        failure_stage = "dino_score";
        result.timing.dino_score_ms = measure_ms([&] {
          patch_features = extract_patch_feature_matrix(model_output, result.aligned_rows, result.aligned_cols, input.patch_size);
          dino_score = derive_dino_score_map(model_output, result.aligned_rows, result.aligned_cols, input.patch_size, input.dst_rows, input.dst_cols);
        });
        if (config.return_patch_features && patch_features.defined()) {
          result.patch_rows = std::max(1, result.aligned_rows / std::max(1, input.patch_size));
          result.patch_cols = std::max(1, result.aligned_cols / std::max(1, input.patch_size));
          result.feature_dim = static_cast<int>(patch_features.size(1));
          if (config.return_patch_features_host) {
            auto patch_features_cpu = patch_features.device().is_cuda() ? patch_features.to(torch::kCPU) : patch_features;
            result.patch_features.resize(static_cast<size_t>(patch_features_cpu.numel()));
            std::memcpy(result.patch_features.data(),
                        patch_features_cpu.data_ptr<float>(),
                        result.patch_features.size() * sizeof(float));
          }
        }
        result.backend_used = "torchscript";
      } else {
        result.timing.dino_score_ms = measure_ms([&] {
          auto resized_placeholder = torch::nn::functional::interpolate(
                                       resized_db.unsqueeze(0).unsqueeze(0),
                                       torch::nn::functional::InterpolateFuncOptions()
                                           .size(std::vector<int64_t>{static_cast<int64_t>(input.dst_rows), static_cast<int64_t>(input.dst_cols)})
                                           .mode(torch::kBilinear)
                                           .align_corners(false))
                                       .squeeze(0)
                                       .squeeze(0);
          dino_score = normalize_map01(resized_placeholder, 0.05, 0.95).contiguous();
        });
        result.backend_used = "pytorch_placeholder";
      }

      if (config.compute_power_score) {
        torch::Tensor power_score;
        failure_stage = "power_score";
        result.timing.power_score_ms = measure_ms([&] {
          auto resized_power = torch::nn::functional::interpolate(
                                   corrected_db.unsqueeze(0).unsqueeze(0),
                                   torch::nn::functional::InterpolateFuncOptions()
                                       .size(std::vector<int64_t>{static_cast<int64_t>(input.dst_rows), static_cast<int64_t>(input.dst_cols)})
                                       .mode(torch::kBilinear)
                                       .align_corners(false))
                                   .squeeze(0)
                                   .squeeze(0)
                                   .contiguous();
          power_score = normalize_map01(resized_power, 0.05, 0.95).contiguous();
          result.power_threshold = scalar_quantile(power_score, std::clamp(config.power_q, 0.0, 1.0));
        });
      }

      torch::Tensor final_score;
      failure_stage = "fusion";
      result.timing.fusion_ms = measure_ms([&] {
        if (config.compute_dino_threshold) {
          result.dino_threshold = scalar_quantile(dino_score, std::clamp(config.dino_group_score_q, 0.0, 1.0));
        } else {
          result.dino_threshold = config.pipeline_final_threshold;
        }
        result.final_threshold = result.dino_threshold;
        final_score = dino_score.contiguous();
      });

      if (config.return_final_mask_device) {
        auto final_score_device = final_score.contiguous();
        result.score_map_device = final_score_device.data_ptr<float>();
        result.score_map_device_owner = std::make_shared<torch::Tensor>(final_score_device);
        result.final_mask_device = final_score_device.data_ptr<float>();
        result.final_mask_device_owner = result.score_map_device_owner;
      }

      if (config.return_final_mask) {
        failure_stage = "final_mask_to_cpu";
        result.timing.score_to_cpu_ms = measure_ms([&] {
          auto final_mask_cpu = final_score.device().is_cuda() ? final_score.to(torch::kCPU) : final_score;
          result.score_map.resize(static_cast<size_t>(input.dst_rows) * static_cast<size_t>(input.dst_cols));
          result.final_mask.resize(static_cast<size_t>(input.dst_rows) * static_cast<size_t>(input.dst_cols));
          std::memcpy(result.score_map.data(), final_mask_cpu.data_ptr<float>(), result.score_map.size() * sizeof(float));
          result.final_mask = result.score_map;
        });
      }
      result.torchscript_forward_ready = torchscript_forward_ready_;
      result.success = true;
      return result;
    } catch (const std::exception& error) {
      result.error_stage = failure_stage;
      result.error_message = error.what();
      result.error_detail = failure_detail + (last_detail_.empty() ? std::string() : std::string(" detail=") + last_detail_);
      return result;
    }
  }

  DinoTorchRuntimeBatchResult run_batch(const DinoTorchRuntimeConfig& config, const DinoTorchRuntimeBatchInput& input) {
    DinoTorchRuntimeBatchResult result;
    result.aligned_rows = std::max(1, input.patch_size);
    result.aligned_cols = std::max(1, input.patch_size);
    std::string failure_stage = "input_validation";
    std::string failure_detail;

    try {
      torch::InferenceMode inference_mode_guard(true);
      if (input.batch_size <= 0 || input.src_rows <= 0 || input.src_cols <= 0 ||
          input.dst_rows <= 0 || input.dst_cols <= 0 || input.corrected_db_batch_device == nullptr) {
        throw std::runtime_error("Invalid batch runtime input");
      }
      if (config.compute_power_score) {
        throw std::runtime_error("Batch runtime path does not support power score computation");
      }
      if (config.return_pre_model_gray) {
        throw std::runtime_error("Batch runtime path does not support pre-model grayscale capture");
      }

      failure_detail = std::string("batch=") + std::to_string(input.batch_size) +
                       " src=" + std::to_string(input.src_rows) + "x" + std::to_string(input.src_cols) +
                       " dst=" + std::to_string(input.dst_rows) + "x" + std::to_string(input.dst_cols) +
                       " patch=" + std::to_string(input.patch_size);

      const auto init_mode = normalize_torchscript_init_mode(config.torchscript_init_mode);
      const bool use_cuda_torch = (config.inference_backend != "torchscript") || torchscript_init_moves_to_cuda(init_mode);
      const bool use_fp16 = use_fp16_torch_dtype(config.torch_dtype) && use_cuda_torch;
      c10::Device compute_device = use_cuda_torch ? c10::Device(torch::kCUDA, 0) : c10::Device(torch::kCPU);
      std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;
      if (use_cuda_torch) {
        const auto torch_stream = input.cuda_stream
                                      ? c10::cuda::getStreamFromExternal(input.cuda_stream, compute_device.index())
                                      : c10::cuda::getDefaultCUDAStream(compute_device.index());
        stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(torch_stream);
      }

      std::string backend = config.inference_backend;
      if (backend == "torchscript") {
        failure_stage = "torchscript_load";
        ensure_loaded(config, compute_device);
        if (torchscript_load_failed_ || !torchscript_forward_ready_) {
          throw std::runtime_error("TorchScript backend requested but is not ready");
        }
      }

      const int safe_patch = std::max(1, input.patch_size);
      const int source_aligned_rows = static_cast<int>((input.src_rows / safe_patch) * safe_patch);
      const int source_aligned_cols = static_cast<int>((input.src_cols / safe_patch) * safe_patch);
      if (source_aligned_rows < safe_patch || source_aligned_cols < safe_patch) {
        throw std::runtime_error("Corrected batch chunk is too small for DINO patch-aligned input");
      }
      const int target_aligned_rows = std::max(safe_patch, (std::max(1, input.dst_rows) / safe_patch) * safe_patch);
      const int target_aligned_cols = std::max(safe_patch, (std::max(1, input.dst_cols) / safe_patch) * safe_patch);
      result.input_resized_to_target = target_aligned_rows < source_aligned_rows || target_aligned_cols < source_aligned_cols;
      result.aligned_rows = result.input_resized_to_target ? target_aligned_rows : source_aligned_rows;
      result.aligned_cols = result.input_resized_to_target ? target_aligned_cols : source_aligned_cols;

      auto device_float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
      auto corrected_batch = torch::from_blob(const_cast<float*>(input.corrected_db_batch_device),
                                              {static_cast<int64_t>(input.batch_size), static_cast<int64_t>(input.src_rows), static_cast<int64_t>(input.src_cols)},
                                              device_float_options);
      torch::Tensor resized_batch;
      if (result.input_resized_to_target) {
        resized_batch = torch::nn::functional::interpolate(
                           corrected_batch.unsqueeze(1),
                           torch::nn::functional::InterpolateFuncOptions()
                               .size(std::vector<int64_t>{static_cast<int64_t>(result.aligned_rows), static_cast<int64_t>(result.aligned_cols)})
                               .mode(torch::kBilinear)
                               .align_corners(false))
                           .squeeze(1)
                           .contiguous();
      } else {
        resized_batch = corrected_batch.index({torch::indexing::Slice(),
                                               torch::indexing::Slice(0, result.aligned_rows),
                                               torch::indexing::Slice(0, result.aligned_cols)})
                           .contiguous();
      }

      failure_stage = "model_prep";
      torch::Tensor model_input;
      result.timing.model_prep_ms = measure_ms([&] {
        const auto target_dtype = use_fp16 ? torch::kHalf : torch::kFloat32;
        auto grayscale_batch = (config.legacy_fast_gray_preprocess
                                    ? legacy_fast_dino_gray_batch(resized_batch)
                                    : signal_agnostic_dino_gray_batch(resized_batch))
                                   .unsqueeze(1)
                                   .to(target_dtype);
        auto rgb_batch = grayscale_batch.expand({static_cast<int64_t>(input.batch_size), 3, grayscale_batch.size(2), grayscale_batch.size(3)})
                             .contiguous(torch::MemoryFormat::ChannelsLast);
        const auto [mean, std] = get_normalization_tensors(config, compute_device, target_dtype);
        model_input = ((rgb_batch - mean) / std).contiguous(torch::MemoryFormat::ChannelsLast);
      });

      torch::Tensor score_maps_batch;
      if (backend == "torchscript") {
        torch::Tensor model_output;
        std::vector<torch::Tensor> patch_features_per_sample;
        failure_stage = "torch_forward";
        result.timing.torch_forward_ms = measure_ms([&] {
          auto raw_output = torchscript_module_->forward({model_input});
          if (raw_output.isTensor()) {
            model_output = raw_output.toTensor();
          } else if (raw_output.isTuple()) {
            auto tuple_ptr = raw_output.toTuple();
            if (tuple_ptr && !tuple_ptr->elements().empty() && tuple_ptr->elements()[0].isTensor()) {
              model_output = tuple_ptr->elements()[0].toTensor();
            }
          }
        });
        if (!model_output.defined()) {
          throw std::runtime_error("TorchScript forward returned non-tensor output");
        }

        failure_stage = "dino_score";
        result.timing.dino_score_ms = measure_ms([&] {
          std::vector<torch::Tensor> score_maps;
          score_maps.reserve(static_cast<size_t>(input.batch_size));
          if (config.return_patch_features) {
            patch_features_per_sample.reserve(static_cast<size_t>(input.batch_size));
          }
          result.dino_thresholds.resize(static_cast<size_t>(input.batch_size), config.pipeline_final_threshold);
          result.final_thresholds.resize(static_cast<size_t>(input.batch_size), config.pipeline_final_threshold);
          for (int sample_index = 0; sample_index < input.batch_size; ++sample_index) {
            torch::Tensor sample_output;
            if (model_output.dim() >= 1 && model_output.size(0) == input.batch_size) {
              sample_output = model_output[sample_index];
            } else if (input.batch_size == 1) {
              sample_output = model_output;
            } else {
              throw std::runtime_error("TorchScript batch output does not expose a batch dimension");
            }
            if (config.return_patch_features) {
              patch_features_per_sample.push_back(extract_patch_feature_matrix(sample_output,
                                                                               result.aligned_rows,
                                                                               result.aligned_cols,
                                                                               input.patch_size));
            }
            auto dino_score = derive_dino_score_map(sample_output,
                                                    result.aligned_rows,
                                                    result.aligned_cols,
                                                    input.patch_size,
                                                    input.dst_rows,
                                                    input.dst_cols);
            const double threshold = config.compute_dino_threshold
                                         ? scalar_quantile(dino_score, std::clamp(config.dino_group_score_q, 0.0, 1.0))
                                         : config.pipeline_final_threshold;
            result.dino_thresholds[static_cast<size_t>(sample_index)] = threshold;
            result.final_thresholds[static_cast<size_t>(sample_index)] = threshold;
            score_maps.push_back(dino_score);
          }
          score_maps_batch = torch::stack(score_maps, 0).contiguous();
        });
        if (config.return_patch_features && !patch_features_per_sample.empty()) {
          auto patch_features_batch = torch::stack(patch_features_per_sample, 0).contiguous();
          auto patch_features_batch_device = patch_features_batch.contiguous();
          auto patch_features_batch_device_owner = std::make_shared<torch::Tensor>(patch_features_batch_device);
          result.patch_features_batch_device = patch_features_batch_device_owner->data_ptr<float>();
          result.patch_features_batch_device_owner = patch_features_batch_device_owner;
          result.patch_rows = std::max(1, result.aligned_rows / std::max(1, input.patch_size));
          result.patch_cols = std::max(1, result.aligned_cols / std::max(1, input.patch_size));
          result.feature_dim = patch_features_batch.dim() >= 3 ? static_cast<int>(patch_features_batch.size(2)) : 0;
          if (config.return_patch_features_host) {
            auto patch_features_cpu = patch_features_batch.device().is_cuda() ? patch_features_batch.to(torch::kCPU) : patch_features_batch;
            result.patch_features_batch.resize(static_cast<size_t>(patch_features_cpu.numel()));
            std::memcpy(result.patch_features_batch.data(),
                        patch_features_cpu.data_ptr<float>(),
                        result.patch_features_batch.size() * sizeof(float));
          }
        }
        result.backend_used = "torchscript";
      } else {
        result.timing.dino_score_ms = measure_ms([&] {
          auto resized_placeholder = torch::nn::functional::interpolate(
                                       resized_batch.unsqueeze(1),
                                       torch::nn::functional::InterpolateFuncOptions()
                                           .size(std::vector<int64_t>{static_cast<int64_t>(input.dst_rows), static_cast<int64_t>(input.dst_cols)})
                                           .mode(torch::kBilinear)
                                           .align_corners(false))
                                       .squeeze(1)
                                       .contiguous();
          std::vector<torch::Tensor> score_maps;
          score_maps.reserve(static_cast<size_t>(input.batch_size));
          result.dino_thresholds.resize(static_cast<size_t>(input.batch_size), config.pipeline_final_threshold);
          result.final_thresholds.resize(static_cast<size_t>(input.batch_size), config.pipeline_final_threshold);
          for (int sample_index = 0; sample_index < input.batch_size; ++sample_index) {
            auto score = normalize_map01(resized_placeholder[sample_index], 0.05, 0.95).contiguous();
            score_maps.push_back(score);
          }
          score_maps_batch = torch::stack(score_maps, 0).contiguous();
        });
        result.backend_used = "pytorch_placeholder";
      }

      if (config.return_final_mask_device) {
        auto score_maps_device = score_maps_batch.contiguous();
        auto score_maps_device_owner = std::make_shared<torch::Tensor>(score_maps_device);
        result.score_maps_device = score_maps_device_owner->data_ptr<float>();
        result.score_maps_device_owner = score_maps_device_owner;
        result.timing.score_to_cpu_ms = 0.0;
      } else {
        result.timing.score_to_cpu_ms = measure_ms([&] {
          auto score_maps_cpu = score_maps_batch.device().is_cuda() ? score_maps_batch.to(torch::kCPU) : score_maps_batch;
          result.score_maps.resize(static_cast<size_t>(score_maps_cpu.numel()));
          std::memcpy(result.score_maps.data(), score_maps_cpu.data_ptr<float>(), result.score_maps.size() * sizeof(float));
        });
      }
      result.torchscript_forward_ready = torchscript_forward_ready_;
      result.success = true;
      return result;
    } catch (const std::exception& error) {
      result.error_stage = failure_stage;
      result.error_message = error.what();
      result.error_detail = failure_detail + (last_detail_.empty() ? std::string() : std::string(" detail=") + last_detail_);
      return result;
    }
  }

  DinoHybridPostGpuResult run_hybrid_post_gpu(const DinoHybridPostGpuInput& input) {
    DinoHybridPostGpuResult result;
    if (input.dst_rows <= 0 || input.dst_cols <= 0 || input.src_rows <= 0 ||
        input.dino_score_device == nullptr || input.coherence_gate_device == nullptr) {
      result.error_message = "Invalid GPU hybrid postprocess input";
      return result;
    }

    try {
      torch::InferenceMode inference_mode_guard(true);
      c10::Device compute_device(torch::kCUDA, 0);
      const auto torch_stream = input.cuda_stream
                                    ? c10::cuda::getStreamFromExternal(input.cuda_stream, compute_device.index())
                                    : c10::cuda::getDefaultCUDAStream(compute_device.index());
      c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

      auto tensor_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
      auto dino_score = torch::from_blob(const_cast<float*>(input.dino_score_device),
                                         {static_cast<int64_t>(input.dst_rows), static_cast<int64_t>(input.dst_cols)},
                                         tensor_options);
      auto coherence_gate = torch::from_blob(const_cast<float*>(input.coherence_gate_device),
                                             {static_cast<int64_t>(input.dst_rows), static_cast<int64_t>(input.dst_cols)},
                                             tensor_options);
      auto valid_mask = make_valid_mask_tensor(input.src_rows,
                                               input.dst_rows,
                                               input.dst_cols,
                                               input.ignore_bins_per_side,
                                               compute_device);

      auto base_map = dino_score * coherence_gate;
      auto base_norm = normalize_map01_masked_minmax(base_map, valid_mask);
      auto envelope_map = normalize_map01_masked_minmax(gaussian_blur_2d(base_norm, 6.0, 1.4), valid_mask);
      auto base_blur = gaussian_blur_2d(base_norm, 4.0, 1.0);
      auto residual_penalty = normalize_map01_masked_minmax(gaussian_blur_2d(torch::abs(base_norm - base_blur), 2.0, 0.8), valid_mask);
      auto freq_curvature_penalty = normalize_map01_masked_minmax(torch::abs(gaussian_second_derivative_rows_2d(base_norm, 0.8)), valid_mask);

      auto keep_freq = normalize_map01_masked_minmax(envelope_map - 0.90 * freq_curvature_penalty, valid_mask);
      auto keep_res = normalize_map01_masked_minmax(envelope_map - 1.00 * residual_penalty, valid_mask);
      auto active_freq = keep_freq.masked_select(valid_mask);
      auto active_res = keep_res.masked_select(valid_mask);

      result.seed_freq_threshold = static_cast<float>(scalar_quantile(active_freq, 0.90));
      result.seed_res_threshold = static_cast<float>(scalar_quantile(active_res, 0.82));
      result.grow_freq_threshold = result.seed_freq_threshold;
      result.grow_res_threshold = result.seed_res_threshold;
      result.combined_threshold = result.seed_freq_threshold;

      auto seed_mask = torch::logical_and(valid_mask,
                                          torch::logical_and(keep_freq >= result.seed_freq_threshold,
                                                             keep_res >= result.seed_res_threshold));

      auto seed_mask_cpu = seed_mask.to(torch::kCPU, torch::kUInt8).contiguous();
      const size_t elements = static_cast<size_t>(input.dst_rows) * static_cast<size_t>(input.dst_cols);
      result.seed_mask.resize(elements);
      std::memcpy(result.seed_mask.data(), seed_mask_cpu.data_ptr<uint8_t>(), elements * sizeof(uint8_t));
      result.success = true;
      return result;
    } catch (const std::exception& error) {
      result.error_message = error.what();
      return result;
    }
  }

  void warmup(const DinoTorchRuntimeConfig& config,
              int src_rows,
              int src_cols,
              int dst_rows,
              int dst_cols,
              int patch_size,
              int batch_size,
              cudaStream_t cuda_stream) {
    const auto init_mode = normalize_torchscript_init_mode(config.torchscript_init_mode);
    if (config.inference_backend != "torchscript") {
      return;
    }

    const bool use_cuda_torch = torchscript_init_moves_to_cuda(init_mode);
    const bool use_fp16 = use_fp16_torch_dtype(config.torch_dtype) && use_cuda_torch;
    const c10::Device compute_device = use_cuda_torch ? c10::Device(torch::kCUDA, 0) : c10::Device(torch::kCPU);

    torch::InferenceMode inference_mode_guard(true);
    std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;
    if (use_cuda_torch) {
      const auto torch_stream = cuda_stream
                                    ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                    : c10::cuda::getDefaultCUDAStream(compute_device.index());
      stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(torch_stream);
    }

    ensure_loaded(config, compute_device);
    if (!torchscript_forward_ready_ || !torchscript_module_) {
      return;
    }

    const auto warmup_dtype = use_fp16 ? torch::kHalf : torch::kFloat32;

    // Materialize normalization tensors ahead of live frames so model_prep can
    // reuse them without paying a first-frame allocation penalty.
    (void)get_normalization_tensors(config, compute_device, warmup_dtype);

    const int safe_batch = std::max(1, batch_size);
    const int safe_src_rows = std::max(1, src_rows);
    const int safe_src_cols = std::max(1, src_cols);
    auto warmup_corrected = torch::zeros({safe_batch, safe_src_rows, safe_src_cols},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(compute_device));

    DinoTorchRuntimeBatchInput warmup_input;
    warmup_input.batch_size = safe_batch;
    warmup_input.src_rows = safe_src_rows;
    warmup_input.src_cols = safe_src_cols;
    warmup_input.dst_rows = std::max(1, dst_rows);
    warmup_input.dst_cols = std::max(1, dst_cols);
    warmup_input.patch_size = std::max(1, patch_size);
    warmup_input.cuda_stream = cuda_stream;
    warmup_input.corrected_db_batch_device = warmup_corrected.data_ptr<float>();

    auto warmup_config = config;
    warmup_config.compute_power_score = false;
    warmup_config.return_patch_features = false;
    warmup_config.return_pre_model_gray = false;

    auto warmup_result = run_batch(warmup_config, warmup_input);
    if (!warmup_result.success) {
      throw std::runtime_error("TorchScript warmup batch failed at " + warmup_result.error_stage + ": " + warmup_result.error_message);
    }

    if (safe_batch == 1) {
      DinoTorchRuntimeInput warmup_single_input;
      warmup_single_input.src_rows = safe_src_rows;
      warmup_single_input.src_cols = safe_src_cols;
      warmup_single_input.dst_rows = std::max(1, dst_rows);
      warmup_single_input.dst_cols = std::max(1, dst_cols);
      warmup_single_input.patch_size = std::max(1, patch_size);
      warmup_single_input.cuda_stream = cuda_stream;
      warmup_single_input.corrected_db_device = warmup_corrected.data_ptr<float>();

      auto warmup_single_config = warmup_config;
      warmup_single_config.return_final_mask = false;
      warmup_single_config.return_final_mask_device = false;

      auto warmup_single_result = run(warmup_single_config, warmup_single_input);
      if (!warmup_single_result.success) {
        throw std::runtime_error("TorchScript warmup single failed at " + warmup_single_result.error_stage + ": " + warmup_single_result.error_message);
      }
    }

    if (use_cuda_torch) {
      const auto sync_result = cudaDeviceSynchronize();
      if (sync_result != cudaSuccess) {
        throw std::runtime_error(std::string("TorchScript warmup synchronization failed: ") +
                                 cudaGetErrorString(sync_result));
      }
    }
  }

 private:
  struct NormalizationTensorCache {
    bool valid = false;
    c10::Device device = c10::Device(torch::kCPU);
    c10::ScalarType dtype = torch::kFloat32;
    std::array<float, 3> mean_values{0.485f, 0.456f, 0.406f};
    std::array<float, 3> std_values{0.229f, 0.224f, 0.225f};
    torch::Tensor mean_tensor;
    torch::Tensor std_tensor;
  };

  std::pair<torch::Tensor, torch::Tensor> get_normalization_tensors(const DinoTorchRuntimeConfig& config,
                                                                    const c10::Device& device,
                                                                    c10::ScalarType dtype) {
    std::lock_guard<std::mutex> lock(normalization_cache_mutex_);

    const std::array<float, 3> mean_values{
        static_cast<float>(resolve_stat_value(config.imagenet_mean, 0, 0.485)),
        static_cast<float>(resolve_stat_value(config.imagenet_mean, 1, 0.456)),
        static_cast<float>(resolve_stat_value(config.imagenet_mean, 2, 0.406)),
    };
    const std::array<float, 3> std_values{
        static_cast<float>(resolve_stat_value(config.imagenet_std, 0, 0.229)),
        static_cast<float>(resolve_stat_value(config.imagenet_std, 1, 0.224)),
        static_cast<float>(resolve_stat_value(config.imagenet_std, 2, 0.225)),
    };

    const bool cache_matches = normalization_cache_.valid &&
                               normalization_cache_.device == device &&
                               normalization_cache_.dtype == dtype &&
                               normalization_cache_.mean_values == mean_values &&
                               normalization_cache_.std_values == std_values;
    if (!cache_matches) {
      auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
      normalization_cache_.valid = true;
      normalization_cache_.device = device;
      normalization_cache_.dtype = dtype;
      normalization_cache_.mean_values = mean_values;
      normalization_cache_.std_values = std_values;
      normalization_cache_.mean_tensor = torch::from_blob(const_cast<float*>(mean_values.data()), {3}, cpu_options)
                                            .clone()
                                            .to(device, dtype)
                                            .view({1, 3, 1, 1});
      normalization_cache_.std_tensor = torch::from_blob(const_cast<float*>(std_values.data()), {3}, cpu_options)
                                           .clone()
                                           .to(device, dtype)
                                           .view({1, 3, 1, 1});
    }

    return {normalization_cache_.mean_tensor, normalization_cache_.std_tensor};
  }

  void ensure_loaded(const DinoTorchRuntimeConfig& config, const c10::Device& device) {
    std::lock_guard<std::mutex> lock(torchscript_load_mutex_);
    if (torchscript_model_loaded_ || torchscript_load_failed_) {
      return;
    }

    const auto init_mode = normalize_torchscript_init_mode(config.torchscript_init_mode);
    const bool load_on_cuda = torchscript_init_moves_to_cuda(init_mode);
    const auto load_device = load_on_cuda ? device : c10::Device(torch::kCPU);
    const bool use_fp16 = use_fp16_torch_dtype(config.torch_dtype) && load_on_cuda;
    try {
      torchscript_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(config.model_script_path, load_device));
      if (use_fp16) {
        torchscript_module_->to(load_device, torch::kHalf, false);
      }
      if (torchscript_init_runs_eval(init_mode)) {
        torchscript_module_->eval();
        try {
          auto frozen_module = torch::jit::freeze(*torchscript_module_);
          auto optimized_module = torch::jit::optimize_for_inference(frozen_module);
          torchscript_module_ = std::make_unique<torch::jit::script::Module>(std::move(optimized_module));
        } catch (const std::exception&) {
        }
      }
      torchscript_model_loaded_ = true;
      torchscript_load_failed_ = false;
      torchscript_forward_ready_ = true;
      last_detail_ = std::string("loaded script=") + config.model_script_path + " mode=" + init_mode + " dtype=" + (use_fp16 ? "fp16" : "fp32");
    } catch (...) {
      torchscript_module_.reset();
      torchscript_model_loaded_ = false;
      torchscript_load_failed_ = true;
      torchscript_forward_ready_ = false;
      throw;
    }
  }

  std::mutex torchscript_load_mutex_;
  std::mutex normalization_cache_mutex_;
  NormalizationTensorCache normalization_cache_;
  std::unique_ptr<torch::jit::script::Module> torchscript_module_;
  bool torchscript_model_loaded_ = false;
  bool torchscript_load_failed_ = false;
  bool torchscript_forward_ready_ = false;
  std::string last_detail_;
};

DinoTorchRuntime::DinoTorchRuntime() : impl_(std::make_unique<Impl>()) {}
DinoTorchRuntime::~DinoTorchRuntime() = default;
DinoTorchRuntime::DinoTorchRuntime(DinoTorchRuntime&&) noexcept = default;
DinoTorchRuntime& DinoTorchRuntime::operator=(DinoTorchRuntime&&) noexcept = default;

DinoTorchRuntimeResult DinoTorchRuntime::run(const DinoTorchRuntimeConfig& config, const DinoTorchRuntimeInput& input) {
  return impl_->run(config, input);
}

DinoTorchRuntimeBatchResult DinoTorchRuntime::run_batch(const DinoTorchRuntimeConfig& config,
                                                        const DinoTorchRuntimeBatchInput& input) {
  return impl_->run_batch(config, input);
}

DinoHybridPostGpuResult DinoTorchRuntime::run_hybrid_post_gpu(const DinoHybridPostGpuInput& input) {
  return impl_->run_hybrid_post_gpu(input);
}

void DinoTorchRuntime::warmup(const DinoTorchRuntimeConfig& config,
                              int src_rows,
                              int src_cols,
                              int dst_rows,
                              int dst_cols,
                              int patch_size,
                              int batch_size,
                              cudaStream_t cuda_stream) {
  impl_->warmup(config, src_rows, src_cols, dst_rows, dst_cols, patch_size, batch_size, cuda_stream);
}

}  // namespace holoscan::ops
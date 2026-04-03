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

int64_t quantile_index(int64_t size, double q) {
  if (size <= 1) {
    return 0;
  }
  const double clamped = std::clamp(q, 0.0, 1.0);
  return static_cast<int64_t>(std::llround(clamped * static_cast<double>(size - 1)));
}

torch::Tensor select_quantile_along_dim(const torch::Tensor& input, double q, int64_t dim) {
  auto sorted = std::get<0>(torch::sort(input, dim));
  return sorted.select(dim, quantile_index(input.size(dim), q));
}

double scalar_quantile(const torch::Tensor& input, double q) {
  auto flat = input.reshape({-1});
  auto sorted = std::get<0>(torch::sort(flat, 0));
  return sorted[quantile_index(sorted.size(0), q)].item<double>();
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

torch::Tensor normalize_map01(const torch::Tensor& input, double low_q, double high_q) {
  const double lo = scalar_quantile(input, low_q);
  const double hi = scalar_quantile(input, high_q);
  const double scale = std::max(hi - lo, 1e-6);
  return torch::clamp((input - lo) / scale, 0.0, 1.0);
}

torch::Tensor derive_dino_score_map(torch::Tensor model_output,
                                    int aligned_rows,
                                    int aligned_cols,
                                    int patch_size,
                                    int dst_rows,
                                    int dst_cols) {
  if (model_output.dim() == 4 && model_output.size(0) == 1) {
    model_output = model_output.squeeze(0);
  }

  torch::Tensor base_map;
  if (model_output.dim() == 3) {
    if (model_output.size(0) > 1 && model_output.size(1) > 1 && model_output.size(2) > 1) {
      base_map = torch::sqrt(torch::mean(model_output * model_output, 0) + 1e-6);
    } else if (model_output.size(0) == 1) {
      base_map = model_output.squeeze(0);
    }
  } else if (model_output.dim() == 2) {
    const int patch_rows = std::max(1, aligned_rows / std::max(1, patch_size));
    const int patch_cols = std::max(1, aligned_cols / std::max(1, patch_size));
    const int64_t patch_count = static_cast<int64_t>(patch_rows) * static_cast<int64_t>(patch_cols);
    if (model_output.size(0) == patch_count) {
      base_map = torch::sqrt(torch::mean(model_output * model_output, 1) + 1e-6).view({patch_rows, patch_cols});
    } else if (model_output.size(1) == patch_count) {
      auto transposed = model_output.transpose(0, 1);
      base_map = torch::sqrt(torch::mean(transposed * transposed, 1) + 1e-6).view({patch_rows, patch_cols});
    } else {
      base_map = model_output;
    }
  } else if (model_output.dim() == 1) {
    base_map = model_output.view({1, -1});
  }

  if (!base_map.defined()) {
    throw std::runtime_error("TorchScript forward returned an unsupported tensor shape for DINO scoring");
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
    result.aligned_rows = std::max(1, (input.dst_rows / std::max(1, input.patch_size)) * std::max(1, input.patch_size));
    result.aligned_cols = std::max(1, (input.dst_cols / std::max(1, input.patch_size)) * std::max(1, input.patch_size));
    result.final_threshold = config.pipeline_final_threshold;
    std::string failure_stage = "input_validation";
    std::string failure_detail;

    try {
      if (!input.power_db || input.power_db->size() != static_cast<size_t>(input.src_rows) * static_cast<size_t>(input.src_cols)) {
        throw std::runtime_error("Invalid host power_db input buffer");
      }

      failure_detail = std::string("channel=") + std::to_string(input.channel_number) +
                       " frame=" + std::to_string(input.frame_number) +
                       " src=" + std::to_string(input.src_rows) + "x" + std::to_string(input.src_cols) +
                       " dst=" + std::to_string(input.dst_rows) + "x" + std::to_string(input.dst_cols) +
                       " patch=" + std::to_string(input.patch_size);

      const auto init_mode = normalize_torchscript_init_mode(config.torchscript_init_mode);
      const bool use_cuda_torch = (config.inference_backend != "torchscript") || torchscript_init_moves_to_cuda(init_mode);
      c10::Device compute_device = use_cuda_torch ? c10::Device(torch::kCUDA, 0) : c10::Device(torch::kCPU);
      std::unique_ptr<c10::cuda::CUDAGuard> device_guard;
      if (use_cuda_torch) {
        device_guard = std::make_unique<c10::cuda::CUDAGuard>(compute_device);
      }

      failure_stage = "power_db_tensor_create";
      auto cpu_float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
      auto power_db = torch::from_blob(const_cast<float*>(input.power_db->data()),
                                       {static_cast<int64_t>(input.src_rows), static_cast<int64_t>(input.src_cols)},
                                       cpu_float_options)
                          .clone();
      if (use_cuda_torch) {
        failure_stage = "power_db_to_device";
        power_db = power_db.to(compute_device);
      }

      auto backend = config.inference_backend;
      if (backend == "torchscript") {
        failure_stage = "torchscript_load";
        try {
          ensure_loaded(config, compute_device);
        } catch (const std::exception& error) {
          last_detail_ = error.what();
          backend = "pytorch_placeholder";
        }
        if (torchscript_load_failed_ || !torchscript_forward_ready_) {
          backend = "pytorch_placeholder";
        }
      }

      torch::Tensor corrected_db;
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

      failure_stage = "crop_align";
      result.timing.crop_align_ms = measure_ms([&] {
        result.freq_bin_hz = input.resolution_hz > 0.0 ? input.resolution_hz : (input.span_hz > 0.0 ? input.span_hz / static_cast<double>(input.src_cols) : 0.0);
        if (config.ignore_sideband_hz > 0.0 && result.freq_bin_hz > 0.0) {
          const int requested_bins = static_cast<int>(std::ceil(config.ignore_sideband_hz / result.freq_bin_hz));
          const int max_ignore_bins = std::max(0, (input.src_cols - input.patch_size) / 2);
          result.ignore_bins_per_side = std::min(requested_bins, max_ignore_bins);
        }
        if (result.ignore_bins_per_side > 0 && (2 * result.ignore_bins_per_side) < input.src_cols) {
          corrected_db = corrected_db.index({torch::indexing::Slice(), torch::indexing::Slice(result.ignore_bins_per_side, input.src_cols - result.ignore_bins_per_side)});
        }
      });

      torch::Tensor resized_db;
      failure_stage = "resize";
      result.timing.resize_ms = measure_ms([&] {
        resized_db = torch::nn::functional::interpolate(
                         corrected_db.unsqueeze(0).unsqueeze(0),
                         torch::nn::functional::InterpolateFuncOptions()
                             .size(std::vector<int64_t>{static_cast<int64_t>(result.aligned_rows), static_cast<int64_t>(result.aligned_cols)})
                             .mode(torch::kBilinear)
                             .align_corners(false))
                         .squeeze(0)
                         .squeeze(0)
                         .contiguous();
      });

      torch::Tensor model_input;
      failure_stage = "model_prep";
      result.timing.model_prep_ms = measure_ms([&] {
        auto grayscale = normalize_map01(resized_db, 0.01, 0.99).to(torch::kFloat32);
        auto rgb = torch::stack({grayscale, grayscale, grayscale}, 0).unsqueeze(0).contiguous();
        auto mean = torch::tensor(
                        std::vector<float>{static_cast<float>(resolve_stat_value(config.imagenet_mean, 0, 0.485)),
                                           static_cast<float>(resolve_stat_value(config.imagenet_mean, 1, 0.456)),
                                           static_cast<float>(resolve_stat_value(config.imagenet_mean, 2, 0.406))},
                        torch::TensorOptions().dtype(torch::kFloat32).device(compute_device))
                        .view({1, 3, 1, 1});
        auto std = torch::tensor(
                       std::vector<float>{static_cast<float>(resolve_stat_value(config.imagenet_std, 0, 0.229)),
                                          static_cast<float>(resolve_stat_value(config.imagenet_std, 1, 0.224)),
                                          static_cast<float>(resolve_stat_value(config.imagenet_std, 2, 0.225))},
                       torch::TensorOptions().dtype(torch::kFloat32).device(compute_device))
                       .view({1, 3, 1, 1});
        model_input = (rgb - mean) / std;
      });

      torch::Tensor dino_score;
      if (backend == "torchscript") {
        torch::Tensor model_output;
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
          dino_score = derive_dino_score_map(model_output, result.aligned_rows, result.aligned_cols, input.patch_size, input.dst_rows, input.dst_cols);
        });
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

      torch::Tensor final_mask;
      failure_stage = "fusion";
      result.timing.fusion_ms = measure_ms([&] {
        result.dino_threshold = scalar_quantile(dino_score, std::clamp(config.dino_group_score_q, 0.0, 1.0));
        auto dino_mask = dino_score.ge(result.dino_threshold).to(torch::kFloat32);
        auto power_mask = power_score.ge(result.power_threshold).to(torch::kFloat32);
        auto agreement_map = 0.5f * (dino_mask + power_mask);
        auto rescue = torch::clamp((power_score - config.pipeline_power_rescue_floor) /
                                       std::max(1.0 - config.pipeline_power_rescue_floor, 1e-6),
                                   0.0,
                                   1.0) * config.pipeline_power_rescue_gain;
        auto gap_weight = torch::clamp(dino_score - config.pipeline_gap_floor, 0.0, 1.0);
        auto final_score = torch::maximum(agreement_map, rescue * gap_weight);
        final_mask = final_score.ge(config.pipeline_final_threshold).to(torch::kFloat32).contiguous();
      });

      failure_stage = "final_mask_to_cpu";
      auto final_mask_cpu = final_mask.device().is_cuda() ? final_mask.to(torch::kCPU) : final_mask;
      result.final_mask.resize(static_cast<size_t>(input.dst_rows) * static_cast<size_t>(input.dst_cols));
      std::memcpy(result.final_mask.data(), final_mask_cpu.data_ptr<float>(), result.final_mask.size() * sizeof(float));
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

 private:
  void ensure_loaded(const DinoTorchRuntimeConfig& config, const c10::Device& device) {
    std::lock_guard<std::mutex> lock(torchscript_load_mutex_);
    if (torchscript_model_loaded_ || torchscript_load_failed_) {
      return;
    }

    const auto init_mode = normalize_torchscript_init_mode(config.torchscript_init_mode);
    const bool load_on_cuda = torchscript_init_moves_to_cuda(init_mode);
    const auto load_device = load_on_cuda ? device : c10::Device(torch::kCPU);
    try {
      torchscript_module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(config.model_script_path, load_device));
      if (torchscript_init_runs_eval(init_mode)) {
        torchscript_module_->eval();
      }
      torchscript_model_loaded_ = true;
      torchscript_load_failed_ = false;
      torchscript_forward_ready_ = true;
      last_detail_ = std::string("loaded script=") + config.model_script_path + " mode=" + init_mode;
    } catch (...) {
      torchscript_module_.reset();
      torchscript_model_loaded_ = false;
      torchscript_load_failed_ = true;
      torchscript_forward_ready_ = false;
      throw;
    }
  }

  std::mutex torchscript_load_mutex_;
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

}  // namespace holoscan::ops
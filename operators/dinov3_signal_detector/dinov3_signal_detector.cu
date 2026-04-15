// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "dinov3_signal_detector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

enum TimingStageIndex : size_t {
  kInputStage = 0,
  kPowerDbStage,
  kFrontendStage,
  kCoherenceStage,
  kTorchRuntimeStage,
  kHybridStage,
  kMaskSaveStage,
  kTotalStage,
};

constexpr std::array<const char*, holoscan::ops::DinoV3SignalDetector::kTimingStageCount>
    kTimingStageNames = {
        "input_ms",
        "power_db_ms",
        "frontend_ms",
        "coherence_ms",
        "torch_runtime_ms",
        "hybrid_post_ms",
        "mask_save_ms",
        "total_ms",
    };

struct CanonicalTensorView {
  int rows = 0;
  int cols = 0;
  bool transposed = false;
};

struct ComponentLabelling {
  std::vector<int> labels;
  std::vector<int> sizes;
  int count = 0;
};

struct HybridPostprocessResult {
  std::vector<uint8_t> mask;
  float seed_freq_threshold = 1.0f;
  float seed_res_threshold = 1.0f;
  float grow_freq_threshold = 1.0f;
  float grow_res_threshold = 1.0f;
  float combined_threshold = 1.0f;
  float final_fraction = 0.0f;
  float connected_fraction = 0.0f;
  int component_count = 0;
};

float quantile_from_values(std::vector<float> values, double q, float fallback);
std::vector<uint8_t> keep_large_components(const std::vector<uint8_t>& mask,
                                           int rows,
                                           int cols,
                                           int min_size,
                                           int* component_count = nullptr);
std::vector<uint8_t> binary_closing_rect(const std::vector<uint8_t>& mask,
                                         int rows,
                                         int cols,
                                         int kernel_rows,
                                         int kernel_cols);
std::vector<uint8_t> binary_fill_holes(const std::vector<uint8_t>& mask, int rows, int cols);
std::vector<uint8_t> binary_propagation(const std::vector<uint8_t>& seed,
                                        const std::vector<uint8_t>& mask,
                                        int rows,
                                        int cols);
float mean_mask_value(const std::vector<uint8_t>& mask);
float connected_fraction(const std::vector<uint8_t>& mask, const std::vector<uint8_t>& valid_mask);

HybridPostprocessResult run_fast_dino_postprocess(const std::vector<float>& dino_score,
                                                  const std::vector<uint8_t>& valid_mask,
                                                  int rows,
                                                  int cols,
                                                  float score_threshold,
                                                  int min_component_size) {
  HybridPostprocessResult result;
  result.mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  if (dino_score.size() != result.mask.size() || valid_mask.size() != result.mask.size()) {
    return result;
  }

  const auto active_scores = [&]() {
    std::vector<float> values;
    values.reserve(dino_score.size());
    for (size_t index = 0; index < dino_score.size(); ++index) {
      if (valid_mask[index] && std::isfinite(dino_score[index])) {
        values.push_back(dino_score[index]);
      }
    }
    return values;
  }();

  const float fallback_threshold = quantile_from_values(active_scores, 0.85, 1.0f);
  const float base_threshold = std::isfinite(score_threshold) && score_threshold > 0.0f
                                   ? score_threshold
                                   : fallback_threshold;
  const float grow_threshold = std::max(0.0f, base_threshold * 0.92f);

  std::vector<uint8_t> seed_mask(result.mask.size(), 0);
  std::vector<uint8_t> grow_mask(result.mask.size(), 0);
  for (size_t index = 0; index < result.mask.size(); ++index) {
    if (!valid_mask[index]) {
      continue;
    }
    seed_mask[index] = dino_score[index] >= base_threshold ? 1 : 0;
    grow_mask[index] = dino_score[index] >= grow_threshold ? 1 : 0;
  }

  seed_mask = keep_large_components(seed_mask, rows, cols, std::max(8, min_component_size / 2));
  grow_mask = binary_closing_rect(grow_mask, rows, cols, 5, 3);
  grow_mask = binary_fill_holes(grow_mask, rows, cols);
  auto final_mask = binary_propagation(seed_mask, grow_mask, rows, cols);
  final_mask = binary_closing_rect(final_mask, rows, cols, 5, 3);
  final_mask = binary_fill_holes(final_mask, rows, cols);
  final_mask = keep_large_components(final_mask, rows, cols, std::max(12, min_component_size), &result.component_count);
  for (size_t index = 0; index < final_mask.size(); ++index) {
    final_mask[index] = (final_mask[index] && valid_mask[index]) ? 1 : 0;
  }
  final_mask = keep_large_components(final_mask, rows, cols, std::max(12, min_component_size), &result.component_count);

  result.seed_freq_threshold = base_threshold;
  result.seed_res_threshold = base_threshold;
  result.grow_freq_threshold = grow_threshold;
  result.grow_res_threshold = grow_threshold;
  result.combined_threshold = grow_threshold;
  result.final_fraction = mean_mask_value(final_mask);
  result.connected_fraction = connected_fraction(final_mask, valid_mask);
  result.mask = std::move(final_mask);
  return result;
}

CanonicalTensorView canonical_tensor_view(int input_rows, int input_cols) {
  CanonicalTensorView view;
  view.transposed = input_rows < input_cols;
  view.rows = view.transposed ? input_cols : input_rows;
  view.cols = view.transposed ? input_rows : input_cols;
  return view;
}

template <typename T>
__host__ __device__ inline T clamp_value(T value, T low, T high) {
  return value < low ? low : (value > high ? high : value);
}

__host__ __device__ inline size_t flat_index(int cols, int row, int col) {
  return static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
}

std::string make_mask_output_path(const std::string& output_dir,
                                  uint16_t channel,
                                  uint64_t frame_number,
                                  int rows,
                                  int cols) {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  std::ostringstream oss;
  oss << output_dir
      << "/dino_mask_ch" << channel
      << "_f" << frame_number
      << "_" << ms
      << "_" << rows << "x" << cols
      << ".pgm";
  return oss.str();
}

bool write_pgm(const std::string& path, const std::vector<uint8_t>& image, int width, int height) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }

  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
  return out.good();
}

__global__ void dino_transpose_kernel(const holoscan::ops::dino_complex* input,
                                      int input_rows,
                                      int input_cols,
                                      holoscan::ops::dino_complex* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = input_rows * input_cols;
  if (index >= total) {
    return;
  }
  const int row = index / input_cols;
  const int col = index % input_cols;
  output[flat_index(input_rows, col, row)] = input[index];
}

__global__ void dino_power_db_kernel(const holoscan::ops::dino_complex* input,
                                     int rows,
                                     int cols,
                                     float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }

  const auto value = input[index];
  const float re = value.real();
  const float im = value.imag();
  const float power = re * re + im * im + 1e-12f;
  output[index] = 10.0f * log10f(power);
}

__global__ void dino_row_mean_kernel(const float* input, int rows, int cols, float* row_mean) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float partial[256];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    sum += input[flat_index(cols, row, col)];
  }
  partial[tid] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    row_mean[row] = partial[0] / static_cast<float>(max(cols, 1));
  }
}

__global__ void dino_gaussian_smooth_rows_kernel(const float* input,
                                                 int rows,
                                                 int radius,
                                                 float sigma,
                                                 float* output) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  float sum = 0.0f;
  float weight_sum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const int src_row = max(0, min(rows - 1, row + offset));
    const float weight = expf(-(static_cast<float>(offset * offset)) / (2.0f * sigma * sigma));
    sum += input[src_row] * weight;
    weight_sum += weight;
  }
  output[row] = weight_sum > 0.0f ? sum / weight_sum : input[row];
}

__global__ void dino_frontend_reference_kernel(const float* row_smooth,
                                               int rows,
                                               float quantile,
                                               float* reference_level) {
  __shared__ float partial_sum[256];
  __shared__ float partial_max[256];

  const int tid = threadIdx.x;
  float thread_sum = 0.0f;
  float thread_max = -1.0e30f;
  for (int row = tid; row < rows; row += blockDim.x) {
    const float value = row_smooth[row];
    thread_sum += value;
    thread_max = fmaxf(thread_max, value);
  }

  partial_sum[tid] = thread_sum;
  partial_max[tid] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_sum[tid] += partial_sum[tid + stride];
      partial_max[tid] = fmaxf(partial_max[tid], partial_max[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float mean_value = partial_sum[0] / static_cast<float>(max(rows, 1));
    const float blend = fminf(fmaxf((quantile - 0.5f) / 0.5f, 0.0f), 1.0f);
    reference_level[0] = mean_value + blend * (partial_max[0] - mean_value);
  }
}

__global__ void dino_frontend_correction_kernel(const float* input,
                                                int rows,
                                                int cols,
                                                const float* row_smooth,
                                                const float* reference_level,
                                                float max_boost_db,
                                                float* corrected) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }

  const int row = index / cols;
  const float boost = fminf(fmaxf(reference_level[0] - row_smooth[row], 0.0f), max_boost_db);
  corrected[index] = input[index] + boost;
}

__global__ void dino_box_mean_cols_kernel(const float* input,
                                          int rows,
                                          int cols,
                                          int radius_cols,
                                          float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }

  const int row = index / cols;
  const int col = index % cols;
  const int col_start = max(0, col - radius_cols);
  const int col_stop = min(cols - 1, col + radius_cols);

  float sum = 0.0f;
  int count = 0;
  for (int src_col = col_start; src_col <= col_stop; ++src_col) {
    sum += input[flat_index(cols, row, src_col)];
    ++count;
  }
  output[index] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void dino_box_mean_rows_kernel(const float* input,
                                          int rows,
                                          int cols,
                                          int radius_rows,
                                          float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }

  const int row = index / cols;
  const int col = index % cols;
  const int row_start = max(0, row - radius_rows);
  const int row_stop = min(rows - 1, row + radius_rows);

  float sum = 0.0f;
  int count = 0;
  for (int src_row = row_start; src_row <= row_stop; ++src_row) {
    sum += input[flat_index(cols, src_row, col)];
    ++count;
  }
  output[index] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void dino_coherence_gate_kernel(const float* time_mean,
                                           const float* freq_mean,
                                           int rows,
                                           int cols,
                                           int ignore_bins_per_side,
                                           float coherence_floor_db,
                                           float coherence_span_db,
                                           float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }

  const int row = index / cols;
  if (row < ignore_bins_per_side || row >= (rows - ignore_bins_per_side)) {
    output[index] = 0.0f;
    return;
  }

  const float coherence_db = time_mean[index] - freq_mean[index];
  output[index] = fminf(fmaxf((coherence_db - coherence_floor_db) / fmaxf(coherence_span_db, 1e-6f),
                              0.0f),
                        1.0f);
}

__global__ void dino_resize_bilinear_kernel(const float* input,
                                            int src_rows,
                                            int src_cols,
                                            float* output,
                                            int dst_rows,
                                            int dst_cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = dst_rows * dst_cols;
  if (index >= total) {
    return;
  }

  const int dst_row = index / dst_cols;
  const int dst_col = index % dst_cols;
  const float src_row_f = dst_rows > 1
                              ? (static_cast<float>(dst_row) * static_cast<float>(src_rows - 1)) /
                                    static_cast<float>(dst_rows - 1)
                              : 0.0f;
  const float src_col_f = dst_cols > 1
                              ? (static_cast<float>(dst_col) * static_cast<float>(src_cols - 1)) /
                                    static_cast<float>(dst_cols - 1)
                              : 0.0f;

  const int src_row0 = clamp_value(static_cast<int>(floorf(src_row_f)), 0, src_rows - 1);
  const int src_col0 = clamp_value(static_cast<int>(floorf(src_col_f)), 0, src_cols - 1);
  const int src_row1 = min(src_row0 + 1, src_rows - 1);
  const int src_col1 = min(src_col0 + 1, src_cols - 1);
  const float row_t = src_row_f - static_cast<float>(src_row0);
  const float col_t = src_col_f - static_cast<float>(src_col0);

  const float v00 = input[flat_index(src_cols, src_row0, src_col0)];
  const float v01 = input[flat_index(src_cols, src_row0, src_col1)];
  const float v10 = input[flat_index(src_cols, src_row1, src_col0)];
  const float v11 = input[flat_index(src_cols, src_row1, src_col1)];
  const float top = v00 + (v01 - v00) * col_t;
  const float bottom = v10 + (v11 - v10) * col_t;
  output[flat_index(dst_cols, dst_row, dst_col)] = top + (bottom - top) * row_t;
}

float quantile_from_values(std::vector<float> values, double q, float fallback = 1.0f) {
  values.erase(std::remove_if(values.begin(), values.end(), [](float value) {
                 return !std::isfinite(value);
               }),
               values.end());
  if (values.empty()) {
    return fallback;
  }
  q = clamp_value(q, 0.0, 1.0);
  const size_t index = static_cast<size_t>(std::llround(q * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
  return values[index];
}

std::vector<float> resize_bilinear(const std::vector<float>& input,
                                   int src_rows,
                                   int src_cols,
                                   int dst_rows,
                                   int dst_cols) {
  if (src_rows <= 0 || src_cols <= 0 || dst_rows <= 0 || dst_cols <= 0) {
    return {};
  }
  if (src_rows == dst_rows && src_cols == dst_cols) {
    return input;
  }

  std::vector<float> output(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), 0.0f);
  const float row_scale = dst_rows > 1 ? static_cast<float>(src_rows - 1) / static_cast<float>(dst_rows - 1) : 0.0f;
  const float col_scale = dst_cols > 1 ? static_cast<float>(src_cols - 1) / static_cast<float>(dst_cols - 1) : 0.0f;

  for (int dst_row = 0; dst_row < dst_rows; ++dst_row) {
    const float src_row_f = row_scale * static_cast<float>(dst_row);
    const int src_row0 = clamp_value(static_cast<int>(std::floor(src_row_f)), 0, src_rows - 1);
    const int src_row1 = clamp_value(src_row0 + 1, 0, src_rows - 1);
    const float row_t = src_row_f - static_cast<float>(src_row0);
    for (int dst_col = 0; dst_col < dst_cols; ++dst_col) {
      const float src_col_f = col_scale * static_cast<float>(dst_col);
      const int src_col0 = clamp_value(static_cast<int>(std::floor(src_col_f)), 0, src_cols - 1);
      const int src_col1 = clamp_value(src_col0 + 1, 0, src_cols - 1);
      const float col_t = src_col_f - static_cast<float>(src_col0);

      const float v00 = input[flat_index(src_cols, src_row0, src_col0)];
      const float v01 = input[flat_index(src_cols, src_row0, src_col1)];
      const float v10 = input[flat_index(src_cols, src_row1, src_col0)];
      const float v11 = input[flat_index(src_cols, src_row1, src_col1)];
      const float top = (1.0f - col_t) * v00 + col_t * v01;
      const float bottom = (1.0f - col_t) * v10 + col_t * v11;
      output[flat_index(dst_cols, dst_row, dst_col)] = (1.0f - row_t) * top + row_t * bottom;
    }
  }
  return output;
}

std::vector<uint8_t> resize_valid_row_mask(int src_rows, int dst_rows, int dst_cols, int ignore_bins_per_side) {
  std::vector<uint8_t> mask(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), 0);
  if (src_rows <= 0 || dst_rows <= 0 || dst_cols <= 0) {
    return mask;
  }

  for (int dst_row = 0; dst_row < dst_rows; ++dst_row) {
    const int src_row = std::min(src_rows - 1,
                                 static_cast<int>((static_cast<int64_t>(dst_row) * static_cast<int64_t>(src_rows)) /
                                                  static_cast<int64_t>(std::max(dst_rows, 1))));
    const bool valid = src_row >= ignore_bins_per_side && src_row < (src_rows - ignore_bins_per_side);
    if (!valid) {
      continue;
    }
    const size_t row_offset = static_cast<size_t>(dst_row) * static_cast<size_t>(dst_cols);
    std::fill(mask.begin() + static_cast<std::ptrdiff_t>(row_offset),
              mask.begin() + static_cast<std::ptrdiff_t>(row_offset + static_cast<size_t>(dst_cols)),
              static_cast<uint8_t>(1));
  }
  return mask;
}

std::vector<float> normalize01_masked_minmax(const std::vector<float>& input,
                                             const std::vector<uint8_t>& mask) {
  std::vector<float> output(input.size(), 0.0f);
  float low = std::numeric_limits<float>::infinity();
  float high = -std::numeric_limits<float>::infinity();
  for (size_t index = 0; index < input.size() && index < mask.size(); ++index) {
    if (!mask[index] || !std::isfinite(input[index])) {
      continue;
    }
    low = std::min(low, input[index]);
    high = std::max(high, input[index]);
  }
  if (!std::isfinite(low) || !std::isfinite(high) || high <= low + 1e-12f) {
    return output;
  }

  const float scale = high - low;
  for (size_t index = 0; index < input.size() && index < mask.size(); ++index) {
    if (!mask[index]) {
      continue;
    }
    output[index] = clamp_value((input[index] - low) / scale, 0.0f, 1.0f);
  }
  return output;
}

std::vector<float> gaussian_kernel(double sigma) {
  if (sigma <= 0.0) {
    return {1.0f};
  }
  const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1), 0.0f);
  double sum = 0.0;
  for (int offset = -radius; offset <= radius; ++offset) {
    const double weight = std::exp(-(static_cast<double>(offset * offset)) / (2.0 * sigma * sigma));
    kernel[static_cast<size_t>(offset + radius)] = static_cast<float>(weight);
    sum += weight;
  }
  for (float& value : kernel) {
    value = static_cast<float>(value / sum);
  }
  return kernel;
}

std::vector<float> convolve_axis(const std::vector<float>& input,
                                 int rows,
                                 int cols,
                                 const std::vector<float>& kernel,
                                 bool along_rows) {
  const int radius = static_cast<int>(kernel.size() / 2);
  std::vector<float> output(input.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      float sum = 0.0f;
      for (int offset = -radius; offset <= radius; ++offset) {
        const int src_row = along_rows ? clamp_value(row + offset, 0, rows - 1) : row;
        const int src_col = along_rows ? col : clamp_value(col + offset, 0, cols - 1);
        sum += kernel[static_cast<size_t>(offset + radius)] * input[flat_index(cols, src_row, src_col)];
      }
      output[flat_index(cols, row, col)] = sum;
    }
  }
  return output;
}

ComponentLabelling label_components(const std::vector<uint8_t>& mask, int rows, int cols) {
  ComponentLabelling result;
  result.labels.assign(mask.size(), 0);
  int next_label = 0;
  constexpr std::array<int, 8> d_row = {-1, -1, -1, 0, 0, 1, 1, 1};
  constexpr std::array<int, 8> d_col = {-1, 0, 1, -1, 1, -1, 0, 1};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t index = flat_index(cols, row, col);
      if (!mask[index] || result.labels[index] != 0) {
        continue;
      }

      ++next_label;
      int component_size = 0;
      std::queue<std::pair<int, int>> pending;
      pending.push({row, col});
      result.labels[index] = next_label;
      while (!pending.empty()) {
        const auto [cur_row, cur_col] = pending.front();
        pending.pop();
        ++component_size;
        for (size_t neighbor_index = 0; neighbor_index < d_row.size(); ++neighbor_index) {
          const int next_row = cur_row + d_row[neighbor_index];
          const int next_col = cur_col + d_col[neighbor_index];
          if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
            continue;
          }
          const size_t flat = flat_index(cols, next_row, next_col);
          if (!mask[flat] || result.labels[flat] != 0) {
            continue;
          }
          result.labels[flat] = next_label;
          pending.push({next_row, next_col});
        }
      }
      result.sizes.push_back(component_size);
    }
  }

  result.count = next_label;
  return result;
}

std::vector<uint8_t> keep_large_components(const std::vector<uint8_t>& mask,
                                           int rows,
                                           int cols,
                                           int min_size,
                                           int* kept_component_count) {
  auto labelled = label_components(mask, rows, cols);
  std::vector<uint8_t> output(mask.size(), 0);
  for (size_t index = 0; index < labelled.labels.size(); ++index) {
    const int label = labelled.labels[index];
    if (label <= 0) {
      continue;
    }
    if (labelled.sizes[static_cast<size_t>(label - 1)] >= std::max(1, min_size)) {
      output[index] = 1;
    }
  }
  if (kept_component_count != nullptr) {
    auto kept = label_components(output, rows, cols);
    *kept_component_count = kept.count;
  }
  return output;
}

std::vector<uint8_t> binary_dilate_rect(const std::vector<uint8_t>& mask,
                                        int rows,
                                        int cols,
                                        int kernel_rows,
                                        int kernel_cols) {
  const int row_radius = std::max(0, kernel_rows / 2);
  const int col_radius = std::max(0, kernel_cols / 2);
  std::vector<uint8_t> output(mask.size(), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      bool active = false;
      for (int d_row = -row_radius; d_row <= row_radius && !active; ++d_row) {
        const int src_row = clamp_value(row + d_row, 0, rows - 1);
        for (int d_col = -col_radius; d_col <= col_radius; ++d_col) {
          const int src_col = clamp_value(col + d_col, 0, cols - 1);
          if (mask[flat_index(cols, src_row, src_col)]) {
            active = true;
            break;
          }
        }
      }
      output[flat_index(cols, row, col)] = active ? 1 : 0;
    }
  }
  return output;
}

std::vector<uint8_t> binary_erode_rect(const std::vector<uint8_t>& mask,
                                       int rows,
                                       int cols,
                                       int kernel_rows,
                                       int kernel_cols) {
  const int row_radius = std::max(0, kernel_rows / 2);
  const int col_radius = std::max(0, kernel_cols / 2);
  std::vector<uint8_t> output(mask.size(), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      bool active = true;
      for (int d_row = -row_radius; d_row <= row_radius && active; ++d_row) {
        const int src_row = clamp_value(row + d_row, 0, rows - 1);
        for (int d_col = -col_radius; d_col <= col_radius; ++d_col) {
          const int src_col = clamp_value(col + d_col, 0, cols - 1);
          if (!mask[flat_index(cols, src_row, src_col)]) {
            active = false;
            break;
          }
        }
      }
      output[flat_index(cols, row, col)] = active ? 1 : 0;
    }
  }
  return output;
}

std::vector<uint8_t> binary_closing_rect(const std::vector<uint8_t>& mask,
                                         int rows,
                                         int cols,
                                         int kernel_rows,
                                         int kernel_cols) {
  return binary_erode_rect(binary_dilate_rect(mask, rows, cols, kernel_rows, kernel_cols),
                           rows,
                           cols,
                           kernel_rows,
                           kernel_cols);
}

std::vector<uint8_t> binary_fill_holes(const std::vector<uint8_t>& mask, int rows, int cols) {
  std::vector<uint8_t> visited(mask.size(), 0);
  std::queue<std::pair<int, int>> pending;
  auto maybe_enqueue = [&](int row, int col) {
    const size_t index = flat_index(cols, row, col);
    if (mask[index] || visited[index]) {
      return;
    }
    visited[index] = 1;
    pending.push({row, col});
  };

  for (int row = 0; row < rows; ++row) {
    maybe_enqueue(row, 0);
    maybe_enqueue(row, cols - 1);
  }
  for (int col = 0; col < cols; ++col) {
    maybe_enqueue(0, col);
    maybe_enqueue(rows - 1, col);
  }

  constexpr std::array<int, 4> d_row = {-1, 0, 0, 1};
  constexpr std::array<int, 4> d_col = {0, -1, 1, 0};
  while (!pending.empty()) {
    const auto [row, col] = pending.front();
    pending.pop();
    for (size_t index = 0; index < d_row.size(); ++index) {
      const int next_row = row + d_row[index];
      const int next_col = col + d_col[index];
      if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
        continue;
      }
      maybe_enqueue(next_row, next_col);
    }
  }

  std::vector<uint8_t> output = mask;
  for (size_t index = 0; index < output.size(); ++index) {
    if (!output[index] && !visited[index]) {
      output[index] = 1;
    }
  }
  return output;
}

std::vector<uint8_t> binary_propagation(const std::vector<uint8_t>& seed,
                                        const std::vector<uint8_t>& grow_mask,
                                        int rows,
                                        int cols) {
  std::vector<uint8_t> output(seed.size(), 0);
  std::queue<std::pair<int, int>> pending;
  constexpr std::array<int, 8> d_row = {-1, -1, -1, 0, 0, 1, 1, 1};
  constexpr std::array<int, 8> d_col = {-1, 0, 1, -1, 1, -1, 0, 1};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t index = flat_index(cols, row, col);
      if (!seed[index] || !grow_mask[index]) {
        continue;
      }
      output[index] = 1;
      pending.push({row, col});
    }
  }

  while (!pending.empty()) {
    const auto [row, col] = pending.front();
    pending.pop();
    for (size_t index = 0; index < d_row.size(); ++index) {
      const int next_row = row + d_row[index];
      const int next_col = col + d_col[index];
      if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
        continue;
      }
      const size_t next_flat = flat_index(cols, next_row, next_col);
      if (!grow_mask[next_flat] || output[next_flat]) {
        continue;
      }
      output[next_flat] = 1;
      pending.push({next_row, next_col});
    }
  }
  return output;
}

float mean_mask_value(const std::vector<uint8_t>& mask) {
  if (mask.empty()) {
    return 0.0f;
  }
  size_t active = 0;
  for (uint8_t value : mask) {
    active += value ? 1U : 0U;
  }
  return static_cast<float>(active) / static_cast<float>(mask.size());
}

float connected_fraction(const std::vector<uint8_t>& mask, const std::vector<uint8_t>& valid_mask) {
  size_t active = 0;
  size_t valid = 0;
  for (size_t index = 0; index < mask.size() && index < valid_mask.size(); ++index) {
    if (!valid_mask[index]) {
      continue;
    }
    ++valid;
    active += mask[index] ? 1U : 0U;
  }
  if (valid == 0) {
    return 0.0f;
  }
  return static_cast<float>(active) / static_cast<float>(valid);
}

HybridPostprocessResult run_residual_veto_hybrid(const std::vector<float>& dino_score,
                                                 const std::vector<float>& coherence_gate,
                                                 const std::vector<uint8_t>& valid_mask,
                                                 int rows,
                                                 int cols) {
  HybridPostprocessResult result;
  result.mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  if (dino_score.size() != result.mask.size() ||
      coherence_gate.size() != result.mask.size() ||
      valid_mask.size() != result.mask.size()) {
    return result;
  }

  std::vector<float> base_map(result.mask.size(), 0.0f);
  for (size_t index = 0; index < base_map.size(); ++index) {
    base_map[index] = dino_score[index] * coherence_gate[index];
  }

  const auto base_norm = normalize01_masked_minmax(base_map, valid_mask);
  const auto envelope_map = normalize01_masked_minmax(gaussian_blur(base_norm, rows, cols, 6.0, 1.4), valid_mask);

  auto base_blur = gaussian_blur(base_norm, rows, cols, 4.0, 1.0);
  std::vector<float> residual_abs(base_norm.size(), 0.0f);
  for (size_t index = 0; index < residual_abs.size(); ++index) {
    residual_abs[index] = std::fabs(base_norm[index] - base_blur[index]);
  }
  const auto residual_penalty =
      normalize01_masked_minmax(gaussian_blur(residual_abs, rows, cols, 2.0, 0.8), valid_mask);

  auto curvature = gaussian_second_derivative_rows(base_norm, rows, cols, 0.8);
  for (float& value : curvature) {
    value = std::fabs(value);
  }
  const auto freq_curvature_penalty = normalize01_masked_minmax(curvature, valid_mask);

  std::vector<float> keep_freq(base_norm.size(), 0.0f);
  std::vector<float> keep_res(base_norm.size(), 0.0f);
  for (size_t index = 0; index < keep_freq.size(); ++index) {
    keep_freq[index] = envelope_map[index] - 0.90f * freq_curvature_penalty[index];
    keep_res[index] = envelope_map[index] - 1.00f * residual_penalty[index];
  }
  keep_freq = normalize01_masked_minmax(keep_freq, valid_mask);
  keep_res = normalize01_masked_minmax(keep_res, valid_mask);

  std::vector<float> residual_veto_gate(base_norm.size(), 0.0f);
  std::vector<float> combined_input(base_norm.size(), 0.0f);
  for (size_t index = 0; index < residual_veto_gate.size(); ++index) {
    residual_veto_gate[index] = clamp_value((keep_res[index] - 0.30f) / 0.70f, 0.0f, 1.0f);
    combined_input[index] = keep_freq[index] * (0.35f + 0.65f * residual_veto_gate[index]);
  }
  const auto combined_score = normalize01_masked_minmax(combined_input, valid_mask);

  std::vector<float> active_freq;
  std::vector<float> active_res;
  std::vector<float> active_combined;
  active_freq.reserve(base_norm.size());
  active_res.reserve(base_norm.size());
  active_combined.reserve(base_norm.size());
  for (size_t index = 0; index < valid_mask.size(); ++index) {
    if (!valid_mask[index]) {
      continue;
    }
    active_freq.push_back(keep_freq[index]);
    active_res.push_back(keep_res[index]);
    active_combined.push_back(combined_score[index]);
  }

  result.seed_freq_threshold = quantile_from_values(active_freq, 0.90, 1.0f);
  result.seed_res_threshold = quantile_from_values(active_res, 0.82, 1.0f);
  result.grow_freq_threshold = quantile_from_values(active_freq, 0.74, 1.0f);
  result.grow_res_threshold = quantile_from_values(active_res, 0.55, 1.0f);
  result.combined_threshold = quantile_from_values(active_combined, 0.78, 1.0f);

  std::vector<uint8_t> seed_mask(base_norm.size(), 0);
  std::vector<uint8_t> grow_mask(base_norm.size(), 0);
  for (size_t index = 0; index < seed_mask.size(); ++index) {
    seed_mask[index] = (valid_mask[index] &&
                        keep_freq[index] >= result.seed_freq_threshold &&
                        keep_res[index] >= result.seed_res_threshold)
                           ? 1
                           : 0;
    grow_mask[index] = (valid_mask[index] &&
                        keep_freq[index] >= result.grow_freq_threshold &&
                        keep_res[index] >= result.grow_res_threshold)
                           ? 1
                           : 0;
  }

  seed_mask = keep_large_components(seed_mask, rows, cols, 8);
  grow_mask = binary_closing_rect(grow_mask, rows, cols, 5, 3);
  grow_mask = binary_fill_holes(grow_mask, rows, cols);

  auto seed_components = label_components(seed_mask, rows, cols);
  std::vector<uint8_t> final_mask(base_norm.size(), 0);
  for (int label = 1; label <= seed_components.count; ++label) {
    std::vector<uint8_t> seed_component(base_norm.size(), 0);
    for (size_t index = 0; index < seed_component.size(); ++index) {
      if (seed_components.labels[index] == label) {
        seed_component[index] = 1;
      }
    }
    auto grown_component = binary_propagation(seed_component, grow_mask, rows, cols);
    if (std::count(grown_component.begin(), grown_component.end(), static_cast<uint8_t>(1)) < 12) {
      continue;
    }
    for (size_t index = 0; index < final_mask.size(); ++index) {
      final_mask[index] = final_mask[index] || grown_component[index];
    }
  }

  final_mask = keep_large_components(final_mask, rows, cols, 18);
  for (size_t index = 0; index < final_mask.size(); ++index) {
    final_mask[index] = (final_mask[index] && combined_score[index] >= (result.combined_threshold * 0.85f)) ? 1 : 0;
  }
  final_mask = binary_closing_rect(final_mask, rows, cols, 7, 3);
  final_mask = binary_fill_holes(final_mask, rows, cols);
  final_mask = keep_large_components(final_mask, rows, cols, 24, &result.component_count);
  for (size_t index = 0; index < final_mask.size(); ++index) {
    final_mask[index] = (final_mask[index] && valid_mask[index]) ? 1 : 0;
  }
  final_mask = keep_large_components(final_mask, rows, cols, 24, &result.component_count);

  result.final_fraction = mean_mask_value(final_mask);
  result.connected_fraction = connected_fraction(final_mask, valid_mask);
  result.mask = std::move(final_mask);
  return result;
}

}  // namespace

namespace holoscan::ops {

DinoV3SignalDetector::~DinoV3SignalDetector() {
  for (auto& buffers : channel_buffers_) {
    cudaFree(buffers.analysis_tensor_device);
    cudaFree(buffers.power_db_device);
    cudaFree(buffers.corrected_db_device);
    cudaFree(buffers.row_stat_device);
    cudaFree(buffers.row_smooth_device);
    cudaFree(buffers.frontend_reference_device);
    cudaFree(buffers.time_mean_device);
    cudaFree(buffers.freq_mean_device);
    cudaFree(buffers.background_device);
    cudaFree(buffers.box_filter_scratch_device);
    cudaFree(buffers.coherence_gate_device);
    cudaFree(buffers.coherence_gate_resized_device);
    cudaFreeHost(buffers.coherence_gate_host);
    cudaFreeHost(buffers.mask_host);
    if (buffers.coherence_gate_ready_event != nullptr) {
      cudaEventDestroy(buffers.coherence_gate_ready_event);
    }
    if (buffers.staging_stream != nullptr) {
      cudaStreamDestroy(buffers.staging_stream);
    }
    buffers = ChannelBuffers {};
  }
}

void DinoV3SignalDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<dino_in_t>("in");

  const std::vector<double> imagenet_mean_default{0.485, 0.456, 0.406};
  const std::vector<double> imagenet_std_default{0.229, 0.224, 0.225};

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(input_height_, "input_height", "Input height", "Detector output height.", 256);
  spec.param(input_width_, "input_width", "Input width", "Detector output width.", 512);
  spec.param(patch_size_, "patch_size", "Patch size", "Patch size used for DINO-aligned input shaping.", 16);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one output every N input frames per channel.", 1);
  spec.param(mask_threshold_db_, "mask_threshold_db", "Mask threshold (legacy)", "Legacy placeholder threshold retained for compatibility.", -20.0f);
  spec.param(log_detections_, "log_detections", "Log detections", "If true, logs detector execution details.", false);
  spec.param(backend_mode_, "backend_mode", "Backend mode", "Detector backend mode: reference or fast_gpu.", std::string("reference"));
  spec.param(enable_mask_save_, "enable_mask_save", "Enable mask save", "Enable writing detector masks to disk for debug runs.", false);
  spec.param(save_every_n_frames_, "save_every_n_frames", "Save stride", "Save one detector mask every N frames per channel.", 1);
  spec.param(max_masks_per_channel_, "max_masks_per_channel", "Max masks per channel", "Maximum number of detector masks to save per channel for a run.", 5);
  spec.param(output_dir_, "output_dir", "Output directory", "Directory where detector masks are written.", std::string("/workspace/dino_masks"));
  spec.param(use_pytorch_backend_, "use_pytorch_backend", "Use PyTorch backend", "If true, uses the LibTorch runtime for DINO score extraction when available.", true);
  spec.param(inference_backend_, "inference_backend", "Inference backend", "Backend mode: torchscript, pytorch_placeholder, or cuda_threshold_fallback.", std::string("torchscript"));
  spec.param(model_name_, "model_name", "Model name", "DINOv3 model name.", std::string("dinov3_vitb16"));
  spec.param(model_repo_path_, "model_repo_path", "Model repo path", "Path to local DINOv3 repository.", std::string("/workspace/models/dinov3"));
  spec.param(weights_path_, "weights_path", "Weights path", "Path to model weights.", std::string("/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"));
  spec.param(model_script_path_, "model_script_path", "Model script path", "Path to TorchScript model for model-forward backend.", std::string("/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts"));
  spec.param(torchscript_init_mode_, "torchscript_init_mode", "TorchScript init mode", "TorchScript initialization mode: load_only, load_cpu_eval, load_cuda_no_eval, or load_cuda_eval.", std::string("load_cuda_eval"));
  spec.param(strict_model_forward_, "strict_model_forward", "Strict model forward", "If true, drop frames when the DINO runtime fails instead of falling back.", false);
  spec.param(imagenet_mean_, "imagenet_mean", "ImageNet mean", "Mean used for notebook-aligned model normalization.", imagenet_mean_default);
  spec.param(imagenet_std_, "imagenet_std", "ImageNet std", "Standard deviation used for notebook-aligned model normalization.", imagenet_std_default);
  spec.param(fft_size_, "fft_size", "FFT size", "Notebook-derived FFT size constant for metadata and parity tracking.", 1024);
  spec.param(noverlap_, "noverlap", "FFT overlap", "Notebook-derived overlap constant for parity tracking.", 256);
  spec.param(ignore_sideband_hz_, "ignore_sideband_hz", "Ignore sideband Hz", "Frequency span to ignore on each side of the spectrum before DINO preprocessing.", 7.0e6);
  spec.param(frontend_correction_enable_, "frontend_correction_enable", "Frontend correction enable", "Enable frontend correction before DINO preprocessing.", true);
  spec.param(frontend_correction_row_q_, "frontend_correction_row_q", "Frontend correction row quantile", "Frontend correction row quantile.", 25.0);
  spec.param(frontend_correction_smooth_sigma_, "frontend_correction_smooth_sigma", "Frontend correction smoothing sigma", "Frontend correction smoothing sigma.", 12.0);
  spec.param(frontend_correction_reference_q_, "frontend_correction_reference_q", "Frontend correction reference quantile", "Frontend correction reference quantile.", 75.0);
  spec.param(frontend_correction_max_boost_db_, "frontend_correction_max_boost_db", "Frontend correction max boost", "Frontend correction max boost in dB.", 12.0);
  spec.param(frontend_correction_soft_knee_db_, "frontend_correction_soft_knee_db", "Frontend correction soft knee", "Retained for compatibility with the notebook parity config.", 4.0);
  spec.param(frontend_correction_edge_taper_fraction_, "frontend_correction_edge_taper_fraction", "Frontend correction edge taper fraction", "Retained for compatibility with the notebook parity config.", 0.10);
  spec.param(frontend_correction_edge_taper_sigma_, "frontend_correction_edge_taper_sigma", "Frontend correction edge taper sigma", "Retained for compatibility with the notebook parity config.", 6.0);
  spec.param(frontend_correction_edge_target_drop_db_, "frontend_correction_edge_target_drop_db", "Frontend correction edge target drop", "Retained for compatibility with the notebook parity config.", 2.5);
  spec.param(frontend_edge_guard_floor_, "frontend_edge_guard_floor", "Frontend edge guard floor", "Retained for compatibility with the notebook parity config.", 0.35);
  spec.param(dino_coherence_gate_floor_, "dino_coherence_gate_floor", "DINO coherence gate floor", "Lower bound for the coherent-style gate used in the hybrid detector.", 0.25);
  spec.param(dino_coherence_gate_span_db_, "dino_coherence_gate_span_db", "DINO coherence gate span", "Normalization span in dB for the coherent-style gate used in the hybrid detector.", 3.0);
  spec.param(texture_q_, "texture_q", "Texture quantile", "Retained for compatibility with notebook experiments.", 0.90);
  spec.param(texture_k_, "texture_k", "Texture K", "Retained for compatibility with notebook experiments.", 6);
  spec.param(power_q_, "power_q", "Power quantile", "Retained for compatibility with notebook experiments.", 0.90);
  spec.param(dino_group_k_, "dino_group_k", "DINO grouping K", "Retained for compatibility with notebook experiments.", 8);
  spec.param(dino_group_spatial_weight_, "dino_group_spatial_weight", "DINO grouping spatial weight", "Retained for compatibility with notebook experiments.", 0.35);
  spec.param(dino_group_score_q_, "dino_group_score_q", "DINO grouping score quantile", "Primary DINO score quantile emitted by the runtime.", 0.60);
  spec.param(pipeline_final_threshold_, "pipeline_final_threshold", "Pipeline final threshold", "Retained for compatibility with notebook experiments.", 0.20);
  spec.param(pipeline_final_threshold_no_speckle_, "pipeline_final_threshold_no_speckle", "Pipeline final threshold without speckle", "Retained for compatibility with notebook experiments.", 0.10);
  spec.param(pipeline_gap_floor_, "pipeline_gap_floor", "Pipeline gap floor", "Retained for compatibility with notebook experiments.", 0.10);
  spec.param(pipeline_component_min_size_, "pipeline_component_min_size", "Pipeline minimum component size", "Retained for compatibility with notebook experiments.", 5);
  spec.param(pipeline_component_min_size_no_speckle_, "pipeline_component_min_size_no_speckle", "Pipeline minimum component size without speckle", "Retained for compatibility with notebook experiments.", 2);
  spec.param(pipeline_power_rescue_floor_, "pipeline_power_rescue_floor", "Pipeline power rescue floor", "Retained for compatibility with notebook experiments.", 0.10);
  spec.param(pipeline_power_rescue_gain_, "pipeline_power_rescue_gain", "Pipeline power rescue gain", "Retained for compatibility with notebook experiments.", 2.0);
  spec.param(pipeline_strong_speckle_min_component_, "pipeline_strong_speckle_min_component", "Pipeline strong speckle minimum component", "Retained for compatibility with notebook experiments.", 10);
  spec.param(pipeline_texture_speckle_clean_threshold_, "pipeline_texture_speckle_clean_threshold", "Pipeline texture speckle clean threshold", "Retained for compatibility with notebook experiments.", 0.85);
  spec.param(pipeline_texture_speckle_strong_threshold_, "pipeline_texture_speckle_strong_threshold", "Pipeline texture strong threshold", "Retained for compatibility with notebook experiments.", 0.20);
  spec.param(timing_summary_enable_, "timing_summary_enable", "Timing summary enable", "Enable per-stage detector timing summaries.", true);
  spec.param(timing_summary_every_n_, "timing_summary_every_n", "Timing summary every N", "Emit timing summaries every N emitted detector frames per channel.", 16);
  spec.param(timing_summary_window_, "timing_summary_window", "Timing summary window", "Maximum number of emitted detector frames to accumulate before a timing summary reset.", 16);
}

void DinoV3SignalDetector::initialize() {
  holoscan::Operator::initialize();

  frame_count_.assign(num_channels_.get(), 0);
  masks_saved_.assign(num_channels_.get(), 0);
  timing_stats_.assign(num_channels_.get(), ChannelTimingStats {});
  channel_buffers_.assign(num_channels_.get(), ChannelBuffers {});

  const auto cuda_result = cudaFree(nullptr);
  if (cuda_result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA context initialization failed: ") + cudaGetErrorString(cuda_result));
  }

  if (enable_mask_save_.get()) {
    std::filesystem::create_directories(output_dir_.get());
  }

  pytorch_runtime_ready_ = false;
  pytorch_warning_emitted_ = false;
  torch_runtime_ = std::make_unique<DinoTorchRuntime>();

#ifdef HOLOHUB_HAS_TORCH
  if (use_pytorch_backend_.get()) {
    pytorch_runtime_ready_ = true;
    HOLOSCAN_LOG_INFO(
        "DINO hybrid detector runtime enabled. model_name='{}' repo='{}' weights='{}' script='{}'",
        model_name_.get(),
        model_repo_path_.get(),
        weights_path_.get(),
        model_script_path_.get());
  }
#else
  if (use_pytorch_backend_.get()) {
    HOLOSCAN_LOG_WARN(
        "PyTorch backend requested, but the detector was built without Torch. Falling back when the runtime cannot provide a DINO score.");
  }
#endif
}

void DinoV3SignalDetector::compute(holoscan::InputContext& op_input,
                                   holoscan::OutputContext&,
                                   holoscan::ExecutionContext&) {
  auto input = op_input.receive<dino_in_t>("in").value();
  auto& fft_tensor = std::get<0>(input);
  auto stream = std::get<1>(input);

  auto meta = metadata();
  const uint16_t channel_number = meta->get<uint16_t>("channel_number", 0);
  if (channel_number >= frame_count_.size()) {
    HOLOSCAN_LOG_WARN("DINO hybrid detector received out-of-range channel {} (configured channels: {}).",
                      channel_number,
                      frame_count_.size());
    return;
  }

  const uint64_t frame_number = ++frame_count_[channel_number];
  const int emit_stride = std::max(1, emit_stride_.get());
  if ((frame_number % static_cast<uint64_t>(emit_stride)) != 0) {
    return;
  }

  const int input_rows = static_cast<int>(fft_tensor.Size(0));
  const int input_cols = static_cast<int>(fft_tensor.Size(1));
  if (input_rows <= 0 || input_cols <= 0) {
    HOLOSCAN_LOG_WARN("DINO hybrid detector received empty tensor on channel {}", channel_number);
    return;
  }

  const auto canonical_view = canonical_tensor_view(input_rows, input_cols);
  const int src_rows = canonical_view.rows;
  const int src_cols = canonical_view.cols;
  const int dst_rows = std::max(1, input_height_.get());
  const int dst_cols = std::max(1, input_width_.get());
  const int patch_size = std::max(1, patch_size_.get());
  const int total_bins = src_rows * src_cols;
  const size_t frame_elements = static_cast<size_t>(total_bins);
  const size_t dst_elements = static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols);
  const size_t dst_bytes = dst_elements * sizeof(float);
  const bool timing_enabled = timing_summary_enable_.get();
  const bool should_save_mask = enable_mask_save_.get() &&
                                (frame_number % static_cast<uint64_t>(std::max(1, save_every_n_frames_.get())) == 0) &&
                                (masks_saved_[channel_number] < max_masks_per_channel_.get());
  const auto requested_backend_mode = backend_mode_.get();
  const bool requested_fast_backend = requested_backend_mode == "fast_gpu";
  const bool requested_reference_backend = requested_backend_mode == "reference";
  if (!requested_fast_backend && !requested_reference_backend) {
    HOLOSCAN_LOG_WARN("Unsupported DINO backend_mode='{}'. Falling back to reference.", requested_backend_mode);
  }
  const bool use_fast_backend = requested_fast_backend && strict_model_forward_.get() && !should_save_mask;
  const std::string effective_backend_mode = use_fast_backend ? std::string("fast_gpu") : std::string("reference");
  std::array<double, kTimingStageCount> stage_ms {};
  const auto total_start = std::chrono::steady_clock::now();

  auto time_step_ms = [&](size_t stage_index, auto&& fn) {
    if (!timing_enabled) {
      fn();
      return;
    }

    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    const auto sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
      throw std::runtime_error(std::string("timing synchronization failed at ") +
                               kTimingStageNames[stage_index] + ": " +
                               cudaGetErrorString(sync_result));
    }
    stage_ms[stage_index] = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - stage_start)
                                .count();
  };

  double span_hz = 0.0;
  if (meta->has_key("sample_rate_hz")) {
    span_hz = meta->get<double>("sample_rate_hz");
  } else if (meta->has_key("span")) {
    span_hz = static_cast<double>(meta->get<uint64_t>("span", 0));
  } else if (meta->has_key("bandwidth_hz")) {
    span_hz = meta->get<double>("bandwidth_hz");
  }
  if (!std::isfinite(span_hz) || span_hz <= 0.0) {
    span_hz = 0.0;
  }

  double resolution_hz = static_cast<double>(meta->get<uint64_t>("resolution", 0));
  if ((!std::isfinite(resolution_hz) || resolution_hz <= 0.0) && span_hz > 0.0 && src_rows > 0) {
    resolution_hz = span_hz / static_cast<double>(src_rows);
  }

  int ignore_bins_per_side = 0;
  if (resolution_hz > 0.0 && ignore_sideband_hz_.get() > 0.0) {
    ignore_bins_per_side = static_cast<int>(std::ceil(ignore_sideband_hz_.get() / resolution_hz));
    ignore_bins_per_side = std::clamp(ignore_bins_per_side, 0, std::max(0, (src_rows - patch_size) / 2));
  }

  auto& buffers = channel_buffers_[channel_number];

  try {
    time_step_ms(kInputStage, [&] {
      auto allocate_device_float = [](float*& pointer, size_t requested_elements) {
        const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&pointer), requested_elements * sizeof(float));
        if (alloc_result != cudaSuccess) {
          throw std::runtime_error(std::string("device float buffer allocation failed: ") +
                                   cudaGetErrorString(alloc_result));
        }
      };

      if (buffers.staging_stream == nullptr) {
        const auto stream_result = cudaStreamCreateWithFlags(&buffers.staging_stream, cudaStreamNonBlocking);
        if (stream_result != cudaSuccess) {
          throw std::runtime_error(std::string("coherence staging stream creation failed: ") +
                                   cudaGetErrorString(stream_result));
        }
      }
      if (buffers.coherence_gate_ready_event == nullptr) {
        const auto event_result = cudaEventCreateWithFlags(&buffers.coherence_gate_ready_event, cudaEventDisableTiming);
        if (event_result != cudaSuccess) {
          throw std::runtime_error(std::string("coherence staging event creation failed: ") +
                                   cudaGetErrorString(event_result));
        }
      }

      if (buffers.frame_elements != frame_elements) {
        cudaFree(buffers.analysis_tensor_device);
        cudaFree(buffers.power_db_device);
        cudaFree(buffers.corrected_db_device);
        cudaFree(buffers.time_mean_device);
        cudaFree(buffers.freq_mean_device);
        cudaFree(buffers.background_device);
        cudaFree(buffers.box_filter_scratch_device);
        cudaFree(buffers.coherence_gate_device);

        buffers.analysis_tensor_device = nullptr;
        buffers.power_db_device = nullptr;
        buffers.corrected_db_device = nullptr;
        buffers.time_mean_device = nullptr;
        buffers.freq_mean_device = nullptr;
        buffers.background_device = nullptr;
        buffers.box_filter_scratch_device = nullptr;
        buffers.coherence_gate_device = nullptr;

        const auto analysis_alloc = cudaMalloc(reinterpret_cast<void**>(&buffers.analysis_tensor_device),
                                               frame_elements * sizeof(dino_complex));
        if (analysis_alloc != cudaSuccess) {
          throw std::runtime_error(std::string("analysis tensor allocation failed: ") +
                                   cudaGetErrorString(analysis_alloc));
        }
        allocate_device_float(buffers.power_db_device, frame_elements);
        allocate_device_float(buffers.corrected_db_device, frame_elements);
        allocate_device_float(buffers.time_mean_device, frame_elements);
        allocate_device_float(buffers.freq_mean_device, frame_elements);
        allocate_device_float(buffers.background_device, frame_elements);
        allocate_device_float(buffers.box_filter_scratch_device, frame_elements);
        allocate_device_float(buffers.coherence_gate_device, frame_elements);

        buffers.frame_elements = frame_elements;
      }

      if (buffers.mask_elements != dst_elements) {
        cudaFree(buffers.coherence_gate_resized_device);
        cudaFreeHost(buffers.coherence_gate_host);
        cudaFreeHost(buffers.mask_host);

        buffers.coherence_gate_resized_device = nullptr;
        buffers.coherence_gate_host = nullptr;
        buffers.mask_host = nullptr;

        allocate_device_float(buffers.coherence_gate_resized_device, dst_elements);
        const auto host_gate_alloc = cudaMallocHost(reinterpret_cast<void**>(&buffers.coherence_gate_host), dst_bytes);
        if (host_gate_alloc != cudaSuccess) {
          throw std::runtime_error(std::string("coherence gate host allocation failed: ") +
                                   cudaGetErrorString(host_gate_alloc));
        }

        buffers.mask_elements = dst_elements;
      }

      if (buffers.row_elements != static_cast<size_t>(src_rows)) {
        cudaFree(buffers.row_stat_device);
        cudaFree(buffers.row_smooth_device);
        cudaFree(buffers.frontend_reference_device);
        buffers.row_stat_device = nullptr;
        buffers.row_smooth_device = nullptr;
        buffers.frontend_reference_device = nullptr;

        allocate_device_float(buffers.row_stat_device, static_cast<size_t>(src_rows));
        allocate_device_float(buffers.row_smooth_device, static_cast<size_t>(src_rows));
        allocate_device_float(buffers.frontend_reference_device, 1);
        buffers.row_elements = static_cast<size_t>(src_rows);
      }

      if (should_save_mask && buffers.mask_host == nullptr) {
        const auto alloc_result = cudaMallocHost(reinterpret_cast<void**>(&buffers.mask_host), buffers.mask_elements * sizeof(uint8_t));
        if (alloc_result != cudaSuccess) {
          throw std::runtime_error(std::string("mask host allocation failed: ") + cudaGetErrorString(alloc_result));
        }
      }

      constexpr int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      if (canonical_view.transposed) {
        dino_transpose_kernel<<<blocks, threads, 0, stream>>>(fft_tensor.Data(), input_rows, input_cols, buffers.analysis_tensor_device);
        const auto kernel_result = cudaGetLastError();
        if (kernel_result != cudaSuccess) {
          throw std::runtime_error(std::string("analysis transpose kernel launch failed: ") + cudaGetErrorString(kernel_result));
        }
      } else {
        const auto copy_result = cudaMemcpyAsync(buffers.analysis_tensor_device,
                                                 fft_tensor.Data(),
                                                 frame_elements * sizeof(dino_complex),
                                                 cudaMemcpyDeviceToDevice,
                                                 stream);
        if (copy_result != cudaSuccess) {
          throw std::runtime_error(std::string("analysis tensor copy failed: ") + cudaGetErrorString(copy_result));
        }
      }
    });

    time_step_ms(kPowerDbStage, [&] {
      constexpr int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      dino_power_db_kernel<<<blocks, threads, 0, stream>>>(buffers.analysis_tensor_device,
                                                            src_rows,
                                                            src_cols,
                                                            buffers.power_db_device);
      const auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("power_db kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
    });

    time_step_ms(kFrontendStage, [&] {
      if (use_fast_backend) {
        return;
      }
      constexpr int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      const int row_blocks = (src_rows + threads - 1) / threads;
      const int smooth_radius = std::max(1, static_cast<int>(std::ceil(std::max(frontend_correction_smooth_sigma_.get(), 1.0) * 1.5)));

      dino_row_mean_kernel<<<src_rows, threads, 0, stream>>>(buffers.power_db_device,
                                                              src_rows,
                                                              src_cols,
                                                              buffers.row_stat_device);
      dino_gaussian_smooth_rows_kernel<<<row_blocks, threads, 0, stream>>>(buffers.row_stat_device,
                                                                            src_rows,
                                                                            smooth_radius,
                                                                            static_cast<float>(std::max(frontend_correction_smooth_sigma_.get(), 1.0)),
                                                                            buffers.row_smooth_device);
      dino_frontend_reference_kernel<<<1, threads, 0, stream>>>(buffers.row_smooth_device,
                                                                 src_rows,
                                                                 static_cast<float>(frontend_correction_reference_q_.get() / 100.0),
                                                                 buffers.frontend_reference_device);
      dino_frontend_correction_kernel<<<blocks, threads, 0, stream>>>(buffers.power_db_device,
                                                                       src_rows,
                                                                       src_cols,
                                                                       buffers.row_smooth_device,
                                                                       buffers.frontend_reference_device,
                                                                       static_cast<float>(frontend_correction_max_boost_db_.get()),
                                                                       buffers.corrected_db_device);
      const auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("frontend correction kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
    });

    time_step_ms(kCoherenceStage, [&] {
      if (use_fast_backend) {
        return;
      }
      constexpr int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      dino_box_mean_cols_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                 src_rows,
                                                                 src_cols,
                                                                 4,
                                                                 buffers.time_mean_device);
      dino_box_mean_rows_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                 src_rows,
                                                                 src_cols,
                                                                 3,
                                                                 buffers.freq_mean_device);
      dino_box_mean_cols_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                 src_rows,
                                                                 src_cols,
                                                                 10,
                                                                 buffers.box_filter_scratch_device);
      dino_box_mean_rows_kernel<<<blocks, threads, 0, stream>>>(buffers.box_filter_scratch_device,
                                                                 src_rows,
                                                                 src_cols,
                                                                 8,
                                                                 buffers.background_device);
      dino_coherence_gate_kernel<<<blocks, threads, 0, stream>>>(buffers.time_mean_device,
                                                                  buffers.freq_mean_device,
                                                                  src_rows,
                                                                  src_cols,
                                                                  ignore_bins_per_side,
                                                                  static_cast<float>(dino_coherence_gate_floor_.get()),
                                                                  static_cast<float>(dino_coherence_gate_span_db_.get()),
                                                                  buffers.coherence_gate_device);
      const auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("coherence gate kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      const auto ready_result = cudaEventRecord(buffers.coherence_gate_ready_event, stream);
      if (ready_result != cudaSuccess) {
        throw std::runtime_error(std::string("coherence gate event record failed: ") + cudaGetErrorString(ready_result));
      }
      const auto wait_result = cudaStreamWaitEvent(buffers.staging_stream, buffers.coherence_gate_ready_event, 0);
      if (wait_result != cudaSuccess) {
        throw std::runtime_error(std::string("coherence gate staging wait failed: ") + cudaGetErrorString(wait_result));
      }

      const int resized_blocks = (static_cast<int>(dst_elements) + threads - 1) / threads;
      dino_resize_bilinear_kernel<<<resized_blocks, threads, 0, buffers.staging_stream>>>(buffers.coherence_gate_device,
                                                                                           src_rows,
                                                                                           src_cols,
                                                                                           buffers.coherence_gate_resized_device,
                                                                                           dst_rows,
                                                                                           dst_cols);
      const auto resize_kernel_result = cudaGetLastError();
      if (resize_kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("coherence gate resize kernel launch failed: ") + cudaGetErrorString(resize_kernel_result));
      }

      const auto copy_result = cudaMemcpyAsync(buffers.coherence_gate_host,
                                               buffers.coherence_gate_resized_device,
                                               dst_bytes,
                                               cudaMemcpyDeviceToHost,
                                               buffers.staging_stream);
      if (copy_result != cudaSuccess) {
        throw std::runtime_error(std::string("coherence gate resized copy failed: ") + cudaGetErrorString(copy_result));
      }
    });

    DinoTorchRuntimeResult runtime_result;
    runtime_result.aligned_rows = dst_rows;
    runtime_result.aligned_cols = dst_cols;
    std::string backend_used = "cuda_threshold_fallback";
    time_step_ms(kTorchRuntimeStage, [&] {
      if (!use_pytorch_backend_.get() || !torch_runtime_) {
        return;
      }

      DinoTorchRuntimeConfig runtime_config;
      runtime_config.inference_backend = inference_backend_.get();
      runtime_config.model_script_path = model_script_path_.get();
      runtime_config.torchscript_init_mode = torchscript_init_mode_.get();
      runtime_config.imagenet_mean = imagenet_mean_.get();
      runtime_config.imagenet_std = imagenet_std_.get();
      runtime_config.return_final_mask = true;
      runtime_config.ignore_sideband_hz = ignore_sideband_hz_.get();
      runtime_config.frontend_correction_enable = frontend_correction_enable_.get();
      runtime_config.frontend_correction_row_q = frontend_correction_row_q_.get();
      runtime_config.frontend_correction_smooth_sigma = frontend_correction_smooth_sigma_.get();
      runtime_config.frontend_correction_reference_q = frontend_correction_reference_q_.get();
      runtime_config.frontend_correction_max_boost_db = frontend_correction_max_boost_db_.get();
      runtime_config.frontend_correction_soft_knee_db = frontend_correction_soft_knee_db_.get();
      runtime_config.frontend_correction_edge_taper_fraction = frontend_correction_edge_taper_fraction_.get();
      runtime_config.frontend_correction_edge_taper_sigma = frontend_correction_edge_taper_sigma_.get();
      runtime_config.frontend_correction_edge_target_drop_db = frontend_correction_edge_target_drop_db_.get();
      runtime_config.power_q = power_q_.get();
      runtime_config.dino_group_score_q = dino_group_score_q_.get();
      runtime_config.pipeline_final_threshold = pipeline_final_threshold_.get();
      runtime_config.pipeline_gap_floor = pipeline_gap_floor_.get();
      runtime_config.pipeline_power_rescue_floor = pipeline_power_rescue_floor_.get();
      runtime_config.pipeline_power_rescue_gain = pipeline_power_rescue_gain_.get();

      DinoTorchRuntimeInput runtime_input;
      runtime_input.channel_number = channel_number;
      runtime_input.frame_number = frame_number;
      runtime_input.src_rows = src_rows;
      runtime_input.src_cols = src_cols;
      runtime_input.dst_rows = dst_rows;
      runtime_input.dst_cols = dst_cols;
      runtime_input.patch_size = patch_size;
      runtime_input.cuda_stream = stream;
      runtime_input.resolution_hz = resolution_hz;
      runtime_input.span_hz = span_hz;
      runtime_input.power_db_device = buffers.power_db_device;

      runtime_result = torch_runtime_->run(runtime_config, runtime_input);
      backend_used = runtime_result.backend_used;
      if (!runtime_result.success) {
        if (strict_model_forward_.get()) {
          throw std::runtime_error(std::string("DINO runtime failed at ") + runtime_result.error_stage +
                                   ": " + runtime_result.error_message + " (" + runtime_result.error_detail + ")");
        }
        if (!pytorch_warning_emitted_) {
          HOLOSCAN_LOG_WARN("DINO runtime fallback engaged at stage '{}' with message '{}' detail='{}'.",
                            runtime_result.error_stage,
                            runtime_result.error_message,
                            runtime_result.error_detail);
          pytorch_warning_emitted_ = true;
        }
      }
    });

    HybridPostprocessResult hybrid_result;
    time_step_ms(kHybridStage, [&] {
      auto valid_mask = resize_valid_row_mask(src_rows, dst_rows, dst_cols, ignore_bins_per_side);
      if (use_fast_backend) {
        if (runtime_result.success &&
            runtime_result.final_mask.size() == static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols)) {
          hybrid_result = run_fast_dino_postprocess(runtime_result.final_mask,
                                                    valid_mask,
                                                    dst_rows,
                                                    dst_cols,
                                                    static_cast<float>(runtime_result.dino_threshold),
                                                    std::max(12, pipeline_component_min_size_.get() * 4));
          return;
        }
        throw std::runtime_error("fast_gpu backend requires a valid DINO score map from the runtime");
      }

      const auto sync_result = cudaStreamSynchronize(buffers.staging_stream);
      if (sync_result != cudaSuccess) {
        throw std::runtime_error(std::string("coherence gate staging synchronization failed: ") + cudaGetErrorString(sync_result));
      }

      std::vector<float> coherence_gate_resized(buffers.coherence_gate_host,
                                                buffers.coherence_gate_host + static_cast<std::ptrdiff_t>(dst_elements));
      std::vector<float> dino_score_map;
      if (runtime_result.success &&
          runtime_result.final_mask.size() == static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols)) {
        dino_score_map = runtime_result.final_mask;
      } else {
        dino_score_map = coherence_gate_resized;
        backend_used = runtime_result.success ? std::string("coherence_gate_fallback") : std::string("dino_runtime_fallback");
      }

      hybrid_result = run_residual_veto_hybrid(dino_score_map, coherence_gate_resized, valid_mask, dst_rows, dst_cols);
    });

    time_step_ms(kMaskSaveStage, [&] {
      if (!should_save_mask) {
        return;
      }
      std::vector<uint8_t> image(hybrid_result.mask.size(), 0);
      for (size_t index = 0; index < hybrid_result.mask.size(); ++index) {
        image[index] = hybrid_result.mask[index] ? 255 : 0;
      }

      const auto mask_path = make_mask_output_path(output_dir_.get(), channel_number, frame_number, dst_rows, dst_cols);
      if (!write_pgm(mask_path, image, dst_cols, dst_rows)) {
        HOLOSCAN_LOG_ERROR("Failed to write DINO hybrid mask image: {}", mask_path);
      } else {
        ++masks_saved_[channel_number];
        if (log_detections_.get()) {
          HOLOSCAN_LOG_INFO("Saved DINO hybrid mask for channel {} frame {} to {}",
                            channel_number,
                            frame_number,
                            mask_path);
        }
      }
    });

    if (timing_enabled) {
      stage_ms[kTotalStage] = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - total_start)
                                  .count();
    }

    meta->set("dino_frame_number", frame_number);
    meta->set("dino_mask_height", static_cast<uint32_t>(dst_rows));
    meta->set("dino_mask_width", static_cast<uint32_t>(dst_cols));
    meta->set("dino_mask_threshold_db", mask_threshold_db_.get());
    meta->set("dino_backend", backend_used);
    meta->set("dino_backend_mode", effective_backend_mode);
    meta->set("dino_model_name", model_name_.get());
    meta->set("dino_weights_path", weights_path_.get());
    meta->set("dino_model_script_path", model_script_path_.get());
    meta->set("dino_torchscript_init_mode", torchscript_init_mode_.get());
    meta->set("dino_torchscript_forward_ready", runtime_result.torchscript_forward_ready);
    meta->set("dino_patch_size", patch_size);
    meta->set("dino_fft_size", fft_size_.get());
    meta->set("dino_noverlap", noverlap_.get());
    meta->set("dino_ignore_bins_per_side", ignore_bins_per_side);
    meta->set("dino_freq_bin_hz", resolution_hz);
    meta->set("dino_frontend_correction_enabled", frontend_correction_enable_.get());
    meta->set("dino_input_aligned_height", runtime_result.aligned_rows);
    meta->set("dino_input_aligned_width", runtime_result.aligned_cols);
    meta->set("dino_group_score_threshold", static_cast<double>(hybrid_result.combined_threshold));
    meta->set("dino_power_score_threshold", runtime_result.power_threshold);
    meta->set("dino_pipeline_final_threshold", static_cast<double>(hybrid_result.combined_threshold * 0.85f));
    meta->set("dino_pipeline_variant",
          use_fast_backend ? std::string("dino_score_fast_mask_v1")
                   : std::string("coherent_shell_residual_veto_hybrid_v1"));
    meta->set("dino_coherence_gate_floor", dino_coherence_gate_floor_.get());
    meta->set("dino_coherence_gate_span_db", dino_coherence_gate_span_db_.get());
    meta->set("dino_seed_freq_threshold", static_cast<double>(hybrid_result.seed_freq_threshold));
    meta->set("dino_seed_res_threshold", static_cast<double>(hybrid_result.seed_res_threshold));
    meta->set("dino_grow_freq_threshold", static_cast<double>(hybrid_result.grow_freq_threshold));
    meta->set("dino_grow_res_threshold", static_cast<double>(hybrid_result.grow_res_threshold));
    meta->set("dino_component_count", static_cast<uint32_t>(std::max(0, hybrid_result.component_count)));
    meta->set("dino_mask_fraction", static_cast<double>(hybrid_result.final_fraction));
    meta->set("dino_connected_fraction", static_cast<double>(hybrid_result.connected_fraction));
    meta->set("dino_timing_total_ms", stage_ms[kTotalStage]);
    meta->set("dino_timing_summary_enabled", timing_enabled);

    if (timing_enabled) {
      auto& timing = timing_stats_[channel_number];
      ++timing.window_frames;
      for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
        timing.total_ms[stage_index] += stage_ms[stage_index];
        timing.max_ms[stage_index] = std::max(timing.max_ms[stage_index], stage_ms[stage_index]);
      }

      const uint64_t every_n = static_cast<uint64_t>(std::max(1, timing_summary_every_n_.get()));
      const uint64_t window = static_cast<uint64_t>(std::max(1, timing_summary_window_.get()));
      if (timing.window_frames >= every_n) {
        std::ostringstream summary;
        summary << "DINO hybrid timing summary ch=" << channel_number
                << " frames=" << timing.window_frames;
        for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
          const double mean_ms = timing.total_ms[stage_index] / static_cast<double>(timing.window_frames);
          summary << " " << kTimingStageNames[stage_index] << "(mean=" << mean_ms
                  << ",max=" << timing.max_ms[stage_index] << ")";
        }
        HOLOSCAN_LOG_INFO(summary.str());
      }
      if (timing.window_frames >= window) {
        timing = ChannelTimingStats {};
      }
    }
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("DINO hybrid detector failed on channel {} frame {}: {}",
                       channel_number,
                       frame_number,
                       error.what());
  }
}

}  // namespace holoscan::ops
// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "dinov3_signal_detector.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
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
  std::vector<float> combined_score;
  float seed_freq_threshold = 1.0f;
  float seed_res_threshold = 1.0f;
  float grow_freq_threshold = 1.0f;
  float grow_res_threshold = 1.0f;
  float combined_threshold = 1.0f;
  float final_fraction = 0.0f;
  float connected_fraction = 0.0f;
  int component_count = 0;
};

struct ChunkPlanEntry {
  int chunk_index = 0;
  int row_start = 0;
  int row_stop = 0;
  double freq_start_hz = 0.0;
  double freq_stop_hz = 0.0;
};

struct DetectionBox {
  int freq_start = 0;
  int freq_stop = 0;
  int time_start = 0;
  int time_stop = 0;
  int filled_area = 0;
  float density = 0.0f;
  float bbox_density = 0.0f;
  float envelope_density = 0.0f;
  float score_mean = 0.0f;
  float score_peak = 0.0f;
  std::vector<int> source_chunk_indices;
};

struct GroupingResult {
  std::vector<uint8_t> seed_mask;
  std::vector<uint8_t> bridged_mask;
  std::vector<int> component_labels;
  std::vector<uint8_t> grouped_mask;
  std::vector<DetectionBox> boxes;
  float peak_score_floor = 0.0f;
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

std::vector<float> gaussian_blur(const std::vector<float>& input,
                                 int rows,
                                 int cols,
                                 double sigma_rows,
                                 double sigma_cols) {
  auto row_blurred = convolve_axis(input, rows, cols, gaussian_kernel(sigma_rows), true);
  return convolve_axis(row_blurred, rows, cols, gaussian_kernel(sigma_cols), false);
}

std::vector<float> gaussian_second_derivative_rows(const std::vector<float>& input,
                                                   int rows,
                                                   int cols,
                                                   double sigma) {
  if (sigma <= 0.0) {
    return std::vector<float>(input.size(), 0.0f);
  }

  const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1), 0.0f);
  for (int offset = -radius; offset <= radius; ++offset) {
    const double x = static_cast<double>(offset);
    const double sigma2 = sigma * sigma;
    const double weight = ((x * x - sigma2) / (sigma2 * sigma2)) * std::exp(-(x * x) / (2.0 * sigma2));
    kernel[static_cast<size_t>(offset + radius)] = static_cast<float>(weight);
  }

  return convolve_axis(input, rows, cols, kernel, true);
}

std::vector<float> gaussian_smooth_rows(const std::vector<float>& input,
                                        int rows,
                                        int radius,
                                        float sigma) {
  std::vector<float> output(static_cast<size_t>(rows), 0.0f);
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    float weight_sum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
      const int src_row = clamp_value(row + offset, 0, rows - 1);
      const float weight = std::exp(-(static_cast<float>(offset * offset)) / (2.0f * sigma * sigma));
      sum += input[static_cast<size_t>(src_row)] * weight;
      weight_sum += weight;
    }
    output[static_cast<size_t>(row)] = weight_sum > 0.0f ? sum / weight_sum : input[static_cast<size_t>(row)];
  }
  return output;
}

std::vector<float> box_mean_cols(const std::vector<float>& input, int rows, int cols, int radius_cols) {
  std::vector<float> output(input.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int col_start = std::max(0, col - radius_cols);
      const int col_stop = std::min(cols - 1, col + radius_cols);
      float sum = 0.0f;
      int count = 0;
      for (int src_col = col_start; src_col <= col_stop; ++src_col) {
        sum += input[flat_index(cols, row, src_col)];
        ++count;
      }
      output[flat_index(cols, row, col)] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }
  }
  return output;
}

std::vector<float> box_mean_rows(const std::vector<float>& input, int rows, int cols, int radius_rows) {
  std::vector<float> output(input.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int row_start = std::max(0, row - radius_rows);
      const int row_stop = std::min(rows - 1, row + radius_rows);
      float sum = 0.0f;
      int count = 0;
      for (int src_row = row_start; src_row <= row_stop; ++src_row) {
        sum += input[flat_index(cols, src_row, col)];
        ++count;
      }
      output[flat_index(cols, row, col)] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }
  }
  return output;
}

std::vector<float> box_mean_2d(const std::vector<float>& input,
                               int rows,
                               int cols,
                               int radius_rows,
                               int radius_cols) {
  return box_mean_rows(box_mean_cols(input, rows, cols, radius_cols), rows, cols, radius_rows);
}

std::vector<float> gaussian_first_derivative_rows(const std::vector<float>& input,
                                                  int rows,
                                                  int cols,
                                                  double sigma) {
  if (sigma <= 0.0) {
    return std::vector<float>(input.size(), 0.0f);
  }
  const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1), 0.0f);
  const double sigma2 = sigma * sigma;
  for (int offset = -radius; offset <= radius; ++offset) {
    const double x = static_cast<double>(offset);
    kernel[static_cast<size_t>(offset + radius)] =
        static_cast<float>((-x / sigma2) * std::exp(-(x * x) / (2.0 * sigma2)));
  }
  const auto smoothed_cols = convolve_axis(input, rows, cols, gaussian_kernel(sigma), false);
  return convolve_axis(smoothed_cols, rows, cols, kernel, true);
}

std::vector<float> gaussian_first_derivative_cols(const std::vector<float>& input,
                                                  int rows,
                                                  int cols,
                                                  double sigma) {
  if (sigma <= 0.0) {
    return std::vector<float>(input.size(), 0.0f);
  }
  const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1), 0.0f);
  const double sigma2 = sigma * sigma;
  for (int offset = -radius; offset <= radius; ++offset) {
    const double x = static_cast<double>(offset);
    kernel[static_cast<size_t>(offset + radius)] =
        static_cast<float>((-x / sigma2) * std::exp(-(x * x) / (2.0 * sigma2)));
  }
  const auto smoothed_rows = convolve_axis(input, rows, cols, gaussian_kernel(sigma), true);
  return convolve_axis(smoothed_rows, rows, cols, kernel, false);
}

std::vector<float> normalize01_quantile(const std::vector<float>& input,
                                        double low_q,
                                        double high_q) {
  std::vector<float> values;
  values.reserve(input.size());
  for (float value : input) {
    if (std::isfinite(value)) {
      values.push_back(value);
    }
  }

  std::vector<float> output(input.size(), 0.0f);
  if (values.empty()) {
    return output;
  }

  const float low = quantile_from_values(values, clamp_value(low_q / 100.0, 0.0, 1.0), 0.0f);
  const float high = quantile_from_values(values, clamp_value(high_q / 100.0, 0.0, 1.0), 1.0f);
  const float scale = std::max(high - low, 1.0e-6f);
  for (size_t index = 0; index < input.size(); ++index) {
    if (!std::isfinite(input[index])) {
      continue;
    }
    output[index] = clamp_value((input[index] - low) / scale, 0.0f, 1.0f);
  }
  return output;
}

std::vector<float> embed_aligned_map_in_source_canvas(const std::vector<float>& aligned_map,
                                                      int aligned_rows,
                                                      int aligned_cols,
                                                      int source_rows,
                                                      int source_cols,
                                                      int row_offset,
                                                      int col_offset) {
  std::vector<float> canvas(static_cast<size_t>(std::max(source_rows, 0)) * static_cast<size_t>(std::max(source_cols, 0)), 0.0f);
  if (source_rows <= 0 || source_cols <= 0 || aligned_rows <= 0 || aligned_cols <= 0 ||
      aligned_map.size() != static_cast<size_t>(aligned_rows) * static_cast<size_t>(aligned_cols)) {
    return canvas;
  }

  const int clamped_row_offset = clamp_value(row_offset, 0, std::max(0, source_rows - 1));
  const int clamped_col_offset = clamp_value(col_offset, 0, std::max(0, source_cols - 1));
  const int copy_rows = std::min(aligned_rows, std::max(0, source_rows - clamped_row_offset));
  const int copy_cols = std::min(aligned_cols, std::max(0, source_cols - clamped_col_offset));
  for (int row = 0; row < copy_rows; ++row) {
    const int dst_row = clamped_row_offset + row;
    for (int col = 0; col < copy_cols; ++col) {
      const int dst_col = clamped_col_offset + col;
      canvas[flat_index(source_cols, dst_row, dst_col)] = aligned_map[flat_index(aligned_cols, row, col)];
    }
  }
  return canvas;
}

std::vector<float> project_aligned_map_to_output(const std::vector<float>& aligned_map,
                                                 int aligned_rows,
                                                 int aligned_cols,
                                                 int source_rows,
                                                 int source_cols,
                                                 int row_offset,
                                                 int col_offset,
                                                 int output_rows,
                                                 int output_cols) {
  const auto source_canvas = embed_aligned_map_in_source_canvas(aligned_map,
                                                                aligned_rows,
                                                                aligned_cols,
                                                                source_rows,
                                                                source_cols,
                                                                row_offset,
                                                                col_offset);
  return resize_bilinear(source_canvas, source_rows, source_cols, output_rows, output_cols);
}

std::vector<uint8_t> resize_mask_nearest(const std::vector<uint8_t>& input,
                                         int input_rows,
                                         int input_cols,
                                         int output_rows,
                                         int output_cols) {
  std::vector<uint8_t> output(static_cast<size_t>(std::max(output_rows, 0)) * static_cast<size_t>(std::max(output_cols, 0)), 0);
  if (input_rows <= 0 || input_cols <= 0 || output_rows <= 0 || output_cols <= 0 ||
      input.size() != static_cast<size_t>(input_rows) * static_cast<size_t>(input_cols)) {
    return output;
  }
  for (int out_row = 0; out_row < output_rows; ++out_row) {
    const int src_row = std::min(input_rows - 1,
                                 static_cast<int>((static_cast<int64_t>(out_row) * static_cast<int64_t>(input_rows)) /
                                                  static_cast<int64_t>(std::max(output_rows, 1))));
    for (int out_col = 0; out_col < output_cols; ++out_col) {
      const int src_col = std::min(input_cols - 1,
                                   static_cast<int>((static_cast<int64_t>(out_col) * static_cast<int64_t>(input_cols)) /
                                                    static_cast<int64_t>(std::max(output_cols, 1))));
      output[flat_index(output_cols, out_row, out_col)] = input[flat_index(input_cols, src_row, src_col)];
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
  std::vector<int> pending;
  pending.reserve(mask.size());

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t index = flat_index(cols, row, col);
      if (!mask[index] || result.labels[index] != 0) {
        continue;
      }

      ++next_label;
      int component_size = 0;
      pending.clear();
      pending.push_back(static_cast<int>(index));
      size_t pending_head = 0;
      result.labels[index] = next_label;
      while (pending_head < pending.size()) {
        const int flat_index_value = pending[pending_head++];
        const int cur_row = flat_index_value / cols;
        const int cur_col = flat_index_value % cols;
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
          pending.push_back(static_cast<int>(flat));
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
  int kept_count = 0;
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
    for (int component_size : labelled.sizes) {
      if (component_size >= std::max(1, min_size)) {
        ++kept_count;
      }
    }
    *kept_component_count = kept_count;
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
  std::vector<int> pending;
  pending.reserve(mask.size());
  auto maybe_enqueue = [&](int row, int col) {
    const size_t index = flat_index(cols, row, col);
    if (mask[index] || visited[index]) {
      return;
    }
    visited[index] = 1;
    pending.push_back(static_cast<int>(index));
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
  size_t pending_head = 0;
  while (pending_head < pending.size()) {
    const int flat_index_value = pending[pending_head++];
    const int row = flat_index_value / cols;
    const int col = flat_index_value % cols;
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

std::vector<uint8_t> fill_nearly_continuous_time_gaps(const std::vector<uint8_t>& mask,
                                                      int rows,
                                                      int cols,
                                                      int max_gap_px,
                                                      float min_continuity_ratio = 0.85f) {
  std::vector<uint8_t> output = mask;
  max_gap_px = std::max(0, max_gap_px);
  min_continuity_ratio = clamp_value(min_continuity_ratio, 0.0f, 1.0f);
  if (max_gap_px <= 0) {
    return output;
  }

  for (int row = 0; row < rows; ++row) {
    std::vector<int> active_cols;
    for (int col = 0; col < cols; ++col) {
      if (output[flat_index(cols, row, col)] != 0) {
        active_cols.push_back(col);
      }
    }
    if (active_cols.size() < 2) {
      continue;
    }

    std::vector<int> run_starts;
    std::vector<int> run_stops;
    run_starts.push_back(active_cols.front());
    int previous = active_cols.front();
    for (size_t index = 1; index < active_cols.size(); ++index) {
      const int current = active_cols[index];
      if (current != previous + 1) {
        run_stops.push_back(previous + 1);
        run_starts.push_back(current);
      }
      previous = current;
    }
    run_stops.push_back(previous + 1);

    for (size_t run_index = 0; run_index + 1 < run_starts.size(); ++run_index) {
      const int left_start = run_starts[run_index];
      const int left_stop = run_stops[run_index];
      const int right_start = run_starts[run_index + 1];
      const int right_stop = run_stops[run_index + 1];
      const int gap_width = right_start - left_stop;
      if (gap_width <= 0 || gap_width > max_gap_px) {
        continue;
      }
      const int left_width = left_stop - left_start;
      const int right_width = right_stop - right_start;
      const float continuity_ratio = static_cast<float>(left_width + right_width) /
                                     static_cast<float>(std::max(1, left_width + gap_width + right_width));
      if (continuity_ratio >= min_continuity_ratio) {
        for (int fill_col = left_stop; fill_col < right_start; ++fill_col) {
          output[flat_index(cols, row, fill_col)] = 1;
        }
      }
    }
  }
  return output;
}

std::vector<float> frontend_corrected_db(const std::vector<float>& power_db,
                                         int rows,
                                         int cols,
                                         double smooth_sigma,
                                         double reference_q,
                                         double max_boost_db,
                                         float& reference_level_out) {
  std::vector<float> row_mean(static_cast<size_t>(rows), 0.0f);
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += power_db[flat_index(cols, row, col)];
    }
    row_mean[static_cast<size_t>(row)] = sum / static_cast<float>(std::max(cols, 1));
  }
  const float sigma = static_cast<float>(std::max(smooth_sigma, 1.0));
  const int radius = std::max(1, static_cast<int>(std::ceil(sigma * 1.5f)));
  const auto row_smooth = gaussian_smooth_rows(row_mean, rows, radius, sigma);

  float sum = 0.0f;
  float max_value = -1.0e30f;
  for (float value : row_smooth) {
    sum += value;
    max_value = std::max(max_value, value);
  }
  const float mean_value = sum / static_cast<float>(std::max(rows, 1));
  const float quantile = static_cast<float>(reference_q / 100.0);
  const float blend = clamp_value((quantile - 0.5f) / 0.5f, 0.0f, 1.0f);
  reference_level_out = mean_value + blend * (max_value - mean_value);

  std::vector<float> corrected(power_db.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const float boost = std::min(std::max(reference_level_out - row_smooth[static_cast<size_t>(row)], 0.0f),
                                 static_cast<float>(max_boost_db));
    for (int col = 0; col < cols; ++col) {
      corrected[flat_index(cols, row, col)] = power_db[flat_index(cols, row, col)] + boost;
    }
  }
  return corrected;
}

std::vector<float> slice_rows(const std::vector<float>& input,
                              int rows,
                              int cols,
                              int row_start,
                              int row_stop) {
  row_start = clamp_value(row_start, 0, rows);
  row_stop = clamp_value(row_stop, row_start, rows);
  const int out_rows = row_stop - row_start;
  std::vector<float> output(static_cast<size_t>(out_rows) * static_cast<size_t>(cols), 0.0f);
  for (int row = 0; row < out_rows; ++row) {
    const size_t src_offset = flat_index(cols, row_start + row, 0);
    const size_t dst_offset = flat_index(cols, row, 0);
    std::memcpy(output.data() + dst_offset, input.data() + src_offset, static_cast<size_t>(cols) * sizeof(float));
  }
  return output;
}

std::vector<uint8_t> expand_row_valid_mask(const std::vector<uint8_t>& src_valid_rows,
                                           int cols) {
  std::vector<uint8_t> output(static_cast<size_t>(src_valid_rows.size()) * static_cast<size_t>(std::max(cols, 0)), 0);
  if (cols <= 0) {
    return output;
  }
  for (int row = 0; row < static_cast<int>(src_valid_rows.size()); ++row) {
    if (src_valid_rows[static_cast<size_t>(row)] == 0) {
      continue;
    }
    const size_t offset = static_cast<size_t>(row) * static_cast<size_t>(cols);
    std::fill(output.begin() + static_cast<std::ptrdiff_t>(offset),
              output.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cols)),
              static_cast<uint8_t>(1));
  }
  return output;
}

std::vector<uint8_t> compute_ignore_sideband_rows(int num_rows,
                                                  double bin_hz,
                                                  double ignore_sideband_hz,
                                                  int min_keep_rows,
                                                  int* applied_bins) {
  std::vector<uint8_t> valid_row_mask(static_cast<size_t>(std::max(num_rows, 0)), 1);
  if (applied_bins != nullptr) {
    *applied_bins = 0;
  }
  if (num_rows < 2 || !std::isfinite(bin_hz) || bin_hz <= 0.0 || !std::isfinite(ignore_sideband_hz) || ignore_sideband_hz <= 0.0) {
    return valid_row_mask;
  }

  const int max_bins = std::max(0, (num_rows - std::max(1, min_keep_rows)) / 2);
  const int requested_bins = static_cast<int>(std::ceil(ignore_sideband_hz / bin_hz));
  const int clipped_bins = clamp_value(requested_bins, 0, max_bins);
  if (applied_bins != nullptr) {
    *applied_bins = clipped_bins;
  }
  if (clipped_bins > 0) {
    std::fill(valid_row_mask.begin(), valid_row_mask.begin() + clipped_bins, static_cast<uint8_t>(0));
    std::fill(valid_row_mask.end() - clipped_bins, valid_row_mask.end(), static_cast<uint8_t>(0));
  }
  return valid_row_mask;
}

std::vector<double> build_frequency_axis_hz(int num_rows, double resolution_hz) {
  std::vector<double> axis(static_cast<size_t>(std::max(num_rows, 0)), 0.0);
  const bool calibrated = std::isfinite(resolution_hz) && resolution_hz > 0.0;
  for (int row = 0; row < num_rows; ++row) {
    axis[static_cast<size_t>(row)] = calibrated ? static_cast<double>(row) * resolution_hz : static_cast<double>(row);
  }
  return axis;
}

std::vector<ChunkPlanEntry> build_frequency_chunks(const std::vector<double>& freq_axis_hz,
                                                   double chunk_bandwidth_hz,
                                                   double chunk_overlap_hz,
                                                   int min_rows,
                                                   const std::vector<uint8_t>& valid_row_mask,
                                                   double uncalibrated_chunk_fraction,
                                                   double uncalibrated_overlap_fraction) {
  std::vector<ChunkPlanEntry> chunks;
  if (freq_axis_hz.empty() || valid_row_mask.size() != freq_axis_hz.size()) {
    return chunks;
  }

  std::vector<int> valid_idx;
  valid_idx.reserve(valid_row_mask.size());
  for (size_t index = 0; index < valid_row_mask.size(); ++index) {
    if (valid_row_mask[index] != 0) {
      valid_idx.push_back(static_cast<int>(index));
    }
  }
  if (valid_idx.empty()) {
    return chunks;
  }
  if (!(std::isfinite(chunk_bandwidth_hz) && chunk_bandwidth_hz > 0.0)) {
    throw std::runtime_error("chunk_bandwidth_hz must be positive");
  }
  const double step_hz = chunk_bandwidth_hz - chunk_overlap_hz;
  if (!(std::isfinite(step_hz) && step_hz > 0.0)) {
    throw std::runtime_error("chunk_bandwidth_hz must be larger than chunk_overlap_hz");
  }

  double freq_min = freq_axis_hz[static_cast<size_t>(valid_idx.front())];
  double freq_max = freq_axis_hz[static_cast<size_t>(valid_idx.front())];
  for (int index : valid_idx) {
    freq_min = std::min(freq_min, freq_axis_hz[static_cast<size_t>(index)]);
    freq_max = std::max(freq_max, freq_axis_hz[static_cast<size_t>(index)]);
  }
  const double freq_span = freq_max - freq_min;

  if (!(std::isfinite(freq_span)) || freq_span <= 0.0 || chunk_bandwidth_hz >= freq_span) {
    const int valid_count = static_cast<int>(valid_idx.size());
    const double chunk_fraction = clamp_value(uncalibrated_chunk_fraction, 0.10, 1.0);
    const double overlap_fraction = clamp_value(uncalibrated_overlap_fraction, 0.0, 0.95);
    const int chunk_rows = clamp_value(static_cast<int>(std::llround(static_cast<double>(valid_count) * chunk_fraction)),
                                       min_rows,
                                       valid_count);
    if (chunk_rows >= valid_count) {
      chunks.push_back(ChunkPlanEntry{0,
                                      valid_idx.front(),
                                      valid_idx.back() + 1,
                                      freq_axis_hz[static_cast<size_t>(valid_idx.front())],
                                      freq_axis_hz[static_cast<size_t>(valid_idx.back())]});
      return chunks;
    }

    const int overlap_rows = clamp_value(static_cast<int>(std::llround(static_cast<double>(chunk_rows) * overlap_fraction)),
                                         0,
                                         chunk_rows - 1);
    const int step_rows = std::max(1, chunk_rows - overlap_rows);
    int chunk_index = 0;
    for (int start_pos = 0; start_pos < valid_count; start_pos += step_rows) {
      const int stop_pos = std::min(start_pos + chunk_rows, valid_count);
      if ((stop_pos - start_pos) < min_rows) {
        if (stop_pos >= valid_count) {
          break;
        }
        continue;
      }
      chunks.push_back(ChunkPlanEntry{chunk_index++,
                                      valid_idx[static_cast<size_t>(start_pos)],
                                      valid_idx[static_cast<size_t>(stop_pos - 1)] + 1,
                                      freq_axis_hz[static_cast<size_t>(valid_idx[static_cast<size_t>(start_pos)])],
                                      freq_axis_hz[static_cast<size_t>(valid_idx[static_cast<size_t>(stop_pos - 1)])]});
      if (stop_pos >= valid_count) {
        break;
      }
    }
    return chunks;
  }

  int chunk_index = 0;
  for (double chunk_start_hz = freq_min; chunk_start_hz < freq_max + 1.0e-6; chunk_start_hz += step_hz) {
    const double chunk_stop_hz = std::min(chunk_start_hz + chunk_bandwidth_hz, freq_max);
    int row_start = -1;
    int row_stop = -1;
    for (int index : valid_idx) {
      const double value = freq_axis_hz[static_cast<size_t>(index)];
      if (value >= chunk_start_hz && value <= chunk_stop_hz) {
        if (row_start < 0) {
          row_start = index;
        }
        row_stop = index + 1;
      }
    }
    if (row_start >= 0 && row_stop > row_start && (row_stop - row_start) >= min_rows) {
      chunks.push_back(ChunkPlanEntry{chunk_index++,
                                      row_start,
                                      row_stop,
                                      freq_axis_hz[static_cast<size_t>(row_start)],
                                      freq_axis_hz[static_cast<size_t>(row_stop - 1)]});
    }
    if (chunk_stop_hz >= freq_max) {
      break;
    }
  }
  return chunks;
}

std::vector<float> structure_tensor_gate(const std::vector<float>& corrected,
                                         int rows,
                                         int cols,
                                         const std::vector<uint8_t>& valid_mask) {
  const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
  const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
  const auto background = box_mean_2d(corrected,
                                      rows,
                                      cols,
                                      std::max(1, bg_freq / 2),
                                      std::max(1, bg_time / 2));

  std::vector<float> residual_db(corrected.size(), 0.0f);
  for (size_t index = 0; index < residual_db.size(); ++index) {
    residual_db[index] = std::max(corrected[index] - background[index], 0.0f);
  }
  const auto residual_n = normalize01_quantile(residual_db, 5.0, 99.0);

  const std::array<double, 3> scales = {0.8, 1.6, 3.2};
  std::vector<float> gate_max(corrected.size(), 0.0f);
  for (double grad_sigma : scales) {
    const double integ_sigma = std::max(1.0, 1.8 * grad_sigma);
    const auto grad_f = gaussian_first_derivative_rows(residual_n, rows, cols, grad_sigma);
    const auto grad_t = gaussian_first_derivative_cols(residual_n, rows, cols, grad_sigma);

    std::vector<float> grad_ff(corrected.size(), 0.0f);
    std::vector<float> grad_ft(corrected.size(), 0.0f);
    std::vector<float> grad_tt(corrected.size(), 0.0f);
    for (size_t index = 0; index < corrected.size(); ++index) {
      grad_ff[index] = grad_f[index] * grad_f[index];
      grad_ft[index] = grad_f[index] * grad_t[index];
      grad_tt[index] = grad_t[index] * grad_t[index];
    }

    const auto j_ff = gaussian_blur(grad_ff, rows, cols, integ_sigma, integ_sigma);
    const auto j_ft = gaussian_blur(grad_ft, rows, cols, integ_sigma, integ_sigma);
    const auto j_tt = gaussian_blur(grad_tt, rows, cols, integ_sigma, integ_sigma);

    std::vector<float> coherence(corrected.size(), 0.0f);
    std::vector<float> energy(corrected.size(), 0.0f);
    for (size_t index = 0; index < corrected.size(); ++index) {
      const float delta = std::sqrt(std::max((j_ff[index] - j_tt[index]) * (j_ff[index] - j_tt[index]) +
                                                4.0f * (j_ft[index] * j_ft[index]),
                                            0.0f));
      const float lambda1 = 0.5f * (j_ff[index] + j_tt[index] + delta);
      const float lambda2 = 0.5f * (j_ff[index] + j_tt[index] - delta);
      coherence[index] = (lambda1 - lambda2) / std::max(lambda1 + lambda2, 1.0e-6f);
      energy[index] = lambda1 + lambda2;
    }

    const auto coherence_n = normalize01_quantile(coherence, 5.0, 99.0);
    const auto energy_n = normalize01_quantile(energy, 5.0, 99.0);
    for (size_t index = 0; index < gate_max.size(); ++index) {
      const float gate_value = coherence_n[index] * std::sqrt(std::max(energy_n[index], 0.0f));
      gate_max[index] = std::max(gate_max[index], gate_value);
    }
  }

  auto gate_px = normalize01_quantile(gate_max, 5.0, 99.0);
  for (size_t index = 0; index < gate_px.size() && index < valid_mask.size(); ++index) {
    if (valid_mask[index] == 0) {
      gate_px[index] = 0.0f;
    }
  }
  return gate_px;
}

int component_envelope_area(const std::vector<uint8_t>& mask, int rows, int cols) {
  int area = 0;
  for (int col = 0; col < cols; ++col) {
    int min_row = rows;
    int max_row = -1;
    for (int row = 0; row < rows; ++row) {
      if (mask[flat_index(cols, row, col)] == 0) {
        continue;
      }
      min_row = std::min(min_row, row);
      max_row = std::max(max_row, row);
    }
    if (max_row >= min_row) {
      area += (max_row - min_row + 1);
    }
  }
  return area;
}

GroupingResult group_mask_regions(const std::vector<uint8_t>& mask,
                                  const std::vector<float>& score_map,
                                  const std::vector<uint8_t>& valid_mask,
                                  int rows,
                                  int cols,
                                  bool filter_detection_mask,
                                  int bridge_freq_px,
                                  int bridge_time_px,
                                  int min_component_size,
                                  int min_freq_span_px,
                                  int min_time_span_px,
                                  float min_density,
                                  float time_continuity_ratio) {
  GroupingResult result;
  if (rows <= 0 || cols <= 0 || mask.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
    return result;
  }

  result.seed_mask = mask;
  for (size_t index = 0; index < result.seed_mask.size() && index < valid_mask.size(); ++index) {
    if (valid_mask[index] == 0) {
      result.seed_mask[index] = 0;
    }
  }

  if (!filter_detection_mask) {
    result.bridged_mask = result.seed_mask;
    const auto labelled = label_components(result.bridged_mask, rows, cols);
    result.component_labels.assign(labelled.labels.begin(), labelled.labels.end());
    result.grouped_mask = result.seed_mask;
    for (size_t label_index = 0; label_index < labelled.sizes.size(); ++label_index) {
      const int component_id = static_cast<int>(label_index) + 1;
      int min_row = rows;
      int max_row = -1;
      int min_col = cols;
      int max_col = -1;
      int filled_area = 0;
      float score_peak = 0.0f;
      float score_sum = 0.0f;
      for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
          const size_t flat = flat_index(cols, row, col);
          if (labelled.labels[flat] != component_id) {
            continue;
          }
          min_row = std::min(min_row, row);
          max_row = std::max(max_row, row);
          min_col = std::min(min_col, col);
          max_col = std::max(max_col, col);
          ++filled_area;
          if (flat < score_map.size()) {
            const float score = score_map[flat];
            score_sum += score;
            score_peak = std::max(score_peak, score);
          }
        }
      }
      if (filled_area <= 0 || max_row < min_row || max_col < min_col) {
        continue;
      }
      DetectionBox box;
      box.freq_start = min_row;
      box.freq_stop = max_row + 1;
      box.time_start = min_col;
      box.time_stop = max_col + 1;
      box.filled_area = filled_area;
      const int bbox_area = std::max(1, (box.freq_stop - box.freq_start) * (box.time_stop - box.time_start));
      box.bbox_density = static_cast<float>(filled_area) / static_cast<float>(bbox_area);
      box.envelope_density = box.bbox_density;
      box.density = box.bbox_density;
      box.score_mean = filled_area > 0 ? score_sum / static_cast<float>(filled_area) : 0.0f;
      box.score_peak = score_peak;
      result.boxes.push_back(std::move(box));
    }
    return result;
  }

  result.bridged_mask = result.seed_mask;
  if (bridge_freq_px > 1 || bridge_time_px > 1) {
    result.bridged_mask = binary_closing_rect(result.bridged_mask,
                                              rows,
                                              cols,
                                              std::max(1, bridge_freq_px),
                                              std::max(1, bridge_time_px));
  }
  result.bridged_mask = fill_nearly_continuous_time_gaps(result.bridged_mask, rows, cols, bridge_time_px, time_continuity_ratio);

  const auto labelled = label_components(result.bridged_mask, rows, cols);
  result.component_labels.assign(labelled.labels.begin(), labelled.labels.end());
  std::vector<float> active_scores;
  active_scores.reserve(score_map.size());
  for (size_t index = 0; index < score_map.size() && index < result.seed_mask.size(); ++index) {
    if (result.seed_mask[index] != 0) {
      active_scores.push_back(score_map[index]);
    }
  }
  result.peak_score_floor = quantile_from_values(active_scores, 0.50, 0.0f);
  result.grouped_mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);

  for (size_t label_index = 0; label_index < labelled.sizes.size(); ++label_index) {
    const int component_id = static_cast<int>(label_index) + 1;
    int min_row = rows;
    int max_row = -1;
    int min_col = cols;
    int max_col = -1;
    int filled_area = 0;
    std::vector<float> component_scores;
    std::vector<uint8_t> component_local_mask;

    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const size_t flat = flat_index(cols, row, col);
        if (labelled.labels[flat] != component_id) {
          continue;
        }
        min_row = std::min(min_row, row);
        max_row = std::max(max_row, row);
        min_col = std::min(min_col, col);
        max_col = std::max(max_col, col);
        ++filled_area;
      }
    }
    if (filled_area <= 0 || max_row < min_row || max_col < min_col) {
      continue;
    }

    const int freq_start = min_row;
    const int freq_stop = max_row + 1;
    const int time_start = min_col;
    const int time_stop = max_col + 1;
    const int freq_span = freq_stop - freq_start;
    const int time_span = time_stop - time_start;
    const int local_rows = freq_span;
    const int local_cols = time_span;
    component_local_mask.assign(static_cast<size_t>(local_rows) * static_cast<size_t>(local_cols), 0);
    for (int row = freq_start; row < freq_stop; ++row) {
      for (int col = time_start; col < time_stop; ++col) {
        const size_t src_flat = flat_index(cols, row, col);
        if (labelled.labels[src_flat] != component_id) {
          continue;
        }
        component_local_mask[flat_index(local_cols, row - freq_start, col - time_start)] = 1;
        if (src_flat < score_map.size()) {
          component_scores.push_back(score_map[src_flat]);
        }
      }
    }

    const int bbox_area = std::max(1, freq_span * time_span);
    const int envelope_area = std::max(1, component_envelope_area(component_local_mask, local_rows, local_cols));
    const float bbox_density = static_cast<float>(filled_area) / static_cast<float>(bbox_area);
    const float envelope_density = static_cast<float>(filled_area) / static_cast<float>(envelope_area);
    const float density = envelope_density;
    const float score_peak = component_scores.empty() ? 0.0f : *std::max_element(component_scores.begin(), component_scores.end());
    float score_sum = 0.0f;
    for (float score : component_scores) {
      score_sum += score;
    }
    const float score_mean = component_scores.empty() ? 0.0f : score_sum / static_cast<float>(component_scores.size());

    const bool keep = filled_area >= std::max(1, min_component_size) &&
                      freq_span >= std::max(1, min_freq_span_px) &&
                      time_span >= std::max(1, min_time_span_px) &&
                      density >= min_density &&
                      score_peak >= result.peak_score_floor;
    if (!keep) {
      continue;
    }

    for (int row = freq_start; row < freq_stop; ++row) {
      for (int col = time_start; col < time_stop; ++col) {
        const size_t src_flat = flat_index(cols, row, col);
        if (labelled.labels[src_flat] == component_id) {
          result.grouped_mask[src_flat] = 1;
        }
      }
    }
    DetectionBox box;
    box.freq_start = freq_start;
    box.freq_stop = freq_stop;
    box.time_start = time_start;
    box.time_stop = time_stop;
    box.filled_area = filled_area;
    box.density = density;
    box.bbox_density = bbox_density;
    box.envelope_density = envelope_density;
    box.score_mean = score_mean;
    box.score_peak = score_peak;
    result.boxes.push_back(std::move(box));
  }
  return result;
}

void accumulate_chunk_grouped_result(const std::vector<uint8_t>& chunk_grouped_mask,
                                     const std::vector<float>& chunk_combined_score,
                                     int chunk_output_rows,
                                     int chunk_output_cols,
                                     int chunk_source_rows,
                                     int chunk_source_cols,
                                     int global_row_start,
                                     std::vector<uint8_t>& projected_mask,
                                     std::vector<float>& projected_score_sum,
                                     std::vector<float>& projected_score_weight,
                                     int global_rows,
                                     int global_cols) {
  for (int row = 0; row < chunk_output_rows; ++row) {
    const int global_row_span_start = global_row_start + static_cast<int>((static_cast<int64_t>(row) * static_cast<int64_t>(chunk_source_rows)) /
                                                                          static_cast<int64_t>(std::max(chunk_output_rows, 1)));
    const int global_row_span_stop = global_row_start + static_cast<int>((static_cast<int64_t>(row + 1) * static_cast<int64_t>(chunk_source_rows)) /
                                                                         static_cast<int64_t>(std::max(chunk_output_rows, 1)));
    const int clamped_row_start = clamp_value(global_row_span_start, 0, global_rows);
    const int clamped_row_stop = clamp_value(std::max(global_row_span_stop, global_row_span_start + 1), 0, global_rows);
    if (clamped_row_start >= clamped_row_stop) {
      continue;
    }
    for (int col = 0; col < chunk_output_cols; ++col) {
      const size_t chunk_flat = flat_index(chunk_output_cols, row, col);
      if (chunk_flat >= chunk_grouped_mask.size() || chunk_flat >= chunk_combined_score.size() || chunk_grouped_mask[chunk_flat] == 0) {
        continue;
      }
      const int global_col_span_start = static_cast<int>((static_cast<int64_t>(col) * static_cast<int64_t>(chunk_source_cols)) /
                                                         static_cast<int64_t>(std::max(chunk_output_cols, 1)));
      const int global_col_span_stop = static_cast<int>((static_cast<int64_t>(col + 1) * static_cast<int64_t>(chunk_source_cols)) /
                                                        static_cast<int64_t>(std::max(chunk_output_cols, 1)));
      const int clamped_col_start = clamp_value(global_col_span_start, 0, global_cols);
      const int clamped_col_stop = clamp_value(std::max(global_col_span_stop, global_col_span_start + 1), 0, global_cols);
      if (clamped_col_start >= clamped_col_stop) {
        continue;
      }
      for (int global_row = clamped_row_start; global_row < clamped_row_stop; ++global_row) {
        for (int global_col = clamped_col_start; global_col < clamped_col_stop; ++global_col) {
          const size_t global_flat = flat_index(global_cols, global_row, global_col);
          projected_mask[global_flat] = 1;
          projected_score_sum[global_flat] += chunk_combined_score[chunk_flat];
          projected_score_weight[global_flat] += 1.0f;
        }
      }
    }
  }
}

std::vector<uint8_t> binary_propagation(const std::vector<uint8_t>& seed,
                                        const std::vector<uint8_t>& grow_mask,
                                        int rows,
                                        int cols) {
  std::vector<uint8_t> output(seed.size(), 0);
  std::vector<int> pending;
  pending.reserve(seed.size());
  constexpr std::array<int, 8> d_row = {-1, -1, -1, 0, 0, 1, 1, 1};
  constexpr std::array<int, 8> d_col = {-1, 0, 1, -1, 1, -1, 0, 1};

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t index = flat_index(cols, row, col);
      if (!seed[index] || !grow_mask[index]) {
        continue;
      }
      output[index] = 1;
      pending.push_back(static_cast<int>(index));
    }
  }

  size_t pending_head = 0;
  while (pending_head < pending.size()) {
    const int flat_index_value = pending[pending_head++];
    const int row = flat_index_value / cols;
    const int col = flat_index_value % cols;
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
      pending.push_back(static_cast<int>(next_flat));
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

HybridPostprocessResult finalize_residual_veto_hybrid(const std::vector<uint8_t>& seed_mask_input,
                                                      const std::vector<uint8_t>& grow_mask_input,
                                                      const std::vector<uint8_t>& combined_gate_mask,
                                                      const std::vector<uint8_t>& valid_mask,
                                                      int rows,
                                                      int cols,
                                                      float seed_freq_threshold,
                                                      float seed_res_threshold,
                                                      float grow_freq_threshold,
                                                      float grow_res_threshold,
                                                      float combined_threshold) {
  HybridPostprocessResult result;
  result.mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  result.seed_freq_threshold = seed_freq_threshold;
  result.seed_res_threshold = seed_res_threshold;
  result.grow_freq_threshold = grow_freq_threshold;
  result.grow_res_threshold = grow_res_threshold;
  result.combined_threshold = combined_threshold;
  if (seed_mask_input.size() != result.mask.size() || valid_mask.size() != result.mask.size()) {
    return result;
  }

  (void)grow_mask_input;
  (void)combined_gate_mask;

  auto seed_mask = keep_large_components(seed_mask_input, rows, cols, 8);
  std::vector<uint8_t> final_mask = std::move(seed_mask);
  for (size_t index = 0; index < final_mask.size(); ++index) {
    final_mask[index] = (final_mask[index] && valid_mask[index]) ? 1 : 0;
  }
  final_mask = keep_large_components(final_mask, rows, cols, 8, &result.component_count);

  result.final_fraction = mean_mask_value(final_mask);
  result.connected_fraction = connected_fraction(final_mask, valid_mask);
  result.mask = std::move(final_mask);
  return result;
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
  result.combined_score = combined_score;

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
  result.grow_freq_threshold = result.seed_freq_threshold;
  result.grow_res_threshold = result.seed_res_threshold;
  result.combined_threshold = quantile_from_values(active_combined, 0.78, 1.0f);

  std::vector<uint8_t> seed_mask(base_norm.size(), 0);
  for (size_t index = 0; index < seed_mask.size(); ++index) {
    seed_mask[index] = (valid_mask[index] &&
                        keep_freq[index] >= result.seed_freq_threshold &&
                        keep_res[index] >= result.seed_res_threshold)
                           ? 1
                           : 0;
  }

  std::vector<uint8_t> final_mask(seed_mask.size(), 0);
  for (size_t index = 0; index < final_mask.size(); ++index) {
    final_mask[index] = (seed_mask[index] &&
                         valid_mask[index] &&
                         combined_score[index] >= result.combined_threshold * 0.85f)
                            ? 1
                            : 0;
  }

  final_mask = binary_closing_rect(final_mask, rows, cols, 7, 3);
  final_mask = binary_fill_holes(final_mask, rows, cols);
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

namespace {

constexpr std::array<const char*, 7> kRuntimeStageNames = {
  "frontend",
  "crop_align",
  "resize",
  "model_prep",
  "torch_forward",
  "dino_score",
  "fusion",
};

constexpr std::array<const char*, holoscan::ops::DinoV3SignalDetector::kReferenceStageCount>
    kReferenceStageNames = {
        "host_copy",
        "host_frontend",
        "chunk_plan",
        "chunk_upload",
        "score_project",
        "coherence_hybrid",
        "chunk_group",
        "global_merge",
    };

uint64_t steady_time_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

double elapsed_ms(uint64_t start_ns, uint64_t end_ns) {
  if (start_ns == 0 || end_ns <= start_ns) {
    return 0.0;
  }
  return static_cast<double>(end_ns - start_ns) / 1.0e6;
}

}  // namespace

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
    if (buffers.analysis_ready_event != nullptr) {
      cudaEventDestroy(buffers.analysis_ready_event);
    }
    if (buffers.coherence_gate_ready_event != nullptr) {
      cudaEventDestroy(buffers.coherence_gate_ready_event);
    }
    if (buffers.processing_stream != nullptr) {
      cudaStreamDestroy(buffers.processing_stream);
    }
    if (buffers.staging_stream != nullptr) {
      cudaStreamDestroy(buffers.staging_stream);
    }
    buffers = ChannelBuffers {};
  }
}

void DinoV3SignalDetector::setup(holoscan::OperatorSpec& spec) {
  auto& input_port = spec.input<dino_in_t>("in", holoscan::IOSpec::IOSize{8});
  input_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));

  const std::vector<double> imagenet_mean_default{0.485, 0.456, 0.406};
  const std::vector<double> imagenet_std_default{0.229, 0.224, 0.225};

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(input_height_, "input_height", "Input height", "Detector output height.", 256);
  spec.param(input_width_, "input_width", "Input width", "Detector output width.", 512);
  spec.param(patch_size_, "patch_size", "Patch size", "Patch size used for DINO-aligned input shaping.", 16);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one output every N input frames per channel.", 1);
  spec.param(mask_threshold_db_, "mask_threshold_db", "Mask threshold (legacy)", "Legacy placeholder threshold retained for compatibility.", -20.0f);
  spec.param(channel_filter_,
             "channel_filter",
             "Channel filter",
             "Optional channel filter. When >= 0, process only that channel in this operator instance.",
             -1);
  spec.param(log_detections_, "log_detections", "Log detections", "If true, logs detector execution details.", false);
  spec.param(backend_mode_,
             "backend_mode",
             "Backend mode",
             "Detector backend mode. The validated live path requires reference.",
             std::string("reference"));
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
  spec.param(torch_dtype_, "torch_dtype", "Torch dtype", "Torch inference precision: fp32 or fp16.", std::string("fp32"));
  spec.param(strict_model_forward_, "strict_model_forward", "Strict model forward", "If true, drop frames when the DINO runtime fails instead of falling back.", false);
  spec.param(imagenet_mean_, "imagenet_mean", "ImageNet mean", "Mean used for notebook-aligned model normalization.", imagenet_mean_default);
  spec.param(imagenet_std_, "imagenet_std", "ImageNet std", "Standard deviation used for notebook-aligned model normalization.", imagenet_std_default);
  spec.param(fft_size_, "fft_size", "FFT size", "Notebook-derived FFT size constant for metadata and parity tracking.", 1024);
  spec.param(noverlap_, "noverlap", "FFT overlap", "Notebook-derived overlap constant for parity tracking.", 256);
  spec.param(chunk_bandwidth_hz_, "chunk_bandwidth_hz", "Chunk bandwidth", "Chunk bandwidth in Hz for validated wideband subsection planning.", 25000000.0);
  spec.param(chunk_overlap_hz_, "chunk_overlap_hz", "Chunk overlap", "Chunk overlap in Hz for validated wideband subsection planning.", 6250000.0);
  spec.param(uncalibrated_chunk_fraction_, "uncalibrated_chunk_fraction", "Uncalibrated chunk fraction", "Fallback chunk fraction when the frequency axis is not calibrated.", 0.40);
  spec.param(uncalibrated_overlap_fraction_, "uncalibrated_overlap_fraction", "Uncalibrated overlap fraction", "Fallback chunk overlap fraction when the frequency axis is not calibrated.", 0.20);
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
  spec.param(filter_detection_mask_, "filter_detection_mask", "Filter detection mask", "When true, apply the validated grouping and filtering path before boxing.", true);
  spec.param(grouping_bridge_freq_px_, "grouping_bridge_freq_px", "Grouping bridge frequency px", "Frequency-axis bridge size for validated region grouping.", 33);
  spec.param(grouping_bridge_time_px_, "grouping_bridge_time_px", "Grouping bridge time px", "Time-axis bridge size for validated region grouping.", 5);
  spec.param(grouping_min_component_size_, "grouping_min_component_size", "Grouping minimum component size", "Minimum grouped component area for validated region grouping.", 24);
  spec.param(grouping_min_freq_span_px_, "grouping_min_freq_span_px", "Grouping minimum frequency span", "Minimum grouped component frequency span in pixels.", 18);
  spec.param(grouping_min_time_span_px_, "grouping_min_time_span_px", "Grouping minimum time span", "Minimum grouped component time span in pixels.", 2);
  spec.param(grouping_min_density_, "grouping_min_density", "Grouping minimum density", "Minimum grouped component density.", 0.06);
  spec.param(grouping_time_continuity_ratio_, "grouping_time_continuity_ratio", "Grouping time continuity ratio", "Minimum continuity ratio for validated time-gap filling.", 0.85);
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

  const int configured_channels = std::max(1, num_channels_.get());
  const int channel_filter = channel_filter_.get();
  const size_t local_channel_count = channel_filter >= 0 ? 1u : static_cast<size_t>(configured_channels);

  frame_count_.assign(local_channel_count, 0);
  masks_saved_.assign(local_channel_count, 0);
  timing_stats_.assign(local_channel_count, ChannelTimingStats {});
  ingress_stats_.assign(local_channel_count, ChannelIngressStats {});
  service_stats_.assign(local_channel_count, ChannelServiceStats {});
  channel_buffers_.assign(local_channel_count, ChannelBuffers {});

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
    if (inference_backend_.get() == "torchscript") {
      DinoTorchRuntimeConfig runtime_config;
      runtime_config.inference_backend = inference_backend_.get();
      runtime_config.model_script_path = model_script_path_.get();
      runtime_config.torchscript_init_mode = torchscript_init_mode_.get();
      runtime_config.torch_dtype = torch_dtype_.get();
      runtime_config.imagenet_mean = imagenet_mean_.get();
      runtime_config.imagenet_std = imagenet_std_.get();
      runtime_config.return_final_mask = false;
      runtime_config.return_final_mask_device = false;
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

      torch_runtime_->warmup(runtime_config,
                            std::max(1, input_height_.get()),
                            std::max(1, input_width_.get()),
                            std::max(1, patch_size_.get()));
    }
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

void DinoV3SignalDetector::stop() {
  for (size_t channel_index = 0; channel_index < ingress_stats_.size(); ++channel_index) {
    const auto& stats = ingress_stats_[channel_index];
    HOLOSCAN_LOG_INFO(
        "DINO ingress latency ch={} samples={} mean_chdr_to_dino_ms={:.3f} max_chdr_to_dino_ms={:.3f} mean_fft_to_dino_ms={:.3f} max_fft_to_dino_ms={:.3f}",
        channel_index,
        stats.samples,
        stats.samples == 0 ? 0.0 : stats.total_chdr_to_dino_ms / static_cast<double>(stats.samples),
        stats.max_chdr_to_dino_ms,
        stats.samples == 0 ? 0.0 : stats.total_fft_to_dino_ms / static_cast<double>(stats.samples),
        stats.max_fft_to_dino_ms);
  }
  for (size_t channel_index = 0; channel_index < service_stats_.size(); ++channel_index) {
    const auto& stats = service_stats_[channel_index];
    std::ostringstream summary;
    summary << "DINO service timing ch=" << channel_index
            << " samples=" << stats.samples
            << " mean_wall_ms=" << (stats.samples == 0 ? 0.0 : stats.total_wall_ms / static_cast<double>(stats.samples))
            << " max_wall_ms=" << stats.max_wall_ms
            << " mean_runtime_call_ms=" << (stats.samples == 0 ? 0.0 : stats.total_runtime_call_ms / static_cast<double>(stats.samples))
            << " max_runtime_call_ms=" << stats.max_runtime_call_ms
            << " mean_hybrid_call_ms=" << (stats.samples == 0 ? 0.0 : stats.total_hybrid_call_ms / static_cast<double>(stats.samples))
          << " max_hybrid_call_ms=" << stats.max_hybrid_call_ms
          << " mean_chunk_count=" << (stats.samples == 0 ? 0.0 : static_cast<double>(stats.total_chunk_count) / static_cast<double>(stats.samples))
          << " max_chunk_count=" << stats.max_chunk_count;
    for (size_t stage_index = 0; stage_index < kRuntimeStageNames.size(); ++stage_index) {
      const double mean_stage_ms = stats.samples == 0 ? 0.0 : stats.total_runtime_stage_ms[stage_index] / static_cast<double>(stats.samples);
      summary << " " << kRuntimeStageNames[stage_index] << "(mean=" << mean_stage_ms
              << ",max=" << stats.max_runtime_stage_ms[stage_index] << ")";
    }
        for (size_t stage_index = 0; stage_index < kReferenceStageNames.size(); ++stage_index) {
          const double mean_stage_ms = stats.samples == 0 ? 0.0 : stats.total_reference_stage_ms[stage_index] / static_cast<double>(stats.samples);
          summary << " " << kReferenceStageNames[stage_index] << "(mean=" << mean_stage_ms
            << ",max=" << stats.max_reference_stage_ms[stage_index] << ")";
        }
    HOLOSCAN_LOG_INFO(summary.str());
  }
  holoscan::Operator::stop();
}

void DinoV3SignalDetector::compute(holoscan::InputContext& op_input,
                                   holoscan::OutputContext&,
                                   holoscan::ExecutionContext&) {
  auto input = op_input.receive<dino_in_t>("in").value();
  auto& fft_tensor = std::get<0>(input);
  auto stream = std::get<1>(input);

  auto meta = metadata();
  const uint64_t dino_enter_ns = steady_time_ns();
  const uint16_t channel_number = meta->get<uint16_t>("channel_number", 0);
  const int channel_filter = channel_filter_.get();
  if (channel_filter >= 0 && channel_number != static_cast<uint16_t>(channel_filter)) {
    return;
  }

  const size_t local_channel_index = channel_filter >= 0 ? 0u : static_cast<size_t>(channel_number);
  if (local_channel_index >= frame_count_.size()) {
    HOLOSCAN_LOG_WARN("DINO hybrid detector received out-of-range channel {} (configured channels: {}).",
                      channel_number,
                      frame_count_.size());
    return;
  }

  {
    auto& ingress = ingress_stats_[local_channel_index];
    const uint64_t chdr_emit_ns = meta->get<uint64_t>("chdr_emit_ts_ns", 0);
    const uint64_t fft_emit_ns = meta->get<uint64_t>("fft_emit_ts_ns", 0);
    const double chdr_to_dino_ms = elapsed_ms(chdr_emit_ns, dino_enter_ns);
    const double fft_to_dino_ms = elapsed_ms(fft_emit_ns, dino_enter_ns);
    ingress.samples++;
    ingress.total_chdr_to_dino_ms += chdr_to_dino_ms;
    ingress.max_chdr_to_dino_ms = std::max(ingress.max_chdr_to_dino_ms, chdr_to_dino_ms);
    ingress.total_fft_to_dino_ms += fft_to_dino_ms;
    ingress.max_fft_to_dino_ms = std::max(ingress.max_fft_to_dino_ms, fft_to_dino_ms);
  }
  meta->set("dino_enter_ts_ns", dino_enter_ns);
  const auto compute_wall_start = std::chrono::steady_clock::now();

  const uint64_t frame_number = ++frame_count_[local_channel_index];
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
                                (masks_saved_[local_channel_index] < max_masks_per_channel_.get());
  const auto requested_backend_mode = backend_mode_.get();
  if (requested_backend_mode != "reference" && !backend_mode_warning_emitted_) {
    HOLOSCAN_LOG_WARN(
        "Unsupported or deprecated DINO backend_mode='{}'. Falling back to the validated reference path.",
        requested_backend_mode);
    backend_mode_warning_emitted_ = true;
  }
  const std::string effective_backend_mode = std::string("reference");
  std::array<double, kTimingStageCount> stage_ms {};
  const auto total_start = std::chrono::steady_clock::now();

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

  auto& buffers = channel_buffers_[local_channel_index];

  auto time_step_ms = [&](size_t stage_index, auto&& fn) {
    if (!timing_enabled) {
      fn();
      return;
    }

    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    const auto sync_stream = buffers.processing_stream != nullptr ? buffers.processing_stream : stream;
    const auto sync_result = cudaStreamSynchronize(sync_stream);
    if (sync_result != cudaSuccess) {
      throw std::runtime_error(std::string("timing synchronization failed at ") +
                               kTimingStageNames[stage_index] + ": " +
                               cudaGetErrorString(sync_result));
    }
    stage_ms[stage_index] = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - stage_start)
                                .count();
  };

  const auto work_stream = [&]() -> cudaStream_t {
    return buffers.processing_stream != nullptr ? buffers.processing_stream : stream;
  };

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
      if (buffers.processing_stream == nullptr) {
        const auto stream_result = cudaStreamCreateWithFlags(&buffers.processing_stream, cudaStreamNonBlocking);
        if (stream_result != cudaSuccess) {
          throw std::runtime_error(std::string("processing stream creation failed: ") +
                                   cudaGetErrorString(stream_result));
        }
      }
      if (buffers.analysis_ready_event == nullptr) {
        const auto event_result = cudaEventCreateWithFlags(&buffers.analysis_ready_event, cudaEventDisableTiming);
        if (event_result != cudaSuccess) {
          throw std::runtime_error(std::string("analysis ready event creation failed: ") +
                                   cudaGetErrorString(event_result));
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

      const auto ready_result = cudaEventRecord(buffers.analysis_ready_event, stream);
      if (ready_result != cudaSuccess) {
        throw std::runtime_error(std::string("analysis ready event record failed: ") + cudaGetErrorString(ready_result));
      }
      const auto wait_result = cudaStreamWaitEvent(buffers.processing_stream, buffers.analysis_ready_event, 0);
      if (wait_result != cudaSuccess) {
        throw std::runtime_error(std::string("processing stream wait failed: ") + cudaGetErrorString(wait_result));
      }
    });

    time_step_ms(kPowerDbStage, [&] {
      constexpr int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      dino_power_db_kernel<<<blocks, threads, 0, work_stream()>>>(buffers.analysis_tensor_device,
                                                            src_rows,
                                                            src_cols,
                                                            buffers.power_db_device);
      const auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("power_db kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
    });

    time_step_ms(kFrontendStage, [&] {
      constexpr int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      const int row_blocks = (src_rows + threads - 1) / threads;
      const int smooth_radius = std::max(1, static_cast<int>(std::ceil(std::max(frontend_correction_smooth_sigma_.get(), 1.0) * 1.5)));

      dino_row_mean_kernel<<<src_rows, threads, 0, work_stream()>>>(buffers.power_db_device,
                                                              src_rows,
                                                              src_cols,
                                                              buffers.row_stat_device);
      dino_gaussian_smooth_rows_kernel<<<row_blocks, threads, 0, work_stream()>>>(buffers.row_stat_device,
                                                                            src_rows,
                                                                            smooth_radius,
                                                                            static_cast<float>(std::max(frontend_correction_smooth_sigma_.get(), 1.0)),
                                                                            buffers.row_smooth_device);
      dino_frontend_reference_kernel<<<1, threads, 0, work_stream()>>>(buffers.row_smooth_device,
                                                                 src_rows,
                                                                 static_cast<float>(frontend_correction_reference_q_.get() / 100.0),
                                                                 buffers.frontend_reference_device);
      dino_frontend_correction_kernel<<<blocks, threads, 0, work_stream()>>>(buffers.power_db_device,
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
    });

    std::string backend_used = "reference_uninitialized";
    double runtime_call_wall_ms = 0.0;
    double hybrid_call_wall_ms = 0.0;
    HybridPostprocessResult hybrid_result;
    bool torchscript_forward_ready = false;
    holoscan::ops::DinoTorchRuntimeTiming aggregated_runtime_timing;
    std::array<double, DinoV3SignalDetector::kReferenceStageCount> reference_stage_ms {};
    int aligned_input_rows = 0;
    int aligned_input_cols = 0;
    double power_threshold = 0.0;
    int chunk_count = 0;

    time_step_ms(kTorchRuntimeStage, [&] {
    });

    time_step_ms(kHybridStage, [&] {
      const auto hybrid_call_start = std::chrono::steady_clock::now();
      if (!use_pytorch_backend_.get() || !torch_runtime_) {
        throw std::runtime_error("validated live reference path requires the DINO torch runtime");
      }

      const auto sync_result = cudaStreamSynchronize(work_stream());
      if (sync_result != cudaSuccess) {
        throw std::runtime_error(std::string("reference-path synchronization failed: ") + cudaGetErrorString(sync_result));
      }

      std::vector<float> power_db_host(frame_elements, 0.0f);
      {
        const auto stage_start = std::chrono::steady_clock::now();
        const auto copy_result = cudaMemcpy(power_db_host.data(),
                                            buffers.power_db_device,
                                            frame_elements * sizeof(float),
                                            cudaMemcpyDeviceToHost);
        if (copy_result != cudaSuccess) {
          throw std::runtime_error(std::string("failed to copy power_db to host for chunked live reference path: ") +
                                   cudaGetErrorString(copy_result));
        }
        reference_stage_ms[0] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
      }

      float frontend_reference_level = 0.0f;
      std::vector<float> corrected_db_host;
      {
        const auto stage_start = std::chrono::steady_clock::now();
        corrected_db_host = frontend_correction_enable_.get()
                                ? frontend_corrected_db(power_db_host,
                                                        src_rows,
                                                        src_cols,
                                                        frontend_correction_smooth_sigma_.get(),
                                                        frontend_correction_reference_q_.get(),
                                                        frontend_correction_max_boost_db_.get(),
                                                        frontend_reference_level)
                                : power_db_host;
        reference_stage_ms[1] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
      }
      (void)frontend_reference_level;

      const double chunk_bin_hz = (std::isfinite(resolution_hz) && resolution_hz > 0.0) ? resolution_hz : 1.0;
      std::vector<uint8_t> source_valid_rows;
      std::vector<ChunkPlanEntry> chunk_plan;
      {
        const auto stage_start = std::chrono::steady_clock::now();
        source_valid_rows = compute_ignore_sideband_rows(src_rows,
                                                         chunk_bin_hz,
                                                         ignore_sideband_hz_.get(),
                                                         16,
                                                         &ignore_bins_per_side);
        const auto source_freq_axis_hz = build_frequency_axis_hz(src_rows, resolution_hz);
        chunk_plan = build_frequency_chunks(source_freq_axis_hz,
                                            chunk_bandwidth_hz_.get(),
                                            chunk_overlap_hz_.get(),
                                            16,
                                            source_valid_rows,
                                            uncalibrated_chunk_fraction_.get(),
                                            uncalibrated_overlap_fraction_.get());
        reference_stage_ms[2] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
      }
      if (chunk_plan.empty()) {
        throw std::runtime_error("validated live reference path produced an empty chunk plan");
      }
      chunk_count = static_cast<int>(chunk_plan.size());

      std::vector<uint8_t> projected_grouped_mask(frame_elements, 0);
      std::vector<float> projected_grouped_score_sum(frame_elements, 0.0f);
      std::vector<float> projected_grouped_score_weight(frame_elements, 0.0f);

      DinoTorchRuntimeConfig runtime_config;
      runtime_config.inference_backend = inference_backend_.get();
      runtime_config.model_script_path = model_script_path_.get();
      runtime_config.torchscript_init_mode = torchscript_init_mode_.get();
      runtime_config.torch_dtype = torch_dtype_.get();
      runtime_config.imagenet_mean = imagenet_mean_.get();
      runtime_config.imagenet_std = imagenet_std_.get();
      runtime_config.return_final_mask = true;
      runtime_config.return_final_mask_device = false;
      runtime_config.return_pre_model_gray = false;
      runtime_config.return_patch_features = false;
      runtime_config.compute_dino_threshold = true;
      runtime_config.compute_power_score = false;
      runtime_config.ignore_sideband_hz = 0.0;
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
      runtime_config.dino_group_k = dino_group_k_.get();
      runtime_config.dino_group_spatial_weight = dino_group_spatial_weight_.get();
      runtime_config.dino_group_score_q = dino_group_score_q_.get();
      runtime_config.pipeline_final_threshold = pipeline_final_threshold_.get();
      runtime_config.pipeline_gap_floor = pipeline_gap_floor_.get();
      runtime_config.pipeline_power_rescue_floor = pipeline_power_rescue_floor_.get();
      runtime_config.pipeline_power_rescue_gain = pipeline_power_rescue_gain_.get();

      float seed_freq_sum = 0.0f;
      float seed_res_sum = 0.0f;
      float grow_freq_sum = 0.0f;
      float grow_res_sum = 0.0f;
      float combined_threshold_sum = 0.0f;

      for (const auto& chunk : chunk_plan) {
        const int chunk_rows = std::max(0, chunk.row_stop - chunk.row_start);
        if (chunk_rows <= 0) {
          continue;
        }

        const auto power_chunk = slice_rows(power_db_host, src_rows, src_cols, chunk.row_start, chunk.row_stop);
        const auto corrected_chunk = slice_rows(corrected_db_host, src_rows, src_cols, chunk.row_start, chunk.row_stop);
        std::vector<uint8_t> chunk_valid_rows(static_cast<size_t>(chunk_rows), 1);
        for (int row = 0; row < chunk_rows; ++row) {
          const int src_row = chunk.row_start + row;
          if (src_row >= 0 && src_row < static_cast<int>(source_valid_rows.size())) {
            chunk_valid_rows[static_cast<size_t>(row)] = source_valid_rows[static_cast<size_t>(src_row)];
          }
        }

        const int runtime_rows = std::max(patch_size,
                                          (std::max(1, chunk_rows) / std::max(1, patch_size)) * std::max(1, patch_size));
        const int runtime_cols = std::max(patch_size,
                                          (std::max(1, src_cols) / std::max(1, patch_size)) * std::max(1, patch_size));
        float* power_chunk_device = nullptr;
        float* corrected_chunk_device = nullptr;
        const size_t chunk_bytes = static_cast<size_t>(chunk_rows) * static_cast<size_t>(src_cols) * sizeof(float);
        {
          const auto stage_start = std::chrono::steady_clock::now();
          if (cudaMalloc(reinterpret_cast<void**>(&power_chunk_device), chunk_bytes) != cudaSuccess ||
              cudaMalloc(reinterpret_cast<void**>(&corrected_chunk_device), chunk_bytes) != cudaSuccess) {
            if (power_chunk_device != nullptr) {
              cudaFree(power_chunk_device);
            }
            if (corrected_chunk_device != nullptr) {
              cudaFree(corrected_chunk_device);
            }
            throw std::runtime_error("failed to allocate chunk GPU buffers for live DINO reference path");
          }
          if (cudaMemcpy(power_chunk_device, power_chunk.data(), chunk_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
              cudaMemcpy(corrected_chunk_device, corrected_chunk.data(), chunk_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(power_chunk_device);
            cudaFree(corrected_chunk_device);
            throw std::runtime_error("failed to upload chunk tensors for live DINO reference path");
          }
          reference_stage_ms[3] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
        }

        DinoTorchRuntimeInput runtime_input;
        runtime_input.channel_number = channel_number;
        runtime_input.frame_number = frame_number;
        runtime_input.src_rows = chunk_rows;
        runtime_input.src_cols = src_cols;
        runtime_input.dst_rows = runtime_rows;
        runtime_input.dst_cols = runtime_cols;
        runtime_input.patch_size = patch_size;
        runtime_input.cuda_stream = work_stream();
        runtime_input.resolution_hz = resolution_hz;
        runtime_input.span_hz = std::max(0.0, chunk.freq_stop_hz - chunk.freq_start_hz);
        runtime_input.power_db_device = power_chunk_device;
        runtime_input.corrected_db_device = corrected_chunk_device;

        const auto runtime_call_start = std::chrono::steady_clock::now();
        const auto runtime_result = torch_runtime_->run(runtime_config, runtime_input);
        runtime_call_wall_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - runtime_call_start).count();

        cudaFree(power_chunk_device);
        cudaFree(corrected_chunk_device);

        if (!runtime_result.success) {
          throw std::runtime_error(std::string("chunked live DINO runtime failed at ") + runtime_result.error_stage +
                                   ": " + runtime_result.error_message + " (" + runtime_result.error_detail + ")");
        }
        if (runtime_result.score_map.size() != static_cast<size_t>(runtime_rows) * static_cast<size_t>(runtime_cols)) {
          throw std::runtime_error("unexpected chunk DINO score map size in live reference path");
        }

        backend_used = runtime_result.backend_used;
        torchscript_forward_ready = torchscript_forward_ready || runtime_result.torchscript_forward_ready;
        aligned_input_rows = runtime_result.aligned_rows;
        aligned_input_cols = runtime_result.aligned_cols;
        power_threshold = runtime_result.power_threshold;
        aggregated_runtime_timing.frontend_correction_ms += runtime_result.timing.frontend_correction_ms;
        aggregated_runtime_timing.crop_align_ms += runtime_result.timing.crop_align_ms;
        aggregated_runtime_timing.resize_ms += runtime_result.timing.resize_ms;
        aggregated_runtime_timing.model_prep_ms += runtime_result.timing.model_prep_ms;
        aggregated_runtime_timing.torch_forward_ms += runtime_result.timing.torch_forward_ms;
        aggregated_runtime_timing.dino_score_ms += runtime_result.timing.dino_score_ms;
        aggregated_runtime_timing.fusion_ms += runtime_result.timing.fusion_ms;

        std::vector<float> dino_score_norm;
        {
          const auto stage_start = std::chrono::steady_clock::now();
          const auto raw_aligned_score = resize_bilinear(runtime_result.score_map,
                                                         runtime_rows,
                                                         runtime_cols,
                                                         runtime_result.aligned_rows,
                                                         runtime_result.aligned_cols);
          const auto grouped_dino_score = project_aligned_map_to_output(raw_aligned_score,
                                                                        runtime_result.aligned_rows,
                                                                        runtime_result.aligned_cols,
                                                                        chunk_rows,
                                                                        src_cols,
                                                                        0,
                                                                        0,
                                                                        chunk_rows,
                                                                        src_cols);
          dino_score_norm = normalize01_quantile(grouped_dino_score, 5.0, 95.0);
          reference_stage_ms[4] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
        }
        const auto chunk_valid_mask = expand_row_valid_mask(chunk_valid_rows, src_cols);
        HybridPostprocessResult chunk_hybrid;
        {
          const auto stage_start = std::chrono::steady_clock::now();
          const auto coherence_gate = structure_tensor_gate(corrected_chunk, chunk_rows, src_cols, chunk_valid_mask);
          const auto coherence_gate_norm = normalize01_quantile(coherence_gate, 5.0, 99.0);
          chunk_hybrid = run_residual_veto_hybrid(dino_score_norm,
                                                  coherence_gate_norm,
                                                  chunk_valid_mask,
                                                  chunk_rows,
                                                  src_cols);
          reference_stage_ms[5] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
        }

        GroupingResult chunk_grouping;
        {
          const auto stage_start = std::chrono::steady_clock::now();
          chunk_grouping = group_mask_regions(chunk_hybrid.mask,
                                              chunk_hybrid.combined_score,
                                              chunk_valid_mask,
                                              chunk_rows,
                                              src_cols,
                                              filter_detection_mask_.get(),
                                              grouping_bridge_freq_px_.get(),
                                              grouping_bridge_time_px_.get(),
                                              grouping_min_component_size_.get(),
                                              grouping_min_freq_span_px_.get(),
                                              grouping_min_time_span_px_.get(),
                                              static_cast<float>(grouping_min_density_.get()),
                                              static_cast<float>(grouping_time_continuity_ratio_.get()));
          reference_stage_ms[6] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
        }
        const int chunk_output_rows = std::max(1, input_height_.get());
        const int chunk_output_cols = std::max(1, input_width_.get());
        const auto chunk_grouped_mask_output = resize_mask_nearest(chunk_grouping.grouped_mask,
                                                                   chunk_rows,
                                                                   src_cols,
                                                                   chunk_output_rows,
                                                                   chunk_output_cols);
        const auto chunk_combined_score_output = resize_bilinear(chunk_hybrid.combined_score,
                                                                 chunk_rows,
                                                                 src_cols,
                                                                 chunk_output_rows,
                                                                 chunk_output_cols);
        accumulate_chunk_grouped_result(chunk_grouped_mask_output,
                                        chunk_combined_score_output,
                                        chunk_output_rows,
                                        chunk_output_cols,
                                        chunk_rows,
                                        src_cols,
                                        chunk.row_start,
                                        projected_grouped_mask,
                                        projected_grouped_score_sum,
                                        projected_grouped_score_weight,
                                        src_rows,
                                        src_cols);

        seed_freq_sum += chunk_hybrid.seed_freq_threshold;
        seed_res_sum += chunk_hybrid.seed_res_threshold;
        grow_freq_sum += chunk_hybrid.grow_freq_threshold;
        grow_res_sum += chunk_hybrid.grow_res_threshold;
        combined_threshold_sum += chunk_hybrid.combined_threshold;
      }

      std::vector<float> projected_grouped_score(frame_elements, 0.0f);
      for (size_t index = 0; index < projected_grouped_score.size(); ++index) {
        if (projected_grouped_score_weight[index] > 0.0f) {
          projected_grouped_score[index] = projected_grouped_score_sum[index] / projected_grouped_score_weight[index];
        }
      }

      const auto source_valid_mask = expand_row_valid_mask(source_valid_rows, src_cols);
      for (int row = 0; row < src_rows; ++row) {
        if (source_valid_rows[static_cast<size_t>(row)] != 0) {
          continue;
        }
        for (int col = 0; col < src_cols; ++col) {
          const size_t flat = flat_index(src_cols, row, col);
          projected_grouped_mask[flat] = 0;
          projected_grouped_score[flat] = 0.0f;
        }
      }

      GroupingResult global_grouping;
      {
        const auto stage_start = std::chrono::steady_clock::now();
        global_grouping = group_mask_regions(projected_grouped_mask,
                                             projected_grouped_score,
                                             source_valid_mask,
                                             src_rows,
                                             src_cols,
                                             filter_detection_mask_.get(),
                                             grouping_bridge_freq_px_.get(),
                                             grouping_bridge_time_px_.get(),
                                             grouping_min_component_size_.get(),
                                             grouping_min_freq_span_px_.get(),
                                             grouping_min_time_span_px_.get(),
                                             static_cast<float>(grouping_min_density_.get()),
                                             static_cast<float>(grouping_time_continuity_ratio_.get()));
        reference_stage_ms[7] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
      }

      hybrid_result.mask = std::move(global_grouping.grouped_mask);
      hybrid_result.final_fraction = mean_mask_value(hybrid_result.mask);
      hybrid_result.connected_fraction = connected_fraction(hybrid_result.mask, source_valid_mask);
      hybrid_result.component_count = static_cast<int>(global_grouping.boxes.size());
      const float chunk_divisor = static_cast<float>(std::max(1, chunk_count));
      hybrid_result.seed_freq_threshold = seed_freq_sum / chunk_divisor;
      hybrid_result.seed_res_threshold = seed_res_sum / chunk_divisor;
      hybrid_result.grow_freq_threshold = grow_freq_sum / chunk_divisor;
      hybrid_result.grow_res_threshold = grow_res_sum / chunk_divisor;
      hybrid_result.combined_threshold = combined_threshold_sum / chunk_divisor;
      hybrid_result.combined_score = std::move(projected_grouped_score);

      hybrid_call_wall_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - hybrid_call_start)
                                 .count();
    });

    time_step_ms(kMaskSaveStage, [&] {
      if (!should_save_mask) {
        return;
      }
      std::vector<uint8_t> image(hybrid_result.mask.size(), 0);
      for (size_t index = 0; index < hybrid_result.mask.size(); ++index) {
        image[index] = hybrid_result.mask[index] ? 255 : 0;
      }

      const auto mask_path = make_mask_output_path(output_dir_.get(), channel_number, frame_number, src_rows, src_cols);
      if (!write_pgm(mask_path, image, src_cols, src_rows)) {
        HOLOSCAN_LOG_ERROR("Failed to write DINO hybrid mask image: {}", mask_path);
      } else {
        ++masks_saved_[local_channel_index];
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

    {
      auto& service = service_stats_[local_channel_index];
      const double wall_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - compute_wall_start)
                                 .count();
      service.samples++;
      service.total_wall_ms += wall_ms;
      service.max_wall_ms = std::max(service.max_wall_ms, wall_ms);
      service.total_runtime_call_ms += runtime_call_wall_ms;
      service.max_runtime_call_ms = std::max(service.max_runtime_call_ms, runtime_call_wall_ms);
      service.total_hybrid_call_ms += hybrid_call_wall_ms;
      service.max_hybrid_call_ms = std::max(service.max_hybrid_call_ms, hybrid_call_wall_ms);
      service.total_runtime_stage_ms[0] += aggregated_runtime_timing.frontend_correction_ms;
      service.total_runtime_stage_ms[1] += aggregated_runtime_timing.crop_align_ms;
      service.total_runtime_stage_ms[2] += aggregated_runtime_timing.resize_ms;
      service.total_runtime_stage_ms[3] += aggregated_runtime_timing.model_prep_ms;
      service.total_runtime_stage_ms[4] += aggregated_runtime_timing.torch_forward_ms;
      service.total_runtime_stage_ms[5] += aggregated_runtime_timing.dino_score_ms;
      service.total_runtime_stage_ms[6] += aggregated_runtime_timing.fusion_ms;
      service.max_runtime_stage_ms[0] = std::max(service.max_runtime_stage_ms[0], aggregated_runtime_timing.frontend_correction_ms);
      service.max_runtime_stage_ms[1] = std::max(service.max_runtime_stage_ms[1], aggregated_runtime_timing.crop_align_ms);
      service.max_runtime_stage_ms[2] = std::max(service.max_runtime_stage_ms[2], aggregated_runtime_timing.resize_ms);
      service.max_runtime_stage_ms[3] = std::max(service.max_runtime_stage_ms[3], aggregated_runtime_timing.model_prep_ms);
      service.max_runtime_stage_ms[4] = std::max(service.max_runtime_stage_ms[4], aggregated_runtime_timing.torch_forward_ms);
      service.max_runtime_stage_ms[5] = std::max(service.max_runtime_stage_ms[5], aggregated_runtime_timing.dino_score_ms);
      service.max_runtime_stage_ms[6] = std::max(service.max_runtime_stage_ms[6], aggregated_runtime_timing.fusion_ms);
      for (size_t stage_index = 0; stage_index < kReferenceStageNames.size(); ++stage_index) {
        service.total_reference_stage_ms[stage_index] += reference_stage_ms[stage_index];
        service.max_reference_stage_ms[stage_index] = std::max(service.max_reference_stage_ms[stage_index], reference_stage_ms[stage_index]);
      }
      service.total_chunk_count += static_cast<uint64_t>(std::max(0, chunk_count));
      service.max_chunk_count = std::max<uint64_t>(service.max_chunk_count, static_cast<uint64_t>(std::max(0, chunk_count)));
    }

    meta->set("dino_frame_number", frame_number);
    meta->set("dino_mask_height", static_cast<uint32_t>(src_rows));
    meta->set("dino_mask_width", static_cast<uint32_t>(src_cols));
    meta->set("dino_mask_threshold_db", mask_threshold_db_.get());
    meta->set("dino_backend", backend_used);
    meta->set("dino_backend_mode", effective_backend_mode);
    meta->set("dino_model_name", model_name_.get());
    meta->set("dino_weights_path", weights_path_.get());
    meta->set("dino_model_script_path", model_script_path_.get());
    meta->set("dino_torchscript_init_mode", torchscript_init_mode_.get());
    meta->set("dino_torchscript_forward_ready", torchscript_forward_ready);
    meta->set("dino_patch_size", patch_size);
    meta->set("dino_fft_size", fft_size_.get());
    meta->set("dino_noverlap", noverlap_.get());
    meta->set("dino_ignore_bins_per_side", ignore_bins_per_side);
    meta->set("dino_freq_bin_hz", resolution_hz);
    meta->set("dino_frontend_correction_enabled", frontend_correction_enable_.get());
    meta->set("dino_input_aligned_height", aligned_input_rows);
    meta->set("dino_input_aligned_width", aligned_input_cols);
    meta->set("dino_group_score_threshold", static_cast<double>(hybrid_result.combined_threshold));
    meta->set("dino_power_score_threshold", power_threshold);
    meta->set("dino_pipeline_final_threshold", static_cast<double>(hybrid_result.combined_threshold * 0.85f));
    meta->set("dino_pipeline_variant", std::string("chunked_retry_merge_reference_v1"));
    meta->set("dino_chunk_count", static_cast<uint32_t>(std::max(0, chunk_count)));
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
      auto& timing = timing_stats_[local_channel_index];
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
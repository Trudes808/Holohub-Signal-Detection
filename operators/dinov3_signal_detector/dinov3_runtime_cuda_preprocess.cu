// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "dinov3_runtime_cuda_preprocess.hpp"

#include <cmath>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace holoscan::ops {
namespace {

template <typename T>
__host__ __device__ inline T clamp_value(T value, T low, T high) {
  return value < low ? low : (value > high ? high : value);
}

__host__ __device__ inline size_t flat_index(int cols, int row, int col) {
  return static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
}

constexpr int kHistogramBins = 256;
constexpr float kPositiveInfinity = std::numeric_limits<float>::infinity();
constexpr int kRawPositionalBasisDim = 16;

struct PreprocessScratch {
  float* plane_a = nullptr;
  float* plane_b = nullptr;
  float* plane_c = nullptr;
  float* plane_d = nullptr;
  float* row_mean = nullptr;
  float* row_trend = nullptr;
  float* col_mean = nullptr;
  float* col_trend = nullptr;
  float* scalars = nullptr;
  uint32_t* histogram = nullptr;
  size_t plane_capacity = 0;
  size_t row_capacity = 0;
  size_t col_capacity = 0;

  ~PreprocessScratch() {
    release();
  }

  void release() {
    auto free_ptr = [](auto*& ptr) {
      if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
      }
    };
    free_ptr(plane_a);
    free_ptr(plane_b);
    free_ptr(plane_c);
    free_ptr(plane_d);
    free_ptr(row_mean);
    free_ptr(row_trend);
    free_ptr(col_mean);
    free_ptr(col_trend);
    free_ptr(scalars);
    free_ptr(histogram);
    plane_capacity = 0;
    row_capacity = 0;
    col_capacity = 0;
  }

  bool ensure_capacity(size_t plane_elements, size_t rows, size_t cols) {
    if (plane_elements > plane_capacity) {
      if (plane_a != nullptr) cudaFree(plane_a);
      if (plane_b != nullptr) cudaFree(plane_b);
      if (plane_c != nullptr) cudaFree(plane_c);
      if (plane_d != nullptr) cudaFree(plane_d);
      plane_a = nullptr;
      plane_b = nullptr;
      plane_c = nullptr;
      plane_d = nullptr;
      if (cudaMalloc(reinterpret_cast<void**>(&plane_a), plane_elements * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&plane_b), plane_elements * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&plane_c), plane_elements * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&plane_d), plane_elements * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      plane_capacity = plane_elements;
    }
    if (rows > row_capacity) {
      if (row_mean != nullptr) cudaFree(row_mean);
      if (row_trend != nullptr) cudaFree(row_trend);
      row_mean = nullptr;
      row_trend = nullptr;
      if (cudaMalloc(reinterpret_cast<void**>(&row_mean), rows * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&row_trend), rows * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      row_capacity = rows;
    }
    if (cols > col_capacity) {
      if (col_mean != nullptr) cudaFree(col_mean);
      if (col_trend != nullptr) cudaFree(col_trend);
      col_mean = nullptr;
      col_trend = nullptr;
      if (cudaMalloc(reinterpret_cast<void**>(&col_mean), cols * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&col_trend), cols * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      col_capacity = cols;
    }
    if (scalars == nullptr && cudaMalloc(reinterpret_cast<void**>(&scalars), 8 * sizeof(float)) != cudaSuccess) {
      release();
      return false;
    }
    if (histogram == nullptr && cudaMalloc(reinterpret_cast<void**>(&histogram), kHistogramBins * sizeof(uint32_t)) != cudaSuccess) {
      release();
      return false;
    }
    return true;
  }
};

PreprocessScratch& preprocess_scratch() {
  static PreprocessScratch scratch;
  return scratch;
}

struct RawScoreScratch {
  float* xty = nullptr;
  float* beta = nullptr;
  float* raw_patch = nullptr;
  float* aligned_map = nullptr;
  size_t xty_capacity = 0;
  size_t raw_patch_capacity = 0;
  size_t aligned_capacity = 0;

  ~RawScoreScratch() {
    release();
  }

  void release() {
    auto free_ptr = [](float*& ptr) {
      if (ptr != nullptr) {
        cudaFree(ptr);
        ptr = nullptr;
      }
    };
    free_ptr(xty);
    free_ptr(beta);
    free_ptr(raw_patch);
    free_ptr(aligned_map);
    xty_capacity = 0;
    raw_patch_capacity = 0;
    aligned_capacity = 0;
  }

  bool ensure_capacity(size_t requested_xty_capacity,
                       size_t requested_raw_patch_capacity,
                       size_t requested_aligned_capacity) {
    if (requested_xty_capacity > xty_capacity) {
      if (xty != nullptr) cudaFree(xty);
      if (beta != nullptr) cudaFree(beta);
      xty = nullptr;
      beta = nullptr;
      if (requested_xty_capacity > 0 &&
          (cudaMalloc(reinterpret_cast<void**>(&xty), requested_xty_capacity * sizeof(float)) != cudaSuccess ||
           cudaMalloc(reinterpret_cast<void**>(&beta), requested_xty_capacity * sizeof(float)) != cudaSuccess)) {
        release();
        return false;
      }
      xty_capacity = requested_xty_capacity;
    }
    if (requested_raw_patch_capacity > raw_patch_capacity) {
      if (raw_patch != nullptr) cudaFree(raw_patch);
      raw_patch = nullptr;
      if (requested_raw_patch_capacity > 0 &&
          cudaMalloc(reinterpret_cast<void**>(&raw_patch), requested_raw_patch_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      raw_patch_capacity = requested_raw_patch_capacity;
    }
    if (requested_aligned_capacity > aligned_capacity) {
      if (aligned_map != nullptr) cudaFree(aligned_map);
      aligned_map = nullptr;
      if (requested_aligned_capacity > 0 &&
          cudaMalloc(reinterpret_cast<void**>(&aligned_map), requested_aligned_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      aligned_capacity = requested_aligned_capacity;
    }
    return true;
  }
};

RawScoreScratch& raw_score_scratch() {
  static RawScoreScratch scratch;
  return scratch;
}

std::vector<float> positional_design_matrix_host(int patch_rows, int patch_cols) {
  constexpr float kPi = 3.14159265358979323846f;
  const int patch_count = patch_rows * patch_cols;
  std::vector<float> design(static_cast<size_t>(std::max(patch_count, 0)) * static_cast<size_t>(kRawPositionalBasisDim), 0.0f);
  for (int row = 0; row < patch_rows; ++row) {
    const float row_coord = patch_rows > 1 ? -1.0f + 2.0f * static_cast<float>(row) / static_cast<float>(patch_rows - 1) : 0.0f;
    for (int col = 0; col < patch_cols; ++col) {
      const float col_coord = patch_cols > 1 ? -1.0f + 2.0f * static_cast<float>(col) / static_cast<float>(patch_cols - 1) : 0.0f;
      const size_t base = flat_index(kRawPositionalBasisDim, row * patch_cols + col, 0);
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
  return design;
}

bool invert_small_square_matrix_host(const std::vector<float>& input, int dim, std::vector<float>& inverse) {
  if (dim <= 0 || input.size() != static_cast<size_t>(dim) * static_cast<size_t>(dim)) {
    return false;
  }

  std::vector<float> augmented(static_cast<size_t>(dim) * static_cast<size_t>(dim * 2), 0.0f);
  for (int row = 0; row < dim; ++row) {
    for (int col = 0; col < dim; ++col) {
      augmented[static_cast<size_t>(row) * static_cast<size_t>(dim * 2) + static_cast<size_t>(col)] =
          input[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col)];
    }
    augmented[static_cast<size_t>(row) * static_cast<size_t>(dim * 2) + static_cast<size_t>(dim + row)] = 1.0f;
  }

  for (int pivot = 0; pivot < dim; ++pivot) {
    int best_row = pivot;
    float best_value = std::fabs(augmented[static_cast<size_t>(pivot) * static_cast<size_t>(dim * 2) + static_cast<size_t>(pivot)]);
    for (int row = pivot + 1; row < dim; ++row) {
      const float candidate = std::fabs(augmented[static_cast<size_t>(row) * static_cast<size_t>(dim * 2) + static_cast<size_t>(pivot)]);
      if (candidate > best_value) {
        best_value = candidate;
        best_row = row;
      }
    }
    if (best_value < 1.0e-8f) {
      return false;
    }
    if (best_row != pivot) {
      for (int col = 0; col < dim * 2; ++col) {
        std::swap(augmented[static_cast<size_t>(pivot) * static_cast<size_t>(dim * 2) + static_cast<size_t>(col)],
                  augmented[static_cast<size_t>(best_row) * static_cast<size_t>(dim * 2) + static_cast<size_t>(col)]);
      }
    }
    const float pivot_value = augmented[static_cast<size_t>(pivot) * static_cast<size_t>(dim * 2) + static_cast<size_t>(pivot)];
    for (int col = 0; col < dim * 2; ++col) {
      augmented[static_cast<size_t>(pivot) * static_cast<size_t>(dim * 2) + static_cast<size_t>(col)] /= pivot_value;
    }
    for (int row = 0; row < dim; ++row) {
      if (row == pivot) {
        continue;
      }
      const float factor = augmented[static_cast<size_t>(row) * static_cast<size_t>(dim * 2) + static_cast<size_t>(pivot)];
      if (std::fabs(factor) < 1.0e-12f) {
        continue;
      }
      for (int col = 0; col < dim * 2; ++col) {
        augmented[static_cast<size_t>(row) * static_cast<size_t>(dim * 2) + static_cast<size_t>(col)] -=
            factor * augmented[static_cast<size_t>(pivot) * static_cast<size_t>(dim * 2) + static_cast<size_t>(col)];
      }
    }
  }

  inverse.assign(static_cast<size_t>(dim) * static_cast<size_t>(dim), 0.0f);
  for (int row = 0; row < dim; ++row) {
    for (int col = 0; col < dim; ++col) {
      inverse[static_cast<size_t>(row) * static_cast<size_t>(dim) + static_cast<size_t>(col)] =
          augmented[static_cast<size_t>(row) * static_cast<size_t>(dim * 2) + static_cast<size_t>(dim + col)];
    }
  }
  return true;
}

struct PositionalSuppressionDeviceCache {
  int patch_count = 0;
  float* design_device = nullptr;
  float* inverse_gram_device = nullptr;

  ~PositionalSuppressionDeviceCache() {
    if (design_device != nullptr) {
      cudaFree(design_device);
    }
    if (inverse_gram_device != nullptr) {
      cudaFree(inverse_gram_device);
    }
  }
};

std::shared_ptr<const PositionalSuppressionDeviceCache> get_positional_suppression_device_cache(int patch_rows, int patch_cols) {
  const uint64_t cache_key = (static_cast<uint64_t>(static_cast<uint32_t>(patch_rows)) << 32U) |
                             static_cast<uint64_t>(static_cast<uint32_t>(patch_cols));
  static std::mutex cache_mutex;
  static std::unordered_map<uint64_t, std::shared_ptr<const PositionalSuppressionDeviceCache>> cache_by_shape;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto found = cache_by_shape.find(cache_key);
    if (found != cache_by_shape.end()) {
      return found->second;
    }
  }

  auto cache = std::make_shared<PositionalSuppressionDeviceCache>();
  cache->patch_count = patch_rows * patch_cols;
  const auto design = positional_design_matrix_host(patch_rows, patch_cols);
  std::vector<float> gram(static_cast<size_t>(kRawPositionalBasisDim) * static_cast<size_t>(kRawPositionalBasisDim), 0.0f);
  for (int patch_index = 0; patch_index < cache->patch_count; ++patch_index) {
    const size_t design_base = static_cast<size_t>(patch_index) * static_cast<size_t>(kRawPositionalBasisDim);
    for (int left = 0; left < kRawPositionalBasisDim; ++left) {
      const float left_value = design[design_base + static_cast<size_t>(left)];
      for (int right = 0; right < kRawPositionalBasisDim; ++right) {
        gram[static_cast<size_t>(left) * static_cast<size_t>(kRawPositionalBasisDim) + static_cast<size_t>(right)] +=
            left_value * design[design_base + static_cast<size_t>(right)];
      }
    }
  }
  for (int diag = 0; diag < kRawPositionalBasisDim; ++diag) {
    gram[static_cast<size_t>(diag) * static_cast<size_t>(kRawPositionalBasisDim) + static_cast<size_t>(diag)] += 1.0e-3f;
  }

  std::vector<float> inverse_gram;
  if (!invert_small_square_matrix_host(gram, kRawPositionalBasisDim, inverse_gram)) {
    return nullptr;
  }
  if (cudaMalloc(reinterpret_cast<void**>(&cache->design_device), design.size() * sizeof(float)) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&cache->inverse_gram_device), inverse_gram.size() * sizeof(float)) != cudaSuccess) {
    return nullptr;
  }
  if (cudaMemcpy(cache->design_device, design.data(), design.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(cache->inverse_gram_device, inverse_gram.data(), inverse_gram.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
    return nullptr;
  }

  std::lock_guard<std::mutex> lock(cache_mutex);
  const auto [iter, inserted] = cache_by_shape.emplace(cache_key, cache);
  return inserted ? cache : iter->second;
}

__global__ void sample_row_mean_kernel(const float* input, int rows, int cols, float* output) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float partial[256];
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
    output[row] = partial[0] / static_cast<float>(max(cols, 1));
  }
}

__global__ void sample_col_mean_kernel(const float* input, int rows, int cols, float* output) {
  const int col = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float partial[256];
  float sum = 0.0f;
  for (int row = tid; row < rows; row += blockDim.x) {
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
    output[col] = partial[0] / static_cast<float>(max(rows, 1));
  }
}

__global__ void mean_reduce_kernel(const float* input, int count, float* output_mean) {
  __shared__ float partial[256];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int index = tid; index < count; index += blockDim.x) {
    sum += input[index];
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
    output_mean[0] = partial[0] / static_cast<float>(max(count, 1));
  }
}

__global__ void mean_std_reduce_kernel(const float* input, int count, float* output_mean, float* output_std) {
  __shared__ float partial_sum[256];
  __shared__ float partial_sq[256];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  float sq_sum = 0.0f;
  for (int index = tid; index < count; index += blockDim.x) {
    const float value = input[index];
    sum += value;
    sq_sum += value * value;
  }
  partial_sum[tid] = sum;
  partial_sq[tid] = sq_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_sum[tid] += partial_sum[tid + stride];
      partial_sq[tid] += partial_sq[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const float mean = partial_sum[0] / static_cast<float>(max(count, 1));
    const float variance = fmaxf(partial_sq[0] / static_cast<float>(max(count, 1)) - mean * mean, 0.0f);
    output_mean[0] = mean;
    output_std[0] = sqrtf(variance);
  }
}

__global__ void minmax_reduce_kernel(const float* input, int count, float* output_min, float* output_max) {
  __shared__ float partial_min[256];
  __shared__ float partial_max[256];
  const int tid = threadIdx.x;
  float local_min = kPositiveInfinity;
  float local_max = -kPositiveInfinity;
  for (int index = tid; index < count; index += blockDim.x) {
    const float value = input[index];
    local_min = fminf(local_min, value);
    local_max = fmaxf(local_max, value);
  }
  partial_min[tid] = local_min;
  partial_max[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_min[tid] = fminf(partial_min[tid], partial_min[tid + stride]);
      partial_max[tid] = fmaxf(partial_max[tid], partial_max[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    output_min[0] = isfinite(partial_min[0]) ? partial_min[0] : 0.0f;
    output_max[0] = isfinite(partial_max[0]) ? partial_max[0] : 1.0f;
  }
}

__global__ void histogram_kernel(const float* input,
                                 int count,
                                 const float* input_min,
                                 const float* input_max,
                                 uint32_t* histogram) {
  __shared__ uint32_t local_hist[kHistogramBins];
  const int tid = threadIdx.x;
  if (tid < kHistogramBins) {
    local_hist[tid] = 0;
  }
  __syncthreads();

  const float min_value = input_min[0];
  const float max_value = input_max[0];
  const float inv_range = 1.0f / fmaxf(max_value - min_value, 1.0e-6f);
  for (int index = tid; index < count; index += blockDim.x) {
    const float normalized = fminf(fmaxf((input[index] - min_value) * inv_range, 0.0f), 1.0f);
    const int bin = clamp_value(static_cast<int>(llroundf(normalized * static_cast<float>(kHistogramBins - 1))), 0, kHistogramBins - 1);
    atomicAdd(&local_hist[bin], 1U);
  }
  __syncthreads();
  if (tid < kHistogramBins) {
    histogram[tid] = local_hist[tid];
  }
}

__global__ void histogram_quantile_kernel(const uint32_t* histogram,
                                          int count,
                                          float low_q,
                                          float high_q,
                                          float* output_low_q,
                                          float* output_high_q) {
  if (threadIdx.x != 0) {
    return;
  }
  const int low_target = clamp_value(static_cast<int>(llroundf(low_q * static_cast<float>(max(count - 1, 0)))), 0, max(count - 1, 0));
  const int high_target = clamp_value(static_cast<int>(llroundf(high_q * static_cast<float>(max(count - 1, 0)))), 0, max(count - 1, 0));
  int cumulative = 0;
  int low_bin = 0;
  int high_bin = 0;
  bool low_found = false;
  for (int bin = 0; bin < kHistogramBins; ++bin) {
    cumulative += static_cast<int>(histogram[bin]);
    if (!low_found && cumulative > low_target) {
      low_bin = bin;
      low_found = true;
    }
    if (cumulative > high_target) {
      high_bin = bin;
      break;
    }
  }
  output_low_q[0] = static_cast<float>(low_bin) / static_cast<float>(kHistogramBins - 1);
  output_high_q[0] = static_cast<float>(high_bin) / static_cast<float>(kHistogramBins - 1);
}

__global__ void gaussian_smooth_1d_kernel(const float* input, int count, int radius, float sigma, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  float sum = 0.0f;
  float weight_sum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const int src = clamp_value(index + offset, 0, count - 1);
    const float weight = expf(-(static_cast<float>(offset * offset)) / (2.0f * sigma * sigma));
    sum += input[src] * weight;
    weight_sum += weight;
  }
  output[index] = weight_sum > 0.0f ? sum / weight_sum : input[index];
}

__global__ void detrend_kernel(const float* input,
                               const float* row_trend,
                               const float* col_trend,
                               const float* global_mean,
                               int rows,
                               int cols,
                               float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (index >= total) {
    return;
  }
  const int row = index / cols;
  const int col = index % cols;
  output[index] = input[index] - (row_trend[row] + col_trend[col] - global_mean[0]);
}

__global__ void box_mean_kernel(const float* input,
                                int rows,
                                int cols,
                                int radius_rows,
                                int radius_cols,
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
  const int col_start = max(0, col - radius_cols);
  const int col_stop = min(cols - 1, col + radius_cols);
  float sum = 0.0f;
  int count = 0;
  for (int src_row = row_start; src_row <= row_stop; ++src_row) {
    for (int src_col = col_start; src_col <= col_stop; ++src_col) {
      sum += input[flat_index(cols, src_row, src_col)];
      ++count;
    }
  }
  output[index] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void subtract_kernel(const float* lhs, const float* rhs, int count, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = lhs[index] - rhs[index];
  }
}

__global__ void square_kernel(const float* input, int count, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = input[index] * input[index];
  }
}

__global__ void abs_kernel(const float* input, int count, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = fabsf(input[index]);
  }
}

__global__ void local_z_kernel(const float* local_resid, const float* local_scale_mean, int count, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float scale = sqrtf(local_scale_mean[index] + 1.0e-6f);
    output[index] = local_resid[index] / fmaxf(scale, 1.0e-4f);
  }
}

__global__ void normalize_quantile_kernel(const float* input,
                                          int count,
                                          const float* input_min,
                                          const float* input_max,
                                          const float* low_q,
                                          const float* high_q,
                                          float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float min_value = input_min[0];
  const float max_value = input_max[0];
  const float range = fmaxf(max_value - min_value, 1.0e-6f);
  const float low = min_value + low_q[0] * range;
  const float high = min_value + high_q[0] * range;
  output[index] = fminf(fmaxf((input[index] - low) / fmaxf(high - low, 1.0e-6f), 0.0f), 1.0f);
}

__global__ void quantile_value_kernel(const float* input_min,
                                      const float* input_max,
                                      const float* quantile_norm,
                                      float* output_value) {
  if (threadIdx.x == 0) {
    output_value[0] = input_min[0] + quantile_norm[0] * fmaxf(input_max[0] - input_min[0], 1.0e-6f);
  }
}

__global__ void combine_signal_agnostic_kernel(const float* local_z,
                                               const float* abs_detrended,
                                               const float* scale_value,
                                               int count,
                                               float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float local_resid_n = fminf(fmaxf(0.5f + 0.5f * (local_z[index] / fmaxf(scale_value[0], 1.0e-6f)), 0.0f), 1.0f);
  output[index] = 0.70f * local_resid_n + 0.30f * abs_detrended[index];
}

__global__ void quantize_kernel(const float* input, int count, float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = roundf(fminf(fmaxf(input[index], 0.0f), 1.0f) * 255.0f) / 255.0f;
  }
}

__global__ void select_and_quantize_kernel(const float* fallback,
                                           const float* combined,
                                           const float* combined_std,
                                           int count,
                                           float* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const float source = combined_std[0] < 0.02f ? fallback[index] : combined[index];
  output[index] = roundf(fminf(fmaxf(source, 0.0f), 1.0f) * 255.0f) / 255.0f;
}

__global__ void raw_positional_xty_kernel(const float* patch_features,
                                          const float* design,
                                          int patch_count,
                                          int feature_dim,
                                          int basis_dim,
                                          float* xty) {
  const int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int basis_index = blockIdx.y;
  const int batch_index = blockIdx.z;
  if (feature_index >= feature_dim || basis_index >= basis_dim) {
    return;
  }

  const size_t batch_feature_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim);
  float sum = 0.0f;
  for (int patch_index = 0; patch_index < patch_count; ++patch_index) {
    sum += design[static_cast<size_t>(patch_index) * static_cast<size_t>(basis_dim) + static_cast<size_t>(basis_index)] *
           patch_features[batch_feature_offset + static_cast<size_t>(patch_index) * static_cast<size_t>(feature_dim) + static_cast<size_t>(feature_index)];
  }
  xty[(static_cast<size_t>(batch_index) * static_cast<size_t>(basis_dim) + static_cast<size_t>(basis_index)) *
          static_cast<size_t>(feature_dim) +
      static_cast<size_t>(feature_index)] = sum;
}

__global__ void raw_positional_beta_kernel(const float* inverse_gram,
                                           const float* xty,
                                           int feature_dim,
                                           int basis_dim,
                                           float* beta) {
  const int feature_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int basis_row = blockIdx.y;
  const int batch_index = blockIdx.z;
  if (feature_index >= feature_dim || basis_row >= basis_dim) {
    return;
  }

  float sum = 0.0f;
  for (int basis_col = 0; basis_col < basis_dim; ++basis_col) {
    sum += inverse_gram[static_cast<size_t>(basis_row) * static_cast<size_t>(basis_dim) + static_cast<size_t>(basis_col)] *
           xty[(static_cast<size_t>(batch_index) * static_cast<size_t>(basis_dim) + static_cast<size_t>(basis_col)) *
                   static_cast<size_t>(feature_dim) +
               static_cast<size_t>(feature_index)];
  }
  beta[(static_cast<size_t>(batch_index) * static_cast<size_t>(basis_dim) + static_cast<size_t>(basis_row)) *
           static_cast<size_t>(feature_dim) +
       static_cast<size_t>(feature_index)] = sum;
}

__global__ void raw_patch_energy_kernel(const float* patch_features,
                                        const float* design,
                                        const float* beta,
                                        int patch_count,
                                        int feature_dim,
                                        int basis_dim,
                                        float suppression,
                                        float* raw_patch) {
  const int patch_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_index = blockIdx.y;
  if (patch_index >= patch_count) {
    return;
  }

  const size_t batch_feature_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim);
  const size_t beta_batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(basis_dim) * static_cast<size_t>(feature_dim);
  const size_t design_offset = static_cast<size_t>(patch_index) * static_cast<size_t>(basis_dim);
  float sum_sq = 0.0f;
  for (int feature_index = 0; feature_index < feature_dim; ++feature_index) {
    float trend = 0.0f;
    for (int basis_index = 0; basis_index < basis_dim; ++basis_index) {
      trend += design[design_offset + static_cast<size_t>(basis_index)] *
               beta[beta_batch_offset + static_cast<size_t>(basis_index) * static_cast<size_t>(feature_dim) + static_cast<size_t>(feature_index)];
    }
    const float value = patch_features[batch_feature_offset + static_cast<size_t>(patch_index) * static_cast<size_t>(feature_dim) +
                                       static_cast<size_t>(feature_index)] -
                        suppression * trend;
    sum_sq += value * value;
  }
  raw_patch[static_cast<size_t>(batch_index) * static_cast<size_t>(patch_count) + static_cast<size_t>(patch_index)] =
      sqrtf(fmaxf(sum_sq / static_cast<float>(feature_dim), 1.0e-6f));
}

__global__ void raw_patch_energy_plain_kernel(const float* patch_features,
                                              int patch_count,
                                              int feature_dim,
                                              float* raw_patch) {
  const int patch_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int batch_index = blockIdx.y;
  if (patch_index >= patch_count) {
    return;
  }

  const size_t batch_feature_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim);
  float sum_sq = 0.0f;
  for (int feature_index = 0; feature_index < feature_dim; ++feature_index) {
    const float value = patch_features[batch_feature_offset + static_cast<size_t>(patch_index) * static_cast<size_t>(feature_dim) +
                                       static_cast<size_t>(feature_index)];
    sum_sq += value * value;
  }
  raw_patch[static_cast<size_t>(batch_index) * static_cast<size_t>(patch_count) + static_cast<size_t>(patch_index)] =
      sqrtf(fmaxf(sum_sq / static_cast<float>(feature_dim), 1.0e-6f));
}

__device__ __forceinline__ void bilinear_align_corners_false_coordinate(int dst_index,
                                                                        int src_size,
                                                                        int dst_size,
                                                                        int& src0,
                                                                        int& src1,
                                                                        float& t) {
  if (dst_size <= 0 || src_size <= 0) {
    src0 = 0;
    src1 = 0;
    t = 0.0f;
    return;
  }
  const float src = ((static_cast<float>(dst_index) + 0.5f) * static_cast<float>(src_size) / static_cast<float>(dst_size)) - 0.5f;
  src0 = static_cast<int>(floorf(src));
  t = src - static_cast<float>(src0);
  if (src0 < 0) {
    src0 = 0;
    src1 = 0;
    t = 0.0f;
    return;
  }
  src1 = src0 + 1;
  if (src1 >= src_size) {
    src1 = src_size - 1;
    if (src0 >= src_size - 1) {
      src0 = src_size - 1;
      t = 0.0f;
    }
  }
}

__global__ void resize_bilinear_batch_kernel(const float* input,
                                             int batch_size,
                                             int src_rows,
                                             int src_cols,
                                             int dst_rows,
                                             int dst_cols,
                                             float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = dst_rows * dst_cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int dst_row = local_index / dst_cols;
  const int dst_col = local_index % dst_cols;
  int src_row0 = 0;
  int src_row1 = 0;
  int src_col0 = 0;
  int src_col1 = 0;
  float row_t = 0.0f;
  float col_t = 0.0f;
  bilinear_align_corners_false_coordinate(dst_row, src_rows, dst_rows, src_row0, src_row1, row_t);
  bilinear_align_corners_false_coordinate(dst_col, src_cols, dst_cols, src_col0, src_col1, col_t);

  const size_t input_batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols);
  const float v00 = input[input_batch_offset + flat_index(src_cols, src_row0, src_col0)];
  const float v01 = input[input_batch_offset + flat_index(src_cols, src_row0, src_col1)];
  const float v10 = input[input_batch_offset + flat_index(src_cols, src_row1, src_col0)];
  const float v11 = input[input_batch_offset + flat_index(src_cols, src_row1, src_col1)];
  const float top = (1.0f - col_t) * v00 + col_t * v01;
  const float bottom = (1.0f - col_t) * v10 + col_t * v11;
  output[idx] = (1.0f - row_t) * top + row_t * bottom;
}

__global__ void project_aligned_maps_to_output_kernel(const float* aligned_maps,
                                                      int batch_size,
                                                      int aligned_rows,
                                                      int aligned_cols,
                                                      int output_rows,
                                                      int output_cols,
                                                      float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = output_rows * output_cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / output_cols;
  const int col = local_index % output_cols;
  float value = 0.0f;
  if (row < aligned_rows && col < aligned_cols) {
    const size_t aligned_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(aligned_rows) * static_cast<size_t>(aligned_cols);
    value = aligned_maps[aligned_offset + flat_index(aligned_cols, row, col)];
  }
  output[idx] = value;
}

bool record_error(cudaError_t status, std::string* error_message, const char* message) {
  if (status == cudaSuccess) {
    return true;
  }
  if (error_message != nullptr) {
    *error_message = std::string(message) + ": " + cudaGetErrorString(status);
  }
  return false;
}

bool quantile_normalize_sample(const float* input,
                               int count,
                               float low_q,
                               float high_q,
                               PreprocessScratch& scratch,
                               cudaStream_t stream,
                               float* output,
                               std::string* error_message) {
  constexpr int threads = 256;
  if (!record_error(cudaMemsetAsync(scratch.histogram, 0, kHistogramBins * sizeof(uint32_t), stream), error_message, "preprocess histogram memset failed")) {
    return false;
  }
  minmax_reduce_kernel<<<1, threads, 0, stream>>>(input, count, scratch.scalars + 2, scratch.scalars + 3);
  histogram_kernel<<<1, threads, 0, stream>>>(input, count, scratch.scalars + 2, scratch.scalars + 3, scratch.histogram);
  histogram_quantile_kernel<<<1, threads, 0, stream>>>(scratch.histogram, count, low_q, high_q, scratch.scalars + 4, scratch.scalars + 5);
  if (!record_error(cudaGetLastError(), error_message, "preprocess quantile kernels failed")) {
    return false;
  }
  const int blocks = (count + threads - 1) / threads;
  normalize_quantile_kernel<<<blocks, threads, 0, stream>>>(input,
                                                             count,
                                                             scratch.scalars + 2,
                                                             scratch.scalars + 3,
                                                             scratch.scalars + 4,
                                                             scratch.scalars + 5,
                                                             output);
  return record_error(cudaGetLastError(), error_message, "preprocess normalize kernel failed");
}

bool prepare_signal_agnostic_sample_cuda(const float* input,
                                         int rows,
                                         int cols,
                                         PreprocessScratch& scratch,
                                         cudaStream_t stream,
                                         float* output,
                                         std::string* error_message) {
  constexpr int threads = 256;
  const int count = rows * cols;
  const int plane_blocks = (count + threads - 1) / threads;
  const int row_blocks = (rows + threads - 1) / threads;
  const int col_blocks = (cols + threads - 1) / threads;
  const float row_sigma = static_cast<float>(std::max(1.0, static_cast<double>(rows) / 32.0));
  const float col_sigma = static_cast<float>(std::max(1.0, static_cast<double>(cols) / 32.0));
  const int row_radius = std::max(1, static_cast<int>(std::ceil(3.0 * row_sigma)));
  const int col_radius = std::max(1, static_cast<int>(std::ceil(3.0 * col_sigma)));

  sample_row_mean_kernel<<<rows, threads, 0, stream>>>(input, rows, cols, scratch.row_mean);
  sample_col_mean_kernel<<<cols, threads, 0, stream>>>(input, rows, cols, scratch.col_mean);
  mean_reduce_kernel<<<1, threads, 0, stream>>>(input, count, scratch.scalars);
  if (!record_error(cudaGetLastError(), error_message, "preprocess mean kernels failed")) {
    return false;
  }

  gaussian_smooth_1d_kernel<<<row_blocks, threads, 0, stream>>>(scratch.row_mean, rows, row_radius, row_sigma, scratch.row_trend);
  gaussian_smooth_1d_kernel<<<col_blocks, threads, 0, stream>>>(scratch.col_mean, cols, col_radius, col_sigma, scratch.col_trend);
  if (!record_error(cudaGetLastError(), error_message, "preprocess gaussian kernels failed")) {
    return false;
  }

  detrend_kernel<<<plane_blocks, threads, 0, stream>>>(input,
                                                        scratch.row_trend,
                                                        scratch.col_trend,
                                                        scratch.scalars,
                                                        rows,
                                                        cols,
                                                        scratch.plane_a);
  box_mean_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_a, rows, cols, 3, 3, scratch.plane_b);
  subtract_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_a, scratch.plane_b, count, scratch.plane_b);
  square_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_b, count, scratch.plane_c);
  box_mean_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_c, rows, cols, 4, 4, scratch.plane_d);
  local_z_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_b, scratch.plane_d, count, scratch.plane_c);
  if (!record_error(cudaGetLastError(), error_message, "preprocess local residual kernels failed")) {
    return false;
  }

  if (!quantile_normalize_sample(scratch.plane_a, count, 0.02f, 0.98f, scratch, stream, scratch.plane_b, error_message)) {
    return false;
  }

  abs_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_c, count, scratch.plane_d);
  if (!record_error(cudaGetLastError(), error_message, "preprocess abs kernel failed")) {
    return false;
  }
  if (!record_error(cudaMemsetAsync(scratch.histogram, 0, kHistogramBins * sizeof(uint32_t), stream), error_message, "preprocess histogram memset failed")) {
    return false;
  }
  minmax_reduce_kernel<<<1, threads, 0, stream>>>(scratch.plane_d, count, scratch.scalars + 2, scratch.scalars + 3);
  histogram_kernel<<<1, threads, 0, stream>>>(scratch.plane_d, count, scratch.scalars + 2, scratch.scalars + 3, scratch.histogram);
  histogram_quantile_kernel<<<1, threads, 0, stream>>>(scratch.histogram, count, 0.95f, 0.95f, scratch.scalars + 4, scratch.scalars + 5);
  quantile_value_kernel<<<1, threads, 0, stream>>>(scratch.scalars + 2, scratch.scalars + 3, scratch.scalars + 5, scratch.scalars + 6);
  if (!record_error(cudaGetLastError(), error_message, "preprocess scale kernels failed")) {
    return false;
  }

  combine_signal_agnostic_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_c,
                                                                        scratch.plane_b,
                                                                        scratch.scalars + 6,
                                                                        count,
                                                                        scratch.plane_d);
  mean_std_reduce_kernel<<<1, threads, 0, stream>>>(scratch.plane_d, count, scratch.scalars, scratch.scalars + 1);
  if (!record_error(cudaGetLastError(), error_message, "preprocess combined kernels failed")) {
    return false;
  }

  if (!quantile_normalize_sample(scratch.plane_a, count, 0.01f, 0.99f, scratch, stream, scratch.plane_b, error_message)) {
    return false;
  }
  select_and_quantize_kernel<<<plane_blocks, threads, 0, stream>>>(scratch.plane_b,
                                                                    scratch.plane_d,
                                                                    scratch.scalars + 1,
                                                                    count,
                                                                    output);
  return record_error(cudaGetLastError(), error_message, "preprocess finalize kernel failed");
}

}  // namespace

bool prepare_tensorrt_grayscale_batch_cuda(const float* resized_batch_device,
                                           int batch_size,
                                           int rows,
                                           int cols,
                                           bool legacy_fast_gray_preprocess,
                                           cudaStream_t cuda_stream,
                                           DinoCudaGrayBatch* output,
                                           std::string* error_message) {
  if (output == nullptr || resized_batch_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    if (error_message != nullptr) {
      *error_message = "invalid TensorRT grayscale preprocess input";
    }
    return false;
  }

  static std::mutex scratch_mutex;
  std::lock_guard<std::mutex> lock(scratch_mutex);

  auto& scratch = preprocess_scratch();
  const size_t plane_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  if (!scratch.ensure_capacity(plane_elements, static_cast<size_t>(rows), static_cast<size_t>(cols))) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate TensorRT grayscale preprocess scratch buffers";
    }
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  float* output_device = nullptr;
  if (!record_error(cudaMalloc(reinterpret_cast<void**>(&output_device), static_cast<size_t>(batch_size) * plane_elements * sizeof(float)),
                    error_message,
                    "failed to allocate TensorRT grayscale batch")) {
    return false;
  }
  auto owner = std::shared_ptr<void>(output_device, [](void* ptr) {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
  });

  constexpr int threads = 256;
  const int count = rows * cols;
  const int blocks = (count + threads - 1) / threads;
  for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
    const float* input_sample = resized_batch_device + static_cast<size_t>(sample_index) * plane_elements;
    float* output_sample = output_device + static_cast<size_t>(sample_index) * plane_elements;
    if (legacy_fast_gray_preprocess) {
      if (!quantile_normalize_sample(input_sample, count, 0.01f, 0.99f, scratch, stream, output_sample, error_message)) {
        return false;
      }
      quantize_kernel<<<blocks, threads, 0, stream>>>(output_sample, count, output_sample);
      if (!record_error(cudaGetLastError(), error_message, "legacy preprocess finalize kernel failed")) {
        return false;
      }
    } else if (!prepare_signal_agnostic_sample_cuda(input_sample, rows, cols, scratch, stream, output_sample, error_message)) {
      return false;
    }
  }

  output->data = output_device;
  output->owner = std::move(owner);
  return true;
}

bool project_tensorrt_raw_score_batch_cuda(const float* patch_features_batch_device,
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
                                           cudaStream_t cuda_stream,
                                           float* output_score_device,
                                           DinoCudaScoreBatch* output,
                                           std::string* error_message) {
  const int patch_count = patch_rows * patch_cols;
  if (output == nullptr || patch_features_batch_device == nullptr || batch_size <= 0 || patch_rows <= 0 || patch_cols <= 0 ||
      feature_dim <= 0 || aligned_rows <= 0 || aligned_cols <= 0 || output_rows <= 0 || output_cols <= 0 || patch_count <= 0) {
    if (error_message != nullptr) {
      *error_message = "invalid TensorRT raw score projection input";
    }
    return false;
  }

  static std::mutex scratch_mutex;
  std::lock_guard<std::mutex> lock(scratch_mutex);

  const float clamped_suppression = clamp_value(positional_suppression, 0.0f, 1.0f);
  const size_t xty_capacity = clamped_suppression > 0.0f
                                  ? static_cast<size_t>(batch_size) * static_cast<size_t>(kRawPositionalBasisDim) * static_cast<size_t>(feature_dim)
                                  : 0;
  const size_t raw_patch_capacity = static_cast<size_t>(batch_size) * static_cast<size_t>(patch_count);
  const size_t aligned_capacity = resized_full_chunk ? 0 : static_cast<size_t>(batch_size) * static_cast<size_t>(aligned_rows) * static_cast<size_t>(aligned_cols);
  auto& scratch = raw_score_scratch();
  if (!scratch.ensure_capacity(xty_capacity, raw_patch_capacity, aligned_capacity)) {
    if (error_message != nullptr) {
      *error_message = "failed to allocate TensorRT raw score projection scratch buffers";
    }
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  float* output_device = output_score_device;
  std::shared_ptr<void> owner;
  if (output_device == nullptr) {
    if (!record_error(cudaMalloc(reinterpret_cast<void**>(&output_device),
                                 static_cast<size_t>(batch_size) * static_cast<size_t>(output_rows) * static_cast<size_t>(output_cols) * sizeof(float)),
                      error_message,
                      "failed to allocate TensorRT raw score batch")) {
      return false;
    }
    owner = std::shared_ptr<void>(output_device, [](void* ptr) {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    });
  }

  constexpr int threads = 256;
  const dim3 patch_grid(static_cast<unsigned int>((patch_count + threads - 1) / threads), static_cast<unsigned int>(batch_size), 1U);
  if (clamped_suppression > 0.0f) {
    const auto cache = get_positional_suppression_device_cache(patch_rows, patch_cols);
    if (!cache || cache->patch_count != patch_count) {
      if (error_message != nullptr) {
        *error_message = "failed to prepare TensorRT positional suppression cache";
      }
      return false;
    }
    const dim3 feature_grid(static_cast<unsigned int>((feature_dim + threads - 1) / threads),
                            static_cast<unsigned int>(kRawPositionalBasisDim),
                            static_cast<unsigned int>(batch_size));
    raw_positional_xty_kernel<<<feature_grid, threads, 0, stream>>>(patch_features_batch_device,
                                                                     cache->design_device,
                                                                     patch_count,
                                                                     feature_dim,
                                                                     kRawPositionalBasisDim,
                                                                     scratch.xty);
    raw_positional_beta_kernel<<<feature_grid, threads, 0, stream>>>(cache->inverse_gram_device,
                                                                      scratch.xty,
                                                                      feature_dim,
                                                                      kRawPositionalBasisDim,
                                                                      scratch.beta);
    raw_patch_energy_kernel<<<patch_grid, threads, 0, stream>>>(patch_features_batch_device,
                                                                 cache->design_device,
                                                                 scratch.beta,
                                                                 patch_count,
                                                                 feature_dim,
                                                                 kRawPositionalBasisDim,
                                                                 clamped_suppression,
                                                                 scratch.raw_patch);
  } else {
    raw_patch_energy_plain_kernel<<<patch_grid, threads, 0, stream>>>(patch_features_batch_device,
                                                                       patch_count,
                                                                       feature_dim,
                                                                       scratch.raw_patch);
  }
  if (!record_error(cudaGetLastError(), error_message, "TensorRT raw score patch-energy kernels failed")) {
    return false;
  }

  if (resized_full_chunk) {
    if (output_rows == patch_rows && output_cols == patch_cols) {
      if (!record_error(cudaMemcpyAsync(output_device,
                                        scratch.raw_patch,
                                        static_cast<size_t>(batch_size) * static_cast<size_t>(patch_count) * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        stream),
                        error_message,
                        "TensorRT raw score direct copy failed")) {
        return false;
      }
      output->data = output_device;
      output->owner = std::move(owner);
      return true;
    }
    const int output_total = batch_size * output_rows * output_cols;
    const int output_blocks = (output_total + threads - 1) / threads;
    resize_bilinear_batch_kernel<<<output_blocks, threads, 0, stream>>>(scratch.raw_patch,
                                                                         batch_size,
                                                                         patch_rows,
                                                                         patch_cols,
                                                                         output_rows,
                                                                         output_cols,
                                                                         output_device);
    if (!record_error(cudaGetLastError(), error_message, "TensorRT raw score direct resize failed")) {
      return false;
    }
  } else {
    const int aligned_total = batch_size * aligned_rows * aligned_cols;
    const int aligned_blocks = (aligned_total + threads - 1) / threads;
    resize_bilinear_batch_kernel<<<aligned_blocks, threads, 0, stream>>>(scratch.raw_patch,
                                                                          batch_size,
                                                                          patch_rows,
                                                                          patch_cols,
                                                                          aligned_rows,
                                                                          aligned_cols,
                                                                          scratch.aligned_map);
    if (!record_error(cudaGetLastError(), error_message, "TensorRT raw score aligned resize failed")) {
      return false;
    }

    const int output_total = batch_size * output_rows * output_cols;
    const int output_blocks = (output_total + threads - 1) / threads;
    project_aligned_maps_to_output_kernel<<<output_blocks, threads, 0, stream>>>(scratch.aligned_map,
                                                                                  batch_size,
                                                                                  aligned_rows,
                                                                                  aligned_cols,
                                                                                  output_rows,
                                                                                  output_cols,
                                                                                  output_device);
    if (!record_error(cudaGetLastError(), error_message, "TensorRT raw score output projection failed")) {
      return false;
    }
  }

  output->data = output_device;
  output->owner = std::move(owner);
  return true;
}

}  // namespace holoscan::ops
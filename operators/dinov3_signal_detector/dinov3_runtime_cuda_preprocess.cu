// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "dinov3_runtime_cuda_preprocess.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

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

}  // namespace holoscan::ops
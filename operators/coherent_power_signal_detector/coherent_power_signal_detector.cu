// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "coherent_power_signal_detector.hpp"
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

#include <gxf/core/gxf.h>

namespace {

using coherent_power_complex = holoscan::ops::coherent_power_complex;

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

constexpr char kHardwiredPowerAssistMode[] = "absolute_direct";
constexpr char kHardwiredCoherenceSourceMode[] = "power_assist";

enum TimingStageIndex : size_t {
  kInputStage = 0,
  kPowerDbStage,
  kPipelineStage,
  kDeviceCopyStage,
  kMaskSaveStage,
  kTotalStage,
};

constexpr std::array<const char*, holoscan::ops::CoherentPowerSignalDetector::kTimingStageCount>
    kTimingStageNames = {
        "input_ms",
        "power_db_ms",
      "pipeline_ms",
        "device_copy_ms",
        "mask_save_ms",
        "total_ms",
    };

constexpr uint64_t kPathArtifactCaptureFrame = 8;
constexpr uint64_t kPathArtifactStopGraceFrames = 2;
constexpr std::string_view kReferencePathArtifactDir = "/workspace/coherent_power_snapshots/live_reference_debug";
constexpr int kEmitMorphOpenRows = 3;
constexpr int kEmitMorphOpenCols = 7;
constexpr int kEmitMorphCloseRows = 5;
constexpr int kEmitMorphCloseCols = 21;
constexpr std::string_view kPerformancePathArtifactDir = "/workspace/coherent_power_snapshots/performance_path_debug";

struct PathArtifactStopTracker {
  std::mutex mutex;
  std::vector<int> active_channels;
  std::vector<int> captured_channels;
  uint64_t stop_after_frame = 0;
  bool stop_pending = false;
  bool interrupt_requested = false;
};

PathArtifactStopTracker& path_artifact_stop_tracker() {
  static PathArtifactStopTracker tracker;
  return tracker;
}

std::vector<uint8_t> reduce_mask_for_history_rows(const std::vector<uint8_t>& mask_pixels,
                                                  int src_width,
                                                  int src_height,
                                                  int dst_width,
                                                  int dst_rows) {
  if (src_width <= 0 || src_height <= 0 || mask_pixels.empty() || dst_width <= 0 || dst_rows <= 0) {
    return {};
  }

  std::vector<uint8_t> reduced(static_cast<size_t>(dst_width) * static_cast<size_t>(dst_rows), 0);
  for (int row = 0; row < dst_rows; ++row) {
    const int src_row_start = (row * src_height) / dst_rows;
    const int src_row_end = std::max(src_row_start + 1, ((row + 1) * src_height) / dst_rows);
    for (int col = 0; col < dst_width; ++col) {
      const int src_col_start = (col * src_width) / dst_width;
      const int src_col_end = std::max(src_col_start + 1, ((col + 1) * src_width) / dst_width);
      int active = 0;
      int count = 0;
      for (int src_row = src_row_start; src_row < src_row_end; ++src_row) {
        for (int src_col = src_col_start; src_col < src_col_end; ++src_col) {
          active += mask_pixels[static_cast<size_t>(src_row) * static_cast<size_t>(src_width) +
                                static_cast<size_t>(src_col)] > 0
                        ? 1
                        : 0;
          ++count;
        }
      }
      const float occupancy = count > 0 ? static_cast<float>(active) / static_cast<float>(count) : 0.0f;
      reduced[static_cast<size_t>(row) * static_cast<size_t>(dst_width) + static_cast<size_t>(col)] =
          static_cast<uint8_t>(std::lround(std::clamp(occupancy, 0.0f, 1.0f) * 255.0f));
    }
  }
  return reduced;
}

std::mutex& mask_output_buffer_pool_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<size_t, std::vector<void*>>& mask_output_buffer_pool() {
  static std::unordered_map<size_t, std::vector<void*>> pool;
  return pool;
}

void* acquire_mask_output_buffer(size_t bytes) {
  std::lock_guard<std::mutex> lock(mask_output_buffer_pool_mutex());
  auto& pool = mask_output_buffer_pool();
  auto it = pool.find(bytes);
  if (it != pool.end() && !it->second.empty()) {
    void* ptr = it->second.back();
    it->second.pop_back();
    return ptr;
  }
  void* ptr = nullptr;
  auto result = cudaMalloc(&ptr, bytes);
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(result));
  }
  return ptr;
}

void recycle_mask_output_buffer(void* ptr, size_t bytes) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mask_output_buffer_pool_mutex());
  mask_output_buffer_pool()[bytes].push_back(ptr);
}

std::shared_ptr<unsigned int> acquire_pooled_u32_buffer() {
  constexpr size_t kBytes = sizeof(unsigned int);
  return std::shared_ptr<unsigned int>(static_cast<unsigned int*>(acquire_mask_output_buffer(kBytes)),
                                       [](unsigned int* ptr) { recycle_mask_output_buffer(ptr, kBytes); });
}

std::shared_ptr<uint8_t> allocate_owned_u8_buffer(size_t bytes) {
  void* ptr = nullptr;
  auto result = cudaMalloc(&ptr, bytes);
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(result));
  }
  return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(ptr), [](uint8_t* buffer) {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
  });
}

std::shared_ptr<unsigned int> allocate_owned_u32_buffer() {
  void* ptr = nullptr;
  auto result = cudaMalloc(&ptr, sizeof(unsigned int));
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(result));
  }
  return std::shared_ptr<unsigned int>(static_cast<unsigned int*>(ptr), [](unsigned int* buffer) {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
  });
}

struct FastGpuMetadataSummary {
  int subsection_count = 1;
  int grouped_box_count = 0;
  int ignore_bins_per_side = 0;
  float merged_threshold = 0.0f;
  float seed_threshold = 0.0f;
  uint32_t raw_mask_nonzero_pixels = 0;
  uint32_t post_emit_close_mask_nonzero_pixels = 0;
  uint32_t post_emit_persistence_mask_nonzero_pixels = 0;
  uint32_t post_smooth_mask_nonzero_pixels = 0;
  float always_on_floor_db = 0.0f;
  uint32_t always_on_stripe_count = 0;
};

struct CanonicalTensorView {
  int rows = 0;
  int cols = 0;
  bool transposed = false;
};

int compute_ignore_bins_per_side(int rows,
                                 double resolution_hz,
                                 double ignore_sideband_percent,
                                 double ignore_sideband_hz) {
  int ignore_bins_per_side = 0;
  if (ignore_sideband_percent > 0.0) {
    ignore_bins_per_side = static_cast<int>(
        std::floor(static_cast<double>(rows) * ignore_sideband_percent / 100.0));
  } else if (resolution_hz > 0.0 && ignore_sideband_hz > 0.0) {
    ignore_bins_per_side = static_cast<int>(std::ceil(ignore_sideband_hz / resolution_hz));
  }
  return std::clamp(ignore_bins_per_side, 0, std::max(0, (rows - 16) / 2));
}

std::string json_bool(bool value) {
  return value ? "true" : "false";
}


std::string json_escape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += ch;
        break;
    }
  }
  return escaped;
}

std::string make_debug_artifact_stem(const std::string& output_dir,
                                     uint16_t channel,
                                     uint64_t frame_number,
                                     int rows,
                                     int cols) {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  std::ostringstream oss;
  oss << output_dir
      << "/coherent_power_snapshot_ch" << channel
      << "_f" << frame_number
      << "_" << ms
      << "_" << rows << "x" << cols;
  return oss.str();
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
      << "/coherent_power_mask_ch" << channel
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

bool write_npy_2d(const std::string& path,
                  const void* payload,
                  size_t payload_bytes,
                  int rows,
                  int cols,
                  const std::string& dtype_descr) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }

  std::ostringstream header_builder;
  header_builder << "{'descr': '" << dtype_descr << "', 'fortran_order': False, 'shape': ("
                 << rows << ", " << cols << "), }";
  std::string header = header_builder.str();

  const size_t preamble_size = 10;
  const size_t padding = (16 - ((preamble_size + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');

  if (header.size() > std::numeric_limits<uint16_t>::max()) {
    return false;
  }

  out.write("\x93NUMPY", 6);
  const unsigned char version[2] = {1, 0};
  out.write(reinterpret_cast<const char*>(version), 2);
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  const unsigned char header_len_le[2] = {
      static_cast<unsigned char>(header_len & 0xFF),
      static_cast<unsigned char>((header_len >> 8) & 0xFF),
  };
  out.write(reinterpret_cast<const char*>(header_len_le), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(payload), static_cast<std::streamsize>(payload_bytes));
  return out.good();
}

// Minimal reader for a 1-D little-endian float32 (.npy v1.0) array, used to load the
// calibrated per-frequency noise floor. Returns the flattened element count regardless of
// 1-D (rows,) or 2-D (rows,1) shape. Sets ok=false on any parse/type mismatch.
std::vector<float> read_npy_float32(const std::string& path, bool& ok) {
  ok = false;
  std::vector<float> values;
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    return values;
  }
  char magic[6];
  in.read(magic, 6);
  if (!in.good() || std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
    return values;
  }
  unsigned char version[2];
  in.read(reinterpret_cast<char*>(version), 2);
  unsigned char header_len_le[2];
  in.read(reinterpret_cast<char*>(header_len_le), 2);
  const size_t header_len = static_cast<size_t>(header_len_le[0]) |
                            (static_cast<size_t>(header_len_le[1]) << 8);
  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header_len));
  if (!in.good()) {
    return values;
  }
  // dtype must be little-endian / not-byte-order float32.
  if (header.find("'<f4'") == std::string::npos && header.find("'|f4'") == std::string::npos) {
    return values;
  }
  // Parse the shape tuple and multiply its dims to get the element count.
  const auto shape_pos = header.find("'shape':");
  const auto open_paren = header.find('(', shape_pos);
  const auto close_paren = header.find(')', open_paren);
  if (shape_pos == std::string::npos || open_paren == std::string::npos ||
      close_paren == std::string::npos) {
    return values;
  }
  size_t count = 1;
  bool any_dim = false;
  const std::string dims = header.substr(open_paren + 1, close_paren - open_paren - 1);
  size_t cursor = 0;
  while (cursor < dims.size()) {
    while (cursor < dims.size() && !std::isdigit(static_cast<unsigned char>(dims[cursor]))) {
      ++cursor;
    }
    if (cursor >= dims.size()) {
      break;
    }
    size_t value = 0;
    while (cursor < dims.size() && std::isdigit(static_cast<unsigned char>(dims[cursor]))) {
      value = value * 10 + static_cast<size_t>(dims[cursor] - '0');
      ++cursor;
    }
    count *= value;
    any_dim = true;
  }
  if (!any_dim || count == 0) {
    return values;
  }
  values.resize(count);
  in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
  if (!in.good() && !in.eof()) {
    values.clear();
    return values;
  }
  ok = true;
  return values;
}

std::vector<uint8_t> float_unit_map_to_u8(const std::vector<float>& values) {
  std::vector<uint8_t> image(values.size(), 0);
  for (size_t index = 0; index < values.size(); ++index) {
    const float clamped = std::min(1.0f, std::max(0.0f, values[index]));
    image[index] = static_cast<uint8_t>(std::lround(clamped * 255.0f));
  }
  return image;
}

std::vector<uint8_t> binary_float_mask_to_u8(const std::vector<float>& values, float threshold = 0.5f) {
  std::vector<uint8_t> image(values.size(), 0);
  for (size_t index = 0; index < values.size(); ++index) {
    image[index] = values[index] > threshold ? 255 : 0;
  }
  return image;
}

CanonicalTensorView canonical_tensor_view(int input_rows, int input_cols) {
  CanonicalTensorView view;
  view.transposed = input_rows < input_cols;
  view.rows = view.transposed ? input_cols : input_rows;
  view.cols = view.transposed ? input_rows : input_cols;
  return view;
}

constexpr float kPi = 3.14159265358979323846f;

__host__ __device__ inline size_t flat_index(int rows, int cols, int row, int col) {
  return static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
}

__global__ void coherent_power_power_db_kernel(const cuda::std::complex<float>* input,
                                               int src_rows,
                                               int src_cols,
                                               float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = src_rows * src_cols;
  if (idx >= total) {
    return;
  }

  const auto value = input[idx];
  const float re = value.real();
  const float im = value.imag();
  const float power = re * re + im * im + 1e-12f;
  output[idx] = 10.0f * log10f(power);
}

__global__ void coherent_power_row_mean_kernel(const float* input,
                                               int rows,
                                               int cols,
                                               float* row_mean) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float partial[256];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    sum += input[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
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

__global__ void coherent_power_row_sampled_mean_kernel(const float* input,
                                                       int rows,
                                                       int cols,
                                                       int col_stride,
                                                       float* row_mean) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float partial_sum[256];
  __shared__ unsigned int partial_count[256];
  const int tid = threadIdx.x;
  const int sample_stride = max(col_stride, 1);
  float sum = 0.0f;
  unsigned int count = 0;
  for (int col = tid * sample_stride; col < cols; col += blockDim.x * sample_stride) {
    sum += input[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    ++count;
  }
  partial_sum[tid] = sum;
  partial_count[tid] = count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_sum[tid] += partial_sum[tid + stride];
      partial_count[tid] += partial_count[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    row_mean[row] = partial_count[0] > 0 ? partial_sum[0] / static_cast<float>(partial_count[0]) : 0.0f;
  }
}

__global__ void coherent_power_row_capped_mean_from_reference_kernel(const float* input,
                                                                     int rows,
                                                                     int cols,
                                                                     const float* reference_level,
                                                                     float cap_headroom_db,
                                                                     float* row_mean) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float partial[256];
  const int tid = threadIdx.x;
  const float cap_db = reference_level[0] + fmaxf(cap_headroom_db, 0.0f);
  float sum = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float value = input[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    sum += fminf(value, cap_db);
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

__global__ void coherent_power_gaussian_smooth_rows_kernel(const float* input,
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

__global__ void coherent_power_frontend_reference_kernel(const float* row_smooth,
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

__global__ void coherent_power_frontend_correction_kernel(const float* input,
                                                          int rows,
                                                          int cols,
                                                          const float* row_smooth,
                                                          const float* reference_level,
                                                          float max_boost_db,
                                                          float* corrected) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const float boost = fminf(fmaxf(reference_level[0] - row_smooth[row], 0.0f), max_boost_db);
  corrected[idx] = input[idx] + boost;
}

__global__ void coherent_power_box_mean_cols_kernel(const float* input,
                                                    int rows,
                                                    int cols,
                                                    int radius_cols,
                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const int col = idx % cols;
  const int col_start = max(0, col - radius_cols);
  const int col_stop = min(cols - 1, col + radius_cols);

  const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(cols);
  float sum = 0.0f;
  int count = 0;
  for (int src_col = col_start; src_col <= col_stop; ++src_col) {
    sum += input[row_offset + static_cast<size_t>(src_col)];
    ++count;
  }
  output[idx] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void coherent_power_box_mean_rows_kernel(const float* input,
                                                    int rows,
                                                    int cols,
                                                    int radius_rows,
                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const int col = idx % cols;
  const int row_start = max(0, row - radius_rows);
  const int row_stop = min(rows - 1, row + radius_rows);

  float sum = 0.0f;
  int count = 0;
  for (int src_row = row_start; src_row <= row_stop; ++src_row) {
    const size_t src_index = static_cast<size_t>(src_row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
    sum += input[src_index];
    ++count;
  }
  output[idx] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void coherent_power_fast_power_assist_score_kernel(const float* corrected,
                                                              const float* background,
                                                              int rows,
                                                              int cols,
                                                              int ignore_bins_per_side,
                                                              float power_floor_db,
                                                              float power_span_db,
                                                              float score_threshold,
                                                              float* score,
                                                              uint8_t* mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const bool valid_row = row >= ignore_bins_per_side && row < (rows - ignore_bins_per_side);
  if (!valid_row) {
    score[idx] = 0.0f;
    mask[idx] = 0;
    return;
  }

  const float support_db = corrected[idx] - background[idx];
  const float support = fminf(fmaxf((support_db - power_floor_db) / fmaxf(power_span_db, 1e-6f), 0.0f), 1.0f);
  score[idx] = support;
  mask[idx] = (support >= score_threshold && support > 0.0f) ? 1 : 0;
}

// Calibrated per-frequency noise-floor fill: OR a pixel into the mask when its absolute
// corrected power exceeds the per-row floor by offset_db. Unlike the local box-mean support
// (which hollows out the interior of signals wider than the box), this uses the per-row noise
// floor, so a strong steady/wideband signal fills in solidly. Runs AFTER the box score kernel
// and only raises the score, never lowers it.
__global__ void coherent_power_per_freq_fill_kernel(const float* corrected,
                                                    const float* per_freq_floor_db,
                                                    int rows,
                                                    int cols,
                                                    int ignore_bins_per_side,
                                                    float offset_db,
                                                    float span_db,
                                                    float* score,
                                                    uint8_t* mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  if (row < ignore_bins_per_side || row >= (rows - ignore_bins_per_side)) {
    return;
  }
  const float floor_db = per_freq_floor_db[row];
  if (corrected[idx] > floor_db + offset_db) {
    mask[idx] = 1;
    const float grade = fminf(fmaxf((corrected[idx] - floor_db) / fmaxf(span_db, 1e-6f), 0.0f), 1.0f);
    score[idx] = fmaxf(score[idx], grade);
  }
}

// Fill a float buffer with a constant. Used to seed the dynamic per-frequency floor to a high bar
// so every bin starts above any plausible signal and can only descend from there.
__global__ void coherent_power_fill_float_kernel(float* buf, int n, float value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  buf[idx] = value;
}

// Dynamic per-frequency floor update (one block per frequency row). Computes a robust per-row
// high-power statistic over this frame's time bins -- mean + std_k * std of corrected_db, a cheap
// single-pass proxy for the noise high-quantile the offline calibration measures -- then folds it
// into a windowed minimum: a small ring of sub-window minima per bin. The current slot accumulates
// the running min of the statistic; on the first frame of a slot the slot is overwritten (so the
// oldest window ages out), and the published floor is the min across all slots. This tracks the
// noise floor downward while bounding the creep a strictly-global min would accumulate over a long
// run. Bins that are noise most of the time settle onto their floor within a window; an always-on
// signal never presents a quiet frame and keeps its bin high (ignored by design). Ignore-band rows
// are left untouched.
__global__ void coherent_power_dynamic_floor_update_kernel(const float* corrected,
                                                           int rows,
                                                           int cols,
                                                           int ignore_bins_per_side,
                                                           float std_k,
                                                           float* ring,
                                                           int window_slots,
                                                           int cur_slot,
                                                           int first_frame_of_slot,
                                                           float* floor_db) {
  const int row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float partial_sum[256];
  __shared__ float partial_sq[256];
  __shared__ unsigned int partial_count[256];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  float sq = 0.0f;
  unsigned int count = 0;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float v = corrected[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    sum += v;
    sq += v * v;
    ++count;
  }
  partial_sum[tid] = sum;
  partial_sq[tid] = sq;
  partial_count[tid] = count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial_sum[tid] += partial_sum[tid + stride];
      partial_sq[tid] += partial_sq[tid + stride];
      partial_count[tid] += partial_count[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (row < ignore_bins_per_side || row >= (rows - ignore_bins_per_side)) {
      return;
    }
    const unsigned int n = partial_count[0];
    if (n == 0) {
      return;
    }
    const float mean = partial_sum[0] / static_cast<float>(n);
    const float var = fmaxf(partial_sq[0] / static_cast<float>(n) - mean * mean, 0.0f);
    const float stat = mean + std_k * sqrtf(var);

    const size_t base = static_cast<size_t>(row) * static_cast<size_t>(window_slots);
    // Overwrite the slot on its first frame (retiring the window it previously held), otherwise fold
    // the statistic into the slot's running minimum.
    ring[base + cur_slot] = first_frame_of_slot ? stat : fminf(ring[base + cur_slot], stat);
    float f = ring[base];
    for (int w = 1; w < window_slots; ++w) {
      f = fminf(f, ring[base + w]);
    }
    floor_db[row] = f;
  }
}

// Strong-signal rescue candidate: mark a pixel when its absolute corrected power exceeds the
// per-row (per-frequency) noise floor by excess_db. Built in the detector's internal
// orientation (rows = frequency, cols = time). Unlike the box-mean support (which the emit-mask
// opening + frequency-persistence pass erases for frequency-narrow features), this candidate is
// OR-ed back in AFTER those width filters, so a strong narrow signal survives regardless of its
// frequency width. Time persistence is enforced separately downstream.
__global__ void coherent_power_strong_rescue_kernel(const float* corrected,
                                                    const float* row_floor_db,
                                                    int rows,
                                                    int cols,
                                                    int ignore_bins_per_side,
                                                    float excess_db,
                                                    uint8_t* strong_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  if (row < ignore_bins_per_side || row >= (rows - ignore_bins_per_side)) {
    strong_mask[idx] = 0;
    return;
  }
  strong_mask[idx] = (corrected[idx] >= row_floor_db[row] + excess_db) ? 1 : 0;
}

// Elementwise OR of a source mask into a destination mask (both u8, same layout).
__global__ void coherent_power_or_u8_kernel(uint8_t* dst, const uint8_t* src, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  if (src[idx]) {
    dst[idx] = 1;
  }
}

__global__ void coherent_power_majority_smooth_kernel(const uint8_t* input,
                                                      int rows,
                                                      int cols,
                                                      int ignore_bins_per_side,
                                                      uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const int col = idx % cols;
  if (row < ignore_bins_per_side || row >= (rows - ignore_bins_per_side)) {
    output[idx] = 0;
    return;
  }

  int sum = 0;
  int count = 0;
  for (int d_row = -1; d_row <= 1; ++d_row) {
    const int src_row = max(0, min(rows - 1, row + d_row));
    if (src_row < ignore_bins_per_side || src_row >= (rows - ignore_bins_per_side)) {
      continue;
    }
    for (int d_col = -1; d_col <= 1; ++d_col) {
      const int src_col = max(0, min(cols - 1, col + d_col));
      sum += input[static_cast<size_t>(src_row) * static_cast<size_t>(cols) + static_cast<size_t>(src_col)] ? 1 : 0;
      ++count;
    }
  }

  output[idx] = (sum * 2 >= max(count, 1)) ? 1 : 0;
}

__global__ void count_nonzero_u8_kernel(const uint8_t* input, int total, unsigned int* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  if (input[idx] > 0) {
    atomicAdd(output, 1U);
  }
}

float clamp_float(float value, float low, float high) {
  return std::max(low, std::min(high, value));
}

__global__ void coherent_power_binary_dilate_freq_kernel(const uint8_t* input,
                                                         int rows,
                                                         int cols,
                                                         int radius,
                                                         uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  const int col = idx % cols;
  uint8_t value = 0;
  for (int src_row = max(0, row - radius); src_row <= min(rows - 1, row + radius); ++src_row) {
    if (input[flat_index(rows, cols, src_row, col)]) {
      value = 1;
      break;
    }
  }
  output[idx] = value;
}

__global__ void coherent_power_binary_erode_freq_kernel(const uint8_t* input,
                                                        int rows,
                                                        int cols,
                                                        int radius,
                                                        uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  const int col = idx % cols;
  uint8_t value = 1;
  for (int src_row = max(0, row - radius); src_row <= min(rows - 1, row + radius); ++src_row) {
    if (!input[flat_index(rows, cols, src_row, col)]) {
      value = 0;
      break;
    }
  }
  output[idx] = value;
}

__global__ void coherent_power_binary_dilate_cols_kernel(const uint8_t* input,
                                                         int rows,
                                                         int cols,
                                                         int radius,
                                                         uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  const int col = idx % cols;
  uint8_t value = 0;
  for (int src_col = max(0, col - radius); src_col <= min(cols - 1, col + radius); ++src_col) {
    if (input[flat_index(rows, cols, row, src_col)]) {
      value = 1;
      break;
    }
  }
  output[idx] = value;
}

__global__ void coherent_power_binary_erode_cols_kernel(const uint8_t* input,
                                                        int rows,
                                                        int cols,
                                                        int radius,
                                                        uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  const int col = idx % cols;
  uint8_t value = 1;
  for (int src_col = max(0, col - radius); src_col <= min(cols - 1, col + radius); ++src_col) {
    if (!input[flat_index(rows, cols, row, src_col)]) {
      value = 0;
      break;
    }
  }
  output[idx] = value;
}

__global__ void coherent_power_frequency_persistence_kernel(const uint8_t* input,
                                                            int rows,
                                                            int cols,
                                                            int radius,
                                                            int min_hits,
                                                            uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  if (!input[idx]) {
    output[idx] = 0;
    return;
  }

  const int row = idx / cols;
  const int col = idx % cols;
  int hits = 0;
  for (int offset = -radius; offset <= radius; ++offset) {
    const int src_col = max(0, min(cols - 1, col + offset));
    hits += input[flat_index(rows, cols, row, src_col)] ? 1 : 0;
    if (hits >= min_hits) {
      output[idx] = 1;
      return;
    }
  }

  output[idx] = 0;
}

int clamp_int(int value, int low, int high) {
  return std::max(low, std::min(high, value));
}

__global__ void coherent_power_transpose_kernel(const holoscan::ops::coherent_power_complex* input,
                                                int input_rows,
                                                int input_cols,
                                                holoscan::ops::coherent_power_complex* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = input_rows * input_cols;
  if (index >= total) {
    return;
  }
  const int row = index / input_cols;
  const int col = index % input_cols;
  output[flat_index(input_cols, input_rows, col, row)] = input[index];
}

__global__ void transpose_u8_kernel(const uint8_t* input,
                                    int input_rows,
                                    int input_cols,
                                    uint8_t* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = input_rows * input_cols;
  if (index >= total) {
    return;
  }
  const int row = index / input_cols;
  const int col = index % input_cols;
  output[flat_index(input_cols, input_rows, col, row)] = input[index];
}

void apply_emit_mask_morphology(uint8_t* mask_device,
                                int rows,
                                int cols,
                                uint8_t* scratch0_device,
                                uint8_t* scratch1_device,
                                int freq_persistence_window,
                                int freq_persistence_min_hits,
                                unsigned int* post_close_nonzero_device,
                                unsigned int* post_persistence_nonzero_device,
                                cudaStream_t stream) {
  if (mask_device == nullptr || scratch0_device == nullptr || scratch1_device == nullptr || rows <= 0 || cols <= 0) {
    return;
  }

  constexpr int threads = 256;
  const int total = rows * cols;
  const int blocks = (total + threads - 1) / threads;
  const int open_row_radius = std::max(0, (kEmitMorphOpenRows - 1) / 2);
  const int open_col_radius = std::max(0, (kEmitMorphOpenCols - 1) / 2);
  const int close_row_radius = std::max(0, (kEmitMorphCloseRows - 1) / 2);
  const int close_col_radius = std::max(0, (kEmitMorphCloseCols - 1) / 2);

  coherent_power_binary_erode_freq_kernel<<<blocks, threads, 0, stream>>>(mask_device,
                                                                           rows,
                                                                           cols,
                                                                           open_row_radius,
                                                                           scratch0_device);
  auto kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask open-axis0 erode kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  coherent_power_binary_erode_cols_kernel<<<blocks, threads, 0, stream>>>(scratch0_device,
                                                                           rows,
                                                                           cols,
                                                                           open_col_radius,
                                                                           scratch1_device);
  kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask open-axis1 erode kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  coherent_power_binary_dilate_freq_kernel<<<blocks, threads, 0, stream>>>(scratch1_device,
                                                                            rows,
                                                                            cols,
                                                                            open_row_radius,
                                                                            scratch0_device);
  kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask open-axis0 dilate kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  coherent_power_binary_dilate_cols_kernel<<<blocks, threads, 0, stream>>>(scratch0_device,
                                                                            rows,
                                                                            cols,
                                                                            open_col_radius,
                                                                            mask_device);
  kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask open-axis1 dilate kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  coherent_power_binary_dilate_freq_kernel<<<blocks, threads, 0, stream>>>(mask_device,
                                                                            rows,
                                                                            cols,
                                                                            close_row_radius,
                                                                            scratch0_device);
  kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask close-axis0 dilate kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  coherent_power_binary_dilate_cols_kernel<<<blocks, threads, 0, stream>>>(scratch0_device,
                                                                            rows,
                                                                            cols,
                                                                            close_col_radius,
                                                                            scratch1_device);
  kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask close-axis1 dilate kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  coherent_power_binary_erode_freq_kernel<<<blocks, threads, 0, stream>>>(scratch1_device,
                                                                           rows,
                                                                           cols,
                                                                           close_row_radius,
                                                                           scratch0_device);
  kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask close-axis0 erode kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  coherent_power_binary_erode_cols_kernel<<<blocks, threads, 0, stream>>>(scratch0_device,
                                                                           rows,
                                                                           cols,
                                                                           close_col_radius,
                                                                           mask_device);
  kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("emit mask close-axis1 erode kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  if (post_close_nonzero_device != nullptr) {
    kernel_result = cudaMemsetAsync(post_close_nonzero_device, 0, sizeof(unsigned int), stream);
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("emit mask post-close counter reset failed: ") +
                               cudaGetErrorString(kernel_result));
    }
    count_nonzero_u8_kernel<<<blocks, threads, 0, stream>>>(mask_device,
                                                             total,
                                                             post_close_nonzero_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("emit mask post-close count kernel launch failed: ") +
                               cudaGetErrorString(kernel_result));
    }
  }

  const int persistence_window = std::max(1, freq_persistence_window | 1);
  const int persistence_min_hits = clamp_int(freq_persistence_min_hits, 1, persistence_window);
  if (persistence_window > 1 && persistence_min_hits > 1) {
    const int persistence_radius = (persistence_window - 1) / 2;
    coherent_power_frequency_persistence_kernel<<<blocks, threads, 0, stream>>>(mask_device,
                                                                                 rows,
                                                                                 cols,
                                                                                 persistence_radius,
                                                                                 persistence_min_hits,
                                                                                 scratch0_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("emit mask frequency persistence kernel launch failed: ") +
                               cudaGetErrorString(kernel_result));
    }

    kernel_result = cudaMemcpyAsync(mask_device,
                                    scratch0_device,
                                    static_cast<size_t>(total) * sizeof(uint8_t),
                                    cudaMemcpyDeviceToDevice,
                                    stream);
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("emit mask frequency persistence device copy failed: ") +
                               cudaGetErrorString(kernel_result));
    }

    if (post_persistence_nonzero_device != nullptr) {
      kernel_result = cudaMemsetAsync(post_persistence_nonzero_device, 0, sizeof(unsigned int), stream);
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("emit mask post-persistence counter reset failed: ") +
                                 cudaGetErrorString(kernel_result));
      }
      count_nonzero_u8_kernel<<<blocks, threads, 0, stream>>>(mask_device,
                                                               total,
                                                               post_persistence_nonzero_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("emit mask post-persistence count kernel launch failed: ") +
                                 cudaGetErrorString(kernel_result));
      }
    }
  } else if (post_persistence_nonzero_device != nullptr) {
    kernel_result = cudaMemsetAsync(post_persistence_nonzero_device, 0, sizeof(unsigned int), stream);
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("emit mask post-persistence counter reset failed: ") +
                               cudaGetErrorString(kernel_result));
    }
    count_nonzero_u8_kernel<<<blocks, threads, 0, stream>>>(mask_device,
                                                             total,
                                                             post_persistence_nonzero_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("emit mask post-persistence count kernel launch failed: ") +
                               cudaGetErrorString(kernel_result));
    }
  }
}

float quantile_from_values(std::vector<float> values, float q) {
  if (values.empty()) {
    return 0.0f;
  }
  q = clamp_float(q, 0.0f, 1.0f);
  const size_t nth_index = static_cast<size_t>(std::llround(q * static_cast<float>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(nth_index), values.end());
  return values[nth_index];
}

float percentile_from_values(std::vector<float> values, float percentile) {
  return quantile_from_values(std::move(values), percentile / 100.0f);
}

std::vector<float> normalize_map01_local(const std::vector<float>& input,
                                         float low_q,
                                         float high_q) {
  std::vector<float> finite_values;
  finite_values.reserve(input.size());
  for (const float value : input) {
    if (std::isfinite(value)) {
      finite_values.push_back(value);
    }
  }
  if (finite_values.empty()) {
    return std::vector<float>(input.size(), 0.0f);
  }
  float low = percentile_from_values(finite_values, low_q);
  float high = percentile_from_values(std::move(finite_values), high_q);
  if (high <= low) {
    high = low + 1e-6f;
  }
  std::vector<float> output(input.size(), 0.0f);
  const float denom = high - low;
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = clamp_float((input[index] - low) / denom, 0.0f, 1.0f);
  }
  return output;
}

bool local_relative_power_support_map_cuda(const float* sxx_db_local,
                                           int rows,
                                           int cols,
                                           const std::vector<uint8_t>& valid_row_mask,
                                           float floor_q,
                                           int freq_window,
                                           int time_window,
                                           std::vector<float>& support);

std::vector<float> label_mask_connected_components(const std::vector<uint8_t>& mask,
                                                   int rows,
                                                   int cols,
                                                   const std::vector<uint8_t>& valid_row_mask) {
  std::vector<float> component_labels(mask.size(), 0.0f);
  if (mask.empty()) {
    return component_labels;
  }

  std::vector<uint8_t> visited(mask.size(), 0);
  const std::array<std::pair<int, int>, 4> neighbors{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
  int component_id = 0;
  for (int row = 0; row < rows; ++row) {
    if (!valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      const size_t seed = flat_index(rows, cols, row, col);
      if (!mask[seed] || visited[seed]) {
        continue;
      }
      ++component_id;
      std::queue<std::pair<int, int>> queue;
      queue.push({row, col});
      visited[seed] = 1;
      component_labels[seed] = static_cast<float>(component_id);
      while (!queue.empty()) {
        const auto [current_row, current_col] = queue.front();
        queue.pop();
        for (const auto& [delta_row, delta_col] : neighbors) {
          const int next_row = current_row + delta_row;
          const int next_col = current_col + delta_col;
          if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
            continue;
          }
          if (!valid_row_mask[static_cast<size_t>(next_row)]) {
            continue;
          }
          const size_t next_index = flat_index(rows, cols, next_row, next_col);
          if (!mask[next_index] || visited[next_index]) {
            continue;
          }
          visited[next_index] = 1;
          component_labels[next_index] = static_cast<float>(component_id);
          queue.push({next_row, next_col});
        }
      }
    }
  }
  return component_labels;
}



}  // namespace

namespace holoscan::ops {


CoherentPowerSignalDetector::~CoherentPowerSignalDetector() {
  for (auto& buffers : channel_buffers_) {
    cudaFreeHost(buffers.input_tensor_host);
    cudaFree(buffers.analysis_tensor_device);
    cudaFree(buffers.power_db_device);
    cudaFree(buffers.corrected_db_device);
    cudaFree(buffers.time_mean_device);
    cudaFree(buffers.freq_mean_device);
    cudaFree(buffers.background_device);
    cudaFree(buffers.box_filter_scratch_device);
    cudaFree(buffers.score_device);
    cudaFree(buffers.row_stat_device);
    cudaFree(buffers.row_smooth_device);
    cudaFree(buffers.frontend_reference_device);
    cudaFree(buffers.mask_device);
    cudaFree(buffers.scratch_mask_device);
    cudaFree(buffers.strong_mask_device);
    cudaFree(buffers.strong_scratch_device);
    cudaFree(buffers.strong_row_floor_device);
    cudaFree(buffers.dynamic_floor_device);
    cudaFree(buffers.dynamic_floor_ring_device);
    cudaFreeHost(buffers.power_db_host);
    cudaFreeHost(buffers.mask_host);

    buffers = ChannelBuffers {};
  }
  cudaFree(per_freq_threshold_device_);
  per_freq_threshold_device_ = nullptr;
}

void CoherentPowerSignalDetector::reset_channel_state(uint16_t channel_number,
                                                      size_t row_elements,
                                                      size_t frame_elements,
                                                      cudaStream_t stream) {
  auto& buffers = channel_buffers_[channel_number];

  auto reset_bytes = [&](void* ptr, size_t bytes, const char* label) {
    if (ptr == nullptr || bytes == 0) {
      return;
    }
    const auto reset_result = cudaMemsetAsync(ptr, 0, bytes, stream);
    if (reset_result != cudaSuccess) {
      throw std::runtime_error(std::string("failed to reset detector ") + label + ": " +
                               cudaGetErrorString(reset_result));
    }
  };

  reset_bytes(buffers.background_device, frame_elements * sizeof(float), "background buffer");
  reset_bytes(buffers.frontend_reference_device, sizeof(float), "always-on floor buffer");
  reset_bytes(buffers.mask_device, frame_elements * sizeof(uint8_t), "mask buffer");
  reset_bytes(buffers.scratch_mask_device, frame_elements * sizeof(uint8_t), "scratch mask buffer");
  reset_bytes(buffers.strong_mask_device, frame_elements * sizeof(uint8_t), "strong rescue mask buffer");
  reset_bytes(buffers.strong_scratch_device, frame_elements * sizeof(uint8_t), "strong rescue scratch buffer");
  reset_bytes(buffers.strong_row_floor_device, row_elements * sizeof(float), "strong rescue row floor buffer");

  // The dynamic per-frequency floor is a running minimum, not a per-frame buffer: zeroing it would
  // wrongly pin every bin to 0 dB. Instead flag it so the next dynamic frame re-seeds it to the
  // high init bar and it relearns the noise floor from scratch.
  if (channel_number < dynamic_floor_seed_pending_.size()) {
    dynamic_floor_seed_pending_[channel_number] = 1;
  }

  HOLOSCAN_LOG_INFO("Reset coherent detector state for channel {} before processing the next full batch",
                    channel_number);
}

void CoherentPowerSignalDetector::setup(holoscan::OperatorSpec& spec) {
  auto& input_port = spec.input<coherent_power_in_t>("in", holoscan::IOSpec::IOSize{16});
  input_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(input_height_, "input_height", "Input height", "Detector output height.", 256);
  spec.param(input_width_, "input_width", "Input width", "Detector output width.", 512);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one output every N input frames per channel.", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter", "If non-negative, only process frames for this channel number.", -1);
  spec.param(log_detections_, "log_detections", "Log detections", "If true, logs detector execution details.", false);
  spec.param(fast_performance_, "fast_performance", "Fast performance path", "Legacy compatibility flag. The detector now always uses the live fast path.", true);
  spec.param(save_performance_path_artifacts_, "save_performance_path_artifacts", "Save path artifacts", "If true, save frame-8 artifacts for each active channel from the live fast path and stop after all channels are captured.", false);
  spec.param(enable_mask_save_, "enable_mask_save", "Enable mask save", "Enable writing detector masks to disk for debug runs.", false);
  spec.param(enable_tensor_snapshot_save_, "enable_tensor_snapshot_save", "Enable tensor snapshot save", "Enable writing frozen detector input snapshots for offline parity runs.", false);
  spec.param(save_every_n_frames_, "save_every_n_frames", "Save stride", "Save one detector mask every N frames per channel.", 1);
  spec.param(max_snapshots_per_channel_, "max_snapshots_per_channel", "Max snapshots per channel", "Maximum number of frozen detector input snapshots to save per channel for a run.", 2);
  spec.param(output_dir_, "output_dir", "Output directory", "Directory where detector masks are written.", std::string("/workspace/coherent_power_masks"));
  spec.param(tensor_snapshot_dir_, "tensor_snapshot_dir", "Tensor snapshot directory", "Directory where frozen detector input snapshots are written.", std::string("/workspace/coherent_power_snapshots"));
  spec.param(save_power_db_snapshot_, "save_power_db_snapshot", "Save power dB snapshot", "If true, also saves the post power_db frame alongside the complex tensor snapshot.", true);
  spec.param(save_coherent_power_stats_, "save_coherent_power_stats", "Save fast-path stats", "If true, dump per-frame corrected_db and local background maps for offline fast-path threshold calibration.", false);
  spec.param(coherent_power_stats_dir_, "coherent_power_stats_dir", "Coherent power stats directory", "Directory where fast-path calibration stats (corrected_db + background) are written.", std::string("/tmp/usrp_spectrograms/coherent_power_cal"));
  spec.param(per_freq_threshold_path_, "per_freq_threshold_path", "Per-frequency floor path", "Path to a calibrated per-row noise-floor .npy (length src_rows, float32 dB). Empty disables the per-frequency fill.", std::string(""));
  spec.param(per_freq_threshold_offset_db_, "per_freq_threshold_offset_db", "Per-frequency threshold offset", "dB above the calibrated per-row floor required to fire the per-frequency fill.", 2.0);
  spec.param(per_freq_threshold_mode_, "per_freq_threshold_mode", "Per-frequency threshold mode", "Source of the per-frequency floor when the fill is enabled: 'calibrated' (load the .npy at per_freq_threshold_path), 'dynamic' (learn a monotone running-min floor live from the stream), or 'static'/empty (disable the per-frequency fill).", std::string(""));
  spec.param(dynamic_floor_init_db_, "dynamic_floor_init_db", "Dynamic floor init", "Initial high per-bin floor (dB) for dynamic mode; each bin only ever descends from here. Reset to this on startup and on a center-frequency change.", 40.0);
  spec.param(dynamic_floor_std_k_, "dynamic_floor_std_k", "Dynamic floor std multiplier", "The per-frame per-row statistic folded into the running min is mean + k*std of corrected_db; k approximates the noise high-quantile the offline calibration uses. Combine with per_freq_threshold_offset_db for the final firing margin.", 2.0);
  spec.param(dynamic_floor_warmup_frames_, "dynamic_floor_warmup_frames", "Dynamic floor warmup", "Frames to accumulate into the running min before the dynamic floor is allowed to feed the fill kernel (0 = use it immediately; the high init bar already keeps early frames conservative).", 0);
  spec.param(dynamic_floor_window_slots_, "dynamic_floor_window_slots", "Dynamic floor window slots", "Number of sub-window minima kept per bin. The published floor is the min across all slots; stale lows age out after all slots rotate, bounding creep. Effective window = window_slots * slot_frames frames.", 8);
  spec.param(dynamic_floor_slot_frames_, "dynamic_floor_slot_frames", "Dynamic floor slot frames", "Frames each sub-window slot accumulates before the cursor rotates to the next slot.", 16);
  spec.param(ignore_sideband_percent_, "ignore_sideband_percent", "Ignore sideband percent", "Fraction of band edges to ignore when not frequency calibrated.", 0.0);
  spec.param(ignore_sideband_hz_, "ignore_sideband_hz", "Ignore sideband Hz", "Frequency span to ignore on each side of the band.", 7.0e6);
  spec.param(frontend_reference_q_, "frontend_reference_q", "Frontend reference quantile", "Notebook-derived frontend reference quantile.", 75.0);
  spec.param(frontend_smooth_sigma_, "frontend_smooth_sigma", "Frontend smoothing sigma", "Notebook-derived frontend smoothing sigma.", 12.0);
  spec.param(frontend_max_boost_db_, "frontend_max_boost_db", "Frontend max boost", "Notebook-derived frontend max boost in dB.", 12.0);
  spec.param(frontend_signal_cap_db_, "frontend_signal_cap_db", "Frontend signal cap", "Additional dB above the frame reference allowed to influence the frontend row baseline estimate before correction. Set to 0 to disable capped row mean.", 6.0);
  spec.param(filter_detection_mask_, "filter_detection_mask", "Filter detection mask", "If true, apply bridging and component filtering before boxing. If false, box raw connected mask regions directly.", true);
  spec.param(fast_power_floor_db_, "fast_power_floor_db", "Fast path power floor", "Support floor in dB for the fast GPU detector path.", 1.5);
  spec.param(fast_power_span_db_, "fast_power_span_db", "Fast path power span", "Support normalization span in dB for the fast GPU detector path.", 8.0);
  spec.param(fast_score_threshold_, "fast_score_threshold", "Fast path score threshold", "Score threshold for the fast GPU detector path.", 0.58);
  spec.param(fast_strong_rescue_enable_, "fast_strong_rescue_enable", "Fast path strong-signal rescue enable", "If true, OR strong signals (>= excess_db above the per-row noise floor) into the emitted mask after the emit morphology + frequency-persistence pass, so strong but frequency-narrow signals still detect.", false);
  spec.param(fast_strong_rescue_excess_db_, "fast_strong_rescue_excess_db", "Fast path strong-signal rescue excess dB", "dB above the per-row (per-frequency) noise floor required to rescue a strong narrow signal.", 8.0);
  spec.param(fast_strong_rescue_min_time_bins_, "fast_strong_rescue_min_time_bins", "Fast path strong-signal rescue minimum time bins", "Minimum in-window strong time bins required to rescue a pixel, so isolated impulsive spikes are not admitted. Values <= 1 disable the time-persistence guard.", 3);
  spec.param(live_emit_mask_rows_, "live_emit_mask_rows", "Live emit mask rows", "Target history rows for artifact snapshots reduced with the live visualizer rule.", 16);
  spec.param(live_emit_mask_cols_, "live_emit_mask_cols", "Live emit mask cols", "Target history columns for artifact snapshots reduced with the live visualizer rule.", 20480);
  spec.param(live_emit_freq_persistence_window_,
             "live_emit_freq_persistence_window",
             "Live emit frequency persistence window",
             "Odd-width horizontal support window applied after live emit mask morphology. Values <= 1 disable the extra frequency persistence pass.",
             0);
  spec.param(live_emit_freq_persistence_min_hits_,
             "live_emit_freq_persistence_min_hits",
             "Live emit frequency persistence minimum hits",
             "Minimum in-window horizontal hits required by the post-morph live emit frequency persistence pass.",
             1);
  spec.param(fast_time_smooth_radius_, "fast_time_smooth_radius", "Fast path time radius", "Time-axis radius for the fast GPU coherence proxy.", 4);
  spec.param(fast_freq_smooth_radius_, "fast_freq_smooth_radius", "Fast path frequency radius", "Frequency-axis radius for the fast GPU coherence proxy.", 3);
  spec.param(fast_background_freq_radius_, "fast_background_freq_radius", "Fast path background frequency radius", "Frequency-axis radius for the fast GPU local background.", 8);
  spec.param(fast_background_time_radius_, "fast_background_time_radius", "Fast path background time radius", "Time-axis radius for the fast GPU local background.", 10);
  spec.param(fast_mask_smooth_iterations_, "fast_mask_smooth_iterations", "Fast path mask smoothing", "Number of 3x3 majority-filter iterations for the fast GPU mask.", 1);
  spec.param(timing_summary_enable_, "timing_summary_enable", "Timing summary enable", "Enable per-stage timing summaries.", true);
  spec.param(timing_summary_every_n_, "timing_summary_every_n", "Timing summary every N", "Emit timing summaries every N emitted frames per channel.", 16);
  spec.param(timing_summary_window_, "timing_summary_window", "Timing summary window", "Maximum number of emitted frames to accumulate before reset.", 16);
}

void CoherentPowerSignalDetector::initialize() {
  holoscan::Operator::initialize();
  stop_requested_.store(false, std::memory_order_relaxed);

  if (save_performance_path_artifacts_.get()) {
    auto& tracker = path_artifact_stop_tracker();
    std::lock_guard<std::mutex> lock(tracker.mutex);
    const size_t channel_count = static_cast<size_t>(std::max(1, num_channels_.get()));
    if (tracker.active_channels.size() != channel_count || tracker.captured_channels.size() != channel_count) {
      tracker.active_channels.assign(channel_count, 0);
      tracker.captured_channels.assign(channel_count, 0);
      tracker.stop_after_frame = 0;
      tracker.stop_pending = false;
      tracker.interrupt_requested = false;
    }
    const int channel_filter = channel_filter_.get();
    if (channel_filter >= 0 && channel_filter < static_cast<int>(channel_count)) {
      tracker.active_channels[static_cast<size_t>(channel_filter)] = 1;
      tracker.captured_channels[static_cast<size_t>(channel_filter)] = 0;
    } else {
      std::fill(tracker.active_channels.begin(), tracker.active_channels.end(), 1);
      std::fill(tracker.captured_channels.begin(), tracker.captured_channels.end(), 0);
    }
  }

  frame_count_.assign(num_channels_.get(), 0);
  masks_saved_.assign(num_channels_.get(), 0);
  snapshots_saved_.assign(num_channels_.get(), 0);
  path_artifacts_saved_.assign(num_channels_.get(), 0);
  timing_stats_.assign(num_channels_.get(), ChannelTimingStats {});
  channel_buffers_.assign(num_channels_.get(), ChannelBuffers {});
  reset_detector_state_on_next_full_batch_.assign(num_channels_.get(), 0);
  last_seen_chdr_soft_resync_epoch_.assign(num_channels_.get(), 0);
  last_seen_center_frequency_.assign(num_channels_.get(), 0);
  // Seed the dynamic per-frequency floor on the first dynamic frame of each channel.
  dynamic_floor_seed_pending_.assign(num_channels_.get(), 1);
  dynamic_floor_slot_.assign(num_channels_.get(), 0);
  dynamic_floor_slot_frame_.assign(num_channels_.get(), 0);

  auto cuda_result = cudaFree(nullptr);
  if (cuda_result != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA context initialization failed: ") + cudaGetErrorString(cuda_result));
  }

  const int configured_rows = std::max(1, input_height_.get());
  const int configured_cols = std::max(1, input_width_.get());
  const size_t configured_elements = static_cast<size_t>(configured_rows) * static_cast<size_t>(configured_cols);

  auto allocate_device_float = [](float*& pointer, size_t requested_elements) {
    const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&pointer), requested_elements * sizeof(float));
    if (alloc_result != cudaSuccess) {
      throw std::runtime_error(std::string("device float buffer allocation failed: ") + cudaGetErrorString(alloc_result));
    }
  };
  auto allocate_device_u8 = [](uint8_t*& pointer, size_t requested_elements) {
    const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&pointer), requested_elements * sizeof(uint8_t));
    if (alloc_result != cudaSuccess) {
      throw std::runtime_error(std::string("device mask buffer allocation failed: ") + cudaGetErrorString(alloc_result));
    }
  };

  if (!fast_performance_.get()) {
    HOLOSCAN_LOG_WARN("coherent_power_signal_detector.fast_performance=false is ignored; the detector now always runs the live fast path.");
  }

  for (auto& buffers : channel_buffers_) {
    allocate_device_float(buffers.power_db_device, configured_elements);
    allocate_device_float(buffers.corrected_db_device, configured_elements);
    allocate_device_float(buffers.time_mean_device, configured_elements);
    allocate_device_float(buffers.freq_mean_device, configured_elements);
    allocate_device_float(buffers.background_device, configured_elements);
    allocate_device_float(buffers.box_filter_scratch_device, configured_elements);
    allocate_device_float(buffers.score_device, configured_elements);
    allocate_device_float(buffers.row_stat_device, static_cast<size_t>(configured_rows));
    allocate_device_float(buffers.row_smooth_device, static_cast<size_t>(configured_rows));
    allocate_device_float(buffers.frontend_reference_device, 1);
    allocate_device_u8(buffers.mask_device, configured_elements);
    allocate_device_u8(buffers.scratch_mask_device, configured_elements);
    allocate_device_u8(buffers.strong_mask_device, configured_elements);
    allocate_device_u8(buffers.strong_scratch_device, configured_elements);
    allocate_device_float(buffers.strong_row_floor_device, static_cast<size_t>(configured_rows));
    allocate_device_float(buffers.dynamic_floor_device, static_cast<size_t>(configured_rows));
    allocate_device_float(buffers.dynamic_floor_ring_device,
                          static_cast<size_t>(configured_rows) *
                              static_cast<size_t>(std::max(1, dynamic_floor_window_slots_.get())));
    const auto analysis_tensor_result = cudaMalloc(reinterpret_cast<void**>(&buffers.analysis_tensor_device),
                                                   configured_elements * sizeof(coherent_power_complex));
    if (analysis_tensor_result != cudaSuccess) {
      throw std::runtime_error(std::string("analysis tensor buffer allocation failed: ") + cudaGetErrorString(analysis_tensor_result));
    }

    buffers.frame_elements = configured_elements;
    buffers.row_elements = static_cast<size_t>(configured_rows);
    buffers.mask_elements = configured_elements;

    if (enable_mask_save_.get()) {
      const auto host_mask_result = cudaMallocHost(reinterpret_cast<void**>(&buffers.mask_host), configured_elements * sizeof(uint8_t));
      if (host_mask_result != cudaSuccess) {
        throw std::runtime_error(std::string("mask host buffer allocation failed: ") + cudaGetErrorString(host_mask_result));
      }
    }
  }

  if (!channel_buffers_.empty()) {
    const size_t frame_bytes = configured_elements * sizeof(float);
    const size_t row_bytes = static_cast<size_t>(configured_rows) * sizeof(float);
    const size_t mask_bytes = configured_elements * sizeof(uint8_t);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((configured_elements + threads - 1) / threads);
    const int row_blocks = (configured_rows + threads - 1) / threads;
    const int smooth_radius = std::max(1, static_cast<int>(std::ceil(std::max(frontend_smooth_sigma_.get(), 1.0) * 1.5)));
    const int channel_filter = channel_filter_.get();

    for (size_t channel_index = 0; channel_index < channel_buffers_.size(); ++channel_index) {
      if (channel_filter >= 0 && static_cast<int>(channel_index) != channel_filter) {
        continue;
      }

      auto& buffers = channel_buffers_[channel_index];
      cuda_result = cudaMemset(buffers.power_db_device, 0, frame_bytes);
      if (cuda_result != cudaSuccess) {
        throw std::runtime_error(std::string("power_db warmup memset failed: ") + cudaGetErrorString(cuda_result));
      }
      cuda_result = cudaMemset(buffers.corrected_db_device, 0, frame_bytes);
      if (cuda_result != cudaSuccess) {
        throw std::runtime_error(std::string("corrected_db warmup memset failed: ") + cudaGetErrorString(cuda_result));
      }
      cuda_result = cudaMemset(buffers.row_stat_device, 0, row_bytes);
      if (cuda_result != cudaSuccess) {
        throw std::runtime_error(std::string("row_stat warmup memset failed: ") + cudaGetErrorString(cuda_result));
      }
      cuda_result = cudaMemset(buffers.row_smooth_device, 0, row_bytes);
      if (cuda_result != cudaSuccess) {
        throw std::runtime_error(std::string("row_smooth warmup memset failed: ") + cudaGetErrorString(cuda_result));
      }
      cuda_result = cudaMemset(buffers.mask_device, 0, mask_bytes);
      if (cuda_result != cudaSuccess) {
        throw std::runtime_error(std::string("mask warmup memset failed: ") + cudaGetErrorString(cuda_result));
      }

      coherent_power_row_mean_kernel<<<configured_rows, threads>>>(buffers.power_db_device,
                                                                   configured_rows,
                                                                   configured_cols,
                                                                   buffers.row_stat_device);
      coherent_power_gaussian_smooth_rows_kernel<<<row_blocks, threads>>>(buffers.row_stat_device,
                                                                          configured_rows,
                                                                          smooth_radius,
                                                                          static_cast<float>(std::max(frontend_smooth_sigma_.get(), 1.0)),
                                                                          buffers.row_smooth_device);
      coherent_power_frontend_reference_kernel<<<1, threads>>>(buffers.row_smooth_device,
                                                               configured_rows,
                                                               static_cast<float>(frontend_reference_q_.get() / 100.0),
                                                               buffers.frontend_reference_device);
      if (frontend_signal_cap_db_.get() > 0.0) {
        coherent_power_row_capped_mean_from_reference_kernel<<<configured_rows, threads>>>(buffers.power_db_device,
                                                                                            configured_rows,
                                                                                            configured_cols,
                                                                                            buffers.frontend_reference_device,
                                                                                            static_cast<float>(frontend_signal_cap_db_.get()),
                                                                                            buffers.row_stat_device);
        coherent_power_gaussian_smooth_rows_kernel<<<row_blocks, threads>>>(buffers.row_stat_device,
                                                                            configured_rows,
                                                                            smooth_radius,
                                                                            static_cast<float>(std::max(frontend_smooth_sigma_.get(), 1.0)),
                                                                            buffers.row_smooth_device);
        coherent_power_frontend_reference_kernel<<<1, threads>>>(buffers.row_smooth_device,
                                                                 configured_rows,
                                                                 static_cast<float>(frontend_reference_q_.get() / 100.0),
                                                                 buffers.frontend_reference_device);
      }
      coherent_power_frontend_correction_kernel<<<blocks, threads>>>(buffers.power_db_device,
                                                                     configured_rows,
                                                                     configured_cols,
                                                                     buffers.row_smooth_device,
                                                                     buffers.frontend_reference_device,
                                                                     static_cast<float>(frontend_max_boost_db_.get()),
                                                                     buffers.corrected_db_device);
      coherent_power_box_mean_cols_kernel<<<blocks, threads>>>(buffers.corrected_db_device,
                                                               configured_rows,
                                                               configured_cols,
                                                               std::max(1, fast_background_time_radius_.get()),
                                                               buffers.box_filter_scratch_device);
      coherent_power_box_mean_rows_kernel<<<blocks, threads>>>(buffers.box_filter_scratch_device,
                                                               configured_rows,
                                                               configured_cols,
                                                               std::max(1, fast_background_freq_radius_.get()),
                                                               buffers.background_device);
      coherent_power_fast_power_assist_score_kernel<<<blocks, threads>>>(buffers.corrected_db_device,
                             buffers.background_device,
                             configured_rows,
                             configured_cols,
                             0,
                             static_cast<float>(fast_power_floor_db_.get()),
                             static_cast<float>(fast_power_span_db_.get()),
                             static_cast<float>(fast_score_threshold_.get()),
                             buffers.score_device,
                             buffers.mask_device);
      coherent_power_majority_smooth_kernel<<<blocks, threads>>>(buffers.mask_device,
                                                                 configured_rows,
                                                                 configured_cols,
                                                                 0,
                                                                 buffers.scratch_mask_device);
    }

    cuda_result = cudaDeviceSynchronize();
    if (cuda_result != cudaSuccess) {
      throw std::runtime_error(std::string("fast path warmup failed: ") + cudaGetErrorString(cuda_result));
    }
  }

  if (enable_mask_save_.get()) {
    std::filesystem::create_directories(output_dir_.get());
  }
  if (enable_tensor_snapshot_save_.get()) {
    std::filesystem::create_directories(tensor_snapshot_dir_.get());
  }
  if (save_performance_path_artifacts_.get()) {
    std::filesystem::create_directories(std::string(fast_performance_.get() ? kPerformancePathArtifactDir : kReferencePathArtifactDir));
  }
}

void CoherentPowerSignalDetector::compute(holoscan::InputContext& op_input,
                                          holoscan::OutputContext& op_output,
                                          holoscan::ExecutionContext& context) {
  auto input = op_input.receive<coherent_power_in_t>("in").value();
  auto& fft_tensor = std::get<0>(input);
  auto stream = std::get<1>(input);

  auto meta = metadata();
  const uint16_t channel_number = meta->get<uint16_t>("channel_number", 0);
  if (channel_number >= frame_count_.size()) {
    HOLOSCAN_LOG_WARN("Coherent power detector received out-of-range channel {} (configured channels: {}).",
                      channel_number,
                      frame_count_.size());
    return;
  }

  const int channel_filter = channel_filter_.get();
  if (channel_filter >= 0 && static_cast<int>(channel_number) != channel_filter) {
    return;
  }

  const bool chdr_partial_batch = meta->get<bool>("chdr_partial_batch", false);
  const uint64_t chdr_soft_resync_epoch = meta->get<uint64_t>("chdr_soft_resync_epoch", 0);
  if (chdr_soft_resync_epoch > 0 &&
      chdr_soft_resync_epoch != last_seen_chdr_soft_resync_epoch_[channel_number]) {
    last_seen_chdr_soft_resync_epoch_[channel_number] = chdr_soft_resync_epoch;
    reset_detector_state_on_next_full_batch_[channel_number] = 1;
    HOLOSCAN_LOG_WARN(
        "Observed CHDR soft resync epoch {} on channel {}; detector state will reset on the next full batch",
        chdr_soft_resync_epoch,
        channel_number);
  }
  // A tuning change invalidates the learned per-frequency floor (the noise floor is frequency
  // dependent), so re-seed the dynamic floor to the high bar when the center frequency changes.
  const uint64_t center_frequency = meta->get<uint64_t>("center_frequency", 0);
  if (center_frequency != last_seen_center_frequency_[channel_number]) {
    if (last_seen_center_frequency_[channel_number] != 0) {
      dynamic_floor_seed_pending_[channel_number] = 1;
      HOLOSCAN_LOG_INFO(
          "Center frequency changed to {} Hz on channel {}; dynamic per-frequency floor will re-seed",
          center_frequency,
          channel_number);
    }
    last_seen_center_frequency_[channel_number] = center_frequency;
  }
  const uint32_t chdr_packets_in_batch = meta->get<uint32_t>("chdr_packets_in_batch", 0);
  const uint32_t chdr_expected_packets_in_batch = meta->get<uint32_t>("chdr_expected_packets_in_batch", 0);
  if (chdr_partial_batch) {
    reset_detector_state_on_next_full_batch_[channel_number] = 1;
    const uint64_t partial_frame_number = meta->has_key("fft_emitted_frame_number")
                                              ? meta->get<uint64_t>("fft_emitted_frame_number", 0)
                                              : 0;
    HOLOSCAN_LOG_WARN(
        "Skipping partial CHDR batch in coherent detector ch={} frame={} packets_in_batch={} expected_packets={}"
        ", detector state will reset on the next full batch",
        channel_number,
        partial_frame_number,
        chdr_packets_in_batch,
        chdr_expected_packets_in_batch);
    meta->set("coherent_skipped_partial_batch", true);
    return;
  }
  meta->set("coherent_skipped_partial_batch", false);

  const uint64_t processing_frame_number = ++frame_count_[channel_number];
  const int emit_stride = std::max(1, emit_stride_.get());
  if ((processing_frame_number % static_cast<uint64_t>(emit_stride)) != 0) {
    return;
  }
  const uint64_t frame_number = meta->has_key("fft_emitted_frame_number")
                                    ? meta->get<uint64_t>("fft_emitted_frame_number", processing_frame_number)
                                    : processing_frame_number;
  const uint64_t detector_enter_ns = steady_time_ns();
  const uint64_t fft_emit_ts_ns = meta->get<uint64_t>("fft_emit_ts_ns", 0);

  if (save_performance_path_artifacts_.get()) {
    bool should_interrupt_graph = false;
    uint64_t stop_after_frame = 0;
    {
      auto& tracker = path_artifact_stop_tracker();
      std::lock_guard<std::mutex> lock(tracker.mutex);
      if (tracker.stop_pending && !tracker.interrupt_requested && frame_number >= tracker.stop_after_frame) {
        tracker.interrupt_requested = true;
        should_interrupt_graph = true;
        stop_after_frame = tracker.stop_after_frame;
      }
    }
    if (should_interrupt_graph && !stop_requested_.exchange(true, std::memory_order_relaxed)) {
      HOLOSCAN_LOG_INFO("Stopping graph after saving path artifacts for all active channels and draining through frame {}",
                        stop_after_frame);
      GxfGraphInterrupt(context.context());
      return;
    }
  }

  const int input_rows = static_cast<int>(fft_tensor.Size(0));
  const int input_cols = static_cast<int>(fft_tensor.Size(1));
  const auto canonical_view = canonical_tensor_view(input_rows, input_cols);
  const int src_rows = canonical_view.rows;
  const int src_cols = canonical_view.cols;
  const int configured_rows = std::max(1, input_height_.get());
  const int configured_cols = std::max(1, input_width_.get());
  if (input_rows <= 0 || input_cols <= 0) {
    HOLOSCAN_LOG_WARN("Coherent power detector received empty tensor on channel {}", channel_number);
    return;
  }

  const int configured_analysis_rows =
      (canonical_view.transposed && configured_rows == input_rows && configured_cols == input_cols) ? input_cols : configured_rows;
  const int configured_analysis_cols =
      (canonical_view.transposed && configured_rows == input_rows && configured_cols == input_cols) ? input_rows : configured_cols;

  const auto total_start = std::chrono::steady_clock::now();
  std::array<double, kTimingStageCount> stage_ms {};
  const bool timing_enabled = timing_summary_enable_.get();

  auto time_step_ms = [&](size_t stage_index, auto&& fn) {
    if (!timing_enabled) {
      fn();
      return;
    }

    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    auto sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Coherent power detector timing sync failed at {}: {}",
                         kTimingStageNames[stage_index],
                         cudaGetErrorString(sync_result));
      return;
    }
    stage_ms[stage_index] =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
  };

  int ignore_bins_per_side = 0;
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
  const double sample_rate_hz = span_hz;
  ignore_bins_per_side = compute_ignore_bins_per_side(
      src_rows, resolution_hz, ignore_sideband_percent_.get(), ignore_sideband_hz_.get());

  const int total_bins = src_rows * src_cols;
  const size_t power_db_bytes = static_cast<size_t>(total_bins) * sizeof(float);
  const size_t mask_bytes = static_cast<size_t>(total_bins) * sizeof(uint8_t);
  auto& buffers = channel_buffers_[channel_number];

  if (reset_detector_state_on_next_full_batch_[channel_number] != 0) {
    reset_channel_state(channel_number, static_cast<size_t>(src_rows), static_cast<size_t>(total_bins), stream);
    reset_detector_state_on_next_full_batch_[channel_number] = 0;
  }

  const bool require_reference_dimensions = src_rows != configured_analysis_rows || src_cols != configured_analysis_cols;
  const bool save_requested = enable_mask_save_.get();
  const bool should_save_path_artifacts = save_performance_path_artifacts_.get() &&
                                          frame_number == kPathArtifactCaptureFrame &&
                                          path_artifacts_saved_[channel_number] == 0;
  const int output_rows = configured_analysis_rows;
  const int output_cols = configured_analysis_cols;
  const bool debug_bundle_requested = enable_tensor_snapshot_save_.get();
  const bool should_save_snapshot_bundle = debug_bundle_requested &&
                                           (frame_number % static_cast<uint64_t>(std::max(1, save_every_n_frames_.get())) == 0) &&
                                           (snapshots_saved_[channel_number] < max_snapshots_per_channel_.get());
  const bool should_save_mask = false && save_requested;
  const bool should_write_mask_image = should_save_mask;
  const bool should_save_tensor_snapshot = enable_tensor_snapshot_save_.get() && should_save_snapshot_bundle;
  const bool should_save_power_db_snapshot = should_save_tensor_snapshot && save_power_db_snapshot_.get();
  const bool should_save_coherent_power_stats =
      save_coherent_power_stats_.get() &&
      (frame_number % static_cast<uint64_t>(std::max(1, save_every_n_frames_.get())) == 0);
  const bool should_run_debug_save_stage = should_write_mask_image ||
                                           should_save_tensor_snapshot ||
                                           should_save_path_artifacts ||
                                           should_save_coherent_power_stats;
  const bool frequency_axis_calibrated = resolution_hz > 0.0;
  const int emitted_mask_rows = input_rows;
  const int emitted_mask_cols = input_cols;

  time_step_ms(kInputStage, [&] {
    auto allocate_device_float = [](float*& pointer, size_t requested_elements) {
      const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&pointer), requested_elements * sizeof(float));
      if (alloc_result != cudaSuccess) {
        throw std::runtime_error(std::string("device float buffer allocation failed: ") + cudaGetErrorString(alloc_result));
      }
    };
    auto allocate_device_u8 = [](uint8_t*& pointer, size_t requested_elements) {
      const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&pointer), requested_elements * sizeof(uint8_t));
      if (alloc_result != cudaSuccess) {
        throw std::runtime_error(std::string("device mask buffer allocation failed: ") + cudaGetErrorString(alloc_result));
      }
    };

    if (buffers.frame_elements != static_cast<size_t>(total_bins)) {
      cudaFreeHost(buffers.input_tensor_host);
      cudaFree(buffers.power_db_device);
      cudaFree(buffers.corrected_db_device);
      cudaFree(buffers.time_mean_device);
      cudaFree(buffers.freq_mean_device);
      cudaFree(buffers.background_device);
      cudaFree(buffers.box_filter_scratch_device);
      cudaFree(buffers.score_device);
      cudaFree(buffers.mask_device);
      cudaFree(buffers.scratch_mask_device);
      cudaFree(buffers.strong_mask_device);
      cudaFree(buffers.strong_scratch_device);
      cudaFree(buffers.frontend_reference_device);
      cudaFree(buffers.analysis_tensor_device);
      cudaFreeHost(buffers.power_db_host);
      cudaFreeHost(buffers.mask_host);

      buffers.input_tensor_host = nullptr;
      buffers.analysis_tensor_device = nullptr;
      buffers.power_db_device = nullptr;
      buffers.corrected_db_device = nullptr;
      buffers.time_mean_device = nullptr;
      buffers.freq_mean_device = nullptr;
      buffers.background_device = nullptr;
      buffers.box_filter_scratch_device = nullptr;
      buffers.score_device = nullptr;
      buffers.mask_device = nullptr;
      buffers.scratch_mask_device = nullptr;
      buffers.strong_mask_device = nullptr;
      buffers.strong_scratch_device = nullptr;
      buffers.frontend_reference_device = nullptr;
      buffers.power_db_host = nullptr;
      buffers.mask_host = nullptr;

      allocate_device_float(buffers.power_db_device, static_cast<size_t>(total_bins));
      allocate_device_float(buffers.corrected_db_device, static_cast<size_t>(total_bins));
      allocate_device_float(buffers.time_mean_device, static_cast<size_t>(total_bins));
      allocate_device_float(buffers.freq_mean_device, static_cast<size_t>(total_bins));
      allocate_device_float(buffers.background_device, static_cast<size_t>(total_bins));
      allocate_device_float(buffers.box_filter_scratch_device, static_cast<size_t>(total_bins));
      allocate_device_float(buffers.score_device, static_cast<size_t>(total_bins));
      allocate_device_float(buffers.frontend_reference_device, 1);
      allocate_device_u8(buffers.mask_device, static_cast<size_t>(total_bins));
      allocate_device_u8(buffers.scratch_mask_device, static_cast<size_t>(total_bins));
      allocate_device_u8(buffers.strong_mask_device, static_cast<size_t>(total_bins));
      allocate_device_u8(buffers.strong_scratch_device, static_cast<size_t>(total_bins));
      const auto analysis_tensor_result = cudaMalloc(reinterpret_cast<void**>(&buffers.analysis_tensor_device),
                                                     static_cast<size_t>(total_bins) * sizeof(coherent_power_complex));
      if (analysis_tensor_result != cudaSuccess) {
        throw std::runtime_error(std::string("analysis tensor buffer allocation failed: ") + cudaGetErrorString(analysis_tensor_result));
      }

      buffers.frame_elements = static_cast<size_t>(total_bins);
      buffers.mask_elements = static_cast<size_t>(total_bins);
    }

    if (buffers.analysis_tensor_device == nullptr) {
      const auto analysis_tensor_result = cudaMalloc(reinterpret_cast<void**>(&buffers.analysis_tensor_device),
                                                     static_cast<size_t>(total_bins) * sizeof(coherent_power_complex));
      if (analysis_tensor_result != cudaSuccess) {
        throw std::runtime_error(std::string("analysis tensor buffer allocation failed: ") + cudaGetErrorString(analysis_tensor_result));
      }
    }

    if (buffers.row_elements != static_cast<size_t>(src_rows)) {
      if (buffers.row_stat_device != nullptr) {
        cudaFree(buffers.row_stat_device);
        buffers.row_stat_device = nullptr;
      }
      if (buffers.row_smooth_device != nullptr) {
        cudaFree(buffers.row_smooth_device);
        buffers.row_smooth_device = nullptr;
      }
      if (buffers.strong_row_floor_device != nullptr) {
        cudaFree(buffers.strong_row_floor_device);
        buffers.strong_row_floor_device = nullptr;
      }
      if (buffers.dynamic_floor_device != nullptr) {
        cudaFree(buffers.dynamic_floor_device);
        buffers.dynamic_floor_device = nullptr;
      }
      if (buffers.dynamic_floor_ring_device != nullptr) {
        cudaFree(buffers.dynamic_floor_ring_device);
        buffers.dynamic_floor_ring_device = nullptr;
      }
      const auto row_stat_result = cudaMalloc(reinterpret_cast<void**>(&buffers.row_stat_device), static_cast<size_t>(src_rows) * sizeof(float));
      if (row_stat_result != cudaSuccess) {
        throw std::runtime_error(std::string("row_stat buffer allocation failed: ") + cudaGetErrorString(row_stat_result));
      }
      const auto row_smooth_result = cudaMalloc(reinterpret_cast<void**>(&buffers.row_smooth_device), static_cast<size_t>(src_rows) * sizeof(float));
      if (row_smooth_result != cudaSuccess) {
        throw std::runtime_error(std::string("row_smooth buffer allocation failed: ") + cudaGetErrorString(row_smooth_result));
      }
      const auto strong_row_floor_result = cudaMalloc(reinterpret_cast<void**>(&buffers.strong_row_floor_device), static_cast<size_t>(src_rows) * sizeof(float));
      if (strong_row_floor_result != cudaSuccess) {
        throw std::runtime_error(std::string("strong rescue row floor buffer allocation failed: ") + cudaGetErrorString(strong_row_floor_result));
      }
      const auto dynamic_floor_result = cudaMalloc(reinterpret_cast<void**>(&buffers.dynamic_floor_device), static_cast<size_t>(src_rows) * sizeof(float));
      if (dynamic_floor_result != cudaSuccess) {
        throw std::runtime_error(std::string("dynamic per-frequency floor buffer allocation failed: ") + cudaGetErrorString(dynamic_floor_result));
      }
      const auto dynamic_floor_ring_result = cudaMalloc(
          reinterpret_cast<void**>(&buffers.dynamic_floor_ring_device),
          static_cast<size_t>(src_rows) *
              static_cast<size_t>(std::max(1, dynamic_floor_window_slots_.get())) * sizeof(float));
      if (dynamic_floor_ring_result != cudaSuccess) {
        throw std::runtime_error(std::string("dynamic per-frequency floor ring buffer allocation failed: ") + cudaGetErrorString(dynamic_floor_ring_result));
      }
      // Row count changed: the running floor is stale, re-seed it on the next dynamic frame.
      dynamic_floor_seed_pending_[channel_number] = 1;
      buffers.row_elements = static_cast<size_t>(src_rows);
    }

    if (buffers.frontend_reference_device == nullptr) {
      const auto frontend_reference_result = cudaMalloc(reinterpret_cast<void**>(&buffers.frontend_reference_device), sizeof(float));
      if (frontend_reference_result != cudaSuccess) {
        throw std::runtime_error(std::string("frontend_reference buffer allocation failed: ") + cudaGetErrorString(frontend_reference_result));
      }
    }

    if (should_save_power_db_snapshot && buffers.power_db_host == nullptr) {
      const auto alloc_result = cudaMallocHost(reinterpret_cast<void**>(&buffers.power_db_host), power_db_bytes);
      if (alloc_result != cudaSuccess) {
        throw std::runtime_error(std::string("power_db snapshot host buffer allocation failed: ") + cudaGetErrorString(alloc_result));
      }
    }

    if (should_save_tensor_snapshot && buffers.input_tensor_host == nullptr) {
      const auto alloc_result = cudaMallocHost(reinterpret_cast<void**>(&buffers.input_tensor_host),
                                               static_cast<size_t>(total_bins) * sizeof(coherent_power_complex));
      if (alloc_result != cudaSuccess) {
        throw std::runtime_error(std::string("input tensor snapshot host buffer allocation failed: ") + cudaGetErrorString(alloc_result));
      }
    }

    if (should_save_mask && buffers.mask_host == nullptr) {
      const auto alloc_result = cudaMallocHost(reinterpret_cast<void**>(&buffers.mask_host), mask_bytes);
      if (alloc_result != cudaSuccess) {
        throw std::runtime_error(std::string("mask host buffer allocation failed: ") + cudaGetErrorString(alloc_result));
      }
    }

    constexpr int threads = 256;
    const int blocks = (total_bins + threads - 1) / threads;
    if (canonical_view.transposed) {
      coherent_power_transpose_kernel<<<blocks, threads, 0, stream>>>(fft_tensor.Data(),
                                                                       input_rows,
                                                                       input_cols,
                                                                       buffers.analysis_tensor_device);
      auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("analysis transpose kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
    } else {
      auto copy_result = cudaMemcpyAsync(buffers.analysis_tensor_device,
                                         fft_tensor.Data(),
                                         static_cast<size_t>(total_bins) * sizeof(coherent_power_complex),
                                         cudaMemcpyDeviceToDevice,
                                         stream);
      if (copy_result != cudaSuccess) {
        throw std::runtime_error(std::string("analysis tensor copy failed: ") + cudaGetErrorString(copy_result));
      }
    }

    if (should_save_tensor_snapshot) {
      auto sync_result = cudaStreamSynchronize(stream);
      if (sync_result != cudaSuccess) {
        throw std::runtime_error(std::string("input tensor snapshot pre-copy sync failed: ") + cudaGetErrorString(sync_result));
      }
      auto copy_result = cudaMemcpy(buffers.input_tensor_host,
                                    buffers.analysis_tensor_device,
                                    static_cast<size_t>(total_bins) * sizeof(coherent_power_complex),
                                    cudaMemcpyDeviceToHost);
      if (copy_result != cudaSuccess) {
        throw std::runtime_error(std::string("input tensor snapshot copy failed: ") + cudaGetErrorString(copy_result));
      }
    }
  });

  time_step_ms(kPowerDbStage, [&] {
    constexpr int threads = 256;
    const int blocks = (total_bins + threads - 1) / threads;
    coherent_power_power_db_kernel<<<blocks, threads, 0, stream>>>(buffers.analysis_tensor_device,
                                                                    src_rows,
                                                                    src_cols,
                                                                    buffers.power_db_device);
    auto kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("power_db kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }
  });

  FastGpuMetadataSummary fast_summary;
  std::shared_ptr<unsigned int> fast_raw_mask_nonzero_device;
  std::shared_ptr<unsigned int> fast_post_emit_close_mask_nonzero_device;
  std::shared_ptr<unsigned int> fast_post_smooth_mask_nonzero_device;
  std::shared_ptr<unsigned int> fast_always_on_stripe_count_device;
  std::string coherent_backend_name = "coherent_power_live_v1";
  std::string coherent_variant_name = "frontend_live_mask_v1";
  time_step_ms(kPipelineStage, [&] {
    constexpr int threads = 256;
    const int blocks = (total_bins + threads - 1) / threads;
    coherent_power_row_mean_kernel<<<src_rows, threads, 0, stream>>>(buffers.power_db_device,
                                                                      src_rows,
                                                                      src_cols,
                                                                      buffers.row_stat_device);
    auto kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("row_mean kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    const int smooth_radius = std::max(1, static_cast<int>(std::ceil(std::max(frontend_smooth_sigma_.get(), 1.0) * 1.5)));
    const int row_blocks = (src_rows + threads - 1) / threads;
    coherent_power_gaussian_smooth_rows_kernel<<<row_blocks, threads, 0, stream>>>(buffers.row_stat_device,
                                                                                     src_rows,
                                                                                     smooth_radius,
                                                                                     static_cast<float>(std::max(frontend_smooth_sigma_.get(), 1.0)),
                                                                                     buffers.row_smooth_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("row_smooth kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    coherent_power_frontend_reference_kernel<<<1, threads, 0, stream>>>(buffers.row_smooth_device,
                                                                         src_rows,
                                                                         static_cast<float>(frontend_reference_q_.get() / 100.0),
                                                                         buffers.frontend_reference_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("frontend_reference kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    coherent_power_frontend_correction_kernel<<<blocks, threads, 0, stream>>>(buffers.power_db_device,
                                                                               src_rows,
                                                                               src_cols,
                                                                               buffers.row_smooth_device,
                                                                               buffers.frontend_reference_device,
                                                                               static_cast<float>(frontend_max_boost_db_.get()),
                                                                               buffers.corrected_db_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("frontend_correction kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    if (frontend_signal_cap_db_.get() > 0.0) {
      coherent_power_row_capped_mean_from_reference_kernel<<<src_rows, threads, 0, stream>>>(buffers.power_db_device,
                                                                                               src_rows,
                                                                                               src_cols,
                                                                                               buffers.frontend_reference_device,
                                                                                               static_cast<float>(frontend_signal_cap_db_.get()),
                                                                                               buffers.row_stat_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("capped row_mean kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      coherent_power_gaussian_smooth_rows_kernel<<<row_blocks, threads, 0, stream>>>(buffers.row_stat_device,
                                                                                       src_rows,
                                                                                       smooth_radius,
                                                                                       static_cast<float>(std::max(frontend_smooth_sigma_.get(), 1.0)),
                                                                                       buffers.row_smooth_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("capped row_smooth kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      coherent_power_frontend_reference_kernel<<<1, threads, 0, stream>>>(buffers.row_smooth_device,
                                                                           src_rows,
                                                                           static_cast<float>(frontend_reference_q_.get() / 100.0),
                                                                           buffers.frontend_reference_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("capped frontend_reference kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      coherent_power_frontend_correction_kernel<<<blocks, threads, 0, stream>>>(buffers.power_db_device,
                                                                                 src_rows,
                                                                                 src_cols,
                                                                                 buffers.row_smooth_device,
                                                                                 buffers.frontend_reference_device,
                                                                                 static_cast<float>(frontend_max_boost_db_.get()),
                                                                                 buffers.corrected_db_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("capped frontend_correction kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
    }

    coherent_power_box_mean_cols_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                         src_rows,
                                                                         src_cols,
                                                                         std::max(1, fast_background_time_radius_.get()),
                                                                         buffers.box_filter_scratch_device);
    coherent_power_box_mean_rows_kernel<<<blocks, threads, 0, stream>>>(buffers.box_filter_scratch_device,
                                                                         src_rows,
                                                                         src_cols,
                                                                         std::max(1, fast_background_freq_radius_.get()),
                                                                         buffers.background_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("separable box_mean kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    coherent_power_fast_power_assist_score_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                                   buffers.background_device,
                                                                                   src_rows,
                                                                                   src_cols,
                                                                                   ignore_bins_per_side,
                                                                                   static_cast<float>(fast_power_floor_db_.get()),
                                                                                   static_cast<float>(fast_power_span_db_.get()),
                                                                                   static_cast<float>(fast_score_threshold_.get()),
                                                                                   buffers.score_device,
                                                                                   buffers.mask_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("fast_score kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    // Per-frequency noise-floor fill (OR-ed into the box mask): fires where absolute corrected_db
    // exceeds a per-row floor + offset, filling wide-signal interiors the local box hollows out.
    // The floor comes from one of three sources selected by per_freq_threshold_mode:
    //   "calibrated" -> load a static .npy once (shared across channels);
    //   "dynamic"    -> a per-channel monotone running minimum learned live from the stream;
    //   "static"/empty -> disabled.
    std::string per_freq_mode = per_freq_threshold_mode_.get();
    for (char& c : per_freq_mode) {
      if (c >= 'A' && c <= 'Z') {
        c = static_cast<char>(c - 'A' + 'a');
      }
    }

    const float* fill_floor_device = nullptr;

    if (per_freq_mode == "calibrated") {
      // Loaded once, shared across channels; a length/shape mismatch or unreadable file disables it.
      if (!per_freq_threshold_ready_ && !per_freq_threshold_failed_) {
        bool loaded_ok = false;
        const std::string floor_path = per_freq_threshold_path_.get();
        std::vector<float> floor_host = floor_path.empty()
                                            ? std::vector<float>{}
                                            : read_npy_float32(floor_path, loaded_ok);
        if (loaded_ok && static_cast<int>(floor_host.size()) == src_rows) {
          const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&per_freq_threshold_device_),
                                               floor_host.size() * sizeof(float));
          if (alloc_result == cudaSuccess &&
              cudaMemcpy(per_freq_threshold_device_, floor_host.data(),
                         floor_host.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess) {
            per_freq_threshold_len_ = static_cast<int>(floor_host.size());
            per_freq_threshold_ready_ = true;
            HOLOSCAN_LOG_INFO("Loaded per-frequency noise floor ({} rows) from {}",
                              per_freq_threshold_len_, floor_path);
          } else {
            per_freq_threshold_failed_ = true;
            cudaFree(per_freq_threshold_device_);
            per_freq_threshold_device_ = nullptr;
            HOLOSCAN_LOG_ERROR("Failed to upload per-frequency noise floor; per-frequency fill disabled.");
          }
        } else {
          per_freq_threshold_failed_ = true;
          HOLOSCAN_LOG_ERROR("Per-frequency floor '{}' unreadable or wrong length (got {}, need {}); "
                             "per-frequency fill disabled.",
                             floor_path, floor_host.size(), src_rows);
        }
      }
      if (per_freq_threshold_ready_) {
        fill_floor_device = per_freq_threshold_device_;
      }
    } else if (per_freq_mode == "dynamic") {
      const int window_slots = std::max(1, dynamic_floor_window_slots_.get());
      const int slot_frames = std::max(1, dynamic_floor_slot_frames_.get());
      // Re-seed the ring + published floor to the high init bar on the first frame / after a reset,
      // and rewind the slot cursor so the window relearns from scratch.
      if (dynamic_floor_seed_pending_[channel_number]) {
        const float init_db = static_cast<float>(dynamic_floor_init_db_.get());
        const int ring_elems = src_rows * window_slots;
        const int ring_blocks = (ring_elems + threads - 1) / threads;
        coherent_power_fill_float_kernel<<<ring_blocks, threads, 0, stream>>>(
            buffers.dynamic_floor_ring_device, ring_elems, init_db);
        const int floor_blocks = (src_rows + threads - 1) / threads;
        coherent_power_fill_float_kernel<<<floor_blocks, threads, 0, stream>>>(
            buffers.dynamic_floor_device, src_rows, init_db);
        kernel_result = cudaGetLastError();
        if (kernel_result != cudaSuccess) {
          throw std::runtime_error(std::string("dynamic floor seed kernel launch failed: ") + cudaGetErrorString(kernel_result));
        }
        dynamic_floor_slot_[channel_number] = 0;
        dynamic_floor_slot_frame_[channel_number] = 0;
        dynamic_floor_seed_pending_[channel_number] = 0;
      }
      // Fold this frame's per-row robust power statistic into the current sub-window slot, then
      // publish floor = min across all slots. The slot is overwritten on its first frame so the
      // window it previously held ages out (bounded creep).
      const int cur_slot = dynamic_floor_slot_[channel_number];
      const int first_frame_of_slot = (dynamic_floor_slot_frame_[channel_number] == 0) ? 1 : 0;
      coherent_power_dynamic_floor_update_kernel<<<src_rows, threads, 0, stream>>>(
          buffers.corrected_db_device,
          src_rows,
          src_cols,
          ignore_bins_per_side,
          static_cast<float>(dynamic_floor_std_k_.get()),
          buffers.dynamic_floor_ring_device,
          window_slots,
          cur_slot,
          first_frame_of_slot,
          buffers.dynamic_floor_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("dynamic floor update kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
      // Advance the ring cursor: rotate to the next slot once this one has accumulated slot_frames.
      if (++dynamic_floor_slot_frame_[channel_number] >= slot_frames) {
        dynamic_floor_slot_frame_[channel_number] = 0;
        dynamic_floor_slot_[channel_number] = (cur_slot + 1) % window_slots;
      }
      // Only feed the fill once enough frames have been folded in (the high init bar keeps early
      // frames conservative regardless, so warmup defaults to 0).
      const uint64_t warmup = static_cast<uint64_t>(std::max(0, dynamic_floor_warmup_frames_.get()));
      if (processing_frame_number >= warmup) {
        fill_floor_device = buffers.dynamic_floor_device;
      }
    }

    if (fill_floor_device != nullptr) {
      coherent_power_per_freq_fill_kernel<<<blocks, threads, 0, stream>>>(
          buffers.corrected_db_device,
          fill_floor_device,
          src_rows,
          src_cols,
          ignore_bins_per_side,
          static_cast<float>(per_freq_threshold_offset_db_.get()),
          static_cast<float>(fast_power_span_db_.get()),
          buffers.score_device,
          buffers.mask_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("per_freq_fill kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
    }

    if (fast_strong_rescue_enable_.get()) {
      // Per-frequency (per-row) noise floor = mean over time of the local box-mean background.
      // Averaging the box mean (freq radius covers many bins) over time dilutes a narrow carrier,
      // so this stays a noise-floor estimate even when a persistent narrow signal is present.
      coherent_power_row_sampled_mean_kernel<<<src_rows, threads, 0, stream>>>(buffers.background_device,
                                                                                src_rows,
                                                                                src_cols,
                                                                                1,
                                                                                buffers.strong_row_floor_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("strong rescue row-floor kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      coherent_power_strong_rescue_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                           buffers.strong_row_floor_device,
                                                                           src_rows,
                                                                           src_cols,
                                                                           ignore_bins_per_side,
                                                                           static_cast<float>(fast_strong_rescue_excess_db_.get()),
                                                                           buffers.strong_mask_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("strong rescue kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      // Time-persistence guard. cols = time in the internal orientation, so the frequency
      // persistence kernel enforces horizontal (time) support: keep a strong pixel only when at
      // least min_time_bins strong pixels fall within +/-(min_time_bins-1) time bins on its row,
      // rejecting isolated impulsive spikes while leaving the frequency width unconstrained.
      const int strong_min_time_bins = std::max(1, fast_strong_rescue_min_time_bins_.get());
      if (strong_min_time_bins > 1) {
        coherent_power_frequency_persistence_kernel<<<blocks, threads, 0, stream>>>(buffers.strong_mask_device,
                                                                                     src_rows,
                                                                                     src_cols,
                                                                                     strong_min_time_bins - 1,
                                                                                     strong_min_time_bins,
                                                                                     buffers.strong_scratch_device);
        kernel_result = cudaGetLastError();
        if (kernel_result != cudaSuccess) {
          throw std::runtime_error(std::string("strong rescue time-persistence kernel launch failed: ") + cudaGetErrorString(kernel_result));
        }
        std::swap(buffers.strong_mask_device, buffers.strong_scratch_device);
      }
    }


    fast_raw_mask_nonzero_device = acquire_pooled_u32_buffer();
    if (cudaMemsetAsync(fast_raw_mask_nonzero_device.get(), 0, sizeof(unsigned int), stream) != cudaSuccess) {
      throw std::runtime_error("failed to reset fast raw mask nonzero counter");
    }
    count_nonzero_u8_kernel<<<blocks, threads, 0, stream>>>(buffers.mask_device,
                                                             total_bins,
                                                             fast_raw_mask_nonzero_device.get());
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("fast raw mask count kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    for (int iter = 0; iter < std::max(0, fast_mask_smooth_iterations_.get()); ++iter) {
      coherent_power_majority_smooth_kernel<<<blocks, threads, 0, stream>>>(buffers.mask_device,
                                                                             src_rows,
                                                                             src_cols,
                                                                             ignore_bins_per_side,
                                                                             buffers.scratch_mask_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("mask_smooth kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
      std::swap(buffers.mask_device, buffers.scratch_mask_device);
    }

    fast_post_smooth_mask_nonzero_device = acquire_pooled_u32_buffer();
    if (cudaMemsetAsync(fast_post_smooth_mask_nonzero_device.get(), 0, sizeof(unsigned int), stream) != cudaSuccess) {
      throw std::runtime_error("failed to reset fast post-smooth mask nonzero counter");
    }
    count_nonzero_u8_kernel<<<blocks, threads, 0, stream>>>(buffers.mask_device,
                                                             total_bins,
                                                             fast_post_smooth_mask_nonzero_device.get());
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("fast post-smooth mask count kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }

    fast_summary.ignore_bins_per_side = ignore_bins_per_side;
    fast_summary.merged_threshold = static_cast<float>(fast_score_threshold_.get());
    fast_summary.seed_threshold = static_cast<float>(fast_score_threshold_.get());
  });

  stage_ms[kDeviceCopyStage] = 0.0;

  auto maybe_save_debug_artifacts = [&] {
    if (should_save_coherent_power_stats) {
      // Offline fast-path threshold calibration dump: corrected_db and the local box-mean
      // background so support_db = corrected - background can be reconstructed per pixel.
      std::vector<float> corrected_stats_host(static_cast<size_t>(total_bins), 0.0f);
      std::vector<float> background_stats_host(static_cast<size_t>(total_bins), 0.0f);
      cudaMemcpyAsync(corrected_stats_host.data(), buffers.corrected_db_device, power_db_bytes,
                      cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(background_stats_host.data(), buffers.background_device, power_db_bytes,
                      cudaMemcpyDeviceToHost, stream);
      const auto stats_sync = cudaStreamSynchronize(stream);
      if (stats_sync != cudaSuccess) {
        throw std::runtime_error(std::string("coherent power stats synchronization failed: ") +
                                 cudaGetErrorString(stats_sync));
      }
      const std::string stats_dir = coherent_power_stats_dir_.get();
      std::error_code stats_dir_ec;
      std::filesystem::create_directories(stats_dir, stats_dir_ec);
      std::ostringstream stem;
      stem << stats_dir << "/coherent_power_stats_ch" << channel_number << "_f" << frame_number
           << "_" << src_rows << "x" << src_cols;
      const std::string corrected_stats_path = stem.str() + "_corrected_sxx_db.npy";
      const std::string background_stats_path = stem.str() + "_background_db.npy";
      if (!write_npy_2d(corrected_stats_path, corrected_stats_host.data(),
                        corrected_stats_host.size() * sizeof(float), src_rows, src_cols, "<f4") ||
          !write_npy_2d(background_stats_path, background_stats_host.data(),
                        background_stats_host.size() * sizeof(float), src_rows, src_cols, "<f4")) {
        HOLOSCAN_LOG_ERROR("Failed to write coherent power calibration stats under {}", stats_dir);
      } else {
        std::ofstream meta_out(stats_dir + "/meta.json", std::ios::trunc);
        if (meta_out.is_open()) {
          meta_out << "{\n"
                   << "  \"artifact_contract\": \"coherent_power_fast_stats_v1\",\n"
                   << "  \"src_rows\": " << src_rows << ",\n"
                   << "  \"src_cols\": " << src_cols << ",\n"
                   << "  \"ignore_bins_per_side\": " << ignore_bins_per_side << ",\n"
                   << "  \"fast_power_floor_db\": " << fast_power_floor_db_.get() << ",\n"
                   << "  \"fast_power_span_db\": " << fast_power_span_db_.get() << ",\n"
                   << "  \"fast_score_threshold\": " << fast_score_threshold_.get() << ",\n"
                   << "  \"fast_background_freq_radius\": " << fast_background_freq_radius_.get() << ",\n"
                   << "  \"fast_background_time_radius\": " << fast_background_time_radius_.get() << ",\n"
                   << "  \"frontend_max_boost_db\": " << frontend_max_boost_db_.get() << ",\n"
                   << "  \"frontend_signal_cap_db\": " << frontend_signal_cap_db_.get() << "\n"
                   << "}\n";
        }
      }
    }

    std::string mask_path;
    if (should_write_mask_image) {
      std::vector<uint8_t> image;
      image.reserve(static_cast<size_t>(output_rows) * static_cast<size_t>(output_cols));
      {
        auto copy_result = cudaMemcpyAsync(buffers.mask_host,
                                           buffers.mask_device,
                                           mask_bytes,
                                           cudaMemcpyDeviceToHost,
                                           stream);
        if (copy_result != cudaSuccess) {
          throw std::runtime_error(std::string("mask device-to-host copy failed: ") + cudaGetErrorString(copy_result));
        }
        auto sync_result = cudaStreamSynchronize(stream);
        if (sync_result != cudaSuccess) {
          throw std::runtime_error(std::string("mask synchronization failed: ") + cudaGetErrorString(sync_result));
        }
        image.assign(buffers.mask_host, buffers.mask_host + static_cast<size_t>(total_bins));
        for (uint8_t& pixel : image) {
          pixel = pixel ? 255 : 0;
        }
      }

      mask_path = make_mask_output_path(output_dir_.get(), channel_number, frame_number, output_rows, output_cols);
      if (!write_pgm(mask_path, image, output_cols, output_rows)) {
        HOLOSCAN_LOG_ERROR("Failed to write coherent power mask image: {}", mask_path);
      } else {
        ++masks_saved_[channel_number];
        if (log_detections_.get()) {
          HOLOSCAN_LOG_INFO("Saved coherent power mask for channel {} frame {} to {}",
                            channel_number,
                            frame_number,
                            mask_path);
        }
      }
    }

    if (!should_save_tensor_snapshot) {
      if (!should_save_path_artifacts) {
        return;
      }
    }

    if (should_save_power_db_snapshot) {
      auto copy_result = cudaMemcpyAsync(buffers.power_db_host,
                                         buffers.power_db_device,
                                         power_db_bytes,
                                         cudaMemcpyDeviceToHost,
                                         stream);
      if (copy_result != cudaSuccess) {
        throw std::runtime_error(std::string("power_db snapshot copy failed: ") + cudaGetErrorString(copy_result));
      }
    }

    auto sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
      throw std::runtime_error(std::string("snapshot synchronization failed: ") + cudaGetErrorString(sync_result));
    }

    const auto snapshot_stem = make_debug_artifact_stem(tensor_snapshot_dir_.get(), channel_number, frame_number, src_rows, src_cols);
    const auto tensor_path = snapshot_stem + "_tensor.npy";
    const auto power_db_path = snapshot_stem + "_power_db.npy";
    const auto metadata_path = snapshot_stem + ".json";
    if (should_save_tensor_snapshot &&
        !write_npy_2d(tensor_path,
                      buffers.input_tensor_host,
                      static_cast<size_t>(total_bins) * sizeof(coherent_power_complex),
                      src_rows,
                      src_cols,
                      "<c8")) {
      throw std::runtime_error("failed to write coherent input tensor snapshot");
    }

    if (should_save_power_db_snapshot &&
        !write_npy_2d(power_db_path,
                      buffers.power_db_host,
                      power_db_bytes,
                      src_rows,
                      src_cols,
                      "<f4")) {
      throw std::runtime_error("failed to write coherent power_db snapshot");
    }

    if (should_save_path_artifacts) {
      std::vector<float> corrected_db_host(static_cast<size_t>(total_bins), 0.0f);
      const auto corrected_copy_result = cudaMemcpy(corrected_db_host.data(),
                                                    buffers.corrected_db_device,
                                                    power_db_bytes,
                                                    cudaMemcpyDeviceToHost);
      if (corrected_copy_result != cudaSuccess) {
        throw std::runtime_error(std::string("path artifact corrected_db copy failed: ") + cudaGetErrorString(corrected_copy_result));
      }
      const std::string path_artifact_root = std::string(kPerformancePathArtifactDir);
      const auto path_snapshot_stem = make_debug_artifact_stem(path_artifact_root, channel_number, frame_number, src_rows, src_cols);
      const auto path_metadata_path = path_snapshot_stem + ".json";
      const auto path_corrected_db_path = path_snapshot_stem + "_corrected_sxx_db.npy";
      const auto path_corrected_db_pgm_path = path_snapshot_stem + "_corrected_preview.pgm";
      const auto path_final_mask_path = path_snapshot_stem + "_performance_final_mask.npy";
      const auto path_final_mask_pgm_path = path_snapshot_stem + "_performance_final_mask.pgm";
      const auto path_merged_surface_path = path_snapshot_stem + "_merged_surface.npy";
      const auto path_merged_surface_pgm_path = path_snapshot_stem + "_merged_surface.pgm";
      const auto path_support_surface_path = path_snapshot_stem + "_support_surface.npy";
      const auto path_support_surface_pgm_path = path_snapshot_stem + "_support_surface.pgm";
      const auto path_coherence_surface_path = path_snapshot_stem + "_coherence_surface.npy";
      const auto path_coherence_surface_pgm_path = path_snapshot_stem + "_coherence_surface.pgm";
      const auto path_mask_components_path = path_snapshot_stem + "_mask_components.npy";
      const auto path_mask_components_pgm_path = path_snapshot_stem + "_mask_components.pgm";
      const auto path_emitted_live_mask_path = path_snapshot_stem + "_emitted_live_mask.npy";
      const auto path_emitted_live_mask_pgm_path = path_snapshot_stem + "_emitted_live_mask.pgm";
      const auto path_emitted_live_history_mask_path = path_snapshot_stem + "_emitted_live_history_mask.npy";
      const auto path_emitted_live_history_mask_pgm_path = path_snapshot_stem + "_emitted_live_history_mask.pgm";
      if (!write_npy_2d(path_corrected_db_path,
                        corrected_db_host.data(),
                        corrected_db_host.size() * sizeof(float),
                        src_rows,
                        src_cols,
                        "<f4")) {
        throw std::runtime_error("failed to write path artifact corrected spectrogram");
      }
      if (!write_pgm(path_corrected_db_pgm_path,
                     float_unit_map_to_u8(normalize_map01_local(corrected_db_host, 1.0f, 99.0f)),
                     src_cols,
                     src_rows)) {
        throw std::runtime_error("failed to write path artifact corrected preview");
      }

      {
        std::vector<float> merged_surface_host(static_cast<size_t>(total_bins), 0.0f);
        const auto merged_surface_copy_result = cudaMemcpy(merged_surface_host.data(),
                                                           buffers.score_device,
                                                           power_db_bytes,
                                                           cudaMemcpyDeviceToHost);
        if (merged_surface_copy_result != cudaSuccess) {
          throw std::runtime_error(std::string("path artifact merged surface copy failed: ") + cudaGetErrorString(merged_surface_copy_result));
        }
        std::vector<float> background_host(static_cast<size_t>(total_bins), 0.0f);
        const auto background_copy_result = cudaMemcpy(background_host.data(),
                                                       buffers.background_device,
                                                       power_db_bytes,
                                                       cudaMemcpyDeviceToHost);
        if (background_copy_result != cudaSuccess) {
          throw std::runtime_error(std::string("path artifact background copy failed: ") + cudaGetErrorString(background_copy_result));
        }
        std::vector<uint8_t> final_mask_host(static_cast<size_t>(total_bins), 0);
        const auto final_mask_copy_result = cudaMemcpy(final_mask_host.data(),
                                                       buffers.mask_device,
                                                       mask_bytes,
                                                       cudaMemcpyDeviceToHost);
        if (final_mask_copy_result != cudaSuccess) {
          throw std::runtime_error(std::string("path artifact final mask copy failed: ") + cudaGetErrorString(final_mask_copy_result));
        }
        std::vector<uint8_t> per_row_valid_mask(static_cast<size_t>(src_rows), 1);
        for (int row = 0; row < src_rows; ++row) {
          const bool valid = row >= ignore_bins_per_side && row < (src_rows - ignore_bins_per_side);
          per_row_valid_mask[static_cast<size_t>(row)] = valid ? 1 : 0;
          if (!valid) {
            const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(src_cols);
            std::fill(final_mask_host.begin() + row_offset,
                      final_mask_host.begin() + row_offset + static_cast<size_t>(src_cols),
                      static_cast<uint8_t>(0));
          }
        }
        std::vector<float> support_surface_host(static_cast<size_t>(total_bins), 0.0f);
        std::vector<float> coherence_surface_host(static_cast<size_t>(total_bins), 0.0f);
        for (int row = 0; row < src_rows; ++row) {
          const bool valid = per_row_valid_mask[static_cast<size_t>(row)] != 0;
          const size_t row_offset = static_cast<size_t>(row) * static_cast<size_t>(src_cols);
          for (int col = 0; col < src_cols; ++col) {
            const size_t index = row_offset + static_cast<size_t>(col);
            if (!valid) {
              support_surface_host[index] = 0.0f;
              coherence_surface_host[index] = 0.0f;
              continue;
            }
            const float support_db = corrected_db_host[index] - background_host[index];
            support_surface_host[index] = std::clamp(
                (support_db - static_cast<float>(fast_power_floor_db_.get())) /
                    std::max(static_cast<float>(fast_power_span_db_.get()), 1e-6f),
                0.0f,
                1.0f);
            coherence_surface_host[index] = 0.0f;
          }
        }
        const std::vector<float> mask_components_host =
            label_mask_connected_components(final_mask_host, src_rows, src_cols, per_row_valid_mask);
        std::vector<float> final_mask_float(final_mask_host.size(), 0.0f);
        for (size_t index = 0; index < final_mask_host.size(); ++index) {
          final_mask_float[index] = final_mask_host[index] ? 1.0f : 0.0f;
        }
        if (!write_npy_2d(path_merged_surface_path,
                          merged_surface_host.data(),
                          merged_surface_host.size() * sizeof(float),
                          src_rows,
                          src_cols,
                          "<f4")) {
          throw std::runtime_error("failed to write path artifact merged surface");
        }
        if (!write_pgm(path_merged_surface_pgm_path,
                       float_unit_map_to_u8(normalize_map01_local(merged_surface_host, 1.0f, 99.0f)),
                       src_cols,
                       src_rows)) {
          throw std::runtime_error("failed to write path artifact merged surface preview");
        }
        if (!write_npy_2d(path_support_surface_path,
                          support_surface_host.data(),
                          support_surface_host.size() * sizeof(float),
                          src_rows,
                          src_cols,
                          "<f4")) {
          throw std::runtime_error("failed to write path artifact support surface");
        }
        if (!write_pgm(path_support_surface_pgm_path,
                       float_unit_map_to_u8(support_surface_host),
                       src_cols,
                       src_rows)) {
          throw std::runtime_error("failed to write path artifact support surface preview");
        }
        if (!write_npy_2d(path_coherence_surface_path,
                          coherence_surface_host.data(),
                          coherence_surface_host.size() * sizeof(float),
                          src_rows,
                          src_cols,
                          "<f4")) {
          throw std::runtime_error("failed to write path artifact coherence surface");
        }
        if (!write_pgm(path_coherence_surface_pgm_path,
                       float_unit_map_to_u8(coherence_surface_host),
                       src_cols,
                       src_rows)) {
          throw std::runtime_error("failed to write path artifact coherence surface preview");
        }
        if (!write_npy_2d(path_mask_components_path,
                          mask_components_host.data(),
                          mask_components_host.size() * sizeof(float),
                          src_rows,
                          src_cols,
                          "<f4")) {
          throw std::runtime_error("failed to write path artifact mask components");
        }
        if (!write_pgm(path_mask_components_pgm_path,
                       float_unit_map_to_u8(normalize_map01_local(mask_components_host, 0.0f, 100.0f)),
                       src_cols,
                       src_rows)) {
          throw std::runtime_error("failed to write path artifact mask components preview");
        }
        if (!write_npy_2d(path_final_mask_path,
                          final_mask_float.data(),
                          final_mask_float.size() * sizeof(float),
                          src_rows,
                          src_cols,
                          "<f4")) {
          throw std::runtime_error("failed to write path artifact final mask");
        }
        if (!write_pgm(path_final_mask_pgm_path,
                       binary_float_mask_to_u8(final_mask_float),
                       src_cols,
                       src_rows)) {
          throw std::runtime_error("failed to write path artifact final mask preview");
        }

        std::vector<uint8_t> emitted_live_mask_host(static_cast<size_t>(input_rows) * static_cast<size_t>(input_cols), 0);
        const size_t emitted_live_mask_bytes = emitted_live_mask_host.size() * sizeof(uint8_t);
        auto emitted_live_mask_device = allocate_owned_u8_buffer(emitted_live_mask_bytes);
        if (canonical_view.transposed) {
          constexpr int transpose_threads = 256;
          const int transpose_blocks = (total_bins + transpose_threads - 1) / transpose_threads;
          transpose_u8_kernel<<<transpose_blocks, transpose_threads, 0, stream>>>(buffers.mask_device,
                                                                                   src_rows,
                                                                                   src_cols,
                                                                                   emitted_live_mask_device.get());
          auto transpose_result = cudaGetLastError();
          if (transpose_result != cudaSuccess) {
            throw std::runtime_error(std::string("emitted live mask transpose kernel launch failed: ") +
                                     cudaGetErrorString(transpose_result));
          }
        } else {
          const auto emitted_copy_result = cudaMemcpyAsync(emitted_live_mask_device.get(),
                                                           buffers.mask_device,
                                                           emitted_live_mask_bytes,
                                                           cudaMemcpyDeviceToDevice,
                                                           stream);
          if (emitted_copy_result != cudaSuccess) {
            throw std::runtime_error(std::string("path artifact emitted live mask copy failed: ") +
                                     cudaGetErrorString(emitted_copy_result));
          }
        }

        if (filter_detection_mask_.get()) {
          auto emit_scratch0_device = allocate_owned_u8_buffer(emitted_live_mask_bytes);
          auto emit_scratch1_device = allocate_owned_u8_buffer(emitted_live_mask_bytes);
          apply_emit_mask_morphology(emitted_live_mask_device.get(),
                                     input_rows,
                                     input_cols,
                                     emit_scratch0_device.get(),
                                     emit_scratch1_device.get(),
                                     live_emit_freq_persistence_window_.get(),
                                     live_emit_freq_persistence_min_hits_.get(),
                                     nullptr,
                                     nullptr,
                                     stream);
        }

        const auto emitted_copy_result = cudaMemcpyAsync(emitted_live_mask_host.data(),
                                                         emitted_live_mask_device.get(),
                                                         emitted_live_mask_bytes,
                                                         cudaMemcpyDeviceToHost,
                                                         stream);
        if (emitted_copy_result != cudaSuccess) {
          throw std::runtime_error(std::string("path artifact emitted live mask copy failed: ") +
                                   cudaGetErrorString(emitted_copy_result));
        }
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
          throw std::runtime_error("failed to synchronize emitted live mask artifact copy");
        }
        std::vector<float> emitted_live_mask_float(emitted_live_mask_host.size(), 0.0f);
        for (size_t index = 0; index < emitted_live_mask_host.size(); ++index) {
          emitted_live_mask_float[index] = emitted_live_mask_host[index] ? 1.0f : 0.0f;
        }
        const int live_history_rows = std::max(1, live_emit_mask_rows_.get());
        const int live_history_cols = std::max(1, live_emit_mask_cols_.get());
        std::vector<uint8_t> emitted_live_history_mask_u8 =
          reduce_mask_for_history_rows(emitted_live_mask_host,
                         input_cols,
                         input_rows,
                         live_history_cols,
                         live_history_rows);
        std::vector<float> emitted_live_history_mask_float(emitted_live_history_mask_u8.size(), 0.0f);
        for (size_t index = 0; index < emitted_live_history_mask_u8.size(); ++index) {
          emitted_live_history_mask_float[index] =
              static_cast<float>(emitted_live_history_mask_u8[index]) / 255.0f;
        }
        if (!write_npy_2d(path_emitted_live_mask_path,
                          emitted_live_mask_float.data(),
                          emitted_live_mask_float.size() * sizeof(float),
                          input_rows,
                          input_cols,
                          "<f4")) {
          throw std::runtime_error("failed to write path artifact emitted live mask");
        }
        if (!write_pgm(path_emitted_live_mask_pgm_path,
                       emitted_live_mask_host,
                       input_cols,
                       input_rows)) {
          throw std::runtime_error("failed to write path artifact emitted live mask preview");
        }
        if (!write_npy_2d(path_emitted_live_history_mask_path,
                          emitted_live_history_mask_float.data(),
                          emitted_live_history_mask_float.size() * sizeof(float),
                          live_history_rows,
                          live_history_cols,
                          "<f4")) {
          throw std::runtime_error("failed to write path artifact emitted live history mask");
        }
        if (!write_pgm(path_emitted_live_history_mask_pgm_path,
                       emitted_live_history_mask_u8,
                       live_history_cols,
                       live_history_rows)) {
          throw std::runtime_error("failed to write path artifact emitted live history mask preview");
        }
      }

      std::ofstream path_meta_out(path_metadata_path, std::ios::binary);
      if (!path_meta_out.is_open()) {
        throw std::runtime_error("failed to open path artifact metadata sidecar");
      }
      path_meta_out << "{\n";
      path_meta_out << "  \"channel_number\": " << channel_number << ",\n";
      path_meta_out << "  \"frame_number\": " << frame_number << ",\n";
      path_meta_out << "  \"rows\": " << src_rows << ",\n";
      path_meta_out << "  \"cols\": " << src_cols << ",\n";
      path_meta_out << "  \"original_input_rows\": " << input_rows << ",\n";
      path_meta_out << "  \"original_input_cols\": " << input_cols << ",\n";
      path_meta_out << "  \"tensor_axis_order\": \"frequency_time\",\n";
      path_meta_out << "  \"input_height\": " << output_rows << ",\n";
      path_meta_out << "  \"input_width\": " << output_cols << ",\n";
      path_meta_out << "  \"resolution_hz\": " << resolution_hz << ",\n";
      path_meta_out << "  \"sample_rate_hz\": " << sample_rate_hz << ",\n";
      path_meta_out << "  \"span_hz\": " << span_hz << ",\n";
      path_meta_out << "  \"frequency_axis_calibrated\": " << json_bool(frequency_axis_calibrated) << ",\n";
      path_meta_out << "  \"ignore_bins_per_side\": " << ignore_bins_per_side << ",\n";
      path_meta_out << "  \"fast_performance\": true,\n";
      path_meta_out << "  \"path_mode_effective\": \"fast_performance\",\n";
      path_meta_out << "  \"pipeline_variant\": \"" << json_escape(coherent_variant_name) << "\",\n";
      path_meta_out << "  \"tensor_snapshot_path\": null,\n";
      path_meta_out << "  \"power_db_snapshot_path\": null,\n";
      path_meta_out << "  \"mask_path\": null,\n";
      path_meta_out << "  \"performance_path_artifacts\": {\n";
      path_meta_out << "    \"corrected_db_path\": \"" << json_escape(path_corrected_db_path) << "\",\n";
      path_meta_out << "    \"corrected_db_pgm_path\": \"" << json_escape(path_corrected_db_pgm_path) << "\",\n";
      path_meta_out << "    \"merged_surface_path\": \"" << json_escape(path_merged_surface_path) << "\",\n";
      path_meta_out << "    \"merged_surface_pgm_path\": \"" << json_escape(path_merged_surface_pgm_path) << "\",\n";
      path_meta_out << "    \"support_surface_path\": \"" << json_escape(path_support_surface_path) << "\",\n";
      path_meta_out << "    \"support_surface_pgm_path\": \"" << json_escape(path_support_surface_pgm_path) << "\",\n";
      path_meta_out << "    \"coherence_surface_path\": \"" << json_escape(path_coherence_surface_path) << "\",\n";
      path_meta_out << "    \"coherence_surface_pgm_path\": \"" << json_escape(path_coherence_surface_pgm_path) << "\",\n";
      path_meta_out << "    \"mask_components_path\": \"" << json_escape(path_mask_components_path) << "\",\n";
      path_meta_out << "    \"mask_components_pgm_path\": \"" << json_escape(path_mask_components_pgm_path) << "\",\n";
      path_meta_out << "    \"grouped_artifact_mask_path\": \"" << json_escape(path_final_mask_path) << "\",\n";
      path_meta_out << "    \"grouped_artifact_mask_pgm_path\": \"" << json_escape(path_final_mask_pgm_path) << "\",\n";
      path_meta_out << "    \"final_mask_path\": \"" << json_escape(path_final_mask_path) << "\",\n";
      path_meta_out << "    \"final_mask_pgm_path\": \"" << json_escape(path_final_mask_pgm_path) << "\",\n";
      path_meta_out << "    \"detector_emitted_visualizer_input_mask_path\": \"" << json_escape(path_emitted_live_mask_path) << "\",\n";
      path_meta_out << "    \"detector_emitted_visualizer_input_mask_pgm_path\": \"" << json_escape(path_emitted_live_mask_pgm_path) << "\",\n";
      path_meta_out << "    \"detector_emitted_visualizer_history_mask_path\": \"" << json_escape(path_emitted_live_history_mask_path) << "\",\n";
      path_meta_out << "    \"detector_emitted_visualizer_history_mask_pgm_path\": \"" << json_escape(path_emitted_live_history_mask_pgm_path) << "\",\n";
      path_meta_out << "    \"detector_emitted_visualizer_history_rows\": " << std::max(1, live_emit_mask_rows_.get()) << ",\n";
      path_meta_out << "    \"detector_emitted_visualizer_history_cols\": " << std::max(1, live_emit_mask_cols_.get()) << ",\n";
      path_meta_out << "    \"emitted_live_mask_path\": \"" << json_escape(path_emitted_live_mask_path) << "\",\n";
      path_meta_out << "    \"emitted_live_mask_pgm_path\": \"" << json_escape(path_emitted_live_mask_pgm_path) << "\",\n";
      path_meta_out << "    \"artifact_capture_preserves_fast_path\": true,\n";
      path_meta_out << "    \"detector_output_matches_live_visualizer_input\": true\n";
      path_meta_out << "  },\n";
      path_meta_out << "  \"reference_debug_artifacts\": null,\n";
      path_meta_out << "  \"config\": {\n";
      path_meta_out << "    \"ignore_sideband_percent\": " << ignore_sideband_percent_.get() << ",\n";
      path_meta_out << "    \"ignore_sideband_hz\": " << ignore_sideband_hz_.get() << ",\n";
      path_meta_out << "    \"frontend_reference_q\": " << frontend_reference_q_.get() << ",\n";
      path_meta_out << "    \"frontend_smooth_sigma\": " << frontend_smooth_sigma_.get() << ",\n";
      path_meta_out << "    \"frontend_max_boost_db\": " << frontend_max_boost_db_.get() << ",\n";
      path_meta_out << "    \"frontend_signal_cap_db\": " << frontend_signal_cap_db_.get() << ",\n";
      path_meta_out << "    \"power_assist_mode\": \"" << kHardwiredPowerAssistMode << "\",\n";
      path_meta_out << "    \"coherence_source_mode\": \"" << kHardwiredCoherenceSourceMode << "\"\n";
      path_meta_out << "  }\n";
      path_meta_out << "}\n";
      if (!path_meta_out.good()) {
        throw std::runtime_error("failed to write path artifact metadata sidecar");
      }
      path_meta_out.flush();
      path_meta_out.close();
      path_artifacts_saved_[channel_number] = 1;

      {
        auto& tracker = path_artifact_stop_tracker();
        std::lock_guard<std::mutex> lock(tracker.mutex);
        if (static_cast<size_t>(channel_number) >= tracker.captured_channels.size()) {
          tracker.captured_channels.resize(static_cast<size_t>(channel_number + 1), 0);
          tracker.active_channels.resize(static_cast<size_t>(channel_number + 1), 1);
        }
        tracker.captured_channels[static_cast<size_t>(channel_number)] = 1;
        bool all_channels_captured = false;
        if (!tracker.active_channels.empty()) {
          all_channels_captured = true;
          for (size_t channel_index = 0; channel_index < tracker.active_channels.size(); ++channel_index) {
            if (tracker.active_channels[channel_index] == 0) {
              continue;
            }
            if (channel_index >= tracker.captured_channels.size() || tracker.captured_channels[channel_index] == 0) {
              all_channels_captured = false;
              break;
            }
          }
        }
        if (all_channels_captured) {
          tracker.stop_pending = true;
          tracker.stop_after_frame = std::max(tracker.stop_after_frame,
                                              frame_number + kPathArtifactStopGraceFrames);
        }
      }
      if (!should_save_tensor_snapshot) {
        return;
      }
    }

    const double center_frequency_hz = static_cast<double>(meta->get<uint64_t>("center_frequency", 0));
    std::ofstream meta_out(metadata_path, std::ios::binary);
    if (!meta_out.is_open()) {
      throw std::runtime_error("failed to open coherent snapshot metadata sidecar");
    }
    meta_out << "{\n";
    meta_out << "  \"channel_number\": " << channel_number << ",\n";
    meta_out << "  \"frame_number\": " << frame_number << ",\n";
    meta_out << "  \"rows\": " << src_rows << ",\n";
    meta_out << "  \"cols\": " << src_cols << ",\n";
    meta_out << "  \"original_input_rows\": " << input_rows << ",\n";
    meta_out << "  \"original_input_cols\": " << input_cols << ",\n";
    meta_out << "  \"tensor_axis_order\": \"frequency_time\",\n";
    meta_out << "  \"input_height\": " << output_rows << ",\n";
    meta_out << "  \"input_width\": " << output_cols << ",\n";
    meta_out << "  \"resolution_hz\": " << resolution_hz << ",\n";
    meta_out << "  \"sample_rate_hz\": " << sample_rate_hz << ",\n";
    meta_out << "  \"span_hz\": " << span_hz << ",\n";
    meta_out << "  \"center_frequency_hz\": " << center_frequency_hz << ",\n";
    meta_out << "  \"frequency_axis_calibrated\": " << json_bool(frequency_axis_calibrated) << ",\n";
    meta_out << "  \"ignore_bins_per_side\": " << ignore_bins_per_side << ",\n";
    meta_out << "  \"fast_performance\": true,\n";
    meta_out << "  \"path_mode_effective\": \"fast_performance\",\n";
    meta_out << "  \"pipeline_variant\": \"" << json_escape(coherent_variant_name) << "\",\n";
    if (should_save_tensor_snapshot) {
      meta_out << "  \"tensor_snapshot_path\": \"" << json_escape(tensor_path) << "\",\n";
    } else {
      meta_out << "  \"tensor_snapshot_path\": null,\n";
    }
    if (should_save_power_db_snapshot) {
      meta_out << "  \"power_db_snapshot_path\": \"" << json_escape(power_db_path) << "\",\n";
    } else {
      meta_out << "  \"power_db_snapshot_path\": null,\n";
    }
    if (!mask_path.empty()) {
      meta_out << "  \"mask_path\": \"" << json_escape(mask_path) << "\",\n";
    } else {
      meta_out << "  \"mask_path\": null,\n";
    }
    meta_out << "  \"reference_debug_artifacts\": null,\n";
    meta_out << "  \"config\": {\n";
    meta_out << "    \"ignore_sideband_percent\": " << ignore_sideband_percent_.get() << ",\n";
    meta_out << "    \"ignore_sideband_hz\": " << ignore_sideband_hz_.get() << ",\n";
    meta_out << "    \"frontend_reference_q\": " << frontend_reference_q_.get() << ",\n";
    meta_out << "    \"frontend_smooth_sigma\": " << frontend_smooth_sigma_.get() << ",\n";
    meta_out << "    \"frontend_max_boost_db\": " << frontend_max_boost_db_.get() << ",\n";
    meta_out << "    \"frontend_signal_cap_db\": " << frontend_signal_cap_db_.get() << ",\n";
    meta_out << "    \"power_assist_mode\": \"" << kHardwiredPowerAssistMode << "\",\n";
    meta_out << "    \"coherence_source_mode\": \"" << kHardwiredCoherenceSourceMode << "\"\n";
    meta_out << "  }\n";
    meta_out << "}\n";
    if (!meta_out.good()) {
      throw std::runtime_error("failed to write coherent snapshot metadata sidecar");
    }
    meta_out.flush();
    if (!meta_out.good()) {
      throw std::runtime_error("failed to flush coherent snapshot metadata sidecar");
    }
    meta_out.close();
    if (!meta_out) {
      throw std::runtime_error("failed to close coherent snapshot metadata sidecar");
    }

    ++snapshots_saved_[channel_number];
    if (log_detections_.get()) {
      if (should_save_tensor_snapshot) {
        HOLOSCAN_LOG_INFO("Saved coherent tensor snapshot for channel {} frame {} to {}",
                          channel_number,
                          frame_number,
                          tensor_path);
      } else {
        HOLOSCAN_LOG_INFO("Saved coherent reference debug bundle for channel {} frame {} under {}",
                          channel_number,
                          frame_number,
                          snapshot_stem);
      }
    }

  };

  if (should_run_debug_save_stage) {
    time_step_ms(kMaskSaveStage, maybe_save_debug_artifacts);
  } else {
    stage_ms[kMaskSaveStage] = 0.0;
  }

  stage_ms[kTotalStage] =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();

  const double fft_to_detector_enter_ms = elapsed_ms(fft_emit_ts_ns, detector_enter_ns);
  const double fft_to_detector_done_ms = fft_to_detector_enter_ms + stage_ms[kTotalStage];

  meta->set("coherent_frame_number", frame_number);
  meta->set("coherent_mask_height", static_cast<uint32_t>(emitted_mask_rows));
  meta->set("coherent_mask_width", static_cast<uint32_t>(emitted_mask_cols));
  meta->set("coherent_backend", coherent_backend_name);
  meta->set("coherent_chunk_count", static_cast<uint32_t>(fast_summary.subsection_count));
    const uint32_t coherent_grouped_box_count =
      static_cast<uint32_t>(fast_summary.grouped_box_count);
    meta->set("coherent_grouped_box_count", coherent_grouped_box_count);
  meta->set("coherent_ignore_bins_per_side", fast_summary.ignore_bins_per_side);
  meta->set("coherent_merged_threshold", fast_summary.merged_threshold);
  meta->set("coherent_seed_threshold", fast_summary.seed_threshold);
  meta->set("coherent_pipeline_variant", coherent_variant_name);
  meta->set("coherent_timing_total_ms", stage_ms[kTotalStage]);
  meta->set("coherent_grouped_box_count_meaningful",
            true);
  meta->set("coherent_final_mask_component_count",
            static_cast<uint32_t>(0));
  meta->set("coherent_final_mask_grouped_box_count",
            static_cast<uint32_t>(0));
  meta->set("coherent_final_mask_grouped_box_count_meaningful",
            false);
  const uint32_t coherent_reference_mask_nonzero_pixels = 0U;
  uint32_t coherent_emitted_mask_nonzero_pixels = coherent_reference_mask_nonzero_pixels;
  bool coherent_emitted_mask_metrics_meaningful = false;

  {
    auto& buffers = channel_buffers_[static_cast<size_t>(channel_number)];
    if (buffers.mask_device != nullptr && buffers.mask_elements > 0) {
      const int dst_rows = emitted_mask_rows;
      const int dst_cols = emitted_mask_cols;
        const size_t emitted_mask_bytes = static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols) * sizeof(uint8_t);
      auto mask_buffer = allocate_owned_u8_buffer(emitted_mask_bytes);
      auto emitted_nonzero_device = allocate_owned_u32_buffer();
      if (cudaMemsetAsync(emitted_nonzero_device.get(), 0, sizeof(unsigned int), stream) != cudaSuccess) {
        throw std::runtime_error("failed to reset emitted mask nonzero counter");
      }
      auto emit_mask_result = cudaSuccess;
      if (canonical_view.transposed) {
        constexpr int transpose_threads = 256;
        const int transpose_blocks = (total_bins + transpose_threads - 1) / transpose_threads;
        transpose_u8_kernel<<<transpose_blocks, transpose_threads, 0, stream>>>(buffers.mask_device,
                                                                                 src_rows,
                                                                                 src_cols,
                                                                                 mask_buffer.get());
        emit_mask_result = cudaGetLastError();
        if (emit_mask_result != cudaSuccess) {
          throw std::runtime_error(std::string("mask transpose-to-emit kernel launch failed: ") + cudaGetErrorString(emit_mask_result));
        }
      } else {
        emit_mask_result = cudaMemcpyAsync(mask_buffer.get(),
                                           buffers.mask_device,
                                           emitted_mask_bytes,
                                           cudaMemcpyDeviceToDevice,
                                           stream);
        if (emit_mask_result != cudaSuccess) {
          throw std::runtime_error(std::string("mask emit device copy failed: ") + cudaGetErrorString(emit_mask_result));
        }
      }

      if (filter_detection_mask_.get()) {
        auto emit_scratch0_device = allocate_owned_u8_buffer(emitted_mask_bytes);
        auto emit_scratch1_device = allocate_owned_u8_buffer(emitted_mask_bytes);
        fast_post_emit_close_mask_nonzero_device = allocate_owned_u32_buffer();
        auto fast_post_emit_persistence_mask_nonzero_device = allocate_owned_u32_buffer();
        apply_emit_mask_morphology(mask_buffer.get(),
                                   dst_rows,
                                   dst_cols,
                                   emit_scratch0_device.get(),
                                   emit_scratch1_device.get(),
                                   live_emit_freq_persistence_window_.get(),
                                   live_emit_freq_persistence_min_hits_.get(),
                                   fast_post_emit_close_mask_nonzero_device.get(),
                                   fast_post_emit_persistence_mask_nonzero_device.get(),
                                   stream);
        unsigned int post_persistence_nonzero_count = 0;
        if (cudaMemcpy(&post_persistence_nonzero_count,
                       fast_post_emit_persistence_mask_nonzero_device.get(),
                       sizeof(unsigned int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          throw std::runtime_error("failed to copy fast post-persistence mask nonzero count");
        }
        fast_summary.post_emit_persistence_mask_nonzero_pixels = post_persistence_nonzero_count;
      }

      const int compact_total = dst_rows * dst_cols;
      constexpr int count_threads = 256;
      const int count_blocks = (compact_total + count_threads - 1) / count_threads;

      // Strong-signal rescue: OR the (time-persisted) strong mask back in AFTER the emit
      // morphology + frequency-persistence pass, transposed into the emit orientation to match.
      // This is what lets a strong but frequency-narrow signal survive the width filters.
      if (fast_strong_rescue_enable_.get()) {
        auto strong_emit_buffer = allocate_owned_u8_buffer(emitted_mask_bytes);
        if (canonical_view.transposed) {
          transpose_u8_kernel<<<count_blocks, count_threads, 0, stream>>>(buffers.strong_mask_device,
                                                                           src_rows,
                                                                           src_cols,
                                                                           strong_emit_buffer.get());
          emit_mask_result = cudaGetLastError();
          if (emit_mask_result != cudaSuccess) {
            throw std::runtime_error(std::string("strong rescue transpose-to-emit kernel launch failed: ") + cudaGetErrorString(emit_mask_result));
          }
        } else {
          emit_mask_result = cudaMemcpyAsync(strong_emit_buffer.get(),
                                             buffers.strong_mask_device,
                                             emitted_mask_bytes,
                                             cudaMemcpyDeviceToDevice,
                                             stream);
          if (emit_mask_result != cudaSuccess) {
            throw std::runtime_error(std::string("strong rescue emit copy failed: ") + cudaGetErrorString(emit_mask_result));
          }
        }
        coherent_power_or_u8_kernel<<<count_blocks, count_threads, 0, stream>>>(mask_buffer.get(),
                                                                                 strong_emit_buffer.get(),
                                                                                 compact_total);
        emit_mask_result = cudaGetLastError();
        if (emit_mask_result != cudaSuccess) {
          throw std::runtime_error(std::string("strong rescue OR kernel launch failed: ") + cudaGetErrorString(emit_mask_result));
        }
      }

      count_nonzero_u8_kernel<<<count_blocks, count_threads, 0, stream>>>(mask_buffer.get(),
                                                                           compact_total,
                                                                           emitted_nonzero_device.get());
      auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("live mask count kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
      unsigned int emitted_nonzero_count = 0;
      if (cudaMemcpyAsync(&emitted_nonzero_count,
                          emitted_nonzero_device.get(),
                          sizeof(unsigned int),
                          cudaMemcpyDeviceToHost,
                          stream) != cudaSuccess) {
        throw std::runtime_error("failed to copy emitted mask nonzero count");
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("failed to synchronize live mask emission stream");
      }
      if (fast_raw_mask_nonzero_device) {
        unsigned int raw_nonzero_count = 0;
        if (cudaMemcpy(&raw_nonzero_count,
                       fast_raw_mask_nonzero_device.get(),
                       sizeof(unsigned int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          throw std::runtime_error("failed to copy fast raw mask nonzero count");
        }
        fast_summary.raw_mask_nonzero_pixels = raw_nonzero_count;
      }
      if (fast_post_emit_close_mask_nonzero_device) {
        unsigned int post_close_nonzero_count = 0;
        if (cudaMemcpy(&post_close_nonzero_count,
                       fast_post_emit_close_mask_nonzero_device.get(),
                       sizeof(unsigned int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          throw std::runtime_error("failed to copy fast post-close mask nonzero count");
        }
        fast_summary.post_emit_close_mask_nonzero_pixels = post_close_nonzero_count;
      }
      if (fast_post_smooth_mask_nonzero_device) {
        unsigned int post_smooth_nonzero_count = 0;
        if (cudaMemcpy(&post_smooth_nonzero_count,
                       fast_post_smooth_mask_nonzero_device.get(),
                       sizeof(unsigned int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          throw std::runtime_error("failed to copy fast post-smooth mask nonzero count");
        }
        fast_summary.post_smooth_mask_nonzero_pixels = post_smooth_nonzero_count;
      }
      if (fast_always_on_stripe_count_device) {
        unsigned int always_on_stripe_count = 0;
        if (cudaMemcpy(&always_on_stripe_count,
                       fast_always_on_stripe_count_device.get(),
                       sizeof(unsigned int),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
          throw std::runtime_error("failed to copy always-on stripe count");
        }
        fast_summary.always_on_stripe_count = always_on_stripe_count;
      }
      holoscan::ops::DetectorMaskMessage mask_msg;
      mask_msg.device_pixels = std::move(mask_buffer);
      coherent_emitted_mask_nonzero_pixels = emitted_nonzero_count;
      coherent_emitted_mask_metrics_meaningful = true;
      mask_msg.width = dst_cols;
      mask_msg.height = dst_rows;
      mask_msg.channel = channel_number;
      mask_msg.frame_number = frame_number;
      HOLOSCAN_LOG_DEBUG("Mask emit audit ch={} frame={} dims={}x{} raw_nonzero={} post_smooth_nonzero={} always_on_floor_db={} always_on_stripes={} post_close_nonzero={} post_persistence_nonzero={} emitted_nonzero={} device_ptr=yes",
                        channel_number,
                        frame_number,
                        dst_cols,
                        dst_rows,
                        fast_summary.raw_mask_nonzero_pixels,
                        fast_summary.post_smooth_mask_nonzero_pixels,
                        fast_summary.always_on_floor_db,
                        fast_summary.always_on_stripe_count,
                        fast_summary.post_emit_close_mask_nonzero_pixels,
            fast_summary.post_emit_persistence_mask_nonzero_pixels,
                        emitted_nonzero_count);
      op_output.emit(mask_msg, "mask_out");
    }
  }

  meta->set("coherent_emitted_mask_metrics_meaningful", coherent_emitted_mask_metrics_meaningful);
  meta->set("coherent_emitted_mask_nonzero_pixels", coherent_emitted_mask_nonzero_pixels);
  meta->set("coherent_fast_raw_mask_nonzero_pixels",
            fast_summary.raw_mask_nonzero_pixels);
  meta->set("coherent_fast_post_emit_close_mask_nonzero_pixels",
            fast_summary.post_emit_close_mask_nonzero_pixels);
  meta->set("coherent_fast_post_emit_persistence_mask_nonzero_pixels",
            fast_summary.post_emit_persistence_mask_nonzero_pixels);
  meta->set("coherent_fast_post_smooth_mask_nonzero_pixels",
            fast_summary.post_smooth_mask_nonzero_pixels);
  meta->set("coherent_fast_always_on_floor_db",
            fast_summary.always_on_floor_db);
  meta->set("coherent_fast_always_on_stripe_count",
            fast_summary.always_on_stripe_count);
  meta->set("coherent_final_mask_metrics_meaningful",
            coherent_emitted_mask_metrics_meaningful);
  meta->set("coherent_final_mask_nonzero_pixels",
            coherent_emitted_mask_nonzero_pixels);

  if (!timing_enabled) {
    return;
  }

  auto& stats = timing_stats_[channel_number];
  ++stats.window_frames;
  stats.fft_to_detector_enter_total_ms += fft_to_detector_enter_ms;
  stats.fft_to_detector_enter_max_ms = std::max(stats.fft_to_detector_enter_max_ms,
                                                fft_to_detector_enter_ms);
  stats.fft_to_detector_done_total_ms += fft_to_detector_done_ms;
  stats.fft_to_detector_done_max_ms = std::max(stats.fft_to_detector_done_max_ms,
                                               fft_to_detector_done_ms);
  stats.grouped_box_count_total += coherent_grouped_box_count;
  stats.grouped_box_count_max = std::max<uint64_t>(stats.grouped_box_count_max, coherent_grouped_box_count);
  stats.emitted_mask_nonzero_total += coherent_emitted_mask_nonzero_pixels;
  stats.emitted_mask_nonzero_max = std::max<uint64_t>(stats.emitted_mask_nonzero_max, coherent_emitted_mask_nonzero_pixels);
  stats.final_mask_nonzero_total += coherent_reference_mask_nonzero_pixels;
  stats.final_mask_nonzero_max = std::max<uint64_t>(stats.final_mask_nonzero_max, coherent_reference_mask_nonzero_pixels);
  stats.final_mask_component_count_total += static_cast<uint64_t>(std::max(0, 0));
  stats.final_mask_component_count_max = std::max<uint64_t>(
      stats.final_mask_component_count_max,
      static_cast<uint64_t>(std::max(0, 0)));
  stats.final_mask_grouped_box_count_total += static_cast<uint64_t>(std::max(0, 0));
  stats.final_mask_grouped_box_count_max = std::max<uint64_t>(
      stats.final_mask_grouped_box_count_max,
      static_cast<uint64_t>(std::max(0, 0)));
  for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
    stats.total_ms[stage_index] += stage_ms[stage_index];
    stats.max_ms[stage_index] = std::max(stats.max_ms[stage_index], stage_ms[stage_index]);
  }
  const int summary_every = std::max(1, timing_summary_every_n_.get());
  const int summary_window = std::max(1, timing_summary_window_.get());
  const bool emit_summary = (frame_number % static_cast<uint64_t>(summary_every) == 0) ||
                            (stats.window_frames >= static_cast<uint64_t>(summary_window));
  if (!emit_summary) {
    return;
  }

  const double inv_frames = 1.0 / static_cast<double>(std::max<uint64_t>(1, stats.window_frames));
  std::ostringstream oss;
  oss << "Coherent power timing summary ch=" << channel_number
      << " frames=" << stats.window_frames
      << " fft_to_detector_enter_mean=" << (stats.fft_to_detector_enter_total_ms * inv_frames)
      << " fft_to_detector_enter_max=" << stats.fft_to_detector_enter_max_ms
      << " fft_to_detector_done_mean=" << (stats.fft_to_detector_done_total_ms * inv_frames)
      << " fft_to_detector_done_max=" << stats.fft_to_detector_done_max_ms
      << " grouped_box_count_mean=" << (static_cast<double>(stats.grouped_box_count_total) * inv_frames)
      << " grouped_box_count_max=" << stats.grouped_box_count_max
      << " grouped_box_count_meaningful="
      << 1
      << " emitted_mask_metrics_meaningful=" << (coherent_emitted_mask_metrics_meaningful ? 1 : 0)
      << " emitted_mask_nonzero_mean=" << (static_cast<double>(stats.emitted_mask_nonzero_total) * inv_frames)
      << " emitted_mask_nonzero_max=" << stats.emitted_mask_nonzero_max;
  for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
    const double mean_ms = stats.total_ms[stage_index] * inv_frames;
    oss << ' ' << kTimingStageNames[stage_index] << "_mean=" << mean_ms
        << ' ' << kTimingStageNames[stage_index] << "_max=" << stats.max_ms[stage_index];
  }
  HOLOSCAN_LOG_INFO("{}", oss.str());
  stats = ChannelTimingStats {};


  // count nonzero pixel

}

}   // namespace holoscan::ops
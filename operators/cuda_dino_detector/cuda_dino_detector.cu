// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_dino_detector.hpp"
#include "cuda_dino_torch_helpers.hpp"
#include "cuda_dino_types.hpp"
#include "../../applications/usrp_wideband_signal_detection/spectrogram_visualization.hpp"

#include <cuda_fp16.h>
#include <dinov3_torch_runtime.hpp>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace holoscan::ops {

namespace {

bool use_fp16_precision(const std::string& dtype_text) {
  std::string lowered = dtype_text;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return lowered == "fp16" || lowered == "half" || lowered == "float16";
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
  const auto result = cudaMalloc(&ptr, bytes);
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

std::shared_ptr<uint8_t> acquire_pooled_u8_buffer(size_t bytes) {
  return std::shared_ptr<uint8_t>(static_cast<uint8_t*>(acquire_mask_output_buffer(bytes)),
                                  [bytes](uint8_t* ptr) { recycle_mask_output_buffer(ptr, bytes); });
}

}  // namespace

namespace {

enum TimingStageIndex {
  kStagePowerDb = 0,
  kStageFrontend = 1,
  kStageChunkPlan = 2,
  kStageChunkPack = 3,
  kStageRuntime = 4,
  kStageRawDino = 5,
};

struct IgnoreSidebandInfo {
  int applied_bins = 0;
  double applied_hz = 0.0;
  double bin_hz = 0.0;
  std::vector<uint8_t> valid_row_mask;
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
  std::vector<uint8_t> grouped_mask;
  std::vector<DetectionBox> boxes;
  float peak_score_floor = 0.0f;
};

struct LinearResizeSample {
  int index0 = 0;
  int index1 = 0;
  float t = 0.0f;
};

struct DebugChunkResult {
  int chunk_index = 0;
  int row_start = 0;
  int row_stop = 0;
  int src_rows = 0;
  int src_cols = 0;
  int dst_rows = 0;
  int dst_cols = 0;
  std::vector<float> hybrid_keep_freq;
  std::vector<float> hybrid_keep_res;
  std::vector<uint8_t> hybrid_filled_mask;
  std::vector<uint8_t> hybrid_filled_mask_source;
  std::vector<uint8_t> hybrid_seed_mask;
  std::vector<uint8_t> hybrid_closed_mask;
  std::vector<uint8_t> hybrid_component_filtered_mask;
  std::vector<uint8_t> hybrid_component_filtered_mask_source;
  std::vector<uint8_t> final_mask;
  std::vector<uint8_t> final_mask_source;
  std::vector<float> combined_score;
  float hybrid_seed_freq_threshold = 0.0f;
  float hybrid_seed_res_threshold = 0.0f;
  float hybrid_combined_threshold = 0.0f;
  std::vector<uint8_t> grouped_mask_source;
  std::vector<DetectionBox> grouped_boxes;
};

struct OperatorTimingProfile {
  double total_compute_ms = 0.0;
  double power_db_ms = 0.0;
  double frontend_ms = 0.0;
  double chunk_plan_ms = 0.0;
  double chunk_pack_ms = 0.0;
  double coherence_batch_ms = 0.0;
  double runtime_batch_ms = 0.0;
  double runtime_crop_align_ms = 0.0;
  double runtime_resize_ms = 0.0;
  double runtime_model_prep_ms = 0.0;
  double runtime_torch_forward_ms = 0.0;
  double runtime_dino_score_ms = 0.0;
  double raw_score_projection_ms = 0.0;
  double hybrid_batch_ms = 0.0;
  double hybrid_normalization_ms = 0.0;
  double hybrid_residual_stack_ms = 0.0;
  double hybrid_threshold_extract_ms = 0.0;
  double hybrid_closing_ms = 0.0;
  double hybrid_fill_holes_ms = 0.0;
  double hybrid_component_filter_ms = 0.0;
  double hybrid_output_copy_ms = 0.0;
  double debug_device_to_host_ms = 0.0;
  double debug_chunk_grouping_ms = 0.0;
  double global_merge_ms = 0.0;
  double artifact_serialization_ms = 0.0;
};

double elapsed_ms_since(const std::chrono::steady_clock::time_point& start_time) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count();
}

bool write_npy_2d(const std::filesystem::path& path,
                  const void* payload,
                  size_t payload_bytes,
                  int rows,
                  int cols,
                  const std::string& dtype_descr) {
  const auto tmp_path = path.string() + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }
  std::ostringstream header_stream;
  header_stream << "{'descr': '" << dtype_descr << "', 'fortran_order': False, 'shape': (" << rows << ", " << cols << "), }";
  std::string header = header_stream.str();
  const size_t preamble = 10;
  size_t padding = 16 - ((preamble + header.size() + 1) % 16);
  if (padding == 16) {
    padding = 0;
  }
  header.append(padding, ' ');
  header.push_back('\n');

  out.write("\x93NUMPY", 6);
  const unsigned char version[2] = {1, 0};
  out.write(reinterpret_cast<const char*>(version), 2);
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  const unsigned char header_bytes[2] = {
      static_cast<unsigned char>(header_len & 0xFFU),
      static_cast<unsigned char>((header_len >> 8U) & 0xFFU),
  };
  out.write(reinterpret_cast<const char*>(header_bytes), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(payload), static_cast<std::streamsize>(payload_bytes));
  out.close();
  if (!out.good()) {
    std::error_code cleanup_error;
    std::filesystem::remove(tmp_path, cleanup_error);
    return false;
  }
  std::error_code remove_error;
  std::filesystem::remove(path, remove_error);
  std::error_code rename_error;
  std::filesystem::rename(tmp_path, path, rename_error);
  if (rename_error) {
    std::error_code cleanup_error;
    std::filesystem::remove(tmp_path, cleanup_error);
    return false;
  }
  return true;
}

bool write_pgm(const std::filesystem::path& path,
               const std::vector<uint8_t>& image,
               int width,
               int height) {
  const auto tmp_path = path.string() + ".tmp";
  std::ofstream out(tmp_path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
  out.close();
  if (!out.good()) {
    std::error_code cleanup_error;
    std::filesystem::remove(tmp_path, cleanup_error);
    return false;
  }
  std::error_code remove_error;
  std::filesystem::remove(path, remove_error);
  std::error_code rename_error;
  std::filesystem::rename(tmp_path, path, rename_error);
  if (rename_error) {
    std::error_code cleanup_error;
    std::filesystem::remove(tmp_path, cleanup_error);
    return false;
  }
  return true;
}

std::vector<uint8_t> build_spectrogram_preview(const std::vector<cuda_dino_complex>& host_fft,
                                               int src_rows,
                                               int src_cols,
                                               int dst_rows,
                                               int dst_cols) {
  std::vector<float> reduced(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), -120.0f);
  for (int r = 0; r < dst_rows; ++r) {
    const int r0 = (r * src_rows) / dst_rows;
    const int r1 = ((r + 1) * src_rows) / dst_rows;
    for (int c = 0; c < dst_cols; ++c) {
      const int c0 = (c * src_cols) / dst_cols;
      const int c1 = ((c + 1) * src_cols) / dst_cols;

      double accum = 0.0;
      int count = 0;
      for (int rr = r0; rr < std::max(r0 + 1, r1); ++rr) {
        for (int cc = c0; cc < std::max(c0 + 1, c1); ++cc) {
          const auto& value = host_fft[static_cast<size_t>(rr) * static_cast<size_t>(src_cols) +
                                       static_cast<size_t>(cc)];
          const float power = value.real() * value.real() + value.imag() * value.imag() + 1.0e-12f;
          accum += 10.0 * std::log10(power);
          ++count;
        }
      }
      reduced[static_cast<size_t>(r) * static_cast<size_t>(dst_cols) + static_cast<size_t>(c)] =
          static_cast<float>(accum / static_cast<double>(std::max(1, count)));
    }
  }

  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  for (const float value : reduced) {
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }
  const float denom = std::max(1.0e-6f, max_value - min_value);

  std::vector<uint8_t> image(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), 0);
  for (size_t index = 0; index < reduced.size(); ++index) {
    const float normalized = (reduced[index] - min_value) / denom;
    image[index] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
  }
  return image;
}

template <typename T>
std::vector<T> transpose_host_row_major_matrix(const std::vector<T>& input, int rows, int cols) {
  if (rows <= 0 || cols <= 0 || input.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
    return {};
  }

  std::vector<T> output(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      output[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] =
          input[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    }
  }
  return output;
}

std::filesystem::path make_aligned_spectrogram_path(const std::filesystem::path& root,
                                                    const std::string& subdir,
                                                    const std::string& prefix,
                                                    int channel,
                                                    uint64_t frame_number,
                                                    int rows,
                                                    int cols,
                                                    const std::string& extension) {
  std::ostringstream filename;
  filename << prefix << "_ch" << channel << "_f" << frame_number << "_" << rows << "x" << cols << extension;
  return root / subdir / filename.str();
}

std::vector<float> mask_to_float(const std::vector<uint8_t>& input) {
  std::vector<float> output(input.size(), 0.0f);
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index] ? 1.0f : 0.0f;
  }
  return output;
}

std::string json_escape(const std::string& input) {
  std::string output;
  output.reserve(input.size());
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      default:
        output.push_back(ch);
        break;
    }
  }
  return output;
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open output file: " + path.string());
  }
  out << text;
  if (!out.good()) {
    throw std::runtime_error("failed to write output file: " + path.string());
  }
}

std::string detection_boxes_to_json(const std::vector<DetectionBox>& boxes) {
  std::ostringstream out;
  out << "[\n";
  for (size_t index = 0; index < boxes.size(); ++index) {
    const auto& box = boxes[index];
    out << "  {\n";
    out << "    \"freq_start\": " << box.freq_start << ",\n";
    out << "    \"freq_stop\": " << box.freq_stop << ",\n";
    out << "    \"time_start\": " << box.time_start << ",\n";
    out << "    \"time_stop\": " << box.time_stop << ",\n";
    out << "    \"filled_area\": " << box.filled_area << ",\n";
    out << "    \"density\": " << std::setprecision(8) << box.density << ",\n";
    out << "    \"bbox_density\": " << std::setprecision(8) << box.bbox_density << ",\n";
    out << "    \"envelope_density\": " << std::setprecision(8) << box.envelope_density << ",\n";
    out << "    \"score_mean\": " << std::setprecision(8) << box.score_mean << ",\n";
    out << "    \"score_peak\": " << std::setprecision(8) << box.score_peak << "\n";
    out << "  }";
    if (index + 1 < boxes.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "]\n";
  return out.str();
}

std::string chunk_plan_to_json(const std::vector<ChunkPlanEntry>& chunk_plan) {
  std::ostringstream out;
  out << "[\n";
  for (size_t index = 0; index < chunk_plan.size(); ++index) {
    const auto& chunk = chunk_plan[index];
    out << "  {\n";
    out << "    \"chunk_index\": " << chunk.chunk_index << ",\n";
    out << "    \"row_start\": " << chunk.row_start << ",\n";
    out << "    \"row_stop\": " << chunk.row_stop << ",\n";
    out << "    \"freq_start_hz\": " << std::setprecision(16) << chunk.freq_start_hz << ",\n";
    out << "    \"freq_stop_hz\": " << std::setprecision(16) << chunk.freq_stop_hz << "\n";
    out << "  }";
    if (index + 1 < chunk_plan.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "]\n";
  return out.str();
}

struct GlobalMergedResult {
  std::vector<uint8_t> projected_grouped_mask;
  std::vector<float> projected_grouped_score;
  std::vector<uint8_t> stitched_final_mask;
  std::vector<DetectionBox> projected_boxes;
  std::vector<DetectionBox> merged_boxes;
  std::vector<uint8_t> merged_box_mask;
};

struct ChunkOwnershipRange {
  int row_start = 0;
  int row_stop = 0;
};

struct ComponentLabelling {
  std::vector<int> labels;
  std::vector<int> sizes;
  int count = 0;
};

int chunk_row_count(const ChunkPlanEntry& chunk) {
  return std::max(0, chunk.row_stop - chunk.row_start);
}

struct UniformChunkGeometry {
  int chunk_rows = 0;
  int overlap_rows = 0;
  int step_rows = 0;
};

struct PlannedIgnoreSidebandSelection {
  int applied_bins = 0;
  std::vector<uint8_t> valid_row_mask;
  std::vector<ChunkPlanEntry> chunk_plan;
};

__host__ __device__ __forceinline__ size_t flat_index(int cols, int row, int col) {
  return static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
}

__global__ void transpose_complex_matrix_kernel(const cuda_dino_complex* input,
                                                int input_rows,
                                                int input_cols,
                                                cuda_dino_complex* output) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= input_rows || col >= input_cols) {
    return;
  }

  output[flat_index(input_rows, col, row)] = input[flat_index(input_cols, row, col)];
}

__global__ void transpose_u8_matrix_kernel(const uint8_t* input,
                                           int input_rows,
                                           int input_cols,
                                           uint8_t* output) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= input_rows || col >= input_cols) {
    return;
  }

  output[flat_index(input_rows, col, row)] = input[flat_index(input_cols, row, col)];
}

template <typename T>
__host__ __device__ __forceinline__ T clamp_value(T value, T low, T high) {
  return value < low ? low : (value > high ? high : value);
}

struct DirectionalCoherenceCudaScratch {
  float* background_cols = nullptr;
  float* background = nullptr;
  float* residual = nullptr;
  float* directional_delta = nullptr;
  uint8_t* valid_row_mask = nullptr;
  size_t float_capacity = 0;
  size_t row_mask_capacity = 0;

  ~DirectionalCoherenceCudaScratch() {
    release();
  }

  void release() {
    if (background_cols != nullptr) {
      cudaFree(background_cols);
      background_cols = nullptr;
    }
    if (background != nullptr) {
      cudaFree(background);
      background = nullptr;
    }
    if (residual != nullptr) {
      cudaFree(residual);
      residual = nullptr;
    }
    if (directional_delta != nullptr) {
      cudaFree(directional_delta);
      directional_delta = nullptr;
    }
    if (valid_row_mask != nullptr) {
      cudaFree(valid_row_mask);
      valid_row_mask = nullptr;
    }
    float_capacity = 0;
    row_mask_capacity = 0;
  }

  bool ensure_capacity(size_t requested_float_capacity, size_t requested_row_mask_capacity) {
    if (requested_float_capacity > float_capacity) {
      release_float_buffers();
      if (cudaMalloc(reinterpret_cast<void**>(&background_cols), requested_float_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&background), requested_float_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&residual), requested_float_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&directional_delta), requested_float_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      float_capacity = requested_float_capacity;
    }
    if (requested_row_mask_capacity > row_mask_capacity) {
      if (valid_row_mask != nullptr) {
        cudaFree(valid_row_mask);
        valid_row_mask = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&valid_row_mask), requested_row_mask_capacity * sizeof(uint8_t)) != cudaSuccess) {
        release();
        return false;
      }
      row_mask_capacity = requested_row_mask_capacity;
    }
    return true;
  }

 private:
  void release_float_buffers() {
    if (background_cols != nullptr) {
      cudaFree(background_cols);
      background_cols = nullptr;
    }
    if (background != nullptr) {
      cudaFree(background);
      background = nullptr;
    }
    if (residual != nullptr) {
      cudaFree(residual);
      residual = nullptr;
    }
    if (directional_delta != nullptr) {
      cudaFree(directional_delta);
      directional_delta = nullptr;
    }
    float_capacity = 0;
  }
};

DirectionalCoherenceCudaScratch& directional_coherence_cuda_scratch() {
  static DirectionalCoherenceCudaScratch scratch;
  return scratch;
}

struct FillHolesCudaScratch {
  uint8_t* background = nullptr;
  uint8_t* grown_a = nullptr;
  uint8_t* grown_b = nullptr;
  uint32_t* changed = nullptr;
  size_t mask_capacity = 0;

  ~FillHolesCudaScratch() {
    release();
  }

  void release() {
    if (background != nullptr) {
      cudaFree(background);
      background = nullptr;
    }
    if (grown_a != nullptr) {
      cudaFree(grown_a);
      grown_a = nullptr;
    }
    if (grown_b != nullptr) {
      cudaFree(grown_b);
      grown_b = nullptr;
    }
    if (changed != nullptr) {
      cudaFree(changed);
      changed = nullptr;
    }
    mask_capacity = 0;
  }

  bool ensure_capacity(size_t requested_mask_capacity) {
    if (requested_mask_capacity <= mask_capacity && background != nullptr && grown_a != nullptr && grown_b != nullptr && changed != nullptr) {
      return true;
    }
    release();
    if (cudaMalloc(reinterpret_cast<void**>(&background), requested_mask_capacity * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&grown_a), requested_mask_capacity * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&grown_b), requested_mask_capacity * sizeof(uint8_t)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&changed), sizeof(uint32_t)) != cudaSuccess) {
      release();
      return false;
    }
    mask_capacity = requested_mask_capacity;
    return true;
  }
};

FillHolesCudaScratch& fill_holes_cuda_scratch() {
  static FillHolesCudaScratch scratch;
  return scratch;
}

constexpr int kRawScoreProjectionBasis = 16;
constexpr int kRawScoreProjectionThreads = 128;

bool invert_small_square_matrix(const std::array<double, kRawScoreProjectionBasis * kRawScoreProjectionBasis>& input,
                                std::array<double, kRawScoreProjectionBasis * kRawScoreProjectionBasis>& inverse) {
  auto working = input;
  inverse.fill(0.0);
  for (int index = 0; index < kRawScoreProjectionBasis; ++index) {
    inverse[static_cast<size_t>(index) * kRawScoreProjectionBasis + static_cast<size_t>(index)] = 1.0;
  }

  for (int column = 0; column < kRawScoreProjectionBasis; ++column) {
    int pivot_row = column;
    double pivot_value = std::fabs(working[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(column)]);
    for (int row = column + 1; row < kRawScoreProjectionBasis; ++row) {
      const double candidate = std::fabs(working[static_cast<size_t>(row) * kRawScoreProjectionBasis + static_cast<size_t>(column)]);
      if (candidate > pivot_value) {
        pivot_value = candidate;
        pivot_row = row;
      }
    }

    if (pivot_value <= 1.0e-12) {
      return false;
    }

    if (pivot_row != column) {
      for (int entry = 0; entry < kRawScoreProjectionBasis; ++entry) {
        std::swap(working[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(entry)],
                  working[static_cast<size_t>(pivot_row) * kRawScoreProjectionBasis + static_cast<size_t>(entry)]);
        std::swap(inverse[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(entry)],
                  inverse[static_cast<size_t>(pivot_row) * kRawScoreProjectionBasis + static_cast<size_t>(entry)]);
      }
    }

    const double pivot = working[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(column)];
    for (int entry = 0; entry < kRawScoreProjectionBasis; ++entry) {
      working[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(entry)] /= pivot;
      inverse[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(entry)] /= pivot;
    }

    for (int row = 0; row < kRawScoreProjectionBasis; ++row) {
      if (row == column) {
        continue;
      }
      const double factor = working[static_cast<size_t>(row) * kRawScoreProjectionBasis + static_cast<size_t>(column)];
      if (std::fabs(factor) <= 1.0e-18) {
        continue;
      }
      for (int entry = 0; entry < kRawScoreProjectionBasis; ++entry) {
        working[static_cast<size_t>(row) * kRawScoreProjectionBasis + static_cast<size_t>(entry)] -=
            factor * working[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(entry)];
        inverse[static_cast<size_t>(row) * kRawScoreProjectionBasis + static_cast<size_t>(entry)] -=
            factor * inverse[static_cast<size_t>(column) * kRawScoreProjectionBasis + static_cast<size_t>(entry)];
      }
    }
  }

  return true;
}

bool build_positional_suppression_matrices_host(int patch_rows,
                                                int patch_cols,
                                                std::vector<float>& design,
                                                std::vector<float>& projection_left) {
  constexpr float kPi = 3.14159265358979323846f;
  const int patch_count = patch_rows * patch_cols;
  if (patch_rows <= 0 || patch_cols <= 0 || patch_count <= 0) {
    design.clear();
    projection_left.clear();
    return false;
  }

  design.assign(static_cast<size_t>(patch_count) * static_cast<size_t>(kRawScoreProjectionBasis), 0.0f);
  for (int row = 0; row < patch_rows; ++row) {
    const float row_coord = patch_rows > 1 ? -1.0f + 2.0f * static_cast<float>(row) / static_cast<float>(patch_rows - 1) : 0.0f;
    for (int col = 0; col < patch_cols; ++col) {
      const float col_coord = patch_cols > 1 ? -1.0f + 2.0f * static_cast<float>(col) / static_cast<float>(patch_cols - 1) : 0.0f;
      const size_t base = (static_cast<size_t>(row) * static_cast<size_t>(patch_cols) + static_cast<size_t>(col)) *
                          static_cast<size_t>(kRawScoreProjectionBasis);
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

  std::array<double, kRawScoreProjectionBasis * kRawScoreProjectionBasis> gram{};
  for (int lhs = 0; lhs < kRawScoreProjectionBasis; ++lhs) {
    for (int rhs = 0; rhs < kRawScoreProjectionBasis; ++rhs) {
      double sum = lhs == rhs ? 1.0e-3 : 0.0;
      for (int patch = 0; patch < patch_count; ++patch) {
        const size_t base = static_cast<size_t>(patch) * static_cast<size_t>(kRawScoreProjectionBasis);
        sum += static_cast<double>(design[base + static_cast<size_t>(lhs)]) *
               static_cast<double>(design[base + static_cast<size_t>(rhs)]);
      }
      gram[static_cast<size_t>(lhs) * kRawScoreProjectionBasis + static_cast<size_t>(rhs)] = sum;
    }
  }

  std::array<double, kRawScoreProjectionBasis * kRawScoreProjectionBasis> gram_inverse{};
  if (!invert_small_square_matrix(gram, gram_inverse)) {
    design.clear();
    projection_left.clear();
    return false;
  }

  projection_left.assign(static_cast<size_t>(kRawScoreProjectionBasis) * static_cast<size_t>(patch_count), 0.0f);
  for (int basis = 0; basis < kRawScoreProjectionBasis; ++basis) {
    for (int patch = 0; patch < patch_count; ++patch) {
      double sum = 0.0;
      const size_t design_base = static_cast<size_t>(patch) * static_cast<size_t>(kRawScoreProjectionBasis);
      for (int coeff = 0; coeff < kRawScoreProjectionBasis; ++coeff) {
        sum += gram_inverse[static_cast<size_t>(basis) * kRawScoreProjectionBasis + static_cast<size_t>(coeff)] *
               static_cast<double>(design[design_base + static_cast<size_t>(coeff)]);
      }
      projection_left[static_cast<size_t>(basis) * static_cast<size_t>(patch_count) + static_cast<size_t>(patch)] =
          static_cast<float>(sum);
    }
  }

  return true;
}

struct PositionalSuppressionCudaCache {
  int patch_rows = 0;
  int patch_cols = 0;
  float* design = nullptr;
  float* projection_left = nullptr;
  size_t patch_count_capacity = 0;

  ~PositionalSuppressionCudaCache() {
    release();
  }

  void release() {
    if (design != nullptr) {
      cudaFree(design);
      design = nullptr;
    }
    if (projection_left != nullptr) {
      cudaFree(projection_left);
      projection_left = nullptr;
    }
    patch_rows = 0;
    patch_cols = 0;
    patch_count_capacity = 0;
  }

  bool ensure_capacity(int requested_patch_rows, int requested_patch_cols, cudaStream_t stream) {
    const int requested_patch_count = requested_patch_rows * requested_patch_cols;
    if (requested_patch_rows <= 0 || requested_patch_cols <= 0 || requested_patch_count <= 0) {
      return false;
    }
    if (patch_rows == requested_patch_rows && patch_cols == requested_patch_cols && patch_count_capacity == static_cast<size_t>(requested_patch_count) &&
        design != nullptr && projection_left != nullptr) {
      return true;
    }

    std::vector<float> host_design;
    std::vector<float> host_projection_left;
    if (!build_positional_suppression_matrices_host(requested_patch_rows,
                                                    requested_patch_cols,
                                                    host_design,
                                                    host_projection_left)) {
      return false;
    }

    release();
    const size_t design_count = static_cast<size_t>(requested_patch_count) * static_cast<size_t>(kRawScoreProjectionBasis);
    const size_t projection_count = static_cast<size_t>(kRawScoreProjectionBasis) * static_cast<size_t>(requested_patch_count);
    if (cudaMalloc(reinterpret_cast<void**>(&design), design_count * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&projection_left), projection_count * sizeof(float)) != cudaSuccess) {
      release();
      return false;
    }

    if (cudaMemcpyAsync(design,
                        host_design.data(),
                        design_count * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream) != cudaSuccess ||
        cudaMemcpyAsync(projection_left,
                        host_projection_left.data(),
                        projection_count * sizeof(float),
                        cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      release();
      return false;
    }

    patch_rows = requested_patch_rows;
    patch_cols = requested_patch_cols;
    patch_count_capacity = static_cast<size_t>(requested_patch_count);
    return true;
  }
};

PositionalSuppressionCudaCache& positional_suppression_cuda_cache() {
  static thread_local PositionalSuppressionCudaCache cache;
  return cache;
}

struct RawScoreProjectionCudaScratch {
  float* beta = nullptr;
  float* patch_values = nullptr;
  float* aligned_maps = nullptr;
  float* temp_plane = nullptr;
  float* low_values = nullptr;
  float* high_values = nullptr;
  size_t beta_capacity = 0;
  size_t patch_capacity = 0;
  size_t aligned_capacity = 0;
  size_t temp_plane_capacity = 0;
  size_t batch_capacity = 0;

  ~RawScoreProjectionCudaScratch() {
    release();
  }

  void release() {
    if (beta != nullptr) {
      cudaFree(beta);
      beta = nullptr;
    }
    if (patch_values != nullptr) {
      cudaFree(patch_values);
      patch_values = nullptr;
    }
    if (aligned_maps != nullptr) {
      cudaFree(aligned_maps);
      aligned_maps = nullptr;
    }
    if (temp_plane != nullptr) {
      cudaFree(temp_plane);
      temp_plane = nullptr;
    }
    if (low_values != nullptr) {
      cudaFree(low_values);
      low_values = nullptr;
    }
    if (high_values != nullptr) {
      cudaFree(high_values);
      high_values = nullptr;
    }
    beta_capacity = 0;
    patch_capacity = 0;
    aligned_capacity = 0;
    temp_plane_capacity = 0;
    batch_capacity = 0;
  }

  bool ensure_capacity(size_t requested_beta_capacity,
                       size_t requested_patch_capacity,
                       size_t requested_aligned_capacity,
                       size_t requested_temp_plane_capacity,
                       size_t requested_batch_capacity) {
    if (requested_beta_capacity > beta_capacity) {
      if (beta != nullptr) {
        cudaFree(beta);
        beta = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&beta), requested_beta_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      beta_capacity = requested_beta_capacity;
    }
    if (requested_patch_capacity > patch_capacity) {
      if (patch_values != nullptr) {
        cudaFree(patch_values);
        patch_values = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&patch_values), requested_patch_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      patch_capacity = requested_patch_capacity;
    }
    if (requested_aligned_capacity > aligned_capacity) {
      if (aligned_maps != nullptr) {
        cudaFree(aligned_maps);
        aligned_maps = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&aligned_maps), requested_aligned_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      aligned_capacity = requested_aligned_capacity;
    }
    if (requested_temp_plane_capacity > temp_plane_capacity) {
      if (temp_plane != nullptr) {
        cudaFree(temp_plane);
        temp_plane = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&temp_plane), requested_temp_plane_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      temp_plane_capacity = requested_temp_plane_capacity;
    }
    if (requested_batch_capacity > batch_capacity) {
      if (low_values != nullptr) {
        cudaFree(low_values);
        low_values = nullptr;
      }
      if (high_values != nullptr) {
        cudaFree(high_values);
        high_values = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&low_values), requested_batch_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&high_values), requested_batch_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      batch_capacity = requested_batch_capacity;
    }
    return true;
  }
};

RawScoreProjectionCudaScratch& raw_score_projection_cuda_scratch() {
  static RawScoreProjectionCudaScratch scratch;
  return scratch;
}

struct ComponentFilterCudaScratch {
  int* labels_a = nullptr;
  int* labels_b = nullptr;
  int* component_counts = nullptr;
  uint32_t* changed = nullptr;
  size_t label_capacity = 0;
  size_t count_capacity = 0;

  ~ComponentFilterCudaScratch() {
    release();
  }

  void release() {
    if (labels_a != nullptr) {
      cudaFree(labels_a);
      labels_a = nullptr;
    }
    if (labels_b != nullptr) {
      cudaFree(labels_b);
      labels_b = nullptr;
    }
    if (component_counts != nullptr) {
      cudaFree(component_counts);
      component_counts = nullptr;
    }
    if (changed != nullptr) {
      cudaFree(changed);
      changed = nullptr;
    }
    label_capacity = 0;
    count_capacity = 0;
  }

  bool ensure_capacity(size_t requested_label_capacity, size_t requested_count_capacity) {
    if (requested_label_capacity > label_capacity) {
      if (labels_a != nullptr) {
        cudaFree(labels_a);
        labels_a = nullptr;
      }
      if (labels_b != nullptr) {
        cudaFree(labels_b);
        labels_b = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&labels_a), requested_label_capacity * sizeof(int)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&labels_b), requested_label_capacity * sizeof(int)) != cudaSuccess) {
        release();
        return false;
      }
      label_capacity = requested_label_capacity;
    }
    if (requested_count_capacity > count_capacity) {
      if (component_counts != nullptr) {
        cudaFree(component_counts);
        component_counts = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&component_counts), requested_count_capacity * sizeof(int)) != cudaSuccess) {
        release();
        return false;
      }
      count_capacity = requested_count_capacity;
    }
    if (changed == nullptr) {
      if (cudaMalloc(reinterpret_cast<void**>(&changed), sizeof(uint32_t)) != cudaSuccess) {
        release();
        return false;
      }
    }
    return true;
  }
};

ComponentFilterCudaScratch& component_filter_cuda_scratch() {
  static ComponentFilterCudaScratch scratch;
  return scratch;
}

__global__ void directional_box_mean_cols_batch_kernel(const float* input,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       int radius_cols,
                                                       float* output) {
  const int line = blockIdx.y;
  if (line >= batch_size * rows || cols <= 0) {
    return;
  }

  const int tile_start = blockIdx.x * blockDim.x;
  const int local = threadIdx.x;
  const int col = tile_start + local;
  const int batch_index = line / rows;
  const int row = line - batch_index * rows;
  const size_t base_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(rows) * static_cast<size_t>(cols) +
                             static_cast<size_t>(row) * static_cast<size_t>(cols);

  extern __shared__ float shared_line[];
  shared_line[local + radius_cols] = (col < cols) ? input[base_offset + static_cast<size_t>(col)] : 0.0f;
  if (local < radius_cols) {
    const int left_col = tile_start + local - radius_cols;
    const int right_col = tile_start + static_cast<int>(blockDim.x) + local;
    shared_line[local] = (left_col >= 0 && left_col < cols) ? input[base_offset + static_cast<size_t>(left_col)] : 0.0f;
    shared_line[local + radius_cols + blockDim.x] =
        (right_col >= 0 && right_col < cols) ? input[base_offset + static_cast<size_t>(right_col)] : 0.0f;
  }
  __syncthreads();

  if (col >= cols) {
    return;
  }

  const int window_start = max(0, col - radius_cols);
  const int window_stop = min(cols - 1, col + radius_cols);
  float sum = 0.0f;
  for (int src_col = window_start; src_col <= window_stop; ++src_col) {
    sum += shared_line[src_col - tile_start + radius_cols];
  }
  const int count = window_stop - window_start + 1;
  output[base_offset + static_cast<size_t>(col)] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void directional_box_mean_rows_batch_kernel(const float* input,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       int radius_rows,
                                                       float* output) {
  const int line = blockIdx.y;
  if (line >= batch_size * cols || rows <= 0) {
    return;
  }

  const int tile_start = blockIdx.x * blockDim.x;
  const int local = threadIdx.x;
  const int row = tile_start + local;
  const int batch_index = line / cols;
  const int col = line - batch_index * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(rows) * static_cast<size_t>(cols);

  extern __shared__ float shared_line[];
  shared_line[local + radius_rows] = (row < rows) ? input[batch_offset + flat_index(cols, row, col)] : 0.0f;
  if (local < radius_rows) {
    const int top_row = tile_start + local - radius_rows;
    const int bottom_row = tile_start + static_cast<int>(blockDim.x) + local;
    shared_line[local] = (top_row >= 0 && top_row < rows) ? input[batch_offset + flat_index(cols, top_row, col)] : 0.0f;
    shared_line[local + radius_rows + blockDim.x] =
        (bottom_row >= 0 && bottom_row < rows) ? input[batch_offset + flat_index(cols, bottom_row, col)] : 0.0f;
  }
  __syncthreads();

  if (row >= rows) {
    return;
  }

  const int window_start = max(0, row - radius_rows);
  const int window_stop = min(rows - 1, row + radius_rows);
  float sum = 0.0f;
  for (int src_row = window_start; src_row <= window_stop; ++src_row) {
    sum += shared_line[src_row - tile_start + radius_rows];
  }
  const int count = window_stop - window_start + 1;
  output[batch_offset + flat_index(cols, row, col)] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void directional_subtract_clamp_batch_kernel(const float* input,
                                                        const float* baseline,
                                                        int total,
                                                        float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = fmaxf(input[idx] - baseline[idx], 0.0f);
}

__global__ void directional_weighted_sum_batch_kernel(const float* lhs,
                                                      const float* rhs,
                                                      int total,
                                                      float lhs_weight,
                                                      float rhs_weight,
                                                      float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = lhs_weight * lhs[idx] + rhs_weight * rhs[idx];
}

__global__ void directional_normalize_clamp_batch_kernel(const float* input,
                                                         int total,
                                                         float low,
                                                         float inv_denom,
                                                         float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = fminf(fmaxf((input[idx] - low) * inv_denom, 0.0f), 1.0f);
}

__global__ void directional_apply_valid_rows_batch_kernel(float* values,
                                                          int batch_size,
                                                          int rows,
                                                          int cols,
                                                          const uint8_t* valid_row_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    values[idx] = 0.0f;
  }
}

__global__ void binary_dilate_rows_batch_kernel(const uint8_t* input,
                                                int batch_size,
                                                int rows,
                                                int cols,
                                                int radius,
                                                uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  uint8_t value = 0;
  for (int src_row = max(0, row - radius); src_row <= min(rows - 1, row + radius); ++src_row) {
    if (input[batch_offset + flat_index(cols, src_row, col)] != 0) {
      value = 1;
      break;
    }
  }
  output[idx] = value;
}

__global__ void binary_erode_rows_batch_kernel(const uint8_t* input,
                                               int batch_size,
                                               int rows,
                                               int cols,
                                               int radius,
                                               uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  uint8_t value = 1;
  for (int src_row = max(0, row - radius); src_row <= min(rows - 1, row + radius); ++src_row) {
    if (input[batch_offset + flat_index(cols, src_row, col)] == 0) {
      value = 0;
      break;
    }
  }
  output[idx] = value;
}

__global__ void binary_dilate_cols_batch_kernel(const uint8_t* input,
                                                int batch_size,
                                                int rows,
                                                int cols,
                                                int radius,
                                                uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  uint8_t value = 0;
  for (int src_col = max(0, col - radius); src_col <= min(cols - 1, col + radius); ++src_col) {
    if (input[batch_offset + flat_index(cols, row, src_col)] != 0) {
      value = 1;
      break;
    }
  }
  output[idx] = value;
}

__global__ void binary_erode_cols_batch_kernel(const uint8_t* input,
                                               int batch_size,
                                               int rows,
                                               int cols,
                                               int radius,
                                               uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  uint8_t value = 1;
  for (int src_col = max(0, col - radius); src_col <= min(cols - 1, col + radius); ++src_col) {
    if (input[batch_offset + flat_index(cols, row, src_col)] == 0) {
      value = 0;
      break;
    }
  }
  output[idx] = value;
}

__global__ void fill_holes_init_kernel(const uint8_t* mask,
                                       int batch_size,
                                       int rows,
                                       int cols,
                                       uint8_t* background) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const uint8_t bg = mask[idx] == 0 ? 1 : 0;
  background[idx] = bg;
}

__global__ void fill_holes_seed_border_kernel(const uint8_t* background,
                                              int batch_size,
                                              int rows,
                                              int cols,
                                              uint8_t* exterior) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int local_index = idx % plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  exterior[idx] = (background[idx] != 0 && (row == 0 || row == rows - 1 || col == 0 || col == cols - 1)) ? 1 : 0;
}

__global__ void fill_holes_expand_rows_kernel(const uint8_t* background,
                                              const uint8_t* current,
                                              int batch_size,
                                              int rows,
                                              int cols,
                                              uint8_t* output,
                                              uint32_t* changed) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_lines = batch_size * rows;
  if (line >= total_lines) {
    return;
  }

  const int batch_index = line / rows;
  const int row = line - batch_index * rows;
  const size_t row_offset = (static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)) *
                            static_cast<size_t>(cols);
  int col = 0;
  while (col < cols) {
    const size_t index = row_offset + static_cast<size_t>(col);
    if (background[index] == 0) {
      output[index] = 0;
      ++col;
      continue;
    }

    const int segment_start = col;
    bool reached = false;
    bool already_full = true;
    while (col < cols) {
      const size_t segment_index = row_offset + static_cast<size_t>(col);
      if (background[segment_index] == 0) {
        break;
      }
      const bool active = current[segment_index] != 0;
      reached = reached || active;
      already_full = already_full && active;
      ++col;
    }

    if (reached && !already_full) {
      atomicExch(changed, 1U);
    }
    const uint8_t fill_value = reached ? 1 : 0;
    for (int fill_col = segment_start; fill_col < col; ++fill_col) {
      output[row_offset + static_cast<size_t>(fill_col)] = fill_value;
    }
  }
}

__global__ void fill_holes_expand_cols_kernel(const uint8_t* background,
                                              const uint8_t* current,
                                              int batch_size,
                                              int rows,
                                              int cols,
                                              uint8_t* output,
                                              uint32_t* changed) {
  const int line = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_lines = batch_size * cols;
  if (line >= total_lines) {
    return;
  }

  const int batch_index = line / cols;
  const int col = line - batch_index * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  int row = 0;
  while (row < rows) {
    const size_t index = batch_offset + flat_index(cols, row, col);
    if (background[index] == 0) {
      output[index] = 0;
      ++row;
      continue;
    }

    const int segment_start = row;
    bool reached = false;
    bool already_full = true;
    while (row < rows) {
      const size_t segment_index = batch_offset + flat_index(cols, row, col);
      if (background[segment_index] == 0) {
        break;
      }
      const bool active = current[segment_index] != 0;
      reached = reached || active;
      already_full = already_full && active;
      ++row;
    }

    if (reached && !already_full) {
      atomicExch(changed, 1U);
    }
    const uint8_t fill_value = reached ? 1 : 0;
    for (int fill_row = segment_start; fill_row < row; ++fill_row) {
      output[batch_offset + flat_index(cols, fill_row, col)] = fill_value;
    }
  }
}

__device__ __forceinline__ int connected_components_find_root(const int* parents, int label) {
  int root = label;
  while (root > 0) {
    const int parent = parents[root - 1];
    if (parent == root || parent == 0) {
      break;
    }
    root = parent;
  }
  return root;
}

__device__ __forceinline__ void connected_components_union_labels(int* parents, int label_a, int label_b) {
  while (label_a > 0 && label_b > 0) {
    const int root_a = connected_components_find_root(parents, label_a);
    const int root_b = connected_components_find_root(parents, label_b);
    if (root_a == root_b) {
      return;
    }

    const int high = max(root_a, root_b);
    const int low = min(root_a, root_b);
    const int previous = atomicMin(&parents[high - 1], low);
    if (previous == high || previous == low) {
      return;
    }

    label_a = previous;
    label_b = low;
  }
}

__global__ void connected_components_init_labels_kernel(const uint8_t* mask,
                                                        int batch_size,
                                                        int rows,
                                                        int cols,
                                                        int* parents) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  parents[idx] = mask[idx] != 0 ? (idx + 1) : 0;
}

__global__ void connected_components_union_kernel(const uint8_t* mask,
                                                  int batch_size,
                                                  int rows,
                                                  int cols,
                                                  int* parents) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total || mask[idx] == 0) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  const int label = idx + 1;

  if (col > 0) {
    const size_t neighbor = batch_offset + flat_index(cols, row, col - 1);
    if (mask[neighbor] != 0) {
      connected_components_union_labels(parents, label, static_cast<int>(neighbor) + 1);
    }
  }
  if (row > 0) {
    const size_t neighbor = batch_offset + flat_index(cols, row - 1, col);
    if (mask[neighbor] != 0) {
      connected_components_union_labels(parents, label, static_cast<int>(neighbor) + 1);
    }
    if (col > 0) {
      const size_t diagonal = batch_offset + flat_index(cols, row - 1, col - 1);
      if (mask[diagonal] != 0) {
        connected_components_union_labels(parents, label, static_cast<int>(diagonal) + 1);
      }
    }
    if (col + 1 < cols) {
      const size_t diagonal = batch_offset + flat_index(cols, row - 1, col + 1);
      if (mask[diagonal] != 0) {
        connected_components_union_labels(parents, label, static_cast<int>(diagonal) + 1);
      }
    }
  }
}

__global__ void connected_components_union_4_kernel(const uint8_t* mask,
                                                    int batch_size,
                                                    int rows,
                                                    int cols,
                                                    int* parents) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total || mask[idx] == 0) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  const int label = idx + 1;

  if (col > 0) {
    const size_t neighbor = batch_offset + flat_index(cols, row, col - 1);
    if (mask[neighbor] != 0) {
      connected_components_union_labels(parents, label, static_cast<int>(neighbor) + 1);
    }
  }
  if (row > 0) {
    const size_t neighbor = batch_offset + flat_index(cols, row - 1, col);
    if (mask[neighbor] != 0) {
      connected_components_union_labels(parents, label, static_cast<int>(neighbor) + 1);
    }
  }
}

__global__ void connected_components_compress_kernel(const uint8_t* mask,
                                                     int batch_size,
                                                     int rows,
                                                     int cols,
                                                     int* parents) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  parents[idx] = mask[idx] != 0 ? connected_components_find_root(parents, idx + 1) : 0;
}

__global__ void connected_components_count_labels_kernel(const int* labels,
                                                         int batch_size,
                                                         int rows,
                                                         int cols,
                                                         int* counts) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int label = labels[idx];
  if (label > 0) {
    atomicAdd(&counts[label - 1], 1);
  }
}

__global__ void connected_components_mark_border_labels_kernel(const uint8_t* mask,
                                                               const int* labels,
                                                               int batch_size,
                                                               int rows,
                                                               int cols,
                                                               int* border_labels) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total || mask[idx] == 0) {
    return;
  }

  const int local_index = idx % plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
    const int label = labels[idx];
    if (label > 0) {
      atomicExch(&border_labels[label - 1], 1);
    }
  }
}

__global__ void fill_holes_finalize_kernel(const uint8_t* mask,
                                           const uint8_t* background,
                                           const int* background_labels,
                                           const int* border_labels,
                                           int total,
                                           uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int label = background_labels[idx];
  const bool fill_pixel =
      background[idx] != 0 && label > 0 && border_labels[label - 1] == 0;
  output[idx] = (mask[idx] != 0 || fill_pixel) ? 1 : 0;
}

__global__ void fill_holes_finalize_border_grow_kernel(const uint8_t* mask,
                                                       const uint8_t* background,
                                                       const uint8_t* exterior,
                                                       int total,
                                                       uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const bool fill_pixel = background[idx] != 0 && exterior[idx] == 0;
  output[idx] = (mask[idx] != 0 || fill_pixel) ? 1 : 0;
}

__global__ void raw_dino_rms_energy_batch_kernel(const float* patch_features,
                                                 int batch_size,
                                                 int patch_count,
                                                 int feature_dim,
                                                 float* output_patch_scores) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * patch_count;
  if (idx >= total) {
    return;
  }

  const size_t feature_offset = static_cast<size_t>(idx) * static_cast<size_t>(feature_dim);
  float sum_sq = 0.0f;
  for (int feature = 0; feature < feature_dim; ++feature) {
    const float value = patch_features[feature_offset + static_cast<size_t>(feature)];
    sum_sq += value * value;
  }
  output_patch_scores[idx] = sqrtf(fmaxf(sum_sq / static_cast<float>(max(feature_dim, 1)), 1.0e-6f));
}

__global__ void raw_dino_beta_batch_kernel(const float* patch_features,
                                           int batch_size,
                                           int patch_count,
                                           int feature_dim,
                                           const float* projection_left,
                                           float* beta) {
  const int batch_feature = blockIdx.x;
  const int batch_index = batch_feature / feature_dim;
  if (batch_index >= batch_size) {
    return;
  }
  const int feature_index = batch_feature - batch_index * feature_dim;
  const int tid = threadIdx.x;

  __shared__ float partial[kRawScoreProjectionBasis * kRawScoreProjectionThreads];
  float accum[kRawScoreProjectionBasis];
#pragma unroll
  for (int basis = 0; basis < kRawScoreProjectionBasis; ++basis) {
    accum[basis] = 0.0f;
  }

  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim);
  for (int patch = tid; patch < patch_count; patch += blockDim.x) {
    const float value = patch_features[batch_offset + static_cast<size_t>(patch) * static_cast<size_t>(feature_dim) +
                                       static_cast<size_t>(feature_index)];
#pragma unroll
    for (int basis = 0; basis < kRawScoreProjectionBasis; ++basis) {
      accum[basis] += projection_left[static_cast<size_t>(basis) * static_cast<size_t>(patch_count) + static_cast<size_t>(patch)] * value;
    }
  }

#pragma unroll
  for (int basis = 0; basis < kRawScoreProjectionBasis; ++basis) {
    partial[static_cast<size_t>(basis) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(tid)] = accum[basis];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int basis = 0; basis < kRawScoreProjectionBasis; ++basis) {
        partial[static_cast<size_t>(basis) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(tid)] +=
            partial[static_cast<size_t>(basis) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(tid + stride)];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const size_t beta_base = static_cast<size_t>(batch_index) * static_cast<size_t>(kRawScoreProjectionBasis) *
                             static_cast<size_t>(feature_dim) + static_cast<size_t>(feature_index);
#pragma unroll
    for (int basis = 0; basis < kRawScoreProjectionBasis; ++basis) {
      beta[beta_base + static_cast<size_t>(basis) * static_cast<size_t>(feature_dim)] =
          partial[static_cast<size_t>(basis) * static_cast<size_t>(blockDim.x)];
    }
  }
}

__global__ void raw_dino_project_energy_batch_kernel(const float* patch_features,
                                                     int batch_size,
                                                     int patch_count,
                                                     int feature_dim,
                                                     const float* design,
                                                     const float* beta,
                                                     float suppression,
                                                     float* output_patch_scores) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * patch_count;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / patch_count;
  const int patch_index = idx - batch_index * patch_count;
  const float* design_row = design + static_cast<size_t>(patch_index) * static_cast<size_t>(kRawScoreProjectionBasis);
  const size_t feature_offset = (static_cast<size_t>(batch_index) * static_cast<size_t>(patch_count) + static_cast<size_t>(patch_index)) *
                                static_cast<size_t>(feature_dim);
  const size_t beta_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(kRawScoreProjectionBasis) *
                             static_cast<size_t>(feature_dim);

  float sum_sq = 0.0f;
  for (int feature = 0; feature < feature_dim; ++feature) {
    float fitted = 0.0f;
#pragma unroll
    for (int basis = 0; basis < kRawScoreProjectionBasis; ++basis) {
      fitted += design_row[basis] * beta[beta_offset + static_cast<size_t>(basis) * static_cast<size_t>(feature_dim) + static_cast<size_t>(feature)];
    }
    const float value = patch_features[feature_offset + static_cast<size_t>(feature)] - suppression * fitted;
    sum_sq += value * value;
  }
  output_patch_scores[idx] = sqrtf(fmaxf(sum_sq / static_cast<float>(max(feature_dim, 1)), 1.0e-6f));
}

__global__ void component_filter_finalize_kernel(const uint8_t* mask,
                                                 const int* labels,
                                                 int batch_size,
                                                 int rows,
                                                 int cols,
                                                 int min_size,
                                                 const int* counts,
                                                 uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }
  if (mask[idx] == 0) {
    output[idx] = 0;
    return;
  }
  const int label = labels[idx];
  output[idx] = (label > 0 && counts[label - 1] >= min_size) ? 1 : 0;
}

bool label_connected_components_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                                     int batch_size,
                                                     int rows,
                                                     int cols,
                                                     int* labels_device,
                                                     cudaStream_t cuda_stream) {
  if (mask_batch_device == nullptr || labels_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t total = static_cast<size_t>(batch_size) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const int threads = 256;
  const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  connected_components_init_labels_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                           batch_size,
                                                                           rows,
                                                                           cols,
                                                                           labels_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  connected_components_union_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                     batch_size,
                                                                     rows,
                                                                     cols,
                                                                     labels_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  connected_components_compress_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                        batch_size,
                                                                        rows,
                                                                        cols,
                                                                        labels_device);
  return cudaGetLastError() == cudaSuccess;
}

bool label_connected_components_4_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       int* labels_device,
                                                       cudaStream_t cuda_stream) {
  if (mask_batch_device == nullptr || labels_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t total = static_cast<size_t>(batch_size) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const int threads = 256;
  const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  connected_components_init_labels_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                           batch_size,
                                                                           rows,
                                                                           cols,
                                                                           labels_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  connected_components_union_4_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                       batch_size,
                                                                       rows,
                                                                       cols,
                                                                       labels_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  connected_components_compress_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                        batch_size,
                                                                        rows,
                                                                        cols,
                                                                        labels_device);
  return cudaGetLastError() == cudaSuccess;
}

std::vector<uint8_t> expand_row_valid_mask(const std::vector<uint8_t>& src_valid_rows, int cols) {
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

void write_operator_artifact_bundle(const std::filesystem::path& output_dir,
                                    int chunk_count,
                                    int selected_chunk_index,
                                    int src_rows,
                                    int src_cols,
                                    int aligned_rows,
                                    int aligned_cols,
                                    bool runtime_resized_full_chunk,
                                    bool projected_full_chunk,
                                    const ChunkPlanEntry& selected_chunk,
                                    const std::vector<float>& corrected_resized,
                                    const std::vector<float>& raw_score_resized,
                                    const std::vector<float>& raw_score_deweighted_resized,
                                    const std::vector<float>& coherence_gate_resized,
                                    const DebugChunkResult& selected_debug_chunk,
                                    const GlobalMergedResult& global_merged,
                                    const std::vector<ChunkPlanEntry>& chunk_plan,
                                    const OperatorTimingProfile& timing_profile) {
  std::filesystem::create_directories(output_dir / "chunk_debug");

  const auto corrected_resized_path = output_dir / "chunk_debug" / "chunk_corrected_resized.npy";
  const auto raw_score_path = output_dir / "chunk_debug" / "chunk_dino_score_raw.npy";
  const auto raw_score_deweighted_path = output_dir / "chunk_debug" / "chunk_dino_score_raw_deweighted.npy";
  const auto coherence_gate_path = output_dir / "chunk_debug" / "chunk_coherence_gate.npy";
  const auto combined_score_path = output_dir / "chunk_debug" / "chunk_combined_score.npy";
  const auto hybrid_keep_freq_path = output_dir / "chunk_debug" / "chunk_hybrid_keep_freq.npy";
  const auto hybrid_keep_res_path = output_dir / "chunk_debug" / "chunk_hybrid_keep_res.npy";
  const auto hybrid_seed_mask_path = output_dir / "chunk_debug" / "chunk_hybrid_seed_mask.npy";
  const auto hybrid_closed_mask_path = output_dir / "chunk_debug" / "chunk_hybrid_closed_mask.npy";
  const auto hybrid_filled_mask_path = output_dir / "chunk_debug" / "chunk_hybrid_filled_mask.npy";
  const auto hybrid_component_filtered_mask_path = output_dir / "chunk_debug" / "chunk_hybrid_component_filtered_mask.npy";
  const auto grouped_mask_path = output_dir / "chunk_debug" / "chunk_grouped_mask.npy";
  const auto final_mask_path = output_dir / "chunk_debug" / "chunk_final_mask.npy";
  const auto final_mask_source_path = output_dir / "chunk_debug" / "chunk_final_mask_source.npy";
  const auto final_mask_projected_path = output_dir / "chunk_debug" / "chunk_final_mask_projected.npy";
  const auto grouped_boxes_path = output_dir / "chunk_debug" / "chunk_grouped_boxes.json";
  const auto chunk_debug_summary_path = output_dir / "chunk_debug" / "chunk_debug_summary.json";

  const auto projected_grouped_mask_path = output_dir / "offline_projected_grouped_mask.npy";
  const auto projected_grouped_score_path = output_dir / "offline_projected_grouped_score.npy";
  const auto merged_box_mask_path = output_dir / "offline_merged_box_mask.npy";
  const auto final_mask_global_path = output_dir / "offline_final_mask.npy";
  const auto chunk_plan_path = output_dir / "offline_chunk_plan.json";
  const auto projected_boxes_path = output_dir / "offline_projected_boxes.json";
  const auto merged_boxes_path = output_dir / "offline_merged_boxes.json";
  const auto summary_path = output_dir / "offline_validation_summary.json";

  const auto grouped_mask_float = mask_to_float(selected_debug_chunk.grouped_mask_source);
  const auto hybrid_seed_mask_float = mask_to_float(selected_debug_chunk.hybrid_seed_mask);
  const auto hybrid_closed_mask_float = mask_to_float(selected_debug_chunk.hybrid_closed_mask);
  const auto hybrid_filled_mask_float = mask_to_float(selected_debug_chunk.hybrid_filled_mask);
  const auto hybrid_component_filtered_mask_float = mask_to_float(selected_debug_chunk.hybrid_component_filtered_mask);
  const auto final_mask_float = mask_to_float(selected_debug_chunk.final_mask);
  const auto projected_mask_float = mask_to_float(global_merged.projected_grouped_mask);
  const auto merged_box_mask_float = mask_to_float(global_merged.merged_box_mask);
  const auto stitched_final_mask_float = mask_to_float(global_merged.stitched_final_mask);
  const auto grouped_mask_projected_float = mask_to_float(resize_mask_nearest(selected_debug_chunk.grouped_mask_source,
                                                                              selected_debug_chunk.src_rows,
                                                                              selected_debug_chunk.src_cols,
                                                                              selected_debug_chunk.dst_rows,
                                                                              selected_debug_chunk.dst_cols));
  const auto final_mask_source_float = mask_to_float(selected_debug_chunk.final_mask_source);
  const auto final_mask_projected_float = mask_to_float(resize_mask_nearest(selected_debug_chunk.final_mask,
                                                                            selected_debug_chunk.dst_rows,
                                                                            selected_debug_chunk.dst_cols,
                                                                            selected_debug_chunk.src_rows,
                                                                            selected_debug_chunk.src_cols));

  if (!write_npy_2d(corrected_resized_path,
                    corrected_resized.data(),
                    corrected_resized.size() * sizeof(float),
                    selected_debug_chunk.dst_rows,
                    selected_debug_chunk.dst_cols,
                    "<f4") ||
      !write_npy_2d(raw_score_path,
                    raw_score_resized.data(),
                    raw_score_resized.size() * sizeof(float),
                    selected_debug_chunk.dst_rows,
                    selected_debug_chunk.dst_cols,
                    "<f4") ||
      !write_npy_2d(raw_score_deweighted_path,
                    raw_score_deweighted_resized.data(),
                    raw_score_deweighted_resized.size() * sizeof(float),
                    selected_debug_chunk.dst_rows,
                    selected_debug_chunk.dst_cols,
                    "<f4") ||
      !write_npy_2d(coherence_gate_path,
                    coherence_gate_resized.data(),
                    coherence_gate_resized.size() * sizeof(float),
                    selected_debug_chunk.dst_rows,
                    selected_debug_chunk.dst_cols,
                    "<f4") ||
      !write_npy_2d(combined_score_path,
                    selected_debug_chunk.combined_score.data(),
                    selected_debug_chunk.combined_score.size() * sizeof(float),
                    selected_debug_chunk.dst_rows,
                    selected_debug_chunk.dst_cols,
                    "<f4") ||
      !write_npy_2d(hybrid_keep_freq_path,
            selected_debug_chunk.hybrid_keep_freq.data(),
            selected_debug_chunk.hybrid_keep_freq.size() * sizeof(float),
            selected_debug_chunk.dst_rows,
            selected_debug_chunk.dst_cols,
            "<f4") ||
      !write_npy_2d(hybrid_keep_res_path,
            selected_debug_chunk.hybrid_keep_res.data(),
            selected_debug_chunk.hybrid_keep_res.size() * sizeof(float),
            selected_debug_chunk.dst_rows,
            selected_debug_chunk.dst_cols,
            "<f4") ||
      !write_npy_2d(hybrid_seed_mask_path,
            hybrid_seed_mask_float.data(),
            hybrid_seed_mask_float.size() * sizeof(float),
            selected_debug_chunk.dst_rows,
            selected_debug_chunk.dst_cols,
            "<f4") ||
      !write_npy_2d(hybrid_closed_mask_path,
            hybrid_closed_mask_float.data(),
            hybrid_closed_mask_float.size() * sizeof(float),
            selected_debug_chunk.dst_rows,
            selected_debug_chunk.dst_cols,
            "<f4") ||
      !write_npy_2d(hybrid_filled_mask_path,
            hybrid_filled_mask_float.data(),
            hybrid_filled_mask_float.size() * sizeof(float),
            selected_debug_chunk.dst_rows,
            selected_debug_chunk.dst_cols,
            "<f4") ||
      !write_npy_2d(hybrid_component_filtered_mask_path,
            hybrid_component_filtered_mask_float.data(),
            hybrid_component_filtered_mask_float.size() * sizeof(float),
            selected_debug_chunk.dst_rows,
            selected_debug_chunk.dst_cols,
            "<f4") ||
      !write_npy_2d(grouped_mask_path,
                    grouped_mask_projected_float.data(),
                    grouped_mask_projected_float.size() * sizeof(float),
                    selected_debug_chunk.dst_rows,
                    selected_debug_chunk.dst_cols,
                    "<f4") ||
      !write_npy_2d(final_mask_path,
                    final_mask_float.data(),
                    final_mask_float.size() * sizeof(float),
                    selected_debug_chunk.dst_rows,
                    selected_debug_chunk.dst_cols,
                    "<f4") ||
      !write_npy_2d(final_mask_source_path,
                    final_mask_source_float.data(),
                    final_mask_source_float.size() * sizeof(float),
                    selected_debug_chunk.src_rows,
                    selected_debug_chunk.src_cols,
                    "<f4") ||
      !write_npy_2d(final_mask_projected_path,
                    final_mask_projected_float.data(),
                    final_mask_projected_float.size() * sizeof(float),
                    selected_debug_chunk.src_rows,
                    selected_debug_chunk.src_cols,
                    "<f4") ||
      !write_npy_2d(projected_grouped_mask_path,
                    projected_mask_float.data(),
                    projected_mask_float.size() * sizeof(float),
                    src_rows,
                    src_cols,
                    "<f4") ||
      !write_npy_2d(projected_grouped_score_path,
                    global_merged.projected_grouped_score.data(),
                    global_merged.projected_grouped_score.size() * sizeof(float),
                    src_rows,
                    src_cols,
                    "<f4") ||
      !write_npy_2d(merged_box_mask_path,
                    merged_box_mask_float.data(),
                    merged_box_mask_float.size() * sizeof(float),
                    src_rows,
                    src_cols,
                    "<f4") ||
      !write_npy_2d(final_mask_global_path,
                    stitched_final_mask_float.data(),
                    stitched_final_mask_float.size() * sizeof(float),
                    src_rows,
                    src_cols,
                    "<f4")) {
    throw std::runtime_error("failed to serialize CUDA DINO operator artifact bundle to " + output_dir.string());
  }

  write_text_file(grouped_boxes_path, detection_boxes_to_json(selected_debug_chunk.grouped_boxes));
  write_text_file(chunk_plan_path, chunk_plan_to_json(chunk_plan));
  write_text_file(projected_boxes_path, detection_boxes_to_json(global_merged.projected_boxes));
  write_text_file(merged_boxes_path, detection_boxes_to_json(global_merged.merged_boxes));

  std::ostringstream chunk_debug_summary;
  chunk_debug_summary << "{\n";
  chunk_debug_summary << "  \"artifact_contract\": \"operator_live_cuda_dino_v1\",\n";
  chunk_debug_summary << "  \"chunk_index\": " << selected_debug_chunk.chunk_index << ",\n";
  chunk_debug_summary << "  \"chunk_count\": " << chunk_count << ",\n";
  chunk_debug_summary << "  \"row_start\": " << selected_chunk.row_start << ",\n";
  chunk_debug_summary << "  \"row_stop\": " << selected_chunk.row_stop << ",\n";
  chunk_debug_summary << "  \"src_rows\": " << selected_debug_chunk.src_rows << ",\n";
  chunk_debug_summary << "  \"src_cols\": " << selected_debug_chunk.src_cols << ",\n";
  chunk_debug_summary << "  \"aligned_rows\": " << aligned_rows << ",\n";
  chunk_debug_summary << "  \"aligned_cols\": " << aligned_cols << ",\n";
  chunk_debug_summary << "  \"dst_rows\": " << selected_debug_chunk.dst_rows << ",\n";
  chunk_debug_summary << "  \"dst_cols\": " << selected_debug_chunk.dst_cols << ",\n";
  chunk_debug_summary << "  \"runtime_resized_full_chunk\": "
                      << (runtime_resized_full_chunk ? "true" : "false") << ",\n";
  chunk_debug_summary << "  \"projected_full_chunk\": "
                      << (projected_full_chunk ? "true" : "false") << ",\n";
  chunk_debug_summary << "  \"grouped_box_count\": " << selected_debug_chunk.grouped_boxes.size() << ",\n";
  chunk_debug_summary << "  \"operator_timing_ms\": {\n";
  chunk_debug_summary << "    \"total_compute\": " << timing_profile.total_compute_ms << ",\n";
  chunk_debug_summary << "    \"power_db\": " << timing_profile.power_db_ms << ",\n";
  chunk_debug_summary << "    \"frontend\": " << timing_profile.frontend_ms << ",\n";
  chunk_debug_summary << "    \"chunk_plan\": " << timing_profile.chunk_plan_ms << ",\n";
  chunk_debug_summary << "    \"chunk_pack\": " << timing_profile.chunk_pack_ms << ",\n";
  chunk_debug_summary << "    \"coherence_batch\": " << timing_profile.coherence_batch_ms << ",\n";
  chunk_debug_summary << "    \"runtime_batch\": " << timing_profile.runtime_batch_ms << ",\n";
  chunk_debug_summary << "    \"runtime_crop_align\": " << timing_profile.runtime_crop_align_ms << ",\n";
  chunk_debug_summary << "    \"runtime_resize\": " << timing_profile.runtime_resize_ms << ",\n";
  chunk_debug_summary << "    \"runtime_model_prep\": " << timing_profile.runtime_model_prep_ms << ",\n";
  chunk_debug_summary << "    \"runtime_torch_forward\": " << timing_profile.runtime_torch_forward_ms << ",\n";
  chunk_debug_summary << "    \"runtime_dino_score\": " << timing_profile.runtime_dino_score_ms << ",\n";
  chunk_debug_summary << "    \"raw_score_projection\": " << timing_profile.raw_score_projection_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_batch\": " << timing_profile.hybrid_batch_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_normalization\": " << timing_profile.hybrid_normalization_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_residual_stack\": " << timing_profile.hybrid_residual_stack_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_threshold_extract\": " << timing_profile.hybrid_threshold_extract_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_closing\": " << timing_profile.hybrid_closing_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_fill_holes\": " << timing_profile.hybrid_fill_holes_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_component_filter\": " << timing_profile.hybrid_component_filter_ms << ",\n";
  chunk_debug_summary << "    \"hybrid_output_copy\": " << timing_profile.hybrid_output_copy_ms << ",\n";
  chunk_debug_summary << "    \"debug_device_to_host\": " << timing_profile.debug_device_to_host_ms << ",\n";
  chunk_debug_summary << "    \"debug_chunk_grouping\": " << timing_profile.debug_chunk_grouping_ms << ",\n";
  chunk_debug_summary << "    \"global_merge\": " << timing_profile.global_merge_ms << ",\n";
  chunk_debug_summary << "    \"artifact_serialization\": " << timing_profile.artifact_serialization_ms << "\n";
  chunk_debug_summary << "  },\n";
  chunk_debug_summary << "  \"hybrid_thresholds\": {\n";
  chunk_debug_summary << "    \"seed_freq\": " << selected_debug_chunk.hybrid_seed_freq_threshold << ",\n";
  chunk_debug_summary << "    \"seed_res\": " << selected_debug_chunk.hybrid_seed_res_threshold << ",\n";
  chunk_debug_summary << "    \"combined\": " << selected_debug_chunk.hybrid_combined_threshold << "\n";
  chunk_debug_summary << "  },\n";
  chunk_debug_summary << "  \"corrected_resized_npy\": \"" << json_escape(corrected_resized_path.string()) << "\",\n";
  chunk_debug_summary << "  \"dino_score_raw_npy\": \"" << json_escape(raw_score_path.string()) << "\",\n";
  chunk_debug_summary << "  \"dino_score_raw_deweighted_npy\": \"" << json_escape(raw_score_deweighted_path.string()) << "\",\n";
  chunk_debug_summary << "  \"coherence_gate_npy\": \"" << json_escape(coherence_gate_path.string()) << "\",\n";
  chunk_debug_summary << "  \"combined_score_npy\": \"" << json_escape(combined_score_path.string()) << "\",\n";
  chunk_debug_summary << "  \"hybrid_keep_freq_npy\": \"" << json_escape(hybrid_keep_freq_path.string()) << "\",\n";
  chunk_debug_summary << "  \"hybrid_keep_res_npy\": \"" << json_escape(hybrid_keep_res_path.string()) << "\",\n";
  chunk_debug_summary << "  \"hybrid_seed_mask_npy\": \"" << json_escape(hybrid_seed_mask_path.string()) << "\",\n";
  chunk_debug_summary << "  \"hybrid_closed_mask_npy\": \"" << json_escape(hybrid_closed_mask_path.string()) << "\",\n";
  chunk_debug_summary << "  \"hybrid_filled_mask_npy\": \"" << json_escape(hybrid_filled_mask_path.string()) << "\",\n";
  chunk_debug_summary << "  \"hybrid_component_filtered_mask_npy\": \"" << json_escape(hybrid_component_filtered_mask_path.string()) << "\",\n";
  chunk_debug_summary << "  \"grouped_mask_npy\": \"" << json_escape(grouped_mask_path.string()) << "\",\n";
  chunk_debug_summary << "  \"grouped_boxes_json\": \"" << json_escape(grouped_boxes_path.string()) << "\",\n";
  chunk_debug_summary << "  \"final_mask_npy\": \"" << json_escape(final_mask_path.string()) << "\",\n";
  chunk_debug_summary << "  \"final_mask_source_npy\": \"" << json_escape(final_mask_source_path.string()) << "\",\n";
  chunk_debug_summary << "  \"final_mask_projected_npy\": \"" << json_escape(final_mask_projected_path.string()) << "\"\n";
  chunk_debug_summary << "}\n";
  write_text_file(chunk_debug_summary_path, chunk_debug_summary.str());

  std::ostringstream summary;
  summary << "{\n";
  summary << "  \"artifact_contract\": \"operator_live_cuda_dino_v1\",\n";
  summary << "  \"chunk_count\": " << chunk_count << ",\n";
  summary << "  \"selected_chunk_index\": " << selected_chunk_index << ",\n";
  summary << "  \"src_rows\": " << src_rows << ",\n";
  summary << "  \"src_cols\": " << src_cols << ",\n";
  summary << "  \"operator_timing_ms\": {\n";
  summary << "    \"total_compute\": " << timing_profile.total_compute_ms << ",\n";
  summary << "    \"power_db\": " << timing_profile.power_db_ms << ",\n";
  summary << "    \"frontend\": " << timing_profile.frontend_ms << ",\n";
  summary << "    \"chunk_plan\": " << timing_profile.chunk_plan_ms << ",\n";
  summary << "    \"chunk_pack\": " << timing_profile.chunk_pack_ms << ",\n";
  summary << "    \"coherence_batch\": " << timing_profile.coherence_batch_ms << ",\n";
  summary << "    \"runtime_batch\": " << timing_profile.runtime_batch_ms << ",\n";
  summary << "    \"runtime_crop_align\": " << timing_profile.runtime_crop_align_ms << ",\n";
  summary << "    \"runtime_resize\": " << timing_profile.runtime_resize_ms << ",\n";
  summary << "    \"runtime_model_prep\": " << timing_profile.runtime_model_prep_ms << ",\n";
  summary << "    \"runtime_torch_forward\": " << timing_profile.runtime_torch_forward_ms << ",\n";
  summary << "    \"runtime_dino_score\": " << timing_profile.runtime_dino_score_ms << ",\n";
  summary << "    \"raw_score_projection\": " << timing_profile.raw_score_projection_ms << ",\n";
  summary << "    \"hybrid_batch\": " << timing_profile.hybrid_batch_ms << ",\n";
  summary << "    \"hybrid_normalization\": " << timing_profile.hybrid_normalization_ms << ",\n";
  summary << "    \"hybrid_residual_stack\": " << timing_profile.hybrid_residual_stack_ms << ",\n";
  summary << "    \"hybrid_threshold_extract\": " << timing_profile.hybrid_threshold_extract_ms << ",\n";
  summary << "    \"hybrid_closing\": " << timing_profile.hybrid_closing_ms << ",\n";
  summary << "    \"hybrid_fill_holes\": " << timing_profile.hybrid_fill_holes_ms << ",\n";
  summary << "    \"hybrid_component_filter\": " << timing_profile.hybrid_component_filter_ms << ",\n";
  summary << "    \"hybrid_output_copy\": " << timing_profile.hybrid_output_copy_ms << ",\n";
  summary << "    \"debug_device_to_host\": " << timing_profile.debug_device_to_host_ms << ",\n";
  summary << "    \"debug_chunk_grouping\": " << timing_profile.debug_chunk_grouping_ms << ",\n";
  summary << "    \"global_merge\": " << timing_profile.global_merge_ms << ",\n";
  summary << "    \"artifact_serialization\": " << timing_profile.artifact_serialization_ms << "\n";
  summary << "  },\n";
  summary << "  \"projected_grouped_mask_npy\": \"" << json_escape(projected_grouped_mask_path.string()) << "\",\n";
  summary << "  \"projected_grouped_score_npy\": \"" << json_escape(projected_grouped_score_path.string()) << "\",\n";
  summary << "  \"merged_box_mask_npy\": \"" << json_escape(merged_box_mask_path.string()) << "\",\n";
  summary << "  \"final_mask_npy\": \"" << json_escape(final_mask_global_path.string()) << "\",\n";
  summary << "  \"chunk_plan_json\": \"" << json_escape(chunk_plan_path.string()) << "\",\n";
  summary << "  \"projected_boxes_json\": \"" << json_escape(projected_boxes_path.string()) << "\",\n";
  summary << "  \"merged_boxes_json\": \"" << json_escape(merged_boxes_path.string()) << "\",\n";
  summary << "  \"chunk_debug_summary_json\": \"" << json_escape(chunk_debug_summary_path.string()) << "\"\n";
  summary << "}\n";
  write_text_file(summary_path, summary.str());
}

std::vector<int> build_nearest_resize_indices(int input_size, int output_size) {
  std::vector<int> indices(static_cast<size_t>(std::max(output_size, 0)), 0);
  if (input_size <= 0 || output_size <= 0) {
    return indices;
  }
  for (int output_index = 0; output_index < output_size; ++output_index) {
    indices[static_cast<size_t>(output_index)] = std::min(input_size - 1,
                                                          static_cast<int>((static_cast<int64_t>(output_index) * static_cast<int64_t>(input_size)) /
                                                                           static_cast<int64_t>(std::max(output_size, 1))));
  }
  return indices;
}

std::vector<LinearResizeSample> build_linear_resize_samples(int input_size, int output_size) {
  std::vector<LinearResizeSample> samples(static_cast<size_t>(std::max(output_size, 0)));
  if (input_size <= 0 || output_size <= 0) {
    return samples;
  }
  const float scale = output_size > 1 ? static_cast<float>(input_size - 1) / static_cast<float>(output_size - 1) : 0.0f;
  for (int output_index = 0; output_index < output_size; ++output_index) {
    const float input_f = scale * static_cast<float>(output_index);
    const int index0 = clamp_value(static_cast<int>(std::floor(input_f)), 0, input_size - 1);
    const int index1 = clamp_value(index0 + 1, 0, input_size - 1);
    samples[static_cast<size_t>(output_index)] = LinearResizeSample{index0, index1, input_f - static_cast<float>(index0)};
  }
  return samples;
}

float sample_bilinear_resized_value(const std::vector<float>& input,
                                    int input_cols,
                                    const LinearResizeSample& row_sample,
                                    const LinearResizeSample& col_sample) {
  const float v00 = input[flat_index(input_cols, row_sample.index0, col_sample.index0)];
  const float v01 = input[flat_index(input_cols, row_sample.index0, col_sample.index1)];
  const float v10 = input[flat_index(input_cols, row_sample.index1, col_sample.index0)];
  const float v11 = input[flat_index(input_cols, row_sample.index1, col_sample.index1)];
  const float top = (1.0f - col_sample.t) * v00 + col_sample.t * v01;
  const float bottom = (1.0f - col_sample.t) * v10 + col_sample.t * v11;
  return (1.0f - row_sample.t) * top + row_sample.t * bottom;
}

DetectionBox scale_box_to_shape(const DetectionBox& box,
                                int source_rows,
                                int source_cols,
                                int target_rows,
                                int target_cols) {
  DetectionBox scaled = box;
  scaled.freq_start = static_cast<int>(std::floor(static_cast<double>(box.freq_start) * static_cast<double>(target_rows) /
                                                  static_cast<double>(std::max(source_rows, 1))));
  scaled.freq_stop = static_cast<int>(std::ceil(static_cast<double>(box.freq_stop) * static_cast<double>(target_rows) /
                                                static_cast<double>(std::max(source_rows, 1))));
  scaled.time_start = static_cast<int>(std::floor(static_cast<double>(box.time_start) * static_cast<double>(target_cols) /
                                                  static_cast<double>(std::max(source_cols, 1))));
  scaled.time_stop = static_cast<int>(std::ceil(static_cast<double>(box.time_stop) * static_cast<double>(target_cols) /
                                                static_cast<double>(std::max(source_cols, 1))));
  scaled.freq_start = clamp_value(scaled.freq_start, 0, target_rows);
  scaled.freq_stop = clamp_value(std::max(scaled.freq_stop, scaled.freq_start), 0, target_rows);
  scaled.time_start = clamp_value(scaled.time_start, 0, target_cols);
  scaled.time_stop = clamp_value(std::max(scaled.time_stop, scaled.time_start), 0, target_cols);
  return scaled;
}

DetectionBox merge_box_cluster(const std::vector<DetectionBox>& cluster) {
  DetectionBox merged;
  merged.freq_start = cluster.front().freq_start;
  merged.freq_stop = cluster.front().freq_stop;
  merged.time_start = cluster.front().time_start;
  merged.time_stop = cluster.front().time_stop;
  int weighted_area = 0;
  float weighted_score_sum = 0.0f;
  std::vector<int> source_chunk_indices;
  for (const auto& box : cluster) {
    merged.freq_start = std::min(merged.freq_start, box.freq_start);
    merged.freq_stop = std::max(merged.freq_stop, box.freq_stop);
    merged.time_start = std::min(merged.time_start, box.time_start);
    merged.time_stop = std::max(merged.time_stop, box.time_stop);
    merged.filled_area += box.filled_area;
    weighted_area += std::max(1, box.filled_area);
    weighted_score_sum += box.score_mean * static_cast<float>(std::max(1, box.filled_area));
    merged.score_peak = std::max(merged.score_peak, box.score_peak);
    source_chunk_indices.insert(source_chunk_indices.end(), box.source_chunk_indices.begin(), box.source_chunk_indices.end());
  }
  const int bbox_area = std::max(1, (merged.freq_stop - merged.freq_start) * (merged.time_stop - merged.time_start));
  merged.density = static_cast<float>(merged.filled_area) / static_cast<float>(bbox_area);
  merged.bbox_density = merged.density;
  merged.envelope_density = merged.density;
  merged.score_mean = weighted_area > 0 ? weighted_score_sum / static_cast<float>(weighted_area) : 0.0f;
  std::sort(source_chunk_indices.begin(), source_chunk_indices.end());
  source_chunk_indices.erase(std::unique(source_chunk_indices.begin(), source_chunk_indices.end()), source_chunk_indices.end());
  merged.source_chunk_indices = std::move(source_chunk_indices);
  return merged;
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

std::vector<uint8_t> binary_filter_rows(const std::vector<uint8_t>& mask,
                                        int rows,
                                        int cols,
                                        int radius,
                                        bool require_all) {
  std::vector<uint8_t> output(mask.size(), 0);
  if (rows <= 0 || cols <= 0) {
    return output;
  }
  const int window = 2 * std::max(0, radius) + 1;
  std::vector<int> prefix(static_cast<size_t>(cols + 1), 0);
  for (int row = 0; row < rows; ++row) {
    prefix[0] = 0;
    for (int col = 0; col < cols; ++col) {
      prefix[static_cast<size_t>(col + 1)] = prefix[static_cast<size_t>(col)] +
                                             static_cast<int>(mask[flat_index(cols, row, col)] != 0);
    }
    for (int col = 0; col < cols; ++col) {
      const int left = std::max(0, col - radius);
      const int right = std::min(cols - 1, col + radius);
      int active = prefix[static_cast<size_t>(right + 1)] - prefix[static_cast<size_t>(left)];
      if (col - radius < 0 && mask[flat_index(cols, row, 0)]) {
        active += -(col - radius);
      }
      if (col + radius >= cols && mask[flat_index(cols, row, cols - 1)]) {
        active += col + radius - (cols - 1);
      }
      output[flat_index(cols, row, col)] = require_all ? static_cast<uint8_t>(active == window) : static_cast<uint8_t>(active > 0);
    }
  }
  return output;
}

std::vector<uint8_t> binary_filter_cols(const std::vector<uint8_t>& mask,
                                        int rows,
                                        int cols,
                                        int radius,
                                        bool require_all) {
  std::vector<uint8_t> output(mask.size(), 0);
  if (rows <= 0 || cols <= 0) {
    return output;
  }
  const int window = 2 * std::max(0, radius) + 1;
  std::vector<int> prefix(static_cast<size_t>(rows + 1), 0);
  for (int col = 0; col < cols; ++col) {
    prefix[0] = 0;
    for (int row = 0; row < rows; ++row) {
      prefix[static_cast<size_t>(row + 1)] = prefix[static_cast<size_t>(row)] +
                                             static_cast<int>(mask[flat_index(cols, row, col)] != 0);
    }
    for (int row = 0; row < rows; ++row) {
      const int top = std::max(0, row - radius);
      const int bottom = std::min(rows - 1, row + radius);
      int active = prefix[static_cast<size_t>(bottom + 1)] - prefix[static_cast<size_t>(top)];
      if (row - radius < 0 && mask[flat_index(cols, 0, col)]) {
        active += -(row - radius);
      }
      if (row + radius >= rows && mask[flat_index(cols, rows - 1, col)]) {
        active += row + radius - (rows - 1);
      }
      output[flat_index(cols, row, col)] = require_all ? static_cast<uint8_t>(active == window) : static_cast<uint8_t>(active > 0);
    }
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
  return binary_filter_cols(binary_filter_rows(mask, rows, cols, col_radius, false), rows, cols, row_radius, false);
}

std::vector<uint8_t> binary_erode_rect(const std::vector<uint8_t>& mask,
                                       int rows,
                                       int cols,
                                       int kernel_rows,
                                       int kernel_cols) {
  const int row_radius = std::max(0, kernel_rows / 2);
  const int col_radius = std::max(0, kernel_cols / 2);
  return binary_filter_cols(binary_filter_rows(mask, rows, cols, col_radius, true), rows, cols, row_radius, true);
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
      if (output[flat_index(cols, row, col)]) {
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

std::vector<DetectionBox> group_boxes_fast_only(const std::vector<uint8_t>& mask,
                                                const std::vector<float>& score_map,
                                                const std::vector<uint8_t>& valid_mask,
                                                int rows,
                                                int cols,
                                                int bridge_freq_px,
                                                int bridge_time_px,
                                                int min_component_size,
                                                int min_freq_span_px,
                                                int min_time_span_px,
                                                float min_density) {
  std::vector<DetectionBox> boxes;
  if (rows <= 0 || cols <= 0 || mask.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
    return boxes;
  }

  std::vector<uint8_t> active = mask;
  std::vector<float> active_scores;
  active_scores.reserve(score_map.size());
  for (size_t index = 0; index < active.size(); ++index) {
    if (index < valid_mask.size() && !valid_mask[index]) {
      active[index] = 0;
      continue;
    }
    if (active[index] && index < score_map.size()) {
      active_scores.push_back(score_map[index]);
    }
  }
  if (std::none_of(active.begin(), active.end(), [](uint8_t value) { return value != 0; })) {
    return boxes;
  }

  struct FastComponentStats {
    int min_row = std::numeric_limits<int>::max();
    int max_row = -1;
    int min_col = std::numeric_limits<int>::max();
    int max_col = -1;
    int filled_area = 0;
    float score_sum = 0.0f;
    float score_peak = 0.0f;
  };

  const auto labelled = label_components(active, rows, cols);
  std::vector<FastComponentStats> stats(labelled.sizes.size());
  for (size_t flat = 0; flat < labelled.labels.size(); ++flat) {
    const int label = labelled.labels[flat];
    if (label <= 0) {
      continue;
    }
    auto& component = stats[static_cast<size_t>(label - 1)];
    const int row = static_cast<int>(flat / static_cast<size_t>(cols));
    const int col = static_cast<int>(flat % static_cast<size_t>(cols));
    component.min_row = std::min(component.min_row, row);
    component.max_row = std::max(component.max_row, row);
    component.min_col = std::min(component.min_col, col);
    component.max_col = std::max(component.max_col, col);
    ++component.filled_area;
    if (flat < score_map.size()) {
      const float score = score_map[flat];
      component.score_sum += score;
      component.score_peak = std::max(component.score_peak, score);
    }
  }

  const float peak_score_floor = quantile_from_values(active_scores, 0.50, 0.0f);
  std::vector<DetectionBox> component_boxes;
  component_boxes.reserve(stats.size());
  for (const auto& component : stats) {
    if (component.filled_area <= 0 || component.max_row < component.min_row || component.max_col < component.min_col) {
      continue;
    }
    DetectionBox box;
    box.freq_start = component.min_row;
    box.freq_stop = component.max_row + 1;
    box.time_start = component.min_col;
    box.time_stop = component.max_col + 1;
    box.filled_area = component.filled_area;
    const int freq_span = box.freq_stop - box.freq_start;
    const int time_span = box.time_stop - box.time_start;
    const int bbox_area = std::max(1, freq_span * time_span);
    const float density = static_cast<float>(component.filled_area) / static_cast<float>(bbox_area);
    box.density = density;
    box.bbox_density = density;
    box.envelope_density = density;
    box.score_mean = component.filled_area > 0 ? component.score_sum / static_cast<float>(component.filled_area) : 0.0f;
    box.score_peak = component.score_peak;
    component_boxes.push_back(std::move(box));
  }
  if (component_boxes.empty()) {
    return boxes;
  }

  struct LocalBoxEvent {
    int freq_start = 0;
    int freq_stop = 0;
    size_t index = 0;
  };

  struct LocalBoxDisjointSet {
    std::vector<size_t> parent;
    std::vector<uint16_t> rank;

    explicit LocalBoxDisjointSet(size_t count) : parent(count), rank(count, 0) {
      for (size_t index = 0; index < count; ++index) {
        parent[index] = index;
      }
    }

    size_t find(size_t index) {
      size_t root = index;
      while (parent[root] != root) {
        root = parent[root];
      }
      while (parent[index] != index) {
        const size_t next = parent[index];
        parent[index] = root;
        index = next;
      }
      return root;
    }

    void unite(size_t lhs, size_t rhs) {
      lhs = find(lhs);
      rhs = find(rhs);
      if (lhs == rhs) {
        return;
      }
      if (rank[lhs] < rank[rhs]) {
        std::swap(lhs, rhs);
      }
      parent[rhs] = lhs;
      if (rank[lhs] == rank[rhs]) {
        ++rank[lhs];
      }
    }
  };

  auto axis_gap = [](int start_a, int stop_a, int start_b, int stop_b) {
    if (stop_a < start_b) {
      return start_b - stop_a;
    }
    if (stop_b < start_a) {
      return start_a - stop_b;
    }
    return 0;
  };

  auto boxes_should_merge_fast = [&](const DetectionBox& lhs, const DetectionBox& rhs) {
    const int freq_gap = axis_gap(lhs.freq_start, lhs.freq_stop, rhs.freq_start, rhs.freq_stop);
    if (freq_gap > std::max(0, bridge_freq_px)) {
      return false;
    }
    const int time_gap = axis_gap(lhs.time_start, lhs.time_stop, rhs.time_start, rhs.time_stop);
    return time_gap <= std::max(0, bridge_time_px);
  };

  std::vector<LocalBoxEvent> events(component_boxes.size());
  for (size_t index = 0; index < component_boxes.size(); ++index) {
    events[index] = LocalBoxEvent{component_boxes[index].freq_start, component_boxes[index].freq_stop, index};
  }
  std::sort(events.begin(), events.end(), [](const LocalBoxEvent& lhs, const LocalBoxEvent& rhs) {
    if (lhs.freq_start != rhs.freq_start) {
      return lhs.freq_start < rhs.freq_start;
    }
    if (lhs.freq_stop != rhs.freq_stop) {
      return lhs.freq_stop < rhs.freq_stop;
    }
    return lhs.index < rhs.index;
  });

  LocalBoxDisjointSet sets(component_boxes.size());
  std::vector<size_t> active_indices;
  active_indices.reserve(component_boxes.size());
  for (const auto& event : events) {
    const auto& current_box = component_boxes[event.index];
    size_t write_index = 0;
    for (size_t read_index = 0; read_index < active_indices.size(); ++read_index) {
      const size_t active_index = active_indices[read_index];
      if (component_boxes[active_index].freq_stop + std::max(0, bridge_freq_px) >= current_box.freq_start) {
        active_indices[write_index++] = active_index;
      }
    }
    active_indices.resize(write_index);
    for (const size_t active_index : active_indices) {
      if (boxes_should_merge_fast(component_boxes[active_index], current_box)) {
        sets.unite(active_index, event.index);
      }
    }
    active_indices.push_back(event.index);
  }

  std::vector<std::vector<DetectionBox>> merged_clusters(component_boxes.size());
  for (size_t index = 0; index < component_boxes.size(); ++index) {
    merged_clusters[sets.find(index)].push_back(component_boxes[index]);
  }

  boxes.reserve(merged_clusters.size());
  for (auto& cluster : merged_clusters) {
    if (cluster.empty()) {
      continue;
    }
    DetectionBox box = merge_box_cluster(cluster);
    if (box.filled_area < std::max(1, min_component_size) ||
        (box.freq_stop - box.freq_start) < std::max(1, min_freq_span_px) ||
        (box.time_stop - box.time_start) < std::max(1, min_time_span_px) ||
        box.density < min_density ||
        box.score_peak < peak_score_floor) {
      continue;
    }
    boxes.push_back(std::move(box));
  }
  return boxes;
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
    if (!valid_mask[index]) {
      result.seed_mask[index] = 0;
    }
  }
  if (std::none_of(result.seed_mask.begin(), result.seed_mask.end(), [](uint8_t value) { return value != 0; })) {
    result.bridged_mask = result.seed_mask;
    result.grouped_mask = result.seed_mask;
    return result;
  }

  struct ComponentStats {
    int min_row = std::numeric_limits<int>::max();
    int max_row = -1;
    int min_col = std::numeric_limits<int>::max();
    int max_col = -1;
    int filled_area = 0;
    float score_sum = 0.0f;
    float score_peak = 0.0f;
    bool keep = false;
    std::vector<int> col_min_row;
    std::vector<int> col_max_row;
  };

  auto build_component_stats = [&](const ComponentLabelling& labelled_state) {
    std::vector<ComponentStats> stats(labelled_state.sizes.size());
    for (size_t flat = 0; flat < labelled_state.labels.size(); ++flat) {
      const int label = labelled_state.labels[flat];
      if (label <= 0) {
        continue;
      }
      auto& component = stats[static_cast<size_t>(label - 1)];
      const int row = static_cast<int>(flat / static_cast<size_t>(cols));
      const int col = static_cast<int>(flat % static_cast<size_t>(cols));
      component.min_row = std::min(component.min_row, row);
      component.max_row = std::max(component.max_row, row);
      component.min_col = std::min(component.min_col, col);
      component.max_col = std::max(component.max_col, col);
      ++component.filled_area;
      if (flat < score_map.size()) {
        const float score = score_map[flat];
        component.score_sum += score;
        component.score_peak = std::max(component.score_peak, score);
      }
    }
    return stats;
  };

  if (!filter_detection_mask) {
    result.bridged_mask = result.seed_mask;
    const auto labelled = label_components(result.bridged_mask, rows, cols);
    result.grouped_mask = result.seed_mask;
    const auto component_stats = build_component_stats(labelled);
    for (const auto& component : component_stats) {
      if (component.filled_area <= 0 || component.max_row < component.min_row || component.max_col < component.min_col) {
        continue;
      }
      DetectionBox box;
      box.freq_start = component.min_row;
      box.freq_stop = component.max_row + 1;
      box.time_start = component.min_col;
      box.time_stop = component.max_col + 1;
      box.filled_area = component.filled_area;
      const int bbox_area = std::max(1, (box.freq_stop - box.freq_start) * (box.time_stop - box.time_start));
      box.bbox_density = static_cast<float>(component.filled_area) / static_cast<float>(bbox_area);
      box.envelope_density = box.bbox_density;
      box.density = box.bbox_density;
      box.score_mean = component.filled_area > 0 ? component.score_sum / static_cast<float>(component.filled_area) : 0.0f;
      box.score_peak = component.score_peak;
      result.boxes.push_back(std::move(box));
    }
    return result;
  }

  result.bridged_mask = result.seed_mask;
  if (bridge_freq_px > 1 || bridge_time_px > 1) {
    result.bridged_mask = binary_closing_rect(result.bridged_mask, rows, cols, std::max(1, bridge_freq_px), std::max(1, bridge_time_px));
  }
  result.bridged_mask = fill_nearly_continuous_time_gaps(result.bridged_mask, rows, cols, bridge_time_px, time_continuity_ratio);

  const auto labelled = label_components(result.bridged_mask, rows, cols);
  auto component_stats = build_component_stats(labelled);
  std::vector<float> active_scores;
  active_scores.reserve(score_map.size());
  for (size_t index = 0; index < score_map.size() && index < result.seed_mask.size(); ++index) {
    if (result.seed_mask[index]) {
      active_scores.push_back(score_map[index]);
    }
  }
  result.peak_score_floor = quantile_from_values(active_scores, 0.50, 0.0f);

  for (auto& component : component_stats) {
    if (component.filled_area <= 0 || component.max_row < component.min_row || component.max_col < component.min_col) {
      continue;
    }
    const int local_cols = component.max_col - component.min_col + 1;
    component.col_min_row.assign(static_cast<size_t>(local_cols), rows);
    component.col_max_row.assign(static_cast<size_t>(local_cols), -1);
  }

  for (size_t flat = 0; flat < labelled.labels.size(); ++flat) {
    const int label = labelled.labels[flat];
    if (label <= 0) {
      continue;
    }
    auto& component = component_stats[static_cast<size_t>(label - 1)];
    if (component.filled_area <= 0 || component.max_row < component.min_row || component.max_col < component.min_col) {
      continue;
    }
    const int row = static_cast<int>(flat / static_cast<size_t>(cols));
    const int col = static_cast<int>(flat % static_cast<size_t>(cols));
    const size_t local_col = static_cast<size_t>(col - component.min_col);
    component.col_min_row[local_col] = std::min(component.col_min_row[local_col], row);
    component.col_max_row[local_col] = std::max(component.col_max_row[local_col], row);
  }

  result.grouped_mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  for (size_t label_index = 0; label_index < component_stats.size(); ++label_index) {
    auto& component = component_stats[label_index];
    if (component.filled_area <= 0 || component.max_row < component.min_row || component.max_col < component.min_col) {
      continue;
    }

    const int freq_start = component.min_row;
    const int freq_stop = component.max_row + 1;
    const int time_start = component.min_col;
    const int time_stop = component.max_col + 1;
    const int freq_span = freq_stop - freq_start;
    const int time_span = time_stop - time_start;
    const int bbox_area = std::max(1, freq_span * time_span);
    int envelope_area = 0;
    for (size_t local_col = 0; local_col < component.col_min_row.size(); ++local_col) {
      if (component.col_max_row[local_col] >= component.col_min_row[local_col]) {
        envelope_area += component.col_max_row[local_col] - component.col_min_row[local_col] + 1;
      }
    }
    envelope_area = std::max(1, envelope_area);
    const float bbox_density = static_cast<float>(component.filled_area) / static_cast<float>(bbox_area);
    const float envelope_density = static_cast<float>(component.filled_area) / static_cast<float>(envelope_area);
    const float density = envelope_density;
    const float score_peak = component.score_peak;
    const float score_mean = component.filled_area > 0 ? component.score_sum / static_cast<float>(component.filled_area) : 0.0f;

    component.keep = component.filled_area >= std::max(1, min_component_size) &&
                     freq_span >= std::max(1, min_freq_span_px) &&
                     time_span >= std::max(1, min_time_span_px) &&
                     density >= min_density &&
                     score_peak >= result.peak_score_floor;
    if (!component.keep) {
      continue;
    }

    DetectionBox box;
    box.freq_start = freq_start;
    box.freq_stop = freq_stop;
    box.time_start = time_start;
    box.time_stop = time_stop;
    box.filled_area = component.filled_area;
    box.density = density;
    box.bbox_density = bbox_density;
    box.envelope_density = envelope_density;
    box.score_mean = score_mean;
    box.score_peak = score_peak;
    result.boxes.push_back(std::move(box));
  }

  for (size_t flat = 0; flat < labelled.labels.size(); ++flat) {
    const int label = labelled.labels[flat];
    if (label > 0 && component_stats[static_cast<size_t>(label - 1)].keep) {
      result.grouped_mask[flat] = 1;
    }
  }
  return result;
}

bool boxes_overlap(const DetectionBox& box_a, const DetectionBox& box_b) {
  return box_a.freq_start < box_b.freq_stop && box_b.freq_start < box_a.freq_stop &&
         box_a.time_start < box_b.time_stop && box_b.time_start < box_a.time_stop;
}

bool boxes_share_source_chunk(const DetectionBox& box_a, const DetectionBox& box_b) {
  if (box_a.source_chunk_indices.empty() || box_b.source_chunk_indices.empty()) {
    return false;
  }
  for (int lhs_chunk : box_a.source_chunk_indices) {
    for (int rhs_chunk : box_b.source_chunk_indices) {
      if (lhs_chunk == rhs_chunk) {
        return true;
      }
    }
  }
  return false;
}

bool boxes_should_merge(const DetectionBox& box_a, const DetectionBox& box_b) {
  return boxes_overlap(box_a, box_b) && !boxes_share_source_chunk(box_a, box_b);
}

void rasterize_box_mask(std::vector<uint8_t>& mask,
                        int rows,
                        int cols,
                        const DetectionBox& box,
                        const std::vector<uint8_t>* valid_row_mask = nullptr) {
  const int freq_start = clamp_value(box.freq_start, 0, rows);
  const int freq_stop = clamp_value(box.freq_stop, freq_start, rows);
  const int time_start = clamp_value(box.time_start, 0, cols);
  const int time_stop = clamp_value(box.time_stop, time_start, cols);
  if (freq_stop <= freq_start || time_stop <= time_start) {
    return;
  }
  for (int row = freq_start; row < freq_stop; ++row) {
    if (valid_row_mask != nullptr && row < static_cast<int>(valid_row_mask->size()) && !(*valid_row_mask)[static_cast<size_t>(row)]) {
      continue;
    }
    auto* row_begin = mask.data() + static_cast<std::ptrdiff_t>(flat_index(cols, row, time_start));
    std::fill(row_begin, row_begin + static_cast<std::ptrdiff_t>(time_stop - time_start), static_cast<uint8_t>(1));
  }
}

std::vector<uint8_t> boxes_to_mask(const std::vector<DetectionBox>& boxes,
                                   int rows,
                                   int cols,
                                   const std::vector<uint8_t>& valid_row_mask) {
  std::vector<uint8_t> mask(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  for (const auto& box : boxes) {
    rasterize_box_mask(mask, rows, cols, box, &valid_row_mask);
  }
  return mask;
}

std::vector<DetectionBox> merge_projected_boxes(const std::vector<DetectionBox>& projected_boxes) {
  if (projected_boxes.empty()) {
    return {};
  }

  struct BoxSweepEvent {
    int freq_start = 0;
    int freq_stop = 0;
    size_t index = 0;
  };

  struct BoxDisjointSet {
    std::vector<size_t> parent;
    std::vector<uint16_t> rank;

    explicit BoxDisjointSet(size_t count) : parent(count), rank(count, 0) {
      for (size_t index = 0; index < count; ++index) {
        parent[index] = index;
      }
    }

    size_t find(size_t index) {
      size_t root = index;
      while (parent[root] != root) {
        root = parent[root];
      }
      while (parent[index] != index) {
        const size_t next = parent[index];
        parent[index] = root;
        index = next;
      }
      return root;
    }

    void unite(size_t lhs, size_t rhs) {
      lhs = find(lhs);
      rhs = find(rhs);
      if (lhs == rhs) {
        return;
      }
      if (rank[lhs] < rank[rhs]) {
        std::swap(lhs, rhs);
      }
      parent[rhs] = lhs;
      if (rank[lhs] == rank[rhs]) {
        ++rank[lhs];
      }
    }
  };

  auto build_merged_boxes = [&](BoxDisjointSet& sets) {
    std::vector<std::vector<DetectionBox>> clusters(projected_boxes.size());
    for (size_t index = 0; index < projected_boxes.size(); ++index) {
      clusters[sets.find(index)].push_back(projected_boxes[index]);
    }

    std::vector<DetectionBox> merged_boxes;
    merged_boxes.reserve(projected_boxes.size());
    for (auto& cluster : clusters) {
      if (!cluster.empty()) {
        merged_boxes.push_back(merge_box_cluster(cluster));
      }
    }
    return merged_boxes;
  };

  std::array<int, 2> source_chunks{-1, -1};
  size_t unique_source_chunk_count = 0;
  std::vector<uint8_t> source_group(projected_boxes.size(), 0);
  bool two_chunk_fast_path_supported = true;
  for (size_t index = 0; index < projected_boxes.size(); ++index) {
    const auto& box = projected_boxes[index];
    if (box.source_chunk_indices.size() != 1) {
      two_chunk_fast_path_supported = false;
      break;
    }
    const int chunk_index = box.source_chunk_indices.front();
    size_t group_index = 0;
    for (; group_index < unique_source_chunk_count; ++group_index) {
      if (source_chunks[group_index] == chunk_index) {
        break;
      }
    }
    if (group_index == unique_source_chunk_count) {
      if (unique_source_chunk_count >= source_chunks.size()) {
        two_chunk_fast_path_supported = false;
        break;
      }
      source_chunks[unique_source_chunk_count++] = chunk_index;
    }
    source_group[index] = static_cast<uint8_t>(group_index);
  }

  std::vector<BoxSweepEvent> sweep_boxes(projected_boxes.size());
  for (size_t index = 0; index < projected_boxes.size(); ++index) {
    sweep_boxes[index] = BoxSweepEvent{projected_boxes[index].freq_start, projected_boxes[index].freq_stop, index};
  }
  std::sort(sweep_boxes.begin(), sweep_boxes.end(), [](const BoxSweepEvent& lhs, const BoxSweepEvent& rhs) {
    if (lhs.freq_start != rhs.freq_start) {
      return lhs.freq_start < rhs.freq_start;
    }
    if (lhs.freq_stop != rhs.freq_stop) {
      return lhs.freq_stop < rhs.freq_stop;
    }
    return lhs.index < rhs.index;
  });

  if (two_chunk_fast_path_supported && unique_source_chunk_count == 2) {
    BoxDisjointSet sets(projected_boxes.size());
    std::array<std::vector<size_t>, 2> active_indices_by_group;
    active_indices_by_group[0].reserve(projected_boxes.size());
    active_indices_by_group[1].reserve(projected_boxes.size());

    for (const auto& current : sweep_boxes) {
      const auto& current_box = projected_boxes[current.index];
      const uint8_t current_group = source_group[current.index];
      for (auto& active_indices : active_indices_by_group) {
        size_t write_index = 0;
        for (size_t read_index = 0; read_index < active_indices.size(); ++read_index) {
          const size_t active_index = active_indices[read_index];
          if (projected_boxes[active_index].freq_stop > current_box.freq_start) {
            active_indices[write_index++] = active_index;
          }
        }
        active_indices.resize(write_index);
      }

      auto& opposite_active = active_indices_by_group[1 - current_group];
      for (const size_t active_index : opposite_active) {
        if (boxes_overlap(projected_boxes[active_index], current_box)) {
          sets.unite(active_index, current.index);
        }
      }
      active_indices_by_group[current_group].push_back(current.index);
    }
    return build_merged_boxes(sets);
  }

  BoxDisjointSet sets(projected_boxes.size());
  std::vector<size_t> active_indices;
  active_indices.reserve(projected_boxes.size());
  for (const auto& current : sweep_boxes) {
    const auto& current_box = projected_boxes[current.index];
    size_t write_index = 0;
    for (size_t read_index = 0; read_index < active_indices.size(); ++read_index) {
      const size_t active_index = active_indices[read_index];
      if (projected_boxes[active_index].freq_stop > current_box.freq_start) {
        active_indices[write_index++] = active_index;
      }
    }
    active_indices.resize(write_index);
    for (const size_t active_index : active_indices) {
      if (boxes_should_merge(projected_boxes[active_index], current_box)) {
        sets.unite(active_index, current.index);
      }
    }
    active_indices.push_back(current.index);
  }
  return build_merged_boxes(sets);
}

std::vector<ChunkOwnershipRange> compute_chunk_row_ownership_ranges(const std::vector<DebugChunkResult>& chunk_results) {
  std::vector<ChunkOwnershipRange> ownership(chunk_results.size());
  for (size_t index = 0; index < chunk_results.size(); ++index) {
    ownership[index].row_start = chunk_results[index].row_start;
    ownership[index].row_stop = chunk_results[index].row_stop;
  }
  for (size_t index = 0; index + 1 < chunk_results.size(); ++index) {
    const auto& current = chunk_results[index];
    const auto& next = chunk_results[index + 1];
    if (current.row_stop <= next.row_start) {
      continue;
    }
    const int split_row = clamp_value((current.row_stop + next.row_start) / 2,
                                      next.row_start,
                                      current.row_stop);
    ownership[index].row_stop = std::min(ownership[index].row_stop, split_row);
    ownership[index + 1].row_start = std::max(ownership[index + 1].row_start, split_row);
  }
  for (size_t index = 0; index < ownership.size(); ++index) {
    ownership[index].row_start = clamp_value(ownership[index].row_start,
                                             chunk_results[index].row_start,
                                             chunk_results[index].row_stop);
    ownership[index].row_stop = clamp_value(ownership[index].row_stop,
                                            ownership[index].row_start,
                                            chunk_results[index].row_stop);
  }
  return ownership;
}

GlobalMergedResult build_global_merged_result(const std::vector<DebugChunkResult>& chunk_results,
                                              bool filter_detection_mask,
                                              int bridge_freq_px,
                                              int bridge_time_px,
                                              int min_component_size,
                                              int min_freq_span_px,
                                              int min_time_span_px,
                                              float min_density,
                                              float time_continuity_ratio,
                                              int global_rows,
                                              int global_cols,
                                              const std::vector<uint8_t>& source_valid_row_mask) {
  GlobalMergedResult result;
  const auto ownership_ranges = compute_chunk_row_ownership_ranges(chunk_results);
  result.projected_grouped_mask.assign(static_cast<size_t>(global_rows) * static_cast<size_t>(global_cols), 0);
  result.projected_grouped_score.assign(static_cast<size_t>(global_rows) * static_cast<size_t>(global_cols), 0.0f);
  result.stitched_final_mask.assign(static_cast<size_t>(global_rows) * static_cast<size_t>(global_cols), 0);
  int deferred_group_cols = 0;
  for (const auto& chunk : chunk_results) {
    if (chunk.grouped_boxes.empty() && !chunk.final_mask.empty() && !chunk.combined_score.empty() && chunk.dst_rows > 0 && chunk.dst_cols > 0) {
      deferred_group_cols = std::max(deferred_group_cols, chunk.dst_cols);
    }
  }
  std::vector<uint8_t> deferred_group_mask(static_cast<size_t>(global_rows) * static_cast<size_t>(std::max(1, deferred_group_cols)), 0);
  std::vector<float> deferred_group_score(static_cast<size_t>(global_rows) * static_cast<size_t>(std::max(1, deferred_group_cols)), 0.0f);
  bool has_deferred_group_inputs = false;
  auto append_projected_box = [&](const DetectionBox& box) {
    result.projected_boxes.push_back(box);
  };

  for (size_t chunk_index = 0; chunk_index < chunk_results.size(); ++chunk_index) {
    const auto& chunk = chunk_results[chunk_index];
    const int projected_rows = clamp_value(chunk.row_stop - chunk.row_start, 0, global_rows);
    const int owned_row_start = clamp_value(ownership_ranges[chunk_index].row_start, chunk.row_start, chunk.row_stop);
    const int owned_row_stop = clamp_value(ownership_ranges[chunk_index].row_stop, owned_row_start, chunk.row_stop);
    const int owned_projected_row_start = owned_row_start - chunk.row_start;
    const int owned_projected_row_stop = owned_row_stop - chunk.row_start;
    const auto row_nearest_global = build_nearest_resize_indices(chunk.dst_rows, projected_rows);

    if (!chunk.final_mask.empty() && chunk.dst_rows > 0 && chunk.dst_cols > 0 && projected_rows > 0 &&
        owned_projected_row_stop > owned_projected_row_start) {
      const auto col_nearest_global = build_nearest_resize_indices(chunk.dst_cols, global_cols);
      for (int local_row = owned_projected_row_start; local_row < owned_projected_row_stop; ++local_row) {
        const int global_row = chunk.row_start + local_row;
        if (global_row < 0 || global_row >= global_rows) {
          continue;
        }
        const int src_row_nearest = row_nearest_global[static_cast<size_t>(local_row)];
        for (int col = 0; col < global_cols; ++col) {
          const size_t global_flat = flat_index(global_cols, global_row, col);
          const int src_col_nearest = col_nearest_global[static_cast<size_t>(col)];
          if (chunk.final_mask[flat_index(chunk.dst_cols, src_row_nearest, src_col_nearest)]) {
            result.stitched_final_mask[global_flat] = 1;
          }
        }
      }
    }

    if (!chunk.grouped_mask_source.empty() && !chunk.combined_score.empty() && chunk.src_rows > 0 && chunk.src_cols > 0 &&
        chunk.dst_rows > 0 && chunk.dst_cols > 0 && projected_rows > 0 && owned_projected_row_stop > owned_projected_row_start) {
      const auto row_linear_global = build_linear_resize_samples(chunk.dst_rows, projected_rows);
      const auto col_linear_global = build_linear_resize_samples(chunk.dst_cols, global_cols);
      for (int local_row = owned_projected_row_start; local_row < owned_projected_row_stop; ++local_row) {
        const int global_row = chunk.row_start + local_row;
        if (global_row < 0 || global_row >= global_rows) {
          continue;
        }
        const auto& row_sample = row_linear_global[static_cast<size_t>(local_row)];
        for (int col = 0; col < global_cols; ++col) {
          const size_t global_flat = flat_index(global_cols, global_row, col);
          if (chunk.grouped_mask_source[flat_index(chunk.src_cols, local_row, col)]) {
            result.projected_grouped_mask[global_flat] = 1;
            result.projected_grouped_score[global_flat] = std::max(
                result.projected_grouped_score[global_flat],
                sample_bilinear_resized_value(chunk.combined_score,
                                              chunk.dst_cols,
                                              row_sample,
                                              col_linear_global[static_cast<size_t>(col)]));
          }
        }
      }
    }

    if (deferred_group_cols > 0 && chunk.grouped_boxes.empty() && !chunk.final_mask.empty() && !chunk.combined_score.empty() &&
        chunk.dst_rows > 0 && chunk.dst_cols > 0 && projected_rows > 0 && owned_projected_row_stop > owned_projected_row_start) {
      has_deferred_group_inputs = true;
      const auto col_nearest_deferred = build_nearest_resize_indices(chunk.dst_cols, deferred_group_cols);
      const auto row_linear_deferred = build_linear_resize_samples(chunk.dst_rows, projected_rows);
      const auto col_linear_deferred = build_linear_resize_samples(chunk.dst_cols, deferred_group_cols);
      for (int local_row = owned_projected_row_start; local_row < owned_projected_row_stop; ++local_row) {
        const int global_row = chunk.row_start + local_row;
        if (global_row < 0 || global_row >= global_rows) {
          continue;
        }
        const int src_row_nearest = row_nearest_global[static_cast<size_t>(local_row)];
        const auto& row_sample = row_linear_deferred[static_cast<size_t>(local_row)];
        for (int col = 0; col < deferred_group_cols; ++col) {
          const size_t global_flat = flat_index(deferred_group_cols, global_row, col);
          const int src_col_nearest = col_nearest_deferred[static_cast<size_t>(col)];
          if (chunk.final_mask[flat_index(chunk.dst_cols, src_row_nearest, src_col_nearest)]) {
            deferred_group_mask[global_flat] = 1;
            deferred_group_score[global_flat] = std::max(
                deferred_group_score[global_flat],
                sample_bilinear_resized_value(chunk.combined_score,
                                              chunk.dst_cols,
                                              row_sample,
                                              col_linear_deferred[static_cast<size_t>(col)]));
          }
        }
      }
      continue;
    }

    for (const auto& box : chunk.grouped_boxes) {
      DetectionBox projected_box = scale_box_to_shape(box,
                                                      chunk.dst_rows,
                                                      chunk.dst_cols,
                                                      projected_rows,
                                                      global_cols);
      projected_box.freq_start = chunk.row_start + projected_box.freq_start;
      projected_box.freq_stop = chunk.row_start + projected_box.freq_stop;
      projected_box.source_chunk_indices = {chunk.chunk_index};
      append_projected_box(projected_box);
    }
  }

  for (int row = 0; row < global_rows; ++row) {
    const bool row_valid = row < static_cast<int>(source_valid_row_mask.size()) ? static_cast<bool>(source_valid_row_mask[static_cast<size_t>(row)]) : true;
    if (row_valid) {
      continue;
    }
    auto* grouped_mask_row = result.projected_grouped_mask.data() + static_cast<std::ptrdiff_t>(flat_index(global_cols, row, 0));
    std::fill(grouped_mask_row, grouped_mask_row + global_cols, static_cast<uint8_t>(0));
    auto* grouped_score_row = result.projected_grouped_score.data() + static_cast<std::ptrdiff_t>(flat_index(global_cols, row, 0));
    std::fill(grouped_score_row, grouped_score_row + global_cols, 0.0f);
    auto* deferred_mask_row = deferred_group_mask.data() + static_cast<std::ptrdiff_t>(flat_index(std::max(1, deferred_group_cols), row, 0));
    std::fill(deferred_mask_row, deferred_mask_row + std::max(1, deferred_group_cols), static_cast<uint8_t>(0));
    auto* deferred_score_row = deferred_group_score.data() + static_cast<std::ptrdiff_t>(flat_index(std::max(1, deferred_group_cols), row, 0));
    std::fill(deferred_score_row, deferred_score_row + std::max(1, deferred_group_cols), 0.0f);
  }
  for (int row = 0; row < global_rows; ++row) {
    const bool row_valid = row < static_cast<int>(source_valid_row_mask.size()) ? static_cast<bool>(source_valid_row_mask[static_cast<size_t>(row)]) : true;
    if (row_valid) {
      continue;
    }
    for (int col = 0; col < global_cols; ++col) {
      result.stitched_final_mask[flat_index(global_cols, row, col)] = 0;
    }
  }

  if (has_deferred_group_inputs) {
    const auto global_valid_mask = expand_row_valid_mask(source_valid_row_mask, std::max(1, deferred_group_cols));
    const auto deferred_grouping = group_mask_regions(deferred_group_mask,
                                                      deferred_group_score,
                                                      global_valid_mask,
                                                      global_rows,
                                                      deferred_group_cols,
                                                      filter_detection_mask,
                                                      bridge_freq_px,
                                                      bridge_time_px,
                                                      min_component_size,
                                                      min_freq_span_px,
                                                      min_time_span_px,
                                                      min_density,
                                                      time_continuity_ratio);
    for (const auto& source_box : deferred_grouping.boxes) {
      auto projected_box = scale_box_to_shape(source_box,
                                              global_rows,
                                              deferred_group_cols,
                                              global_rows,
                                              global_cols);
      append_projected_box(projected_box);
    }
  }

  result.merged_boxes = merge_projected_boxes(result.projected_boxes);
  result.merged_box_mask = boxes_to_mask(result.merged_boxes, global_rows, global_cols, source_valid_row_mask);
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

void throw_if_cuda_error(cudaError_t result, const char* message) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(result));
  }
}

void allocate_device_float(float*& ptr, size_t count) {
  const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(float));
  throw_if_cuda_error(alloc_result, "device float allocation failed");
}

void allocate_device_uint8(uint8_t*& ptr, size_t count) {
  const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(uint8_t));
  throw_if_cuda_error(alloc_result, "device uint8 allocation failed");
}

void allocate_device_int(int*& ptr, size_t count) {
  const auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(int));
  throw_if_cuda_error(alloc_result, "device int allocation failed");
}

bool is_truthy_backend_mode(const std::string& backend_mode) {
  return backend_mode == "cuda_partial" || backend_mode == "cuda_full_detector";
}

std::vector<double> build_frequency_axis_hz(int num_rows, double resolution_hz) {
  std::vector<double> axis(static_cast<size_t>(std::max(num_rows, 0)), 0.0);
  const bool calibrated = std::isfinite(resolution_hz) && resolution_hz > 0.0;
  for (int row = 0; row < num_rows; ++row) {
    axis[static_cast<size_t>(row)] = calibrated ? static_cast<double>(row) * resolution_hz : static_cast<double>(row);
  }
  return axis;
}

IgnoreSidebandInfo compute_ignore_sideband_rows(int num_rows,
                                                double bin_hz,
                                                double ignore_sideband_hz,
                                                int min_keep_rows) {
  IgnoreSidebandInfo info;
  info.valid_row_mask.assign(static_cast<size_t>(std::max(num_rows, 0)), 1);
  if (num_rows < 2 || !std::isfinite(bin_hz) || bin_hz <= 0.0 ||
      !std::isfinite(ignore_sideband_hz) || ignore_sideband_hz <= 0.0) {
    return info;
  }

  info.bin_hz = bin_hz;
  const int max_bins = std::max(0, (num_rows - std::max(1, min_keep_rows)) / 2);
  const int requested_bins = static_cast<int>(std::ceil(ignore_sideband_hz / bin_hz));
  info.applied_bins = clamp_value(requested_bins, 0, max_bins);
  info.applied_hz = static_cast<double>(info.applied_bins) * bin_hz;
  if (info.applied_bins > 0) {
    std::fill(info.valid_row_mask.begin(), info.valid_row_mask.begin() + info.applied_bins, static_cast<uint8_t>(0));
    std::fill(info.valid_row_mask.end() - info.applied_bins, info.valid_row_mask.end(), static_cast<uint8_t>(0));
  }
  return info;
}

bool chunk_plan_has_uniform_rows(const std::vector<ChunkPlanEntry>& chunks) {
  if (chunks.empty()) {
    return false;
  }
  const int reference_rows = chunk_row_count(chunks.front());
  if (reference_rows <= 0) {
    return false;
  }
  for (const auto& chunk : chunks) {
    if (chunk_row_count(chunk) != reference_rows) {
      return false;
    }
  }
  return true;
}

std::optional<UniformChunkGeometry> calibrated_uniform_chunk_geometry(double bin_hz,
                                                                      double chunk_bandwidth_hz,
                                                                      double chunk_overlap_hz,
                                                                      int min_chunk_rows) {
  if (!std::isfinite(bin_hz) || bin_hz <= 0.0 || !std::isfinite(chunk_bandwidth_hz) || chunk_bandwidth_hz <= 0.0) {
    return std::nullopt;
  }
  const double step_hz = chunk_bandwidth_hz - chunk_overlap_hz;
  if (!std::isfinite(step_hz) || step_hz <= 0.0) {
    return std::nullopt;
  }

  UniformChunkGeometry geometry;
  geometry.chunk_rows = std::max(min_chunk_rows, static_cast<int>(std::llround(chunk_bandwidth_hz / bin_hz)));
  geometry.overlap_rows = clamp_value(static_cast<int>(std::llround(chunk_overlap_hz / bin_hz)), 0, geometry.chunk_rows - 1);
  geometry.step_rows = std::max(1, geometry.chunk_rows - geometry.overlap_rows);
  if (geometry.chunk_rows < min_chunk_rows) {
    return std::nullopt;
  }
  return geometry;
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
    if (valid_row_mask[index]) {
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
  double freq_max = freq_min;
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

std::vector<ChunkPlanEntry> build_uniform_row_chunks(const std::vector<double>& freq_axis_hz,
                                                     const std::vector<uint8_t>& valid_row_mask,
                                                     int chunk_rows,
                                                     int step_rows,
                                                     int min_chunk_rows) {
  std::vector<ChunkPlanEntry> chunks;
  if (freq_axis_hz.empty() || valid_row_mask.size() != freq_axis_hz.size() || chunk_rows < min_chunk_rows || step_rows <= 0) {
    return chunks;
  }

  std::vector<int> valid_idx;
  valid_idx.reserve(valid_row_mask.size());
  for (size_t index = 0; index < valid_row_mask.size(); ++index) {
    if (valid_row_mask[index]) {
      valid_idx.push_back(static_cast<int>(index));
    }
  }
  const int valid_count = static_cast<int>(valid_idx.size());
  if (valid_count < chunk_rows) {
    return chunks;
  }

  int chunk_index = 0;
  for (int start_pos = 0; start_pos + chunk_rows <= valid_count; start_pos += step_rows) {
    const int row_start = valid_idx[static_cast<size_t>(start_pos)];
    const int row_stop = valid_idx[static_cast<size_t>(start_pos + chunk_rows - 1)] + 1;
    chunks.push_back(ChunkPlanEntry{chunk_index++,
                                    row_start,
                                    row_stop,
                                    freq_axis_hz[static_cast<size_t>(row_start)],
                                    freq_axis_hz[static_cast<size_t>(row_stop - 1)]});
  }
  return chunks;
}

PlannedIgnoreSidebandSelection select_uniform_chunk_plan_with_minimal_sideband_trim(
    int num_rows,
    double bin_hz,
    double ignore_sideband_hz,
    int min_keep_rows,
    const std::vector<double>& freq_axis_hz,
    double chunk_bandwidth_hz,
    double chunk_overlap_hz,
    int min_chunk_rows,
    double uncalibrated_chunk_fraction,
    double uncalibrated_overlap_fraction) {
  PlannedIgnoreSidebandSelection selection;
  const auto ignore_info = compute_ignore_sideband_rows(num_rows, bin_hz, ignore_sideband_hz, min_keep_rows);
  selection.applied_bins = ignore_info.applied_bins;
  selection.valid_row_mask = ignore_info.valid_row_mask;

  const auto calibrated_geometry =
      calibrated_uniform_chunk_geometry(bin_hz, chunk_bandwidth_hz, chunk_overlap_hz, min_chunk_rows);

  if (calibrated_geometry.has_value()) {
    const int max_bins = std::max(0, (num_rows - std::max(1, min_keep_rows)) / 2);
    for (int applied_bins = selection.applied_bins; applied_bins <= max_bins; ++applied_bins) {
      const int valid_count = num_rows - 2 * applied_bins;
      if (valid_count < calibrated_geometry->chunk_rows) {
        break;
      }
      if (((valid_count - calibrated_geometry->chunk_rows) % calibrated_geometry->step_rows) != 0) {
        continue;
      }

      std::vector<uint8_t> valid_row_mask(static_cast<size_t>(std::max(num_rows, 0)), static_cast<uint8_t>(1));
      if (applied_bins > 0) {
        std::fill(valid_row_mask.begin(), valid_row_mask.begin() + applied_bins, static_cast<uint8_t>(0));
        std::fill(valid_row_mask.end() - applied_bins, valid_row_mask.end(), static_cast<uint8_t>(0));
      }
      auto chunk_plan = build_uniform_row_chunks(freq_axis_hz,
                                                 valid_row_mask,
                                                 calibrated_geometry->chunk_rows,
                                                 calibrated_geometry->step_rows,
                                                 min_chunk_rows);
      if (!chunk_plan.empty() && chunk_plan_has_uniform_rows(chunk_plan)) {
        selection.applied_bins = applied_bins;
        selection.valid_row_mask = std::move(valid_row_mask);
        selection.chunk_plan = std::move(chunk_plan);
        return selection;
      }
    }
  }

  auto build_plan_for_bins = [&](int applied_bins) {
    std::vector<uint8_t> valid_row_mask(static_cast<size_t>(std::max(num_rows, 0)), static_cast<uint8_t>(1));
    if (applied_bins > 0) {
      std::fill(valid_row_mask.begin(), valid_row_mask.begin() + applied_bins, static_cast<uint8_t>(0));
      std::fill(valid_row_mask.end() - applied_bins, valid_row_mask.end(), static_cast<uint8_t>(0));
    }
    auto chunk_plan = build_frequency_chunks(freq_axis_hz,
                                             chunk_bandwidth_hz,
                                             chunk_overlap_hz,
                                             min_chunk_rows,
                                             valid_row_mask,
                                             uncalibrated_chunk_fraction,
                                             uncalibrated_overlap_fraction);
    return std::make_pair(std::move(valid_row_mask), std::move(chunk_plan));
  };

  auto [valid_row_mask, chunk_plan] = build_plan_for_bins(selection.applied_bins);
  selection.valid_row_mask = std::move(valid_row_mask);
  selection.chunk_plan = std::move(chunk_plan);
  return selection;
}

__global__ void cuda_dino_power_db_kernel(const cuda_dino_complex* input,
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

__global__ void cuda_dino_row_mean_kernel(const float* input, int rows, int cols, float* row_mean) {
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

__global__ void cuda_dino_gaussian_smooth_rows_kernel(const float* input,
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

__global__ void cuda_dino_frontend_reference_kernel(const float* row_smooth,
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

__global__ void cuda_dino_frontend_correction_kernel(const float* input,
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

__global__ void cuda_dino_pack_reference_chunks_kernel(const float* corrected_full,
                                                       int full_rows,
                                                       int cols,
                                                       const int* chunk_row_starts,
                                                       int chunk_rows,
                                                       int batch_size,
                                                       float* corrected_batch) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * chunk_rows * cols;
  if (index >= total) {
    return;
  }

  const int row_col = index % (chunk_rows * cols);
  const int batch_index = index / (chunk_rows * cols);
  const int local_row = row_col / cols;
  const int col = row_col % cols;
  const int global_row = chunk_row_starts[batch_index] + local_row;
  if (global_row < 0 || global_row >= full_rows) {
    corrected_batch[index] = 0.0f;
    return;
  }
  corrected_batch[index] = corrected_full[flat_index(cols, global_row, col)];
}

__global__ void cuda_dino_stitch_reference_chunk_mask_kernel(const float* input_mask_batch,
                                                             int full_rows,
                                                             int cols,
                                                             const int* chunk_row_starts,
                                                             int chunk_rows,
                                                             int batch_size,
                                                             uint8_t* output_mask) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * chunk_rows * cols;
  if (index >= total) {
    return;
  }

  const int row_col = index % (chunk_rows * cols);
  const int batch_index = index / (chunk_rows * cols);
  const int local_row = row_col / cols;
  const int col = row_col % cols;
  const int global_row = chunk_row_starts[batch_index] + local_row;
  if (global_row < 0 || global_row >= full_rows) {
    return;
  }
  if (input_mask_batch[index] > 0.5f) {
    output_mask[flat_index(cols, global_row, col)] = 255;
  }
}

__global__ void project_aligned_maps_copy_batch_kernel(const float* input,
                                                       int batch_size,
                                                       int input_rows,
                                                       int input_cols,
                                                       int output_rows,
                                                       int output_cols,
                                                       float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int input_plane = input_rows * input_cols;
  const int output_plane = output_rows * output_cols;
  const int total = batch_size * output_plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / output_plane;
  const int local_index = idx - batch_index * output_plane;
  const int row = local_index / output_cols;
  const int col = local_index % output_cols;
  float value = 0.0f;
  if (row < input_rows && col < input_cols) {
    const size_t input_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(input_plane);
    value = input[input_offset + flat_index(input_cols, row, col)];
  }
  output[idx] = value;
}

__global__ void project_aligned_maps_bilinear_batch_kernel(const float* input,
                                                           int batch_size,
                                                           int input_rows,
                                                           int input_cols,
                                                           int output_rows,
                                                           int output_cols,
                                                           float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int input_plane = input_rows * input_cols;
  const int output_plane = output_rows * output_cols;
  const int total = batch_size * output_plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / output_plane;
  const int local_index = idx - batch_index * output_plane;
  const int out_row = local_index / output_cols;
  const int out_col = local_index % output_cols;
  const size_t input_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(input_plane);

  const float src_row = ((static_cast<float>(out_row) + 0.5f) * static_cast<float>(input_rows) /
                         static_cast<float>(std::max(output_rows, 1))) -
                        0.5f;
  const float src_col = ((static_cast<float>(out_col) + 0.5f) * static_cast<float>(input_cols) /
                         static_cast<float>(std::max(output_cols, 1))) -
                        0.5f;

  int row0 = static_cast<int>(floorf(src_row));
  int col0 = static_cast<int>(floorf(src_col));
  float row_lerp = src_row - static_cast<float>(row0);
  float col_lerp = src_col - static_cast<float>(col0);

  if (row0 < 0) {
    row0 = 0;
    row_lerp = 0.0f;
  }
  if (col0 < 0) {
    col0 = 0;
    col_lerp = 0.0f;
  }

  int row1 = row0 + 1;
  int col1 = col0 + 1;
  if (row1 >= input_rows) {
    row1 = input_rows - 1;
    row_lerp = 0.0f;
  }
  if (col1 >= input_cols) {
    col1 = input_cols - 1;
    col_lerp = 0.0f;
  }

  const float top_left = input[input_offset + flat_index(input_cols, row0, col0)];
  const float top_right = input[input_offset + flat_index(input_cols, row0, col1)];
  const float bottom_left = input[input_offset + flat_index(input_cols, row1, col0)];
  const float bottom_right = input[input_offset + flat_index(input_cols, row1, col1)];
  const float top = top_left + (top_right - top_left) * col_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * col_lerp;
  output[idx] = top + (bottom - top) * row_lerp;
}

bool resize_maps_bilinear_cuda_batch_to_device(const float* input_maps_batch_device,
                                               int batch_size,
                                               int input_rows,
                                               int input_cols,
                                               int output_rows,
                                               int output_cols,
                                               float* output_maps_device,
                                               cudaStream_t cuda_stream) {
  if (input_maps_batch_device == nullptr || output_maps_device == nullptr || batch_size <= 0 || input_rows <= 0 || input_cols <= 0 ||
      output_rows <= 0 || output_cols <= 0) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const size_t input_total = static_cast<size_t>(batch_size) * static_cast<size_t>(input_rows) * static_cast<size_t>(input_cols);
  if (input_rows == output_rows && input_cols == output_cols) {
    return cudaMemcpyAsync(output_maps_device,
                           input_maps_batch_device,
                           input_total * sizeof(float),
                           cudaMemcpyDeviceToDevice,
                           stream) == cudaSuccess;
  }

  const size_t output_total = static_cast<size_t>(batch_size) * static_cast<size_t>(output_rows) * static_cast<size_t>(output_cols);
  const int threads = 256;
  const int blocks = static_cast<int>((output_total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  project_aligned_maps_bilinear_batch_kernel<<<blocks, threads, 0, stream>>>(input_maps_batch_device,
                                                                              batch_size,
                                                                              input_rows,
                                                                              input_cols,
                                                                              output_rows,
                                                                              output_cols,
                                                                              output_maps_device);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool project_aligned_maps_cuda_batch_to_device(const float* aligned_maps_batch_device,
                                               int batch_size,
                                               int aligned_rows,
                                               int aligned_cols,
                                               int output_rows,
                                               int output_cols,
                                               bool resized_full_chunk,
                                               float* output_score_device,
                                               cudaStream_t cuda_stream) {
  if (aligned_maps_batch_device == nullptr || output_score_device == nullptr || batch_size <= 0 || aligned_rows <= 0 || aligned_cols <= 0 ||
      output_rows <= 0 || output_cols <= 0) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const size_t input_total = static_cast<size_t>(batch_size) * static_cast<size_t>(aligned_rows) * static_cast<size_t>(aligned_cols);
  if (aligned_rows == output_rows && aligned_cols == output_cols) {
    return cudaMemcpyAsync(output_score_device,
                           aligned_maps_batch_device,
                           input_total * sizeof(float),
                           cudaMemcpyDeviceToDevice,
                           stream) == cudaSuccess;
  }

  const size_t output_total = static_cast<size_t>(batch_size) * static_cast<size_t>(output_rows) * static_cast<size_t>(output_cols);
  const int threads = 256;
  const int blocks = static_cast<int>((output_total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  if (resized_full_chunk) {
    project_aligned_maps_bilinear_batch_kernel<<<blocks, threads, 0, stream>>>(aligned_maps_batch_device,
                                                                                batch_size,
                                                                                aligned_rows,
                                                                                aligned_cols,
                                                                                output_rows,
                                                                                output_cols,
                                                                                output_score_device);
  } else {
    project_aligned_maps_copy_batch_kernel<<<blocks, threads, 0, stream>>>(aligned_maps_batch_device,
                                                                            batch_size,
                                                                            aligned_rows,
                                                                            aligned_cols,
                                                                            output_rows,
                                                                            output_cols,
                                                                            output_score_device);
  }
  return cudaGetLastError() == cudaSuccess;
}

namespace {

bool normalize_map01_quantile_exact_cuda_batch_to_device(const float* input_batch_device,
                                                         int batch_size,
                                                         int rows,
                                                         int cols,
                                                         float low_q,
                                                         float high_q,
                                                         float* temp_plane_device,
                                                         float* low_values_device,
                                                         float* high_values_device,
                                                         float* output_batch_device,
                                                         cudaStream_t stream);

}  // namespace

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
                                                                   cudaStream_t cuda_stream) {
  const int patch_count = patch_rows * patch_cols;
  if (batch_size <= 0 || patch_rows <= 0 || patch_cols <= 0 || feature_dim <= 0 || aligned_rows <= 0 || aligned_cols <= 0 ||
      output_rows <= 0 || output_cols <= 0 || patch_features_batch_device == nullptr || output_score_device == nullptr || patch_count <= 0) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const float clamped_suppression = clamp_value(positional_suppression, 0.0f, 1.0f);
  const size_t patch_values = static_cast<size_t>(batch_size) * static_cast<size_t>(patch_count);
  const size_t aligned_values = static_cast<size_t>(batch_size) * static_cast<size_t>(aligned_rows) * static_cast<size_t>(aligned_cols);
  const size_t beta_values = clamped_suppression > 0.0f
                                 ? static_cast<size_t>(batch_size) * static_cast<size_t>(kRawScoreProjectionBasis) * static_cast<size_t>(feature_dim)
                                 : 0;

  auto& scratch = raw_score_projection_cuda_scratch();
  if (!scratch.ensure_capacity(beta_values,
                               patch_values,
                               aligned_values,
                               static_cast<size_t>(patch_count),
                               static_cast<size_t>(batch_size))) {
    return false;
  }

  const int threads = 256;
  const int patch_blocks = static_cast<int>((patch_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  if (clamped_suppression > 0.0f) {
    auto& cache = positional_suppression_cuda_cache();
    if (!cache.ensure_capacity(patch_rows, patch_cols, stream)) {
      return false;
    }

    raw_dino_beta_batch_kernel<<<batch_size * feature_dim, kRawScoreProjectionThreads, 0, stream>>>(patch_features_batch_device,
                                                                                                      batch_size,
                                                                                                      patch_count,
                                                                                                      feature_dim,
                                                                                                      cache.projection_left,
                                                                                                      scratch.beta);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }

    raw_dino_project_energy_batch_kernel<<<patch_blocks, threads, 0, stream>>>(patch_features_batch_device,
                                                                                batch_size,
                                                                                patch_count,
                                                                                feature_dim,
                                                                                cache.design,
                                                                                scratch.beta,
                                                                                clamped_suppression,
                                                                                scratch.patch_values);
  } else {
    raw_dino_rms_energy_batch_kernel<<<patch_blocks, threads, 0, stream>>>(patch_features_batch_device,
                                                                             batch_size,
                                                                             patch_count,
                                                                             feature_dim,
                                                                             scratch.patch_values);
  }
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  if (!normalize_map01_quantile_exact_cuda_batch_to_device(scratch.patch_values,
                                                           batch_size,
                                                           patch_rows,
                                                           patch_cols,
                                                           0.05f,
                                                           0.95f,
                                                           scratch.temp_plane,
                                                           scratch.low_values,
                                                           scratch.high_values,
                                                           scratch.patch_values,
                                                           stream)) {
    return false;
  }

  const float* aligned_input = scratch.patch_values;
  if (patch_rows != aligned_rows || patch_cols != aligned_cols) {
    if (!resize_maps_bilinear_cuda_batch_to_device(scratch.patch_values,
                                                   batch_size,
                                                   patch_rows,
                                                   patch_cols,
                                                   aligned_rows,
                                                   aligned_cols,
                                                   scratch.aligned_maps,
                                                   stream)) {
      return false;
    }
    aligned_input = scratch.aligned_maps;
  }

  return project_aligned_maps_cuda_batch_to_device(aligned_input,
                                                   batch_size,
                                                   aligned_rows,
                                                   aligned_cols,
                                                   output_rows,
                                                   output_cols,
                                                   resized_full_chunk,
                                                   output_score_device,
                                                   stream);
}

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
                                                       cudaStream_t cuda_stream) {
  if (batch_size <= 0 || runtime_rows <= 0 || runtime_cols <= 0 || aligned_rows <= 0 || aligned_cols <= 0 ||
      output_rows <= 0 || output_cols <= 0 || score_maps_batch_device == nullptr || output_score_device == nullptr) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const float* aligned_input = score_maps_batch_device;
  if (runtime_rows != aligned_rows || runtime_cols != aligned_cols) {
    const size_t aligned_values = static_cast<size_t>(batch_size) * static_cast<size_t>(aligned_rows) * static_cast<size_t>(aligned_cols);
    auto& scratch = raw_score_projection_cuda_scratch();
    if (!scratch.ensure_capacity(0, 0, aligned_values, 0, 0)) {
      return false;
    }
    if (!resize_maps_bilinear_cuda_batch_to_device(score_maps_batch_device,
                                                   batch_size,
                                                   runtime_rows,
                                                   runtime_cols,
                                                   aligned_rows,
                                                   aligned_cols,
                                                   scratch.aligned_maps,
                                                   stream)) {
      return false;
    }
    aligned_input = scratch.aligned_maps;
  }

  return project_aligned_maps_cuda_batch_to_device(aligned_input,
                                                   batch_size,
                                                   aligned_rows,
                                                   aligned_cols,
                                                   output_rows,
                                                   output_cols,
                                                   resized_full_chunk,
                                                   output_score_device,
                                                   stream);
}

bool binary_fill_holes_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                            int batch_size,
                                            int rows,
                                            int cols,
                                            uint8_t* output_mask_batch_device,
                                            cudaStream_t cuda_stream) {
  if (mask_batch_device == nullptr || output_mask_batch_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total = static_cast<size_t>(batch_size) * plane;
  auto& fill_scratch = fill_holes_cuda_scratch();
  auto& component_scratch = component_filter_cuda_scratch();
  if (!fill_scratch.ensure_capacity(total) || !component_scratch.ensure_capacity(total, total)) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const int threads = 256;
  const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  fill_holes_init_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                          batch_size,
                                                          rows,
                                                          cols,
                                                          fill_scratch.background);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  if (!label_connected_components_4_cuda_batch_to_device(fill_scratch.background,
                                                         batch_size,
                                                         rows,
                                                         cols,
                                                         component_scratch.labels_a,
                                                         stream)) {
    return false;
  }

  if (cudaMemsetAsync(component_scratch.component_counts, 0, total * sizeof(int), stream) != cudaSuccess) {
    return false;
  }

  connected_components_mark_border_labels_kernel<<<blocks, threads, 0, stream>>>(fill_scratch.background,
                                                                                  component_scratch.labels_a,
                                                                                  batch_size,
                                                                                  rows,
                                                                                  cols,
                                                                                  component_scratch.component_counts);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  fill_holes_finalize_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                              fill_scratch.background,
                                                              component_scratch.labels_a,
                                                              component_scratch.component_counts,
                                                              static_cast<int>(total),
                                                              output_mask_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool binary_closing_rect_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                              int batch_size,
                                              int rows,
                                              int cols,
                                              int kernel_rows,
                                              int kernel_cols,
                                              uint8_t* output_mask_batch_device,
                                              cudaStream_t cuda_stream) {
  if (mask_batch_device == nullptr || output_mask_batch_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const int row_radius = std::max(0, (std::max(1, kernel_rows) - 1) / 2);
  const int col_radius = std::max(0, (std::max(1, kernel_cols) - 1) / 2);
  const size_t total = static_cast<size_t>(batch_size) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  auto& scratch = fill_holes_cuda_scratch();
  if (!scratch.ensure_capacity(total)) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const int threads = 256;
  const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  binary_dilate_rows_batch_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                   batch_size,
                                                                   rows,
                                                                   cols,
                                                                   row_radius,
                                                                   scratch.grown_a);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  binary_dilate_cols_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.grown_a,
                                                                   batch_size,
                                                                   rows,
                                                                   cols,
                                                                   col_radius,
                                                                   scratch.grown_b);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  binary_erode_rows_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.grown_b,
                                                                  batch_size,
                                                                  rows,
                                                                  cols,
                                                                  row_radius,
                                                                  scratch.grown_a);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  binary_erode_cols_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.grown_a,
                                                                  batch_size,
                                                                  rows,
                                                                  cols,
                                                                  col_radius,
                                                                  output_mask_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool keep_large_components_cuda_batch_to_device(const uint8_t* mask_batch_device,
                                                int batch_size,
                                                int rows,
                                                int cols,
                                                int min_size,
                                                uint8_t* output_mask_batch_device,
                                                cudaStream_t cuda_stream) {
  if (mask_batch_device == nullptr || output_mask_batch_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total = static_cast<size_t>(batch_size) * plane;
  if (min_size <= 1) {
    return cudaMemcpyAsync(output_mask_batch_device,
                           mask_batch_device,
                           total * sizeof(uint8_t),
                           cudaMemcpyDeviceToDevice,
                           cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread) == cudaSuccess;
  }

  auto& scratch = component_filter_cuda_scratch();
  if (!scratch.ensure_capacity(total, total)) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const int threads = 256;
  const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  if (!label_connected_components_cuda_batch_to_device(mask_batch_device,
                                                       batch_size,
                                                       rows,
                                                       cols,
                                                       scratch.labels_a,
                                                       stream)) {
    return false;
  }

  if (cudaMemsetAsync(scratch.component_counts, 0, total * sizeof(int), stream) != cudaSuccess) {
    return false;
  }
  connected_components_count_labels_kernel<<<blocks, threads, 0, stream>>>(scratch.labels_a,
                                                                            batch_size,
                                                                            rows,
                                                                            cols,
                                                                            scratch.component_counts);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  component_filter_finalize_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                    scratch.labels_a,
                                                                    batch_size,
                                                                    rows,
                                                                    cols,
                                                                    min_size,
                                                                    scratch.component_counts,
                                                                    output_mask_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

namespace {

constexpr int kHybridThresholdHistogramBins = 256;
constexpr int kHybridThresholdHistogramMapCount = 3;
constexpr int kHybridMaskedMinmaxTripletMapCount = 3;
constexpr int kHybridFp16UnitHistogramPairMapCount = 2;
constexpr int kHybridFp16UnitHistogramBins = 0x3C00 + 1;
constexpr int kHybridReductionThreads = 256;
constexpr int kHybridReductionTileBlocks = 32;
constexpr int kHybridThresholdHistogramSampleStride = 2;

struct ResidualVetoKernelBuffer {
  float* values = nullptr;
  size_t size = 0;
  int radius = 0;

  ~ResidualVetoKernelBuffer() {
    release();
  }

  void release() {
    if (values != nullptr) {
      cudaFree(values);
      values = nullptr;
    }
    size = 0;
    radius = 0;
  }

  bool ensure(const std::vector<float>& host_values) {
    if (host_values.empty()) {
      return false;
    }
    if (values != nullptr && size == host_values.size()) {
      return true;
    }
    release();
    if (cudaMalloc(reinterpret_cast<void**>(&values), host_values.size() * sizeof(float)) != cudaSuccess) {
      release();
      return false;
    }
    if (cudaMemcpy(values, host_values.data(), host_values.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      release();
      return false;
    }
    size = host_values.size();
    radius = static_cast<int>(host_values.size() / 2);
    return true;
  }
};

std::vector<float> build_gaussian_kernel_values(double sigma) {
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
  if (sum <= 0.0) {
    return {1.0f};
  }
  for (float& value : kernel) {
    value = static_cast<float>(static_cast<double>(value) / sum);
  }
  return kernel;
}

std::vector<float> build_gaussian_second_derivative_kernel_values(double sigma) {
  if (sigma <= 0.0) {
    return {0.0f};
  }
  const int radius = std::max(1, static_cast<int>(std::ceil(3.0 * sigma)));
  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1), 0.0f);
  const double sigma2 = sigma * sigma;
  for (int offset = -radius; offset <= radius; ++offset) {
    const double x = static_cast<double>(offset);
    const double value = ((x * x - sigma2) / (sigma2 * sigma2)) * std::exp(-(x * x) / (2.0 * sigma2));
    kernel[static_cast<size_t>(offset + radius)] = static_cast<float>(value);
  }
  return kernel;
}

struct ResidualVetoKernelCache {
  ResidualVetoKernelBuffer gaussian_rows_6;
  ResidualVetoKernelBuffer gaussian_cols_14;
  ResidualVetoKernelBuffer gaussian_rows_4;
  ResidualVetoKernelBuffer gaussian_cols_1;
  ResidualVetoKernelBuffer gaussian_rows_2;
  ResidualVetoKernelBuffer gaussian_cols_08;
  ResidualVetoKernelBuffer second_derivative_rows_08;

  bool ensure_initialized() {
    return gaussian_rows_6.ensure(build_gaussian_kernel_values(6.0)) &&
           gaussian_cols_14.ensure(build_gaussian_kernel_values(1.4)) &&
           gaussian_rows_4.ensure(build_gaussian_kernel_values(4.0)) &&
           gaussian_cols_1.ensure(build_gaussian_kernel_values(1.0)) &&
           gaussian_rows_2.ensure(build_gaussian_kernel_values(2.0)) &&
           gaussian_cols_08.ensure(build_gaussian_kernel_values(0.8)) &&
           second_derivative_rows_08.ensure(build_gaussian_second_derivative_kernel_values(0.8));
  }
};

ResidualVetoKernelCache& residual_veto_kernel_cache() {
  static ResidualVetoKernelCache cache;
  return cache;
}

struct ResidualVetoCudaScratch {
  float* values_a = nullptr;
  float* values_b = nullptr;
  float* values_c = nullptr;
  float* values_d = nullptr;
  float* values_e = nullptr;
  float* temp_plane = nullptr;
  float* batch_a = nullptr;
  float* batch_b = nullptr;
  float* batch_c = nullptr;
  float* batch_d = nullptr;
  float* batch_e = nullptr;
  float* batch_f = nullptr;
  float* reduction_partial_min = nullptr;
  float* reduction_partial_max = nullptr;
  int* reduction_partial_valid = nullptr;
  unsigned int* histograms = nullptr;
  uint8_t* valid_row_mask = nullptr;
  uint8_t* mask_a = nullptr;
  uint8_t* mask_b = nullptr;
  size_t value_capacity = 0;
  size_t plane_capacity = 0;
  size_t batch_capacity = 0;
  size_t reduction_partial_capacity = 0;
  size_t histogram_capacity = 0;
  size_t row_mask_capacity = 0;
  size_t mask_capacity = 0;

  ~ResidualVetoCudaScratch() {
    release();
  }

  void release() {
    release_value_buffers();
    if (temp_plane != nullptr) {
      cudaFree(temp_plane);
      temp_plane = nullptr;
    }
    if (batch_a != nullptr) {
      cudaFree(batch_a);
      batch_a = nullptr;
    }
    if (batch_b != nullptr) {
      cudaFree(batch_b);
      batch_b = nullptr;
    }
    if (batch_c != nullptr) {
      cudaFree(batch_c);
      batch_c = nullptr;
    }
    if (batch_d != nullptr) {
      cudaFree(batch_d);
      batch_d = nullptr;
    }
    if (batch_e != nullptr) {
      cudaFree(batch_e);
      batch_e = nullptr;
    }
    if (batch_f != nullptr) {
      cudaFree(batch_f);
      batch_f = nullptr;
    }
    if (reduction_partial_min != nullptr) {
      cudaFree(reduction_partial_min);
      reduction_partial_min = nullptr;
    }
    if (reduction_partial_max != nullptr) {
      cudaFree(reduction_partial_max);
      reduction_partial_max = nullptr;
    }
    if (reduction_partial_valid != nullptr) {
      cudaFree(reduction_partial_valid);
      reduction_partial_valid = nullptr;
    }
    if (histograms != nullptr) {
      cudaFree(histograms);
      histograms = nullptr;
    }
    if (valid_row_mask != nullptr) {
      cudaFree(valid_row_mask);
      valid_row_mask = nullptr;
    }
    if (mask_a != nullptr) {
      cudaFree(mask_a);
      mask_a = nullptr;
    }
    if (mask_b != nullptr) {
      cudaFree(mask_b);
      mask_b = nullptr;
    }
    plane_capacity = 0;
    batch_capacity = 0;
    reduction_partial_capacity = 0;
    histogram_capacity = 0;
    row_mask_capacity = 0;
    mask_capacity = 0;
  }

  bool ensure_capacity(size_t requested_value_capacity,
                       size_t requested_plane_capacity,
                       size_t requested_batch_capacity,
                       size_t requested_row_mask_capacity,
                       size_t requested_mask_capacity) {
    if (requested_value_capacity > value_capacity) {
      release_value_buffers();
      if (cudaMalloc(reinterpret_cast<void**>(&values_a), requested_value_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&values_b), requested_value_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&values_c), requested_value_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&values_d), requested_value_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&values_e), requested_value_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      value_capacity = requested_value_capacity;
    }
    if (requested_plane_capacity > plane_capacity) {
      if (temp_plane != nullptr) {
        cudaFree(temp_plane);
        temp_plane = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&temp_plane), requested_plane_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      plane_capacity = requested_plane_capacity;
    }
    if (requested_batch_capacity > batch_capacity) {
      if (batch_a != nullptr) {
        cudaFree(batch_a);
        batch_a = nullptr;
      }
      if (batch_b != nullptr) {
        cudaFree(batch_b);
        batch_b = nullptr;
      }
      if (batch_c != nullptr) {
        cudaFree(batch_c);
        batch_c = nullptr;
      }
      if (batch_d != nullptr) {
        cudaFree(batch_d);
        batch_d = nullptr;
      }
      if (batch_e != nullptr) {
        cudaFree(batch_e);
        batch_e = nullptr;
      }
      if (batch_f != nullptr) {
        cudaFree(batch_f);
        batch_f = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&batch_a), requested_batch_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&batch_b), requested_batch_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&batch_c), requested_batch_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&batch_d), requested_batch_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&batch_e), requested_batch_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&batch_f), requested_batch_capacity * sizeof(float)) != cudaSuccess) {
        release();
        return false;
      }
      batch_capacity = requested_batch_capacity;
    }
    const size_t requested_reduction_partial_capacity = requested_batch_capacity * static_cast<size_t>(kHybridReductionTileBlocks) *
                                                      static_cast<size_t>(kHybridMaskedMinmaxTripletMapCount);
    if (requested_reduction_partial_capacity > reduction_partial_capacity) {
      if (reduction_partial_min != nullptr) {
        cudaFree(reduction_partial_min);
        reduction_partial_min = nullptr;
      }
      if (reduction_partial_max != nullptr) {
        cudaFree(reduction_partial_max);
        reduction_partial_max = nullptr;
      }
      if (reduction_partial_valid != nullptr) {
        cudaFree(reduction_partial_valid);
        reduction_partial_valid = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&reduction_partial_min), requested_reduction_partial_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&reduction_partial_max), requested_reduction_partial_capacity * sizeof(float)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&reduction_partial_valid), requested_reduction_partial_capacity * sizeof(int)) != cudaSuccess) {
        release();
        return false;
      }
      reduction_partial_capacity = requested_reduction_partial_capacity;
    }
    const size_t requested_histogram_capacity =
      requested_batch_capacity * static_cast<size_t>(std::max(kHybridThresholdHistogramMapCount * kHybridThresholdHistogramBins,
                                  kHybridFp16UnitHistogramPairMapCount * kHybridFp16UnitHistogramBins));
    if (requested_histogram_capacity > histogram_capacity) {
      if (histograms != nullptr) {
        cudaFree(histograms);
        histograms = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&histograms), requested_histogram_capacity * sizeof(unsigned int)) != cudaSuccess) {
        release();
        return false;
      }
      histogram_capacity = requested_histogram_capacity;
    }
    if (requested_row_mask_capacity > row_mask_capacity) {
      if (valid_row_mask != nullptr) {
        cudaFree(valid_row_mask);
        valid_row_mask = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&valid_row_mask), requested_row_mask_capacity * sizeof(uint8_t)) != cudaSuccess) {
        release();
        return false;
      }
      row_mask_capacity = requested_row_mask_capacity;
    }
    if (requested_mask_capacity > mask_capacity) {
      if (mask_a != nullptr) {
        cudaFree(mask_a);
        mask_a = nullptr;
      }
      if (mask_b != nullptr) {
        cudaFree(mask_b);
        mask_b = nullptr;
      }
      if (cudaMalloc(reinterpret_cast<void**>(&mask_a), requested_mask_capacity * sizeof(uint8_t)) != cudaSuccess ||
          cudaMalloc(reinterpret_cast<void**>(&mask_b), requested_mask_capacity * sizeof(uint8_t)) != cudaSuccess) {
        release();
        return false;
      }
      mask_capacity = requested_mask_capacity;
    }
    return true;
  }

 private:
  void release_value_buffers() {
    if (values_a != nullptr) {
      cudaFree(values_a);
      values_a = nullptr;
    }
    if (values_b != nullptr) {
      cudaFree(values_b);
      values_b = nullptr;
    }
    if (values_c != nullptr) {
      cudaFree(values_c);
      values_c = nullptr;
    }
    if (values_d != nullptr) {
      cudaFree(values_d);
      values_d = nullptr;
    }
    if (values_e != nullptr) {
      cudaFree(values_e);
      values_e = nullptr;
    }
    value_capacity = 0;
  }
};

ResidualVetoCudaScratch& residual_veto_cuda_scratch() {
  static ResidualVetoCudaScratch scratch;
  return scratch;
}

__global__ void residual_veto_round_fp16_copy_kernel(const float* input, int total, float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = __half2float(__float2half_rn(input[idx]));
}

__global__ void residual_veto_round_fp16_inplace_kernel(float* values, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  values[idx] = __half2float(__float2half_rn(values[idx]));
}

__global__ void residual_veto_normalize_quantile_batch_kernel(const float* input,
                                                              int batch_size,
                                                              int rows,
                                                              int cols,
                                                              const float* low_values,
                                                              const float* high_values,
                                                              float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const float low = low_values[batch_index];
  const float high = high_values[batch_index];
  const float inv_scale = 1.0f / fmaxf(high - low, 1.0e-6f);
  output[idx] = fminf(fmaxf((input[idx] - low) * inv_scale, 0.0f), 1.0f);
}

__global__ void residual_veto_normalize_quantile_pair_round_fp16_batch_kernel(const float* input_a,
                                                                               const float* input_b,
                                                                               int batch_size,
                                                                               int rows,
                                                                               int cols,
                                                                               const float* low_values_a,
                                                                               const float* high_values_a,
                                                                               const float* low_values_b,
                                                                               const float* high_values_b,
                                                                               float* output_a,
                                                                               float* output_b) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const float low_a = low_values_a[batch_index];
  const float high_a = high_values_a[batch_index];
  const float low_b = low_values_b[batch_index];
  const float high_b = high_values_b[batch_index];
  const float inv_scale_a = 1.0f / fmaxf(high_a - low_a, 1.0e-6f);
  const float inv_scale_b = 1.0f / fmaxf(high_b - low_b, 1.0e-6f);
  const float value_a = fminf(fmaxf((input_a[idx] - low_a) * inv_scale_a, 0.0f), 1.0f);
  const float value_b = fminf(fmaxf((input_b[idx] - low_b) * inv_scale_b, 0.0f), 1.0f);
  output_a[idx] = __half2float(__float2half_rn(value_a));
  output_b[idx] = __half2float(__float2half_rn(value_b));
}

__global__ void residual_veto_normalize_quantile_pair_multiply_batch_kernel(const float* input_a,
                                                                             const float* input_b,
                                                                             int batch_size,
                                                                             int rows,
                                                                             int cols,
                                                                             const float* low_values_a,
                                                                             const float* high_values_a,
                                                                             const float* low_values_b,
                                                                             const float* high_values_b,
                                                                             float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const float low_a = low_values_a[batch_index];
  const float high_a = high_values_a[batch_index];
  const float low_b = low_values_b[batch_index];
  const float high_b = high_values_b[batch_index];
  const float inv_scale_a = 1.0f / fmaxf(high_a - low_a, 1.0e-6f);
  const float inv_scale_b = 1.0f / fmaxf(high_b - low_b, 1.0e-6f);
  const float value_a = fminf(fmaxf((input_a[idx] - low_a) * inv_scale_a, 0.0f), 1.0f);
  const float value_b = fminf(fmaxf((input_b[idx] - low_b) * inv_scale_b, 0.0f), 1.0f);
  output[idx] = value_a * value_b;
}

__global__ void residual_veto_normalize_quantile_pair_round_fp16_multiply_batch_kernel(const float* input_a,
                                                                                         const float* input_b,
                                                                                         int batch_size,
                                                                                         int rows,
                                                                                         int cols,
                                                                                         const float* low_values_a,
                                                                                         const float* high_values_a,
                                                                                         const float* low_values_b,
                                                                                         const float* high_values_b,
                                                                                         float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const float low_a = low_values_a[batch_index];
  const float high_a = high_values_a[batch_index];
  const float low_b = low_values_b[batch_index];
  const float high_b = high_values_b[batch_index];
  const float inv_scale_a = 1.0f / fmaxf(high_a - low_a, 1.0e-6f);
  const float inv_scale_b = 1.0f / fmaxf(high_b - low_b, 1.0e-6f);
  const float value_a = __half2float(__float2half_rn(fminf(fmaxf((input_a[idx] - low_a) * inv_scale_a, 0.0f), 1.0f)));
  const float value_b = __half2float(__float2half_rn(fminf(fmaxf((input_b[idx] - low_b) * inv_scale_b, 0.0f), 1.0f)));
  output[idx] = value_a * value_b;
}

__global__ void residual_veto_masked_minmax_reduce_batch_kernel(const float* input,
                                                                int batch_size,
                                                                int rows,
                                                                int cols,
                                                                const uint8_t* valid_row_mask,
                                                                float* min_values,
                                                                float* max_values) {
  const int batch_index = blockIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const int tid = threadIdx.x;
  const int plane = rows * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  __shared__ float shared_min[kHybridReductionThreads];
  __shared__ float shared_max[kHybridReductionThreads];
  __shared__ int shared_valid[kHybridReductionThreads];

  float local_min = 1.0e30f;
  float local_max = -1.0e30f;
  int local_valid = 0;
  for (int local_index = tid; local_index < plane; local_index += blockDim.x) {
    const int row = local_index / cols;
    if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
      continue;
    }
    const float value = input[batch_offset + static_cast<size_t>(local_index)];
    local_min = fminf(local_min, value);
    local_max = fmaxf(local_max, value);
    local_valid = 1;
  }

  shared_min[tid] = local_min;
  shared_max[tid] = local_max;
  shared_valid[tid] = local_valid;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_min[tid] = fminf(shared_min[tid], shared_min[tid + stride]);
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
      shared_valid[tid] = shared_valid[tid] | shared_valid[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (shared_valid[0] != 0) {
      min_values[batch_index] = shared_min[0];
      max_values[batch_index] = shared_max[0];
    } else {
      min_values[batch_index] = 0.0f;
      max_values[batch_index] = 1.0f;
    }
  }
}

__global__ void residual_veto_masked_minmax_partial_reduce_batch_kernel(const float* input,
                                                                        int batch_size,
                                                                        int rows,
                                                                        int cols,
                                                                        const uint8_t* valid_row_mask,
                                                                        float* partial_min_values,
                                                                        float* partial_max_values,
                                                                        int* partial_valid_values,
                                                                        int partial_count) {
  const int partial_index = blockIdx.x;
  const int batch_index = blockIdx.y;
  if (batch_index >= batch_size || partial_index >= partial_count) {
    return;
  }

  const int tid = threadIdx.x;
  const int plane = rows * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  __shared__ float shared_min[kHybridReductionThreads];
  __shared__ float shared_max[kHybridReductionThreads];
  __shared__ int shared_valid[kHybridReductionThreads];

  float local_min = 1.0e30f;
  float local_max = -1.0e30f;
  int local_valid = 0;
  const int stride = partial_count * blockDim.x;
  for (int local_index = partial_index * blockDim.x + tid; local_index < plane; local_index += stride) {
    const int row = local_index / cols;
    if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
      continue;
    }
    const float value = input[batch_offset + static_cast<size_t>(local_index)];
    local_min = fminf(local_min, value);
    local_max = fmaxf(local_max, value);
    local_valid = 1;
  }

  shared_min[tid] = local_min;
  shared_max[tid] = local_max;
  shared_valid[tid] = local_valid;
  __syncthreads();

  for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
    if (tid < reduce_stride) {
      shared_min[tid] = fminf(shared_min[tid], shared_min[tid + reduce_stride]);
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + reduce_stride]);
      shared_valid[tid] = shared_valid[tid] | shared_valid[tid + reduce_stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const size_t output_index = static_cast<size_t>(batch_index) * static_cast<size_t>(partial_count) + static_cast<size_t>(partial_index);
    partial_min_values[output_index] = shared_min[0];
    partial_max_values[output_index] = shared_max[0];
    partial_valid_values[output_index] = shared_valid[0];
  }
}

__global__ void residual_veto_masked_minmax_triplet_partial_reduce_batch_kernel(const float* input_a,
                                                                                const float* input_b,
                                                                                const float* input_c,
                                                                                int batch_size,
                                                                                int rows,
                                                                                int cols,
                                                                                const uint8_t* valid_row_mask,
                                                                                float* partial_min_values,
                                                                                float* partial_max_values,
                                                                                int* partial_valid_values,
                                                                                int partial_count) {
  const int partial_index = blockIdx.x;
  const int batch_index = blockIdx.y;
  if (batch_index >= batch_size || partial_index >= partial_count) {
    return;
  }

  const int tid = threadIdx.x;
  const int plane = rows * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  __shared__ float shared_min_a[kHybridReductionThreads];
  __shared__ float shared_max_a[kHybridReductionThreads];
  __shared__ float shared_min_b[kHybridReductionThreads];
  __shared__ float shared_max_b[kHybridReductionThreads];
  __shared__ float shared_min_c[kHybridReductionThreads];
  __shared__ float shared_max_c[kHybridReductionThreads];
  __shared__ int shared_valid[kHybridReductionThreads];

  float local_min_a = 1.0e30f;
  float local_max_a = -1.0e30f;
  float local_min_b = 1.0e30f;
  float local_max_b = -1.0e30f;
  float local_min_c = 1.0e30f;
  float local_max_c = -1.0e30f;
  int local_valid = 0;
  const int stride = partial_count * blockDim.x;
  for (int local_index = partial_index * blockDim.x + tid; local_index < plane; local_index += stride) {
    const int row = local_index / cols;
    if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
      continue;
    }
    const size_t source_index = batch_offset + static_cast<size_t>(local_index);
    const float value_a = input_a[source_index];
    const float value_b = input_b[source_index];
    const float value_c = input_c[source_index];
    local_min_a = fminf(local_min_a, value_a);
    local_max_a = fmaxf(local_max_a, value_a);
    local_min_b = fminf(local_min_b, value_b);
    local_max_b = fmaxf(local_max_b, value_b);
    local_min_c = fminf(local_min_c, value_c);
    local_max_c = fmaxf(local_max_c, value_c);
    local_valid = 1;
  }

  shared_min_a[tid] = local_min_a;
  shared_max_a[tid] = local_max_a;
  shared_min_b[tid] = local_min_b;
  shared_max_b[tid] = local_max_b;
  shared_min_c[tid] = local_min_c;
  shared_max_c[tid] = local_max_c;
  shared_valid[tid] = local_valid;
  __syncthreads();

  for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
    if (tid < reduce_stride) {
      shared_min_a[tid] = fminf(shared_min_a[tid], shared_min_a[tid + reduce_stride]);
      shared_max_a[tid] = fmaxf(shared_max_a[tid], shared_max_a[tid + reduce_stride]);
      shared_min_b[tid] = fminf(shared_min_b[tid], shared_min_b[tid + reduce_stride]);
      shared_max_b[tid] = fmaxf(shared_max_b[tid], shared_max_b[tid + reduce_stride]);
      shared_min_c[tid] = fminf(shared_min_c[tid], shared_min_c[tid + reduce_stride]);
      shared_max_c[tid] = fmaxf(shared_max_c[tid], shared_max_c[tid + reduce_stride]);
      shared_valid[tid] = shared_valid[tid] | shared_valid[tid + reduce_stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const size_t per_map_stride = static_cast<size_t>(batch_size) * static_cast<size_t>(partial_count);
    const size_t output_index = static_cast<size_t>(batch_index) * static_cast<size_t>(partial_count) + static_cast<size_t>(partial_index);
    partial_min_values[output_index] = shared_min_a[0];
    partial_max_values[output_index] = shared_max_a[0];
    partial_valid_values[output_index] = shared_valid[0];
    partial_min_values[per_map_stride + output_index] = shared_min_b[0];
    partial_max_values[per_map_stride + output_index] = shared_max_b[0];
    partial_valid_values[per_map_stride + output_index] = shared_valid[0];
    partial_min_values[2 * per_map_stride + output_index] = shared_min_c[0];
    partial_max_values[2 * per_map_stride + output_index] = shared_max_c[0];
    partial_valid_values[2 * per_map_stride + output_index] = shared_valid[0];
  }
}

__device__ __forceinline__ float residual_veto_subtract_scaled_value(float lhs, float rhs, float rhs_scale);

__device__ __forceinline__ float residual_veto_combined_input_value(float keep_freq, float keep_res);

__global__ void residual_veto_subtract_scaled_masked_minmax_partial_reduce_batch_kernel(const float* lhs,
                                                                                         const float* rhs,
                                                                                         int batch_size,
                                                                                         int rows,
                                                                                         int cols,
                                                                                         const uint8_t* valid_row_mask,
                                                                                         float rhs_scale,
                                                                                         float* partial_min_values,
                                                                                         float* partial_max_values,
                                                                                         int* partial_valid_values,
                                                                                         int partial_count) {
  const int partial_index = blockIdx.x;
  const int batch_index = blockIdx.y;
  if (batch_index >= batch_size || partial_index >= partial_count) {
    return;
  }

  const int tid = threadIdx.x;
  const int plane = rows * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  __shared__ float shared_min[kHybridReductionThreads];
  __shared__ float shared_max[kHybridReductionThreads];
  __shared__ int shared_valid[kHybridReductionThreads];

  float local_min = 1.0e30f;
  float local_max = -1.0e30f;
  int local_valid = 0;
  const int stride = partial_count * blockDim.x;
  for (int local_index = partial_index * blockDim.x + tid; local_index < plane; local_index += stride) {
    const int row = local_index / cols;
    if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
      continue;
    }
    const size_t source_index = batch_offset + static_cast<size_t>(local_index);
    const float value = residual_veto_subtract_scaled_value(lhs[source_index], rhs[source_index], rhs_scale);
    local_min = fminf(local_min, value);
    local_max = fmaxf(local_max, value);
    local_valid = 1;
  }

  shared_min[tid] = local_min;
  shared_max[tid] = local_max;
  shared_valid[tid] = local_valid;
  __syncthreads();

  for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
    if (tid < reduce_stride) {
      shared_min[tid] = fminf(shared_min[tid], shared_min[tid + reduce_stride]);
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + reduce_stride]);
      shared_valid[tid] = shared_valid[tid] | shared_valid[tid + reduce_stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const size_t output_index = static_cast<size_t>(batch_index) * static_cast<size_t>(partial_count) + static_cast<size_t>(partial_index);
    partial_min_values[output_index] = shared_min[0];
    partial_max_values[output_index] = shared_max[0];
    partial_valid_values[output_index] = shared_valid[0];
  }
}

__global__ void residual_veto_combined_input_masked_minmax_partial_reduce_batch_kernel(const float* keep_freq,
                                                                                        const float* keep_res,
                                                                                        int batch_size,
                                                                                        int rows,
                                                                                        int cols,
                                                                                        const uint8_t* valid_row_mask,
                                                                                        float* partial_min_values,
                                                                                        float* partial_max_values,
                                                                                        int* partial_valid_values,
                                                                                        int partial_count) {
  const int partial_index = blockIdx.x;
  const int batch_index = blockIdx.y;
  if (batch_index >= batch_size || partial_index >= partial_count) {
    return;
  }

  const int tid = threadIdx.x;
  const int plane = rows * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  __shared__ float shared_min[kHybridReductionThreads];
  __shared__ float shared_max[kHybridReductionThreads];
  __shared__ int shared_valid[kHybridReductionThreads];

  float local_min = 1.0e30f;
  float local_max = -1.0e30f;
  int local_valid = 0;
  const int stride = partial_count * blockDim.x;
  for (int local_index = partial_index * blockDim.x + tid; local_index < plane; local_index += stride) {
    const int row = local_index / cols;
    if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
      continue;
    }
    const size_t source_index = batch_offset + static_cast<size_t>(local_index);
    const float value = residual_veto_combined_input_value(keep_freq[source_index], keep_res[source_index]);
    local_min = fminf(local_min, value);
    local_max = fmaxf(local_max, value);
    local_valid = 1;
  }

  shared_min[tid] = local_min;
  shared_max[tid] = local_max;
  shared_valid[tid] = local_valid;
  __syncthreads();

  for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
    if (tid < reduce_stride) {
      shared_min[tid] = fminf(shared_min[tid], shared_min[tid + reduce_stride]);
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + reduce_stride]);
      shared_valid[tid] = shared_valid[tid] | shared_valid[tid + reduce_stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const size_t output_index = static_cast<size_t>(batch_index) * static_cast<size_t>(partial_count) + static_cast<size_t>(partial_index);
    partial_min_values[output_index] = shared_min[0];
    partial_max_values[output_index] = shared_max[0];
    partial_valid_values[output_index] = shared_valid[0];
  }
}

__global__ void residual_veto_subtract_scaled_masked_minmax_pair_partial_reduce_batch_kernel(const float* lhs,
                                                                                              const float* rhs_a,
                                                                                              const float* rhs_b,
                                                                                              int batch_size,
                                                                                              int rows,
                                                                                              int cols,
                                                                                              const uint8_t* valid_row_mask,
                                                                                              float rhs_a_scale,
                                                                                              float rhs_b_scale,
                                                                                              float* partial_min_values,
                                                                                              float* partial_max_values,
                                                                                              int* partial_valid_values,
                                                                                              int partial_count) {
  const int partial_index = blockIdx.x;
  const int batch_index = blockIdx.y;
  if (batch_index >= batch_size || partial_index >= partial_count) {
    return;
  }

  const int tid = threadIdx.x;
  const int plane = rows * cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  __shared__ float shared_min_a[kHybridReductionThreads];
  __shared__ float shared_max_a[kHybridReductionThreads];
  __shared__ float shared_min_b[kHybridReductionThreads];
  __shared__ float shared_max_b[kHybridReductionThreads];
  __shared__ int shared_valid[kHybridReductionThreads];

  float local_min_a = 1.0e30f;
  float local_max_a = -1.0e30f;
  float local_min_b = 1.0e30f;
  float local_max_b = -1.0e30f;
  int local_valid = 0;
  const int stride = partial_count * blockDim.x;
  for (int local_index = partial_index * blockDim.x + tid; local_index < plane; local_index += stride) {
    const int row = local_index / cols;
    if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
      continue;
    }
    const size_t source_index = batch_offset + static_cast<size_t>(local_index);
    const float lhs_value = lhs[source_index];
    const float value_a = residual_veto_subtract_scaled_value(lhs_value, rhs_a[source_index], rhs_a_scale);
    const float value_b = residual_veto_subtract_scaled_value(lhs_value, rhs_b[source_index], rhs_b_scale);
    local_min_a = fminf(local_min_a, value_a);
    local_max_a = fmaxf(local_max_a, value_a);
    local_min_b = fminf(local_min_b, value_b);
    local_max_b = fmaxf(local_max_b, value_b);
    local_valid = 1;
  }

  shared_min_a[tid] = local_min_a;
  shared_max_a[tid] = local_max_a;
  shared_min_b[tid] = local_min_b;
  shared_max_b[tid] = local_max_b;
  shared_valid[tid] = local_valid;
  __syncthreads();

  for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
    if (tid < reduce_stride) {
      shared_min_a[tid] = fminf(shared_min_a[tid], shared_min_a[tid + reduce_stride]);
      shared_max_a[tid] = fmaxf(shared_max_a[tid], shared_max_a[tid + reduce_stride]);
      shared_min_b[tid] = fminf(shared_min_b[tid], shared_min_b[tid + reduce_stride]);
      shared_max_b[tid] = fmaxf(shared_max_b[tid], shared_max_b[tid + reduce_stride]);
      shared_valid[tid] = shared_valid[tid] | shared_valid[tid + reduce_stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const size_t per_map_stride = static_cast<size_t>(batch_size) * static_cast<size_t>(partial_count);
    const size_t output_index = static_cast<size_t>(batch_index) * static_cast<size_t>(partial_count) + static_cast<size_t>(partial_index);
    partial_min_values[output_index] = shared_min_a[0];
    partial_max_values[output_index] = shared_max_a[0];
    partial_valid_values[output_index] = shared_valid[0];
    partial_min_values[per_map_stride + output_index] = shared_min_b[0];
    partial_max_values[per_map_stride + output_index] = shared_max_b[0];
    partial_valid_values[per_map_stride + output_index] = shared_valid[0];
  }
}

__global__ void residual_veto_masked_minmax_finalize_batch_kernel(const float* partial_min_values,
                                                                  const float* partial_max_values,
                                                                  const int* partial_valid_values,
                                                                  int batch_size,
                                                                  int partial_count,
                                                                  float* min_values,
                                                                  float* max_values) {
  const int batch_index = blockIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const int tid = threadIdx.x;
  __shared__ float shared_min[kHybridReductionThreads];
  __shared__ float shared_max[kHybridReductionThreads];
  __shared__ int shared_valid[kHybridReductionThreads];

  float local_min = 1.0e30f;
  float local_max = -1.0e30f;
  int local_valid = 0;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(partial_count);
  for (int partial_index = tid; partial_index < partial_count; partial_index += blockDim.x) {
    const size_t source_index = batch_offset + static_cast<size_t>(partial_index);
    if (partial_valid_values[source_index] == 0) {
      continue;
    }
    local_min = fminf(local_min, partial_min_values[source_index]);
    local_max = fmaxf(local_max, partial_max_values[source_index]);
    local_valid = 1;
  }

  shared_min[tid] = local_min;
  shared_max[tid] = local_max;
  shared_valid[tid] = local_valid;
  __syncthreads();

  for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1) {
    if (tid < reduce_stride) {
      shared_min[tid] = fminf(shared_min[tid], shared_min[tid + reduce_stride]);
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + reduce_stride]);
      shared_valid[tid] = shared_valid[tid] | shared_valid[tid + reduce_stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (shared_valid[0] != 0) {
      min_values[batch_index] = shared_min[0];
      max_values[batch_index] = shared_max[0];
    } else {
      min_values[batch_index] = 0.0f;
      max_values[batch_index] = 1.0f;
    }
  }
}

__global__ void residual_veto_normalize_masked_minmax_batch_kernel(const float* input,
                                                                   int batch_size,
                                                                   int rows,
                                                                   int cols,
                                                                   const uint8_t* valid_row_mask,
                                                                   const float* min_values,
                                                                   const float* max_values,
                                                                   float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    output[idx] = 0.0f;
    return;
  }

  const float low = min_values[batch_index];
  const float high = max_values[batch_index];
  const float inv_scale = 1.0f / fmaxf(high - low, 1.0e-6f);
  output[idx] = fminf(fmaxf((input[idx] - low) * inv_scale, 0.0f), 1.0f);
}

__global__ void residual_veto_normalize_masked_minmax_triplet_batch_kernel(const float* input_a,
                                                                           const float* input_b,
                                                                           const float* input_c,
                                                                           int batch_size,
                                                                           int rows,
                                                                           int cols,
                                                                           const uint8_t* valid_row_mask,
                                                                           const float* min_values_a,
                                                                           const float* max_values_a,
                                                                           const float* min_values_b,
                                                                           const float* max_values_b,
                                                                           const float* min_values_c,
                                                                           const float* max_values_c,
                                                                           float* output_a,
                                                                           float* output_b,
                                                                           float* output_c) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    output_a[idx] = 0.0f;
    output_b[idx] = 0.0f;
    output_c[idx] = 0.0f;
    return;
  }

  const float low_a = min_values_a[batch_index];
  const float high_a = max_values_a[batch_index];
  const float low_b = min_values_b[batch_index];
  const float high_b = max_values_b[batch_index];
  const float low_c = min_values_c[batch_index];
  const float high_c = max_values_c[batch_index];
  const float inv_scale_a = 1.0f / fmaxf(high_a - low_a, 1.0e-6f);
  const float inv_scale_b = 1.0f / fmaxf(high_b - low_b, 1.0e-6f);
  const float inv_scale_c = 1.0f / fmaxf(high_c - low_c, 1.0e-6f);
  output_a[idx] = fminf(fmaxf((input_a[idx] - low_a) * inv_scale_a, 0.0f), 1.0f);
  output_b[idx] = fminf(fmaxf((input_b[idx] - low_b) * inv_scale_b, 0.0f), 1.0f);
  output_c[idx] = fminf(fmaxf((input_c[idx] - low_c) * inv_scale_c, 0.0f), 1.0f);
}

__global__ void residual_veto_subtract_scaled_normalize_masked_minmax_batch_kernel(const float* lhs,
                                                                                    const float* rhs,
                                                                                    int batch_size,
                                                                                    int rows,
                                                                                    int cols,
                                                                                    const uint8_t* valid_row_mask,
                                                                                    const float* min_values,
                                                                                    const float* max_values,
                                                                                    float rhs_scale,
                                                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    output[idx] = 0.0f;
    return;
  }

  const float low = min_values[batch_index];
  const float high = max_values[batch_index];
  const float inv_scale = 1.0f / fmaxf(high - low, 1.0e-6f);
  const float value = residual_veto_subtract_scaled_value(lhs[idx], rhs[idx], rhs_scale);
  output[idx] = fminf(fmaxf((value - low) * inv_scale, 0.0f), 1.0f);
}

__global__ void residual_veto_subtract_scaled_normalize_masked_minmax_pair_batch_kernel(const float* lhs,
                                                                                         const float* rhs_a,
                                                                                         const float* rhs_b,
                                                                                         int batch_size,
                                                                                         int rows,
                                                                                         int cols,
                                                                                         const uint8_t* valid_row_mask,
                                                                                         const float* min_values_a,
                                                                                         const float* max_values_a,
                                                                                         const float* min_values_b,
                                                                                         const float* max_values_b,
                                                                                         float rhs_a_scale,
                                                                                         float rhs_b_scale,
                                                                                         float* output_a,
                                                                                         float* output_b) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    output_a[idx] = 0.0f;
    output_b[idx] = 0.0f;
    return;
  }

  const float low_a = min_values_a[batch_index];
  const float high_a = max_values_a[batch_index];
  const float low_b = min_values_b[batch_index];
  const float high_b = max_values_b[batch_index];
  const float inv_scale_a = 1.0f / fmaxf(high_a - low_a, 1.0e-6f);
  const float inv_scale_b = 1.0f / fmaxf(high_b - low_b, 1.0e-6f);
  const float lhs_value = lhs[idx];
  const float value_a = residual_veto_subtract_scaled_value(lhs_value, rhs_a[idx], rhs_a_scale);
  const float value_b = residual_veto_subtract_scaled_value(lhs_value, rhs_b[idx], rhs_b_scale);
  output_a[idx] = fminf(fmaxf((value_a - low_a) * inv_scale_a, 0.0f), 1.0f);
  output_b[idx] = fminf(fmaxf((value_b - low_b) * inv_scale_b, 0.0f), 1.0f);
}

__global__ void residual_veto_combined_input_normalize_masked_minmax_batch_kernel(const float* keep_freq,
                                                                                   const float* keep_res,
                                                                                   int batch_size,
                                                                                   int rows,
                                                                                   int cols,
                                                                                   const uint8_t* valid_row_mask,
                                                                                   const float* min_values,
                                                                                   const float* max_values,
                                                                                   float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    output[idx] = 0.0f;
    return;
  }

  const float low = min_values[batch_index];
  const float high = max_values[batch_index];
  const float inv_scale = 1.0f / fmaxf(high - low, 1.0e-6f);
  const float value = residual_veto_combined_input_value(keep_freq[idx], keep_res[idx]);
  output[idx] = fminf(fmaxf((value - low) * inv_scale, 0.0f), 1.0f);
}

__global__ void residual_veto_multiply_batch_kernel(const float* lhs,
                                                    const float* rhs,
                                                    int total,
                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = lhs[idx] * rhs[idx];
}

__global__ void residual_veto_abs_diff_batch_kernel(const float* lhs,
                                                    const float* rhs,
                                                    int total,
                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = fabsf(lhs[idx] - rhs[idx]);
}

__global__ void residual_veto_abs_inplace_batch_kernel(float* values, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  values[idx] = fabsf(values[idx]);
}

__global__ void residual_veto_subtract_scaled_batch_kernel(const float* lhs,
                                                           const float* rhs,
                                                           int total,
                                                           float rhs_scale,
                                                           float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = lhs[idx] - rhs_scale * rhs[idx];
}

__device__ __forceinline__ float residual_veto_subtract_scaled_value(float lhs, float rhs, float rhs_scale) {
  return lhs - rhs_scale * rhs;
}

__global__ void residual_veto_combined_input_batch_kernel(const float* keep_freq,
                                                          const float* keep_res,
                                                          int total,
                                                          float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float residual_veto_gate = fminf(fmaxf((keep_res[idx] - 0.30f) / 0.70f, 0.0f), 1.0f);
  output[idx] = keep_freq[idx] * (0.35f + 0.65f * residual_veto_gate);
}

__device__ __forceinline__ float residual_veto_combined_input_value(float keep_freq, float keep_res) {
  const float residual_veto_gate = fminf(fmaxf((keep_res - 0.30f) / 0.70f, 0.0f), 1.0f);
  return keep_freq * (0.35f + 0.65f * residual_veto_gate);
}

__global__ void residual_veto_convolve_rows_batch_kernel(const float* input,
                                                         int batch_size,
                                                         int rows,
                                                         int cols,
                                                         const float* kernel,
                                                         int radius,
                                                         float* output) {
  const int line = blockIdx.y;
  if (line >= batch_size * cols || rows <= 0) {
    return;
  }

  const int tile_start = blockIdx.x * blockDim.x;
  const int local = threadIdx.x;
  const int row = tile_start + local;
  const int batch_index = line / cols;
  const int col = line - batch_index * cols;
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t batch_offset = static_cast<size_t>(batch_index) * plane;

  extern __shared__ float shared_line[];
  const int clamped_row = clamp_value(row, 0, rows - 1);
  shared_line[local + radius] = input[batch_offset + flat_index(cols, clamped_row, col)];
  if (local < radius) {
    const int left_halo_row = clamp_value(tile_start + local - radius, 0, rows - 1);
    const int right_halo_row = clamp_value(tile_start + static_cast<int>(blockDim.x) + local, 0, rows - 1);
    shared_line[local] = input[batch_offset + flat_index(cols, left_halo_row, col)];
    shared_line[local + radius + blockDim.x] = input[batch_offset + flat_index(cols, right_halo_row, col)];
  }
  __syncthreads();

  if (row >= rows) {
    return;
  }

  float sum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    sum += kernel[offset + radius] * shared_line[local + radius + offset];
  }
  output[batch_offset + flat_index(cols, row, col)] = sum;
}

__global__ void residual_veto_convolve_row_filter_bank_batch_kernel(const float* input,
                                                                    int batch_size,
                                                                    int rows,
                                                                    int cols,
                                                                    const float* kernel_a,
                                                                    int radius_a,
                                                                    float* output_a,
                                                                    const float* kernel_b,
                                                                    int radius_b,
                                                                    float* output_b,
                                                                    const float* kernel_c,
                                                                    int radius_c,
                                                                    float* output_c,
                                                                    int max_radius) {
  const int line = blockIdx.y;
  if (line >= batch_size * cols || rows <= 0) {
    return;
  }

  const int tile_start = blockIdx.x * blockDim.x;
  const int local = threadIdx.x;
  const int row = tile_start + local;
  const int batch_index = line / cols;
  const int col = line - batch_index * cols;
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t batch_offset = static_cast<size_t>(batch_index) * plane;

  extern __shared__ float shared_line[];
  const int clamped_row = clamp_value(row, 0, rows - 1);
  shared_line[local + max_radius] = input[batch_offset + flat_index(cols, clamped_row, col)];
  if (local < max_radius) {
    const int left_halo_row = clamp_value(tile_start + local - max_radius, 0, rows - 1);
    const int right_halo_row = clamp_value(tile_start + static_cast<int>(blockDim.x) + local, 0, rows - 1);
    shared_line[local] = input[batch_offset + flat_index(cols, left_halo_row, col)];
    shared_line[local + max_radius + blockDim.x] = input[batch_offset + flat_index(cols, right_halo_row, col)];
  }
  __syncthreads();

  if (row >= rows) {
    return;
  }

  float sum_a = 0.0f;
  float sum_b = 0.0f;
  float sum_c = 0.0f;
  for (int offset = -max_radius; offset <= max_radius; ++offset) {
    const float value = shared_line[local + max_radius + offset];
    if (offset >= -radius_a && offset <= radius_a) {
      sum_a += kernel_a[offset + radius_a] * value;
    }
    if (offset >= -radius_b && offset <= radius_b) {
      sum_b += kernel_b[offset + radius_b] * value;
    }
    if (offset >= -radius_c && offset <= radius_c) {
      sum_c += kernel_c[offset + radius_c] * value;
    }
  }

  const size_t output_offset = batch_offset + flat_index(cols, row, col);
  output_a[output_offset] = sum_a;
  output_b[output_offset] = sum_b;
  output_c[output_offset] = sum_c;
}

__global__ void residual_veto_normalize_masked_minmax_row_filter_bank_batch_kernel(const float* input,
                                                                                    int batch_size,
                                                                                    int rows,
                                                                                    int cols,
                                                                                    const uint8_t* valid_row_mask,
                                                                                    const float* min_values,
                                                                                    const float* max_values,
                                                                                    float* normalized_output,
                                                                                    const float* kernel_a,
                                                                                    int radius_a,
                                                                                    float* output_a,
                                                                                    const float* kernel_b,
                                                                                    int radius_b,
                                                                                    float* output_b,
                                                                                    const float* kernel_c,
                                                                                    int radius_c,
                                                                                    float* output_c,
                                                                                    int max_radius) {
  const int line = blockIdx.y;
  if (line >= batch_size * cols || rows <= 0) {
    return;
  }

  const int tile_start = blockIdx.x * blockDim.x;
  const int local = threadIdx.x;
  const int row = tile_start + local;
  const int batch_index = line / cols;
  const int col = line - batch_index * cols;
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t batch_offset = static_cast<size_t>(batch_index) * plane;
  const size_t mask_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(rows);
  const float low = min_values[batch_index];
  const float high = max_values[batch_index];
  const float inv_scale = 1.0f / fmaxf(high - low, 1.0e-6f);

  extern __shared__ float shared_line[];
  const int clamped_row = clamp_value(row, 0, rows - 1);
  const bool center_valid = valid_row_mask[mask_offset + static_cast<size_t>(clamped_row)] != 0;
  const float center_value = center_valid
                                 ? fminf(fmaxf((input[batch_offset + flat_index(cols, clamped_row, col)] - low) * inv_scale, 0.0f), 1.0f)
                                 : 0.0f;
  shared_line[local + max_radius] = center_value;
  if (local < max_radius) {
    const int left_halo_row = clamp_value(tile_start + local - max_radius, 0, rows - 1);
    const int right_halo_row = clamp_value(tile_start + static_cast<int>(blockDim.x) + local, 0, rows - 1);
    const bool left_valid = valid_row_mask[mask_offset + static_cast<size_t>(left_halo_row)] != 0;
    const bool right_valid = valid_row_mask[mask_offset + static_cast<size_t>(right_halo_row)] != 0;
    shared_line[local] = left_valid
                             ? fminf(fmaxf((input[batch_offset + flat_index(cols, left_halo_row, col)] - low) * inv_scale, 0.0f), 1.0f)
                             : 0.0f;
    shared_line[local + max_radius + blockDim.x] = right_valid
                                                       ? fminf(fmaxf((input[batch_offset + flat_index(cols, right_halo_row, col)] - low) * inv_scale, 0.0f), 1.0f)
                                                       : 0.0f;
  }
  __syncthreads();

  if (row >= rows) {
    return;
  }

  float sum_a = 0.0f;
  float sum_b = 0.0f;
  float sum_c = 0.0f;
  for (int offset = -max_radius; offset <= max_radius; ++offset) {
    const float value = shared_line[local + max_radius + offset];
    if (offset >= -radius_a && offset <= radius_a) {
      sum_a += kernel_a[offset + radius_a] * value;
    }
    if (offset >= -radius_b && offset <= radius_b) {
      sum_b += kernel_b[offset + radius_b] * value;
    }
    if (offset >= -radius_c && offset <= radius_c) {
      sum_c += kernel_c[offset + radius_c] * value;
    }
  }

  const size_t output_offset = batch_offset + flat_index(cols, row, col);
  normalized_output[output_offset] = shared_line[local + max_radius];
  output_a[output_offset] = sum_a;
  output_b[output_offset] = sum_b;
  output_c[output_offset] = sum_c;
}

__global__ void residual_veto_convolve_cols_batch_kernel(const float* input,
                                                         int batch_size,
                                                         int rows,
                                                         int cols,
                                                         const float* kernel,
                                                         int radius,
                                                         float* output) {
  const int line = blockIdx.y;
  if (line >= batch_size * rows || cols <= 0) {
    return;
  }

  const int tile_start = blockIdx.x * blockDim.x;
  const int local = threadIdx.x;
  const int col = tile_start + local;
  const int batch_index = line / rows;
  const int row = line - batch_index * rows;
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t batch_offset = static_cast<size_t>(batch_index) * plane;

  extern __shared__ float shared_line[];
  const int clamped_col = clamp_value(col, 0, cols - 1);
  shared_line[local + radius] = input[batch_offset + flat_index(cols, row, clamped_col)];
  if (local < radius) {
    const int left_halo_col = clamp_value(tile_start + local - radius, 0, cols - 1);
    const int right_halo_col = clamp_value(tile_start + static_cast<int>(blockDim.x) + local, 0, cols - 1);
    shared_line[local] = input[batch_offset + flat_index(cols, row, left_halo_col)];
    shared_line[local + radius + blockDim.x] = input[batch_offset + flat_index(cols, row, right_halo_col)];
  }
  __syncthreads();

  if (col >= cols) {
    return;
  }

  float sum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    sum += kernel[offset + radius] * shared_line[local + radius + offset];
  }
  output[batch_offset + flat_index(cols, row, col)] = sum;
}

__global__ void residual_veto_dual_convolve_cols_batch_kernel(const float* input_a,
                                                              const float* input_b,
                                                              int batch_size,
                                                              int rows,
                                                              int cols,
                                                              const float* kernel_a,
                                                              int radius_a,
                                                              float* output_a,
                                                              const float* kernel_b,
                                                              int radius_b,
                                                              float* output_b,
                                                              int max_radius) {
  const int line = blockIdx.y;
  if (line >= batch_size * rows || cols <= 0) {
    return;
  }

  const int tile_start = blockIdx.x * blockDim.x;
  const int local = threadIdx.x;
  const int col = tile_start + local;
  const int batch_index = line / rows;
  const int row = line - batch_index * rows;
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t batch_offset = static_cast<size_t>(batch_index) * plane;

  extern __shared__ float shared_storage[];
  float* shared_a = shared_storage;
  float* shared_b = shared_storage + static_cast<size_t>(blockDim.x + 2 * max_radius);

  const int clamped_col = clamp_value(col, 0, cols - 1);
  shared_a[local + max_radius] = input_a[batch_offset + flat_index(cols, row, clamped_col)];
  shared_b[local + max_radius] = input_b[batch_offset + flat_index(cols, row, clamped_col)];
  if (local < max_radius) {
    const int left_halo_col = clamp_value(tile_start + local - max_radius, 0, cols - 1);
    const int right_halo_col = clamp_value(tile_start + static_cast<int>(blockDim.x) + local, 0, cols - 1);
    shared_a[local] = input_a[batch_offset + flat_index(cols, row, left_halo_col)];
    shared_a[local + max_radius + blockDim.x] = input_a[batch_offset + flat_index(cols, row, right_halo_col)];
    shared_b[local] = input_b[batch_offset + flat_index(cols, row, left_halo_col)];
    shared_b[local + max_radius + blockDim.x] = input_b[batch_offset + flat_index(cols, row, right_halo_col)];
  }
  __syncthreads();

  if (col >= cols) {
    return;
  }

  float sum_a = 0.0f;
  float sum_b = 0.0f;
  for (int offset = -max_radius; offset <= max_radius; ++offset) {
    if (offset >= -radius_a && offset <= radius_a) {
      sum_a += kernel_a[offset + radius_a] * shared_a[local + max_radius + offset];
    }
    if (offset >= -radius_b && offset <= radius_b) {
      sum_b += kernel_b[offset + radius_b] * shared_b[local + max_radius + offset];
    }
  }

  const size_t output_offset = batch_offset + flat_index(cols, row, col);
  output_a[output_offset] = sum_a;
  output_b[output_offset] = sum_b;
}

__global__ void residual_veto_histogram_valid_rows_batch_kernel(const float* input,
                                                                int batch_size,
                                                                int rows,
                                                                int cols,
                                                                const uint8_t* valid_row_mask,
                                                                unsigned int* histograms,
                                                                int histogram_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    return;
  }

  const float clamped = fminf(fmaxf(input[idx], 0.0f), 1.0f);
  const int bin = min(histogram_bins - 1,
                      max(0, __float2int_rn(clamped * static_cast<float>(histogram_bins - 1))));
  atomicAdd(histograms + static_cast<size_t>(batch_index) * static_cast<size_t>(histogram_bins) + static_cast<size_t>(bin), 1U);
}

__global__ void residual_veto_histogram_valid_rows_triplet_batch_kernel(const float* input_a,
                                                                        const float* input_b,
                                                                        const float* input_c,
                                                                        int batch_size,
                                                                        int rows,
                                                                        int cols,
                                                                        const uint8_t* valid_row_mask,
                                                                        unsigned int* histograms,
                                                                        int histogram_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    return;
  }

  const size_t histogram_base = static_cast<size_t>(batch_index) * static_cast<size_t>(kHybridThresholdHistogramMapCount) *
                                static_cast<size_t>(histogram_bins);
  const float clamped_a = fminf(fmaxf(input_a[idx], 0.0f), 1.0f);
  const float clamped_b = fminf(fmaxf(input_b[idx], 0.0f), 1.0f);
  const float clamped_c = fminf(fmaxf(input_c[idx], 0.0f), 1.0f);
  const int bin_a = min(histogram_bins - 1,
                        max(0, __float2int_rn(clamped_a * static_cast<float>(histogram_bins - 1))));
  const int bin_b = min(histogram_bins - 1,
                        max(0, __float2int_rn(clamped_b * static_cast<float>(histogram_bins - 1))));
  const int bin_c = min(histogram_bins - 1,
                        max(0, __float2int_rn(clamped_c * static_cast<float>(histogram_bins - 1))));
  atomicAdd(histograms + histogram_base + static_cast<size_t>(bin_a), 1U);
  atomicAdd(histograms + histogram_base + static_cast<size_t>(histogram_bins) + static_cast<size_t>(bin_b), 1U);
  atomicAdd(histograms + histogram_base + 2U * static_cast<size_t>(histogram_bins) + static_cast<size_t>(bin_c), 1U);
}

__global__ void residual_veto_histogram_valid_rows_sampled_batch_kernel(const float* input,
                                                                        int batch_size,
                                                                        int rows,
                                                                        int cols,
                                                                        const uint8_t* valid_row_mask,
                                                                        int row_stride,
                                                                        int col_stride,
                                                                        unsigned int* histograms,
                                                                        int histogram_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0 ||
      (row % row_stride) != 0 ||
      (col % col_stride) != 0) {
    return;
  }

  const float clamped = fminf(fmaxf(input[idx], 0.0f), 1.0f);
  const int bin = min(histogram_bins - 1,
                      max(0, __float2int_rn(clamped * static_cast<float>(histogram_bins - 1))));
  atomicAdd(histograms + static_cast<size_t>(batch_index) * static_cast<size_t>(histogram_bins) + static_cast<size_t>(bin), 1U);
}

__global__ void residual_veto_histogram_valid_rows_sampled_triplet_batch_kernel(const float* input_a,
                                                                                const float* input_b,
                                                                                const float* input_c,
                                                                                int batch_size,
                                                                                int rows,
                                                                                int cols,
                                                                                const uint8_t* valid_row_mask,
                                                                                int row_stride,
                                                                                int col_stride,
                                                                                unsigned int* histograms,
                                                                                int histogram_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0 ||
      (row % row_stride) != 0 ||
      (col % col_stride) != 0) {
    return;
  }

  const size_t histogram_base = static_cast<size_t>(batch_index) * static_cast<size_t>(kHybridThresholdHistogramMapCount) *
                                static_cast<size_t>(histogram_bins);
  const float clamped_a = fminf(fmaxf(input_a[idx], 0.0f), 1.0f);
  const float clamped_b = fminf(fmaxf(input_b[idx], 0.0f), 1.0f);
  const float clamped_c = fminf(fmaxf(input_c[idx], 0.0f), 1.0f);
  const int bin_a = min(histogram_bins - 1,
                        max(0, __float2int_rn(clamped_a * static_cast<float>(histogram_bins - 1))));
  const int bin_b = min(histogram_bins - 1,
                        max(0, __float2int_rn(clamped_b * static_cast<float>(histogram_bins - 1))));
  const int bin_c = min(histogram_bins - 1,
                        max(0, __float2int_rn(clamped_c * static_cast<float>(histogram_bins - 1))));
  atomicAdd(histograms + histogram_base + static_cast<size_t>(bin_a), 1U);
  atomicAdd(histograms + histogram_base + static_cast<size_t>(histogram_bins) + static_cast<size_t>(bin_b), 1U);
  atomicAdd(histograms + histogram_base + 2U * static_cast<size_t>(histogram_bins) + static_cast<size_t>(bin_c), 1U);
}

__global__ void residual_veto_histogram_fp16_unit_batch_kernel(const float* input,
                                                               int batch_size,
                                                               int plane,
                                                               unsigned int* histograms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const float clamped = fminf(fmaxf(input[idx], 0.0f), 1.0f);
  const unsigned short bin = min(static_cast<unsigned short>(kHybridFp16UnitHistogramBins - 1),
                                 __half_as_ushort(__float2half_rn(clamped)));
  atomicAdd(histograms + static_cast<size_t>(batch_index) * static_cast<size_t>(kHybridFp16UnitHistogramBins) + static_cast<size_t>(bin), 1U);
}

__global__ void residual_veto_histogram_fp16_unit_pair_batch_kernel(const float* input_a,
                                                                    const float* input_b,
                                                                    int batch_size,
                                                                    int plane,
                                                                    unsigned int* histograms) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const size_t histogram_stride = static_cast<size_t>(batch_size) * static_cast<size_t>(kHybridFp16UnitHistogramBins);
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(kHybridFp16UnitHistogramBins);
  const float clamped_a = fminf(fmaxf(input_a[idx], 0.0f), 1.0f);
  const float clamped_b = fminf(fmaxf(input_b[idx], 0.0f), 1.0f);
  const unsigned short bin_a = min(static_cast<unsigned short>(kHybridFp16UnitHistogramBins - 1),
                                   __half_as_ushort(__float2half_rn(clamped_a)));
  const unsigned short bin_b = min(static_cast<unsigned short>(kHybridFp16UnitHistogramBins - 1),
                                   __half_as_ushort(__float2half_rn(clamped_b)));
  atomicAdd(histograms + batch_offset + static_cast<size_t>(bin_a), 1U);
  atomicAdd(histograms + histogram_stride + batch_offset + static_cast<size_t>(bin_b), 1U);
}

__global__ void residual_veto_histogram_quantile_batch_kernel(const unsigned int* histograms,
                                                              int batch_size,
                                                              int histogram_bins,
                                                              float q,
                                                              float fallback,
                                                              float* output) {
  const int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const unsigned int* histogram = histograms + static_cast<size_t>(batch_index) * static_cast<size_t>(histogram_bins);
  uint64_t total_count = 0;
  for (int index = 0; index < histogram_bins; ++index) {
    total_count += histogram[index];
  }
  if (total_count == 0) {
    output[batch_index] = fallback;
    return;
  }

  const float clamped_q = fminf(fmaxf(q, 0.0f), 1.0f);
  const uint64_t target = static_cast<uint64_t>(floorf((static_cast<float>(total_count) - 1.0f) * clamped_q + 0.5f)) + 1ULL;
  uint64_t cumulative = 0;
  int selected_bin = histogram_bins - 1;
  for (int index = 0; index < histogram_bins; ++index) {
    cumulative += histogram[index];
    if (cumulative >= target) {
      selected_bin = index;
      break;
    }
  }
  output[batch_index] = static_cast<float>(selected_bin) / static_cast<float>(max(histogram_bins - 1, 1));
}

__global__ void residual_veto_histogram_quantile_triplet_batch_kernel(const unsigned int* histograms,
                                                                      int batch_size,
                                                                      int histogram_bins,
                                                                      float q_a,
                                                                      float fallback_a,
                                                                      float q_b,
                                                                      float fallback_b,
                                                                      float q_c,
                                                                      float fallback_c,
                                                                      float* output_a,
                                                                      float* output_b,
                                                                      float* output_c) {
  const int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const size_t histogram_base = static_cast<size_t>(batch_index) * static_cast<size_t>(kHybridThresholdHistogramMapCount) *
                                static_cast<size_t>(histogram_bins);
  const unsigned int* histogram_a = histograms + histogram_base;
  const unsigned int* histogram_b = histogram_a + static_cast<size_t>(histogram_bins);
  const unsigned int* histogram_c = histogram_b + static_cast<size_t>(histogram_bins);
  uint64_t total_count = 0;
  for (int index = 0; index < histogram_bins; ++index) {
    total_count += histogram_a[index];
  }
  if (total_count == 0) {
    output_a[batch_index] = fallback_a;
  } else {
    const float clamped_q = fminf(fmaxf(q_a, 0.0f), 1.0f);
    const uint64_t target = static_cast<uint64_t>(floorf((static_cast<float>(total_count) - 1.0f) * clamped_q + 0.5f)) + 1ULL;
    uint64_t cumulative = 0;
    int selected_bin = histogram_bins - 1;
    for (int index = 0; index < histogram_bins; ++index) {
      cumulative += histogram_a[index];
      if (cumulative >= target) {
        selected_bin = index;
        break;
      }
    }
    output_a[batch_index] = static_cast<float>(selected_bin) / static_cast<float>(max(histogram_bins - 1, 1));
  }

  total_count = 0;
  for (int index = 0; index < histogram_bins; ++index) {
    total_count += histogram_b[index];
  }
  if (total_count == 0) {
    output_b[batch_index] = fallback_b;
  } else {
    const float clamped_q = fminf(fmaxf(q_b, 0.0f), 1.0f);
    const uint64_t target = static_cast<uint64_t>(floorf((static_cast<float>(total_count) - 1.0f) * clamped_q + 0.5f)) + 1ULL;
    uint64_t cumulative = 0;
    int selected_bin = histogram_bins - 1;
    for (int index = 0; index < histogram_bins; ++index) {
      cumulative += histogram_b[index];
      if (cumulative >= target) {
        selected_bin = index;
        break;
      }
    }
    output_b[batch_index] = static_cast<float>(selected_bin) / static_cast<float>(max(histogram_bins - 1, 1));
  }

  total_count = 0;
  for (int index = 0; index < histogram_bins; ++index) {
    total_count += histogram_c[index];
  }
  if (total_count == 0) {
    output_c[batch_index] = fallback_c;
    return;
  }

  const float clamped_q = fminf(fmaxf(q_c, 0.0f), 1.0f);
  const uint64_t target = static_cast<uint64_t>(floorf((static_cast<float>(total_count) - 1.0f) * clamped_q + 0.5f)) + 1ULL;
  uint64_t cumulative = 0;
  int selected_bin = histogram_bins - 1;
  for (int index = 0; index < histogram_bins; ++index) {
    cumulative += histogram_c[index];
    if (cumulative >= target) {
      selected_bin = index;
      break;
    }
  }
  output_c[batch_index] = static_cast<float>(selected_bin) / static_cast<float>(max(histogram_bins - 1, 1));
}

__global__ void residual_veto_histogram_fp16_unit_quantiles_batch_kernel(const unsigned int* histograms,
                                                                         int batch_size,
                                                                         float low_q,
                                                                         float high_q,
                                                                         float* low_values,
                                                                         float* high_values) {
  const int batch_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_index >= batch_size) {
    return;
  }

  const unsigned int* histogram = histograms + static_cast<size_t>(batch_index) * static_cast<size_t>(kHybridFp16UnitHistogramBins);
  uint64_t total_count = 0;
  for (int index = 0; index < kHybridFp16UnitHistogramBins; ++index) {
    total_count += histogram[index];
  }
  if (total_count == 0) {
    low_values[batch_index] = 0.0f;
    high_values[batch_index] = 1.0f;
    return;
  }

  const float clamped_low_q = fminf(fmaxf(low_q, 0.0f), 1.0f);
  const float clamped_high_q = fminf(fmaxf(high_q, 0.0f), 1.0f);
  const uint64_t low_target = static_cast<uint64_t>(floorf((static_cast<float>(total_count) - 1.0f) * clamped_low_q + 0.5f)) + 1ULL;
  const uint64_t high_target = static_cast<uint64_t>(floorf((static_cast<float>(total_count) - 1.0f) * clamped_high_q + 0.5f)) + 1ULL;

  uint64_t cumulative = 0;
  int low_bin = 0;
  int high_bin = kHybridFp16UnitHistogramBins - 1;
  bool low_found = false;
  for (int index = 0; index < kHybridFp16UnitHistogramBins; ++index) {
    cumulative += histogram[index];
    if (!low_found && cumulative >= low_target) {
      low_bin = index;
      low_found = true;
    }
    if (cumulative >= high_target) {
      high_bin = index;
      break;
    }
  }

  low_values[batch_index] = __half2float(__ushort_as_half(static_cast<unsigned short>(low_bin)));
  high_values[batch_index] = __half2float(__ushort_as_half(static_cast<unsigned short>(high_bin)));
}

__global__ void residual_veto_final_mask_batch_kernel(const float* keep_freq,
                                                      const float* keep_res,
                                                      const float* combined_score,
                                                      int batch_size,
                                                      int rows,
                                                      int cols,
                                                      const uint8_t* valid_row_mask,
                                                      const float* seed_freq_thresholds,
                                                      const float* seed_res_thresholds,
                                                      const float* combined_thresholds,
                                                      uint8_t* output_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    output_mask[idx] = 0;
    return;
  }

  const bool seed = keep_freq[idx] >= seed_freq_thresholds[batch_index] && keep_res[idx] >= seed_res_thresholds[batch_index];
  const bool keep = combined_score[idx] >= combined_thresholds[batch_index] * 0.85f;
  output_mask[idx] = (seed && keep) ? 1 : 0;
}

__global__ void residual_veto_threshold_mask_batch_kernel(const float* input,
                                                          int total,
                                                          float threshold,
                                                          uint8_t* output_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output_mask[idx] = input[idx] >= threshold ? 1 : 0;
}

__global__ void residual_veto_apply_valid_rows_mask_u8_batch_kernel(uint8_t* mask,
                                                                    int batch_size,
                                                                    int rows,
                                                                    int cols,
                                                                    const uint8_t* valid_row_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  if (valid_row_mask[static_cast<size_t>(batch_index) * static_cast<size_t>(rows) + static_cast<size_t>(row)] == 0) {
    mask[idx] = 0;
  }
}

__global__ void residual_veto_uint8_to_float_batch_kernel(const uint8_t* input, int total, float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = input[idx] != 0 ? 1.0f : 0.0f;
}

bool copy_or_round_fp16_cuda_batch_to_device(const float* input_device,
                                             size_t total_values,
                                             bool use_fp16,
                                             float* output_device,
                                             cudaStream_t stream) {
  if (!use_fp16) {
    return cudaMemcpyAsync(output_device,
                           input_device,
                           total_values * sizeof(float),
                           cudaMemcpyDeviceToDevice,
                           stream) == cudaSuccess;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_round_fp16_copy_kernel<<<blocks, threads, 0, stream>>>(input_device,
                                                                        static_cast<int>(total_values),
                                                                        output_device);
  return cudaGetLastError() == cudaSuccess;
}

bool maybe_round_fp16_cuda_batch_inplace(float* values_device,
                                         size_t total_values,
                                         bool use_fp16,
                                         cudaStream_t stream) {
  if (!use_fp16) {
    return true;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_round_fp16_inplace_kernel<<<blocks, threads, 0, stream>>>(values_device,
                                                                           static_cast<int>(total_values));
  return cudaGetLastError() == cudaSuccess;
}

bool compute_exact_quantile_bounds_cuda_batch_to_device(const float* input_batch_device,
                                                        int batch_size,
                                                        size_t plane,
                                                        float low_q,
                                                        float high_q,
                                                        float* temp_plane_device,
                                                        float* low_values_device,
                                                        float* high_values_device,
                                                        cudaStream_t stream) {
  if (input_batch_device == nullptr || temp_plane_device == nullptr || low_values_device == nullptr || high_values_device == nullptr ||
      batch_size <= 0 || plane == 0) {
    return false;
  }

  const double clamped_low_q = std::clamp(static_cast<double>(low_q), 0.0, 1.0);
  const double clamped_high_q = std::clamp(static_cast<double>(high_q), 0.0, 1.0);
  const auto low_rank = static_cast<size_t>(plane <= 1 ? 0 : std::llround(clamped_low_q * static_cast<double>(plane - 1)));
  const auto high_rank = static_cast<size_t>(plane <= 1 ? 0 : std::llround(clamped_high_q * static_cast<double>(plane - 1)));
  auto temp_begin = thrust::device_pointer_cast(temp_plane_device);
  auto temp_end = temp_begin + static_cast<std::ptrdiff_t>(plane);

  for (int batch_index = 0; batch_index < batch_size; ++batch_index) {
    const float* input_sample = input_batch_device + static_cast<size_t>(batch_index) * plane;
    if (cudaMemcpyAsync(temp_plane_device,
                        input_sample,
                        plane * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
      return false;
    }

    if (plane > 1) {
      thrust::sort(thrust::cuda::par.on(stream), temp_begin, temp_end);
    }

    if (cudaMemcpyAsync(low_values_device + batch_index,
                        temp_plane_device + static_cast<std::ptrdiff_t>(low_rank),
                        sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess ||
        cudaMemcpyAsync(high_values_device + batch_index,
                        temp_plane_device + static_cast<std::ptrdiff_t>(high_rank),
                        sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
      return false;
    }
  }

  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_quantile_exact_cuda_batch_to_device(const float* input_batch_device,
                                                         int batch_size,
                                                         int rows,
                                                         int cols,
                                                         float low_q,
                                                         float high_q,
                                                         float* temp_plane_device,
                                                         float* low_values_device,
                                                         float* high_values_device,
                                                         float* output_batch_device,
                                                         cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  if (!compute_exact_quantile_bounds_cuda_batch_to_device(input_batch_device,
                                                          batch_size,
                                                          plane,
                                                          low_q,
                                                          high_q,
                                                          temp_plane_device,
                                                          low_values_device,
                                                          high_values_device,
                                                          stream)) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_normalize_quantile_batch_kernel<<<blocks, threads, 0, stream>>>(input_batch_device,
                                                                                 batch_size,
                                                                                 rows,
                                                                                 cols,
                                                                                 low_values_device,
                                                                                 high_values_device,
                                                                                 output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_quantile_exact_pair_multiply_cuda_batch_to_device(const float* input_a_batch_device,
                                                                       const float* input_b_batch_device,
                                                                       int batch_size,
                                                                       int rows,
                                                                       int cols,
                                                                       float low_q_a,
                                                                       float high_q_a,
                                                                       float low_q_b,
                                                                       float high_q_b,
                                                                       float* temp_plane_device,
                                                                       float* low_values_a_device,
                                                                       float* high_values_a_device,
                                                                       float* low_values_b_device,
                                                                       float* high_values_b_device,
                                                                       float* output_batch_device,
                                                                       cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  if (!compute_exact_quantile_bounds_cuda_batch_to_device(input_a_batch_device,
                                                          batch_size,
                                                          plane,
                                                          low_q_a,
                                                          high_q_a,
                                                          temp_plane_device,
                                                          low_values_a_device,
                                                          high_values_a_device,
                                                          stream) ||
      !compute_exact_quantile_bounds_cuda_batch_to_device(input_b_batch_device,
                                                          batch_size,
                                                          plane,
                                                          low_q_b,
                                                          high_q_b,
                                                          temp_plane_device,
                                                          low_values_b_device,
                                                          high_values_b_device,
                                                          stream)) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_normalize_quantile_pair_multiply_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                                input_b_batch_device,
                                                                                                batch_size,
                                                                                                rows,
                                                                                                cols,
                                                                                                low_values_a_device,
                                                                                                high_values_a_device,
                                                                                                low_values_b_device,
                                                                                                high_values_b_device,
                                                                                                output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_masked_minmax_cuda_batch_to_device(const float* input_batch_device,
                                                        int batch_size,
                                                        int rows,
                                                        int cols,
                                                        const uint8_t* valid_row_mask_device,
                                                        float* min_values_device,
                                                        float* max_values_device,
                                                        float* partial_min_values_device,
                                                        float* partial_max_values_device,
                                                        int* partial_valid_values_device,
                                                        float* output_batch_device,
                                                        cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  const int partial_count = std::max(1, std::min(kHybridReductionTileBlocks,
                                                 static_cast<int>((plane + static_cast<size_t>(kHybridReductionThreads) - 1) /
                                                                  static_cast<size_t>(kHybridReductionThreads))));
  const dim3 partial_grid(partial_count, batch_size);
  residual_veto_masked_minmax_partial_reduce_batch_kernel<<<partial_grid, kHybridReductionThreads, 0, stream>>>(input_batch_device,
                                                                                                                  batch_size,
                                                                                                                  rows,
                                                                                                                  cols,
                                                                                                                  valid_row_mask_device,
                                                                                                                  partial_min_values_device,
                                                                                                                  partial_max_values_device,
                                                                                                                  partial_valid_values_device,
                                                                                                                  partial_count);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device,
                                                                                                          partial_max_values_device,
                                                                                                          partial_valid_values_device,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_device,
                                                                                                          max_values_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_normalize_masked_minmax_batch_kernel<<<blocks, threads, 0, stream>>>(input_batch_device,
                                                                                      batch_size,
                                                                                      rows,
                                                                                      cols,
                                                                                      valid_row_mask_device,
                                                                                      min_values_device,
                                                                                      max_values_device,
                                                                                      output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_masked_minmax_and_row_filter_bank_cuda_batch_to_device(const float* input_batch_device,
                                                                            int batch_size,
                                                                            int rows,
                                                                            int cols,
                                                                            const uint8_t* valid_row_mask_device,
                                                                            float* min_values_device,
                                                                            float* max_values_device,
                                                                            float* partial_min_values_device,
                                                                            float* partial_max_values_device,
                                                                            int* partial_valid_values_device,
                                                                            float* normalized_output_batch_device,
                                                                            const ResidualVetoKernelBuffer& row_kernel_a,
                                                                            float* output_a,
                                                                            const ResidualVetoKernelBuffer& row_kernel_b,
                                                                            float* output_b,
                                                                            const ResidualVetoKernelBuffer& row_kernel_c,
                                                                            float* output_c,
                                                                            cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const int partial_count = std::max(1, std::min(kHybridReductionTileBlocks,
                                                 static_cast<int>((plane + static_cast<size_t>(kHybridReductionThreads) - 1) /
                                                                  static_cast<size_t>(kHybridReductionThreads))));
  const dim3 partial_grid(partial_count, batch_size);
  residual_veto_masked_minmax_partial_reduce_batch_kernel<<<partial_grid, kHybridReductionThreads, 0, stream>>>(input_batch_device,
                                                                                                                  batch_size,
                                                                                                                  rows,
                                                                                                                  cols,
                                                                                                                  valid_row_mask_device,
                                                                                                                  partial_min_values_device,
                                                                                                                  partial_max_values_device,
                                                                                                                  partial_valid_values_device,
                                                                                                                  partial_count);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device,
                                                                                                          partial_max_values_device,
                                                                                                          partial_valid_values_device,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_device,
                                                                                                          max_values_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  constexpr int kLineThreads = 128;
  const int max_radius = std::max(row_kernel_a.radius, std::max(row_kernel_b.radius, row_kernel_c.radius));
  const dim3 row_grid((rows + kLineThreads - 1) / kLineThreads, batch_size * cols);
  const size_t row_shared_bytes = static_cast<size_t>(kLineThreads + 2 * max_radius) * sizeof(float);
  residual_veto_normalize_masked_minmax_row_filter_bank_batch_kernel<<<row_grid, kLineThreads, row_shared_bytes, stream>>>(input_batch_device,
                                                                                                                              batch_size,
                                                                                                                              rows,
                                                                                                                              cols,
                                                                                                                              valid_row_mask_device,
                                                                                                                              min_values_device,
                                                                                                                              max_values_device,
                                                                                                                              normalized_output_batch_device,
                                                                                                                              row_kernel_a.values,
                                                                                                                              row_kernel_a.radius,
                                                                                                                              output_a,
                                                                                                                              row_kernel_b.values,
                                                                                                                              row_kernel_b.radius,
                                                                                                                              output_b,
                                                                                                                              row_kernel_c.values,
                                                                                                                              row_kernel_c.radius,
                                                                                                                              output_c,
                                                                                                                              max_radius);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_masked_minmax_triplet_cuda_batch_to_device(const float* input_a_batch_device,
                                                                const float* input_b_batch_device,
                                                                const float* input_c_batch_device,
                                                                int batch_size,
                                                                int rows,
                                                                int cols,
                                                                const uint8_t* valid_row_mask_device,
                                                                float* min_values_a_device,
                                                                float* max_values_a_device,
                                                                float* min_values_b_device,
                                                                float* max_values_b_device,
                                                                float* min_values_c_device,
                                                                float* max_values_c_device,
                                                                float* partial_min_values_device,
                                                                float* partial_max_values_device,
                                                                int* partial_valid_values_device,
                                                                float* output_a_batch_device,
                                                                float* output_b_batch_device,
                                                                float* output_c_batch_device,
                                                                cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  const int partial_count = std::max(1, std::min(kHybridReductionTileBlocks,
                                                 static_cast<int>((plane + static_cast<size_t>(kHybridReductionThreads) - 1) /
                                                                  static_cast<size_t>(kHybridReductionThreads))));
  const dim3 partial_grid(partial_count, batch_size);
  residual_veto_masked_minmax_triplet_partial_reduce_batch_kernel<<<partial_grid, kHybridReductionThreads, 0, stream>>>(input_a_batch_device,
                                                                                                                         input_b_batch_device,
                                                                                                                         input_c_batch_device,
                                                                                                                         batch_size,
                                                                                                                         rows,
                                                                                                                         cols,
                                                                                                                         valid_row_mask_device,
                                                                                                                         partial_min_values_device,
                                                                                                                         partial_max_values_device,
                                                                                                                         partial_valid_values_device,
                                                                                                                         partial_count);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const size_t partial_map_stride = static_cast<size_t>(batch_size) * static_cast<size_t>(partial_count);
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device,
                                                                                                          partial_max_values_device,
                                                                                                          partial_valid_values_device,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_a_device,
                                                                                                          max_values_a_device);
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device + partial_map_stride,
                                                                                                          partial_max_values_device + partial_map_stride,
                                                                                                          partial_valid_values_device + partial_map_stride,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_b_device,
                                                                                                          max_values_b_device);
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device + 2 * partial_map_stride,
                                                                                                          partial_max_values_device + 2 * partial_map_stride,
                                                                                                          partial_valid_values_device + 2 * partial_map_stride,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_c_device,
                                                                                                          max_values_c_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_normalize_masked_minmax_triplet_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                               input_b_batch_device,
                                                                                               input_c_batch_device,
                                                                                               batch_size,
                                                                                               rows,
                                                                                               cols,
                                                                                               valid_row_mask_device,
                                                                                               min_values_a_device,
                                                                                               max_values_a_device,
                                                                                               min_values_b_device,
                                                                                               max_values_b_device,
                                                                                               min_values_c_device,
                                                                                               max_values_c_device,
                                                                                               output_a_batch_device,
                                                                                               output_b_batch_device,
                                                                                               output_c_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_subtract_scaled_masked_minmax_cuda_batch_to_device(const float* lhs_batch_device,
                                                                  const float* rhs_batch_device,
                                                                  int batch_size,
                                                                  int rows,
                                                                  int cols,
                                                                  const uint8_t* valid_row_mask_device,
                                                                  float rhs_scale,
                                                                  float* min_values_device,
                                                                  float* max_values_device,
                                                                  float* partial_min_values_device,
                                                                  float* partial_max_values_device,
                                                                  int* partial_valid_values_device,
                                                                  float* output_batch_device,
                                                                  cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  const int partial_count = std::max(1, std::min(kHybridReductionTileBlocks,
                                                 static_cast<int>((plane + static_cast<size_t>(kHybridReductionThreads) - 1) /
                                                                  static_cast<size_t>(kHybridReductionThreads))));
  const dim3 partial_grid(partial_count, batch_size);
  residual_veto_subtract_scaled_masked_minmax_partial_reduce_batch_kernel<<<partial_grid, kHybridReductionThreads, 0, stream>>>(lhs_batch_device,
                                                                                                                                rhs_batch_device,
                                                                                                                                batch_size,
                                                                                                                                rows,
                                                                                                                                cols,
                                                                                                                                valid_row_mask_device,
                                                                                                                                rhs_scale,
                                                                                                                                partial_min_values_device,
                                                                                                                                partial_max_values_device,
                                                                                                                                partial_valid_values_device,
                                                                                                                                partial_count);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device,
                                                                                                          partial_max_values_device,
                                                                                                          partial_valid_values_device,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_device,
                                                                                                          max_values_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_subtract_scaled_normalize_masked_minmax_batch_kernel<<<blocks, threads, 0, stream>>>(lhs_batch_device,
                                                                                                      rhs_batch_device,
                                                                                                      batch_size,
                                                                                                      rows,
                                                                                                      cols,
                                                                                                      valid_row_mask_device,
                                                                                                      min_values_device,
                                                                                                      max_values_device,
                                                                                                      rhs_scale,
                                                                                                      output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_subtract_scaled_masked_minmax_pair_cuda_batch_to_device(const float* lhs_batch_device,
                                                                       const float* rhs_a_batch_device,
                                                                       float rhs_a_scale,
                                                                       const float* rhs_b_batch_device,
                                                                       float rhs_b_scale,
                                                                       int batch_size,
                                                                       int rows,
                                                                       int cols,
                                                                       const uint8_t* valid_row_mask_device,
                                                                       float* min_values_a_device,
                                                                       float* max_values_a_device,
                                                                       float* min_values_b_device,
                                                                       float* max_values_b_device,
                                                                       float* partial_min_values_device,
                                                                       float* partial_max_values_device,
                                                                       int* partial_valid_values_device,
                                                                       float* output_a_batch_device,
                                                                       float* output_b_batch_device,
                                                                       cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  const int partial_count = std::max(1, std::min(kHybridReductionTileBlocks,
                                                 static_cast<int>((plane + static_cast<size_t>(kHybridReductionThreads) - 1) /
                                                                  static_cast<size_t>(kHybridReductionThreads))));
  const dim3 partial_grid(partial_count, batch_size);
  residual_veto_subtract_scaled_masked_minmax_pair_partial_reduce_batch_kernel<<<partial_grid, kHybridReductionThreads, 0, stream>>>(lhs_batch_device,
                                                                                                                                       rhs_a_batch_device,
                                                                                                                                       rhs_b_batch_device,
                                                                                                                                       batch_size,
                                                                                                                                       rows,
                                                                                                                                       cols,
                                                                                                                                       valid_row_mask_device,
                                                                                                                                       rhs_a_scale,
                                                                                                                                       rhs_b_scale,
                                                                                                                                       partial_min_values_device,
                                                                                                                                       partial_max_values_device,
                                                                                                                                       partial_valid_values_device,
                                                                                                                                       partial_count);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const size_t partial_map_stride = static_cast<size_t>(batch_size) * static_cast<size_t>(partial_count);
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device,
                                                                                                          partial_max_values_device,
                                                                                                          partial_valid_values_device,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_a_device,
                                                                                                          max_values_a_device);
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device + partial_map_stride,
                                                                                                          partial_max_values_device + partial_map_stride,
                                                                                                          partial_valid_values_device + partial_map_stride,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_b_device,
                                                                                                          max_values_b_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_subtract_scaled_normalize_masked_minmax_pair_batch_kernel<<<blocks, threads, 0, stream>>>(lhs_batch_device,
                                                                                                            rhs_a_batch_device,
                                                                                                            rhs_b_batch_device,
                                                                                                            batch_size,
                                                                                                            rows,
                                                                                                            cols,
                                                                                                            valid_row_mask_device,
                                                                                                            min_values_a_device,
                                                                                                            max_values_a_device,
                                                                                                            min_values_b_device,
                                                                                                            max_values_b_device,
                                                                                                            rhs_a_scale,
                                                                                                            rhs_b_scale,
                                                                                                            output_a_batch_device,
                                                                                                            output_b_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_combined_input_masked_minmax_cuda_batch_to_device(const float* keep_freq_batch_device,
                                                                 const float* keep_res_batch_device,
                                                                 int batch_size,
                                                                 int rows,
                                                                 int cols,
                                                                 const uint8_t* valid_row_mask_device,
                                                                 float* min_values_device,
                                                                 float* max_values_device,
                                                                 float* partial_min_values_device,
                                                                 float* partial_max_values_device,
                                                                 int* partial_valid_values_device,
                                                                 float* output_batch_device,
                                                                 cudaStream_t stream) {
  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  const int partial_count = std::max(1, std::min(kHybridReductionTileBlocks,
                                                 static_cast<int>((plane + static_cast<size_t>(kHybridReductionThreads) - 1) /
                                                                  static_cast<size_t>(kHybridReductionThreads))));
  const dim3 partial_grid(partial_count, batch_size);
  residual_veto_combined_input_masked_minmax_partial_reduce_batch_kernel<<<partial_grid, kHybridReductionThreads, 0, stream>>>(keep_freq_batch_device,
                                                                                                                               keep_res_batch_device,
                                                                                                                               batch_size,
                                                                                                                               rows,
                                                                                                                               cols,
                                                                                                                               valid_row_mask_device,
                                                                                                                               partial_min_values_device,
                                                                                                                               partial_max_values_device,
                                                                                                                               partial_valid_values_device,
                                                                                                                               partial_count);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  residual_veto_masked_minmax_finalize_batch_kernel<<<batch_size, kHybridReductionThreads, 0, stream>>>(partial_min_values_device,
                                                                                                          partial_max_values_device,
                                                                                                          partial_valid_values_device,
                                                                                                          batch_size,
                                                                                                          partial_count,
                                                                                                          min_values_device,
                                                                                                          max_values_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_combined_input_normalize_masked_minmax_batch_kernel<<<blocks, threads, 0, stream>>>(keep_freq_batch_device,
                                                                                                     keep_res_batch_device,
                                                                                                     batch_size,
                                                                                                     rows,
                                                                                                     cols,
                                                                                                     valid_row_mask_device,
                                                                                                     min_values_device,
                                                                                                     max_values_device,
                                                                                                     output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool separable_convolve_cuda_batch_to_device(const float* input_batch_device,
                                             int batch_size,
                                             int rows,
                                             int cols,
                                             const ResidualVetoKernelBuffer& row_kernel,
                                             const ResidualVetoKernelBuffer& col_kernel,
                                             float* temp_batch_device,
                                             float* output_batch_device,
                                             cudaStream_t stream) {
  constexpr int kLineThreads = 128;
  const dim3 row_grid((rows + kLineThreads - 1) / kLineThreads, batch_size * cols);
  const size_t row_shared_bytes = static_cast<size_t>(kLineThreads + 2 * row_kernel.radius) * sizeof(float);
  residual_veto_convolve_rows_batch_kernel<<<row_grid, kLineThreads, row_shared_bytes, stream>>>(input_batch_device,
                                                                                                  batch_size,
                                                                                                  rows,
                                                                                                  cols,
                                                                                                  row_kernel.values,
                                                                                                  row_kernel.radius,
                                                                                                  temp_batch_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  const dim3 col_grid((cols + kLineThreads - 1) / kLineThreads, batch_size * rows);
  const size_t col_shared_bytes = static_cast<size_t>(kLineThreads + 2 * col_kernel.radius) * sizeof(float);
  residual_veto_convolve_cols_batch_kernel<<<col_grid, kLineThreads, col_shared_bytes, stream>>>(temp_batch_device,
                                                                                                  batch_size,
                                                                                                  rows,
                                                                                                  cols,
                                                                                                  col_kernel.values,
                                                                                                  col_kernel.radius,
                                                                                                  output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool row_convolve_cuda_batch_to_device(const float* input_batch_device,
                                       int batch_size,
                                       int rows,
                                       int cols,
                                       const ResidualVetoKernelBuffer& row_kernel,
                                       float* output_batch_device,
                                       cudaStream_t stream) {
  constexpr int kLineThreads = 128;
  const dim3 row_grid((rows + kLineThreads - 1) / kLineThreads, batch_size * cols);
  const size_t row_shared_bytes = static_cast<size_t>(kLineThreads + 2 * row_kernel.radius) * sizeof(float);
  residual_veto_convolve_rows_batch_kernel<<<row_grid, kLineThreads, row_shared_bytes, stream>>>(input_batch_device,
                                                                                                  batch_size,
                                                                                                  rows,
                                                                                                  cols,
                                                                                                  row_kernel.values,
                                                                                                  row_kernel.radius,
                                                                                                  output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool col_convolve_cuda_batch_to_device(const float* input_batch_device,
                                       int batch_size,
                                       int rows,
                                       int cols,
                                       const ResidualVetoKernelBuffer& col_kernel,
                                       float* output_batch_device,
                                       cudaStream_t stream) {
  constexpr int kLineThreads = 128;
  const dim3 col_grid((cols + kLineThreads - 1) / kLineThreads, batch_size * rows);
  const size_t col_shared_bytes = static_cast<size_t>(kLineThreads + 2 * col_kernel.radius) * sizeof(float);
  residual_veto_convolve_cols_batch_kernel<<<col_grid, kLineThreads, col_shared_bytes, stream>>>(input_batch_device,
                                                                                                  batch_size,
                                                                                                  rows,
                                                                                                  cols,
                                                                                                  col_kernel.values,
                                                                                                  col_kernel.radius,
                                                                                                  output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool dual_col_convolve_cuda_batch_to_device(const float* input_a_batch_device,
                                            const float* input_b_batch_device,
                                            int batch_size,
                                            int rows,
                                            int cols,
                                            const ResidualVetoKernelBuffer& col_kernel_a,
                                            float* output_a_batch_device,
                                            const ResidualVetoKernelBuffer& col_kernel_b,
                                            float* output_b_batch_device,
                                            cudaStream_t stream) {
  constexpr int kLineThreads = 128;
  const int max_radius = std::max(col_kernel_a.radius, col_kernel_b.radius);
  const dim3 col_grid((cols + kLineThreads - 1) / kLineThreads, batch_size * rows);
  const size_t col_shared_bytes = 2 * static_cast<size_t>(kLineThreads + 2 * max_radius) * sizeof(float);
  residual_veto_dual_convolve_cols_batch_kernel<<<col_grid, kLineThreads, col_shared_bytes, stream>>>(input_a_batch_device,
                                                                                                       input_b_batch_device,
                                                                                                       batch_size,
                                                                                                       rows,
                                                                                                       cols,
                                                                                                       col_kernel_a.values,
                                                                                                       col_kernel_a.radius,
                                                                                                       output_a_batch_device,
                                                                                                       col_kernel_b.values,
                                                                                                       col_kernel_b.radius,
                                                                                                       output_b_batch_device,
                                                                                                       max_radius);
  return cudaGetLastError() == cudaSuccess;
}

bool row_filter_bank_cuda_batch_to_device(const float* input_batch_device,
                                          int batch_size,
                                          int rows,
                                          int cols,
                                          const ResidualVetoKernelBuffer& row_kernel_a,
                                          float* output_a,
                                          const ResidualVetoKernelBuffer& row_kernel_b,
                                          float* output_b,
                                          const ResidualVetoKernelBuffer& row_kernel_c,
                                          float* output_c,
                                          cudaStream_t stream) {
  constexpr int kLineThreads = 128;
  const int max_radius = std::max(row_kernel_a.radius, std::max(row_kernel_b.radius, row_kernel_c.radius));
  const dim3 row_grid((rows + kLineThreads - 1) / kLineThreads, batch_size * cols);
  const size_t row_shared_bytes = static_cast<size_t>(kLineThreads + 2 * max_radius) * sizeof(float);
  residual_veto_convolve_row_filter_bank_batch_kernel<<<row_grid, kLineThreads, row_shared_bytes, stream>>>(input_batch_device,
                                                                                                              batch_size,
                                                                                                              rows,
                                                                                                              cols,
                                                                                                              row_kernel_a.values,
                                                                                                              row_kernel_a.radius,
                                                                                                              output_a,
                                                                                                              row_kernel_b.values,
                                                                                                              row_kernel_b.radius,
                                                                                                              output_b,
                                                                                                              row_kernel_c.values,
                                                                                                              row_kernel_c.radius,
                                                                                                              output_c,
                                                                                                              max_radius);
  return cudaGetLastError() == cudaSuccess;
}

bool masked_histogram_quantile_cuda_batch_to_device(const float* input_batch_device,
                                                    int batch_size,
                                                    int rows,
                                                    int cols,
                                                    const uint8_t* valid_row_mask_device,
                                                    float q,
                                                    float fallback,
                                                    bool use_sampled_histogram,
                                                    unsigned int* histograms_device,
                                                    float* output_thresholds_device,
                                                    cudaStream_t stream) {
  const size_t histogram_values = static_cast<size_t>(batch_size) * static_cast<size_t>(kHybridThresholdHistogramBins);
  if (cudaMemsetAsync(histograms_device, 0, histogram_values * sizeof(unsigned int), stream) != cudaSuccess) {
    return false;
  }

  const size_t total_values = static_cast<size_t>(batch_size) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  if (use_sampled_histogram) {
    residual_veto_histogram_valid_rows_sampled_batch_kernel<<<blocks, threads, 0, stream>>>(input_batch_device,
                                                                                             batch_size,
                                                                                             rows,
                                                                                             cols,
                                                                                             valid_row_mask_device,
                                                                                             kHybridThresholdHistogramSampleStride,
                                                                                             kHybridThresholdHistogramSampleStride,
                                                                                             histograms_device,
                                                                                             kHybridThresholdHistogramBins);
  } else {
    residual_veto_histogram_valid_rows_batch_kernel<<<blocks, threads, 0, stream>>>(input_batch_device,
                                                                                     batch_size,
                                                                                     rows,
                                                                                     cols,
                                                                                     valid_row_mask_device,
                                                                                     histograms_device,
                                                                                     kHybridThresholdHistogramBins);
  }
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int quantile_blocks = (batch_size + threads - 1) / threads;
  residual_veto_histogram_quantile_batch_kernel<<<quantile_blocks, threads, 0, stream>>>(histograms_device,
                                                                                          batch_size,
                                                                                          kHybridThresholdHistogramBins,
                                                                                          q,
                                                                                          fallback,
                                                                                          output_thresholds_device);
  return cudaGetLastError() == cudaSuccess;
}

bool masked_histogram_quantile_triplet_cuda_batch_to_device(const float* input_a_batch_device,
                                                            const float* input_b_batch_device,
                                                            const float* input_c_batch_device,
                                                            int batch_size,
                                                            int rows,
                                                            int cols,
                                                            const uint8_t* valid_row_mask_device,
                                                            float q_a,
                                                            float fallback_a,
                                                            float q_b,
                                                            float fallback_b,
                                                            float q_c,
                                                            float fallback_c,
                                                            bool use_sampled_histogram,
                                                            unsigned int* histograms_device,
                                                            float* output_a_device,
                                                            float* output_b_device,
                                                            float* output_c_device,
                                                            cudaStream_t stream) {
  if (input_a_batch_device == nullptr || input_b_batch_device == nullptr || input_c_batch_device == nullptr ||
      valid_row_mask_device == nullptr || histograms_device == nullptr || output_a_device == nullptr || output_b_device == nullptr ||
      output_c_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t histogram_values = static_cast<size_t>(batch_size) * static_cast<size_t>(kHybridThresholdHistogramMapCount) *
                                  static_cast<size_t>(kHybridThresholdHistogramBins);
  if (cudaMemsetAsync(histograms_device, 0, histogram_values * sizeof(unsigned int), stream) != cudaSuccess) {
    return false;
  }

  const size_t total_values = static_cast<size_t>(batch_size) * static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  if (use_sampled_histogram) {
    residual_veto_histogram_valid_rows_sampled_triplet_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                                     input_b_batch_device,
                                                                                                     input_c_batch_device,
                                                                                                     batch_size,
                                                                                                     rows,
                                                                                                     cols,
                                                                                                     valid_row_mask_device,
                                                                                                     kHybridThresholdHistogramSampleStride,
                                                                                                     kHybridThresholdHistogramSampleStride,
                                                                                                     histograms_device,
                                                                                                     kHybridThresholdHistogramBins);
  } else {
    residual_veto_histogram_valid_rows_triplet_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                             input_b_batch_device,
                                                                                             input_c_batch_device,
                                                                                             batch_size,
                                                                                             rows,
                                                                                             cols,
                                                                                             valid_row_mask_device,
                                                                                             histograms_device,
                                                                                             kHybridThresholdHistogramBins);
  }
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int quantile_blocks = (batch_size + threads - 1) / threads;
  residual_veto_histogram_quantile_triplet_batch_kernel<<<quantile_blocks, threads, 0, stream>>>(histograms_device,
                                                                                                   batch_size,
                                                                                                   kHybridThresholdHistogramBins,
                                                                                                   q_a,
                                                                                                   fallback_a,
                                                                                                   q_b,
                                                                                                   fallback_b,
                                                                                                   q_c,
                                                                                                   fallback_c,
                                                                                                   output_a_device,
                                                                                                   output_b_device,
                                                                                                   output_c_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_quantile_exact_fp16_unit_cuda_batch_to_device(const float* input_batch_device,
                                                                   int batch_size,
                                                                   int rows,
                                                                   int cols,
                                                                   float low_q,
                                                                   float high_q,
                                                                   unsigned int* histograms_device,
                                                                   float* low_values_device,
                                                                   float* high_values_device,
                                                                   float* output_batch_device,
                                                                   cudaStream_t stream) {
  if (input_batch_device == nullptr || histograms_device == nullptr || low_values_device == nullptr || high_values_device == nullptr ||
      output_batch_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t histogram_values = static_cast<size_t>(batch_size) * static_cast<size_t>(kHybridFp16UnitHistogramBins);
  if (cudaMemsetAsync(histograms_device, 0, histogram_values * sizeof(unsigned int), stream) != cudaSuccess) {
    return false;
  }

  const int plane = rows * cols;
  const size_t total_values = static_cast<size_t>(batch_size) * static_cast<size_t>(plane);
  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_histogram_fp16_unit_batch_kernel<<<blocks, threads, 0, stream>>>(input_batch_device,
                                                                                  batch_size,
                                                                                  plane,
                                                                                  histograms_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int quantile_blocks = (batch_size + threads - 1) / threads;
  residual_veto_histogram_fp16_unit_quantiles_batch_kernel<<<quantile_blocks, threads, 0, stream>>>(histograms_device,
                                                                                                     batch_size,
                                                                                                     low_q,
                                                                                                     high_q,
                                                                                                     low_values_device,
                                                                                                     high_values_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  residual_veto_normalize_quantile_batch_kernel<<<blocks, threads, 0, stream>>>(input_batch_device,
                                                                                 batch_size,
                                                                                 rows,
                                                                                 cols,
                                                                                 low_values_device,
                                                                                 high_values_device,
                                                                                 output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_quantile_exact_fp16_unit_pair_cuda_batch_to_device(const float* input_a_batch_device,
                                                                        const float* input_b_batch_device,
                                                                        int batch_size,
                                                                        int rows,
                                                                        int cols,
                                                                        float low_q_a,
                                                                        float high_q_a,
                                                                        float low_q_b,
                                                                        float high_q_b,
                                                                        unsigned int* histograms_device,
                                                                        float* low_values_a_device,
                                                                        float* high_values_a_device,
                                                                        float* low_values_b_device,
                                                                        float* high_values_b_device,
                                                                        float* output_a_batch_device,
                                                                        float* output_b_batch_device,
                                                                        cudaStream_t stream) {
  if (input_a_batch_device == nullptr || input_b_batch_device == nullptr || histograms_device == nullptr ||
      low_values_a_device == nullptr || high_values_a_device == nullptr || low_values_b_device == nullptr || high_values_b_device == nullptr ||
      output_a_batch_device == nullptr || output_b_batch_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t histogram_stride = static_cast<size_t>(batch_size) * static_cast<size_t>(kHybridFp16UnitHistogramBins);
  const size_t histogram_values = histogram_stride * static_cast<size_t>(kHybridFp16UnitHistogramPairMapCount);
  if (cudaMemsetAsync(histograms_device, 0, histogram_values * sizeof(unsigned int), stream) != cudaSuccess) {
    return false;
  }

  const int plane = rows * cols;
  const size_t total_values = static_cast<size_t>(batch_size) * static_cast<size_t>(plane);
  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_histogram_fp16_unit_pair_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                       input_b_batch_device,
                                                                                       batch_size,
                                                                                       plane,
                                                                                       histograms_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int quantile_blocks = (batch_size + threads - 1) / threads;
  residual_veto_histogram_fp16_unit_quantiles_batch_kernel<<<quantile_blocks, threads, 0, stream>>>(histograms_device,
                                                                                                     batch_size,
                                                                                                     low_q_a,
                                                                                                     high_q_a,
                                                                                                     low_values_a_device,
                                                                                                     high_values_a_device);
  residual_veto_histogram_fp16_unit_quantiles_batch_kernel<<<quantile_blocks, threads, 0, stream>>>(histograms_device + histogram_stride,
                                                                                                     batch_size,
                                                                                                     low_q_b,
                                                                                                     high_q_b,
                                                                                                     low_values_b_device,
                                                                                                     high_values_b_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  residual_veto_normalize_quantile_pair_round_fp16_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                                 input_b_batch_device,
                                                                                                 batch_size,
                                                                                                 rows,
                                                                                                 cols,
                                                                                                 low_values_a_device,
                                                                                                 high_values_a_device,
                                                                                                 low_values_b_device,
                                                                                                 high_values_b_device,
                                                                                                 output_a_batch_device,
                                                                                                 output_b_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

bool normalize_map01_quantile_exact_fp16_unit_pair_multiply_cuda_batch_to_device(const float* input_a_batch_device,
                                                                                  const float* input_b_batch_device,
                                                                                  int batch_size,
                                                                                  int rows,
                                                                                  int cols,
                                                                                  float low_q_a,
                                                                                  float high_q_a,
                                                                                  float low_q_b,
                                                                                  float high_q_b,
                                                                                  unsigned int* histograms_device,
                                                                                  float* low_values_a_device,
                                                                                  float* high_values_a_device,
                                                                                  float* low_values_b_device,
                                                                                  float* high_values_b_device,
                                                                                  float* output_batch_device,
                                                                                  cudaStream_t stream) {
  if (input_a_batch_device == nullptr || input_b_batch_device == nullptr || histograms_device == nullptr ||
      low_values_a_device == nullptr || high_values_a_device == nullptr || low_values_b_device == nullptr || high_values_b_device == nullptr ||
      output_batch_device == nullptr || batch_size <= 0 || rows <= 0 || cols <= 0) {
    return false;
  }

  const size_t histogram_stride = static_cast<size_t>(batch_size) * static_cast<size_t>(kHybridFp16UnitHistogramBins);
  const size_t histogram_values = histogram_stride * static_cast<size_t>(kHybridFp16UnitHistogramPairMapCount);
  if (cudaMemsetAsync(histograms_device, 0, histogram_values * sizeof(unsigned int), stream) != cudaSuccess) {
    return false;
  }

  const int plane = rows * cols;
  const size_t total_values = static_cast<size_t>(batch_size) * static_cast<size_t>(plane);
  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  residual_veto_histogram_fp16_unit_pair_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                       input_b_batch_device,
                                                                                       batch_size,
                                                                                       plane,
                                                                                       histograms_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  const int quantile_blocks = (batch_size + threads - 1) / threads;
  residual_veto_histogram_fp16_unit_quantiles_batch_kernel<<<quantile_blocks, threads, 0, stream>>>(histograms_device,
                                                                                                     batch_size,
                                                                                                     low_q_a,
                                                                                                     high_q_a,
                                                                                                     low_values_a_device,
                                                                                                     high_values_a_device);
  residual_veto_histogram_fp16_unit_quantiles_batch_kernel<<<quantile_blocks, threads, 0, stream>>>(histograms_device + histogram_stride,
                                                                                                     batch_size,
                                                                                                     low_q_b,
                                                                                                     high_q_b,
                                                                                                     low_values_b_device,
                                                                                                     high_values_b_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  residual_veto_normalize_quantile_pair_round_fp16_multiply_batch_kernel<<<blocks, threads, 0, stream>>>(input_a_batch_device,
                                                                                                           input_b_batch_device,
                                                                                                           batch_size,
                                                                                                           rows,
                                                                                                           cols,
                                                                                                           low_values_a_device,
                                                                                                           high_values_a_device,
                                                                                                           low_values_b_device,
                                                                                                           high_values_b_device,
                                                                                                           output_batch_device);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

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
                                                       CudaHybridStageTiming* stage_timing) {
  if (batch_size <= 0 || rows <= 0 || cols <= 0 || dino_score_batch_device == nullptr || coherence_batch_device == nullptr ||
      output_combined_score_device == nullptr ||
      valid_row_mask_batch.size() != static_cast<size_t>(batch_size) * static_cast<size_t>(rows)) {
    return false;
  }

  try {
    const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    const size_t total_values = static_cast<size_t>(batch_size) * plane;
    const size_t total_row_mask = static_cast<size_t>(batch_size) * static_cast<size_t>(rows);
    auto& kernel_cache = residual_veto_kernel_cache();
    auto& scratch = residual_veto_cuda_scratch();
    if (!kernel_cache.ensure_initialized() ||
        !scratch.ensure_capacity(total_values, plane, static_cast<size_t>(batch_size), total_row_mask, total_values)) {
      return false;
    }

    cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
    if (cudaMemcpyAsync(scratch.valid_row_mask,
                        valid_row_mask_batch.data(),
                        total_row_mask * sizeof(uint8_t),
                        cudaMemcpyHostToDevice,
                        stream) != cudaSuccess) {
      return false;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
    auto finish_stage = [&](const std::chrono::steady_clock::time_point& start, double* output_ms) -> bool {
      if (output_ms == nullptr) {
        return true;
      }
      if (cudaStreamSynchronize(stream) != cudaSuccess) {
        return false;
      }
      *output_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      return true;
    };

    const auto normalization_start = std::chrono::steady_clock::now();
    if (!copy_or_round_fp16_cuda_batch_to_device(dino_score_batch_device, total_values, use_fp16, scratch.values_a, stream) ||
        !copy_or_round_fp16_cuda_batch_to_device(coherence_batch_device, total_values, use_fp16, scratch.values_b, stream)) {
      return false;
    }
    const bool normalization_ready = use_fp16
                                         ? normalize_map01_quantile_exact_fp16_unit_pair_multiply_cuda_batch_to_device(scratch.values_a,
                                                                                                                       scratch.values_b,
                                                                                                                       batch_size,
                                                                                                                       rows,
                                                                                                                       cols,
                                                                                                                       0.05f,
                                                                                                                       0.95f,
                                                                                                                       0.05f,
                                                                                                                       0.99f,
                                                                                                                       scratch.histograms,
                                                                                                                       scratch.batch_a,
                                                                                                                       scratch.batch_b,
                                                                                                                       scratch.batch_c,
                                                                                                                       scratch.batch_d,
                                                                                                                       output_combined_score_device,
                                                                                                                       stream)
                                         : normalize_map01_quantile_exact_pair_multiply_cuda_batch_to_device(scratch.values_a,
                                                                                                              scratch.values_b,
                                                                                                              batch_size,
                                                                                                              rows,
                                                                                                              cols,
                                                                                                              0.05f,
                                                                                                              0.95f,
                                                                                                              0.05f,
                                                                                                              0.99f,
                                                                                                              scratch.temp_plane,
                                                                                                              scratch.batch_a,
                                                                                                              scratch.batch_b,
                                                                                                              scratch.batch_c,
                                                                                                              scratch.batch_d,
                                                                                                              output_combined_score_device,
                                                                                                              stream);
    if (!normalization_ready) {
      return false;
    }
    if (cudaGetLastError() != cudaSuccess ||
        !maybe_round_fp16_cuda_batch_inplace(output_combined_score_device, total_values, use_fp16, stream) ||
        !finish_stage(normalization_start, stage_timing != nullptr ? &stage_timing->normalization_ms : nullptr)) {
      return false;
    }

    if (!enable_mask_post_processing) {
      if ((output_final_mask_device != nullptr || output_filled_mask_batch_device != nullptr ||
           output_component_filtered_mask_batch_device != nullptr) &&
          cudaMemcpyAsync(scratch.values_a,
                          output_combined_score_device,
                          total_values * sizeof(float),
                          cudaMemcpyDeviceToDevice,
                          stream) != cudaSuccess) {
        return false;
      }
      if ((output_final_mask_device != nullptr || output_filled_mask_batch_device != nullptr ||
           output_component_filtered_mask_batch_device != nullptr) &&
          cudaMemcpyAsync(scratch.values_d,
                          output_combined_score_device,
                          total_values * sizeof(float),
                          cudaMemcpyDeviceToDevice,
                          stream) != cudaSuccess) {
        return false;
      }

      if (output_final_mask_device != nullptr || output_filled_mask_batch_device != nullptr ||
          output_component_filtered_mask_batch_device != nullptr) {
        std::vector<float> disabled_thresholds(static_cast<size_t>(batch_size), 0.5f);
        if (cudaMemcpyAsync(scratch.batch_a,
                            disabled_thresholds.data(),
                            static_cast<size_t>(batch_size) * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess ||
            cudaMemcpyAsync(scratch.batch_b,
                            disabled_thresholds.data(),
                            static_cast<size_t>(batch_size) * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess ||
            cudaMemcpyAsync(scratch.batch_c,
                            disabled_thresholds.data(),
                            static_cast<size_t>(batch_size) * sizeof(float),
                            cudaMemcpyHostToDevice,
                            stream) != cudaSuccess) {
          return false;
        }

        residual_veto_threshold_mask_batch_kernel<<<blocks, threads, 0, stream>>>(output_combined_score_device,
                                                                                   static_cast<int>(total_values),
                                                                                   0.5f,
                                                                                   scratch.mask_a);
        if (cudaGetLastError() != cudaSuccess) {
          return false;
        }
        residual_veto_apply_valid_rows_mask_u8_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.mask_a,
                                                                                             batch_size,
                                                                                             rows,
                                                                                             cols,
                                                                                             scratch.valid_row_mask);
        if (cudaGetLastError() != cudaSuccess) {
          return false;
        }
        if (cudaMemcpyAsync(scratch.mask_b,
                            scratch.mask_a,
                            total_values * sizeof(uint8_t),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
          return false;
        }
        if (output_filled_mask_batch_device != nullptr &&
            cudaMemcpyAsync(output_filled_mask_batch_device,
                            scratch.mask_a,
                            total_values * sizeof(uint8_t),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
          return false;
        }
        if (output_component_filtered_mask_batch_device != nullptr &&
            cudaMemcpyAsync(output_component_filtered_mask_batch_device,
                            scratch.mask_a,
                            total_values * sizeof(uint8_t),
                            cudaMemcpyDeviceToDevice,
                            stream) != cudaSuccess) {
          return false;
        }
        if (output_final_mask_device != nullptr) {
          const auto output_copy_start = std::chrono::steady_clock::now();
          residual_veto_uint8_to_float_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.mask_a,
                                                                                     static_cast<int>(total_values),
                                                                                     output_final_mask_device);
          if (cudaGetLastError() != cudaSuccess ||
              !finish_stage(output_copy_start, stage_timing != nullptr ? &stage_timing->output_copy_ms : nullptr)) {
            return false;
          }
        }
      }

      return true;
    }

    const auto residual_stack_start = std::chrono::steady_clock::now();
    if (!normalize_map01_masked_minmax_and_row_filter_bank_cuda_batch_to_device(output_combined_score_device,
                                           batch_size,
                                           rows,
                                           cols,
                                           scratch.valid_row_mask,
                                           scratch.batch_a,
                                           scratch.batch_b,
                                           scratch.reduction_partial_min,
                                           scratch.reduction_partial_max,
                                           scratch.reduction_partial_valid,
                                           scratch.values_a,
                                           kernel_cache.gaussian_rows_6,
                                           output_combined_score_device,
                                           kernel_cache.gaussian_rows_4,
                                           scratch.values_b,
                                           kernel_cache.second_derivative_rows_08,
                                           scratch.values_d,
                                           stream) ||
      !dual_col_convolve_cuda_batch_to_device(output_combined_score_device,
                                                scratch.values_b,
                                                batch_size,
                                                rows,
                                                cols,
                                                kernel_cache.gaussian_cols_14,
                                                scratch.values_c,
                                                kernel_cache.gaussian_cols_1,
                                                scratch.values_e,
                                                            stream)) {
      return false;
    }
    residual_veto_abs_diff_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.values_a,
                                                                         scratch.values_e,
                                                                         static_cast<int>(total_values),
                                                                         scratch.values_b);
    if (cudaGetLastError() != cudaSuccess ||
        !separable_convolve_cuda_batch_to_device(scratch.values_b,
                                                 batch_size,
                                                 rows,
                                                 cols,
                                                 kernel_cache.gaussian_rows_2,
                                                 kernel_cache.gaussian_cols_08,
                                                 scratch.values_a,
                                                 scratch.values_b,
                                                 stream) ||
        !normalize_map01_masked_minmax_cuda_batch_to_device(scratch.values_b,
                                                            batch_size,
                                                            rows,
                                                            cols,
                                                            scratch.valid_row_mask,
                                                            scratch.batch_a,
                                                            scratch.batch_b,
                                                            scratch.reduction_partial_min,
                                                            scratch.reduction_partial_max,
                                                            scratch.reduction_partial_valid,
                                                            scratch.values_b,
                                           stream)) {
      return false;
    }
    residual_veto_abs_inplace_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.values_d,
                                                                            static_cast<int>(total_values));
    if (cudaGetLastError() != cudaSuccess ||
        !normalize_map01_masked_minmax_triplet_cuda_batch_to_device(scratch.values_c,
                                                                    scratch.values_b,
                                                                    scratch.values_d,
                                                                    batch_size,
                                                                    rows,
                                                                    cols,
                                                                    scratch.valid_row_mask,
                                                                    scratch.batch_a,
                                                                    scratch.batch_d,
                                                                    scratch.batch_b,
                                                                    scratch.batch_e,
                                                                    scratch.batch_c,
                                                                    scratch.batch_f,
                                                                    scratch.reduction_partial_min,
                                                                    scratch.reduction_partial_max,
                                                                    scratch.reduction_partial_valid,
                                                                    scratch.values_c,
                                                                    scratch.values_b,
                                                                    scratch.values_d,
                                                                    stream)) {
      return false;
    }
    if (!normalize_subtract_scaled_masked_minmax_pair_cuda_batch_to_device(scratch.values_c,
                                                                           scratch.values_d,
                                                                           0.90f,
                                                                           scratch.values_b,
                                                                           1.0f,
                                                                           batch_size,
                                                                           rows,
                                                                           cols,
                                                                           scratch.valid_row_mask,
                                                                           scratch.batch_a,
                                                                           scratch.batch_b,
                                                                           scratch.batch_c,
                                                                           scratch.batch_d,
                                                                           scratch.reduction_partial_min,
                                                                           scratch.reduction_partial_max,
                                                                           scratch.reduction_partial_valid,
                                                                           scratch.values_a,
                                                                           scratch.values_d,
                                                                           stream)) {
      return false;
    }
    if (!normalize_combined_input_masked_minmax_cuda_batch_to_device(scratch.values_a,
                                                                     scratch.values_d,
                                                                     batch_size,
                                                                     rows,
                                                                     cols,
                                                                     scratch.valid_row_mask,
                                                                     scratch.batch_a,
                                                                     scratch.batch_b,
                                                                     scratch.reduction_partial_min,
                                                                     scratch.reduction_partial_max,
                                                                     scratch.reduction_partial_valid,
                                                                     output_combined_score_device,
                                                                     stream) ||
        !finish_stage(residual_stack_start, stage_timing != nullptr ? &stage_timing->residual_stack_ms : nullptr)) {
      return false;
    }

    const auto threshold_extract_start = std::chrono::steady_clock::now();
    const bool use_sampled_threshold_histograms = rows >= 128 && cols >= 256;
    if (!masked_histogram_quantile_triplet_cuda_batch_to_device(scratch.values_a,
                                                                scratch.values_d,
                                                                output_combined_score_device,
                                                                batch_size,
                                                                rows,
                                                                cols,
                                                                scratch.valid_row_mask,
                                                                0.90f,
                                                                1.0f,
                                                                0.82f,
                                                                1.0f,
                                                                0.78f,
                                                                1.0f,
                                                                use_sampled_threshold_histograms,
                                                                scratch.histograms,
                                                                scratch.batch_a,
                                                                scratch.batch_b,
                                                                scratch.batch_c,
                                                                stream) ||
        !finish_stage(threshold_extract_start, stage_timing != nullptr ? &stage_timing->threshold_extract_ms : nullptr)) {
      return false;
    }

    residual_veto_final_mask_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.values_a,
                                                                           scratch.values_d,
                                                                           output_combined_score_device,
                                                                           batch_size,
                                                                           rows,
                                                                           cols,
                                                                           scratch.valid_row_mask,
                                                                           scratch.batch_a,
                                                                           scratch.batch_b,
                                                                           scratch.batch_c,
                                                                           scratch.mask_a);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }

    const auto closing_start = std::chrono::steady_clock::now();
    if (!binary_closing_rect_cuda_batch_to_device(scratch.mask_a,
                                                  batch_size,
                                                  rows,
                                                  cols,
                                                  7,
                                                  3,
                                                  scratch.mask_b,
                                                  stream) ||
        !finish_stage(closing_start, stage_timing != nullptr ? &stage_timing->closing_ms : nullptr)) {
      return false;
    }

    const auto fill_holes_start = std::chrono::steady_clock::now();
    uint8_t* filled_mask_output = output_filled_mask_batch_device != nullptr ? output_filled_mask_batch_device : scratch.mask_a;
    if (!binary_fill_holes_cuda_batch_to_device(scratch.mask_b,
                                                batch_size,
                                                rows,
                                                cols,
                                                filled_mask_output,
                                                stream) ||
        !finish_stage(fill_holes_start, stage_timing != nullptr ? &stage_timing->fill_holes_ms : nullptr)) {
      return false;
    }

    const auto component_filter_start = std::chrono::steady_clock::now();
    uint8_t* component_input = filled_mask_output;
    if (output_filled_mask_batch_device != nullptr) {
      if (cudaMemcpyAsync(scratch.mask_a,
                          output_filled_mask_batch_device,
                          total_values * sizeof(uint8_t),
                          cudaMemcpyDeviceToDevice,
                          stream) != cudaSuccess) {
        return false;
      }
      component_input = scratch.mask_a;
    }
    residual_veto_apply_valid_rows_mask_u8_batch_kernel<<<blocks, threads, 0, stream>>>(component_input,
                                                                                         batch_size,
                                                                                         rows,
                                                                                         cols,
                                                                                         scratch.valid_row_mask);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }
    uint8_t* filtered_mask_output =
        output_component_filtered_mask_batch_device != nullptr ? output_component_filtered_mask_batch_device : scratch.mask_b;
    if (!keep_large_components_cuda_batch_to_device(component_input,
                                                    batch_size,
                                                    rows,
                                                    cols,
                                                    min_component_size,
                                                    filtered_mask_output,
                                                    stream) ||
        !finish_stage(component_filter_start, stage_timing != nullptr ? &stage_timing->component_filter_ms : nullptr)) {
      return false;
    }

    if (output_final_mask_device != nullptr) {
      const auto output_copy_start = std::chrono::steady_clock::now();
      residual_veto_uint8_to_float_batch_kernel<<<blocks, threads, 0, stream>>>(filtered_mask_output,
                                                                                 static_cast<int>(total_values),
                                                                                 output_final_mask_device);
      if (cudaGetLastError() != cudaSuccess ||
          !finish_stage(output_copy_start, stage_timing != nullptr ? &stage_timing->output_copy_ms : nullptr)) {
        return false;
      }
    } else if (stage_timing != nullptr) {
      stage_timing->output_copy_ms = 0.0;
    }

    return true;
  } catch (...) {
    return false;
  }
}

bool compute_fast_directional_coherence_gate_gpu_batch_to_device(const float* corrected_batch_device,
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

  const size_t plane = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  const size_t total_values = static_cast<size_t>(batch_size) * plane;
  const size_t total_row_mask = static_cast<size_t>(batch_size) * static_cast<size_t>(rows);
  auto& scratch = directional_coherence_cuda_scratch();
  if (!scratch.ensure_capacity(total_values, total_row_mask)) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  if (cudaMemcpyAsync(scratch.valid_row_mask,
                      valid_row_mask_batch.data(),
                      total_row_mask * sizeof(uint8_t),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }

  const int bg_freq_radius = std::max(9, 2 * std::max(1, rows / 24) + 1) / 2;
  const int bg_time_radius = std::max(9, 2 * std::max(1, cols / 24) + 1) / 2;
  constexpr int kDirectionalTimeRadius = 7;
  constexpr int kDirectionalFreqRadius = 2;
  constexpr float kResidualMixWeight = 0.25f;

  const int threads = 256;
  const int blocks = static_cast<int>((total_values + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));
  constexpr int kLineThreads = 128;
  const dim3 col_grid((cols + kLineThreads - 1) / kLineThreads, batch_size * rows);
  const dim3 row_grid((rows + kLineThreads - 1) / kLineThreads, batch_size * cols);

  directional_box_mean_cols_batch_kernel<<<col_grid,
                                           kLineThreads,
                                           static_cast<size_t>(kLineThreads + 2 * bg_time_radius) * sizeof(float),
                                           stream>>>(corrected_batch_device,
                                                                                         batch_size,
                                                                                         rows,
                                                                                         cols,
                                                                                         bg_time_radius,
                                                                                         scratch.background_cols);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_box_mean_rows_batch_kernel<<<row_grid,
                                           kLineThreads,
                                           static_cast<size_t>(kLineThreads + 2 * bg_freq_radius) * sizeof(float),
                                           stream>>>(scratch.background_cols,
                                                                                         batch_size,
                                                                                         rows,
                                                                                         cols,
                                                                                         bg_freq_radius,
                                                                                         scratch.background);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_subtract_clamp_batch_kernel<<<blocks, threads, 0, stream>>>(corrected_batch_device,
                                                                           scratch.background,
                                                                           static_cast<int>(total_values),
                                                                           scratch.residual);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_normalize_clamp_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.residual,
                                                                            static_cast<int>(total_values),
                                                                            0.0f,
                                                                            1.0f / 40.0f,
                                                                            scratch.residual);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_box_mean_cols_batch_kernel<<<col_grid,
                                           kLineThreads,
                                           static_cast<size_t>(kLineThreads + 2 * kDirectionalTimeRadius) * sizeof(float),
                                           stream>>>(scratch.residual,
                                                                                         batch_size,
                                                                                         rows,
                                                                                         cols,
                                                                                         kDirectionalTimeRadius,
                                                                                         scratch.background_cols);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_box_mean_rows_batch_kernel<<<row_grid,
                                           kLineThreads,
                                           static_cast<size_t>(kLineThreads + 2 * kDirectionalFreqRadius) * sizeof(float),
                                           stream>>>(scratch.residual,
                                                                                         batch_size,
                                                                                         rows,
                                                                                         cols,
                                                                                         kDirectionalFreqRadius,
                                                                                         scratch.background);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_subtract_clamp_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.background_cols,
                                                                           scratch.background,
                                                                           static_cast<int>(total_values),
                                                                           scratch.directional_delta);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_weighted_sum_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.directional_delta,
                                                                         scratch.residual,
                                                                         static_cast<int>(total_values),
                                                                         1.0f - kResidualMixWeight,
                                                                         kResidualMixWeight,
                                                                         output_gate_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_normalize_clamp_batch_kernel<<<blocks, threads, 0, stream>>>(output_gate_device,
                                                                            static_cast<int>(total_values),
                                                                            0.0f,
                                                                            1.0f,
                                                                            output_gate_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_apply_valid_rows_batch_kernel<<<blocks, threads, 0, stream>>>(output_gate_device,
                                                                             batch_size,
                                                                             rows,
                                                                             cols,
                                                                             scratch.valid_row_mask);
  return cudaGetLastError() == cudaSuccess;
}

CudaDinoDetector::~CudaDinoDetector() {
  release_channel_buffers();
}

void CudaDinoDetector::setup(holoscan::OperatorSpec& spec) {
  const std::vector<double> imagenet_mean_default{0.485, 0.456, 0.406};
  const std::vector<double> imagenet_std_default{0.229, 0.224, 0.225};
  spec.input<cuda_dino_in_t>("in");
  spec.output<holoscan::ops::DetectorMaskMessage>("mask_out").condition(holoscan::ConditionType::kNone);

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of detector channels.", 1);
  spec.param(input_height_, "input_height", "Input height", "Reference DINO input height.", 256);
  spec.param(input_width_, "input_width", "Input width", "Reference DINO input width.", 512);
  spec.param(patch_size_, "patch_size", "Patch size", "DINO patch size.", 16);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Process every Nth frame.", 1);
  spec.param(channel_filter_,
             "channel_filter",
             "Channel filter",
             "Optional single-channel filter; negative means all channels.",
             -1);
  spec.param(debug_mode_,
             "debug_mode",
             "Debug mode",
             "Enable parity-oriented debug behavior and selected intermediate artifact capture.",
             false);
  spec.param(enable_debug_artifact_host_copy_,
             "enable_debug_artifact_host_copy",
             "Enable debug artifact host copy",
             "Allow selected intermediate products to be copied back to host in debug mode.",
             false);
  spec.param(debug_chunk_index_,
             "debug_chunk_index",
             "Debug chunk index",
             "Selected chunk or tile index for artifact extraction in debug mode.",
             13);
  spec.param(debug_artifact_output_dir_,
             "debug_artifact_output_dir",
             "Debug artifact output dir",
             "When set, write a validator-style offline artifact bundle for the selected debug chunk.",
             std::string(""));
  spec.param(save_aligned_spectrogram_preview_,
             "save_aligned_spectrogram_preview",
             "Save aligned spectrogram preview",
             "Save the exact detector-input spectrogram preview used for this frame.",
             false);
  spec.param(save_aligned_spectrogram_tensor_,
             "save_aligned_spectrogram_tensor",
             "Save aligned spectrogram tensor",
             "Save the exact detector-input spectrogram tensor used for this frame.",
             false);
  spec.param(aligned_spectrogram_output_height_,
             "aligned_spectrogram_output_height",
             "Aligned spectrogram output height",
             "Saved preview height for detector-aligned spectrogram artifacts.",
             256);
  spec.param(aligned_spectrogram_output_width_,
             "aligned_spectrogram_output_width",
             "Aligned spectrogram output width",
             "Saved preview width for detector-aligned spectrogram artifacts.",
             512);
  spec.param(aligned_spectrogram_output_dir_,
             "aligned_spectrogram_output_dir",
             "Aligned spectrogram output dir",
             "Root directory where detector-aligned spectrogram artifacts are written.",
             std::string(""));
  spec.param(execution_strategy_,
             "execution_strategy",
             "Execution strategy",
             "CUDA DINO execution strategy: reference_chunks, adaptive_tiles, or coarse_to_fine.",
             std::string("reference_chunks"));
  spec.param(max_tokens_per_inference_,
             "max_tokens_per_inference",
             "Max tokens per inference",
             "Soft upper bound used when selecting a CUDA DINO execution strategy.",
             8192);
  spec.param(chunk_bandwidth_hz_,
             "chunk_bandwidth_hz",
             "Chunk bandwidth Hz",
             "Target chunk bandwidth in Hz for the CUDA DINO path.",
             25.0e6);
  spec.param(chunk_overlap_hz_,
             "chunk_overlap_hz",
             "Chunk overlap Hz",
             "Chunk overlap in Hz for the CUDA DINO path.",
             6.25e6);
  spec.param(uncalibrated_chunk_fraction_,
             "uncalibrated_chunk_fraction",
             "Uncalibrated chunk fraction",
             "Fallback chunk size fraction when the frequency axis is not calibrated.",
             0.40);
  spec.param(uncalibrated_overlap_fraction_,
             "uncalibrated_overlap_fraction",
             "Uncalibrated overlap fraction",
             "Fallback overlap fraction when the frequency axis is not calibrated.",
             0.20);
  spec.param(ignore_sideband_hz_,
             "ignore_sideband_hz",
             "Ignore sideband Hz",
             "Frequency span ignored on each side before chunk planning.",
             7.0e6);
  spec.param(frontend_correction_enable_,
             "frontend_correction_enable",
             "Frontend correction enable",
             "Enable frontend correction before chunk-local CUDA stages.",
             true);
  spec.param(frontend_correction_row_q_,
             "frontend_correction_row_q",
             "Frontend correction row quantile",
             "Row quantile used by the frontend correction reference path.",
             25.0);
  spec.param(frontend_correction_smooth_sigma_,
             "frontend_correction_smooth_sigma",
             "Frontend correction smooth sigma",
             "Smoothing sigma used by the frontend correction reference path.",
             12.0);
  spec.param(frontend_correction_reference_q_,
             "frontend_correction_reference_q",
             "Frontend correction reference quantile",
             "Reference quantile used by the frontend correction path.",
             75.0);
  spec.param(frontend_correction_max_boost_db_,
             "frontend_correction_max_boost_db",
             "Frontend correction max boost dB",
             "Maximum frontend correction boost in dB.",
             12.0);
  spec.param(frontend_correction_soft_knee_db_,
             "frontend_correction_soft_knee_db",
             "Frontend correction soft knee dB",
             "Soft-knee control in dB for the frontend correction path.",
             4.0);
  spec.param(frontend_correction_edge_taper_fraction_,
             "frontend_correction_edge_taper_fraction",
             "Frontend correction edge taper fraction",
             "Edge taper fraction for frontend correction.",
             0.10);
  spec.param(frontend_correction_edge_taper_sigma_,
             "frontend_correction_edge_taper_sigma",
             "Frontend correction edge taper sigma",
             "Edge taper sigma for frontend correction.",
             6.0);
  spec.param(frontend_correction_edge_target_drop_db_,
             "frontend_correction_edge_target_drop_db",
             "Frontend correction edge target drop dB",
             "Target edge attenuation in dB.",
             2.5);
  spec.param(dino_coherence_gate_floor_,
             "dino_coherence_gate_floor",
             "DINO coherence gate floor",
             "Lower bound for the coherence gate normalization.",
             0.25);
  spec.param(dino_coherence_gate_span_db_,
             "dino_coherence_gate_span_db",
             "DINO coherence gate span dB",
             "Span in dB for the coherence gate normalization.",
             3.0);
  spec.param(raw_dino_positional_deweight_,
             "raw_dino_positional_deweight",
             "Raw DINO positional deweight",
             "Trend-only positional suppression weight for raw patch-energy scoring.",
             0.75f);
  spec.param(power_q_,
             "power_q",
             "Power quantile",
             "Reference power quantile used by the CUDA detector path.",
             0.90);
  spec.param(hybrid_torch_dtype_,
             "hybrid_torch_dtype",
             "Hybrid torch dtype",
             "Torch dtype used by the CUDA hybrid residual-veto path.",
             std::string("fp16"));
  spec.param(enable_mask_post_processing_,
             "enable_mask_post_processing",
             "Enable mask post processing",
             "Run the hybrid residual-veto, thresholding, and mask post-processing path after the initial combined score stage.",
             true);
  spec.param(hybrid_component_min_size_,
             "hybrid_component_min_size",
             "Hybrid component minimum size",
             "Minimum connected component size retained after hybrid morphology.",
             24);
  spec.param(grouping_bridge_freq_px_,
             "grouping_bridge_freq_px",
             "Grouping bridge frequency px",
             "Frequency-axis bridge size for validated region grouping.",
             33);
  spec.param(grouping_bridge_time_px_,
             "grouping_bridge_time_px",
             "Grouping bridge time px",
             "Time-axis bridge size for validated region grouping.",
             5);
  spec.param(grouping_min_component_size_,
             "grouping_min_component_size",
             "Grouping minimum component size",
             "Minimum grouped component area for validated region grouping.",
             24);
  spec.param(grouping_min_freq_span_px_,
             "grouping_min_freq_span_px",
             "Grouping minimum frequency span",
             "Minimum grouped component frequency span in pixels.",
             18);
  spec.param(grouping_min_time_span_px_,
             "grouping_min_time_span_px",
             "Grouping minimum time span",
             "Minimum grouped component time span in pixels.",
             2);
  spec.param(grouping_min_density_,
             "grouping_min_density",
             "Grouping minimum density",
             "Minimum grouped component density.",
             0.06);
  spec.param(filter_detection_mask_,
             "filter_detection_mask",
             "Filter detection mask",
             "Apply reference-style grouped region filtering before final box merge.",
             true);
  spec.param(emit_grouped_merged_mask_,
             "emit_grouped_merged_mask",
             "Emit grouped merged mask",
             "Emit the grouped and merged detector box mask instead of the raw stitched threshold mask.",
             false);
  spec.param(grouping_time_continuity_ratio_,
             "grouping_time_continuity_ratio",
             "Grouping time continuity ratio",
             "Continuity threshold used when bridging nearly continuous time gaps.",
             0.85);
  spec.param(backend_mode_,
             "backend_mode",
             "Backend mode",
             "Execution mode selector for CUDA DINO detector bring-up.",
             std::string("cuda_partial"));
  spec.param(inference_backend_,
             "inference_backend",
             "Inference backend",
             "Runtime inference backend used for DINO patch features.",
             std::string("torchscript"));
  spec.param(model_script_path_,
             "model_script_path",
             "Model script path",
             "TorchScript model path for the CUDA DINO runtime path.",
             std::string(""));
  spec.param(torchscript_init_mode_,
             "torchscript_init_mode",
             "TorchScript init mode",
             "TorchScript module initialization mode for the CUDA DINO runtime path.",
             std::string("load_cuda_eval"));
  spec.param(torch_dtype_,
             "torch_dtype",
             "Torch dtype",
             "Torch runtime dtype for CUDA DINO runtime execution.",
             std::string("fp32"));
  spec.param(imagenet_mean_,
             "imagenet_mean",
             "ImageNet mean",
             "Mean used for notebook-aligned model normalization.",
             imagenet_mean_default);
  spec.param(imagenet_std_,
             "imagenet_std",
             "ImageNet std",
             "Standard deviation used for notebook-aligned model normalization.",
             imagenet_std_default);
  spec.param(timing_summary_enable_,
             "timing_summary_enable",
             "Timing summary enable",
             "Enable periodic timing summary logging.",
             true);
  spec.param(timing_summary_every_n_,
             "timing_summary_every_n",
             "Timing summary cadence",
             "Emit one summary every N processed frames.",
             16);
  spec.param(timing_summary_window_,
             "timing_summary_window",
             "Timing summary window",
             "Rolling window size for timing aggregation.",
             16);
}

void CudaDinoDetector::initialize() {
  Operator::initialize();
  release_channel_buffers();
  channel_buffers_.assign(static_cast<size_t>(std::max(1, num_channels_.get())), ChannelBuffers{});
  timing_stats_.assign(static_cast<size_t>(std::max(1, num_channels_.get())), ChannelTimingStats{});
  frame_count_.assign(static_cast<size_t>(std::max(1, num_channels_.get())), 0);
  skipped_partial_batches_.assign(static_cast<size_t>(std::max(1, num_channels_.get())), 0);
  skipped_stride_frames_.assign(static_cast<size_t>(std::max(1, num_channels_.get())), 0);
  if (!runtime_) {
    runtime_ = std::make_shared<DinoTorchRuntime>();
  }
  if (is_truthy_backend_mode(backend_mode_.get()) && inference_backend_.get() == "torchscript" && !model_script_path_.get().empty()) {
    DinoTorchRuntimeConfig runtime_config;
    runtime_config.inference_backend = inference_backend_.get();
    runtime_config.model_script_path = model_script_path_.get();
    runtime_config.torchscript_init_mode = torchscript_init_mode_.get();
    runtime_config.torch_dtype = torch_dtype_.get();
    runtime_config.imagenet_mean = imagenet_mean_.get();
    runtime_config.imagenet_std = imagenet_std_.get();
    runtime_config.return_patch_features = true;
    runtime_config.return_patch_features_host = false;
    runtime_config.return_final_mask_device = true;
    runtime_config.compute_power_score = false;
    runtime_config.frontend_correction_enable = false;

    runtime_->warmup(runtime_config,
                     std::max(1, input_height_.get()),
                     std::max(1, input_width_.get()),
                     std::max(1, input_height_.get()),
                     std::max(1, input_width_.get()),
                     std::max(1, patch_size_.get()));
  }
  std::fprintf(stderr,
               "[cuda_dino_detector] INFO: initialized backend_mode='%s' execution_strategy='%s' debug_mode=%d host_copy_debug_only=%d max_tokens_per_inference=%d input=%dx%d patch_size=%d emit_stride=%d chunk_bw_hz=%.3f overlap_hz=%.3f frontend_correction_enable=%d enable_mask_post_processing=%d\n",
               backend_mode_.get().c_str(),
               execution_strategy_.get().c_str(),
               debug_mode_.get() ? 1 : 0,
               enable_debug_artifact_host_copy_.get() ? 1 : 0,
               max_tokens_per_inference_.get(),
               input_height_.get(),
               input_width_.get(),
               patch_size_.get(),
               emit_stride_.get(),
               chunk_bandwidth_hz_.get(),
               chunk_overlap_hz_.get(),
               frontend_correction_enable_.get() ? 1 : 0,
               enable_mask_post_processing_.get() ? 1 : 0);
}

void CudaDinoDetector::stop() {
  for (size_t channel_index = 0; channel_index < frame_count_.size(); ++channel_index) {
    if (skipped_partial_batches_[channel_index] == 0 && skipped_stride_frames_[channel_index] == 0) {
      continue;
    }
    std::fprintf(stderr,
                 "[cuda_dino_detector] INFO: channel %zu skipped_partial_batches=%llu skipped_stride_frames=%llu\n",
                 channel_index,
                 static_cast<unsigned long long>(skipped_partial_batches_[channel_index]),
                 static_cast<unsigned long long>(skipped_stride_frames_[channel_index]));
  }
  release_channel_buffers();
  Operator::stop();
}

void CudaDinoDetector::compute(holoscan::InputContext& op_input,
                               holoscan::OutputContext& op_output,
                               holoscan::ExecutionContext& context) {
  static_cast<void>(context);

  auto maybe_input = op_input.receive<cuda_dino_in_t>("in");
  if (!maybe_input) {
    std::fprintf(stderr, "[cuda_dino_detector] WARN: received no input tensor\n");
    return;
  }

  const auto& [fft_tensor, fft_stream] = *maybe_input;
  const auto compute_start_time = std::chrono::steady_clock::now();
  auto meta = metadata();
  const uint16_t channel_number = meta ? meta->get<uint16_t>("channel_number", 0) : 0;
  const int channel_filter = channel_filter_.get();
  if (channel_filter >= 0 && channel_number != static_cast<uint16_t>(channel_filter)) {
    return;
  }

  const size_t local_channel_index = channel_filter >= 0 ? 0u : static_cast<size_t>(channel_number);
  if (local_channel_index >= channel_buffers_.size()) {
    std::fprintf(stderr,
                 "[cuda_dino_detector] WARN: received out-of-range channel %u (configured channels: %zu)\n",
                 static_cast<unsigned>(channel_number),
                 channel_buffers_.size());
    return;
  }

  uint64_t frame_number = 0;
  if (meta && meta->has_key("fft_emitted_frame_number")) {
    frame_number = meta->get<uint64_t>("fft_emitted_frame_number", frame_count_[local_channel_index] + 1);
    if (frame_number == 0) {
      frame_number = frame_count_[local_channel_index] + 1;
    }
    frame_count_[local_channel_index] = std::max(frame_count_[local_channel_index], frame_number);
  } else {
    frame_number = ++frame_count_[local_channel_index];
  }

  if (meta) {
    meta->set("cuda_dino_frame_number", frame_number);
  }

  if (meta && meta->get<bool>("offline_source_drain_frame", false)) {
    meta->set("cuda_dino_skipped_partial_batch", false);
    meta->set("cuda_dino_skipped_emit_stride", false);
    meta->set("cuda_dino_skipped_offline_drain_frame", true);
    meta->set("cuda_dino_mask_emitted", false);
    return;
  }

  if (meta && meta->get<bool>("chdr_partial_batch", false)) {
    skipped_partial_batches_[local_channel_index]++;
    meta->set("cuda_dino_skipped_partial_batch", true);
    meta->set("cuda_dino_skipped_emit_stride", false);
    meta->set("cuda_dino_mask_emitted", false);
    return;
  }

  const int stride = std::max(1, emit_stride_.get());
  if ((frame_number % static_cast<uint64_t>(stride)) != 0) {
    skipped_stride_frames_[local_channel_index]++;
    if (meta) {
      meta->set("cuda_dino_skipped_partial_batch", false);
      meta->set("cuda_dino_skipped_emit_stride", true);
      meta->set("cuda_dino_mask_emitted", false);
    }
    return;
  }

  const int input_rows = static_cast<int>(fft_tensor.Size(0));
  const int input_cols = static_cast<int>(fft_tensor.Size(1));
  if (input_rows <= 0 || input_cols <= 0) {
    std::fprintf(stderr,
                 "[cuda_dino_detector] WARN: received empty tensor shape %dx%d\n",
                 input_rows,
                 input_cols);
    if (meta) {
      meta->set("cuda_dino_skipped_partial_batch", false);
      meta->set("cuda_dino_skipped_emit_stride", false);
      meta->set("cuda_dino_mask_emitted", false);
    }
    return;
  }

  // FFT/spectrogram emits time x frequency, while the detector chunking path expects frequency x time.
  const int src_rows = input_cols;
  const int src_cols = input_rows;

  if (meta) {
    meta->set("cuda_dino_skipped_partial_batch", false);
    meta->set("cuda_dino_skipped_emit_stride", false);
    meta->set("cuda_dino_mask_height", static_cast<uint32_t>(input_rows));
    meta->set("cuda_dino_mask_width", static_cast<uint32_t>(input_cols));
  }

  auto& buffers = channel_buffers_[local_channel_index];
  const size_t frame_elements = static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols);
  const bool capture_operator_timing =
      debug_mode_.get() && enable_debug_artifact_host_copy_.get() && !debug_artifact_output_dir_.get().empty();
  OperatorTimingProfile timing_profile;

  if (buffers.processing_stream == nullptr) {
    throw_if_cuda_error(cudaStreamCreateWithFlags(&buffers.processing_stream, cudaStreamNonBlocking),
                        "cuda_dino_detector processing stream creation failed");
  }
  if (buffers.coherence_start_event == nullptr) {
    throw_if_cuda_error(cudaEventCreate(&buffers.coherence_start_event),
                        "cuda_dino_detector coherence-start event creation failed");
  }
  if (buffers.coherence_end_event == nullptr) {
    throw_if_cuda_error(cudaEventCreate(&buffers.coherence_end_event),
                        "cuda_dino_detector coherence-end event creation failed");
  }
  if (buffers.copy_complete_event == nullptr) {
    throw_if_cuda_error(cudaEventCreateWithFlags(&buffers.copy_complete_event, cudaEventDisableTiming),
                        "cuda_dino_detector copy-complete event creation failed");
  }

  if (buffers.frame_elements != frame_elements) {
    if (buffers.analysis_tensor_device != nullptr) {
      cudaFree(buffers.analysis_tensor_device);
      buffers.analysis_tensor_device = nullptr;
    }
    if (buffers.power_db_device != nullptr) {
      cudaFree(buffers.power_db_device);
      buffers.power_db_device = nullptr;
    }
    if (buffers.corrected_db_device != nullptr) {
      cudaFree(buffers.corrected_db_device);
      buffers.corrected_db_device = nullptr;
    }
    if (buffers.corrected_batch_device != nullptr) {
      cudaFree(buffers.corrected_batch_device);
      buffers.corrected_batch_device = nullptr;
      buffers.batch_elements = 0;
    }
    if (buffers.coherence_gate_batch_device != nullptr) {
      cudaFree(buffers.coherence_gate_batch_device);
      buffers.coherence_gate_batch_device = nullptr;
    }
    if (buffers.raw_dino_score_batch_device != nullptr) {
      cudaFree(buffers.raw_dino_score_batch_device);
      buffers.raw_dino_score_batch_device = nullptr;
    }
    if (buffers.hybrid_combined_score_batch_device != nullptr) {
      cudaFree(buffers.hybrid_combined_score_batch_device);
      buffers.hybrid_combined_score_batch_device = nullptr;
    }
    if (buffers.hybrid_final_mask_batch_device != nullptr) {
      cudaFree(buffers.hybrid_final_mask_batch_device);
      buffers.hybrid_final_mask_batch_device = nullptr;
    }
    if (buffers.hybrid_filled_mask_batch_device != nullptr) {
      cudaFree(buffers.hybrid_filled_mask_batch_device);
      buffers.hybrid_filled_mask_batch_device = nullptr;
    }
    if (buffers.hybrid_component_filtered_mask_batch_device != nullptr) {
      cudaFree(buffers.hybrid_component_filtered_mask_batch_device);
      buffers.hybrid_component_filtered_mask_batch_device = nullptr;
    }

    throw_if_cuda_error(cudaMalloc(reinterpret_cast<void**>(&buffers.analysis_tensor_device),
                                   frame_elements * sizeof(cuda_dino_complex)),
                        "cuda_dino_detector analysis tensor allocation failed");
    allocate_device_float(buffers.power_db_device, frame_elements);
    allocate_device_float(buffers.corrected_db_device, frame_elements);
    buffers.frame_elements = frame_elements;
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
    if (buffers.frontend_reference_device != nullptr) {
      cudaFree(buffers.frontend_reference_device);
      buffers.frontend_reference_device = nullptr;
    }
    allocate_device_float(buffers.row_stat_device, static_cast<size_t>(src_rows));
    allocate_device_float(buffers.row_smooth_device, static_cast<size_t>(src_rows));
    allocate_device_float(buffers.frontend_reference_device, 1);
    buffers.row_elements = static_cast<size_t>(src_rows);
  }

  ++compute_count_;
  if (!startup_log_emitted_) {
    startup_log_emitted_ = true;
    std::fprintf(stderr,
                 "[cuda_dino_detector] INFO: scaffold active, received first tensor with shape %ldx%ld; fast path stays device-resident and host debug copies remain opt-in\n",
                 static_cast<long>(fft_tensor.Size(0)),
                 static_cast<long>(fft_tensor.Size(1)));
  }

  constexpr int threads = 256;
  const int blocks = static_cast<int>((frame_elements + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  throw_if_cuda_error(cudaEventRecord(buffers.copy_complete_event, fft_stream),
                      "cuda_dino_detector copy-complete event record failed");
  throw_if_cuda_error(cudaStreamWaitEvent(buffers.processing_stream, buffers.copy_complete_event, 0),
                      "cuda_dino_detector processing stream wait failed");

  {
    constexpr int transpose_block_dim = 16;
    const dim3 transpose_block(transpose_block_dim, transpose_block_dim);
    const dim3 transpose_grid((input_cols + transpose_block_dim - 1) / transpose_block_dim,
                              (input_rows + transpose_block_dim - 1) / transpose_block_dim);
    transpose_complex_matrix_kernel<<<transpose_grid, transpose_block, 0, buffers.processing_stream>>>(
        static_cast<const cuda_dino_complex*>(fft_tensor.Data()),
        input_rows,
        input_cols,
        static_cast<cuda_dino_complex*>(buffers.analysis_tensor_device));
    throw_if_cuda_error(cudaGetLastError(), "cuda_dino_detector analysis tensor transpose failed");
  }

  const bool save_aligned_preview =
      save_aligned_spectrogram_preview_.get() && !aligned_spectrogram_output_dir_.get().empty();
  const bool save_aligned_tensor =
      save_aligned_spectrogram_tensor_.get() && !aligned_spectrogram_output_dir_.get().empty();
  if (save_aligned_preview || save_aligned_tensor) {
    const auto artifact_root = std::filesystem::path(aligned_spectrogram_output_dir_.get());
    std::filesystem::create_directories(artifact_root / "aligned_spectrograms");
    std::filesystem::create_directories(artifact_root / "aligned_spectrogram_tensors");

    std::vector<cuda_dino_complex> host_analysis(frame_elements);
    throw_if_cuda_error(cudaMemcpyAsync(host_analysis.data(),
                                        buffers.analysis_tensor_device,
                                        frame_elements * sizeof(cuda_dino_complex),
                                        cudaMemcpyDeviceToHost,
                                        buffers.processing_stream),
                        "cuda_dino_detector aligned spectrogram host copy failed");
    throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                        "cuda_dino_detector aligned spectrogram synchronization failed");

    const auto host_analysis_display = transpose_host_row_major_matrix(host_analysis, src_rows, src_cols);
    if (host_analysis_display.size() != frame_elements) {
      throw std::runtime_error("failed to transpose detector-aligned spectrogram back to display space");
    }

    if (save_aligned_tensor) {
      const auto tensor_path = make_aligned_spectrogram_path(artifact_root,
                                                             "aligned_spectrogram_tensors",
                                                             "aligned_spectrogram_tensor",
                                                             static_cast<int>(channel_number),
                                                             frame_number,
                                                             input_rows,
                                                             input_cols,
                                                             ".npy");
      if (!write_npy_2d(tensor_path,
                        host_analysis_display.data(),
                        host_analysis_display.size() * sizeof(cuda_dino_complex),
                        input_rows,
                        input_cols,
                        "<c8")) {
        throw std::runtime_error("failed to write detector-aligned spectrogram tensor: " + tensor_path.string());
      }
      if (meta) {
        meta->set("cuda_dino_aligned_spectrogram_tensor_path", tensor_path.string());
        meta->set("cuda_dino_aligned_spectrogram_rows", input_rows);
        meta->set("cuda_dino_aligned_spectrogram_cols", input_cols);
      }
    }

    if (save_aligned_preview) {
      const int preview_rows = std::max(1, aligned_spectrogram_output_height_.get());
      const int preview_cols = std::max(1, aligned_spectrogram_output_width_.get());
      const auto preview_image =
          build_spectrogram_preview(host_analysis_display, input_rows, input_cols, preview_rows, preview_cols);
      const auto preview_path = make_aligned_spectrogram_path(artifact_root,
                                                              "aligned_spectrograms",
                                                              "aligned_spectrogram",
                                                              static_cast<int>(channel_number),
                                                              frame_number,
                                                              preview_rows,
                                                              preview_cols,
                                                              ".pgm");
      if (!write_pgm(preview_path, preview_image, preview_cols, preview_rows)) {
        throw std::runtime_error("failed to write detector-aligned spectrogram preview: " + preview_path.string());
      }
      if (meta) {
        meta->set("cuda_dino_aligned_spectrogram_preview_path", preview_path.string());
        meta->set("cuda_dino_aligned_spectrogram_preview_rows", preview_rows);
        meta->set("cuda_dino_aligned_spectrogram_preview_cols", preview_cols);
      }
    }
  }

  const auto power_db_start_time = std::chrono::steady_clock::now();
  cuda_dino_power_db_kernel<<<blocks, threads, 0, buffers.processing_stream>>>(
      static_cast<const cuda_dino_complex*>(buffers.analysis_tensor_device),
      src_rows,
      src_cols,
      buffers.power_db_device);
  throw_if_cuda_error(cudaGetLastError(), "cuda_dino_detector power_db kernel launch failed");
  if (capture_operator_timing) {
    throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                        "cuda_dino_detector power_db timing synchronization failed");
    timing_profile.power_db_ms = elapsed_ms_since(power_db_start_time);
  }

  const auto frontend_start_time = std::chrono::steady_clock::now();
  if (frontend_correction_enable_.get()) {
    const int row_blocks = (src_rows + threads - 1) / threads;
    const int smooth_radius = std::max(
        1,
        static_cast<int>(std::ceil(std::max(frontend_correction_smooth_sigma_.get(), 1.0) * 1.5)));
    cuda_dino_row_mean_kernel<<<src_rows, threads, 0, buffers.processing_stream>>>(buffers.power_db_device,
                                                                                     src_rows,
                                                                                     src_cols,
                                                                                     buffers.row_stat_device);
    cuda_dino_gaussian_smooth_rows_kernel<<<row_blocks, threads, 0, buffers.processing_stream>>>(buffers.row_stat_device,
                                                                                                   src_rows,
                                                                                                   smooth_radius,
                                                                                                   static_cast<float>(std::max(frontend_correction_smooth_sigma_.get(), 1.0)),
                                                                                                   buffers.row_smooth_device);
    cuda_dino_frontend_reference_kernel<<<1, threads, 0, buffers.processing_stream>>>(buffers.row_smooth_device,
                                                                                       src_rows,
                                                                                       static_cast<float>(frontend_correction_reference_q_.get() / 100.0),
                                                                                       buffers.frontend_reference_device);
    cuda_dino_frontend_correction_kernel<<<blocks, threads, 0, buffers.processing_stream>>>(buffers.power_db_device,
                                                                                              src_rows,
                                                                                              src_cols,
                                                                                              buffers.row_smooth_device,
                                                                                              buffers.frontend_reference_device,
                                                                                              static_cast<float>(frontend_correction_max_boost_db_.get()),
                                                                                              buffers.corrected_db_device);
    throw_if_cuda_error(cudaGetLastError(), "cuda_dino_detector frontend correction kernel launch failed");
  } else {
    throw_if_cuda_error(cudaMemcpyAsync(buffers.corrected_db_device,
                                        buffers.power_db_device,
                                        frame_elements * sizeof(float),
                                        cudaMemcpyDeviceToDevice,
                                        buffers.processing_stream),
                        "cuda_dino_detector corrected_db copy failed");
  }
                  if (capture_operator_timing) {
                    throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                                        "cuda_dino_detector frontend timing synchronization failed");
                    timing_profile.frontend_ms = elapsed_ms_since(frontend_start_time);
                  }

  int chunk_count = 0;
  int ignore_bins_per_side = 0;
  double span_hz = 0.0;
  if (meta) {
    span_hz = meta->get<double>("sample_rate_hz", 0.0);
    if (!std::isfinite(span_hz) || span_hz <= 0.0) {
      span_hz = static_cast<double>(meta->get<uint64_t>("span", 0));
    }
    if (!std::isfinite(span_hz) || span_hz <= 0.0) {
      span_hz = meta->get<double>("bandwidth_hz", 0.0);
    }
  }
  if (!std::isfinite(span_hz) || span_hz <= 0.0) {
    span_hz = 0.0;
  }

  double resolution_hz = meta ? static_cast<double>(meta->get<uint64_t>("resolution", 0)) : 0.0;
  if ((!std::isfinite(resolution_hz) || resolution_hz <= 0.0) && span_hz > 0.0 && src_rows > 0) {
    resolution_hz = span_hz / static_cast<double>(src_rows);
  }

  if (execution_strategy_.get() == "reference_chunks") {
    const auto chunk_plan_start_time = std::chrono::steady_clock::now();
    const double chunk_bin_hz = (std::isfinite(resolution_hz) && resolution_hz > 0.0) ? resolution_hz : 1.0;
    const auto source_freq_axis_hz = build_frequency_axis_hz(src_rows, resolution_hz);
    const auto planned_selection = select_uniform_chunk_plan_with_minimal_sideband_trim(src_rows,
                                                                                         chunk_bin_hz,
                                                                                         ignore_sideband_hz_.get(),
                                                                                         16,
                                                                                         source_freq_axis_hz,
                                                                                         chunk_bandwidth_hz_.get(),
                                                                                         chunk_overlap_hz_.get(),
                                                                                         16,
                                                                                         uncalibrated_chunk_fraction_.get(),
                                                                                         uncalibrated_overlap_fraction_.get());
    ignore_bins_per_side = planned_selection.applied_bins;
    const auto& chunk_plan = planned_selection.chunk_plan;
    if (chunk_plan.empty()) {
      throw std::runtime_error("cuda_dino_detector reference_chunks produced an empty chunk plan");
    }
    if (!chunk_plan_has_uniform_rows(chunk_plan)) {
      throw std::runtime_error("cuda_dino_detector reference_chunks requires uniform chunk row counts");
    }
    timing_profile.chunk_plan_ms = elapsed_ms_since(chunk_plan_start_time);

    chunk_count = static_cast<int>(chunk_plan.size());
    const int uniform_chunk_rows = chunk_row_count(chunk_plan.front());
    const size_t batch_elements = static_cast<size_t>(chunk_count) * static_cast<size_t>(uniform_chunk_rows) * static_cast<size_t>(src_cols);

    if (buffers.batch_elements != batch_elements) {
      if (buffers.corrected_batch_device != nullptr) {
        cudaFree(buffers.corrected_batch_device);
        buffers.corrected_batch_device = nullptr;
      }
      if (buffers.coherence_gate_batch_device != nullptr) {
        cudaFree(buffers.coherence_gate_batch_device);
        buffers.coherence_gate_batch_device = nullptr;
      }
      if (buffers.raw_dino_score_batch_device != nullptr) {
        cudaFree(buffers.raw_dino_score_batch_device);
        buffers.raw_dino_score_batch_device = nullptr;
      }
      if (buffers.hybrid_combined_score_batch_device != nullptr) {
        cudaFree(buffers.hybrid_combined_score_batch_device);
        buffers.hybrid_combined_score_batch_device = nullptr;
      }
      if (buffers.hybrid_final_mask_batch_device != nullptr) {
        cudaFree(buffers.hybrid_final_mask_batch_device);
        buffers.hybrid_final_mask_batch_device = nullptr;
      }
      if (buffers.hybrid_filled_mask_batch_device != nullptr) {
        cudaFree(buffers.hybrid_filled_mask_batch_device);
        buffers.hybrid_filled_mask_batch_device = nullptr;
      }
      if (buffers.hybrid_component_filtered_mask_batch_device != nullptr) {
        cudaFree(buffers.hybrid_component_filtered_mask_batch_device);
        buffers.hybrid_component_filtered_mask_batch_device = nullptr;
      }
      allocate_device_float(buffers.corrected_batch_device, batch_elements);
      allocate_device_float(buffers.coherence_gate_batch_device, batch_elements);
      allocate_device_float(buffers.raw_dino_score_batch_device, batch_elements);
      allocate_device_float(buffers.hybrid_combined_score_batch_device, batch_elements);
      allocate_device_float(buffers.hybrid_final_mask_batch_device, batch_elements);
      allocate_device_uint8(buffers.hybrid_filled_mask_batch_device, batch_elements);
      allocate_device_uint8(buffers.hybrid_component_filtered_mask_batch_device, batch_elements);
      buffers.batch_elements = batch_elements;
    }

    if (buffers.chunk_row_start_capacity < static_cast<size_t>(chunk_count)) {
      if (buffers.chunk_row_starts_device != nullptr) {
        cudaFree(buffers.chunk_row_starts_device);
        buffers.chunk_row_starts_device = nullptr;
      }
      allocate_device_int(buffers.chunk_row_starts_device, static_cast<size_t>(chunk_count));
      buffers.chunk_row_start_capacity = static_cast<size_t>(chunk_count);
    }

    const auto chunk_pack_start_time = std::chrono::steady_clock::now();
    std::vector<int> chunk_row_starts(static_cast<size_t>(chunk_count), 0);
    for (int index = 0; index < chunk_count; ++index) {
      chunk_row_starts[static_cast<size_t>(index)] = chunk_plan[static_cast<size_t>(index)].row_start;
    }
    throw_if_cuda_error(cudaMemcpyAsync(buffers.chunk_row_starts_device,
                                        chunk_row_starts.data(),
                                        static_cast<size_t>(chunk_count) * sizeof(int),
                                        cudaMemcpyHostToDevice,
                                        buffers.processing_stream),
                        "cuda_dino_detector chunk row-start upload failed");

    const int pack_total = chunk_count * uniform_chunk_rows * src_cols;
    const int pack_blocks = (pack_total + threads - 1) / threads;
    cuda_dino_pack_reference_chunks_kernel<<<pack_blocks, threads, 0, buffers.processing_stream>>>(buffers.corrected_db_device,
                                                                                                     src_rows,
                                                                                                     src_cols,
                                                                                                     buffers.chunk_row_starts_device,
                                                                                                     uniform_chunk_rows,
                                                                                                     chunk_count,
                                                                                                     buffers.corrected_batch_device);
    throw_if_cuda_error(cudaGetLastError(), "cuda_dino_detector reference chunk pack kernel launch failed");
    if (capture_operator_timing) {
      throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                          "cuda_dino_detector chunk pack timing synchronization failed");
      timing_profile.chunk_pack_ms = elapsed_ms_since(chunk_pack_start_time);
    }

    std::vector<uint8_t> chunk_valid_rows_batch(static_cast<size_t>(chunk_count) * static_cast<size_t>(uniform_chunk_rows), 1);
    for (int batch_index = 0; batch_index < chunk_count; ++batch_index) {
      const auto& chunk = chunk_plan[static_cast<size_t>(batch_index)];
      for (int row = 0; row < uniform_chunk_rows; ++row) {
        const int src_row = chunk.row_start + row;
        if (src_row >= 0 && src_row < static_cast<int>(planned_selection.valid_row_mask.size())) {
          chunk_valid_rows_batch[static_cast<size_t>(batch_index) * static_cast<size_t>(uniform_chunk_rows) + static_cast<size_t>(row)] =
              planned_selection.valid_row_mask[static_cast<size_t>(src_row)];
        }
      }
    }

    bool coherence_ready = false;
    bool raw_score_ready = false;
    std::string raw_score_source = "none";
    const float* debug_patch_features_batch_device = nullptr;
    int debug_patch_rows = 0;
    int debug_patch_cols = 0;
    int debug_feature_dim = 0;
    int debug_aligned_rows = 0;
    int debug_aligned_cols = 0;
    bool debug_runtime_resized_full_chunk = false;
    bool debug_resized_full_chunk = false;
    if (is_truthy_backend_mode(backend_mode_.get()) && !model_script_path_.get().empty()) {
      DinoTorchRuntimeConfig runtime_config;
      runtime_config.inference_backend = inference_backend_.get();
      runtime_config.model_script_path = model_script_path_.get();
      runtime_config.torchscript_init_mode = torchscript_init_mode_.get();
      runtime_config.torch_dtype = torch_dtype_.get();
      runtime_config.imagenet_mean = imagenet_mean_.get();
      runtime_config.imagenet_std = imagenet_std_.get();
      runtime_config.return_patch_features = true;
      runtime_config.return_patch_features_host = false;
      runtime_config.return_final_mask_device = true;
      runtime_config.compute_power_score = false;
      runtime_config.frontend_correction_enable = false;

      DinoTorchRuntimeBatchInput runtime_input;
      runtime_input.batch_size = chunk_count;
      runtime_input.src_rows = uniform_chunk_rows;
      runtime_input.src_cols = src_cols;
      runtime_input.dst_rows = input_height_.get();
      runtime_input.dst_cols = input_width_.get();
      runtime_input.patch_size = patch_size_.get();
      runtime_input.cuda_stream = buffers.processing_stream;
      runtime_input.resolution_hz = resolution_hz;
      runtime_input.span_hz = span_hz;
      runtime_input.corrected_db_batch_device = buffers.corrected_batch_device;

      if (!runtime_) {
        runtime_ = std::make_shared<DinoTorchRuntime>();
      }
      const auto runtime_start_time = std::chrono::steady_clock::now();
      auto runtime_result = runtime_->run_batch(runtime_config, runtime_input);
      timing_profile.runtime_batch_ms = elapsed_ms_since(runtime_start_time);
      if (runtime_result.success) {
        timing_profile.runtime_crop_align_ms = runtime_result.timing.crop_align_ms;
        timing_profile.runtime_resize_ms = runtime_result.timing.resize_ms;
        timing_profile.runtime_model_prep_ms = runtime_result.timing.model_prep_ms;
        timing_profile.runtime_torch_forward_ms = runtime_result.timing.torch_forward_ms;
        timing_profile.runtime_dino_score_ms = runtime_result.timing.dino_score_ms;
        const bool runtime_resized_full_chunk = runtime_result.input_resized_to_target;
        const bool project_full_chunk =
          runtime_resized_full_chunk ||
          (execution_strategy_.get() == "reference_chunks" && uniform_chunk_rows > runtime_result.aligned_rows);
        debug_patch_features_batch_device = runtime_result.patch_features_batch_device;
        debug_patch_rows = runtime_result.patch_rows;
        debug_patch_cols = runtime_result.patch_cols;
        debug_feature_dim = runtime_result.feature_dim;
        debug_aligned_rows = runtime_result.aligned_rows;
        debug_aligned_cols = runtime_result.aligned_cols;
        debug_runtime_resized_full_chunk = runtime_resized_full_chunk;
        debug_resized_full_chunk = project_full_chunk;
        if (runtime_result.patch_features_batch_device != nullptr && runtime_result.patch_rows > 0 && runtime_result.patch_cols > 0 &&
            runtime_result.feature_dim > 0) {
          const auto raw_score_start_time = std::chrono::steady_clock::now();
          raw_score_ready = compute_deweighted_raw_dino_score_gpu_batch_to_device(runtime_result.patch_features_batch_device,
                                                                                  chunk_count,
                                                                                  runtime_result.patch_rows,
                                                                                  runtime_result.patch_cols,
                                                                                  runtime_result.feature_dim,
                                                                                  runtime_result.aligned_rows,
                                                                                  runtime_result.aligned_cols,
                                                                                  uniform_chunk_rows,
                                                                                  src_cols,
                                                                                  raw_dino_positional_deweight_.get(),
                                                                                  project_full_chunk,
                                                                                  buffers.raw_dino_score_batch_device,
                                                                                  buffers.processing_stream);
          if (capture_operator_timing) {
            throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                                "cuda_dino_detector raw score timing synchronization failed");
            timing_profile.raw_score_projection_ms = elapsed_ms_since(raw_score_start_time);
          }
          raw_score_source = raw_score_ready ? "patch_features" : "none";
        } else if (runtime_result.score_maps_device != nullptr) {
          const auto raw_score_start_time = std::chrono::steady_clock::now();
          raw_score_ready = project_runtime_score_batch_to_device(runtime_result.score_maps_device,
                                                                  chunk_count,
                                                                  input_height_.get(),
                                                                  input_width_.get(),
                                                                  runtime_result.aligned_rows,
                                                                  runtime_result.aligned_cols,
                                                                  uniform_chunk_rows,
                                                                  src_cols,
                                                                  project_full_chunk,
                                                                  buffers.raw_dino_score_batch_device,
                                                                  buffers.processing_stream);
          if (capture_operator_timing) {
            throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                                "cuda_dino_detector raw score timing synchronization failed");
            timing_profile.raw_score_projection_ms = elapsed_ms_since(raw_score_start_time);
          }
          raw_score_source = raw_score_ready ? "score_map_fallback" : "none";
        }
        if (meta) {
          meta->set("cuda_dino_runtime_backend_used", runtime_result.backend_used);
          meta->set("cuda_dino_patch_rows", static_cast<uint32_t>(std::max(0, runtime_result.patch_rows)));
          meta->set("cuda_dino_patch_cols", static_cast<uint32_t>(std::max(0, runtime_result.patch_cols)));
          meta->set("cuda_dino_feature_dim", static_cast<uint32_t>(std::max(0, runtime_result.feature_dim)));
        }
      } else {
        std::fprintf(stderr,
                     "[cuda_dino_detector] WARN: DINO runtime batch failed at stage='%s': %s\n",
                     runtime_result.error_stage.c_str(),
                     runtime_result.error_message.c_str());
      }
    } else if (is_truthy_backend_mode(backend_mode_.get()) && compute_count_ == 1) {
      std::fprintf(stderr,
                   "[cuda_dino_detector] WARN: raw DINO stage skipped because model_script_path is empty\n");
    }

    if (capture_operator_timing) {
      throw_if_cuda_error(cudaEventRecord(buffers.coherence_start_event, buffers.processing_stream),
                          "cuda_dino_detector coherence-start event record failed");
    }
    coherence_ready = compute_structure_tensor_gate_gpu_batch_to_device(buffers.corrected_batch_device,
                                                                        chunk_count,
                                                                        uniform_chunk_rows,
                                                                        src_cols,
                                                                        chunk_valid_rows_batch,
                                                                        buffers.coherence_gate_batch_device,
                                                                        buffers.processing_stream);
    if (coherence_ready && capture_operator_timing) {
      throw_if_cuda_error(cudaEventRecord(buffers.coherence_end_event, buffers.processing_stream),
                          "cuda_dino_detector coherence-end event record failed");
    }

    if (capture_operator_timing && coherence_ready) {
      throw_if_cuda_error(cudaEventSynchronize(buffers.coherence_end_event),
                          "cuda_dino_detector coherence timing synchronization failed");
      float coherence_batch_ms = 0.0f;
      throw_if_cuda_error(cudaEventElapsedTime(&coherence_batch_ms,
                                               buffers.coherence_start_event,
                                               buffers.coherence_end_event),
                          "cuda_dino_detector coherence timing elapsed failed");
      timing_profile.coherence_batch_ms = static_cast<double>(coherence_batch_ms);
    }

    if (meta) {
      meta->set("cuda_dino_coherence_ready", coherence_ready);
      meta->set("cuda_dino_raw_score_ready", raw_score_ready);
      meta->set("cuda_dino_raw_score_source", raw_score_source);
    }

    bool hybrid_ready = false;
    const bool enable_mask_post_processing = enable_mask_post_processing_.get();
    const bool preserve_hybrid_debug_outputs = debug_mode_.get() && enable_debug_artifact_host_copy_.get();
    const bool write_operator_artifacts = preserve_hybrid_debug_outputs && !debug_artifact_output_dir_.get().empty();
    if (coherence_ready && raw_score_ready) {
      const auto hybrid_start_time = std::chrono::steady_clock::now();
      CudaHybridStageTiming hybrid_stage_timing;
      hybrid_ready = compute_residual_veto_hybrid_gpu_batch_to_device(buffers.raw_dino_score_batch_device,
                                                                      buffers.coherence_gate_batch_device,
                                                                      chunk_count,
                                                                      uniform_chunk_rows,
                                                                      src_cols,
                                                                      chunk_valid_rows_batch,
                                                                      use_fp16_precision(hybrid_torch_dtype_.get()),
                                                                      enable_mask_post_processing,
                                                                      hybrid_component_min_size_.get(),
                                                                      buffers.hybrid_combined_score_batch_device,
                                                                      buffers.hybrid_final_mask_batch_device,
                                                                      write_operator_artifacts ? buffers.hybrid_filled_mask_batch_device : nullptr,
                                                                      write_operator_artifacts ? buffers.hybrid_component_filtered_mask_batch_device : nullptr,
                                                                      buffers.processing_stream,
                                                                      &hybrid_stage_timing);
      if (capture_operator_timing) {
        throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                            "cuda_dino_detector hybrid timing synchronization failed");
        timing_profile.hybrid_batch_ms = elapsed_ms_since(hybrid_start_time);
        timing_profile.hybrid_normalization_ms = hybrid_stage_timing.normalization_ms;
        timing_profile.hybrid_residual_stack_ms = hybrid_stage_timing.residual_stack_ms;
        timing_profile.hybrid_threshold_extract_ms = hybrid_stage_timing.threshold_extract_ms;
        timing_profile.hybrid_closing_ms = hybrid_stage_timing.closing_ms;
        timing_profile.hybrid_fill_holes_ms = hybrid_stage_timing.fill_holes_ms;
        timing_profile.hybrid_component_filter_ms = hybrid_stage_timing.component_filter_ms;
        timing_profile.hybrid_output_copy_ms = hybrid_stage_timing.output_copy_ms;
      }
    }
    if (meta) {
      meta->set("cuda_dino_hybrid_ready", hybrid_ready);
      meta->set("cuda_dino_enable_mask_post_processing", enable_mask_post_processing);
      meta->set("cuda_dino_hybrid_gpu_post_morphology", hybrid_ready && enable_mask_post_processing);
      meta->set("cuda_dino_hybrid_torch_dtype", hybrid_torch_dtype_.get());
      meta->set("cuda_dino_hybrid_stage_variant",
                hybrid_ready
                    ? (enable_mask_post_processing ? std::string("residual_veto_post_component_filter")
                                                   : std::string("combined_score_threshold_only"))
                    : std::string("none"));
    }

    const bool emit_grouped_merged_mask = emit_grouped_merged_mask_.get();
    if (meta) {
      meta->set("cuda_dino_emit_grouped_merged_mask_requested", emit_grouped_merged_mask);
    }

    std::vector<uint8_t> emitted_mask_detector_host;
    bool emitted_mask_from_host_merge = false;

    const bool host_projection_merge_enabled = hybrid_ready && (write_operator_artifacts || emit_grouped_merged_mask);
    if (host_projection_merge_enabled) {
      const auto debug_device_to_host_start_time = std::chrono::steady_clock::now();
      std::vector<float> corrected_batch;
      std::vector<float> coherence_gate_batch;
      std::vector<float> raw_score_batch;
      std::vector<float> raw_score_deweighted_batch;
      std::vector<float> hybrid_score_batch(batch_elements, 0.0f);
      std::vector<float> hybrid_mask_batch_float(batch_elements, 0.0f);
      std::vector<float> hybrid_keep_freq_batch;
      std::vector<float> hybrid_keep_res_batch;
      std::vector<uint8_t> hybrid_seed_mask_batch;
      std::vector<uint8_t> hybrid_closed_mask_batch;
      std::vector<uint8_t> hybrid_filled_mask_batch;
      std::vector<uint8_t> hybrid_component_filtered_mask_batch;
      std::vector<float> hybrid_seed_freq_thresholds;
      std::vector<float> hybrid_seed_res_thresholds;
      std::vector<float> hybrid_combined_thresholds;
      if (write_operator_artifacts) {
        auto& hybrid_scratch = residual_veto_cuda_scratch();
        if (hybrid_scratch.values_a == nullptr || hybrid_scratch.values_d == nullptr || hybrid_scratch.mask_a == nullptr ||
            hybrid_scratch.mask_b == nullptr || hybrid_scratch.batch_a == nullptr || hybrid_scratch.batch_b == nullptr ||
            hybrid_scratch.batch_c == nullptr) {
          throw std::runtime_error("hybrid debug artifacts requested before residual scratch buffers were initialized");
        }
        corrected_batch.assign(batch_elements, 0.0f);
        coherence_gate_batch.assign(batch_elements, 0.0f);
        raw_score_batch.assign(batch_elements, 0.0f);
        raw_score_deweighted_batch.assign(batch_elements, 0.0f);
        hybrid_keep_freq_batch.assign(batch_elements, 0.0f);
        hybrid_keep_res_batch.assign(batch_elements, 0.0f);
        hybrid_seed_mask_batch.assign(batch_elements, 0);
        hybrid_closed_mask_batch.assign(batch_elements, 0);
        hybrid_filled_mask_batch.assign(batch_elements, 0);
        hybrid_component_filtered_mask_batch.assign(batch_elements, 0);
        hybrid_seed_freq_thresholds.assign(static_cast<size_t>(chunk_count), 0.0f);
        hybrid_seed_res_thresholds.assign(static_cast<size_t>(chunk_count), 0.0f);
        hybrid_combined_thresholds.assign(static_cast<size_t>(chunk_count), 0.0f);
        throw_if_cuda_error(cudaMemcpyAsync(corrected_batch.data(),
                                            buffers.corrected_batch_device,
                                            batch_elements * sizeof(float),
                                            cudaMemcpyDeviceToHost,
                                            buffers.processing_stream),
                            "cuda_dino_detector corrected batch debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(coherence_gate_batch.data(),
                                            buffers.coherence_gate_batch_device,
                                            batch_elements * sizeof(float),
                                            cudaMemcpyDeviceToHost,
                                            buffers.processing_stream),
                            "cuda_dino_detector coherence gate debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(raw_score_batch.data(),
                                            buffers.raw_dino_score_batch_device,
                                            batch_elements * sizeof(float),
                                            cudaMemcpyDeviceToHost,
                                            buffers.processing_stream),
                            "cuda_dino_detector raw DINO debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_keep_freq_batch.data(),
                    hybrid_scratch.values_a,
                    batch_elements * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    buffers.processing_stream),
                "cuda_dino_detector hybrid keep-freq debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_keep_res_batch.data(),
                    hybrid_scratch.values_d,
                    batch_elements * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    buffers.processing_stream),
                "cuda_dino_detector hybrid keep-res debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_seed_mask_batch.data(),
                    hybrid_scratch.mask_a,
                    batch_elements * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost,
                    buffers.processing_stream),
                "cuda_dino_detector hybrid seed mask debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_closed_mask_batch.data(),
                    hybrid_scratch.mask_b,
                    batch_elements * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost,
                    buffers.processing_stream),
                "cuda_dino_detector hybrid closed mask debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_seed_freq_thresholds.data(),
                    hybrid_scratch.batch_a,
                    static_cast<size_t>(chunk_count) * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    buffers.processing_stream),
                "cuda_dino_detector hybrid seed-freq threshold debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_seed_res_thresholds.data(),
                    hybrid_scratch.batch_b,
                    static_cast<size_t>(chunk_count) * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    buffers.processing_stream),
                "cuda_dino_detector hybrid seed-res threshold debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_combined_thresholds.data(),
                    hybrid_scratch.batch_c,
                    static_cast<size_t>(chunk_count) * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    buffers.processing_stream),
                "cuda_dino_detector hybrid combined threshold debug copy failed");
        raw_score_deweighted_batch = raw_score_batch;

        if (debug_patch_features_batch_device != nullptr && debug_patch_rows > 0 && debug_patch_cols > 0 &&
            debug_feature_dim > 0) {
          float* raw_score_debug_device = nullptr;
          throw_if_cuda_error(cudaMalloc(reinterpret_cast<void**>(&raw_score_debug_device), batch_elements * sizeof(float)),
                              "cuda_dino_detector raw debug device allocation failed");
          const bool raw_debug_ready = compute_deweighted_raw_dino_score_gpu_batch_to_device(debug_patch_features_batch_device,
                                                                                              chunk_count,
                                                                                              debug_patch_rows,
                                                                                              debug_patch_cols,
                                                                                              debug_feature_dim,
                                                                                              debug_aligned_rows,
                                                                                              debug_aligned_cols,
                                                                                              uniform_chunk_rows,
                                                                                              src_cols,
                                                                                              0.0f,
                                                                                              debug_resized_full_chunk,
                                                                                              raw_score_debug_device,
                                                                                              buffers.processing_stream);
          if (raw_debug_ready) {
            throw_if_cuda_error(cudaMemcpyAsync(raw_score_batch.data(),
                                                raw_score_debug_device,
                                                batch_elements * sizeof(float),
                                                cudaMemcpyDeviceToHost,
                                                buffers.processing_stream),
                                "cuda_dino_detector raw non-deweighted DINO debug copy failed");
          }
          throw_if_cuda_error(cudaFree(raw_score_debug_device),
                              "cuda_dino_detector raw debug device free failed");
        }
      }
      throw_if_cuda_error(cudaMemcpyAsync(hybrid_score_batch.data(),
                                          buffers.hybrid_combined_score_batch_device,
                                          batch_elements * sizeof(float),
                                          cudaMemcpyDeviceToHost,
                                          buffers.processing_stream),
                          "cuda_dino_detector hybrid combined score debug copy failed");
      throw_if_cuda_error(cudaMemcpyAsync(hybrid_mask_batch_float.data(),
                                          buffers.hybrid_final_mask_batch_device,
                                          batch_elements * sizeof(float),
                                          cudaMemcpyDeviceToHost,
                                          buffers.processing_stream),
                          "cuda_dino_detector hybrid final mask debug copy failed");
      if (write_operator_artifacts) {
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_filled_mask_batch.data(),
                                            buffers.hybrid_filled_mask_batch_device,
                                            batch_elements * sizeof(uint8_t),
                                            cudaMemcpyDeviceToHost,
                                            buffers.processing_stream),
                            "cuda_dino_detector hybrid filled mask debug copy failed");
        throw_if_cuda_error(cudaMemcpyAsync(hybrid_component_filtered_mask_batch.data(),
                                            buffers.hybrid_component_filtered_mask_batch_device,
                                            batch_elements * sizeof(uint8_t),
                                            cudaMemcpyDeviceToHost,
                                            buffers.processing_stream),
                            "cuda_dino_detector hybrid component filtered mask debug copy failed");
      }
      throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                          "cuda_dino_detector hybrid debug synchronization failed");
      timing_profile.debug_device_to_host_ms = elapsed_ms_since(debug_device_to_host_start_time);

      std::vector<DebugChunkResult> live_chunk_results;
      live_chunk_results.reserve(static_cast<size_t>(chunk_count));

      const size_t chunk_elements = static_cast<size_t>(uniform_chunk_rows) * static_cast<size_t>(src_cols);
      const auto debug_grouping_start_time = std::chrono::steady_clock::now();
      const bool use_debug_projection_shape = write_operator_artifacts;
      for (int batch_index = 0; batch_index < chunk_count; ++batch_index) {
        const auto& chunk = chunk_plan[static_cast<size_t>(batch_index)];
        const size_t batch_offset = static_cast<size_t>(batch_index) * chunk_elements;
        std::vector<uint8_t> chunk_valid_rows(chunk_valid_rows_batch.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(batch_index) * static_cast<size_t>(uniform_chunk_rows)),
                                              chunk_valid_rows_batch.begin() + static_cast<std::ptrdiff_t>(static_cast<size_t>(batch_index + 1) * static_cast<size_t>(uniform_chunk_rows)));
        const auto chunk_valid_mask = expand_row_valid_mask(chunk_valid_rows, src_cols);

        std::vector<uint8_t> chunk_mask(chunk_elements, 0);
        std::vector<float> chunk_score(chunk_elements, 0.0f);
        for (size_t index = 0; index < chunk_elements; ++index) {
          chunk_mask[index] = hybrid_mask_batch_float[batch_offset + index] > 0.5f ? 1 : 0;
          chunk_score[index] = hybrid_score_batch[batch_offset + index];
        }

        auto grouping_source = group_mask_regions(chunk_mask,
                    chunk_score,
                    chunk_valid_mask,
                    uniform_chunk_rows,
                    src_cols,
                    filter_detection_mask_.get(),
                    grouping_bridge_freq_px_.get(),
                    grouping_bridge_time_px_.get(),
                    grouping_min_component_size_.get(),
                    grouping_min_freq_span_px_.get(),
                    grouping_min_time_span_px_.get(),
                    static_cast<float>(grouping_min_density_.get()),
                    static_cast<float>(grouping_time_continuity_ratio_.get()));

        DebugChunkResult live_chunk_result;
        live_chunk_result.chunk_index = chunk.chunk_index;
        live_chunk_result.row_start = chunk.row_start;
        live_chunk_result.row_stop = chunk.row_stop;
        live_chunk_result.src_rows = uniform_chunk_rows;
        live_chunk_result.src_cols = src_cols;
        live_chunk_result.dst_rows = use_debug_projection_shape ? std::max(1, input_height_.get()) : uniform_chunk_rows;
        live_chunk_result.dst_cols = use_debug_projection_shape ? std::max(1, input_width_.get()) : src_cols;
        if (write_operator_artifacts) {
          std::vector<float> chunk_keep_freq(hybrid_keep_freq_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                             hybrid_keep_freq_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          std::vector<float> chunk_keep_res(hybrid_keep_res_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                            hybrid_keep_res_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          std::vector<uint8_t> chunk_seed_mask(hybrid_seed_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                               hybrid_seed_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          std::vector<uint8_t> chunk_closed_mask(hybrid_closed_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                                 hybrid_closed_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          std::vector<uint8_t> chunk_filled_mask(hybrid_filled_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                                 hybrid_filled_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          std::vector<uint8_t> chunk_component_filtered_mask(
              hybrid_component_filtered_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
              hybrid_component_filtered_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          live_chunk_result.hybrid_keep_freq = resize_bilinear(chunk_keep_freq,
                                                               uniform_chunk_rows,
                                                               src_cols,
                                                               live_chunk_result.dst_rows,
                                                               live_chunk_result.dst_cols);
          live_chunk_result.hybrid_keep_res = resize_bilinear(chunk_keep_res,
                                                              uniform_chunk_rows,
                                                              src_cols,
                                                              live_chunk_result.dst_rows,
                                                              live_chunk_result.dst_cols);
          live_chunk_result.hybrid_seed_mask = resize_mask_nearest(chunk_seed_mask,
                                                                   uniform_chunk_rows,
                                                                   src_cols,
                                                                   live_chunk_result.dst_rows,
                                                                   live_chunk_result.dst_cols);
          live_chunk_result.hybrid_closed_mask = resize_mask_nearest(chunk_closed_mask,
                                                                     uniform_chunk_rows,
                                                                     src_cols,
                                                                     live_chunk_result.dst_rows,
                                                                     live_chunk_result.dst_cols);
          live_chunk_result.hybrid_filled_mask_source = chunk_filled_mask;
          live_chunk_result.hybrid_filled_mask = resize_mask_nearest(chunk_filled_mask,
                                                                     uniform_chunk_rows,
                                                                     src_cols,
                                                                     live_chunk_result.dst_rows,
                                                                     live_chunk_result.dst_cols);
          live_chunk_result.hybrid_component_filtered_mask_source = chunk_component_filtered_mask;
          live_chunk_result.hybrid_component_filtered_mask = resize_mask_nearest(chunk_component_filtered_mask,
                                                                                 uniform_chunk_rows,
                                                                                 src_cols,
                                                                                 live_chunk_result.dst_rows,
                                                                                 live_chunk_result.dst_cols);
          live_chunk_result.hybrid_seed_freq_threshold = hybrid_seed_freq_thresholds[static_cast<size_t>(batch_index)];
          live_chunk_result.hybrid_seed_res_threshold = hybrid_seed_res_thresholds[static_cast<size_t>(batch_index)];
          live_chunk_result.hybrid_combined_threshold = hybrid_combined_thresholds[static_cast<size_t>(batch_index)];
        }
        live_chunk_result.final_mask_source = chunk_mask;
        live_chunk_result.grouped_mask_source = write_operator_artifacts
                      ? grouping_source.grouped_mask
                      : std::move(grouping_source.grouped_mask);
        if (use_debug_projection_shape) {
          live_chunk_result.final_mask = resize_mask_nearest(chunk_mask,
                                                             uniform_chunk_rows,
                                                             src_cols,
                                                             live_chunk_result.dst_rows,
                                                             live_chunk_result.dst_cols);
          live_chunk_result.combined_score = resize_bilinear(chunk_score,
                                                             uniform_chunk_rows,
                                                             src_cols,
                                                             live_chunk_result.dst_rows,
                                                             live_chunk_result.dst_cols);
          live_chunk_result.grouped_boxes.reserve(grouping_source.boxes.size());
          for (const auto& source_box : grouping_source.boxes) {
            live_chunk_result.grouped_boxes.push_back(scale_box_to_shape(source_box,
                                                                         uniform_chunk_rows,
                                                                         src_cols,
                                                                         live_chunk_result.dst_rows,
                                                                         live_chunk_result.dst_cols));
          }
        } else {
          live_chunk_result.final_mask = std::move(chunk_mask);
          live_chunk_result.combined_score = std::move(chunk_score);
          live_chunk_result.grouped_boxes = std::move(grouping_source.boxes);
        }
        live_chunk_results.push_back(std::move(live_chunk_result));
      }
      timing_profile.debug_chunk_grouping_ms = elapsed_ms_since(debug_grouping_start_time);

      const auto global_merge_start_time = std::chrono::steady_clock::now();
      const auto global_merged = build_global_merged_result(live_chunk_results,
                                                            filter_detection_mask_.get(),
                                                            grouping_bridge_freq_px_.get(),
                                                            grouping_bridge_time_px_.get(),
                                                            grouping_min_component_size_.get(),
                                                            grouping_min_freq_span_px_.get(),
                                                            grouping_min_time_span_px_.get(),
                                                            static_cast<float>(grouping_min_density_.get()),
                                                            static_cast<float>(grouping_time_continuity_ratio_.get()),
                                                            src_rows,
                                                            src_cols,
                                                            planned_selection.valid_row_mask);
                                          timing_profile.global_merge_ms = elapsed_ms_since(global_merge_start_time);
      const auto source_valid_mask = expand_row_valid_mask(planned_selection.valid_row_mask, src_cols);
      if (meta) {
        meta->set("cuda_dino_group_merge_ready", true);
        meta->set("cuda_dino_debug_projected_box_count", static_cast<uint32_t>(global_merged.projected_boxes.size()));
        meta->set("cuda_dino_debug_merged_box_count", static_cast<uint32_t>(global_merged.merged_boxes.size()));
        meta->set("cuda_dino_debug_projected_fraction", static_cast<double>(mean_mask_value(global_merged.projected_grouped_mask)));
        meta->set("cuda_dino_debug_final_fraction", static_cast<double>(mean_mask_value(global_merged.merged_box_mask)));
        meta->set("cuda_dino_debug_connected_fraction", static_cast<double>(connected_fraction(global_merged.merged_box_mask, source_valid_mask)));
      }
      if (emit_grouped_merged_mask) {
        emitted_mask_detector_host.assign(global_merged.merged_box_mask.size(), static_cast<uint8_t>(0));
        for (size_t index = 0; index < global_merged.merged_box_mask.size(); ++index) {
          emitted_mask_detector_host[index] = global_merged.merged_box_mask[index] ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
        }
        emitted_mask_from_host_merge = !emitted_mask_detector_host.empty();
      }
      if (write_operator_artifacts && !live_chunk_results.empty()) {
        const int selected_batch_index = clamp_value(debug_chunk_index_.get(), 0, chunk_count - 1);
        const auto& selected_debug_chunk = live_chunk_results[static_cast<size_t>(selected_batch_index)];
        const auto& selected_chunk = chunk_plan[static_cast<size_t>(selected_batch_index)];
        const size_t chunk_elements = static_cast<size_t>(uniform_chunk_rows) * static_cast<size_t>(src_cols);
        const size_t batch_offset = static_cast<size_t>(selected_batch_index) * chunk_elements;

        std::vector<float> corrected_chunk(corrected_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                           corrected_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
        std::vector<float> coherence_gate_chunk(coherence_gate_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                                coherence_gate_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
        std::vector<float> raw_score_chunk(raw_score_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                           raw_score_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));

        const auto corrected_resized = resize_bilinear(corrected_chunk,
                                                       uniform_chunk_rows,
                                                       src_cols,
                                                       selected_debug_chunk.dst_rows,
                                                       selected_debug_chunk.dst_cols);
        const auto coherence_gate_resized = resize_bilinear(coherence_gate_chunk,
                                                            uniform_chunk_rows,
                                                            src_cols,
                                                            selected_debug_chunk.dst_rows,
                                                            selected_debug_chunk.dst_cols);
        const auto raw_score_resized = resize_bilinear(raw_score_chunk,
                                                       uniform_chunk_rows,
                                                       src_cols,
                                                       selected_debug_chunk.dst_rows,
                                                       selected_debug_chunk.dst_cols);
        const std::vector<float> raw_score_deweighted_chunk(raw_score_deweighted_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                        raw_score_deweighted_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
        const auto raw_score_deweighted_resized = resize_bilinear(raw_score_deweighted_chunk,
                        uniform_chunk_rows,
                        src_cols,
                        selected_debug_chunk.dst_rows,
                        selected_debug_chunk.dst_cols);

        timing_profile.total_compute_ms = elapsed_ms_since(compute_start_time);
        write_operator_artifact_bundle(debug_artifact_output_dir_.get(),
                                       chunk_count,
                                       selected_batch_index,
                                       src_rows,
                                       src_cols,
                                       debug_aligned_rows,
                                       debug_aligned_cols,
                                       debug_runtime_resized_full_chunk,
                                       debug_resized_full_chunk,
                                       selected_chunk,
                                       corrected_resized,
                                       raw_score_resized,
                                       raw_score_deweighted_resized,
                                       coherence_gate_resized,
                                       selected_debug_chunk,
                                       global_merged,
                                       chunk_plan,
                                       timing_profile);
        ++artifact_dump_count_;
        if (meta) {
          meta->set("cuda_dino_debug_artifact_output_dir", debug_artifact_output_dir_.get());
          meta->set("cuda_dino_debug_artifact_dump_count", artifact_dump_count_);
        }
      }
    } else if (meta) {
      meta->set("cuda_dino_group_merge_ready", false);
    }

    if (meta) {
      meta->set("cuda_dino_emit_grouped_merged_mask_applied", emitted_mask_from_host_merge);
    }

    if (hybrid_ready) {
      const size_t emitted_mask_bytes = frame_elements * sizeof(uint8_t);
      auto emitted_mask_device = acquire_pooled_u8_buffer(emitted_mask_bytes);
      if (emitted_mask_from_host_merge) {
        if (emitted_mask_detector_host.size() != frame_elements) {
          throw std::runtime_error("cuda_dino_detector grouped merged mask size does not match emitted frame shape");
        }
        throw_if_cuda_error(cudaMemcpyAsync(emitted_mask_device.get(),
                                            emitted_mask_detector_host.data(),
                                            emitted_mask_bytes,
                                            cudaMemcpyHostToDevice,
                                            buffers.processing_stream),
                            "cuda_dino_detector emitted grouped mask upload failed");
      } else {
        throw_if_cuda_error(cudaMemsetAsync(emitted_mask_device.get(),
                                            0,
                                            emitted_mask_bytes,
                                            buffers.processing_stream),
                            "cuda_dino_detector emitted mask reset failed");
        const int emit_total = chunk_count * uniform_chunk_rows * src_cols;
        const int emit_blocks = (emit_total + threads - 1) / threads;
        cuda_dino_stitch_reference_chunk_mask_kernel<<<emit_blocks, threads, 0, buffers.processing_stream>>>(
            buffers.hybrid_final_mask_batch_device,
            src_rows,
            src_cols,
            buffers.chunk_row_starts_device,
            uniform_chunk_rows,
            chunk_count,
            emitted_mask_device.get());
        throw_if_cuda_error(cudaGetLastError(), "cuda_dino_detector emitted mask stitch kernel launch failed");
      }

      auto emitted_mask_display_device = acquire_pooled_u8_buffer(emitted_mask_bytes);
      {
        constexpr int transpose_block_dim = 16;
        const dim3 transpose_block(transpose_block_dim, transpose_block_dim);
        const dim3 transpose_grid((src_cols + transpose_block_dim - 1) / transpose_block_dim,
                                  (src_rows + transpose_block_dim - 1) / transpose_block_dim);
        transpose_u8_matrix_kernel<<<transpose_grid, transpose_block, 0, buffers.processing_stream>>>(
            emitted_mask_device.get(),
            src_rows,
            src_cols,
            emitted_mask_display_device.get());
        throw_if_cuda_error(cudaGetLastError(), "cuda_dino_detector emitted mask transpose failed");
      }
      throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                          "cuda_dino_detector emitted mask synchronization failed");

      holoscan::ops::DetectorMaskMessage mask_msg;
      mask_msg.device_pixels = std::move(emitted_mask_display_device);
      mask_msg.width = input_cols;
      mask_msg.height = input_rows;
      mask_msg.channel = channel_number;
      mask_msg.frame_number = frame_number;
      if (meta) {
        mask_msg.file_offset_complex = meta->get<uint64_t>("offline_source_file_offset_complex", 0);
        mask_msg.data_end_complex = meta->get<uint64_t>("offline_source_data_end_complex", 0);
        mask_msg.frame_end_complex = meta->get<uint64_t>("offline_source_frame_end_complex", 0);
        mask_msg.complex_samples_read = meta->get<uint64_t>("offline_source_complex_samples_read", 0);
        mask_msg.complex_samples_padded = meta->get<uint64_t>("offline_source_complex_samples_padded", 0);
      }
      op_output.emit(mask_msg, "mask_out");

      if (meta) {
        meta->set("cuda_dino_mask_emitted", true);
      }
    } else if (meta) {
      meta->set("cuda_dino_mask_emitted", false);
    }
  }

  if (meta) {
    meta->set("cuda_dino_backend_mode", backend_mode_.get());
    meta->set("cuda_dino_execution_strategy", execution_strategy_.get());
    meta->set("cuda_dino_chunk_count", static_cast<uint32_t>(std::max(0, chunk_count)));
    meta->set("cuda_dino_ignore_bins_per_side", ignore_bins_per_side);
    meta->set("cuda_dino_freq_bin_hz", resolution_hz);
    meta->set("cuda_dino_total_compute_ms", elapsed_ms_since(compute_start_time));
  }

  timing_profile.total_compute_ms = elapsed_ms_since(compute_start_time);

  if (timing_summary_enable_.get() &&
      (compute_count_ % static_cast<uint64_t>(std::max(1, timing_summary_every_n_.get())) == 0)) {
    std::fprintf(stderr,
                 "[cuda_dino_detector] INFO: processed %llu tensors with GPU-resident power_db/frontend correction in backend_mode='%s' strategy='%s'\n",
                 static_cast<unsigned long long>(compute_count_),
                 backend_mode_.get().c_str(),
                 execution_strategy_.get().c_str());
  }
}

void CudaDinoDetector::release_channel_buffers() {
  for (auto& buffers : channel_buffers_) {
    if (buffers.analysis_tensor_device != nullptr) {
      cudaFree(buffers.analysis_tensor_device);
      buffers.analysis_tensor_device = nullptr;
    }
    if (buffers.power_db_device != nullptr) {
      cudaFree(buffers.power_db_device);
      buffers.power_db_device = nullptr;
    }
    if (buffers.corrected_db_device != nullptr) {
      cudaFree(buffers.corrected_db_device);
      buffers.corrected_db_device = nullptr;
    }
    if (buffers.corrected_batch_device != nullptr) {
      cudaFree(buffers.corrected_batch_device);
      buffers.corrected_batch_device = nullptr;
    }
    if (buffers.coherence_gate_batch_device != nullptr) {
      cudaFree(buffers.coherence_gate_batch_device);
      buffers.coherence_gate_batch_device = nullptr;
    }
    if (buffers.raw_dino_score_batch_device != nullptr) {
      cudaFree(buffers.raw_dino_score_batch_device);
      buffers.raw_dino_score_batch_device = nullptr;
    }
    if (buffers.hybrid_combined_score_batch_device != nullptr) {
      cudaFree(buffers.hybrid_combined_score_batch_device);
      buffers.hybrid_combined_score_batch_device = nullptr;
    }
    if (buffers.hybrid_final_mask_batch_device != nullptr) {
      cudaFree(buffers.hybrid_final_mask_batch_device);
      buffers.hybrid_final_mask_batch_device = nullptr;
    }
    if (buffers.hybrid_filled_mask_batch_device != nullptr) {
      cudaFree(buffers.hybrid_filled_mask_batch_device);
      buffers.hybrid_filled_mask_batch_device = nullptr;
    }
    if (buffers.hybrid_component_filtered_mask_batch_device != nullptr) {
      cudaFree(buffers.hybrid_component_filtered_mask_batch_device);
      buffers.hybrid_component_filtered_mask_batch_device = nullptr;
    }
    if (buffers.row_stat_device != nullptr) {
      cudaFree(buffers.row_stat_device);
      buffers.row_stat_device = nullptr;
    }
    if (buffers.row_smooth_device != nullptr) {
      cudaFree(buffers.row_smooth_device);
      buffers.row_smooth_device = nullptr;
    }
    if (buffers.frontend_reference_device != nullptr) {
      cudaFree(buffers.frontend_reference_device);
      buffers.frontend_reference_device = nullptr;
    }
    if (buffers.chunk_row_starts_device != nullptr) {
      cudaFree(buffers.chunk_row_starts_device);
      buffers.chunk_row_starts_device = nullptr;
    }
    if (buffers.copy_complete_event != nullptr) {
      cudaEventDestroy(buffers.copy_complete_event);
      buffers.copy_complete_event = nullptr;
    }
    if (buffers.coherence_start_event != nullptr) {
      cudaEventDestroy(buffers.coherence_start_event);
      buffers.coherence_start_event = nullptr;
    }
    if (buffers.coherence_end_event != nullptr) {
      cudaEventDestroy(buffers.coherence_end_event);
      buffers.coherence_end_event = nullptr;
    }
    if (buffers.processing_stream != nullptr) {
      cudaStreamDestroy(buffers.processing_stream);
      buffers.processing_stream = nullptr;
    }
    buffers.frame_elements = 0;
    buffers.batch_elements = 0;
    buffers.chunk_row_start_capacity = 0;
    buffers.row_elements = 0;
  }
  channel_buffers_.clear();
}

}  // namespace holoscan::ops
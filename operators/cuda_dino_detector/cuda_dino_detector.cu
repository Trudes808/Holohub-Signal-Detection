// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_dino_detector.hpp"
#include "cuda_dino_torch_helpers.hpp"
#include "cuda_dino_types.hpp"

#include <dinov3_torch_runtime.hpp>

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
#include <optional>
#include <sstream>
#include <stdexcept>

namespace holoscan::ops {

namespace {

bool use_fp16_precision(const std::string& dtype_text) {
  std::string lowered = dtype_text;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return lowered == "fp16" || lowered == "half" || lowered == "float16";
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
  std::vector<uint8_t> hybrid_filled_mask;
  std::vector<uint8_t> hybrid_filled_mask_source;
  std::vector<uint8_t> hybrid_component_filtered_mask;
  std::vector<uint8_t> hybrid_component_filtered_mask_source;
  std::vector<uint8_t> final_mask;
  std::vector<uint8_t> final_mask_source;
  std::vector<float> combined_score;
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

template <typename T>
T clamp_value(T value, T low, T high) {
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
  const int col_start = max(0, col - radius_cols);
  const int col_stop = min(cols - 1, col + radius_cols);
  const size_t base_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane) +
                             static_cast<size_t>(row) * static_cast<size_t>(cols);
  float sum = 0.0f;
  int count = 0;
  for (int src_col = col_start; src_col <= col_stop; ++src_col) {
    sum += input[base_offset + static_cast<size_t>(src_col)];
    ++count;
  }
  output[idx] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
}

__global__ void directional_box_mean_rows_batch_kernel(const float* input,
                                                       int batch_size,
                                                       int rows,
                                                       int cols,
                                                       int radius_rows,
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
  const int col = local_index % cols;
  const int row_start = max(0, row - radius_rows);
  const int row_stop = min(rows - 1, row + radius_rows);
  float sum = 0.0f;
  int count = 0;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);
  for (int src_row = row_start; src_row <= row_stop; ++src_row) {
    sum += input[batch_offset + flat_index(cols, src_row, col)];
    ++count;
  }
  output[idx] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
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

__global__ void fill_holes_init_kernel(const uint8_t* mask,
                                       int batch_size,
                                       int rows,
                                       int cols,
                                       uint8_t* background,
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
  const uint8_t bg = mask[idx] == 0 ? 1 : 0;
  background[idx] = bg;
  exterior[idx] = (bg != 0 && (row == 0 || row == rows - 1 || col == 0 || col == cols - 1)) ? 1 : 0;
}

__global__ void fill_holes_expand_kernel(const uint8_t* background,
                                         const uint8_t* current,
                                         int batch_size,
                                         int rows,
                                         int cols,
                                         uint8_t* next,
                                         uint32_t* changed) {
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

  uint8_t value = current[idx];
  if (value == 0 && background[idx] != 0) {
    for (int d_row = -1; d_row <= 1 && value == 0; ++d_row) {
      const int src_row = row + d_row;
      if (src_row < 0 || src_row >= rows) {
        continue;
      }
      for (int d_col = -1; d_col <= 1; ++d_col) {
        const int src_col = col + d_col;
        if (src_col < 0 || src_col >= cols) {
          continue;
        }
        if (current[batch_offset + flat_index(cols, src_row, src_col)] != 0) {
          value = 1;
          break;
        }
      }
    }
  }

  next[idx] = value;
  if (value != current[idx]) {
    atomicExch(changed, 1U);
  }
}

__global__ void fill_holes_finalize_kernel(const uint8_t* mask,
                                           const uint8_t* background,
                                           const uint8_t* exterior,
                                           int total,
                                           uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = (mask[idx] != 0 || (background[idx] != 0 && exterior[idx] == 0)) ? 1 : 0;
}

__global__ void component_filter_init_labels_kernel(const uint8_t* mask,
                                                    int batch_size,
                                                    int rows,
                                                    int cols,
                                                    int* labels) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }
  const int local_index = idx % plane;
  labels[idx] = mask[idx] != 0 ? (local_index + 1) : 0;
}

__global__ void component_filter_propagate_labels_kernel(const uint8_t* mask,
                                                         const int* current,
                                                         int batch_size,
                                                         int rows,
                                                         int cols,
                                                         int* next,
                                                         uint32_t* changed) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int plane = rows * cols;
  const int total = batch_size * plane;
  if (idx >= total) {
    return;
  }

  if (mask[idx] == 0) {
    next[idx] = 0;
    return;
  }

  const int batch_index = idx / plane;
  const int local_index = idx - batch_index * plane;
  const int row = local_index / cols;
  const int col = local_index % cols;
  const size_t batch_offset = static_cast<size_t>(batch_index) * static_cast<size_t>(plane);

  int best = current[idx];
  for (int d_row = -1; d_row <= 1; ++d_row) {
    const int src_row = row + d_row;
    if (src_row < 0 || src_row >= rows) {
      continue;
    }
    for (int d_col = -1; d_col <= 1; ++d_col) {
      const int src_col = col + d_col;
      if (src_col < 0 || src_col >= cols) {
        continue;
      }
      const size_t neighbor = batch_offset + flat_index(cols, src_row, src_col);
      if (mask[neighbor] != 0) {
        best = max(best, current[neighbor]);
      }
    }
  }

  next[idx] = best;
  if (best != current[idx]) {
    atomicExch(changed, 1U);
  }
}

__global__ void component_filter_count_labels_kernel(const int* labels,
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
  if (label <= 0) {
    return;
  }
  const int batch_index = idx / plane;
  atomicAdd(&counts[batch_index * (plane + 1) + label], 1);
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
  const int batch_index = idx / plane;
  const int label = labels[idx];
  output[idx] = (label > 0 && counts[batch_index * (plane + 1) + label] >= min_size) ? 1 : 0;
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
  chunk_debug_summary << "  \"dst_rows\": " << selected_debug_chunk.dst_rows << ",\n";
  chunk_debug_summary << "  \"dst_cols\": " << selected_debug_chunk.dst_cols << ",\n";
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
  chunk_debug_summary << "  \"corrected_resized_npy\": \"" << json_escape(corrected_resized_path.string()) << "\",\n";
  chunk_debug_summary << "  \"dino_score_raw_npy\": \"" << json_escape(raw_score_path.string()) << "\",\n";
  chunk_debug_summary << "  \"dino_score_raw_deweighted_npy\": \"" << json_escape(raw_score_deweighted_path.string()) << "\",\n";
  chunk_debug_summary << "  \"coherence_gate_npy\": \"" << json_escape(coherence_gate_path.string()) << "\",\n";
  chunk_debug_summary << "  \"combined_score_npy\": \"" << json_escape(combined_score_path.string()) << "\",\n";
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

}  // namespace

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
  auto& scratch = fill_holes_cuda_scratch();
  if (!scratch.ensure_capacity(total)) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const int threads = 256;
  const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  fill_holes_init_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                          batch_size,
                                                          rows,
                                                          cols,
                                                          scratch.background,
                                                          scratch.grown_a);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  uint8_t* current = scratch.grown_a;
  uint8_t* next = scratch.grown_b;
  const int max_iterations = std::max(1, rows + cols);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (cudaMemsetAsync(scratch.changed, 0, sizeof(uint32_t), stream) != cudaSuccess) {
      return false;
    }
    fill_holes_expand_kernel<<<blocks, threads, 0, stream>>>(scratch.background,
                                                              current,
                                                              batch_size,
                                                              rows,
                                                              cols,
                                                              next,
                                                              scratch.changed);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }

    uint32_t changed_host = 0;
    if (cudaMemcpyAsync(&changed_host, scratch.changed, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return false;
    }

    std::swap(current, next);
    if (changed_host == 0) {
      break;
    }
  }

  fill_holes_finalize_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                              scratch.background,
                                                              current,
                                                              static_cast<int>(total),
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
  if (!scratch.ensure_capacity(total, static_cast<size_t>(batch_size) * (plane + 1))) {
    return false;
  }

  cudaStream_t stream = cuda_stream != nullptr ? cuda_stream : cudaStreamPerThread;
  const int threads = 256;
  const int blocks = static_cast<int>((total + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  component_filter_init_labels_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                       batch_size,
                                                                       rows,
                                                                       cols,
                                                                       scratch.labels_a);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  int* current = scratch.labels_a;
  int* next = scratch.labels_b;
  const int max_iterations = std::max(1, rows + cols);
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (cudaMemsetAsync(scratch.changed, 0, sizeof(uint32_t), stream) != cudaSuccess) {
      return false;
    }
    component_filter_propagate_labels_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                              current,
                                                                              batch_size,
                                                                              rows,
                                                                              cols,
                                                                              next,
                                                                              scratch.changed);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }

    uint32_t changed_host = 0;
    if (cudaMemcpyAsync(&changed_host, scratch.changed, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return false;
    }

    std::swap(current, next);
    if (changed_host == 0) {
      break;
    }
  }

  const size_t count_total = static_cast<size_t>(batch_size) * (plane + 1);
  if (cudaMemsetAsync(scratch.component_counts, 0, count_total * sizeof(int), stream) != cudaSuccess) {
    return false;
  }
  component_filter_count_labels_kernel<<<blocks, threads, 0, stream>>>(current,
                                                                        batch_size,
                                                                        rows,
                                                                        cols,
                                                                        scratch.component_counts);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  component_filter_finalize_kernel<<<blocks, threads, 0, stream>>>(mask_batch_device,
                                                                    current,
                                                                    batch_size,
                                                                    rows,
                                                                    cols,
                                                                    min_size,
                                                                    scratch.component_counts,
                                                                    output_mask_batch_device);
  return cudaGetLastError() == cudaSuccess;
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

  directional_box_mean_cols_batch_kernel<<<blocks, threads, 0, stream>>>(corrected_batch_device,
                                                                          batch_size,
                                                                          rows,
                                                                          cols,
                                                                          bg_time_radius,
                                                                          scratch.background_cols);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_box_mean_rows_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.background_cols,
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
  directional_box_mean_cols_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.residual,
                                                                          batch_size,
                                                                          rows,
                                                                          cols,
                                                                          kDirectionalTimeRadius,
                                                                          scratch.background_cols);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  directional_box_mean_rows_batch_kernel<<<blocks, threads, 0, stream>>>(scratch.residual,
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
               "[cuda_dino_detector] INFO: initialized backend_mode='%s' execution_strategy='%s' debug_mode=%d host_copy_debug_only=%d max_tokens_per_inference=%d input=%dx%d patch_size=%d emit_stride=%d chunk_bw_hz=%.3f overlap_hz=%.3f frontend_correction_enable=%d\n",
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
               frontend_correction_enable_.get() ? 1 : 0);
}

void CudaDinoDetector::stop() {
  release_channel_buffers();
  Operator::stop();
}

void CudaDinoDetector::compute(holoscan::InputContext& op_input,
                               holoscan::OutputContext& op_output,
                               holoscan::ExecutionContext& context) {
  static_cast<void>(op_output);
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

  const int src_rows = static_cast<int>(fft_tensor.Size(0));
  const int src_cols = static_cast<int>(fft_tensor.Size(1));
  if (src_rows <= 0 || src_cols <= 0) {
    std::fprintf(stderr,
                 "[cuda_dino_detector] WARN: received empty tensor shape %dx%d\n",
                 src_rows,
                 src_cols);
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
  auto& buffers = channel_buffers_[local_channel_index];
  const size_t frame_elements = static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols);
  const bool capture_operator_timing =
      debug_mode_.get() && enable_debug_artifact_host_copy_.get() && !debug_artifact_output_dir_.get().empty();
  OperatorTimingProfile timing_profile;

  if (buffers.processing_stream == nullptr) {
    throw_if_cuda_error(cudaStreamCreateWithFlags(&buffers.processing_stream, cudaStreamNonBlocking),
                        "cuda_dino_detector processing stream creation failed");
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

  throw_if_cuda_error(cudaMemcpyAsync(buffers.analysis_tensor_device,
                                      fft_tensor.Data(),
                                      frame_elements * sizeof(cuda_dino_complex),
                                      cudaMemcpyDeviceToDevice,
                                      fft_stream),
                      "cuda_dino_detector analysis tensor copy failed");

  ++compute_count_;
  if (!startup_log_emitted_) {
    startup_log_emitted_ = true;
    std::fprintf(stderr,
                 "[cuda_dino_detector] INFO: scaffold active, received first tensor with shape %ldx%ld; fast path stays device-resident and host debug copies remain opt-in\n",
                 static_cast<long>(fft_tensor.Size(0)),
                 static_cast<long>(fft_tensor.Size(1)));
  }

  const int stride = std::max(1, emit_stride_.get());
  constexpr int threads = 256;
  const int blocks = static_cast<int>((frame_elements + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads));

  cudaEvent_t copy_complete_event = nullptr;
  throw_if_cuda_error(cudaEventCreateWithFlags(&copy_complete_event, cudaEventDisableTiming),
                      "cuda_dino_detector copy-complete event creation failed");
  throw_if_cuda_error(cudaEventRecord(copy_complete_event, fft_stream),
                      "cuda_dino_detector copy-complete event record failed");
  throw_if_cuda_error(cudaStreamWaitEvent(buffers.processing_stream, copy_complete_event, 0),
                      "cuda_dino_detector processing stream wait failed");

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

    const auto coherence_start_time = std::chrono::steady_clock::now();
    const bool coherence_ready = compute_structure_tensor_gate_gpu_batch_to_device(buffers.corrected_batch_device,
                                                                                   chunk_count,
                                                                                   uniform_chunk_rows,
                                                                                   src_cols,
                                                                                   chunk_valid_rows_batch,
                                                                                   buffers.coherence_gate_batch_device,
                                                                                   buffers.processing_stream);
    if (capture_operator_timing) {
      throw_if_cuda_error(cudaStreamSynchronize(buffers.processing_stream),
                          "cuda_dino_detector coherence timing synchronization failed");
      timing_profile.coherence_batch_ms = elapsed_ms_since(coherence_start_time);
    }

    bool raw_score_ready = false;
    std::string raw_score_source = "none";
    const float* debug_patch_features_batch_device = nullptr;
    int debug_patch_rows = 0;
    int debug_patch_cols = 0;
    int debug_feature_dim = 0;
    int debug_aligned_rows = 0;
    int debug_aligned_cols = 0;
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
        const bool resized_full_chunk = runtime_result.input_resized_to_target;
        debug_patch_features_batch_device = runtime_result.patch_features_batch_device;
        debug_patch_rows = runtime_result.patch_rows;
        debug_patch_cols = runtime_result.patch_cols;
        debug_feature_dim = runtime_result.feature_dim;
        debug_aligned_rows = runtime_result.aligned_rows;
        debug_aligned_cols = runtime_result.aligned_cols;
        debug_resized_full_chunk = resized_full_chunk;
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
                                                                                  resized_full_chunk,
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
                                                                  resized_full_chunk,
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

    if (meta) {
      meta->set("cuda_dino_coherence_ready", coherence_ready);
      meta->set("cuda_dino_raw_score_ready", raw_score_ready);
      meta->set("cuda_dino_raw_score_source", raw_score_source);
    }

    bool hybrid_ready = false;
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
                                                                      hybrid_component_min_size_.get(),
                                                                      buffers.hybrid_combined_score_batch_device,
                                                                      buffers.hybrid_final_mask_batch_device,
                                                                      buffers.hybrid_filled_mask_batch_device,
                                                                      buffers.hybrid_component_filtered_mask_batch_device,
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
      meta->set("cuda_dino_hybrid_gpu_post_morphology", hybrid_ready);
      meta->set("cuda_dino_hybrid_torch_dtype", hybrid_torch_dtype_.get());
      meta->set("cuda_dino_hybrid_stage_variant",
                hybrid_ready ? std::string("residual_veto_post_component_filter") : std::string("none"));
    }

    const bool debug_projection_merge_enabled =
        hybrid_ready && debug_mode_.get() && enable_debug_artifact_host_copy_.get();
    if (debug_projection_merge_enabled) {
      const bool write_operator_artifacts = !debug_artifact_output_dir_.get().empty();
      const auto debug_device_to_host_start_time = std::chrono::steady_clock::now();
      std::vector<float> corrected_batch;
      std::vector<float> coherence_gate_batch;
      std::vector<float> raw_score_batch;
      std::vector<float> raw_score_deweighted_batch;
      std::vector<float> hybrid_score_batch(batch_elements, 0.0f);
      std::vector<float> hybrid_mask_batch_float(batch_elements, 0.0f);
      std::vector<uint8_t> hybrid_filled_mask_batch;
      std::vector<uint8_t> hybrid_component_filtered_mask_batch;
      if (write_operator_artifacts) {
        corrected_batch.assign(batch_elements, 0.0f);
        coherence_gate_batch.assign(batch_elements, 0.0f);
        raw_score_batch.assign(batch_elements, 0.0f);
        raw_score_deweighted_batch.assign(batch_elements, 0.0f);
        hybrid_filled_mask_batch.assign(batch_elements, 0);
        hybrid_component_filtered_mask_batch.assign(batch_elements, 0);
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

      std::vector<DebugChunkResult> debug_chunk_results;
      debug_chunk_results.reserve(static_cast<size_t>(chunk_count));

      const size_t chunk_elements = static_cast<size_t>(uniform_chunk_rows) * static_cast<size_t>(src_cols);
      const auto debug_grouping_start_time = std::chrono::steady_clock::now();
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

        const auto grouping_source = group_mask_regions(chunk_mask,
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

        DebugChunkResult debug_chunk_result;
        debug_chunk_result.chunk_index = chunk.chunk_index;
        debug_chunk_result.row_start = chunk.row_start;
        debug_chunk_result.row_stop = chunk.row_stop;
        debug_chunk_result.src_rows = uniform_chunk_rows;
        debug_chunk_result.src_cols = src_cols;
        debug_chunk_result.dst_rows = std::max(1, input_height_.get());
        debug_chunk_result.dst_cols = std::max(1, input_width_.get());
        if (write_operator_artifacts) {
          std::vector<uint8_t> chunk_filled_mask(hybrid_filled_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
                                                 hybrid_filled_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          std::vector<uint8_t> chunk_component_filtered_mask(
              hybrid_component_filtered_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset),
              hybrid_component_filtered_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_offset + chunk_elements));
          debug_chunk_result.hybrid_filled_mask_source = chunk_filled_mask;
          debug_chunk_result.hybrid_filled_mask = resize_mask_nearest(chunk_filled_mask,
                                                                      uniform_chunk_rows,
                                                                      src_cols,
                                                                      debug_chunk_result.dst_rows,
                                                                      debug_chunk_result.dst_cols);
          debug_chunk_result.hybrid_component_filtered_mask_source = chunk_component_filtered_mask;
          debug_chunk_result.hybrid_component_filtered_mask = resize_mask_nearest(chunk_component_filtered_mask,
                                                                                  uniform_chunk_rows,
                                                                                  src_cols,
                                                                                  debug_chunk_result.dst_rows,
                                                                                  debug_chunk_result.dst_cols);
        }
        debug_chunk_result.final_mask_source = chunk_mask;
        debug_chunk_result.final_mask = resize_mask_nearest(chunk_mask,
                                                            uniform_chunk_rows,
                                                            src_cols,
                                                            debug_chunk_result.dst_rows,
                                                            debug_chunk_result.dst_cols);
        debug_chunk_result.combined_score = resize_bilinear(chunk_score,
                                                            uniform_chunk_rows,
                                                            src_cols,
                                                            debug_chunk_result.dst_rows,
                                                            debug_chunk_result.dst_cols);
        debug_chunk_result.grouped_mask_source = grouping_source.grouped_mask;
        debug_chunk_result.grouped_boxes.reserve(grouping_source.boxes.size());
        for (const auto& source_box : grouping_source.boxes) {
          debug_chunk_result.grouped_boxes.push_back(scale_box_to_shape(source_box,
                                                                        uniform_chunk_rows,
                                                                        src_cols,
                                                                        debug_chunk_result.dst_rows,
                                                                        debug_chunk_result.dst_cols));
        }
        debug_chunk_results.push_back(std::move(debug_chunk_result));
      }
      timing_profile.debug_chunk_grouping_ms = elapsed_ms_since(debug_grouping_start_time);

      const auto global_merge_start_time = std::chrono::steady_clock::now();
      const auto global_merged = build_global_merged_result(debug_chunk_results,
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
      if (write_operator_artifacts && !debug_chunk_results.empty()) {
        const int selected_batch_index = clamp_value(debug_chunk_index_.get(), 0, chunk_count - 1);
        const auto& selected_debug_chunk = debug_chunk_results[static_cast<size_t>(selected_batch_index)];
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

  if (copy_complete_event != nullptr) {
    cudaEventDestroy(copy_complete_event);
  }

  if (timing_summary_enable_.get() && (compute_count_ % static_cast<uint64_t>(stride) == 0) &&
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
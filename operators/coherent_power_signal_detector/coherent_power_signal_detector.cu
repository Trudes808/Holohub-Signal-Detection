// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "coherent_power_signal_detector.hpp"

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
#include <vector>

namespace {

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

enum ReferenceTimingStageIndex : size_t {
  kReferenceFrontendStage = 0,
  kReferenceChunkWallStage,
  kReferenceChunkSumStage,
  kReferenceChunkMaxStage,
  kReferenceMergeStage,
  kReferenceGroupingStage,
};

constexpr std::array<const char*, holoscan::ops::CoherentPowerSignalDetector::kReferenceTimingStageCount>
    kReferenceTimingStageNames = {
        "reference_frontend_ms",
        "reference_chunk_wall_ms",
        "reference_chunk_sum_ms",
        "reference_chunk_max_ms",
        "reference_merge_ms",
        "reference_grouping_ms",
    };

enum ChunkTimingStageIndex : size_t {
  kChunkStructureTensorStage = 0,
  kChunkPowerSupportStage,
  kChunkScoreThresholdStage,
  kChunkMaskSmoothStage,
  kChunkGroupingStage,
};

constexpr std::array<const char*, holoscan::ops::CoherentPowerSignalDetector::kChunkTimingStageCount>
  kChunkTimingStageNames = {
    "chunk_structure_tensor_ms",
    "chunk_power_support_ms",
    "chunk_score_threshold_ms",
    "chunk_mask_smooth_ms",
    "chunk_grouping_ms",
  };

enum PowerSupportTimingStageIndex : size_t {
  kPowerSupportFloorEstimateStage = 0,
  kPowerSupportLocalRelativeStage,
  kPowerSupportAbsoluteAssistStage,
  kPowerSupportBlendStage,
};

constexpr std::array<const char*, holoscan::ops::CoherentPowerSignalDetector::kPowerSupportTimingStageCount>
  kPowerSupportTimingStageNames = {
    "power_support_floor_estimate_ms",
    "power_support_local_relative_ms",
    "power_support_absolute_assist_ms",
    "power_support_blend_ms",
  };

constexpr int kQuantileHistogramBins = 1024;

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
  int freq_span = 0;
  int time_span = 0;
  int filled_area = 0;
  float density = 0.0f;
  float bbox_density = 0.0f;
  float envelope_density = 0.0f;
  float score_mean = 0.0f;
  float score_peak = 0.0f;
  std::string split_role = "unsplit";
  bool split_applied = false;
  int parent_component_id = -1;
  std::vector<int> source_chunk_indices;
};

struct DetectionChunkResult {
  ChunkPlanEntry chunk;
  std::vector<float> combined_raw_px;
  std::vector<float> score_px;
  std::vector<uint8_t> support_px;
  std::vector<uint8_t> mask_px;
  std::vector<uint8_t> grouped_mask;
  std::vector<uint8_t> valid_row_mask;
  std::vector<uint8_t> valid_score_mask;
  std::vector<DetectionBox> grouped_boxes;
  float support_threshold = 0.0f;
  float score_threshold = 0.0f;
  double compute_ms = 0.0;
  std::array<double, holoscan::ops::CoherentPowerSignalDetector::kChunkTimingStageCount> stage_ms {};
  std::array<double, holoscan::ops::CoherentPowerSignalDetector::kPowerSupportTimingStageCount> power_support_stage_ms {};
};

class ChunkWorkerPool {
 public:
  explicit ChunkWorkerPool(size_t worker_count) {
    worker_count = std::max<size_t>(1, worker_count);
    workers_.reserve(worker_count);
    for (size_t index = 0; index < worker_count; ++index) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ~ChunkWorkerPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
      has_work_ = false;
    }
    work_cv_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  std::vector<DetectionChunkResult> run(size_t task_count,
                                        const std::function<DetectionChunkResult(size_t)>& task_fn) {
    std::vector<DetectionChunkResult> results(task_count);
    if (task_count == 0) {
      return results;
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      task_fn_ = task_fn;
      results_ = &results;
      task_count_ = task_count;
      next_index_.store(0, std::memory_order_relaxed);
      active_workers_ = 0;
      abort_.store(false, std::memory_order_relaxed);
      exception_ = nullptr;
      has_work_ = true;
    }

    work_cv_.notify_all();

    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this]() { return !has_work_; });
    task_fn_ = {};
    results_ = nullptr;
    if (exception_ != nullptr) {
      std::rethrow_exception(exception_);
    }
    return results;
  }

 private:
  void worker_loop() {
    for (;;) {
      std::function<DetectionChunkResult(size_t)> task_fn;
      std::vector<DetectionChunkResult>* results = nullptr;
      size_t task_count = 0;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [this]() { return stop_ || has_work_; });
        if (stop_) {
          return;
        }
        task_fn = task_fn_;
        results = results_;
        task_count = task_count_;
        ++active_workers_;
      }

      while (!abort_.load(std::memory_order_relaxed)) {
        const size_t index = next_index_.fetch_add(1, std::memory_order_relaxed);
        if (index >= task_count) {
          break;
        }
        try {
          (*results)[index] = task_fn(index);
        } catch (...) {
          abort_.store(true, std::memory_order_relaxed);
          std::lock_guard<std::mutex> lock(mutex_);
          if (exception_ == nullptr) {
            exception_ = std::current_exception();
          }
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--active_workers_ == 0) {
          has_work_ = false;
          done_cv_.notify_one();
        }
      }
    }
  }

  std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  std::function<DetectionChunkResult(size_t)> task_fn_;
  std::vector<DetectionChunkResult>* results_ = nullptr;
  std::atomic<size_t> next_index_ {0};
  std::atomic<bool> abort_ {false};
  size_t task_count_ = 0;
  size_t active_workers_ = 0;
  bool has_work_ = false;
  bool stop_ = false;
  std::exception_ptr exception_;
};

ChunkWorkerPool& chunk_worker_pool() {
  static ChunkWorkerPool pool(std::max<size_t>(1, std::min<size_t>(16, std::thread::hardware_concurrency())));
  return pool;
}

struct ReferenceMergeScratch {
  std::vector<float> merged_score_sum;
  std::vector<float> merged_weight;
  std::vector<float> combined_score;
  std::vector<float> merged_score;
  std::vector<uint8_t> merged_support;
  std::vector<uint8_t> valid_score_mask;
  std::vector<uint8_t> raw_merged_mask;
  std::vector<size_t> active_indices;
  size_t frame_size = 0;

  void ensure_capacity(size_t requested_size) {
    if (frame_size == requested_size) {
      return;
    }
    frame_size = requested_size;
    merged_score_sum.assign(frame_size, 0.0f);
    merged_weight.assign(frame_size, 0.0f);
    combined_score.assign(frame_size, 0.0f);
    merged_score.assign(frame_size, 0.0f);
    merged_support.assign(frame_size, 0);
    valid_score_mask.assign(frame_size, 0);
    raw_merged_mask.assign(frame_size, 0);
    active_indices.clear();
    active_indices.reserve(frame_size / 8);
  }

  void begin_frame() {
    for (const size_t index : active_indices) {
      merged_score_sum[index] = 0.0f;
      merged_weight[index] = 0.0f;
      combined_score[index] = 0.0f;
      merged_score[index] = 0.0f;
      merged_support[index] = 0;
      valid_score_mask[index] = 0;
      raw_merged_mask[index] = 0;
    }
    active_indices.clear();
  }
};

ReferenceMergeScratch& reference_merge_scratch() {
  thread_local auto* scratch = new ReferenceMergeScratch();
  return *scratch;
}

struct GroupingResult {
  std::vector<uint8_t> grouped_mask;
  std::vector<DetectionBox> boxes;
  float peak_score_floor = 0.0f;
};

struct PipelineSummary {
  std::vector<float> final_mask;
  int subsection_count = 0;
  int grouped_box_count = 0;
  int ignore_bins_per_side = 0;
  float merged_threshold = 0.0f;
  float seed_threshold = 0.0f;
  std::array<double, holoscan::ops::CoherentPowerSignalDetector::kReferenceTimingStageCount> reference_stage_ms {};
  std::array<double, holoscan::ops::CoherentPowerSignalDetector::kChunkTimingStageCount> chunk_stage_sum_ms {};
  std::array<double, holoscan::ops::CoherentPowerSignalDetector::kChunkTimingStageCount> chunk_stage_peak_ms {};
  std::array<double, holoscan::ops::CoherentPowerSignalDetector::kPowerSupportTimingStageCount> power_support_stage_sum_ms {};
  std::array<double, holoscan::ops::CoherentPowerSignalDetector::kPowerSupportTimingStageCount> power_support_stage_peak_ms {};
};

struct FastGpuMetadataSummary {
  int subsection_count = 1;
  int grouped_box_count = 0;
  int ignore_bins_per_side = 0;
  float merged_threshold = 0.0f;
  float seed_threshold = 0.0f;
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

__global__ void coherent_power_fast_score_kernel(const float* corrected,
                                                 const float* time_mean,
                                                 const float* freq_mean,
                                                 const float* background,
                                                 int rows,
                                                 int cols,
                                                 int ignore_bins_per_side,
                                                 float coherence_weight,
                                                 float power_weight,
                                                 float power_floor_db,
                                                 float power_span_db,
                                                 float coherence_floor_db,
                                                 float coherence_span_db,
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
  const float coherence_db = time_mean[idx] - freq_mean[idx];
  const float support = fminf(fmaxf((support_db - power_floor_db) / fmaxf(power_span_db, 1e-6f), 0.0f), 1.0f);
  const float coherence = fminf(fmaxf((coherence_db - coherence_floor_db) / fmaxf(coherence_span_db, 1e-6f), 0.0f), 1.0f);
  const float combined = coherence_weight * coherence + power_weight * support;
  score[idx] = combined;
  mask[idx] = (combined >= score_threshold && support > 0.0f && coherence > 0.0f) ? 1 : 0;
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

__global__ void coherent_power_subtract_clamp_kernel(const float* input,
                                                     const float* baseline,
                                                     int total,
                                                     float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  output[idx] = fmaxf(input[idx] - baseline[idx], 0.0f);
}

__global__ void coherent_power_gaussian_rows_kernel(const float* input,
                                                    int rows,
                                                    int cols,
                                                    int radius,
                                                    float sigma,
                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const int col = idx % cols;
  float sum = 0.0f;
  float weight_sum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const int src_col = max(0, min(cols - 1, col + offset));
    const float weight = expf(-(static_cast<float>(offset * offset)) / (2.0f * sigma * sigma));
    sum += input[flat_index(rows, cols, row, src_col)] * weight;
    weight_sum += weight;
  }
  output[idx] = weight_sum > 0.0f ? sum / weight_sum : input[idx];
}

__global__ void coherent_power_gaussian_cols_kernel(const float* input,
                                                    int rows,
                                                    int cols,
                                                    int radius,
                                                    float sigma,
                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const int col = idx % cols;
  float sum = 0.0f;
  float weight_sum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const int src_row = max(0, min(rows - 1, row + offset));
    const float weight = expf(-(static_cast<float>(offset * offset)) / (2.0f * sigma * sigma));
    sum += input[flat_index(rows, cols, src_row, col)] * weight;
    weight_sum += weight;
  }
  output[idx] = weight_sum > 0.0f ? sum / weight_sum : input[idx];
}

__global__ void coherent_power_gradient_kernel(const float* input,
                                               int rows,
                                               int cols,
                                               float* grad_f,
                                               float* grad_t) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  const int col = idx % cols;
  const int prev_row = max(0, row - 1);
  const int next_row = min(rows - 1, row + 1);
  const int prev_col = max(0, col - 1);
  const int next_col = min(cols - 1, col + 1);
  grad_f[idx] = 0.5f * (input[flat_index(rows, cols, next_row, col)] - input[flat_index(rows, cols, prev_row, col)]);
  grad_t[idx] = 0.5f * (input[flat_index(rows, cols, row, next_col)] - input[flat_index(rows, cols, row, prev_col)]);
}

__global__ void coherent_power_tensor_products_kernel(const float* grad_f,
                                                      const float* grad_t,
                                                      int total,
                                                      float* j_ff,
                                                      float* j_ft,
                                                      float* j_tt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const float grad_f_value = grad_f[idx];
  const float grad_t_value = grad_t[idx];
  j_ff[idx] = grad_f_value * grad_f_value;
  j_ft[idx] = grad_f_value * grad_t_value;
  j_tt[idx] = grad_t_value * grad_t_value;
}

__global__ void coherent_power_eigen_metrics_kernel(const float* j_ff,
                                                    const float* j_ft,
                                                    const float* j_tt,
                                                    int total,
                                                    float* coherence,
                                                    float* energy) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const float jff = j_ff[idx];
  const float jft = j_ft[idx];
  const float jtt = j_tt[idx];
  const float delta = sqrtf(fmaxf((jff - jtt) * (jff - jtt) + 4.0f * (jft * jft), 0.0f));
  const float lam1 = 0.5f * (jff + jtt + delta);
  const float lam2 = 0.5f * (jff + jtt - delta);
  const float denom = fmaxf(lam1 + lam2, 1e-6f);
  coherence[idx] = (lam1 - lam2) / denom;
  energy[idx] = lam1 + lam2;
}

__global__ void coherent_power_db_to_linear_kernel(const float* input_db,
                                                   int total,
                                                   float* output_linear) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  output_linear[idx] = powf(10.0f, input_db[idx] / 10.0f);
}

__global__ void coherent_power_relative_db_kernel(const float* input_linear,
                                                  float floor_linear,
                                                  int total,
                                                  float* output_rel_db) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  output_rel_db[idx] = fminf(fmaxf(10.0f * log10f(fmaxf(input_linear[idx], 1e-20f) / floor_linear), -5.0f), 25.0f);
}

__global__ void coherent_power_normalize_clamp_kernel(const float* input,
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

__global__ void coherent_power_weighted_sum_kernel(const float* lhs,
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

__global__ void coherent_power_gate_kernel(const float* coherence,
                                           int total,
                                           float gate_start,
                                           float gate_inv_span,
                                           float* gate) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  gate[idx] = fminf(fmaxf((coherence[idx] - gate_start) * gate_inv_span, 0.0f), 1.0f);
}

__global__ void coherent_power_bridged_joint_score_kernel(const float* power,
                                                          const float* coherence_gate,
                                                          int total,
                                                          float bridge_bias,
                                                          float joint_weight,
                                                          float* combined) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float power_value = fmaxf(power[idx], 0.0f);
  const float gate_value = fmaxf(coherence_gate[idx], 0.0f);
  const float bridged_power = power_value * (bridge_bias + (1.0f - bridge_bias) * gate_value);
  const float joint_power = sqrtf(power_value * gate_value);
  combined[idx] = joint_weight * joint_power + (1.0f - joint_weight) * bridged_power;
}

__global__ void coherent_power_absolute_assist_kernel(const float* corrected_db,
                                                      int rows,
                                                      int cols,
                                                      const uint8_t* valid_row_mask,
                                                      float floor_db,
                                                      float start_db,
                                                      float inv_span_db,
                                                      float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }

  const int row = idx / cols;
  if (!valid_row_mask[row]) {
    output[idx] = 0.0f;
    return;
  }

  const float excess_db = corrected_db[idx] - floor_db;
  output[idx] = fminf(fmaxf((excess_db - start_db) * inv_span_db, 0.0f), 1.0f);
}

__global__ void coherent_power_blend_kernel(const float* absolute_power,
                                            const float* local_power,
                                            int total,
                                            float local_blend,
                                            float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  output[idx] = fminf(fmaxf((1.0f - local_blend) * absolute_power[idx] + local_blend * local_power[idx], 0.0f), 1.0f);
}

__global__ void coherent_power_apply_valid_row_mask_kernel(float* values,
                                                           int rows,
                                                           int cols,
                                                           const uint8_t* valid_row_mask) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  if (!valid_row_mask[row]) {
    values[idx] = 0.0f;
  }
}

__global__ void coherent_power_copy_masked_with_sentinel_kernel(const float* input,
                                                                const uint8_t* mask,
                                                                int total,
                                                                float sentinel,
                                                                float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = mask[idx] ? input[idx] : sentinel;
}

__global__ void coherent_power_copy_valid_rows_with_sentinel_kernel(const float* input,
                                                                    int rows,
                                                                    int cols,
                                                                    const uint8_t* valid_row_mask,
                                                                    float sentinel,
                                                                    float* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  output[idx] = valid_row_mask[row] ? input[idx] : sentinel;
}

__global__ void coherent_power_threshold_mask_kernel(const float* input,
                                                     const uint8_t* gate_mask,
                                                     int total,
                                                     float threshold,
                                                     uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  output[idx] = (gate_mask[idx] && input[idx] >= threshold) ? 1 : 0;
}

__global__ void coherent_power_threshold_valid_rows_kernel(const float* input,
                                                           int rows,
                                                           int cols,
                                                           const uint8_t* valid_row_mask,
                                                           float threshold,
                                                           uint8_t* output) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  output[idx] = (valid_row_mask[row] && input[idx] >= threshold) ? 1 : 0;
}

float clamp_float(float value, float low, float high) {
  return std::max(low, std::min(high, value));
}

__global__ void coherent_power_histogram_valid_rows_kernel(const float* input,
                                                           int rows,
                                                           int cols,
                                                           const uint8_t* valid_row_mask,
                                                           float range_min,
                                                           float range_max,
                                                           unsigned int* histogram,
                                                           int histogram_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * cols;
  if (idx >= total) {
    return;
  }
  const int row = idx / cols;
  if (!valid_row_mask[row]) {
    return;
  }
  const float denom = fmaxf(range_max - range_min, 1e-6f);
  const float clamped = fminf(fmaxf((input[idx] - range_min) / denom, 0.0f), 1.0f);
  const int bin = min(histogram_bins - 1, max(0, static_cast<int>(clamped * static_cast<float>(histogram_bins - 1))));
  atomicAdd(histogram + bin, 1U);
}

__global__ void coherent_power_histogram_masked_kernel(const float* input,
                                                       const uint8_t* mask,
                                                       int total,
                                                       float range_min,
                                                       float range_max,
                                                       unsigned int* histogram,
                                                       int histogram_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total || !mask[idx]) {
    return;
  }
  const float denom = fmaxf(range_max - range_min, 1e-6f);
  const float clamped = fminf(fmaxf((input[idx] - range_min) / denom, 0.0f), 1.0f);
  const int bin = min(histogram_bins - 1, max(0, static_cast<int>(clamped * static_cast<float>(histogram_bins - 1))));
  atomicAdd(histogram + bin, 1U);
}

__global__ void coherent_power_histogram_all_kernel(const float* input,
                                                    int total,
                                                    float range_min,
                                                    float range_max,
                                                    unsigned int* histogram,
                                                    int histogram_bins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const float denom = fmaxf(range_max - range_min, 1e-6f);
  const float clamped = fminf(fmaxf((input[idx] - range_min) / denom, 0.0f), 1.0f);
  const int bin = min(histogram_bins - 1, max(0, static_cast<int>(clamped * static_cast<float>(histogram_bins - 1))));
  atomicAdd(histogram + bin, 1U);
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

std::vector<float> collect_masked_values(const std::vector<float>& values,
                                         const std::vector<uint8_t>& mask) {
  std::vector<float> collected;
  collected.reserve(values.size());
  for (size_t index = 0; index < values.size() && index < mask.size(); ++index) {
    if (mask[index]) {
      const float value = values[index];
      if (std::isfinite(value)) {
        collected.push_back(value);
      }
    }
  }
  return collected;
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

std::vector<float> gaussian_kernel_1d(float sigma) {
  sigma = std::max(0.5f, sigma);
  const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
  std::vector<float> kernel(static_cast<size_t>(2 * radius + 1), 0.0f);
  float weight_sum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const float value = std::exp(-(static_cast<float>(offset * offset)) / (2.0f * sigma * sigma));
    kernel[static_cast<size_t>(offset + radius)] = value;
    weight_sum += value;
  }
  if (weight_sum <= 0.0f) {
    return std::vector<float>{1.0f};
  }
  for (float& value : kernel) {
    value /= weight_sum;
  }
  return kernel;
}

std::vector<float> convolve_rows(const std::vector<float>& input,
                                 int rows,
                                 int cols,
                                 const std::vector<float>& kernel) {
  const int radius = static_cast<int>(kernel.size() / 2);
  std::vector<float> output(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      float sum = 0.0f;
      for (int offset = -radius; offset <= radius; ++offset) {
        const int source_col = clamp_int(col + offset, 0, cols - 1);
        sum += input[flat_index(rows, cols, row, source_col)] * kernel[static_cast<size_t>(offset + radius)];
      }
      output[flat_index(rows, cols, row, col)] = sum;
    }
  }
  return output;
}

std::vector<float> convolve_cols(const std::vector<float>& input,
                                 int rows,
                                 int cols,
                                 const std::vector<float>& kernel) {
  const int radius = static_cast<int>(kernel.size() / 2);
  std::vector<float> output(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      float sum = 0.0f;
      for (int offset = -radius; offset <= radius; ++offset) {
        const int source_row = clamp_int(row + offset, 0, rows - 1);
        sum += input[flat_index(rows, cols, source_row, col)] * kernel[static_cast<size_t>(offset + radius)];
      }
      output[flat_index(rows, cols, row, col)] = sum;
    }
  }
  return output;
}

std::vector<float> gaussian_blur_2d(const std::vector<float>& input,
                                    int rows,
                                    int cols,
                                    float sigma) {
  const auto kernel = gaussian_kernel_1d(sigma);
  return convolve_cols(convolve_rows(input, rows, cols, kernel), rows, cols, kernel);
}

bool gaussian_blur_2d_cuda(const float* input_device,
                           int rows,
                           int cols,
                           float sigma,
                           float* scratch_device,
                           float* output_device,
                           cudaStream_t stream) {
  sigma = std::max(0.5f, sigma);
  const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
  const int total = rows * cols;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  coherent_power_gaussian_rows_kernel<<<blocks, threads, 0, stream>>>(input_device,
                                                                       rows,
                                                                       cols,
                                                                       radius,
                                                                       sigma,
                                                                       scratch_device);
  auto cuda_result = cudaGetLastError();
  if (cuda_result != cudaSuccess) {
    return false;
  }
  coherent_power_gaussian_cols_kernel<<<blocks, threads, 0, stream>>>(scratch_device,
                                                                       rows,
                                                                       cols,
                                                                       radius,
                                                                       sigma,
                                                                       output_device);
  cuda_result = cudaGetLastError();
  return cuda_result == cudaSuccess;
}

std::vector<float> gaussian_blur_1d(const std::vector<float>& input, float sigma) {
  const auto kernel = gaussian_kernel_1d(sigma);
  const int radius = static_cast<int>(kernel.size() / 2);
  std::vector<float> output(input.size(), 0.0f);
  for (int index = 0; index < static_cast<int>(input.size()); ++index) {
    float sum = 0.0f;
    for (int offset = -radius; offset <= radius; ++offset) {
      const int source_index = clamp_int(index + offset, 0, static_cast<int>(input.size()) - 1);
      sum += input[static_cast<size_t>(source_index)] * kernel[static_cast<size_t>(offset + radius)];
    }
    output[static_cast<size_t>(index)] = sum;
  }
  return output;
}

std::vector<float> box_filter_2d(const std::vector<float>& input,
                                 int rows,
                                 int cols,
                                 int kernel_rows,
                                 int kernel_cols) {
  kernel_rows = std::max(1, kernel_rows | 1);
  kernel_cols = std::max(1, kernel_cols | 1);
  const int row_radius = kernel_rows / 2;
  const int col_radius = kernel_cols / 2;
  std::vector<float> output(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const int row_start = std::max(0, row - row_radius);
    const int row_stop = std::min(rows - 1, row + row_radius);
    for (int col = 0; col < cols; ++col) {
      const int col_start = std::max(0, col - col_radius);
      const int col_stop = std::min(cols - 1, col + col_radius);
      float sum = 0.0f;
      int count = 0;
      for (int src_row = row_start; src_row <= row_stop; ++src_row) {
        for (int src_col = col_start; src_col <= col_stop; ++src_col) {
          sum += input[flat_index(rows, cols, src_row, src_col)];
          ++count;
        }
      }
      output[flat_index(rows, cols, row, col)] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }
  }
  return output;
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

std::vector<float> normalize_map01_masked(const std::vector<float>& input,
                                          const std::vector<uint8_t>& mask,
                                          float low_q,
                                          float high_q) {
  std::vector<float> masked_values = collect_masked_values(input, mask);
  if (masked_values.empty()) {
    return std::vector<float>(input.size(), 0.0f);
  }
  float low = percentile_from_values(masked_values, low_q);
  float high = percentile_from_values(std::move(masked_values), high_q);
  if (high <= low) {
    high = low + 1e-6f;
  }
  std::vector<float> output(input.size(), 0.0f);
  const float denom = high - low;
  for (size_t index = 0; index < input.size() && index < mask.size(); ++index) {
    if (mask[index]) {
      output[index] = clamp_float((input[index] - low) / denom, 0.0f, 1.0f);
    }
  }
  return output;
}

std::vector<float> normalize_map01_indices(const std::vector<float>& input,
                                           const std::vector<size_t>& indices,
                                           float low_q,
                                           float high_q) {
  std::vector<float> selected_values;
  selected_values.reserve(indices.size());
  for (const size_t index : indices) {
    if (index >= input.size()) {
      continue;
    }
    const float value = input[index];
    if (std::isfinite(value)) {
      selected_values.push_back(value);
    }
  }
  if (selected_values.empty()) {
    return std::vector<float>(input.size(), 0.0f);
  }
  float low = percentile_from_values(selected_values, low_q);
  float high = percentile_from_values(std::move(selected_values), high_q);
  if (high <= low) {
    high = low + 1e-6f;
  }
  std::vector<float> output(input.size(), 0.0f);
  const float denom = high - low;
  for (const size_t index : indices) {
    if (index >= input.size()) {
      continue;
    }
    output[index] = clamp_float((input[index] - low) / denom, 0.0f, 1.0f);
  }
  return output;
}

void normalize_map01_indices_into(const std::vector<float>& input,
                                  const std::vector<size_t>& indices,
                                  float low_q,
                                  float high_q,
                                  std::vector<float>& output) {
  std::vector<float> selected_values;
  selected_values.reserve(indices.size());
  for (const size_t index : indices) {
    if (index >= input.size()) {
      continue;
    }
    const float value = input[index];
    if (std::isfinite(value)) {
      selected_values.push_back(value);
    }
  }
  output.resize(input.size(), 0.0f);
  if (selected_values.empty()) {
    return;
  }
  float low = percentile_from_values(selected_values, low_q);
  float high = percentile_from_values(std::move(selected_values), high_q);
  if (high <= low) {
    high = low + 1e-6f;
  }
  const float denom = high - low;
  for (const size_t index : indices) {
    if (index >= input.size()) {
      continue;
    }
    output[index] = clamp_float((input[index] - low) / denom, 0.0f, 1.0f);
  }
}

float robust_high_quantile_threshold(const std::vector<float>& input, float q, float saturation = 0.9995f) {
  std::vector<float> finite_values;
  finite_values.reserve(input.size());
  for (const float value : input) {
    if (std::isfinite(value)) {
      finite_values.push_back(value);
    }
  }
  if (finite_values.empty()) {
    return 1.0f;
  }
  q = clamp_float(q, 0.50f, 0.99f);
  const float threshold = quantile_from_values(finite_values, q);
  if (threshold < saturation) {
    return threshold;
  }
  std::vector<float> unsaturated;
  unsaturated.reserve(finite_values.size());
  for (const float value : finite_values) {
    if (value < saturation) {
      unsaturated.push_back(value);
    }
  }
  if (unsaturated.empty()) {
    return saturation;
  }
  return quantile_from_values(std::move(unsaturated), std::min(q, 0.90f));
}

std::vector<uint8_t> prune_small_components(std::vector<uint8_t> output,
                                            int rows,
                                            int cols,
                                            int min_component_size) {
  std::vector<uint8_t> visited(output.size(), 0);
  const std::array<std::pair<int, int>, 4> neighbors{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t seed = flat_index(rows, cols, row, col);
      if (!output[seed] || visited[seed]) {
        continue;
      }
      std::queue<std::pair<int, int>> queue;
      std::vector<size_t> component;
      queue.push({row, col});
      visited[seed] = 1;
      while (!queue.empty()) {
        const auto [current_row, current_col] = queue.front();
        queue.pop();
        component.push_back(flat_index(rows, cols, current_row, current_col));
        for (const auto& [delta_row, delta_col] : neighbors) {
          const int next_row = current_row + delta_row;
          const int next_col = current_col + delta_col;
          if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
            continue;
          }
          const size_t next_index = flat_index(rows, cols, next_row, next_col);
          if (!output[next_index] || visited[next_index]) {
            continue;
          }
          visited[next_index] = 1;
          queue.push({next_row, next_col});
        }
      }
      if (static_cast<int>(component.size()) >= min_component_size) {
        continue;
      }
      for (const size_t index : component) {
        output[index] = 0;
      }
    }
  }
  return output;
}

std::vector<uint8_t> valid_row_mask_to_full_mask(const std::vector<uint8_t>& valid_row_mask,
                                                 int cols) {
  std::vector<uint8_t> mask(static_cast<size_t>(valid_row_mask.size()) * static_cast<size_t>(cols), 0);
  for (int row = 0; row < static_cast<int>(valid_row_mask.size()); ++row) {
    if (!valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      mask[flat_index(static_cast<int>(valid_row_mask.size()), cols, row, col)] = 1;
    }
  }
  return mask;
}

std::vector<uint8_t> smooth_binary_label_map(const std::vector<uint8_t>& input,
                                             int rows,
                                             int cols,
                                             int iters,
                                             int min_component_size) {
  std::vector<uint8_t> output = input;
  for (int iter = 0; iter < std::max(0, iters); ++iter) {
    std::vector<uint8_t> next(output.size(), 0);
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        int sum = 0;
        int count = 0;
        for (int d_row = -1; d_row <= 1; ++d_row) {
          for (int d_col = -1; d_col <= 1; ++d_col) {
            const int src_row = clamp_int(row + d_row, 0, rows - 1);
            const int src_col = clamp_int(col + d_col, 0, cols - 1);
            sum += output[flat_index(rows, cols, src_row, src_col)] ? 1 : 0;
            ++count;
          }
        }
        next[flat_index(rows, cols, row, col)] = (sum * 2 >= count) ? 1 : 0;
      }
    }
    output.swap(next);
  }

  return prune_small_components(std::move(output), rows, cols, min_component_size);
}

std::vector<uint8_t> smooth_binary_label_map_windowed(const std::vector<uint8_t>& input,
                                                      int rows,
                                                      int cols,
                                                      int row_start,
                                                      int row_stop,
                                                      int col_start,
                                                      int col_stop,
                                                      int iters,
                                                      int min_component_size) {
  row_start = clamp_int(row_start, 0, rows);
  row_stop = clamp_int(row_stop, row_start, rows);
  col_start = clamp_int(col_start, 0, cols);
  col_stop = clamp_int(col_stop, col_start, cols);
  if (row_start == row_stop || col_start == col_stop) {
    return std::vector<uint8_t>(input.size(), 0);
  }

  const int window_rows = row_stop - row_start;
  const int window_cols = col_stop - col_start;
  std::vector<uint8_t> window(static_cast<size_t>(window_rows) * static_cast<size_t>(window_cols), 0);
  for (int row = 0; row < window_rows; ++row) {
    for (int col = 0; col < window_cols; ++col) {
      window[flat_index(window_rows, window_cols, row, col)] =
          input[flat_index(rows, cols, row_start + row, col_start + col)];
    }
  }

  auto smoothed_window = smooth_binary_label_map(window,
                                                 window_rows,
                                                 window_cols,
                                                 iters,
                                                 min_component_size);
  std::vector<uint8_t> output(input.size(), 0);
  for (int row = 0; row < window_rows; ++row) {
    for (int col = 0; col < window_cols; ++col) {
      output[flat_index(rows, cols, row_start + row, col_start + col)] =
          smoothed_window[flat_index(window_rows, window_cols, row, col)];
    }
  }
  return output;
}

std::vector<float> local_relative_power_support_map(const std::vector<float>& sxx_db_local,
                                                    int rows,
                                                    int cols,
                                                    const std::vector<uint8_t>& valid_row_mask,
                                                    float floor_q,
                                                    int freq_window,
                                                    int time_window) {
  std::vector<float> p_lin(sxx_db_local.size(), 0.0f);
  for (size_t index = 0; index < sxx_db_local.size(); ++index) {
    p_lin[index] = std::pow(10.0f, sxx_db_local[index] / 10.0f);
  }
  std::vector<float> valid_values;
  valid_values.reserve(p_lin.size());
  for (int row = 0; row < rows; ++row) {
    if (!valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      valid_values.push_back(p_lin[flat_index(rows, cols, row, col)]);
    }
  }
  if (valid_values.empty()) {
    valid_values = p_lin;
  }
  const float p_floor = std::max(percentile_from_values(std::move(valid_values), floor_q), 1e-20f);
  std::vector<float> rel_db(sxx_db_local.size(), 0.0f);
  for (size_t index = 0; index < p_lin.size(); ++index) {
    rel_db[index] = clamp_float(10.0f * std::log10(std::max(p_lin[index], 1e-20f) / p_floor), -5.0f, 25.0f);
  }
  auto local_baseline = box_filter_2d(rel_db,
                                      rows,
                                      cols,
                                      std::max(3, freq_window | 1),
                                      std::max(5, time_window | 1));
  std::vector<float> support(rel_db.size(), 0.0f);
  for (size_t index = 0; index < rel_db.size(); ++index) {
    support[index] = std::max(rel_db[index] - local_baseline[index], 0.0f);
  }
  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      support[flat_index(rows, cols, row, col)] = 0.0f;
    }
  }
  return support;
}

void compute_gradients(const std::vector<float>& input,
                       int rows,
                       int cols,
                       std::vector<float>& grad_f,
                       std::vector<float>& grad_t) {
  grad_f.assign(input.size(), 0.0f);
  grad_t.assign(input.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const int prev_row = clamp_int(row - 1, 0, rows - 1);
      const int next_row = clamp_int(row + 1, 0, rows - 1);
      const int prev_col = clamp_int(col - 1, 0, cols - 1);
      const int next_col = clamp_int(col + 1, 0, cols - 1);
      grad_f[flat_index(rows, cols, row, col)] =
          0.5f * (input[flat_index(rows, cols, next_row, col)] - input[flat_index(rows, cols, prev_row, col)]);
      grad_t[flat_index(rows, cols, row, col)] =
          0.5f * (input[flat_index(rows, cols, row, next_col)] - input[flat_index(rows, cols, row, prev_col)]);
    }
  }
}

std::tuple<std::vector<float>, std::vector<float>> structure_tensor_components(const std::vector<float>& input,
                                                                               int rows,
                                                                               int cols,
                                                                               float grad_sigma,
                                                                               float integ_sigma) {
  const auto grad_blur = gaussian_blur_2d(input, rows, cols, grad_sigma);
  std::vector<float> grad_f;
  std::vector<float> grad_t;
  compute_gradients(grad_blur, rows, cols, grad_f, grad_t);
  std::vector<float> j_ff(input.size(), 0.0f);
  std::vector<float> j_ft(input.size(), 0.0f);
  std::vector<float> j_tt(input.size(), 0.0f);
  for (size_t index = 0; index < input.size(); ++index) {
    j_ff[index] = grad_f[index] * grad_f[index];
    j_ft[index] = grad_f[index] * grad_t[index];
    j_tt[index] = grad_t[index] * grad_t[index];
  }
  j_ff = gaussian_blur_2d(j_ff, rows, cols, integ_sigma);
  j_ft = gaussian_blur_2d(j_ft, rows, cols, integ_sigma);
  j_tt = gaussian_blur_2d(j_tt, rows, cols, integ_sigma);

  std::vector<float> coherence(input.size(), 0.0f);
  std::vector<float> energy(input.size(), 0.0f);
  for (size_t index = 0; index < input.size(); ++index) {
    const float delta = std::sqrt(std::max((j_ff[index] - j_tt[index]) * (j_ff[index] - j_tt[index]) +
                                           4.0f * (j_ft[index] * j_ft[index]),
                                           0.0f));
    const float lam1 = 0.5f * (j_ff[index] + j_tt[index] + delta);
    const float lam2 = 0.5f * (j_ff[index] + j_tt[index] - delta);
    coherence[index] = (lam1 - lam2) / std::max(lam1 + lam2, 1e-6f);
    energy[index] = lam1 + lam2;
  }
  return {coherence, energy};
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
multi_scale_structure_tensor_gate(const std::vector<float>& sxx_db_local, int rows, int cols) {
  const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
  const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
  const auto background = box_filter_2d(sxx_db_local, rows, cols, bg_freq, bg_time);
  std::vector<float> residual_db(sxx_db_local.size(), 0.0f);
  for (size_t index = 0; index < sxx_db_local.size(); ++index) {
    residual_db[index] = std::max(sxx_db_local[index] - background[index], 0.0f);
  }
  const auto residual_n = normalize_map01_local(residual_db, 5.0f, 99.0f);

  const std::array<float, 3> scales{{0.8f, 1.6f, 3.2f}};
  std::vector<float> coherence_max(sxx_db_local.size(), 0.0f);
  std::vector<float> energy_max(sxx_db_local.size(), 0.0f);
  for (const float grad_sigma : scales) {
    auto [coherence, energy] = structure_tensor_components(residual_n,
                                                           rows,
                                                           cols,
                                                           grad_sigma,
                                                           std::max(1.0f, 1.8f * grad_sigma));
    coherence = normalize_map01_local(coherence, 5.0f, 99.0f);
    energy = normalize_map01_local(energy, 5.0f, 99.0f);
    for (size_t index = 0; index < coherence.size(); ++index) {
      coherence_max[index] = std::max(coherence_max[index], coherence[index]);
      energy_max[index] = std::max(energy_max[index], energy[index]);
    }
  }
  std::vector<float> gate(coherence_max.size(), 0.0f);
  for (size_t index = 0; index < gate.size(); ++index) {
    gate[index] = coherence_max[index] * std::sqrt(std::max(energy_max[index], 0.0f));
  }
  gate = normalize_map01_local(gate, 5.0f, 99.0f);
  return {coherence_max, energy_max, gate};
}

struct ChunkCudaScratch {
  cudaStream_t stream = nullptr;
  std::array<float*, 8> buffers {};
  std::array<uint8_t*, 2> mask_buffers {};
  std::array<unsigned int*, 2> histogram_buffers {};
  size_t capacity = 0;

  void release_buffers() {
    for (float*& pointer : buffers) {
      if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
      }
    }
    for (uint8_t*& pointer : mask_buffers) {
      if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
      }
    }
    for (unsigned int*& pointer : histogram_buffers) {
      if (pointer != nullptr) {
        cudaFree(pointer);
        pointer = nullptr;
      }
    }
    capacity = 0;
  }

  bool ensure_capacity(size_t requested_elements) {
    if (stream == nullptr && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
      return false;
    }
    if (capacity >= requested_elements) {
      return true;
    }

    release_buffers();
    for (float*& pointer : buffers) {
      if (cudaMalloc(reinterpret_cast<void**>(&pointer), requested_elements * sizeof(float)) != cudaSuccess) {
        release_buffers();
        return false;
      }
    }
    for (uint8_t*& pointer : mask_buffers) {
      if (cudaMalloc(reinterpret_cast<void**>(&pointer), requested_elements * sizeof(uint8_t)) != cudaSuccess) {
        release_buffers();
        return false;
      }
    }
    for (unsigned int*& pointer : histogram_buffers) {
      if (cudaMalloc(reinterpret_cast<void**>(&pointer), kQuantileHistogramBins * sizeof(unsigned int)) != cudaSuccess) {
        release_buffers();
        return false;
      }
    }
    capacity = requested_elements;
    return true;
  }
};

ChunkCudaScratch& chunk_cuda_scratch() {
  thread_local auto* scratch = new ChunkCudaScratch();
  return *scratch;
}

float quantile_from_histogram_host(const std::vector<unsigned int>& histogram,
                                   float q,
                                   float range_min,
                                   float range_max) {
  uint64_t total_count = 0;
  for (const unsigned int count : histogram) {
    total_count += count;
  }
  if (total_count == 0) {
    return range_min;
  }
  q = clamp_float(q, 0.0f, 1.0f);
  const uint64_t target = static_cast<uint64_t>(std::llround(q * static_cast<float>(total_count - 1)));
  uint64_t prefix = 0;
  size_t bin_index = histogram.size() - 1;
  for (size_t index = 0; index < histogram.size(); ++index) {
    prefix += histogram[index];
    if (prefix > target) {
      bin_index = index;
      break;
    }
  }
  const float bin_width = (range_max - range_min) / static_cast<float>(std::max<size_t>(1, histogram.size() - 1));
  return range_min + static_cast<float>(bin_index) * bin_width;
}

float device_quantile_from_histogram_all(float* values_device,
                                         int total,
                                         float q,
                                         float range_min,
                                         float range_max,
                                         unsigned int* histogram_device,
                                         cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  if (cudaMemsetAsync(histogram_device, 0, kQuantileHistogramBins * sizeof(unsigned int), stream) != cudaSuccess) {
    throw std::runtime_error("histogram memset failed");
  }
  coherent_power_histogram_all_kernel<<<blocks, threads, 0, stream>>>(values_device,
                                                                       total,
                                                                       range_min,
                                                                       range_max,
                                                                       histogram_device,
                                                                       kQuantileHistogramBins);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("histogram all kernel failed");
  }
  std::vector<unsigned int> histogram(static_cast<size_t>(kQuantileHistogramBins), 0);
  if (cudaMemcpyAsync(histogram.data(),
                      histogram_device,
                      histogram.size() * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("histogram readback failed");
  }
  return quantile_from_histogram_host(histogram, q, range_min, range_max);
}

float device_quantile_from_histogram_valid_rows(float* values_device,
                                                int rows,
                                                int cols,
                                                const uint8_t* valid_row_mask_device,
                                                float q,
                                                float range_min,
                                                float range_max,
                                                unsigned int* histogram_device,
                                                cudaStream_t stream) {
  const int total = rows * cols;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  if (cudaMemsetAsync(histogram_device, 0, kQuantileHistogramBins * sizeof(unsigned int), stream) != cudaSuccess) {
    throw std::runtime_error("valid-row histogram memset failed");
  }
  coherent_power_histogram_valid_rows_kernel<<<blocks, threads, 0, stream>>>(values_device,
                                                                              rows,
                                                                              cols,
                                                                              valid_row_mask_device,
                                                                              range_min,
                                                                              range_max,
                                                                              histogram_device,
                                                                              kQuantileHistogramBins);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("valid-row histogram kernel failed");
  }
  std::vector<unsigned int> histogram(static_cast<size_t>(kQuantileHistogramBins), 0);
  if (cudaMemcpyAsync(histogram.data(),
                      histogram_device,
                      histogram.size() * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("valid-row histogram readback failed");
  }
  return quantile_from_histogram_host(histogram, q, range_min, range_max);
}

float device_quantile_from_histogram_masked(float* values_device,
                                            const uint8_t* mask_device,
                                            int total,
                                            float q,
                                            float range_min,
                                            float range_max,
                                            unsigned int* histogram_device,
                                            cudaStream_t stream) {
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  if (cudaMemsetAsync(histogram_device, 0, kQuantileHistogramBins * sizeof(unsigned int), stream) != cudaSuccess) {
    throw std::runtime_error("masked histogram memset failed");
  }
  coherent_power_histogram_masked_kernel<<<blocks, threads, 0, stream>>>(values_device,
                                                                          mask_device,
                                                                          total,
                                                                          range_min,
                                                                          range_max,
                                                                          histogram_device,
                                                                          kQuantileHistogramBins);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("masked histogram kernel failed");
  }
  std::vector<unsigned int> histogram(static_cast<size_t>(kQuantileHistogramBins), 0);
  if (cudaMemcpyAsync(histogram.data(),
                      histogram_device,
                      histogram.size() * sizeof(unsigned int),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("masked histogram readback failed");
  }
  return quantile_from_histogram_host(histogram, q, range_min, range_max);
}

void normalize_map01_device_inplace(float* values_device,
                                    int total,
                                    float low_q,
                                    float high_q,
                                    float range_min,
                                    float range_max,
                                    unsigned int* histogram_device,
                                    cudaStream_t stream) {
  float low = device_quantile_from_histogram_all(values_device, total, low_q / 100.0f, range_min, range_max, histogram_device, stream);
  float high = device_quantile_from_histogram_all(values_device, total, high_q / 100.0f, range_min, range_max, histogram_device, stream);
  if (high <= low) {
    high = low + 1e-6f;
  }
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  coherent_power_normalize_clamp_kernel<<<blocks, threads, 0, stream>>>(values_device,
                                                                         total,
                                                                         low,
                                                                         1.0f / (high - low),
                                                                         values_device);
  const auto cuda_result = cudaGetLastError();
  if (cuda_result != cudaSuccess) {
    throw std::runtime_error(std::string("device normalize kernel failed: ") + cudaGetErrorString(cuda_result));
  }
}

bool local_relative_power_support_map_cuda(const float* sxx_db_local,
                                           int rows,
                                           int cols,
                                           const std::vector<uint8_t>& valid_row_mask,
                                           float floor_q,
                                           int freq_window,
                                           int time_window,
                                           std::vector<float>& support);

float estimate_noise_floor_db(const float* corrected_chunk,
                              int rows,
                              int cols,
                              const std::vector<uint8_t>& valid_row_mask,
                              float time_q,
                              float global_q) {
  std::vector<float> row_floor_db(static_cast<size_t>(rows), 0.0f);
  std::vector<float> row_values(static_cast<size_t>(cols), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const float* row_ptr = corrected_chunk + static_cast<size_t>(row) * static_cast<size_t>(cols);
    std::copy(row_ptr, row_ptr + cols, row_values.begin());
    row_floor_db[static_cast<size_t>(row)] = percentile_from_values(row_values, time_q);
  }

  std::vector<float> valid_row_floor_db;
  valid_row_floor_db.reserve(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      valid_row_floor_db.push_back(row_floor_db[static_cast<size_t>(row)]);
    }
  }
  if (valid_row_floor_db.empty()) {
    valid_row_floor_db = row_floor_db;
  }

  return percentile_from_values(std::move(valid_row_floor_db), global_q);
}

bool build_power_assist_map_device(const float* corrected_chunk,
                                   int rows,
                                   int cols,
                                   const std::vector<uint8_t>& valid_row_mask,
                                   std::string power_assist_mode,
                                   float power_floor_time_q,
                                   float power_floor_global_q,
                                   float power_excess_start_db,
                                   float power_excess_full_db,
                                   float power_local_blend,
                                   std::array<double, holoscan::ops::CoherentPowerSignalDetector::kPowerSupportTimingStageCount>* detail_ms_out,
                                   const float** power_px_device_out) {
  const int total = rows * cols;
  if (total <= 0 || corrected_chunk == nullptr || power_px_device_out == nullptr) {
    return false;
  }

  if (detail_ms_out != nullptr) {
    *detail_ms_out = {};
  }

  auto& scratch = chunk_cuda_scratch();
  if (!scratch.ensure_capacity(static_cast<size_t>(total))) {
    return false;
  }

  float* corrected_db_device = scratch.buffers[0];
  float* linear_power_device = scratch.buffers[1];
  float* relative_db_device = scratch.buffers[2];
  float* local_baseline_device = scratch.buffers[3];
  float* local_support_device = scratch.buffers[4];
  float* local_power_device = scratch.buffers[5];
  float* absolute_power_device = scratch.buffers[6];
  float* blended_power_device = scratch.buffers[7];
  uint8_t* valid_row_mask_device = scratch.mask_buffers[0];
  unsigned int* histogram_device = scratch.histogram_buffers[0];
  cudaStream_t stream = scratch.stream;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;
  auto time_stage_ms = [](auto&& fn) {
    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
  };

  if (cudaMemcpyAsync(corrected_db_device,
                      corrected_chunk,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(valid_row_mask_device,
                      valid_row_mask.data(),
                      static_cast<size_t>(rows) * sizeof(uint8_t),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }

  coherent_power_db_to_linear_kernel<<<blocks, threads, 0, stream>>>(corrected_db_device, total, linear_power_device);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  float valid_min_db = std::numeric_limits<float>::max();
  float valid_max_db = std::numeric_limits<float>::lowest();
  for (int row = 0; row < rows; ++row) {
    if (!valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    const float* row_ptr = corrected_chunk + static_cast<size_t>(row) * static_cast<size_t>(cols);
    for (int col = 0; col < cols; ++col) {
      valid_min_db = std::min(valid_min_db, row_ptr[col]);
      valid_max_db = std::max(valid_max_db, row_ptr[col]);
    }
  }
  if (!(valid_min_db < valid_max_db)) {
    valid_min_db = -120.0f;
    valid_max_db = 120.0f;
  }

  float local_floor_linear = 1e-20f;
  const double floor_estimate_ms = time_stage_ms([&] {
    const float local_floor_db = device_quantile_from_histogram_valid_rows(corrected_db_device,
                                                                           rows,
                                                                           cols,
                                                                           valid_row_mask_device,
                                                                           0.30f,
                                                                           valid_min_db,
                                                                           valid_max_db,
                                                                           histogram_device,
                                                                           stream);
    local_floor_linear = std::max(std::pow(10.0f, local_floor_db / 10.0f), 1e-20f);
  });
  if (detail_ms_out != nullptr) {
    (*detail_ms_out)[kPowerSupportFloorEstimateStage] = floor_estimate_ms;
  }

  const double local_relative_ms = time_stage_ms([&] {
    coherent_power_relative_db_kernel<<<blocks, threads, 0, stream>>>(linear_power_device,
                                                                       local_floor_linear,
                                                                       total,
                                                                       relative_db_device);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("chunk local-relative db kernel failed");
    }
    coherent_power_box_mean_cols_kernel<<<blocks, threads, 0, stream>>>(relative_db_device,
                                                                         rows,
                                                                         cols,
                                                                         std::max(5, 33 | 1) / 2,
                                                                         local_baseline_device);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("chunk local-relative col-mean kernel failed");
    }
    coherent_power_box_mean_rows_kernel<<<blocks, threads, 0, stream>>>(local_baseline_device,
                                                                         rows,
                                                                         cols,
                                                                         std::max(3, 9 | 1) / 2,
                                                                         local_support_device);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("chunk local-relative row-mean kernel failed");
    }
    coherent_power_subtract_clamp_kernel<<<blocks, threads, 0, stream>>>(relative_db_device,
                                                                          local_support_device,
                                                                          total,
                                                                          local_power_device);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("chunk local-relative subtract kernel failed");
    }
    coherent_power_apply_valid_row_mask_kernel<<<blocks, threads, 0, stream>>>(local_power_device,
                                                                                 rows,
                                                                                 cols,
                                                                                 valid_row_mask_device);
    if (cudaGetLastError() != cudaSuccess) {
      throw std::runtime_error("chunk local-relative valid-mask kernel failed");
    }
    normalize_map01_device_inplace(local_power_device, total, 5.0f, 95.0f, 0.0f, 25.0f, histogram_device, stream);
  });
  if (detail_ms_out != nullptr) {
    (*detail_ms_out)[kPowerSupportLocalRelativeStage] = local_relative_ms;
  }

  std::transform(power_assist_mode.begin(),
                 power_assist_mode.end(),
                 power_assist_mode.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  if (power_assist_mode == "local_relative") {
    *power_px_device_out = local_power_device;
    return true;
  }

  const float floor_db = estimate_noise_floor_db(corrected_chunk,
                                                 rows,
                                                 cols,
                                                 valid_row_mask,
                                                 power_floor_time_q,
                                                 power_floor_global_q);
  const float start_db = power_excess_start_db;
  const float full_db = std::max(power_excess_full_db, start_db + 1e-3f);
  const double absolute_assist_ms = time_stage_ms([&] {
    coherent_power_absolute_assist_kernel<<<blocks, threads, 0, stream>>>(corrected_db_device,
                                                                           rows,
                                                                           cols,
                                                                           valid_row_mask_device,
                                                                           floor_db,
                                                                           start_db,
                                                                           1.0f / (full_db - start_db),
                                                                           absolute_power_device);
    if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("chunk absolute-assist kernel failed");
    }
  });
  if (detail_ms_out != nullptr) {
    (*detail_ms_out)[kPowerSupportAbsoluteAssistStage] = absolute_assist_ms;
  }

  if (power_assist_mode == "absolute_floor") {
    *power_px_device_out = absolute_power_device;
    return true;
  }

  const float local_blend = clamp_float(power_local_blend, 0.0f, 1.0f);
  const double blend_ms = time_stage_ms([&] {
    coherent_power_blend_kernel<<<blocks, threads, 0, stream>>>(absolute_power_device,
                                                                 local_power_device,
                                                                 total,
                                                                 local_blend,
                                                                 blended_power_device);
    if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
      throw std::runtime_error("chunk power-assist blend kernel failed");
    }
  });
  if (detail_ms_out != nullptr) {
    (*detail_ms_out)[kPowerSupportBlendStage] = blend_ms;
  }

  *power_px_device_out = blended_power_device;
  return true;
}

void run_chunk_score_masks_gpu(const std::vector<float>& coherence_px,
                               const float* power_px_device,
                               int rows,
                               int cols,
                               const std::vector<uint8_t>& valid_row_mask,
                               double support_q,
                               double final_q,
                               double coherence_gate_start,
                               double coherence_gate_full,
                               double coherence_bridge_bias,
                               double coherence_power_joint_weight,
                               int min_component_size,
                               std::vector<float>& combined_raw_px,
                               std::vector<float>& score_px,
                               std::vector<uint8_t>& support_px,
                               std::vector<uint8_t>& mask_px,
                               float& support_threshold,
                               float& score_threshold) {
  const int total = rows * cols;
  if (total <= 0 || static_cast<int>(coherence_px.size()) != total || power_px_device == nullptr) {
    throw std::invalid_argument("run_chunk_score_masks_gpu received mismatched chunk buffers");
  }

  auto& scratch = chunk_cuda_scratch();
  if (!scratch.ensure_capacity(static_cast<size_t>(total))) {
    throw std::runtime_error("chunk CUDA scratch allocation failed for score masks");
  }

  float* d0 = scratch.buffers[0];
  float* d2 = scratch.buffers[2];
  float* d3 = scratch.buffers[3];
  float* d4 = scratch.buffers[4];
  uint8_t* valid_row_mask_device = scratch.mask_buffers[0];
  uint8_t* mask_device = scratch.mask_buffers[1];
  unsigned int* histogram_device = scratch.histogram_buffers[0];
  cudaStream_t stream = scratch.stream;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  if (cudaMemcpyAsync(d0,
                      coherence_px.data(),
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(valid_row_mask_device,
                      valid_row_mask.data(),
                      static_cast<size_t>(rows) * sizeof(uint8_t),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("chunk score GPU input copy failed");
  }

  normalize_map01_device_inplace(d0, total, 5.0f, 99.0f, 0.0f, 1.0f, histogram_device, stream);
  const float gate_start = static_cast<float>(coherence_gate_start);
  const float gate_full = std::max(static_cast<float>(coherence_gate_full), gate_start + 1e-3f);
  coherent_power_gate_kernel<<<blocks, threads, 0, stream>>>(d0,
                                                              total,
                                                              gate_start,
                                                              1.0f / (gate_full - gate_start),
                                                              d4);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk coherence-gate kernel failed");
  }

  coherent_power_bridged_joint_score_kernel<<<blocks, threads, 0, stream>>>(power_px_device,
                                                                             d4,
                                                                             total,
                                                                             clamp_float(static_cast<float>(coherence_bridge_bias), 0.0f, 1.0f),
                                                                             clamp_float(static_cast<float>(coherence_power_joint_weight), 0.0f, 1.0f),
                                                                             d2);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk bridged/joint score kernel failed");
  }
  if (cudaMemcpyAsync(d3,
                      d2,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyDeviceToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("chunk score device copy failed");
  }
  normalize_map01_device_inplace(d3, total, 5.0f, 95.0f, 0.0f, 1.0f, histogram_device, stream);

  coherent_power_apply_valid_row_mask_kernel<<<blocks, threads, 0, stream>>>(d2, rows, cols, valid_row_mask_device);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk combined valid-mask kernel failed");
  }
  coherent_power_apply_valid_row_mask_kernel<<<blocks, threads, 0, stream>>>(d3, rows, cols, valid_row_mask_device);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk score valid-mask kernel failed");
  }

  coherent_power_copy_valid_rows_with_sentinel_kernel<<<blocks, threads, 0, stream>>>(d3,
                                                                                       rows,
                                                                                       cols,
                                                                                       valid_row_mask_device,
                                                                                       -1.0f,
                                                                                       d4);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk support-mask copy kernel failed");
  }
  support_threshold = device_quantile_from_histogram_valid_rows(d3,
                                                                rows,
                                                                cols,
                                                                valid_row_mask_device,
                                                                clamp_float(static_cast<float>(support_q), 0.50f, 0.99f),
                                                                0.0f,
                                                                1.0f,
                                                                histogram_device,
                                                                stream);

  coherent_power_threshold_valid_rows_kernel<<<blocks, threads, 0, stream>>>(d3,
                                                                              rows,
                                                                              cols,
                                                                              valid_row_mask_device,
                                                                              support_threshold,
                                                                              mask_device);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk support threshold kernel failed");
  }
  coherent_power_majority_smooth_kernel<<<blocks, threads, 0, stream>>>(mask_device,
                                                                         rows,
                                                                         cols,
                                                                         0,
                                                                         scratch.mask_buffers[0]);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk support majority smooth kernel failed");
  }

  support_px.assign(static_cast<size_t>(total), 0);
  if (cudaMemcpyAsync(support_px.data(),
                      scratch.mask_buffers[0],
                      static_cast<size_t>(total) * sizeof(uint8_t),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("chunk support mask readback failed");
  }
  support_px = prune_small_components(std::move(support_px), rows, cols, std::max(3, min_component_size / 2));

  if (cudaMemcpyAsync(mask_device,
                      support_px.data(),
                      static_cast<size_t>(total) * sizeof(uint8_t),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    throw std::runtime_error("chunk support mask upload failed");
  }

  coherent_power_copy_masked_with_sentinel_kernel<<<blocks, threads, 0, stream>>>(d3,
                                                                                   mask_device,
                                                                                   total,
                                                                                   -1.0f,
                                                                                   d4);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk final-mask copy kernel failed");
  }
  score_threshold = device_quantile_from_histogram_masked(d3,
                                                          mask_device,
                                                          total,
                                                          clamp_float(static_cast<float>(final_q), 0.50f, 0.99f),
                                                          0.0f,
                                                          1.0f,
                                                          histogram_device,
                                                          stream);

  coherent_power_threshold_mask_kernel<<<blocks, threads, 0, stream>>>(d3,
                                                                        mask_device,
                                                                        total,
                                                                        score_threshold,
                                                                        scratch.mask_buffers[1]);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk final threshold kernel failed");
  }
  coherent_power_majority_smooth_kernel<<<blocks, threads, 0, stream>>>(scratch.mask_buffers[1],
                                                                         rows,
                                                                         cols,
                                                                         0,
                                                                         mask_device);
  if (cudaGetLastError() != cudaSuccess) {
    throw std::runtime_error("chunk final majority smooth kernel failed");
  }

  combined_raw_px.assign(static_cast<size_t>(total), 0.0f);
  score_px.assign(static_cast<size_t>(total), 0.0f);
  mask_px.assign(static_cast<size_t>(total), 0);
  if (cudaMemcpyAsync(combined_raw_px.data(),
                      d2,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(score_px.data(),
                      d3,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(mask_px.data(),
                      mask_device,
                      static_cast<size_t>(total) * sizeof(uint8_t),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    throw std::runtime_error("chunk score/mask readback failed");
  }
  mask_px = prune_small_components(std::move(mask_px), rows, cols, min_component_size);
}

bool local_relative_power_support_map_cuda(const float* sxx_db_local,
                                           int rows,
                                           int cols,
                                           const std::vector<uint8_t>& valid_row_mask,
                                           float floor_q,
                                           int freq_window,
                                           int time_window,
                                           std::vector<float>& support) {
  const int total = rows * cols;
  if (total <= 0 || sxx_db_local == nullptr) {
    return false;
  }

  auto& scratch = chunk_cuda_scratch();
  if (!scratch.ensure_capacity(static_cast<size_t>(total))) {
    return false;
  }

  auto* d0 = scratch.buffers[0];
  auto* d1 = scratch.buffers[1];
  auto* d2 = scratch.buffers[2];
  auto* d3 = scratch.buffers[3];
  auto stream = scratch.stream;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  if (cudaMemcpyAsync(d0,
                      sxx_db_local,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }

  coherent_power_db_to_linear_kernel<<<blocks, threads, 0, stream>>>(d0, total, d1);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  std::vector<float> p_lin(static_cast<size_t>(total), 0.0f);
  if (cudaMemcpyAsync(p_lin.data(),
                      d1,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }

  std::vector<float> valid_values;
  valid_values.reserve(p_lin.size());
  for (int row = 0; row < rows; ++row) {
    if (!valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      valid_values.push_back(p_lin[flat_index(rows, cols, row, col)]);
    }
  }
  if (valid_values.empty()) {
    valid_values = p_lin;
  }
  const float p_floor = std::max(percentile_from_values(std::move(valid_values), floor_q), 1e-20f);

  coherent_power_relative_db_kernel<<<blocks, threads, 0, stream>>>(d1, p_floor, total, d2);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  coherent_power_box_mean_cols_kernel<<<blocks, threads, 0, stream>>>(d2,
                                                                       rows,
                                                                       cols,
                                                                       std::max(5, time_window | 1) / 2,
                                                                       d3);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  coherent_power_box_mean_rows_kernel<<<blocks, threads, 0, stream>>>(d3,
                                                                       rows,
                                                                       cols,
                                                                       std::max(3, freq_window | 1) / 2,
                                                                       d0);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  coherent_power_subtract_clamp_kernel<<<blocks, threads, 0, stream>>>(d2, d0, total, d3);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  support.assign(static_cast<size_t>(total), 0.0f);
  if (cudaMemcpyAsync(support.data(),
                      d3,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }

  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      support[flat_index(rows, cols, row, col)] = 0.0f;
    }
  }
  return true;
}

bool multi_scale_structure_tensor_gate_cuda(const float* sxx_db_local,
                                            int rows,
                                            int cols,
                                            std::vector<float>& coherence_max,
                                            std::vector<float>& energy_max,
                                            std::vector<float>& gate) {
  const int total = rows * cols;
  if (total <= 0 || sxx_db_local == nullptr) {
    return false;
  }

  auto& scratch = chunk_cuda_scratch();
  if (!scratch.ensure_capacity(static_cast<size_t>(total))) {
    return false;
  }
  auto* d0 = scratch.buffers[0];
  auto* d1 = scratch.buffers[1];
  auto* d2 = scratch.buffers[2];
  auto* d3 = scratch.buffers[3];
  auto* d4 = scratch.buffers[4];
  auto* d5 = scratch.buffers[5];
  auto stream = scratch.stream;

  if (cudaMemcpyAsync(d0,
                      sxx_db_local,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }

  const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
  const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
  const int total_threads = 256;
  const int total_blocks = (total + total_threads - 1) / total_threads;
  coherent_power_box_mean_cols_kernel<<<total_blocks, total_threads, 0, stream>>>(d0,
                                                                                    rows,
                                                                                    cols,
                                                                                    bg_time / 2,
                                                                                    d1);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  coherent_power_box_mean_rows_kernel<<<total_blocks, total_threads, 0, stream>>>(d1,
                                                                                    rows,
                                                                                    cols,
                                                                                    bg_freq / 2,
                                                                                    d2);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  coherent_power_subtract_clamp_kernel<<<total_blocks, total_threads, 0, stream>>>(d0, d2, total, d1);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }

  std::vector<float> residual_db(static_cast<size_t>(total), 0.0f);
  if (cudaMemcpyAsync(residual_db.data(),
                      d1,
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }

  auto residual_n = normalize_map01_local(residual_db, 5.0f, 99.0f);
  if (cudaMemcpyAsync(d0,
                      residual_n.data(),
                      static_cast<size_t>(total) * sizeof(float),
                      cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }

  const std::array<float, 3> scales{{0.8f, 1.6f, 3.2f}};
  coherence_max.assign(static_cast<size_t>(total), 0.0f);
  energy_max.assign(static_cast<size_t>(total), 0.0f);
  std::vector<float> coherence(static_cast<size_t>(total), 0.0f);
  std::vector<float> energy(static_cast<size_t>(total), 0.0f);

  for (const float grad_sigma : scales) {
    if (!gaussian_blur_2d_cuda(d0, rows, cols, grad_sigma, d1, d2, stream)) {
      return false;
    }
    coherent_power_gradient_kernel<<<total_blocks, total_threads, 0, stream>>>(d2, rows, cols, d3, d4);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }
    coherent_power_tensor_products_kernel<<<total_blocks, total_threads, 0, stream>>>(d3, d4, total, d1, d2, d5);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }

    const float integ_sigma = std::max(1.0f, 1.8f * grad_sigma);
    if (!gaussian_blur_2d_cuda(d1, rows, cols, integ_sigma, d0, d1, stream) ||
        !gaussian_blur_2d_cuda(d2, rows, cols, integ_sigma, d0, d2, stream) ||
        !gaussian_blur_2d_cuda(d5, rows, cols, integ_sigma, d0, d5, stream)) {
      return false;
    }

    coherent_power_eigen_metrics_kernel<<<total_blocks, total_threads, 0, stream>>>(d1,
                                                                                      d2,
                                                                                      d5,
                                                                                      total,
                                                                                      d3,
                                                                                      d4);
    if (cudaGetLastError() != cudaSuccess) {
      return false;
    }

    if (cudaMemcpyAsync(coherence.data(),
                        d3,
                        static_cast<size_t>(total) * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess ||
        cudaMemcpyAsync(energy.data(),
                        d4,
                        static_cast<size_t>(total) * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      return false;
    }

    coherence = normalize_map01_local(coherence, 5.0f, 99.0f);
    energy = normalize_map01_local(energy, 5.0f, 99.0f);
    for (size_t index = 0; index < coherence.size(); ++index) {
      coherence_max[index] = std::max(coherence_max[index], coherence[index]);
      energy_max[index] = std::max(energy_max[index], energy[index]);
    }
  }

  gate.assign(coherence_max.size(), 0.0f);
  for (size_t index = 0; index < gate.size(); ++index) {
    gate[index] = coherence_max[index] * std::sqrt(std::max(energy_max[index], 0.0f));
  }
  gate = normalize_map01_local(gate, 5.0f, 99.0f);
  return true;
}

void fill_nearly_continuous_time_gaps(std::vector<uint8_t>& mask,
                                      int rows,
                                      int cols,
                                      int max_gap_px,
                                      float min_continuity_ratio) {
  max_gap_px = std::max(0, max_gap_px);
  if (max_gap_px == 0) {
    return;
  }
  for (int row = 0; row < rows; ++row) {
    std::vector<int> active_cols;
    for (int col = 0; col < cols; ++col) {
      if (mask[flat_index(rows, cols, row, col)]) {
        active_cols.push_back(col);
      }
    }
    if (active_cols.size() < 2) {
      continue;
    }
    std::vector<int> run_starts;
    std::vector<int> run_stops;
    run_starts.push_back(active_cols.front());
    int previous_col = active_cols.front();
    for (size_t index = 1; index < active_cols.size(); ++index) {
      const int current_col = active_cols[index];
      if (current_col != previous_col + 1) {
        run_stops.push_back(previous_col + 1);
        run_starts.push_back(current_col);
      }
      previous_col = current_col;
    }
    run_stops.push_back(previous_col + 1);
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
                                     static_cast<float>(left_width + gap_width + right_width);
      if (continuity_ratio < min_continuity_ratio) {
        continue;
      }
      for (int col = left_stop; col < right_start; ++col) {
        mask[flat_index(rows, cols, row, col)] = 1;
      }
    }
  }
}

void binary_close_freq(std::vector<uint8_t>& mask, int rows, int cols, int bridge_freq_px) {
  bridge_freq_px = std::max(1, bridge_freq_px);
  if (bridge_freq_px <= 1) {
    return;
  }
  const int radius = bridge_freq_px / 2;
  std::vector<uint8_t> dilated(mask.size(), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      uint8_t value = 0;
      for (int src_row = std::max(0, row - radius); src_row <= std::min(rows - 1, row + radius); ++src_row) {
        if (mask[flat_index(rows, cols, src_row, col)]) {
          value = 1;
          break;
        }
      }
      dilated[flat_index(rows, cols, row, col)] = value;
    }
  }
  std::vector<uint8_t> eroded(mask.size(), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      uint8_t value = 1;
      for (int src_row = std::max(0, row - radius); src_row <= std::min(rows - 1, row + radius); ++src_row) {
        if (!dilated[flat_index(rows, cols, src_row, col)]) {
          value = 0;
          break;
        }
      }
      eroded[flat_index(rows, cols, row, col)] = value;
    }
  }
  mask.swap(eroded);
}

std::vector<std::pair<int, int>> true_runs(const std::vector<uint8_t>& mask_1d) {
  std::vector<std::pair<int, int>> runs;
  if (mask_1d.empty()) {
    return runs;
  }

  int run_start = -1;
  for (int index = 0; index < static_cast<int>(mask_1d.size()); ++index) {
    if (mask_1d[static_cast<size_t>(index)]) {
      if (run_start < 0) {
        run_start = index;
      }
      continue;
    }
    if (run_start >= 0) {
      runs.push_back({run_start, index});
      run_start = -1;
    }
  }
  if (run_start >= 0) {
    runs.push_back({run_start, static_cast<int>(mask_1d.size())});
  }
  return runs;
}

struct CandidateMask {
  std::vector<uint8_t> mask;
  std::string split_role = "unsplit";
  bool split_applied = false;
};

int component_envelope_area(const std::vector<uint8_t>& component_mask, int rows, int cols) {
  if (rows <= 0 || cols <= 0) {
    return 0;
  }
  int envelope_area = 0;
  for (int col = 0; col < cols; ++col) {
    int min_row = rows;
    int max_row = -1;
    for (int row = 0; row < rows; ++row) {
      if (!component_mask[flat_index(rows, cols, row, col)]) {
        continue;
      }
      min_row = std::min(min_row, row);
      max_row = std::max(max_row, row);
    }
    if (max_row >= min_row) {
      envelope_area += (max_row - min_row + 1);
    }
  }
  return envelope_area;
}

std::vector<CandidateMask> split_component_candidate_masks(const std::vector<uint8_t>& component_mask,
                                                           int rows,
                                                           int cols,
                                                           int min_freq_span_px,
                                                           int min_time_span_px) {
  std::vector<CandidateMask> unsplit {{component_mask, "unsplit", false}};
  if (rows <= 0 || cols <= 0) {
    return unsplit;
  }

  std::vector<int> active_cols;
  active_cols.reserve(cols);
  for (int col = 0; col < cols; ++col) {
    bool active = false;
    for (int row = 0; row < rows; ++row) {
      if (component_mask[flat_index(rows, cols, row, col)]) {
        active = true;
        break;
      }
    }
    if (active) {
      active_cols.push_back(col);
    }
  }
  if (static_cast<int>(active_cols.size()) < std::max(6, 2 * min_time_span_px)) {
    return unsplit;
  }

  std::vector<int> col_span(static_cast<size_t>(cols), 0);
  std::vector<int> active_spans;
  active_spans.reserve(active_cols.size());
  for (const int col : active_cols) {
    int min_row = rows;
    int max_row = -1;
    for (int row = 0; row < rows; ++row) {
      if (!component_mask[flat_index(rows, cols, row, col)]) {
        continue;
      }
      min_row = std::min(min_row, row);
      max_row = std::max(max_row, row);
    }
    if (max_row >= min_row) {
      col_span[static_cast<size_t>(col)] = max_row - min_row + 1;
      active_spans.push_back(col_span[static_cast<size_t>(col)]);
    }
  }
  if (static_cast<int>(active_spans.size()) < std::max(6, 2 * min_time_span_px)) {
    return unsplit;
  }

  std::sort(active_spans.begin(), active_spans.end());
  const size_t baseline_index = static_cast<size_t>(std::clamp<int>(
      static_cast<int>(std::floor(0.35 * static_cast<double>(std::max<int>(static_cast<int>(active_spans.size()) - 1, 0)))),
      0,
      std::max<int>(static_cast<int>(active_spans.size()) - 1, 0)));
  const float baseline_span = static_cast<float>(active_spans[baseline_index]);

  int global_min_row = rows;
  int global_max_row = -1;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      if (!component_mask[flat_index(rows, cols, row, col)]) {
        continue;
      }
      global_min_row = std::min(global_min_row, row);
      global_max_row = std::max(global_max_row, row);
      break;
    }
  }
  if (global_max_row < global_min_row) {
    return unsplit;
  }

  const int global_span = global_max_row - global_min_row + 1;
  const int burst_span_threshold = std::max({
      static_cast<int>(std::ceil(baseline_span * 1.8f)),
      static_cast<int>(std::ceil(baseline_span + std::max(4.0f, static_cast<float>(min_freq_span_px) * 0.5f))),
      min_freq_span_px,
  });
  if (burst_span_threshold >= global_span) {
    return unsplit;
  }

  std::vector<uint8_t> burst_cols_mask(static_cast<size_t>(cols), 0);
  for (const int col : active_cols) {
    burst_cols_mask[static_cast<size_t>(col)] = col_span[static_cast<size_t>(col)] >= burst_span_threshold ? 1 : 0;
  }

  std::vector<std::pair<int, int>> burst_runs;
  for (const auto& run : true_runs(burst_cols_mask)) {
    const int run_width = run.second - run.first;
    if (run_width < min_time_span_px) {
      continue;
    }
    if (run_width >= std::max(static_cast<int>(active_cols.size() * 0.7f), min_time_span_px + 1)) {
      continue;
    }
    burst_runs.push_back(run);
  }
  if (burst_runs.empty()) {
    return unsplit;
  }

  std::vector<int> non_burst_cols;
  non_burst_cols.reserve(active_cols.size());
  for (const int col : active_cols) {
    if (!burst_cols_mask[static_cast<size_t>(col)]) {
      non_burst_cols.push_back(col);
    }
  }
  if (static_cast<int>(non_burst_cols.size()) < std::max(4, 2 * min_time_span_px)) {
    return unsplit;
  }

  const int min_row_hits = std::max(2, static_cast<int>(std::ceil(static_cast<double>(non_burst_cols.size()) * 0.45)));
  std::vector<uint8_t> carrier_rows_mask(static_cast<size_t>(rows), 0);
  for (int row = 0; row < rows; ++row) {
    int row_hits = 0;
    for (const int col : non_burst_cols) {
      row_hits += component_mask[flat_index(rows, cols, row, col)] ? 1 : 0;
    }
    carrier_rows_mask[static_cast<size_t>(row)] = row_hits >= min_row_hits ? 1 : 0;
  }
  const auto carrier_row_runs = true_runs(carrier_rows_mask);
  if (carrier_row_runs.empty()) {
    return unsplit;
  }

  auto carrier_run = *std::max_element(
      carrier_row_runs.begin(),
      carrier_row_runs.end(),
      [](const auto& left, const auto& right) { return (left.second - left.first) < (right.second - right.first); });
  const int carrier_freq_start = carrier_run.first;
  const int carrier_freq_stop = carrier_run.second;
  const int carrier_freq_span = carrier_freq_stop - carrier_freq_start;
  if (carrier_freq_span < std::max(2, static_cast<int>(std::floor(baseline_span))) || carrier_freq_span >= burst_span_threshold) {
    return unsplit;
  }

  std::vector<uint8_t> carrier_mask(component_mask.size(), 0);
  for (int row = carrier_freq_start; row < carrier_freq_stop; ++row) {
    for (int col = 0; col < cols; ++col) {
      carrier_mask[flat_index(rows, cols, row, col)] = component_mask[flat_index(rows, cols, row, col)];
    }
  }
  const int carrier_count = static_cast<int>(std::count(carrier_mask.begin(), carrier_mask.end(), static_cast<uint8_t>(1)));
  if (carrier_count < std::max(2, min_time_span_px * 2)) {
    return unsplit;
  }

  std::vector<CandidateMask> candidates;
  candidates.push_back({carrier_mask, "persistent_carrier", true});
  for (const auto& run : burst_runs) {
    std::vector<uint8_t> burst_mask(component_mask.size(), 0);
    for (int row = 0; row < rows; ++row) {
      for (int col = run.first; col < run.second; ++col) {
        burst_mask[flat_index(rows, cols, row, col)] = component_mask[flat_index(rows, cols, row, col)];
      }
    }
    if (std::count(burst_mask.begin(), burst_mask.end(), static_cast<uint8_t>(1)) == 0) {
      continue;
    }
    candidates.push_back({burst_mask, "transient_wideband_burst", true});
  }

  if (candidates.size() < 2) {
    return unsplit;
  }
  return candidates;
}

GroupingResult group_signal_mask_regions(const std::vector<uint8_t>& mask,
                                         const std::vector<float>& score_map,
                                         int rows,
                                         int cols,
                                         const std::vector<uint8_t>& valid_row_mask,
                                         bool filter_detection_mask,
                                         int bridge_freq_px,
                                         int bridge_time_px,
                                         int min_component_size,
                                         int min_freq_span_px,
                                         int min_time_span_px,
                                         float min_density,
                                         float time_continuity_ratio = 0.85f) {
  std::vector<uint8_t> working_mask = mask;
  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      working_mask[flat_index(rows, cols, row, col)] = 0;
    }
  }

  std::vector<float> active_scores;
  active_scores.reserve(score_map.size());
  for (size_t index = 0; index < working_mask.size(); ++index) {
    if (working_mask[index]) {
      active_scores.push_back(score_map[index]);
    }
  }
  const float peak_score_floor = active_scores.empty() ? 0.0f : quantile_from_values(std::move(active_scores), 0.50f);

  if (!filter_detection_mask) {
    std::vector<uint8_t> grouped_mask = working_mask;
    std::vector<uint8_t> visited(mask.size(), 0);
    std::vector<DetectionBox> boxes;
    const std::array<std::pair<int, int>, 4> neighbors{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < cols; ++col) {
        const size_t seed = flat_index(rows, cols, row, col);
        if (!grouped_mask[seed] || visited[seed]) {
          continue;
        }
        std::queue<std::pair<int, int>> queue;
        queue.push({row, col});
        visited[seed] = 1;
        int min_row = row;
        int max_row = row;
        int min_col = col;
        int max_col = col;
        int filled_area = 0;
        float score_peak = 0.0f;
        float score_sum = 0.0f;
        while (!queue.empty()) {
          const auto [current_row, current_col] = queue.front();
          queue.pop();
          const size_t current_index = flat_index(rows, cols, current_row, current_col);
          ++filled_area;
          min_row = std::min(min_row, current_row);
          max_row = std::max(max_row, current_row);
          min_col = std::min(min_col, current_col);
          max_col = std::max(max_col, current_col);
          const float score = score_map[current_index];
          score_sum += score;
          score_peak = std::max(score_peak, score);
          for (const auto& [delta_row, delta_col] : neighbors) {
            const int next_row = current_row + delta_row;
            const int next_col = current_col + delta_col;
            if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
              continue;
            }
            const size_t next_index = flat_index(rows, cols, next_row, next_col);
            if (!grouped_mask[next_index] || visited[next_index]) {
              continue;
            }
            visited[next_index] = 1;
            queue.push({next_row, next_col});
          }
        }
        DetectionBox box;
        box.freq_start = min_row;
        box.freq_stop = max_row + 1;
        box.time_start = min_col;
        box.time_stop = max_col + 1;
        box.freq_span = box.freq_stop - box.freq_start;
        box.time_span = box.time_stop - box.time_start;
        box.filled_area = filled_area;
        const int bbox_area = std::max(box.freq_span * box.time_span, 1);
        box.bbox_density = static_cast<float>(filled_area) / static_cast<float>(bbox_area);
        box.envelope_density = box.bbox_density;
        box.density = box.bbox_density;
        box.score_mean = filled_area > 0 ? score_sum / static_cast<float>(filled_area) : 0.0f;
        box.score_peak = score_peak;
        box.parent_component_id = static_cast<int>(boxes.size()) + 1;
        boxes.push_back(std::move(box));
      }
    }
    return {grouped_mask, boxes, peak_score_floor};
  }

  std::vector<uint8_t> bridged_mask = working_mask;
  binary_close_freq(bridged_mask, rows, cols, bridge_freq_px);
  fill_nearly_continuous_time_gaps(bridged_mask, rows, cols, bridge_time_px, time_continuity_ratio);

  std::vector<uint8_t> grouped_mask(mask.size(), 0);
  std::vector<uint8_t> visited(mask.size(), 0);
  std::vector<DetectionBox> boxes;
  const std::array<std::pair<int, int>, 4> neighbors{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
  int output_component_id = 0;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t seed = flat_index(rows, cols, row, col);
      if (!bridged_mask[seed] || visited[seed]) {
        continue;
      }
      std::queue<std::pair<int, int>> queue;
      std::vector<size_t> component;
      queue.push({row, col});
      visited[seed] = 1;
      int min_row = row;
      int max_row = row;
      int min_col = col;
      int max_col = col;
      while (!queue.empty()) {
        const auto [current_row, current_col] = queue.front();
        queue.pop();
        const size_t current_index = flat_index(rows, cols, current_row, current_col);
        component.push_back(current_index);
        min_row = std::min(min_row, current_row);
        max_row = std::max(max_row, current_row);
        min_col = std::min(min_col, current_col);
        max_col = std::max(max_col, current_col);
        for (const auto& [delta_row, delta_col] : neighbors) {
          const int next_row = current_row + delta_row;
          const int next_col = current_col + delta_col;
          if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
            continue;
          }
          const size_t next_index = flat_index(rows, cols, next_row, next_col);
          if (!bridged_mask[next_index] || visited[next_index]) {
            continue;
          }
          visited[next_index] = 1;
          queue.push({next_row, next_col});
        }
      }

      const int parent_freq_start = min_row;
      const int parent_freq_stop = max_row + 1;
      const int parent_time_start = min_col;
      const int parent_time_stop = max_col + 1;
      const int local_rows = parent_freq_stop - parent_freq_start;
      const int local_cols = parent_time_stop - parent_time_start;
      std::vector<uint8_t> component_mask_local(static_cast<size_t>(local_rows) * static_cast<size_t>(local_cols), 0);
      for (const size_t index : component) {
        const int component_row = static_cast<int>(index / static_cast<size_t>(cols));
        const int component_col = static_cast<int>(index % static_cast<size_t>(cols));
        component_mask_local[flat_index(local_rows, local_cols, component_row - parent_freq_start, component_col - parent_time_start)] = 1;
      }

      const auto candidate_masks = split_component_candidate_masks(
          component_mask_local,
          local_rows,
          local_cols,
          min_freq_span_px,
          min_time_span_px);

      for (const auto& candidate : candidate_masks) {
        const auto& candidate_mask_local = candidate.mask;
        int local_min_row = local_rows;
        int local_max_row = -1;
        int local_min_col = local_cols;
        int local_max_col = -1;
        for (int local_row = 0; local_row < local_rows; ++local_row) {
          for (int local_col = 0; local_col < local_cols; ++local_col) {
            if (!candidate_mask_local[flat_index(local_rows, local_cols, local_row, local_col)]) {
              continue;
            }
            local_min_row = std::min(local_min_row, local_row);
            local_max_row = std::max(local_max_row, local_row);
            local_min_col = std::min(local_min_col, local_col);
            local_max_col = std::max(local_max_col, local_col);
          }
        }
        if (local_max_row < local_min_row || local_max_col < local_min_col) {
          continue;
        }

        const int freq_start = parent_freq_start + local_min_row;
        const int freq_stop = parent_freq_start + local_max_row + 1;
        const int time_start = parent_time_start + local_min_col;
        const int time_stop = parent_time_start + local_max_col + 1;
        const int freq_span = freq_stop - freq_start;
        const int time_span = time_stop - time_start;
        const int cropped_rows = local_max_row - local_min_row + 1;
        const int cropped_cols = local_max_col - local_min_col + 1;
        std::vector<uint8_t> cropped_candidate_mask(static_cast<size_t>(cropped_rows) * static_cast<size_t>(cropped_cols), 0);
        int filled_area = 0;
        float score_peak = 0.0f;
        float score_sum = 0.0f;
        for (int local_row = 0; local_row < cropped_rows; ++local_row) {
          for (int local_col = 0; local_col < cropped_cols; ++local_col) {
            const bool active = candidate_mask_local[flat_index(local_rows, local_cols, local_min_row + local_row, local_min_col + local_col)] != 0;
            cropped_candidate_mask[flat_index(cropped_rows, cropped_cols, local_row, local_col)] = active ? 1 : 0;
            if (!active) {
              continue;
            }
            ++filled_area;
            const float score = score_map[flat_index(rows, cols, freq_start + local_row, time_start + local_col)];
            score_sum += score;
            score_peak = std::max(score_peak, score);
          }
        }
        const int bbox_area = std::max(freq_span * time_span, 1);
        const int envelope_area = std::max(component_envelope_area(cropped_candidate_mask, cropped_rows, cropped_cols), 1);
        const float bbox_density = static_cast<float>(filled_area) / static_cast<float>(bbox_area);
        const float envelope_density = static_cast<float>(filled_area) / static_cast<float>(envelope_area);
        const float density = envelope_density;
        const float score_mean = filled_area > 0 ? score_sum / static_cast<float>(filled_area) : 0.0f;

        const bool keep_component = filled_area >= min_component_size &&
                                    freq_span >= min_freq_span_px &&
                                    time_span >= min_time_span_px &&
                                    density >= min_density &&
                                    score_peak >= peak_score_floor;
        ++output_component_id;
        if (!keep_component) {
          continue;
        }

        for (int local_row = 0; local_row < cropped_rows; ++local_row) {
          for (int local_col = 0; local_col < cropped_cols; ++local_col) {
            if (!cropped_candidate_mask[flat_index(cropped_rows, cropped_cols, local_row, local_col)]) {
              continue;
            }
            grouped_mask[flat_index(rows, cols, freq_start + local_row, time_start + local_col)] = 1;
          }
        }

        DetectionBox box;
        box.freq_start = freq_start;
        box.freq_stop = freq_stop;
        box.time_start = time_start;
        box.time_stop = time_stop;
        box.freq_span = freq_span;
        box.time_span = time_span;
        box.filled_area = filled_area;
        box.density = density;
        box.bbox_density = bbox_density;
        box.envelope_density = envelope_density;
        box.score_mean = score_mean;
        box.score_peak = score_peak;
        box.split_role = candidate.split_role;
        box.split_applied = candidate.split_applied;
        box.parent_component_id = output_component_id;
        boxes.push_back(std::move(box));
      }
    }
  }

  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      grouped_mask[flat_index(rows, cols, row, col)] = 0;
    }
  }

  return {grouped_mask, boxes, peak_score_floor};
}

std::vector<uint8_t> boxes_to_mask(int rows,
                                   int cols,
                                   const std::vector<DetectionBox>& boxes,
                                   const std::vector<uint8_t>& valid_row_mask) {
  std::vector<uint8_t> mask(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  for (const auto& box : boxes) {
    const int freq_start = clamp_int(box.freq_start, 0, rows);
    const int freq_stop = clamp_int(box.freq_stop, freq_start, rows);
    const int time_start = clamp_int(box.time_start, 0, cols);
    const int time_stop = clamp_int(box.time_stop, time_start, cols);
    for (int row = freq_start; row < freq_stop; ++row) {
      for (int col = time_start; col < time_stop; ++col) {
        mask[flat_index(rows, cols, row, col)] = 1;
      }
    }
  }
  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      continue;
    }
    for (int col = 0; col < cols; ++col) {
      mask[flat_index(rows, cols, row, col)] = 0;
    }
  }
  return mask;
}

bool boxes_overlap(const DetectionBox& box_a, const DetectionBox& box_b) {
  return box_a.freq_start < box_b.freq_stop && box_b.freq_start < box_a.freq_stop &&
         box_a.time_start < box_b.time_stop && box_b.time_start < box_a.time_stop;
}

bool boxes_should_merge(const DetectionBox& box_a, const DetectionBox& box_b) {
  if (!boxes_overlap(box_a, box_b)) {
    return false;
  }
  if (((box_a.split_role == "persistent_carrier") && (box_b.split_role == "transient_wideband_burst")) ||
      ((box_a.split_role == "transient_wideband_burst") && (box_b.split_role == "persistent_carrier"))) {
    return false;
  }
  return true;
}

bool boxes_should_merge_with_bridge(const DetectionBox& box_a,
                                    const DetectionBox& box_b,
                                    int bridge_freq_px,
                                    int bridge_time_px) {
  if (((box_a.split_role == "persistent_carrier") && (box_b.split_role == "transient_wideband_burst")) ||
      ((box_a.split_role == "transient_wideband_burst") && (box_b.split_role == "persistent_carrier"))) {
    return false;
  }

  const int expanded_a_freq_start = box_a.freq_start - std::max(0, bridge_freq_px / 2);
  const int expanded_a_freq_stop = box_a.freq_stop + std::max(0, bridge_freq_px / 2);
  const int expanded_b_freq_start = box_b.freq_start - std::max(0, bridge_freq_px / 2);
  const int expanded_b_freq_stop = box_b.freq_stop + std::max(0, bridge_freq_px / 2);
  const int expanded_a_time_start = box_a.time_start - std::max(0, bridge_time_px);
  const int expanded_a_time_stop = box_a.time_stop + std::max(0, bridge_time_px);
  const int expanded_b_time_start = box_b.time_start - std::max(0, bridge_time_px);
  const int expanded_b_time_stop = box_b.time_stop + std::max(0, bridge_time_px);

  return expanded_a_freq_start < expanded_b_freq_stop && expanded_b_freq_start < expanded_a_freq_stop &&
         expanded_a_time_start < expanded_b_time_stop && expanded_b_time_start < expanded_a_time_stop;
}

DetectionBox merge_box_cluster(const std::vector<DetectionBox>& cluster) {
  DetectionBox merged;
  merged.freq_start = std::numeric_limits<int>::max();
  merged.time_start = std::numeric_limits<int>::max();
  merged.freq_stop = 0;
  merged.time_stop = 0;
  int score_weight = 0;
  std::vector<std::string> split_roles;

  for (const auto& box : cluster) {
    merged.freq_start = std::min(merged.freq_start, box.freq_start);
    merged.freq_stop = std::max(merged.freq_stop, box.freq_stop);
    merged.time_start = std::min(merged.time_start, box.time_start);
    merged.time_stop = std::max(merged.time_stop, box.time_stop);
    merged.filled_area += box.filled_area;
    merged.score_peak = std::max(merged.score_peak, box.score_peak);
    const int box_weight = std::max(box.filled_area, 1);
    merged.score_mean += box.score_mean * static_cast<float>(box_weight);
    score_weight += box_weight;
    merged.split_applied = merged.split_applied || box.split_applied;
    if (std::find(split_roles.begin(), split_roles.end(), box.split_role) == split_roles.end()) {
      split_roles.push_back(box.split_role);
    }
    for (const int chunk_index : box.source_chunk_indices) {
      if (std::find(merged.source_chunk_indices.begin(), merged.source_chunk_indices.end(), chunk_index) == merged.source_chunk_indices.end()) {
        merged.source_chunk_indices.push_back(chunk_index);
      }
    }
  }

  merged.freq_span = merged.freq_stop - merged.freq_start;
  merged.time_span = merged.time_stop - merged.time_start;
  const int bbox_area = std::max(merged.freq_span * merged.time_span, 1);
  merged.density = static_cast<float>(merged.filled_area) / static_cast<float>(bbox_area);
  merged.score_mean = score_weight > 0 ? merged.score_mean / static_cast<float>(score_weight) : 0.0f;
  std::sort(split_roles.begin(), split_roles.end());
  merged.split_role = split_roles.size() == 1 ? split_roles.front() : "mixed";
  std::sort(merged.source_chunk_indices.begin(), merged.source_chunk_indices.end());
  return merged;
}

std::vector<DetectionBox> project_chunk_boxes_to_global(const std::vector<DetectionChunkResult>& chunk_results,
                                                        int rows,
                                                        int cols) {
  std::vector<DetectionBox> projected_boxes;
  for (const auto& chunk : chunk_results) {
    const int row_start = chunk.chunk.row_start;
    const int chunk_index = chunk.chunk.chunk_index;
    for (const auto& box : chunk.grouped_boxes) {
      DetectionBox projected = box;
      projected.freq_start = std::clamp(row_start + box.freq_start, 0, rows);
      projected.freq_stop = std::clamp(row_start + box.freq_stop, projected.freq_start, rows);
      projected.time_start = std::clamp(box.time_start, 0, cols);
      projected.time_stop = std::clamp(box.time_stop, projected.time_start, cols);
      projected.freq_span = projected.freq_stop - projected.freq_start;
      projected.time_span = projected.time_stop - projected.time_start;
      projected.source_chunk_indices = {chunk_index};
      if (projected.freq_span > 0 && projected.time_span > 0) {
        projected_boxes.push_back(std::move(projected));
      }
    }
  }
  return projected_boxes;
}

GroupingResult merge_projected_subsection_boxes(int rows,
                                                int cols,
                                                const std::vector<DetectionChunkResult>& chunk_results,
                                                const std::vector<float>& merged_score,
                                                const std::vector<uint8_t>& valid_row_mask,
                                                bool filter_detection_mask,
                                                int bridge_freq_px,
                                                int bridge_time_px,
                                                int min_component_size,
                                                int min_freq_span_px,
                                                int min_time_span_px,
                                                float min_density,
                                                float time_continuity_ratio) {
  auto projected_boxes = project_chunk_boxes_to_global(chunk_results, rows, cols);
  if (projected_boxes.empty()) {
    return {std::vector<uint8_t>(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0), {}, 0.0f};
  }

  std::vector<DetectionBox> merged_boxes;
  std::vector<uint8_t> visited(projected_boxes.size(), 0);
  for (size_t start_index = 0; start_index < projected_boxes.size(); ++start_index) {
    if (visited[start_index]) {
      continue;
    }
    std::vector<size_t> pending {start_index};
    visited[start_index] = 1;
    std::vector<DetectionBox> cluster;
    while (!pending.empty()) {
      const size_t current_index = pending.back();
      pending.pop_back();
      cluster.push_back(projected_boxes[current_index]);
      for (size_t other_index = 0; other_index < projected_boxes.size(); ++other_index) {
        if (visited[other_index]) {
          continue;
        }
        if (boxes_should_merge_with_bridge(projected_boxes[current_index],
                                          projected_boxes[other_index],
                                          filter_detection_mask ? bridge_freq_px : 0,
                                          filter_detection_mask ? bridge_time_px : 0)) {
          visited[other_index] = 1;
          pending.push_back(other_index);
        }
      }
    }

    auto merged = merge_box_cluster(cluster);
    const bool keep_box = merged.filled_area >= min_component_size &&
                          merged.freq_span >= min_freq_span_px &&
                          merged.time_span >= min_time_span_px &&
                          merged.density >= min_density;
    if (keep_box) {
      merged_boxes.push_back(std::move(merged));
    }
  }

  return {boxes_to_mask(rows, cols, merged_boxes, valid_row_mask), merged_boxes, 0.0f};
}

std::vector<ChunkPlanEntry> build_frequency_chunks(int rows,
                                                   double resolution_hz,
                                                   double chunk_bandwidth_hz,
                                                   double chunk_overlap_hz,
                                                   const std::vector<uint8_t>& valid_row_mask,
                                                   double uncalibrated_chunk_fraction,
                                                   double uncalibrated_overlap_fraction) {
  std::vector<int> valid_rows;
  valid_rows.reserve(rows);
  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      valid_rows.push_back(row);
    }
  }
  if (valid_rows.empty()) {
    return {};
  }

  int chunk_rows = rows;
  int overlap_rows = 0;
  if (resolution_hz > 0.0 && chunk_bandwidth_hz > 0.0) {
    chunk_rows = clamp_int(static_cast<int>(std::llround(chunk_bandwidth_hz / resolution_hz)), 16, rows);
    overlap_rows = clamp_int(static_cast<int>(std::llround(chunk_overlap_hz / resolution_hz)), 0, chunk_rows - 1);
  } else {
    chunk_rows = clamp_int(static_cast<int>(std::llround(static_cast<double>(valid_rows.size()) * uncalibrated_chunk_fraction)),
                           16,
                           static_cast<int>(valid_rows.size()));
    overlap_rows = clamp_int(static_cast<int>(std::llround(static_cast<double>(chunk_rows) * uncalibrated_overlap_fraction)),
                             0,
                             chunk_rows - 1);
  }
  const int step_rows = std::max(1, chunk_rows - overlap_rows);

  std::vector<ChunkPlanEntry> chunks;
  int chunk_index = 0;
  for (int start_index = 0; start_index < static_cast<int>(valid_rows.size()); start_index += step_rows) {
    const int stop_index = std::min(start_index + chunk_rows, static_cast<int>(valid_rows.size()));
    if (stop_index - start_index < 16) {
      if (!chunks.empty()) {
        break;
      }
      continue;
    }
    const int row_start = valid_rows[static_cast<size_t>(start_index)];
    const int row_stop = valid_rows[static_cast<size_t>(stop_index - 1)] + 1;
    ChunkPlanEntry entry;
    entry.chunk_index = chunk_index++;
    entry.row_start = row_start;
    entry.row_stop = row_stop;
    entry.freq_start_hz = resolution_hz > 0.0 ? static_cast<double>(row_start) * resolution_hz : static_cast<double>(row_start);
    entry.freq_stop_hz = resolution_hz > 0.0 ? static_cast<double>(row_stop - 1) * resolution_hz : static_cast<double>(row_stop - 1);
    chunks.push_back(entry);
    if (stop_index >= static_cast<int>(valid_rows.size())) {
      break;
    }
  }
  return chunks;
}

std::vector<float> row_quantile(const std::vector<float>& image, int rows, int cols, float percentile) {
  std::vector<float> output(static_cast<size_t>(rows), 0.0f);
  std::vector<float> row_values;
  row_values.reserve(static_cast<size_t>(cols));
  for (int row = 0; row < rows; ++row) {
    row_values.clear();
    for (int col = 0; col < cols; ++col) {
      row_values.push_back(image[flat_index(rows, cols, row, col)]);
    }
    output[static_cast<size_t>(row)] = percentile_from_values(row_values, percentile);
  }
  return output;
}

std::vector<float> row_quantile(const float* image, int rows, int cols, float percentile) {
  std::vector<float> output(static_cast<size_t>(rows), 0.0f);
  std::vector<float> row_values;
  row_values.reserve(static_cast<size_t>(cols));
  for (int row = 0; row < rows; ++row) {
    row_values.clear();
    for (int col = 0; col < cols; ++col) {
      row_values.push_back(image[flat_index(rows, cols, row, col)]);
    }
    output[static_cast<size_t>(row)] = percentile_from_values(row_values, percentile);
  }
  return output;
}

std::vector<float> apply_frontend_correction(const float* power_db,
                                             int rows,
                                             int cols,
                                             const std::vector<uint8_t>& valid_row_mask,
                                             float row_q,
                                             float reference_q,
                                             float smooth_sigma,
                                             float max_boost_db,
                                             std::vector<float>& boost_db_out) {
  auto row_floor = row_quantile(power_db, rows, cols, row_q);
  auto response = gaussian_blur_1d(row_floor, smooth_sigma);
  std::vector<float> valid_response;
  valid_response.reserve(static_cast<size_t>(rows));
  for (int row = 0; row < rows; ++row) {
    if (valid_row_mask[static_cast<size_t>(row)]) {
      valid_response.push_back(response[static_cast<size_t>(row)]);
    }
  }
  if (valid_response.empty()) {
    valid_response = response;
  }
  const float reference_db = percentile_from_values(std::move(valid_response), reference_q);
  boost_db_out.assign(static_cast<size_t>(rows), 0.0f);
  const size_t power_db_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  std::vector<float> corrected(power_db_size, 0.0f);
  for (int row = 0; row < rows; ++row) {
    boost_db_out[static_cast<size_t>(row)] = clamp_float(reference_db - response[static_cast<size_t>(row)], 0.0f, max_boost_db);
    for (int col = 0; col < cols; ++col) {
      corrected[flat_index(rows, cols, row, col)] =
          power_db[flat_index(rows, cols, row, col)] + boost_db_out[static_cast<size_t>(row)];
    }
  }
  return corrected;
}

DetectionChunkResult detect_chunk_coherent_power(const float* corrected_chunk,
                                                 int rows,
                                                 int cols,
                                                 const ChunkPlanEntry& chunk,
                                                 const std::vector<uint8_t>& chunk_valid_row_mask,
                                                 double coherence_weight,
                                                 double power_weight,
                                                 const std::string& power_assist_mode,
                                                 double power_floor_time_q,
                                                 double power_floor_global_q,
                                                 double power_excess_start_db,
                                                 double power_excess_full_db,
                                                 double power_local_blend,
                                                 double coherence_gate_start,
                                                 double coherence_gate_full,
                                                 double coherence_bridge_bias,
                                                 double coherence_power_joint_weight,
                                                 double support_q,
                                                 double final_q,
                                                 int min_component_size,
                                                 bool filter_detection_mask,
                                                 int grouping_bridge_freq_px,
                                                 int grouping_bridge_time_px,
                                                 int grouping_min_component_size,
                                                 int grouping_min_freq_span_px,
                                                 int grouping_min_time_span_px,
                                                 double grouping_min_density) {
  auto time_chunk_stage_ms = [](auto&& fn) {
    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
  };

  DetectionChunkResult result;
  result.chunk = chunk;
  result.valid_row_mask = chunk_valid_row_mask;
  result.valid_score_mask = valid_row_mask_to_full_mask(chunk_valid_row_mask, cols);
  const size_t chunk_size = static_cast<size_t>(rows) * static_cast<size_t>(cols);

  std::vector<float> coherence_px;
  std::vector<float> energy_px;
  std::vector<float> gate_px;
  result.stage_ms[kChunkStructureTensorStage] = time_chunk_stage_ms([&] {
    if (!multi_scale_structure_tensor_gate_cuda(corrected_chunk, rows, cols, coherence_px, energy_px, gate_px)) {
      throw std::runtime_error("CUDA structure-tensor path failed in detect_chunk_coherent_power");
    }
  });
  (void)energy_px;
  (void)gate_px;
  (void)coherence_weight;
  (void)power_weight;
  const float* power_px_device = nullptr;
  result.stage_ms[kChunkPowerSupportStage] = time_chunk_stage_ms([&] {
    if (!build_power_assist_map_device(corrected_chunk,
                                       rows,
                                       cols,
                                       chunk_valid_row_mask,
                                       power_assist_mode,
                                       static_cast<float>(power_floor_time_q),
                                       static_cast<float>(power_floor_global_q),
                                       static_cast<float>(power_excess_start_db),
                                       static_cast<float>(power_excess_full_db),
                                       static_cast<float>(power_local_blend),
                                       &result.power_support_stage_ms,
                                       &power_px_device)) {
      throw std::runtime_error("GPU power-assist path failed in detect_chunk_coherent_power");
    }
  });
  result.stage_ms[kChunkScoreThresholdStage] = time_chunk_stage_ms([&] {
    run_chunk_score_masks_gpu(coherence_px,
                              power_px_device,
                              rows,
                              cols,
                              chunk_valid_row_mask,
                              support_q,
                              final_q,
                              coherence_gate_start,
                              coherence_gate_full,
                              coherence_bridge_bias,
                              coherence_power_joint_weight,
                              min_component_size,
                              result.combined_raw_px,
                              result.score_px,
                              result.support_px,
                              result.mask_px,
                              result.support_threshold,
                              result.score_threshold);
  });

  result.stage_ms[kChunkMaskSmoothStage] = time_chunk_stage_ms([&] {
    for (size_t index = 0; index < result.score_px.size(); ++index) {
      if (result.valid_score_mask[index]) {
        continue;
      }
      result.combined_raw_px[index] = 0.0f;
      result.score_px[index] = 0.0f;
      result.support_px[index] = 0;
      result.mask_px[index] = 0;
    }
  });

  result.stage_ms[kChunkGroupingStage] = time_chunk_stage_ms([&] {
    const auto grouping = group_signal_mask_regions(result.mask_px,
                                                    result.score_px,
                                                    rows,
                                                    cols,
                                                    chunk_valid_row_mask,
                                                    filter_detection_mask,
                                                    grouping_bridge_freq_px,
                                                    grouping_bridge_time_px,
                                                    std::max(grouping_min_component_size, min_component_size),
                                                    grouping_min_freq_span_px,
                                                    grouping_min_time_span_px,
                                                    static_cast<float>(grouping_min_density));
    result.grouped_mask = grouping.grouped_mask;
    result.grouped_boxes = grouping.boxes;
  });
  return result;
}

PipelineSummary run_reference_pipeline(const float* power_db,
                                       int src_rows,
                                       int src_cols,
                                       int ignore_bins_per_side,
                                       double resolution_hz,
                                       double chunk_bandwidth_hz,
                                       double chunk_overlap_hz,
                                       double uncalibrated_chunk_fraction,
                                       double uncalibrated_overlap_fraction,
                                       double ignore_sideband_percent,
                                       double frontend_row_q,
                                       double frontend_reference_q,
                                       double frontend_smooth_sigma,
                                       double frontend_max_boost_db,
                                       double coherence_weight,
                                       double power_weight,
                                       const std::string& power_assist_mode,
                                       double power_floor_time_q,
                                       double power_floor_global_q,
                                       double power_excess_start_db,
                                       double power_excess_full_db,
                                       double power_local_blend,
                                       double coherence_gate_start,
                                       double coherence_gate_full,
                                       double coherence_bridge_bias,
                                       double coherence_power_joint_weight,
                                       double coherence_power_support_q,
                                       double coherence_power_q,
                                       int min_component_size,
                                       bool filter_detection_mask,
                                       double grouping_seed_score_q,
                                       int grouping_bridge_freq_px,
                                       int grouping_bridge_time_px,
                                       int grouping_min_component_size,
                                       int grouping_min_freq_span_px,
                                       int grouping_min_time_span_px,
                                       double grouping_min_density,
                                       double grouping_time_continuity_ratio) {
  if (power_db == nullptr) {
    throw std::invalid_argument("reference pipeline requires a power_db buffer");
  }
  std::vector<uint8_t> valid_row_mask(static_cast<size_t>(src_rows), 1);
  if (ignore_bins_per_side == 0) {
    ignore_bins_per_side = compute_ignore_bins_per_side(
        src_rows, resolution_hz, ignore_sideband_percent, 0.0);
  }
  for (int row = 0; row < ignore_bins_per_side; ++row) {
    valid_row_mask[static_cast<size_t>(row)] = 0;
    valid_row_mask[static_cast<size_t>(src_rows - 1 - row)] = 0;
  }

  std::vector<float> boost_db;
  std::vector<float> corrected_sxx_db;
  const auto frontend_start = std::chrono::steady_clock::now();
  corrected_sxx_db = apply_frontend_correction(power_db,
                                               src_rows,
                                               src_cols,
                                               valid_row_mask,
                                               static_cast<float>(frontend_row_q),
                                               static_cast<float>(frontend_reference_q),
                                               static_cast<float>(frontend_smooth_sigma),
                                               static_cast<float>(frontend_max_boost_db),
                                               boost_db);
  const double frontend_stage_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frontend_start).count();
  (void)boost_db;

  PipelineSummary summary;
  summary.reference_stage_ms[kReferenceFrontendStage] = frontend_stage_ms;
  summary.ignore_bins_per_side = ignore_bins_per_side;

  const auto chunk_plan = build_frequency_chunks(src_rows,
                                                 resolution_hz,
                                                 chunk_bandwidth_hz,
                                                 chunk_overlap_hz,
                                                 valid_row_mask,
                                                 uncalibrated_chunk_fraction,
                                                 uncalibrated_overlap_fraction);
  summary.subsection_count = static_cast<int>(chunk_plan.size());
  std::vector<DetectionChunkResult> chunk_results;
  chunk_results.reserve(chunk_plan.size());
  auto time_reference_stage_ms = [](auto&& fn) {
    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
  };
  summary.reference_stage_ms[kReferenceChunkWallStage] = time_reference_stage_ms([&] {
    chunk_results = chunk_worker_pool().run(
        chunk_plan.size(),
        [&](size_t chunk_plan_index) {
          const auto chunk_start = std::chrono::steady_clock::now();
          const auto& chunk = chunk_plan[chunk_plan_index];
          const int chunk_rows = chunk.row_stop - chunk.row_start;
          std::vector<uint8_t> chunk_valid_row_mask(static_cast<size_t>(chunk_rows), 0);
          for (int row = 0; row < chunk_rows; ++row) {
            chunk_valid_row_mask[static_cast<size_t>(row)] = valid_row_mask[static_cast<size_t>(chunk.row_start + row)];
          }
          const float* corrected_chunk = corrected_sxx_db.data() +
                                         static_cast<size_t>(chunk.row_start) * static_cast<size_t>(src_cols);
          auto result = detect_chunk_coherent_power(corrected_chunk,
                                                    chunk_rows,
                                                    src_cols,
                                                    chunk,
                                                    chunk_valid_row_mask,
                                                    coherence_weight,
                                                    power_weight,
                                                    power_assist_mode,
                                                    power_floor_time_q,
                                                    power_floor_global_q,
                                                    power_excess_start_db,
                                                    power_excess_full_db,
                                                    power_local_blend,
                                                    coherence_gate_start,
                                                    coherence_gate_full,
                                                    coherence_bridge_bias,
                                                    coherence_power_joint_weight,
                                                    coherence_power_support_q,
                                                    coherence_power_q,
                                                    min_component_size,
                                                    filter_detection_mask,
                                                    grouping_bridge_freq_px,
                                                    grouping_bridge_time_px,
                                                    grouping_min_component_size,
                                                    grouping_min_freq_span_px,
                                                    grouping_min_time_span_px,
                                                    grouping_min_density);
          result.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - chunk_start).count();
          return result;
        });
  });
  double chunk_sum_ms = 0.0;
  double chunk_max_ms = 0.0;
  for (const auto& chunk_result : chunk_results) {
    chunk_sum_ms += chunk_result.compute_ms;
    chunk_max_ms = std::max(chunk_max_ms, chunk_result.compute_ms);
    for (size_t stage_index = 0;
         stage_index < holoscan::ops::CoherentPowerSignalDetector::kChunkTimingStageCount;
         ++stage_index) {
      summary.chunk_stage_sum_ms[stage_index] += chunk_result.stage_ms[stage_index];
      summary.chunk_stage_peak_ms[stage_index] = std::max(summary.chunk_stage_peak_ms[stage_index], chunk_result.stage_ms[stage_index]);
    }
    for (size_t stage_index = 0;
         stage_index < holoscan::ops::CoherentPowerSignalDetector::kPowerSupportTimingStageCount;
         ++stage_index) {
      summary.power_support_stage_sum_ms[stage_index] += chunk_result.power_support_stage_ms[stage_index];
      summary.power_support_stage_peak_ms[stage_index] = std::max(summary.power_support_stage_peak_ms[stage_index], chunk_result.power_support_stage_ms[stage_index]);
    }
  }
  summary.reference_stage_ms[kReferenceChunkSumStage] = chunk_sum_ms;
  summary.reference_stage_ms[kReferenceChunkMaxStage] = chunk_max_ms;

  GroupingResult merged_grouping;
  summary.reference_stage_ms[kReferenceMergeStage] = time_reference_stage_ms([&] {
    merged_grouping = merge_projected_subsection_boxes(src_rows,
                                                       src_cols,
                                                       chunk_results,
                                                       std::vector<float>{},
                                                       valid_row_mask,
                                                       filter_detection_mask,
                                                       grouping_bridge_freq_px,
                                                       grouping_bridge_time_px,
                                                       std::max(grouping_min_component_size, min_component_size),
                                                       grouping_min_freq_span_px,
                                                       grouping_min_time_span_px,
                                                       static_cast<float>(grouping_min_density),
                                                       static_cast<float>(grouping_time_continuity_ratio));
    summary.grouped_box_count = static_cast<int>(merged_grouping.boxes.size());
    summary.merged_threshold = 0.0f;
    summary.seed_threshold = 0.0f;
  });

  summary.reference_stage_ms[kReferenceGroupingStage] = time_reference_stage_ms([&] {
    const auto final_mask_u8 = boxes_to_mask(src_rows, src_cols, merged_grouping.boxes, valid_row_mask);
    summary.final_mask.assign(final_mask_u8.size(), 0.0f);
    for (size_t index = 0; index < final_mask_u8.size(); ++index) {
      summary.final_mask[index] = final_mask_u8[index] ? 1.0f : 0.0f;
    }
  });
  return summary;
}

PipelineSummary run_reference_pipeline_corrected(const float* corrected_sxx_db,
                                                 int src_rows,
                                                 int src_cols,
                                                 int ignore_bins_per_side,
                                                 double resolution_hz,
                                                 double chunk_bandwidth_hz,
                                                 double chunk_overlap_hz,
                                                 double uncalibrated_chunk_fraction,
                                                 double uncalibrated_overlap_fraction,
                                                 double ignore_sideband_percent,
                                                 double coherence_weight,
                                                 double power_weight,
                                                 const std::string& power_assist_mode,
                                                 double power_floor_time_q,
                                                 double power_floor_global_q,
                                                 double power_excess_start_db,
                                                 double power_excess_full_db,
                                                 double power_local_blend,
                                                 double coherence_gate_start,
                                                 double coherence_gate_full,
                                                 double coherence_bridge_bias,
                                                 double coherence_power_joint_weight,
                                                 double coherence_power_support_q,
                                                 double coherence_power_q,
                                                 int min_component_size,
                                                 bool filter_detection_mask,
                                                 double grouping_seed_score_q,
                                                 int grouping_bridge_freq_px,
                                                 int grouping_bridge_time_px,
                                                 int grouping_min_component_size,
                                                 int grouping_min_freq_span_px,
                                                 int grouping_min_time_span_px,
                                                 double grouping_min_density,
                                                 double grouping_time_continuity_ratio,
                                                 double frontend_stage_ms) {
  if (corrected_sxx_db == nullptr) {
    throw std::invalid_argument("reference pipeline requires a corrected_sxx_db buffer");
  }
  PipelineSummary summary;
  auto time_reference_stage_ms = [](auto&& fn) {
    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - stage_start).count();
  };
  if (ignore_bins_per_side == 0) {
    ignore_bins_per_side = compute_ignore_bins_per_side(
        src_rows, resolution_hz, ignore_sideband_percent, 0.0);
  }
  summary.ignore_bins_per_side = ignore_bins_per_side;

  std::vector<uint8_t> valid_row_mask(static_cast<size_t>(src_rows), 1);
  for (int row = 0; row < ignore_bins_per_side; ++row) {
    valid_row_mask[static_cast<size_t>(row)] = 0;
    valid_row_mask[static_cast<size_t>(src_rows - 1 - row)] = 0;
  }

  summary.reference_stage_ms[kReferenceFrontendStage] = frontend_stage_ms;

  const auto chunk_plan = build_frequency_chunks(src_rows,
                                                 resolution_hz,
                                                 chunk_bandwidth_hz,
                                                 chunk_overlap_hz,
                                                 valid_row_mask,
                                                 uncalibrated_chunk_fraction,
                                                 uncalibrated_overlap_fraction);
  summary.subsection_count = static_cast<int>(chunk_plan.size());
  std::vector<DetectionChunkResult> chunk_results;
  chunk_results.reserve(chunk_plan.size());
  summary.reference_stage_ms[kReferenceChunkWallStage] = time_reference_stage_ms([&] {
    chunk_results = chunk_worker_pool().run(
        chunk_plan.size(),
        [&](size_t chunk_plan_index) {
          const auto chunk_start = std::chrono::steady_clock::now();
          const auto& chunk = chunk_plan[chunk_plan_index];
          const int chunk_rows = chunk.row_stop - chunk.row_start;
          std::vector<uint8_t> chunk_valid_row_mask(static_cast<size_t>(chunk_rows), 0);
          for (int row = 0; row < chunk_rows; ++row) {
            chunk_valid_row_mask[static_cast<size_t>(row)] = valid_row_mask[static_cast<size_t>(chunk.row_start + row)];
          }
          const float* corrected_chunk = corrected_sxx_db +
                                         static_cast<size_t>(chunk.row_start) * static_cast<size_t>(src_cols);
          auto result = detect_chunk_coherent_power(corrected_chunk,
                                                    chunk_rows,
                                                    src_cols,
                                                    chunk,
                                                    chunk_valid_row_mask,
                                                    coherence_weight,
                                                    power_weight,
                                                    power_assist_mode,
                                                    power_floor_time_q,
                                                    power_floor_global_q,
                                                    power_excess_start_db,
                                                    power_excess_full_db,
                                                    power_local_blend,
                                                    coherence_gate_start,
                                                    coherence_gate_full,
                                                    coherence_bridge_bias,
                                                    coherence_power_joint_weight,
                                                    coherence_power_support_q,
                                                    coherence_power_q,
                                                    min_component_size,
                                                    filter_detection_mask,
                                                    grouping_bridge_freq_px,
                                                    grouping_bridge_time_px,
                                                    grouping_min_component_size,
                                                    grouping_min_freq_span_px,
                                                    grouping_min_time_span_px,
                                                    grouping_min_density);
          result.compute_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - chunk_start).count();
          return result;
        });
  });
  double chunk_sum_ms = 0.0;
  double chunk_max_ms = 0.0;
  for (const auto& chunk_result : chunk_results) {
    chunk_sum_ms += chunk_result.compute_ms;
    chunk_max_ms = std::max(chunk_max_ms, chunk_result.compute_ms);
    for (size_t stage_index = 0;
         stage_index < holoscan::ops::CoherentPowerSignalDetector::kChunkTimingStageCount;
         ++stage_index) {
      summary.chunk_stage_sum_ms[stage_index] += chunk_result.stage_ms[stage_index];
      summary.chunk_stage_peak_ms[stage_index] = std::max(summary.chunk_stage_peak_ms[stage_index], chunk_result.stage_ms[stage_index]);
    }
    for (size_t stage_index = 0;
         stage_index < holoscan::ops::CoherentPowerSignalDetector::kPowerSupportTimingStageCount;
         ++stage_index) {
      summary.power_support_stage_sum_ms[stage_index] += chunk_result.power_support_stage_ms[stage_index];
      summary.power_support_stage_peak_ms[stage_index] = std::max(summary.power_support_stage_peak_ms[stage_index], chunk_result.power_support_stage_ms[stage_index]);
    }
  }
  summary.reference_stage_ms[kReferenceChunkSumStage] = chunk_sum_ms;
  summary.reference_stage_ms[kReferenceChunkMaxStage] = chunk_max_ms;

  GroupingResult merged_grouping;
  summary.reference_stage_ms[kReferenceMergeStage] = time_reference_stage_ms([&] {
    merged_grouping = merge_projected_subsection_boxes(src_rows,
                                                       src_cols,
                                                       chunk_results,
                                                       std::vector<float>{},
                                                       valid_row_mask,
                                                       filter_detection_mask,
                                                       grouping_bridge_freq_px,
                                                       grouping_bridge_time_px,
                                                       std::max(grouping_min_component_size, min_component_size),
                                                       grouping_min_freq_span_px,
                                                       grouping_min_time_span_px,
                                                       static_cast<float>(grouping_min_density),
                                                       static_cast<float>(grouping_time_continuity_ratio));
    summary.grouped_box_count = static_cast<int>(merged_grouping.boxes.size());
    summary.merged_threshold = 0.0f;
    summary.seed_threshold = 0.0f;
  });

  summary.reference_stage_ms[kReferenceGroupingStage] = time_reference_stage_ms([&] {
    const auto final_mask_u8 = boxes_to_mask(src_rows, src_cols, merged_grouping.boxes, valid_row_mask);
    summary.final_mask.assign(final_mask_u8.size(), 0.0f);
    for (size_t index = 0; index < final_mask_u8.size(); ++index) {
      summary.final_mask[index] = final_mask_u8[index] ? 1.0f : 0.0f;
    }
  });
  return summary;
}

}  // namespace

namespace holoscan::ops {

CoherentPowerReferenceResult run_coherent_power_reference_validation(
    const std::vector<coherent_power_complex>& input_tensor,
    int src_rows,
    int src_cols,
    double resolution_hz,
    const CoherentPowerReferenceConfig& config) {
  if (src_rows <= 0 || src_cols <= 0) {
    throw std::invalid_argument("coherent power reference validation requires positive tensor dimensions");
  }
  if (input_tensor.size() != static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols)) {
    throw std::invalid_argument("coherent power reference validation received mismatched tensor size");
  }

  CoherentPowerReferenceResult result;
  result.src_rows = src_rows;
  result.src_cols = src_cols;
  result.dst_rows = src_rows;
  result.dst_cols = src_cols;
  result.span_hz = resolution_hz > 0.0 ? resolution_hz * static_cast<double>(src_rows) : 0.0;
  result.sample_rate_hz = result.span_hz;
  result.frequency_axis_calibrated = resolution_hz > 0.0;

  result.power_db.assign(input_tensor.size(), 0.0f);
  for (size_t index = 0; index < input_tensor.size(); ++index) {
    const float re = input_tensor[index].real();
    const float im = input_tensor[index].imag();
    result.power_db[index] = 10.0f * std::log10(re * re + im * im + 1e-12f);
  }

  const int ignore_bins_per_side = compute_ignore_bins_per_side(
      src_rows, resolution_hz, config.ignore_sideband_percent, config.ignore_sideband_hz);
  result.ignore_bins_per_side = ignore_bins_per_side;

  std::vector<uint8_t> valid_row_mask(static_cast<size_t>(src_rows), 1);
  for (int row = 0; row < ignore_bins_per_side; ++row) {
    valid_row_mask[static_cast<size_t>(row)] = 0;
    valid_row_mask[static_cast<size_t>(src_rows - 1 - row)] = 0;
  }

  std::vector<float> boost_db;
  result.corrected_sxx_db = apply_frontend_correction(result.power_db.data(),
                                                      src_rows,
                                                      src_cols,
                                                      valid_row_mask,
                                                      static_cast<float>(config.frontend_row_q),
                                                      static_cast<float>(config.frontend_reference_q),
                                                      static_cast<float>(config.frontend_smooth_sigma),
                                                      static_cast<float>(config.frontend_max_boost_db),
                                                      boost_db);

  const auto summary = run_reference_pipeline(result.power_db.data(),
                                              src_rows,
                                              src_cols,
                                              ignore_bins_per_side,
                                              resolution_hz,
                                              config.chunk_bandwidth_hz,
                                              config.chunk_overlap_hz,
                                              config.uncalibrated_chunk_fraction,
                                              config.uncalibrated_overlap_fraction,
                                              config.ignore_sideband_percent,
                                              config.frontend_row_q,
                                              config.frontend_reference_q,
                                              config.frontend_smooth_sigma,
                                              config.frontend_max_boost_db,
                                              config.coherence_weight,
                                              config.power_weight,
                                              config.power_assist_mode,
                                              config.power_floor_time_q,
                                              config.power_floor_global_q,
                                              config.power_excess_start_db,
                                              config.power_excess_full_db,
                                              config.power_local_blend,
                                              config.coherence_gate_start,
                                              config.coherence_gate_full,
                                              config.coherence_bridge_bias,
                                              config.coherence_power_joint_weight,
                                              config.coherence_power_support_q,
                                              config.coherence_power_q,
                                              config.min_component_size,
                                              config.filter_detection_mask,
                                              config.grouping_seed_score_q,
                                              config.grouping_bridge_freq_px,
                                              config.grouping_bridge_time_px,
                                              config.grouping_min_component_size,
                                              config.grouping_min_freq_span_px,
                                              config.grouping_min_time_span_px,
                                              config.grouping_min_density,
                                              config.grouping_time_continuity_ratio);
  result.final_mask = summary.final_mask;
  result.grouped_box_count = summary.grouped_box_count;
  result.merged_threshold = summary.merged_threshold;
  result.seed_threshold = summary.seed_threshold;
  return result;
}

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
    cudaFreeHost(buffers.power_db_host);
    cudaFreeHost(buffers.mask_host);

    buffers = ChannelBuffers {};
  }
}

void CoherentPowerSignalDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<coherent_power_in_t>("in");

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(input_height_, "input_height", "Input height", "Detector output height.", 256);
  spec.param(input_width_, "input_width", "Input width", "Detector output width.", 512);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one output every N input frames per channel.", 1);
  spec.param(channel_filter_, "channel_filter", "Channel filter", "If non-negative, only process frames for this channel number.", -1);
  spec.param(log_detections_, "log_detections", "Log detections", "If true, logs detector execution details.", false);
  spec.param(backend_mode_, "backend_mode", "Backend mode", "Detector backend mode: auto, fast_low_fidelity_mode, or reference.", std::string("auto"));
  spec.param(enable_mask_save_, "enable_mask_save", "Enable mask save", "Enable writing detector masks to disk for debug runs.", false);
  spec.param(enable_tensor_snapshot_save_, "enable_tensor_snapshot_save", "Enable tensor snapshot save", "Enable writing frozen detector input snapshots for offline parity runs.", false);
  spec.param(save_every_n_frames_, "save_every_n_frames", "Save stride", "Save one detector mask every N frames per channel.", 1);
  spec.param(max_masks_per_channel_, "max_masks_per_channel", "Max masks per channel", "Maximum number of detector masks to save per channel for a run.", 5);
  spec.param(max_snapshots_per_channel_, "max_snapshots_per_channel", "Max snapshots per channel", "Maximum number of frozen detector input snapshots to save per channel for a run.", 2);
  spec.param(output_dir_, "output_dir", "Output directory", "Directory where detector masks are written.", std::string("/workspace/coherent_power_masks"));
  spec.param(tensor_snapshot_dir_, "tensor_snapshot_dir", "Tensor snapshot directory", "Directory where frozen detector input snapshots are written.", std::string("/workspace/coherent_power_snapshots"));
  spec.param(save_power_db_snapshot_, "save_power_db_snapshot", "Save power dB snapshot", "If true, also saves the post power_db frame alongside the complex tensor snapshot.", true);
  spec.param(chunk_bandwidth_hz_, "chunk_bandwidth_hz", "Chunk bandwidth", "Chunk bandwidth in Hz.", 25.0e6);
  spec.param(chunk_overlap_hz_, "chunk_overlap_hz", "Chunk overlap", "Chunk overlap in Hz.", 6.25e6);
  spec.param(uncalibrated_chunk_fraction_, "uncalibrated_chunk_fraction", "Uncalibrated chunk fraction", "Fractional chunk span for uncalibrated inputs.", 0.40);
  spec.param(uncalibrated_overlap_fraction_, "uncalibrated_overlap_fraction", "Uncalibrated overlap fraction", "Fractional chunk overlap for uncalibrated inputs.", 0.20);
  spec.param(ignore_sideband_percent_, "ignore_sideband_percent", "Ignore sideband percent", "Fraction of band edges to ignore when not frequency calibrated.", 0.0);
  spec.param(ignore_sideband_hz_, "ignore_sideband_hz", "Ignore sideband Hz", "Frequency span to ignore on each side of the band.", 7.0e6);
  spec.param(frontend_row_q_, "frontend_row_q", "Frontend row quantile", "Notebook-derived frontend row quantile.", 25.0);
  spec.param(frontend_reference_q_, "frontend_reference_q", "Frontend reference quantile", "Notebook-derived frontend reference quantile.", 75.0);
  spec.param(frontend_smooth_sigma_, "frontend_smooth_sigma", "Frontend smoothing sigma", "Notebook-derived frontend smoothing sigma.", 12.0);
  spec.param(frontend_max_boost_db_, "frontend_max_boost_db", "Frontend max boost", "Notebook-derived frontend max boost in dB.", 12.0);
  spec.param(coherence_weight_, "coherence_weight", "Coherence weight", "Notebook-derived coherence score weight.", 0.55);
  spec.param(power_weight_, "power_weight", "Power weight", "Notebook-derived power score weight.", 0.45);
  spec.param(power_assist_mode_, "power_assist_mode", "Power assist mode", "Notebook-derived power assist mode: hybrid, local_relative, or absolute_floor.", std::string("hybrid"));
  spec.param(power_floor_time_q_, "power_floor_time_q", "Power floor time quantile", "Notebook-derived per-row noise floor quantile.", 25.0);
  spec.param(power_floor_global_q_, "power_floor_global_q", "Power floor global quantile", "Notebook-derived global noise floor quantile.", 30.0);
  spec.param(power_excess_start_db_, "power_excess_start_db", "Power excess start dB", "Notebook-derived absolute power assist ramp start above floor.", 3.0);
  spec.param(power_excess_full_db_, "power_excess_full_db", "Power excess full dB", "Notebook-derived absolute power assist full-scale point above floor.", 15.0);
  spec.param(power_local_blend_, "power_local_blend", "Power local blend", "Notebook-derived hybrid blend between absolute and local-relative power assist.", 0.25);
  spec.param(coherence_gate_start_, "coherence_gate_start", "Coherence gate start", "Notebook-derived coherence gate ramp start.", 0.15);
  spec.param(coherence_gate_full_, "coherence_gate_full", "Coherence gate full", "Notebook-derived coherence gate full-scale point.", 0.45);
  spec.param(coherence_bridge_bias_, "coherence_bridge_bias", "Coherence bridge bias", "Notebook-derived power bridging bias under partial coherence.", 0.05);
  spec.param(coherence_power_joint_weight_, "coherence_power_joint_weight", "Coherence-power joint weight", "Notebook-derived weight between joint and bridged power scores.", 0.70);
  spec.param(coherence_power_support_q_, "coherence_power_support_q", "Support quantile", "Notebook-derived support quantile.", 0.82);
  spec.param(coherence_power_q_, "coherence_power_q", "Final quantile", "Notebook-derived final score quantile.", 0.92);
  spec.param(min_component_size_, "min_component_size", "Minimum component size", "Notebook-derived minimum component size.", 6);
  spec.param(filter_detection_mask_, "filter_detection_mask", "Filter detection mask", "If true, apply bridging and component filtering before boxing. If false, box raw connected mask regions directly.", true);
  spec.param(fast_power_floor_db_, "fast_power_floor_db", "Fast path power floor", "Support floor in dB for the fast GPU detector path.", 1.5);
  spec.param(fast_power_span_db_, "fast_power_span_db", "Fast path power span", "Support normalization span in dB for the fast GPU detector path.", 8.0);
  spec.param(fast_coherence_floor_db_, "fast_coherence_floor_db", "Fast path coherence floor", "Coherence floor in dB for the fast GPU detector path.", 0.4);
  spec.param(fast_coherence_span_db_, "fast_coherence_span_db", "Fast path coherence span", "Coherence normalization span in dB for the fast GPU detector path.", 3.0);
  spec.param(fast_score_threshold_, "fast_score_threshold", "Fast path score threshold", "Score threshold for the fast GPU detector path.", 0.58);
  spec.param(fast_time_smooth_radius_, "fast_time_smooth_radius", "Fast path time radius", "Time-axis radius for the fast GPU coherence proxy.", 4);
  spec.param(fast_freq_smooth_radius_, "fast_freq_smooth_radius", "Fast path frequency radius", "Frequency-axis radius for the fast GPU coherence proxy.", 3);
  spec.param(fast_background_freq_radius_, "fast_background_freq_radius", "Fast path background frequency radius", "Frequency-axis radius for the fast GPU local background.", 8);
  spec.param(fast_background_time_radius_, "fast_background_time_radius", "Fast path background time radius", "Time-axis radius for the fast GPU local background.", 10);
  spec.param(fast_mask_smooth_iterations_, "fast_mask_smooth_iterations", "Fast path mask smoothing", "Number of 3x3 majority-filter iterations for the fast GPU mask.", 1);
  spec.param(grouping_seed_score_q_, "grouping_seed_score_q", "Grouping seed score quantile", "Notebook-derived grouping seed quantile.", 0.72);
  spec.param(grouping_bridge_freq_px_, "grouping_bridge_freq_px", "Grouping bridge frequency", "Notebook-derived grouping bridge size in frequency bins.", 33);
  spec.param(grouping_bridge_time_px_, "grouping_bridge_time_px", "Grouping bridge time", "Notebook-derived grouping bridge size in time bins.", 5);
  spec.param(grouping_min_component_size_, "grouping_min_component_size", "Grouping minimum component size", "Notebook-derived grouping minimum component size.", 24);
  spec.param(grouping_min_freq_span_px_, "grouping_min_freq_span_px", "Grouping minimum frequency span", "Notebook-derived grouping minimum frequency span.", 18);
  spec.param(grouping_min_time_span_px_, "grouping_min_time_span_px", "Grouping minimum time span", "Notebook-derived grouping minimum time span.", 2);
  spec.param(grouping_min_density_, "grouping_min_density", "Grouping minimum density", "Notebook-derived grouping minimum density.", 0.06);
  spec.param(grouping_time_continuity_ratio_, "grouping_time_continuity_ratio", "Grouping time continuity ratio", "Notebook-derived grouping time continuity ratio.", 0.85);
  spec.param(timing_summary_enable_, "timing_summary_enable", "Timing summary enable", "Enable per-stage timing summaries.", true);
  spec.param(timing_summary_every_n_, "timing_summary_every_n", "Timing summary every N", "Emit timing summaries every N emitted frames per channel.", 16);
  spec.param(timing_summary_window_, "timing_summary_window", "Timing summary window", "Maximum number of emitted frames to accumulate before reset.", 16);
}

void CoherentPowerSignalDetector::initialize() {
  holoscan::Operator::initialize();

  frame_count_.assign(num_channels_.get(), 0);
  masks_saved_.assign(num_channels_.get(), 0);
  snapshots_saved_.assign(num_channels_.get(), 0);
  timing_stats_.assign(num_channels_.get(), ChannelTimingStats {});
  channel_buffers_.assign(num_channels_.get(), ChannelBuffers {});

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

  const auto backend_mode = backend_mode_.get();
  const bool reference_only_mode = backend_mode == "reference";

  for (auto& buffers : channel_buffers_) {
    allocate_device_float(buffers.power_db_device, configured_elements);
    allocate_device_float(buffers.corrected_db_device, configured_elements);
    allocate_device_float(buffers.time_mean_device, configured_elements);
    allocate_device_float(buffers.freq_mean_device, configured_elements);
    allocate_device_float(buffers.background_device, configured_elements);
    allocate_device_float(buffers.box_filter_scratch_device, configured_elements);
    allocate_device_float(buffers.score_device, configured_elements);
    if (!reference_only_mode) {
      allocate_device_float(buffers.row_stat_device, static_cast<size_t>(configured_rows));
      allocate_device_float(buffers.row_smooth_device, static_cast<size_t>(configured_rows));
      allocate_device_float(buffers.frontend_reference_device, 1);
    }
    allocate_device_u8(buffers.mask_device, configured_elements);
    allocate_device_u8(buffers.scratch_mask_device, configured_elements);
    const auto analysis_tensor_result = cudaMalloc(reinterpret_cast<void**>(&buffers.analysis_tensor_device),
                                                   configured_elements * sizeof(coherent_power_complex));
    if (analysis_tensor_result != cudaSuccess) {
      throw std::runtime_error(std::string("analysis tensor buffer allocation failed: ") + cudaGetErrorString(analysis_tensor_result));
    }

    buffers.frame_elements = configured_elements;
    buffers.row_elements = reference_only_mode ? 0 : static_cast<size_t>(configured_rows);
    buffers.mask_elements = configured_elements;

    if (enable_mask_save_.get()) {
      const auto host_mask_result = cudaMallocHost(reinterpret_cast<void**>(&buffers.mask_host), configured_elements * sizeof(uint8_t));
      if (host_mask_result != cudaSuccess) {
        throw std::runtime_error(std::string("mask host buffer allocation failed: ") + cudaGetErrorString(host_mask_result));
      }
    }
  }

  if (!channel_buffers_.empty() && backend_mode != "reference") {
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
                                                               std::max(1, fast_time_smooth_radius_.get()),
                                                               buffers.time_mean_device);
      coherent_power_box_mean_rows_kernel<<<blocks, threads>>>(buffers.corrected_db_device,
                                                               configured_rows,
                                                               configured_cols,
                                                               std::max(1, fast_freq_smooth_radius_.get()),
                                                               buffers.freq_mean_device);
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
      coherent_power_fast_score_kernel<<<blocks, threads>>>(buffers.corrected_db_device,
                                                            buffers.time_mean_device,
                                                            buffers.freq_mean_device,
                                                            buffers.background_device,
                                                            configured_rows,
                                                            configured_cols,
                                                            0,
                                                            static_cast<float>(coherence_weight_.get()),
                                                            static_cast<float>(power_weight_.get()),
                                                            static_cast<float>(fast_power_floor_db_.get()),
                                                            static_cast<float>(fast_power_span_db_.get()),
                                                            static_cast<float>(fast_coherence_floor_db_.get()),
                                                            static_cast<float>(fast_coherence_span_db_.get()),
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
}

void CoherentPowerSignalDetector::compute(holoscan::InputContext& op_input,
                                          holoscan::OutputContext&,
                                          holoscan::ExecutionContext&) {
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

  const uint64_t frame_number = ++frame_count_[channel_number];
  const int emit_stride = std::max(1, emit_stride_.get());
  if ((frame_number % static_cast<uint64_t>(emit_stride)) != 0) {
    return;
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

  const auto backend_mode = backend_mode_.get();
  const bool backend_is_reference = backend_mode == "reference";
  const bool backend_is_low_fidelity = backend_mode == "fast_low_fidelity_mode";
  const bool backend_is_legacy_fast = backend_mode == "fast_gpu";
  const bool backend_is_auto = backend_mode == "auto";
  if (backend_is_legacy_fast) {
    HOLOSCAN_LOG_WARN("coherent backend_mode='fast_gpu' is deprecated; use 'fast_low_fidelity_mode' instead.");
  } else if (!backend_is_reference && !backend_is_low_fidelity && !backend_is_auto) {
    HOLOSCAN_LOG_WARN("Unsupported coherent backend_mode='{}'. Falling back to auto.", backend_mode);
  }
  const bool save_requested = enable_mask_save_.get();
  const bool require_reference_dimensions = src_rows != configured_analysis_rows || src_cols != configured_analysis_cols;
  const bool use_reference_backend = backend_is_reference ||
                                     (!(backend_is_low_fidelity || backend_is_legacy_fast) && (save_requested || require_reference_dimensions));
  const int output_rows = use_reference_backend ? src_rows : configured_analysis_rows;
  const int output_cols = use_reference_backend ? src_cols : configured_analysis_cols;
  const bool should_save_mask = save_requested &&
                                (frame_number % static_cast<uint64_t>(std::max(1, save_every_n_frames_.get())) == 0) &&
                                (masks_saved_[channel_number] < max_masks_per_channel_.get());
  const bool should_save_tensor_snapshot = enable_tensor_snapshot_save_.get() &&
                                           (frame_number % static_cast<uint64_t>(std::max(1, save_every_n_frames_.get())) == 0) &&
                                           (snapshots_saved_[channel_number] < max_snapshots_per_channel_.get());
  const bool should_save_power_db_snapshot = should_save_tensor_snapshot && save_power_db_snapshot_.get();
  const bool frequency_axis_calibrated = resolution_hz > 0.0;

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
      const auto row_stat_result = cudaMalloc(reinterpret_cast<void**>(&buffers.row_stat_device), static_cast<size_t>(src_rows) * sizeof(float));
      if (row_stat_result != cudaSuccess) {
        throw std::runtime_error(std::string("row_stat buffer allocation failed: ") + cudaGetErrorString(row_stat_result));
      }
      const auto row_smooth_result = cudaMalloc(reinterpret_cast<void**>(&buffers.row_smooth_device), static_cast<size_t>(src_rows) * sizeof(float));
      if (row_smooth_result != cudaSuccess) {
        throw std::runtime_error(std::string("row_smooth buffer allocation failed: ") + cudaGetErrorString(row_smooth_result));
      }
      buffers.row_elements = static_cast<size_t>(src_rows);
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

  PipelineSummary pipeline_summary;
  FastGpuMetadataSummary fast_summary;
  std::string coherent_backend_name = use_reference_backend ? "coherent_power_reference_v1" : "coherent_power_fast_low_fidelity_v1";
  std::string coherent_variant_name = use_reference_backend ? "frontend_chunked_grouped_box_mask_v1" : "frontend_local_fast_low_fidelity_mask_v1";
  time_step_ms(kPipelineStage, [&] {
    if (use_reference_backend) {
      if (buffers.power_db_host == nullptr) {
        const auto alloc_result = cudaMallocHost(reinterpret_cast<void**>(&buffers.power_db_host), power_db_bytes);
        if (alloc_result != cudaSuccess) {
          throw std::runtime_error(std::string("reference corrected_db host buffer allocation failed: ") + cudaGetErrorString(alloc_result));
        }
      }

      const auto frontend_start = std::chrono::steady_clock::now();
      constexpr int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      coherent_power_row_mean_kernel<<<src_rows, threads, 0, stream>>>(buffers.power_db_device,
                                                                        src_rows,
                                                                        src_cols,
                                                                        buffers.row_stat_device);
      auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("reference row_mean kernel launch failed: ") + cudaGetErrorString(kernel_result));
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
        throw std::runtime_error(std::string("reference row_smooth kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      coherent_power_frontend_reference_kernel<<<1, threads, 0, stream>>>(buffers.row_smooth_device,
                                                                           src_rows,
                                                                           static_cast<float>(frontend_reference_q_.get() / 100.0),
                                                                           buffers.frontend_reference_device);
      kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("reference frontend_reference kernel launch failed: ") + cudaGetErrorString(kernel_result));
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
        throw std::runtime_error(std::string("reference frontend_correction kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }

      auto copy_result = cudaMemcpyAsync(buffers.power_db_host,
                                         buffers.corrected_db_device,
                                         power_db_bytes,
                                         cudaMemcpyDeviceToHost,
                                         stream);
      if (copy_result != cudaSuccess) {
        throw std::runtime_error(std::string("corrected_db device-to-host copy failed: ") + cudaGetErrorString(copy_result));
      }
      auto sync_result = cudaStreamSynchronize(stream);
      if (sync_result != cudaSuccess) {
        throw std::runtime_error(std::string("reference frontend synchronization failed: ") + cudaGetErrorString(sync_result));
      }
      const double frontend_stage_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frontend_start).count();
      pipeline_summary = run_reference_pipeline_corrected(buffers.power_db_host,
                                                          src_rows,
                                                          src_cols,
                                                          ignore_bins_per_side,
                                                          resolution_hz,
                                                          chunk_bandwidth_hz_.get(),
                                                          chunk_overlap_hz_.get(),
                                                          uncalibrated_chunk_fraction_.get(),
                                                          uncalibrated_overlap_fraction_.get(),
                                                          ignore_sideband_percent_.get(),
                                                          coherence_weight_.get(),
                                                          power_weight_.get(),
                                                          power_assist_mode_.get(),
                                                          power_floor_time_q_.get(),
                                                          power_floor_global_q_.get(),
                                                          power_excess_start_db_.get(),
                                                          power_excess_full_db_.get(),
                                                          power_local_blend_.get(),
                                                          coherence_gate_start_.get(),
                                                          coherence_gate_full_.get(),
                                                          coherence_bridge_bias_.get(),
                                                          coherence_power_joint_weight_.get(),
                                                          coherence_power_support_q_.get(),
                                                          coherence_power_q_.get(),
                                                          min_component_size_.get(),
                                                          filter_detection_mask_.get(),
                                                          grouping_seed_score_q_.get(),
                                                          grouping_bridge_freq_px_.get(),
                                                          grouping_bridge_time_px_.get(),
                                                          grouping_min_component_size_.get(),
                                                          grouping_min_freq_span_px_.get(),
                                                          grouping_min_time_span_px_.get(),
                                                          grouping_min_density_.get(),
                                                          grouping_time_continuity_ratio_.get(),
                                                          frontend_stage_ms);
      return;
    }

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

    coherent_power_box_mean_cols_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                         src_rows,
                                                                         src_cols,
                                                                         std::max(1, fast_time_smooth_radius_.get()),
                                                                         buffers.time_mean_device);
    coherent_power_box_mean_rows_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                         src_rows,
                                                                         src_cols,
                                                                         std::max(1, fast_freq_smooth_radius_.get()),
                                                                         buffers.freq_mean_device);
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

    coherent_power_fast_score_kernel<<<blocks, threads, 0, stream>>>(buffers.corrected_db_device,
                                                                      buffers.time_mean_device,
                                                                      buffers.freq_mean_device,
                                                                      buffers.background_device,
                                                                      src_rows,
                                                                      src_cols,
                                                                      ignore_bins_per_side,
                                                                      static_cast<float>(coherence_weight_.get()),
                                                                      static_cast<float>(power_weight_.get()),
                                                                      static_cast<float>(fast_power_floor_db_.get()),
                                                                      static_cast<float>(fast_power_span_db_.get()),
                                                                      static_cast<float>(fast_coherence_floor_db_.get()),
                                                                      static_cast<float>(fast_coherence_span_db_.get()),
                                                                      static_cast<float>(fast_score_threshold_.get()),
                                                                      buffers.score_device,
                                                                      buffers.mask_device);
    kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("fast_score kernel launch failed: ") + cudaGetErrorString(kernel_result));
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

    fast_summary.ignore_bins_per_side = ignore_bins_per_side;
    fast_summary.merged_threshold = static_cast<float>(fast_score_threshold_.get());
    fast_summary.seed_threshold = static_cast<float>(fast_score_threshold_.get());
  });

  stage_ms[kDeviceCopyStage] = 0.0;

  auto maybe_save_debug_artifacts = [&] {
    std::string mask_path;
    if (should_save_mask) {
      std::vector<uint8_t> image;
      image.reserve(static_cast<size_t>(output_rows) * static_cast<size_t>(output_cols));
      if (use_reference_backend) {
        image.assign(pipeline_summary.final_mask.size(), 0);
        for (size_t idx = 0; idx < pipeline_summary.final_mask.size(); ++idx) {
          image[idx] = pipeline_summary.final_mask[idx] > 0.5f ? 255 : 0;
        }
      } else {
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
      return;
    }

    if (should_save_power_db_snapshot && !use_reference_backend) {
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

    if (!write_npy_2d(tensor_path,
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
    meta_out << "  \"backend_mode_requested\": \"" << json_escape(backend_mode) << "\",\n";
    meta_out << "  \"backend_mode_effective\": \"" << json_escape(coherent_backend_name) << "\",\n";
    meta_out << "  \"pipeline_variant\": \"" << json_escape(coherent_variant_name) << "\",\n";
    meta_out << "  \"tensor_snapshot_path\": \"" << json_escape(tensor_path) << "\",\n";
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
    meta_out << "  \"config\": {\n";
    meta_out << "    \"chunk_bandwidth_hz\": " << chunk_bandwidth_hz_.get() << ",\n";
    meta_out << "    \"chunk_overlap_hz\": " << chunk_overlap_hz_.get() << ",\n";
    meta_out << "    \"uncalibrated_chunk_fraction\": " << uncalibrated_chunk_fraction_.get() << ",\n";
    meta_out << "    \"uncalibrated_overlap_fraction\": " << uncalibrated_overlap_fraction_.get() << ",\n";
    meta_out << "    \"ignore_sideband_percent\": " << ignore_sideband_percent_.get() << ",\n";
    meta_out << "    \"ignore_sideband_hz\": " << ignore_sideband_hz_.get() << ",\n";
    meta_out << "    \"frontend_row_q\": " << frontend_row_q_.get() << ",\n";
    meta_out << "    \"frontend_reference_q\": " << frontend_reference_q_.get() << ",\n";
    meta_out << "    \"frontend_smooth_sigma\": " << frontend_smooth_sigma_.get() << ",\n";
    meta_out << "    \"frontend_max_boost_db\": " << frontend_max_boost_db_.get() << ",\n";
    meta_out << "    \"coherence_weight\": " << coherence_weight_.get() << ",\n";
    meta_out << "    \"power_weight\": " << power_weight_.get() << ",\n";
    meta_out << "    \"power_assist_mode\": \"" << power_assist_mode_.get() << "\",\n";
    meta_out << "    \"power_floor_time_q\": " << power_floor_time_q_.get() << ",\n";
    meta_out << "    \"power_floor_global_q\": " << power_floor_global_q_.get() << ",\n";
    meta_out << "    \"power_excess_start_db\": " << power_excess_start_db_.get() << ",\n";
    meta_out << "    \"power_excess_full_db\": " << power_excess_full_db_.get() << ",\n";
    meta_out << "    \"power_local_blend\": " << power_local_blend_.get() << ",\n";
    meta_out << "    \"coherence_gate_start\": " << coherence_gate_start_.get() << ",\n";
    meta_out << "    \"coherence_gate_full\": " << coherence_gate_full_.get() << ",\n";
    meta_out << "    \"coherence_bridge_bias\": " << coherence_bridge_bias_.get() << ",\n";
    meta_out << "    \"coherence_power_joint_weight\": " << coherence_power_joint_weight_.get() << ",\n";
    meta_out << "    \"coherence_power_support_q\": " << coherence_power_support_q_.get() << ",\n";
    meta_out << "    \"coherence_power_q\": " << coherence_power_q_.get() << ",\n";
    meta_out << "    \"min_component_size\": " << min_component_size_.get() << ",\n";
    meta_out << "    \"grouping_seed_score_q\": " << grouping_seed_score_q_.get() << ",\n";
    meta_out << "    \"grouping_bridge_freq_px\": " << grouping_bridge_freq_px_.get() << ",\n";
    meta_out << "    \"grouping_bridge_time_px\": " << grouping_bridge_time_px_.get() << ",\n";
    meta_out << "    \"grouping_min_component_size\": " << grouping_min_component_size_.get() << ",\n";
    meta_out << "    \"grouping_min_freq_span_px\": " << grouping_min_freq_span_px_.get() << ",\n";
    meta_out << "    \"grouping_min_time_span_px\": " << grouping_min_time_span_px_.get() << ",\n";
    meta_out << "    \"grouping_min_density\": " << grouping_min_density_.get() << ",\n";
    meta_out << "    \"grouping_time_continuity_ratio\": " << grouping_time_continuity_ratio_.get() << "\n";
    meta_out << "  }\n";
    meta_out << "}\n";
    if (!meta_out.good()) {
      throw std::runtime_error("failed to write coherent snapshot metadata sidecar");
    }

    ++snapshots_saved_[channel_number];
    if (log_detections_.get()) {
      HOLOSCAN_LOG_INFO("Saved coherent tensor snapshot for channel {} frame {} to {}",
                        channel_number,
                        frame_number,
                        tensor_path);
    }
  };

  time_step_ms(kMaskSaveStage, maybe_save_debug_artifacts);

  stage_ms[kTotalStage] =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();

  meta->set("coherent_frame_number", frame_number);
  meta->set("coherent_mask_height", static_cast<uint32_t>(output_rows));
  meta->set("coherent_mask_width", static_cast<uint32_t>(output_cols));
  meta->set("coherent_backend", coherent_backend_name);
  meta->set("coherent_chunk_count", static_cast<uint32_t>(use_reference_backend ? pipeline_summary.subsection_count : fast_summary.subsection_count));
  meta->set("coherent_grouped_box_count", static_cast<uint32_t>(use_reference_backend ? pipeline_summary.grouped_box_count : fast_summary.grouped_box_count));
  meta->set("coherent_ignore_bins_per_side", use_reference_backend ? pipeline_summary.ignore_bins_per_side : fast_summary.ignore_bins_per_side);
  meta->set("coherent_merged_threshold", use_reference_backend ? pipeline_summary.merged_threshold : fast_summary.merged_threshold);
  meta->set("coherent_seed_threshold", use_reference_backend ? pipeline_summary.seed_threshold : fast_summary.seed_threshold);
  meta->set("coherent_pipeline_variant", coherent_variant_name);
  meta->set("coherent_timing_total_ms", stage_ms[kTotalStage]);

  if (!timing_enabled) {
    return;
  }

  auto& stats = timing_stats_[channel_number];
  ++stats.window_frames;
  for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
    stats.total_ms[stage_index] += stage_ms[stage_index];
    stats.max_ms[stage_index] = std::max(stats.max_ms[stage_index], stage_ms[stage_index]);
  }
  if (use_reference_backend) {
    for (size_t stage_index = 0; stage_index < kReferenceTimingStageCount; ++stage_index) {
      stats.reference_total_ms[stage_index] += pipeline_summary.reference_stage_ms[stage_index];
      stats.reference_max_ms[stage_index] = std::max(stats.reference_max_ms[stage_index], pipeline_summary.reference_stage_ms[stage_index]);
    }
    for (size_t stage_index = 0; stage_index < kChunkTimingStageCount; ++stage_index) {
      stats.chunk_stage_sum_total_ms[stage_index] += pipeline_summary.chunk_stage_sum_ms[stage_index];
      stats.chunk_stage_sum_max_ms[stage_index] = std::max(stats.chunk_stage_sum_max_ms[stage_index], pipeline_summary.chunk_stage_sum_ms[stage_index]);
      stats.chunk_stage_peak_total_ms[stage_index] += pipeline_summary.chunk_stage_peak_ms[stage_index];
      stats.chunk_stage_peak_max_ms[stage_index] = std::max(stats.chunk_stage_peak_max_ms[stage_index], pipeline_summary.chunk_stage_peak_ms[stage_index]);
    }
    for (size_t stage_index = 0; stage_index < kPowerSupportTimingStageCount; ++stage_index) {
      stats.power_support_stage_total_ms[stage_index] += pipeline_summary.power_support_stage_sum_ms[stage_index];
      stats.power_support_stage_max_ms[stage_index] = std::max(stats.power_support_stage_max_ms[stage_index], pipeline_summary.power_support_stage_peak_ms[stage_index]);
    }
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
      << " frames=" << stats.window_frames;
  for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
    const double mean_ms = stats.total_ms[stage_index] * inv_frames;
    oss << ' ' << kTimingStageNames[stage_index] << "_mean=" << mean_ms
        << ' ' << kTimingStageNames[stage_index] << "_max=" << stats.max_ms[stage_index];
  }
  if (use_reference_backend) {
    for (size_t stage_index = 0; stage_index < kReferenceTimingStageCount; ++stage_index) {
      const double mean_ms = stats.reference_total_ms[stage_index] * inv_frames;
      oss << ' ' << kReferenceTimingStageNames[stage_index] << "_mean=" << mean_ms
          << ' ' << kReferenceTimingStageNames[stage_index] << "_max=" << stats.reference_max_ms[stage_index];
    }
    for (size_t stage_index = 0; stage_index < kChunkTimingStageCount; ++stage_index) {
      const double sum_mean_ms = stats.chunk_stage_sum_total_ms[stage_index] * inv_frames;
      const double peak_mean_ms = stats.chunk_stage_peak_total_ms[stage_index] * inv_frames;
      oss << ' ' << kChunkTimingStageNames[stage_index] << "_sum_mean=" << sum_mean_ms
          << ' ' << kChunkTimingStageNames[stage_index] << "_sum_max=" << stats.chunk_stage_sum_max_ms[stage_index]
          << ' ' << kChunkTimingStageNames[stage_index] << "_peak_mean=" << peak_mean_ms
          << ' ' << kChunkTimingStageNames[stage_index] << "_peak_max=" << stats.chunk_stage_peak_max_ms[stage_index];
    }
    for (size_t stage_index = 0; stage_index < kPowerSupportTimingStageCount; ++stage_index) {
      const double mean_ms = stats.power_support_stage_total_ms[stage_index] * inv_frames;
      oss << ' ' << kPowerSupportTimingStageNames[stage_index] << "_sum_mean=" << mean_ms
          << ' ' << kPowerSupportTimingStageNames[stage_index] << "_peak_max=" << stats.power_support_stage_max_ms[stage_index];
    }
  }
  HOLOSCAN_LOG_INFO("{}", oss.str());
  stats = ChannelTimingStats {};
}

}  // namespace holoscan::ops
// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "dinov3_torch_runtime.hpp"

#include <c10/cuda/CUDAGuard.h>
#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <future>
#include <mutex>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <charconv>

#include <torch/torch.h>

namespace {

using dino_complex = cuda::std::complex<float>;

struct ValidatorOptions {
  std::filesystem::path tensor_path;
  std::filesystem::path config_path;
  std::optional<std::filesystem::path> live_mask_path;
  std::filesystem::path output_dir;
  int debug_chunk_index = 13;
  bool verbose = false;
};

struct ValidatorConfig {
  int input_height = 256;
  int input_width = 512;
  int patch_size = 16;
  double resolution_hz = 0.0;
  double span_hz = 0.0;
  double chunk_bandwidth_hz = 25.0e6;
  double chunk_overlap_hz = 6.25e6;
  double uncalibrated_chunk_fraction = 0.40;
  double uncalibrated_overlap_fraction = 0.20;
  double ignore_sideband_hz = 7.0e6;
  bool frontend_correction_enable = true;
  double frontend_correction_row_q = 25.0;
  double frontend_correction_smooth_sigma = 12.0;
  double frontend_correction_reference_q = 75.0;
  double frontend_correction_max_boost_db = 12.0;
  double frontend_correction_soft_knee_db = 4.0;
  double frontend_correction_edge_taper_fraction = 0.10;
  double frontend_correction_edge_taper_sigma = 6.0;
  double frontend_correction_edge_target_drop_db = 2.5;
  double dino_coherence_gate_floor = 0.25;
  double dino_coherence_gate_span_db = 3.0;
  double power_q = 0.90;
  int dino_group_k = 8;
  double dino_group_spatial_weight = 0.35;
  double dino_group_score_q = 0.60;
  int min_component_size = 6;
  bool filter_detection_mask = true;
  int grouping_bridge_freq_px = 33;
  int grouping_bridge_time_px = 5;
  int grouping_min_component_size = 24;
  int grouping_min_freq_span_px = 18;
  int grouping_min_time_span_px = 2;
  double grouping_min_density = 0.06;
  double grouping_time_continuity_ratio = 0.85;
  double pipeline_final_threshold = 0.20;
  double pipeline_gap_floor = 0.10;
  double pipeline_power_rescue_floor = 0.10;
  double pipeline_power_rescue_gain = 2.0;
  std::string inference_backend = "torchscript";
  std::string model_script_path = "/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts";
  std::string torchscript_init_mode = "load_cuda_eval";
  std::string torch_dtype = "fp32";
  std::string hybrid_torch_dtype = "fp32";
  bool legacy_fast_gray_preprocess = false;
  int runtime_batch_size = 4;
};

struct NpyArray2D {
  std::string descr;
  int rows = 0;
  int cols = 0;
  std::vector<uint8_t> payload;
};

struct CanonicalTensor {
  int input_rows = 0;
  int input_cols = 0;
  int rows = 0;
  int cols = 0;
  bool transposed = false;
  std::vector<dino_complex> values;
};

struct MaskComparison {
  bool available = false;
  double pixel_agreement = 0.0;
  double intersection_over_union = 0.0;
  double offline_foreground_fraction = 0.0;
  double live_foreground_fraction = 0.0;
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

struct IgnoreSidebandInfo {
  double requested_percent = 0.0;
  double applied_percent = 0.0;
  double requested_hz = 0.0;
  int requested_bins = 0;
  double applied_hz = 0.0;
  int applied_bins = 0;
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

int chunk_row_count(const ChunkPlanEntry& chunk) {
  return std::max(0, chunk.row_stop - chunk.row_start);
}

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
  std::string split_role = "unsplit";
  bool split_applied = false;
  int parent_component_id = -1;
  int source_box_count = 1;
  std::vector<int> source_chunk_indices;
  std::vector<int> parent_component_ids;
};

struct GroupingResult {
  std::vector<uint8_t> seed_mask;
  std::vector<uint8_t> bridged_mask;
  std::vector<int> component_labels;
  std::vector<uint8_t> grouped_mask;
  std::vector<DetectionBox> boxes;
  float peak_score_floor = 0.0f;
};

struct ComponentLabelling {
  std::vector<int> labels;
  std::vector<int> sizes;
};

struct DinoComponentSummaryRow {
  int cluster = 0;
  float size_fraction = 0.0f;
  float support_mean = 0.0f;
  float support_peak = 0.0f;
  float internal_aff = 0.0f;
  float boundary_aff = 0.0f;
  float seed_mean = 0.0f;
  float smoothness = 0.0f;
  float combined_score = 0.0f;
  float size_penalty = 0.0f;
};

struct GroupedDinoPatchResult {
  std::vector<uint8_t> mask_patch;
  std::vector<float> score_patch;
  std::vector<float> seed_norm_map;
  std::vector<float> seed_persistence_map;
  std::vector<float> seed_contrast_map;
  std::vector<uint8_t> active_mask_patch;
  std::vector<uint8_t> label_map_patch;
  std::vector<uint8_t> label_map_pre_smooth_patch;
  std::vector<int> cluster_map;
  std::vector<float> support_map;
  std::vector<float> support_selected_raw_map;
  std::vector<float> cluster_quality_map;
  std::vector<float> selected_support_map;
  float threshold = 0.0f;
};

ComponentLabelling label_components(const std::vector<uint8_t>& mask, int rows, int cols);
std::vector<uint8_t> keep_large_components(const std::vector<uint8_t>& mask,
                                           int rows,
                                           int cols,
                                           int min_size,
                                           int* kept_component_count);
std::vector<uint8_t> binary_closing_rect(const std::vector<uint8_t>& mask,
                                         int rows,
                                         int cols,
                                         int kernel_rows,
                                         int kernel_cols);
std::vector<uint8_t> binary_fill_holes(const std::vector<uint8_t>& mask, int rows, int cols);
float mean_mask_value(const std::vector<uint8_t>& mask);
float connected_fraction(const std::vector<uint8_t>& mask, const std::vector<uint8_t>& valid_mask);
HybridPostprocessResult run_residual_veto_hybrid(const std::vector<float>& hybrid_dino_contrib,
                                                 const std::vector<uint8_t>& valid_mask,
                                                 int rows,
                                                 int cols);
DetectionBox merge_box_cluster(const std::vector<DetectionBox>& cluster);
std::vector<float> suppress_raw_dino_positional_features(const std::vector<float>& patch_features,
                                                         int patch_rows,
                                                         int patch_cols,
                                                         int feature_dim,
                                                         float suppression);

struct ChunkRetryResult {
  int chunk_index = 0;
  int row_start = 0;
  int row_stop = 0;
  int src_rows = 0;
  int src_cols = 0;
  int dst_rows = 0;
  int dst_cols = 0;
  double freq_start_hz = 0.0;
  double freq_stop_hz = 0.0;
  double span_hz = 0.0;
  int ignore_bins_per_side = 0;
  double dino_threshold = 0.0;
  double runtime_final_threshold = 0.0;
  float seed_freq_threshold = 1.0f;
  float seed_res_threshold = 1.0f;
  float combined_threshold = 1.0f;
  float final_fraction = 0.0f;
  float connected_fraction = 0.0f;
  int component_count = 0;
  int grouped_box_count = 0;
  int runtime_input_gray_rows = 0;
  int runtime_input_gray_cols = 0;
  int patch_rows = 0;
  int patch_cols = 0;
  int feature_dim = 0;
  std::vector<float> runtime_input_gray;
  std::vector<float> patch_features;
  std::vector<float> corrected_resized;
  std::vector<float> raw_dino_score_map;
  std::vector<float> raw_dino_score_deweighted_map;
  std::vector<float> dino_score_map;
  std::vector<float> grouped_seed_score_map;
  std::vector<float> grouped_seed_persistence_map;
  std::vector<float> grouped_seed_contrast_map;
  std::vector<float> grouped_support_map_exact;
  std::vector<float> grouped_active_mask_exact;
  std::vector<float> grouped_cluster_labels_exact;
  std::vector<float> grouped_selected_mask_pre_smooth_exact;
  std::vector<float> grouped_selected_mask_exact;
  std::vector<float> grouped_support_selected_raw_exact;
  std::vector<float> grouped_selected_support_map;
  std::vector<float> grouped_cluster_quality_map;
  std::vector<float> coherence_gate;
  std::vector<float> hybrid_contrib;
  std::vector<uint8_t> valid_mask;
  std::vector<uint8_t> final_mask;
  std::vector<uint8_t> final_mask_source;
  std::vector<uint8_t> grouped_mask;
  std::vector<uint8_t> grouped_mask_source;
  std::vector<uint8_t> bridged_mask;
  std::vector<float> combined_score;
  std::vector<DetectionBox> grouped_boxes;
};

struct ChunkGpuWorkspace {
  float* power_chunk_device = nullptr;
  float* corrected_chunk_device = nullptr;
  float* corrected_full_frame_device = nullptr;
  size_t capacity_elements = 0;
  size_t full_frame_capacity_elements = 0;
  cudaStream_t stream = nullptr;

  ChunkGpuWorkspace() {
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
      throw std::runtime_error("failed to create chunk GPU stream");
    }
  }

  ~ChunkGpuWorkspace() {
    if (power_chunk_device != nullptr) {
      cudaFree(power_chunk_device);
    }
    if (corrected_chunk_device != nullptr) {
      cudaFree(corrected_chunk_device);
    }
    if (corrected_full_frame_device != nullptr) {
      cudaFree(corrected_full_frame_device);
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  ChunkGpuWorkspace(const ChunkGpuWorkspace&) = delete;
  ChunkGpuWorkspace& operator=(const ChunkGpuWorkspace&) = delete;

  void ensure_capacity(size_t required_elements) {
    if (required_elements <= capacity_elements) {
      return;
    }
    if (power_chunk_device != nullptr) {
      cudaFree(power_chunk_device);
      power_chunk_device = nullptr;
    }
    if (corrected_chunk_device != nullptr) {
      cudaFree(corrected_chunk_device);
      corrected_chunk_device = nullptr;
    }
    const size_t required_bytes = required_elements * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&power_chunk_device), required_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&corrected_chunk_device), required_bytes) != cudaSuccess) {
      if (power_chunk_device != nullptr) {
        power_chunk_device = nullptr;
      }
      if (corrected_chunk_device != nullptr) {
        cudaFree(corrected_chunk_device);
        corrected_chunk_device = nullptr;
      }
      throw std::runtime_error("failed to allocate reusable chunk GPU buffers for offline DINO validator");
    }
  }

  void ensure_full_frame_capacity(size_t required_elements) {
    if (required_elements <= full_frame_capacity_elements) {
      return;
    }
    if (corrected_full_frame_device != nullptr) {
      cudaFree(corrected_full_frame_device);
      corrected_full_frame_device = nullptr;
    }
    const size_t required_bytes = required_elements * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&corrected_full_frame_device), required_bytes) != cudaSuccess) {
      throw std::runtime_error("failed to allocate full-frame corrected GPU buffer for offline DINO validator");
    }
    full_frame_capacity_elements = required_elements;
  }
};

struct ChunkInferenceArtifacts {
  ChunkRetryResult result;
  std::vector<uint8_t> source_chunk_valid_mask;
  std::vector<float> grouped_dino_score_source;
  std::vector<float> source_chunk_coherence_gate;
  HybridPostprocessResult precomputed_hybrid_result_source;
  bool has_precomputed_hybrid_result = false;
  bool keep_debug_artifacts = false;
};

struct MemorySnapshot {
  size_t rss_kb = 0;
  size_t hwm_kb = 0;
};

struct StageProfileEntry {
  std::string stage;
  std::string scope;
  int chunk_index = -1;
  double elapsed_ms = 0.0;
  size_t rss_before_kb = 0;
  size_t rss_after_kb = 0;
  size_t rss_delta_kb = 0;
  size_t hwm_before_kb = 0;
  size_t hwm_after_kb = 0;
  size_t hwm_delta_kb = 0;
  size_t component_estimated_bytes = 0;
  bool failed = false;
};

struct StageAggregateEntry {
  std::string stage;
  std::string scope;
  int count = 0;
  int failure_count = 0;
  double total_ms = 0.0;
  double mean_ms = 0.0;
  double max_ms = 0.0;
  size_t max_rss_after_kb = 0;
  size_t max_hwm_after_kb = 0;
  size_t max_component_estimated_bytes = 0;
};

std::vector<StageAggregateEntry> aggregate_stage_entries(const std::vector<StageProfileEntry>& entries) {
  struct AggregateState {
    StageAggregateEntry value;
  };

  std::unordered_map<std::string, AggregateState> grouped;
  for (const auto& entry : entries) {
    const std::string key = entry.scope + "\n" + entry.stage;
    auto& state = grouped[key].value;
    if (state.count == 0) {
      state.scope = entry.scope;
      state.stage = entry.stage;
    }
    state.count += 1;
    state.failure_count += entry.failed ? 1 : 0;
    state.total_ms += entry.elapsed_ms;
    state.max_ms = std::max(state.max_ms, entry.elapsed_ms);
    state.max_rss_after_kb = std::max(state.max_rss_after_kb, entry.rss_after_kb);
    state.max_hwm_after_kb = std::max(state.max_hwm_after_kb, entry.hwm_after_kb);
    state.max_component_estimated_bytes = std::max(state.max_component_estimated_bytes,
                                                   entry.component_estimated_bytes);
  }

  std::vector<StageAggregateEntry> result;
  result.reserve(grouped.size());
  for (auto& [_, state] : grouped) {
    if (state.value.count > 0) {
      state.value.mean_ms = state.value.total_ms / static_cast<double>(state.value.count);
    }
    result.push_back(std::move(state.value));
  }
  std::sort(result.begin(), result.end(), [](const StageAggregateEntry& lhs, const StageAggregateEntry& rhs) {
    if (lhs.total_ms != rhs.total_ms) {
      return lhs.total_ms > rhs.total_ms;
    }
    if (lhs.max_hwm_after_kb != rhs.max_hwm_after_kb) {
      return lhs.max_hwm_after_kb > rhs.max_hwm_after_kb;
    }
    return lhs.stage < rhs.stage;
  });
  return result;
}

class StageProfiler {
 public:
  void add_entry(StageProfileEntry entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.push_back(std::move(entry));
  }

  const std::vector<StageProfileEntry>& entries() const {
    return entries_;
  }

  std::vector<StageAggregateEntry> aggregate() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aggregate_stage_entries(entries_);
  }

 private:
  mutable std::mutex mutex_;
  std::vector<StageProfileEntry> entries_;
};

class ScopedStageProfile {
 public:
  ScopedStageProfile(StageProfiler* profiler,
                     std::string stage_name,
                     std::string scope_name,
                     int chunk_index,
                     size_t component_estimated_bytes,
                     bool emit_log)
      : profiler_(profiler),
        stage_name_(std::move(stage_name)),
        scope_name_(std::move(scope_name)),
        chunk_index_(chunk_index),
        component_estimated_bytes_(component_estimated_bytes),
        emit_log_(emit_log),
        start_snapshot_(capture_memory_snapshot()),
        start_time_(std::chrono::steady_clock::now()),
        uncaught_exceptions_on_entry_(std::uncaught_exceptions()) {}

  ~ScopedStageProfile() {
    if (profiler_ == nullptr) {
      return;
    }
    const auto end_snapshot = capture_memory_snapshot();
    const auto end_time = std::chrono::steady_clock::now();
    StageProfileEntry entry;
    entry.stage = stage_name_;
    entry.scope = scope_name_;
    entry.chunk_index = chunk_index_;
    entry.elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    entry.rss_before_kb = start_snapshot_.rss_kb;
    entry.rss_after_kb = end_snapshot.rss_kb;
    entry.rss_delta_kb = end_snapshot.rss_kb >= start_snapshot_.rss_kb
                             ? end_snapshot.rss_kb - start_snapshot_.rss_kb
                             : 0;
    entry.hwm_before_kb = start_snapshot_.hwm_kb;
    entry.hwm_after_kb = end_snapshot.hwm_kb;
    entry.hwm_delta_kb = end_snapshot.hwm_kb >= start_snapshot_.hwm_kb
                             ? end_snapshot.hwm_kb - start_snapshot_.hwm_kb
                             : 0;
    entry.component_estimated_bytes = component_estimated_bytes_;
    entry.failed = std::uncaught_exceptions() > uncaught_exceptions_on_entry_;
    profiler_->add_entry(entry);
    if (emit_log_) {
      std::cerr << "[offline_dino_validator_performance] stage scope=" << entry.scope
                << " name=" << entry.stage;
      if (entry.chunk_index >= 0) {
        std::cerr << " chunk=" << entry.chunk_index;
      }
      std::cerr << " elapsed_ms=" << std::fixed << std::setprecision(3) << entry.elapsed_ms
                << " rss_kb=" << entry.rss_after_kb
                << " rss_delta_kb=" << entry.rss_delta_kb
                << " hwm_kb=" << entry.hwm_after_kb
                << " hwm_delta_kb=" << entry.hwm_delta_kb
                << " component_estimated_bytes=" << entry.component_estimated_bytes
                << " status=" << (entry.failed ? "failed" : "ok")
                << "\n";
    }
  }

 private:
  static MemorySnapshot capture_memory_snapshot();

  StageProfiler* profiler_ = nullptr;
  std::string stage_name_;
  std::string scope_name_;
  int chunk_index_ = -1;
  size_t component_estimated_bytes_ = 0;
  bool emit_log_ = false;
  MemorySnapshot start_snapshot_;
  std::chrono::steady_clock::time_point start_time_;
  int uncaught_exceptions_on_entry_ = 0;
};

void record_timed_stage(StageProfiler* profiler,
                        std::string_view stage_name,
                        std::string_view scope_name,
                        int chunk_index,
                        double elapsed_ms) {
  if (profiler == nullptr || !(std::isfinite(elapsed_ms)) || elapsed_ms <= 0.0) {
    return;
  }
  StageProfileEntry entry;
  entry.stage = std::string(stage_name);
  entry.scope = std::string(scope_name);
  entry.chunk_index = chunk_index;
  entry.elapsed_ms = elapsed_ms;
  profiler->add_entry(std::move(entry));
}

struct GlobalMergedResult {
  std::vector<uint8_t> projected_grouped_mask;
  std::vector<float> projected_grouped_score;
  std::vector<uint8_t> merged_box_mask;
  std::vector<DetectionBox> projected_boxes;
  std::vector<DetectionBox> merged_boxes;
  std::vector<uint8_t> stitched_final_mask;
};

template <typename T>
T clamp_value(T value, T low, T high) {
  return value < low ? low : (value > high ? high : value);
}

std::string_view trim_ascii_whitespace(std::string_view text) {
  size_t begin = 0;
  while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
    ++begin;
  }
  size_t end = text.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
    --end;
  }
  return text.substr(begin, end - begin);
}

int parse_int_or_throw(std::string_view text, std::string_view field_name) {
  const std::string_view trimmed = trim_ascii_whitespace(text);
  if (trimmed.empty()) {
    throw std::runtime_error("missing integer for " + std::string(field_name));
  }
  int value = 0;
  const auto* begin = trimmed.data();
  const auto* end = trimmed.data() + trimmed.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr != end) {
    throw std::runtime_error("invalid integer for " + std::string(field_name) + ": '" + std::string(trimmed) + "'");
  }
  return value;
}

template <typename Fn>
void parallel_for_ranges(size_t item_count, size_t min_items_per_task, Fn&& fn) {
  if (item_count == 0) {
    return;
  }
  const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
  const size_t clamped_min_items = std::max<size_t>(1, min_items_per_task);
  const size_t max_task_count = (item_count + clamped_min_items - 1) / clamped_min_items;
  const size_t task_count = std::min<size_t>(static_cast<size_t>(hw_threads), std::max<size_t>(1, max_task_count));
  if (task_count <= 1) {
    fn(static_cast<size_t>(0), item_count);
    return;
  }

  const size_t block_size = (item_count + task_count - 1) / task_count;
  std::vector<std::future<void>> tasks;
  tasks.reserve(task_count);
  for (size_t task_index = 0; task_index < task_count; ++task_index) {
    const size_t start = task_index * block_size;
    const size_t stop = std::min(item_count, start + block_size);
    if (start >= stop) {
      break;
    }
    tasks.push_back(std::async(std::launch::async, [start, stop, &fn]() {
      fn(start, stop);
    }));
  }
  for (auto& task : tasks) {
    task.get();
  }
}

size_t flat_index(int cols, int row, int col) {
  return static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
}

template <typename T>
std::optional<T> extract_number(const std::string& text, const std::string& key) {
  const std::regex pattern("(^|\\n)\\s*" + key + "\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return std::nullopt;
  }
  std::istringstream stream(match[2].str());
  T value {};
  stream >> value;
  if (!stream.fail()) {
    return value;
  }
  return std::nullopt;
}

std::optional<std::string> extract_yaml_string(const std::string& text, const std::string& key) {
  const std::regex pattern("(^|\\n)\\s*" + key + "\\s*:\\s*\"([^\"]*)\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return std::nullopt;
  }
  return match[2].str();
}

std::optional<bool> extract_bool(const std::string& text, const std::string& key) {
  const std::regex pattern("(^|\\n)\\s*" + key + "\\s*:\\s*(true|false)", std::regex_constants::icase);
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return std::nullopt;
  }
  std::string value = match[2].str();
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return value == "true";
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

bool use_fp16_precision(const std::string& dtype_text) {
  std::string lowered = dtype_text;
  std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return lowered == "fp16" || lowered == "half";
}

ValidatorConfig load_config(const std::filesystem::path& path) {
  const std::string text = read_text_file(path);
  ValidatorConfig config;
  config.input_height = extract_number<int>(text, "input_height").value_or(config.input_height);
  config.input_width = extract_number<int>(text, "input_width").value_or(config.input_width);
  config.patch_size = extract_number<int>(text, "patch_size").value_or(config.patch_size);
  config.resolution_hz = extract_number<double>(text, "resolution").value_or(config.resolution_hz);
  config.span_hz = extract_number<double>(text, "span").value_or(config.span_hz);
  config.chunk_bandwidth_hz = extract_number<double>(text, "chunk_bandwidth_hz").value_or(config.chunk_bandwidth_hz);
  config.chunk_overlap_hz = extract_number<double>(text, "chunk_overlap_hz").value_or(config.chunk_overlap_hz);
  config.uncalibrated_chunk_fraction = extract_number<double>(text, "uncalibrated_chunk_fraction").value_or(config.uncalibrated_chunk_fraction);
  config.uncalibrated_overlap_fraction = extract_number<double>(text, "uncalibrated_overlap_fraction").value_or(config.uncalibrated_overlap_fraction);
  config.ignore_sideband_hz = extract_number<double>(text, "ignore_sideband_hz").value_or(config.ignore_sideband_hz);
  config.frontend_correction_row_q = extract_number<double>(text, "frontend_correction_row_q").value_or(config.frontend_correction_row_q);
  config.frontend_correction_smooth_sigma = extract_number<double>(text, "frontend_correction_smooth_sigma").value_or(config.frontend_correction_smooth_sigma);
  config.frontend_correction_reference_q = extract_number<double>(text, "frontend_correction_reference_q").value_or(config.frontend_correction_reference_q);
  config.frontend_correction_max_boost_db = extract_number<double>(text, "frontend_correction_max_boost_db").value_or(config.frontend_correction_max_boost_db);
  config.frontend_correction_soft_knee_db = extract_number<double>(text, "frontend_correction_soft_knee_db").value_or(config.frontend_correction_soft_knee_db);
  config.frontend_correction_edge_taper_fraction = extract_number<double>(text, "frontend_correction_edge_taper_fraction").value_or(config.frontend_correction_edge_taper_fraction);
  config.frontend_correction_edge_taper_sigma = extract_number<double>(text, "frontend_correction_edge_taper_sigma").value_or(config.frontend_correction_edge_taper_sigma);
  config.frontend_correction_edge_target_drop_db = extract_number<double>(text, "frontend_correction_edge_target_drop_db").value_or(config.frontend_correction_edge_target_drop_db);
  config.dino_coherence_gate_floor = extract_number<double>(text, "dino_coherence_gate_floor").value_or(config.dino_coherence_gate_floor);
  config.dino_coherence_gate_span_db = extract_number<double>(text, "dino_coherence_gate_span_db").value_or(config.dino_coherence_gate_span_db);
  config.power_q = extract_number<double>(text, "power_q").value_or(config.power_q);
  config.dino_group_k = extract_number<int>(text, "dino_group_k").value_or(config.dino_group_k);
  config.dino_group_spatial_weight = extract_number<double>(text, "dino_group_spatial_weight").value_or(config.dino_group_spatial_weight);
  config.dino_group_score_q = extract_number<double>(text, "dino_group_score_q").value_or(config.dino_group_score_q);
  config.min_component_size = extract_number<int>(text, "min_component_size").value_or(config.min_component_size);
  config.filter_detection_mask = extract_bool(text, "filter_detection_mask").value_or(config.filter_detection_mask);
  config.grouping_bridge_freq_px = extract_number<int>(text, "grouping_bridge_freq_px").value_or(config.grouping_bridge_freq_px);
  config.grouping_bridge_time_px = extract_number<int>(text, "grouping_bridge_time_px").value_or(config.grouping_bridge_time_px);
  config.grouping_min_component_size = extract_number<int>(text, "grouping_min_component_size").value_or(config.grouping_min_component_size);
  config.grouping_min_freq_span_px = extract_number<int>(text, "grouping_min_freq_span_px").value_or(config.grouping_min_freq_span_px);
  config.grouping_min_time_span_px = extract_number<int>(text, "grouping_min_time_span_px").value_or(config.grouping_min_time_span_px);
  config.grouping_min_density = extract_number<double>(text, "grouping_min_density").value_or(config.grouping_min_density);
  config.grouping_time_continuity_ratio = extract_number<double>(text, "grouping_time_continuity_ratio").value_or(config.grouping_time_continuity_ratio);
  config.pipeline_final_threshold = extract_number<double>(text, "pipeline_final_threshold").value_or(config.pipeline_final_threshold);
  config.pipeline_gap_floor = extract_number<double>(text, "pipeline_gap_floor").value_or(config.pipeline_gap_floor);
  config.pipeline_power_rescue_floor = extract_number<double>(text, "pipeline_power_rescue_floor").value_or(config.pipeline_power_rescue_floor);
  config.pipeline_power_rescue_gain = extract_number<double>(text, "pipeline_power_rescue_gain").value_or(config.pipeline_power_rescue_gain);
  config.inference_backend = extract_yaml_string(text, "inference_backend").value_or(config.inference_backend);
  config.model_script_path = extract_yaml_string(text, "model_script_path").value_or(config.model_script_path);
  config.torchscript_init_mode = extract_yaml_string(text, "torchscript_init_mode").value_or(config.torchscript_init_mode);
  config.torch_dtype = extract_yaml_string(text, "torch_dtype").value_or(config.torch_dtype);
  config.hybrid_torch_dtype = extract_yaml_string(text, "hybrid_torch_dtype").value_or(config.hybrid_torch_dtype);
  config.legacy_fast_gray_preprocess = extract_bool(text, "legacy_fast_gray_preprocess").value_or(config.legacy_fast_gray_preprocess);
  config.runtime_batch_size = extract_number<int>(text, "runtime_batch_size").value_or(config.runtime_batch_size);
  return config;
}

NpyArray2D load_npy_2d(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open npy file: " + path.string());
  }

  char magic[6] {};
  in.read(magic, 6);
  if (std::string(magic, 6) != std::string("\x93NUMPY", 6)) {
    throw std::runtime_error("unsupported npy magic in: " + path.string());
  }

  unsigned char version[2] {};
  in.read(reinterpret_cast<char*>(version), 2);
  uint32_t header_len = 0;
  if (version[0] == 1) {
    unsigned char bytes[2] {};
    in.read(reinterpret_cast<char*>(bytes), 2);
    header_len = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8U);
  } else {
    unsigned char bytes[4] {};
    in.read(reinterpret_cast<char*>(bytes), 4);
    header_len = static_cast<uint32_t>(bytes[0]) |
                 (static_cast<uint32_t>(bytes[1]) << 8U) |
                 (static_cast<uint32_t>(bytes[2]) << 16U) |
                 (static_cast<uint32_t>(bytes[3]) << 24U);
  }

  std::string header(header_len, '\0');
  in.read(header.data(), static_cast<std::streamsize>(header.size()));

  std::smatch descr_match;
  const std::regex descr_pattern("'descr'\\s*:\\s*'([^']+)'");
  if (!std::regex_search(header, descr_match, descr_pattern)) {
    throw std::runtime_error("npy header missing descr in: " + path.string());
  }

  std::smatch shape_match;
  const std::regex shape_pattern("'shape'\\s*:\\s*\\((\\d+)\\s*,\\s*(\\d+)\\s*\\)");
  if (!std::regex_search(header, shape_match, shape_pattern)) {
    throw std::runtime_error("npy header missing 2D shape in: " + path.string());
  }

  NpyArray2D array;
  array.descr = descr_match[1].str();
  array.rows = parse_int_or_throw(shape_match[1].str(), "npy shape rows in " + path.string());
  array.cols = parse_int_or_throw(shape_match[2].str(), "npy shape cols in " + path.string());
  const size_t element_bytes = array.descr == "<c8" ? sizeof(float) * 2 : sizeof(float);
  const size_t payload_bytes = static_cast<size_t>(array.rows) * static_cast<size_t>(array.cols) * element_bytes;
  array.payload.resize(payload_bytes);
  in.read(reinterpret_cast<char*>(array.payload.data()), static_cast<std::streamsize>(array.payload.size()));
  if (!in) {
    throw std::runtime_error("truncated npy payload in: " + path.string());
  }
  return array;
}

CanonicalTensor load_canonical_tensor(const std::filesystem::path& path) {
  const NpyArray2D array = load_npy_2d(path);
  if (array.descr != "<c8") {
    throw std::runtime_error("expected complex64 tensor snapshot");
  }

  CanonicalTensor tensor;
  tensor.input_rows = array.rows;
  tensor.input_cols = array.cols;
  tensor.transposed = array.rows < array.cols;
  tensor.rows = tensor.transposed ? array.cols : array.rows;
  tensor.cols = tensor.transposed ? array.rows : array.cols;
  tensor.values.assign(static_cast<size_t>(tensor.rows) * static_cast<size_t>(tensor.cols), dino_complex(0.0f, 0.0f));

  for (int row = 0; row < array.rows; ++row) {
    for (int col = 0; col < array.cols; ++col) {
      float parts[2] {};
      const size_t source_index = (static_cast<size_t>(row) * static_cast<size_t>(array.cols) + static_cast<size_t>(col)) * 2U;
      std::memcpy(parts, array.payload.data() + source_index * sizeof(float), sizeof(parts));
      const int dst_row = tensor.transposed ? col : row;
      const int dst_col = tensor.transposed ? row : col;
      tensor.values[flat_index(tensor.cols, dst_row, dst_col)] = dino_complex(parts[0], parts[1]);
    }
  }
  return tensor;
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

std::vector<uint8_t> load_pgm(const std::filesystem::path& path, int& rows, int& cols) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open PGM: " + path.string());
  }
  std::string magic;
  in >> magic;
  if (magic != "P5") {
    throw std::runtime_error("unsupported PGM magic in: " + path.string());
  }
  in >> cols >> rows;
  int max_value = 0;
  in >> max_value;
  in.get();
  if (rows <= 0 || cols <= 0 || max_value <= 0 || max_value > 255) {
    throw std::runtime_error("invalid PGM header in: " + path.string());
  }
  std::vector<uint8_t> image(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  in.read(reinterpret_cast<char*>(image.data()), static_cast<std::streamsize>(image.size()));
  if (!in) {
    throw std::runtime_error("truncated PGM payload in: " + path.string());
  }
  return image;
}

bool write_pgm(const std::filesystem::path& path, const std::vector<uint8_t>& image, int width, int height) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
  return out.good();
}

std::vector<float> power_db_from_tensor(const CanonicalTensor& tensor) {
  std::vector<float> power_db(tensor.values.size(), 0.0f);
  parallel_for_ranges(tensor.values.size(), static_cast<size_t>(1) << 18, [&](size_t start, size_t stop) {
    for (size_t index = start; index < stop; ++index) {
      const float real = tensor.values[index].real();
      const float imag = tensor.values[index].imag();
      power_db[index] = 10.0f * std::log10(real * real + imag * imag + 1.0e-12f);
    }
  });
  return power_db;
}

std::vector<float> gaussian_smooth_rows(const std::vector<float>& input, int rows, int radius, float sigma) {
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

std::vector<float> frontend_corrected_db(const std::vector<float>& power_db,
                                         int rows,
                                         int cols,
                                         const ValidatorConfig& config,
                                         float& reference_level_out) {
  std::vector<float> row_mean(static_cast<size_t>(rows), 0.0f);
  parallel_for_ranges(static_cast<size_t>(rows), 32, [&](size_t start, size_t stop) {
    for (size_t row = start; row < stop; ++row) {
      const float* row_ptr = power_db.data() + row * static_cast<size_t>(cols);
      float sum = 0.0f;
      for (int col = 0; col < cols; ++col) {
        sum += row_ptr[static_cast<size_t>(col)];
      }
      row_mean[row] = sum / static_cast<float>(std::max(cols, 1));
    }
  });
  const float sigma = static_cast<float>(std::max(config.frontend_correction_smooth_sigma, 1.0));
  const int radius = std::max(1, static_cast<int>(std::ceil(sigma * 1.5f)));
  const auto row_smooth = gaussian_smooth_rows(row_mean, rows, radius, sigma);

  float sum = 0.0f;
  float max_value = -1.0e30f;
  for (float value : row_smooth) {
    sum += value;
    max_value = std::max(max_value, value);
  }
  const float mean_value = sum / static_cast<float>(std::max(rows, 1));
  const float quantile = static_cast<float>(config.frontend_correction_reference_q / 100.0);
  const float blend = clamp_value((quantile - 0.5f) / 0.5f, 0.0f, 1.0f);
  reference_level_out = mean_value + blend * (max_value - mean_value);

  std::vector<float> row_boost(static_cast<size_t>(rows), 0.0f);
  for (int row = 0; row < rows; ++row) {
    row_boost[static_cast<size_t>(row)] = std::min(std::max(reference_level_out - row_smooth[static_cast<size_t>(row)], 0.0f),
                                                   static_cast<float>(config.frontend_correction_max_boost_db));
  }

  std::vector<float> corrected(power_db.size(), 0.0f);
  parallel_for_ranges(static_cast<size_t>(rows), 16, [&](size_t start, size_t stop) {
    for (size_t row = start; row < stop; ++row) {
      const float boost = row_boost[row];
      const size_t row_offset = row * static_cast<size_t>(cols);
      const float* src_row = power_db.data() + row_offset;
      float* dst_row = corrected.data() + row_offset;
      if (boost <= 0.0f) {
        std::memcpy(dst_row, src_row, static_cast<size_t>(cols) * sizeof(float));
        continue;
      }
      for (int col = 0; col < cols; ++col) {
        dst_row[static_cast<size_t>(col)] = src_row[static_cast<size_t>(col)] + boost;
      }
    }
  });
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

void copy_rows_device_to_device(const float* src_device,
                                int src_cols,
                                int row_start,
                                int row_count,
                                float* dst_device,
                                cudaStream_t stream) {
  if (src_device == nullptr || dst_device == nullptr || src_cols <= 0 || row_count <= 0) {
    return;
  }
  const size_t row_bytes = static_cast<size_t>(src_cols) * sizeof(float);
  const float* src_row0 = src_device + static_cast<size_t>(row_start) * static_cast<size_t>(src_cols);
  if (cudaMemcpy2DAsync(dst_device,
                        row_bytes,
                        src_row0,
                        row_bytes,
                        row_bytes,
                        static_cast<size_t>(row_count),
                        cudaMemcpyDeviceToDevice,
                        stream) != cudaSuccess) {
    throw std::runtime_error("failed to copy corrected rows on GPU for offline DINO validator");
  }
}

std::vector<uint8_t> resize_row_valid_mask(const std::vector<uint8_t>& src_valid_rows,
                                           int dst_rows,
                                           int dst_cols) {
  std::vector<uint8_t> output(static_cast<size_t>(std::max(dst_rows, 0)) * static_cast<size_t>(std::max(dst_cols, 0)), 0);
  if (src_valid_rows.empty() || dst_rows <= 0 || dst_cols <= 0) {
    return output;
  }
  const int src_rows = static_cast<int>(src_valid_rows.size());
  for (int dst_row = 0; dst_row < dst_rows; ++dst_row) {
    const int src_row = std::min(src_rows - 1,
                                 static_cast<int>((static_cast<int64_t>(dst_row) * static_cast<int64_t>(src_rows)) /
                                                  static_cast<int64_t>(std::max(dst_rows, 1))));
    if (!src_valid_rows[static_cast<size_t>(src_row)]) {
      continue;
    }
    const size_t offset = static_cast<size_t>(dst_row) * static_cast<size_t>(dst_cols);
    std::fill(output.begin() + static_cast<std::ptrdiff_t>(offset),
              output.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(dst_cols)),
              static_cast<uint8_t>(1));
  }
  return output;
}

IgnoreSidebandInfo compute_ignore_sideband_rows(int num_rows,
                                                double bin_hz,
                                                double ignore_sideband_percent,
                                                int min_keep_rows,
                                                std::optional<double> ignore_sideband_hz) {
  IgnoreSidebandInfo info;
  info.valid_row_mask.assign(static_cast<size_t>(std::max(num_rows, 0)), 1);
  info.requested_percent = clamp_value(ignore_sideband_percent, 0.0, 0.49);
  info.requested_hz = std::max(0.0, ignore_sideband_hz.value_or(0.0));
  if (num_rows < 2 || !std::isfinite(bin_hz) || bin_hz <= 0.0) {
    return info;
  }

  info.bin_hz = bin_hz;
  const int max_bins = std::max(0, (num_rows - std::max(1, min_keep_rows)) / 2);
  if (info.requested_percent > 0.0) {
    info.requested_bins = static_cast<int>(std::ceil(static_cast<double>(num_rows) * info.requested_percent));
    info.requested_hz = static_cast<double>(info.requested_bins) * bin_hz;
  } else if (info.requested_hz > 0.0) {
    info.requested_bins = static_cast<int>(std::ceil(info.requested_hz / bin_hz));
  }

  info.applied_bins = clamp_value(info.requested_bins, 0, max_bins);
  info.applied_hz = static_cast<double>(info.applied_bins) * bin_hz;
  info.applied_percent = static_cast<double>(info.applied_bins) / static_cast<double>(std::max(num_rows, 1));
  if (info.applied_bins > 0) {
    std::fill(info.valid_row_mask.begin(), info.valid_row_mask.begin() + info.applied_bins, static_cast<uint8_t>(0));
    std::fill(info.valid_row_mask.end() - info.applied_bins, info.valid_row_mask.end(), static_cast<uint8_t>(0));
  }
  return info;
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
  if (freq_axis_hz.empty()) {
    return chunks;
  }
  if (valid_row_mask.size() != freq_axis_hz.size()) {
    throw std::runtime_error("valid_row_mask length must match frequency axis length");
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
      chunks.push_back(ChunkPlanEntry {
          .chunk_index = 0,
          .row_start = valid_idx.front(),
          .row_stop = valid_idx.back() + 1,
          .freq_start_hz = freq_axis_hz[static_cast<size_t>(valid_idx.front())],
          .freq_stop_hz = freq_axis_hz[static_cast<size_t>(valid_idx.back())],
      });
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
      chunks.push_back(ChunkPlanEntry {
          .chunk_index = chunk_index++,
          .row_start = valid_idx[static_cast<size_t>(start_pos)],
          .row_stop = valid_idx[static_cast<size_t>(stop_pos - 1)] + 1,
          .freq_start_hz = freq_axis_hz[static_cast<size_t>(valid_idx[static_cast<size_t>(start_pos)])],
          .freq_stop_hz = freq_axis_hz[static_cast<size_t>(valid_idx[static_cast<size_t>(stop_pos - 1)])],
      });
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
      chunks.push_back(ChunkPlanEntry {
          .chunk_index = chunk_index++,
          .row_start = row_start,
          .row_stop = row_stop,
          .freq_start_hz = freq_axis_hz[static_cast<size_t>(row_start)],
          .freq_stop_hz = freq_axis_hz[static_cast<size_t>(row_stop - 1)],
      });
    }
    if (chunk_stop_hz >= freq_max) {
      break;
    }
  }
  return chunks;
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

struct UniformChunkGeometry {
  int chunk_rows = 0;
  int overlap_rows = 0;
  int step_rows = 0;
};

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
    chunks.push_back(ChunkPlanEntry {
        .chunk_index = chunk_index++,
        .row_start = row_start,
        .row_stop = row_stop,
        .freq_start_hz = freq_axis_hz[static_cast<size_t>(row_start)],
        .freq_stop_hz = freq_axis_hz[static_cast<size_t>(row_stop - 1)],
    });
  }
  return chunks;
}

struct PlannedIgnoreSidebandSelection {
  IgnoreSidebandInfo ignore_info;
  std::vector<ChunkPlanEntry> chunk_plan;
};

PlannedIgnoreSidebandSelection select_uniform_chunk_plan_with_minimal_sideband_trim(
    int num_rows,
    double bin_hz,
    double ignore_sideband_percent,
    int min_keep_rows,
    std::optional<double> ignore_sideband_hz,
    const std::vector<double>& freq_axis_hz,
    double chunk_bandwidth_hz,
    double chunk_overlap_hz,
    int min_chunk_rows,
    double uncalibrated_chunk_fraction,
    double uncalibrated_overlap_fraction) {
  PlannedIgnoreSidebandSelection selection;
  selection.ignore_info = compute_ignore_sideband_rows(
      num_rows,
      bin_hz,
      ignore_sideband_percent,
      min_keep_rows,
      ignore_sideband_hz);

  const auto calibrated_geometry = calibrated_uniform_chunk_geometry(
      bin_hz,
      chunk_bandwidth_hz,
      chunk_overlap_hz,
      min_chunk_rows);

  if (calibrated_geometry.has_value()) {
    const int max_bins = std::max(0, (num_rows - std::max(1, min_keep_rows)) / 2);
    for (int applied_bins = selection.ignore_info.applied_bins; applied_bins <= max_bins; ++applied_bins) {
      const int valid_count = num_rows - 2 * applied_bins;
      if (valid_count < calibrated_geometry->chunk_rows) {
        break;
      }
      if (((valid_count - calibrated_geometry->chunk_rows) % calibrated_geometry->step_rows) != 0) {
        continue;
      }

      PlannedIgnoreSidebandSelection candidate;
      candidate.ignore_info = selection.ignore_info;
      candidate.ignore_info.applied_bins = applied_bins;
      candidate.ignore_info.applied_hz = static_cast<double>(applied_bins) * candidate.ignore_info.bin_hz;
      candidate.ignore_info.applied_percent = static_cast<double>(applied_bins) / static_cast<double>(std::max(num_rows, 1));
      candidate.ignore_info.valid_row_mask.assign(static_cast<size_t>(std::max(num_rows, 0)), static_cast<uint8_t>(1));
      if (applied_bins > 0) {
        std::fill(candidate.ignore_info.valid_row_mask.begin(),
                  candidate.ignore_info.valid_row_mask.begin() + applied_bins,
                  static_cast<uint8_t>(0));
        std::fill(candidate.ignore_info.valid_row_mask.end() - applied_bins,
                  candidate.ignore_info.valid_row_mask.end(),
                  static_cast<uint8_t>(0));
      }
      candidate.chunk_plan = build_uniform_row_chunks(
          freq_axis_hz,
          candidate.ignore_info.valid_row_mask,
          calibrated_geometry->chunk_rows,
          calibrated_geometry->step_rows,
          min_chunk_rows);
      if (!candidate.chunk_plan.empty() && chunk_plan_has_uniform_rows(candidate.chunk_plan)) {
        return candidate;
      }
    }
  }

  auto build_plan_for_bins = [&](int applied_bins) {
    PlannedIgnoreSidebandSelection candidate;
    candidate.ignore_info = selection.ignore_info;
    candidate.ignore_info.applied_bins = applied_bins;
    candidate.ignore_info.applied_hz = static_cast<double>(applied_bins) * candidate.ignore_info.bin_hz;
    candidate.ignore_info.applied_percent = static_cast<double>(applied_bins) / static_cast<double>(std::max(num_rows, 1));
    candidate.ignore_info.valid_row_mask.assign(static_cast<size_t>(std::max(num_rows, 0)), static_cast<uint8_t>(1));
    if (applied_bins > 0) {
      std::fill(candidate.ignore_info.valid_row_mask.begin(),
                candidate.ignore_info.valid_row_mask.begin() + applied_bins,
                static_cast<uint8_t>(0));
      std::fill(candidate.ignore_info.valid_row_mask.end() - applied_bins,
                candidate.ignore_info.valid_row_mask.end(),
                static_cast<uint8_t>(0));
    }
    candidate.chunk_plan = build_frequency_chunks(
        freq_axis_hz,
        chunk_bandwidth_hz,
        chunk_overlap_hz,
        min_chunk_rows,
        candidate.ignore_info.valid_row_mask,
        uncalibrated_chunk_fraction,
        uncalibrated_overlap_fraction);
    return candidate;
  };

  selection = build_plan_for_bins(selection.ignore_info.applied_bins);
  if (chunk_plan_has_uniform_rows(selection.chunk_plan)) {
    return selection;
  }

  const int target_chunk_count = static_cast<int>(selection.chunk_plan.size());
  const int max_bins = std::max(0, (num_rows - std::max(1, min_keep_rows)) / 2);

  for (int applied_bins = selection.ignore_info.applied_bins + 1; applied_bins <= max_bins; ++applied_bins) {
    auto candidate = build_plan_for_bins(applied_bins);
    if (static_cast<int>(candidate.chunk_plan.size()) != target_chunk_count) {
      continue;
    }
    if (chunk_plan_has_uniform_rows(candidate.chunk_plan)) {
      return candidate;
    }
  }

  for (int applied_bins = selection.ignore_info.applied_bins + 1; applied_bins <= max_bins; ++applied_bins) {
    auto candidate = build_plan_for_bins(applied_bins);
    if (chunk_plan_has_uniform_rows(candidate.chunk_plan)) {
      return candidate;
    }
  }

  return selection;
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

std::vector<float> coherence_gate(const std::vector<float>& corrected,
                                  int rows,
                                  int cols,
                                  int ignore_bins_per_side,
                                  const ValidatorConfig& config) {
  const auto time_mean = box_mean_cols(corrected, rows, cols, 4);
  const auto freq_mean = box_mean_rows(corrected, rows, cols, 3);
  std::vector<float> output(corrected.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t index = flat_index(cols, row, col);
      if (row < ignore_bins_per_side || row >= (rows - ignore_bins_per_side)) {
        output[index] = 0.0f;
        continue;
      }
      const float coherence_db = time_mean[index] - freq_mean[index];
      output[index] = clamp_value((coherence_db - static_cast<float>(config.dino_coherence_gate_floor)) /
                                      std::max(static_cast<float>(config.dino_coherence_gate_span_db), 1.0e-6f),
                                  0.0f,
                                  1.0f);
    }
  }
  return output;
}

std::vector<float> resize_bilinear(const std::vector<float>& input,
                                   int src_rows,
                                   int src_cols,
                                   int dst_rows,
                                   int dst_cols) {
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
  for (int dst_row = 0; dst_row < dst_rows; ++dst_row) {
    const int src_row = std::min(src_rows - 1,
                                 static_cast<int>((static_cast<int64_t>(dst_row) * static_cast<int64_t>(src_rows)) /
                                                  static_cast<int64_t>(std::max(dst_rows, 1))));
    if (src_row < ignore_bins_per_side || src_row >= (src_rows - ignore_bins_per_side)) {
      continue;
    }
    const size_t offset = static_cast<size_t>(dst_row) * static_cast<size_t>(dst_cols);
    std::fill(mask.begin() + static_cast<std::ptrdiff_t>(offset),
              mask.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(dst_cols)),
              static_cast<uint8_t>(1));
  }
  return mask;
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
  const size_t rank_index = static_cast<size_t>(std::llround(q * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(rank_index), values.end());
  return values[rank_index];
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
    kernel[static_cast<size_t>(offset + radius)] = static_cast<float>(((x * x - sigma2) / (sigma2 * sigma2)) * std::exp(-(x * x) / (2.0 * sigma2)));
  }
  return convolve_axis(input, rows, cols, kernel, true);
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
  if (!std::isfinite(low) || !std::isfinite(high) || high <= low + 1.0e-12f) {
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

std::pair<float, float> quantile_bounds_from_input(const std::vector<float>& input,
                                                   double low_q,
                                                   double high_q) {
  std::vector<float> values;
  values.reserve(input.size());
  for (float value : input) {
    if (std::isfinite(value)) {
      values.push_back(value);
    }
  }
  if (values.empty()) {
    return {0.0f, 1.0f};
  }
  const float low = quantile_from_values(values, clamp_value(low_q / 100.0, 0.0, 1.0), 0.0f);
  const float high = quantile_from_values(values, clamp_value(high_q / 100.0, 0.0, 1.0), 1.0f);
  return {low, std::max(high, low + 1.0e-6f)};
}

void normalize01_quantile_into(const std::vector<float>& input,
                               float low,
                               float high,
                               float* output) {
  if (output == nullptr) {
    return;
  }
  const float scale = std::max(high - low, 1.0e-6f);
  for (size_t index = 0; index < input.size(); ++index) {
    const float value = input[index];
    output[index] = std::isfinite(value) ? clamp_value((value - low) / scale, 0.0f, 1.0f) : 0.0f;
  }
}

void multiply_normalized_quantile_into(const std::vector<float>& input,
                                       float low,
                                       float high,
                                       float* output) {
  if (output == nullptr) {
    return;
  }
  const float scale = std::max(high - low, 1.0e-6f);
  for (size_t index = 0; index < input.size(); ++index) {
    const float value = input[index];
    const float normalized = std::isfinite(value) ? clamp_value((value - low) / scale, 0.0f, 1.0f) : 0.0f;
    output[index] *= normalized;
  }
}

std::vector<float> normalize_vector01(const std::vector<float>& input) {
  std::vector<float> output(input.size(), 1.0f);
  if (input.empty()) {
    return output;
  }
  float low = std::numeric_limits<float>::infinity();
  float high = -std::numeric_limits<float>::infinity();
  for (float value : input) {
    if (!std::isfinite(value)) {
      continue;
    }
    low = std::min(low, value);
    high = std::max(high, value);
  }
  if (!std::isfinite(low) || !std::isfinite(high) || high <= low + 1.0e-8f) {
    return output;
  }
  const float scale = high - low;
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = std::isfinite(input[index]) ? (input[index] - low) / scale : 0.0f;
  }
  return output;
}

struct ProcessMemorySnapshot {
  size_t vm_rss_kib = 0;
  size_t vm_hwm_kib = 0;
};

ProcessMemorySnapshot read_process_memory_snapshot() {
  ProcessMemorySnapshot snapshot;
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    auto parse_kib = [&](const char* prefix, size_t* out) {
      const std::string prefix_text(prefix);
      if (line.rfind(prefix_text, 0) != 0) {
        return false;
      }
      std::istringstream input(line.substr(prefix_text.size()));
      size_t value = 0;
      std::string unit;
      input >> value >> unit;
      if (input && unit == "kB") {
        *out = value;
      }
      return true;
    };
    if (parse_kib("VmRSS:", &snapshot.vm_rss_kib)) {
      continue;
    }
    if (parse_kib("VmHWM:", &snapshot.vm_hwm_kib)) {
      continue;
    }
  }
  return snapshot;
}

double dense_square_matrix_mib(int size) {
  if (size <= 0) {
    return 0.0;
  }
  const double bytes = static_cast<double>(size) * static_cast<double>(size) * static_cast<double>(sizeof(float));
  return bytes / (1024.0 * 1024.0);
}

void log_grouped_patch_memory(bool verbose,
                              std::string_view stage,
                              const char* debug_label,
                              int patch_rows,
                              int patch_cols,
                              int feature_dim) {
  if (!verbose) {
    return;
  }
  const int patch_count = std::max(0, patch_rows * patch_cols);
  const auto memory = read_process_memory_snapshot();
  std::cerr << "[offline_dino_validator_performance] grouped_patch_memory stage=" << stage
            << " label=" << debug_label
            << " patch_grid=" << patch_rows << "x" << patch_cols
            << " patch_count=" << patch_count
            << " feature_dim=" << feature_dim
            << " dense_matrix_mib=" << std::fixed << std::setprecision(1) << dense_square_matrix_mib(patch_count)
            << " rss_mib=" << (static_cast<double>(memory.vm_rss_kib) / 1024.0)
            << " hwm_mib=" << (static_cast<double>(memory.vm_hwm_kib) / 1024.0)
            << "\n";
}

torch::Tensor vector_to_tensor_2d(const std::vector<float>& input, int rows, int cols) {
  if (rows <= 0 || cols <= 0 || input.size() != static_cast<size_t>(rows) * static_cast<size_t>(cols)) {
    return torch::zeros({std::max(rows, 0), std::max(cols, 0)}, torch::TensorOptions().dtype(torch::kFloat32));
  }
  return torch::tensor(input, torch::TensorOptions().dtype(torch::kFloat32)).view({rows, cols}).clone();
}

std::vector<float> tensor_to_vector_float(const torch::Tensor& tensor) {
  const auto contiguous = tensor.contiguous().to(torch::kCPU, torch::kFloat32);
  std::vector<float> output(static_cast<size_t>(contiguous.numel()), 0.0f);
  if (!output.empty()) {
    std::memcpy(output.data(), contiguous.data_ptr<float>(), output.size() * sizeof(float));
  }
  return output;
}

torch::Tensor scalar_quantile_tensor_torch(const torch::Tensor& input, double q) {
  auto flat = input.reshape({-1});
  const auto size = flat.size(0);
  if (size <= 1) {
    return flat[0];
  }
  const double clamped = std::clamp(q, 0.0, 1.0);
  const auto rank = static_cast<int64_t>(std::llround(clamped * static_cast<double>(size - 1)));
  return std::get<0>(torch::kthvalue(flat, rank + 1, 0, false));
}

torch::Tensor normalize_map01_quantile_torch(const torch::Tensor& input, double low_q, double high_q) {
  auto lo = scalar_quantile_tensor_torch(input, low_q);
  auto hi = scalar_quantile_tensor_torch(input, high_q);
  auto scale = torch::clamp_min(hi - lo, 1e-6);
  return torch::clamp((input - lo) / scale, 0.0, 1.0);
}

torch::Tensor normalize_map01_masked_minmax_torch(const torch::Tensor& input, const torch::Tensor& valid_mask) {
  auto output = torch::zeros_like(input);
  auto active = input.masked_select(valid_mask);
  if (active.numel() == 0) {
    return output;
  }
  const double lo = active.min().item<double>();
  const double hi = active.max().item<double>();
  const double scale = std::max(hi - lo, 1e-6);
  auto normalized = torch::clamp((input - lo) / scale, 0.0, 1.0).to(input.scalar_type());
  return torch::where(valid_mask, normalized, output);
}

torch::Tensor gaussian_kernel_tensor_torch(double sigma, const c10::Device& device, c10::ScalarType dtype) {
  if (sigma <= 0.0) {
    return torch::ones({1}, torch::TensorOptions().dtype(dtype).device(device));
  }
  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(dtype).device(device));
  auto kernel = torch::exp(-(x * x) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();
  return kernel.contiguous();
}

torch::Tensor gaussian_first_derivative_kernel_tensor_torch(double sigma, const c10::Device& device, c10::ScalarType dtype) {
  if (sigma <= 0.0) {
    return torch::zeros({1}, torch::TensorOptions().dtype(dtype).device(device));
  }
  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(dtype).device(device));
  const double sigma2 = sigma * sigma;
  auto kernel = (-x / sigma2) * torch::exp(-(x * x) / (2.0 * sigma2));
  return kernel.contiguous();
}

torch::Tensor gaussian_second_derivative_kernel_tensor_torch(double sigma, const c10::Device& device, c10::ScalarType dtype) {
  if (sigma <= 0.0) {
    return torch::zeros({1}, torch::TensorOptions().dtype(dtype).device(device));
  }
  const auto radius = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(3.0 * sigma)));
  auto x = torch::arange(-radius, radius + 1, torch::TensorOptions().dtype(dtype).device(device));
  const double sigma2 = sigma * sigma;
  auto kernel = ((x * x - sigma2) / (sigma2 * sigma2)) * torch::exp(-(x * x) / (2.0 * sigma2));
  return kernel.contiguous();
}

torch::Tensor convolve_rows_2d_torch(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(0).unsqueeze(0), {0, 0, radius, radius});
  return torch::conv2d(padded, kernel.view({1, 1, kernel.size(0), 1})).squeeze(0).squeeze(0);
}

torch::Tensor convolve_cols_2d_torch(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(0).unsqueeze(0), {radius, radius, 0, 0});
  return torch::conv2d(padded, kernel.view({1, 1, 1, kernel.size(0)})).squeeze(0).squeeze(0);
}

torch::Tensor convolve_rows_2d_torch_batch(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(1), {0, 0, radius, radius});
  return torch::conv2d(padded, kernel.view({1, 1, kernel.size(0), 1})).squeeze(1).contiguous();
}

torch::Tensor convolve_cols_2d_torch_batch(const torch::Tensor& input, const torch::Tensor& kernel) {
  const auto radius = kernel.size(0) / 2;
  auto padded = torch::replication_pad2d(input.unsqueeze(1), {radius, radius, 0, 0});
  return torch::conv2d(padded, kernel.view({1, 1, 1, kernel.size(0)})).squeeze(1).contiguous();
}

torch::Tensor gaussian_blur_2d_torch(const torch::Tensor& input, double sigma_rows, double sigma_cols) {
  auto row_kernel = gaussian_kernel_tensor_torch(sigma_rows, input.device(), input.scalar_type());
  auto col_kernel = gaussian_kernel_tensor_torch(sigma_cols, input.device(), input.scalar_type());
  return convolve_cols_2d_torch(convolve_rows_2d_torch(input, row_kernel), col_kernel).contiguous();
}

torch::Tensor gaussian_blur_2d_torch_batch(const torch::Tensor& input, double sigma_rows, double sigma_cols) {
  auto row_kernel = gaussian_kernel_tensor_torch(sigma_rows, input.device(), input.scalar_type());
  auto col_kernel = gaussian_kernel_tensor_torch(sigma_cols, input.device(), input.scalar_type());
  return convolve_cols_2d_torch_batch(convolve_rows_2d_torch_batch(input, row_kernel), col_kernel).contiguous();
}

torch::Tensor gaussian_first_derivative_rows_2d_torch(const torch::Tensor& input, double sigma) {
  auto smooth_kernel = gaussian_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  auto deriv_kernel = gaussian_first_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_rows_2d_torch(convolve_cols_2d_torch(input, smooth_kernel), deriv_kernel).contiguous();
}

torch::Tensor gaussian_first_derivative_cols_2d_torch(const torch::Tensor& input, double sigma) {
  auto smooth_kernel = gaussian_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  auto deriv_kernel = gaussian_first_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_cols_2d_torch(convolve_rows_2d_torch(input, smooth_kernel), deriv_kernel).contiguous();
}

torch::Tensor gaussian_first_derivative_rows_2d_torch_batch(const torch::Tensor& input, double sigma) {
  auto smooth_kernel = gaussian_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  auto deriv_kernel = gaussian_first_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_rows_2d_torch_batch(convolve_cols_2d_torch_batch(input, smooth_kernel), deriv_kernel).contiguous();
}

torch::Tensor gaussian_first_derivative_cols_2d_torch_batch(const torch::Tensor& input, double sigma) {
  auto smooth_kernel = gaussian_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  auto deriv_kernel = gaussian_first_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_cols_2d_torch_batch(convolve_rows_2d_torch_batch(input, smooth_kernel), deriv_kernel).contiguous();
}

torch::Tensor gaussian_second_derivative_rows_2d_torch(const torch::Tensor& input, double sigma) {
  auto kernel = gaussian_second_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_rows_2d_torch(input, kernel).contiguous();
}

torch::Tensor gaussian_second_derivative_rows_2d_torch_batch(const torch::Tensor& input, double sigma) {
  auto kernel = gaussian_second_derivative_kernel_tensor_torch(sigma, input.device(), input.scalar_type());
  return convolve_rows_2d_torch_batch(input, kernel).contiguous();
}

torch::Tensor normalize_map01_masked_minmax_torch_batch(const torch::Tensor& input, const torch::Tensor& valid_mask) {
  auto output = torch::zeros_like(input);
  if (input.dim() != 3 || valid_mask.dim() != 3 || input.sizes() != valid_mask.sizes()) {
    return output;
  }

  const auto batch_size = input.size(0);
  if (batch_size <= 0) {
    return output;
  }

  auto flat_valid = valid_mask.reshape({batch_size, -1});
  auto has_active = flat_valid.any(1);
  if (!has_active.any().item<bool>()) {
    return output;
  }

  auto pos_inf = torch::full_like(input, std::numeric_limits<float>::infinity());
  auto neg_inf = torch::full_like(input, -std::numeric_limits<float>::infinity());
  auto masked_lo = torch::where(valid_mask, input, pos_inf).reshape({batch_size, -1});
  auto masked_hi = torch::where(valid_mask, input, neg_inf).reshape({batch_size, -1});
  auto lo = std::get<0>(masked_lo.min(1)).view({batch_size, 1, 1});
  auto hi = std::get<0>(masked_hi.max(1)).view({batch_size, 1, 1});
  auto scale = torch::clamp_min(hi - lo, 1.0e-6);
  auto normalized = torch::clamp((input - lo) / scale, 0.0, 1.0).to(input.scalar_type());
  auto normalized_masked = torch::where(valid_mask, normalized, output);
  auto active_mask = has_active.view({batch_size, 1, 1}).expand_as(valid_mask);
  return torch::where(active_mask, normalized_masked, output);
}

torch::Tensor select_quantile_flat_batch_torch(const torch::Tensor& input, double q) {
  if (input.dim() != 3 || input.size(0) <= 0) {
    return torch::zeros({0, 1, 1}, input.options());
  }
  auto flat = input.reshape({input.size(0), -1});
  const auto size = flat.size(1);
  if (size <= 1) {
    return flat.select(1, 0).view({input.size(0), 1, 1});
  }
  const double clamped = std::clamp(q, 0.0, 1.0);
  const auto rank = static_cast<int64_t>(std::llround(clamped * static_cast<double>(size - 1)));
  return std::get<0>(torch::kthvalue(flat, rank + 1, 1, false)).view({input.size(0), 1, 1});
}

torch::Tensor normalize_map01_quantile_torch_batch(const torch::Tensor& input, double low_q, double high_q) {
  if (input.dim() != 3) {
    return torch::zeros_like(input);
  }
  auto lo = select_quantile_flat_batch_torch(input, low_q);
  auto hi = select_quantile_flat_batch_torch(input, high_q);
  auto scale = torch::clamp_min(hi - lo, 1e-6);
  return torch::clamp((input - lo) / scale, 0.0, 1.0);
}

torch::Tensor uniform_filter_2d_nearest_torch(const torch::Tensor& input, int kernel_rows, int kernel_cols) {
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

    torch::Tensor uniform_filter_2d_nearest_torch_batch(const torch::Tensor& input, int kernel_rows, int kernel_cols) {
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

std::vector<float> structure_tensor_gate_gpu(const float* corrected_device,
                                             int rows,
                                             int cols,
                                             const std::vector<uint8_t>& valid_row_mask,
                                             cudaStream_t cuda_stream) {
  if (rows <= 0 || cols <= 0 || corrected_device == nullptr ||
      valid_row_mask.size() != static_cast<size_t>(rows)) {
    return {};
  }

  torch::InferenceMode inference_mode_guard(true);
  const c10::Device compute_device(torch::kCUDA, 0);
  const auto torch_stream = cuda_stream
                                ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                : c10::cuda::getDefaultCUDAStream(compute_device.index());
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
  auto corrected = torch::from_blob(const_cast<float*>(corrected_device),
                                    {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                    float_options)
                       .contiguous();
  auto valid_row_mask_gpu = torch::from_blob(const_cast<uint8_t*>(valid_row_mask.data()),
                                             {static_cast<int64_t>(rows), 1},
                                             torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                                .to(compute_device, torch::kBool)
                                .expand({static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

  const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
  const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
  auto background = uniform_filter_2d_nearest_torch(corrected,
                                                    std::max(1, bg_freq),
                                                    std::max(1, bg_time));
  auto residual_db = torch::clamp_min(corrected - background, 0.0);
  auto residual_n = normalize_map01_quantile_torch(residual_db, 0.05, 0.99);

  const std::array<double, 3> scales = {0.8, 1.6, 3.2};
  auto gate_max = torch::zeros_like(corrected, float_options);
  for (double grad_sigma : scales) {
    const double integ_sigma = std::max(1.0, 1.8 * grad_sigma);
    auto grad_f = gaussian_first_derivative_rows_2d_torch(residual_n, grad_sigma);
    auto grad_t = gaussian_first_derivative_cols_2d_torch(residual_n, grad_sigma);
    auto j_ff = gaussian_blur_2d_torch(grad_f * grad_f, integ_sigma, integ_sigma);
    auto j_ft = gaussian_blur_2d_torch(grad_f * grad_t, integ_sigma, integ_sigma);
    auto j_tt = gaussian_blur_2d_torch(grad_t * grad_t, integ_sigma, integ_sigma);

    auto delta = torch::sqrt(torch::clamp_min((j_ff - j_tt) * (j_ff - j_tt) + 4.0f * (j_ft * j_ft), 0.0));
    auto lambda1 = 0.5f * (j_ff + j_tt + delta);
    auto lambda2 = 0.5f * (j_ff + j_tt - delta);
    auto coherence = (lambda1 - lambda2) / torch::clamp_min(lambda1 + lambda2, 1.0e-6);
    auto energy = lambda1 + lambda2;

    auto coherence_n = normalize_map01_quantile_torch(coherence, 0.05, 0.99);
    auto energy_n = normalize_map01_quantile_torch(energy, 0.05, 0.99);
    auto gate_value = coherence_n * torch::sqrt(torch::clamp_min(energy_n, 0.0));
    gate_max = torch::maximum(gate_max, gate_value);
  }

  auto gate = normalize_map01_quantile_torch(gate_max, 0.05, 0.99).to(torch::kFloat32);
  gate = torch::where(valid_row_mask_gpu, gate, torch::zeros_like(gate));
  return tensor_to_vector_float(gate);
}

std::vector<float> structure_tensor_gate_gpu_batch(const float* corrected_batch_device,
                                                   int batch_size,
                                                   int rows,
                                                   int cols,
                                                   const std::vector<uint8_t>& valid_row_mask_batch,
                                                   cudaStream_t cuda_stream) {
  auto gate_tensor = [&]() -> torch::Tensor {
    if (batch_size <= 0 || rows <= 0 || cols <= 0 || corrected_batch_device == nullptr ||
        valid_row_mask_batch.size() != static_cast<size_t>(batch_size) * static_cast<size_t>(rows)) {
      return {};
    }

    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto torch_stream = cuda_stream
                                  ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                  : c10::cuda::getDefaultCUDAStream(compute_device.index());
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

    auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
    auto corrected = torch::from_blob(const_cast<float*>(corrected_batch_device),
                                      {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                      float_options)
                         .contiguous();
    auto valid_mask_gpu = torch::from_blob(const_cast<uint8_t*>(valid_row_mask_batch.data()),
                         {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), 1},
                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                  .to(compute_device, torch::kBool)
                  .expand({static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

    const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
    const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
    auto background = uniform_filter_2d_nearest_torch_batch(corrected,
                                                            std::max(1, bg_freq),
                                                            std::max(1, bg_time));
    auto residual_db = torch::clamp_min(corrected - background, 0.0);
    auto residual_n = normalize_map01_quantile_torch_batch(residual_db, 0.05, 0.99);

    const std::array<double, 3> scales = {0.8, 1.6, 3.2};
    auto gate_max = torch::zeros_like(corrected, float_options);
    for (double grad_sigma : scales) {
      const double integ_sigma = std::max(1.0, 1.8 * grad_sigma);
      auto grad_f = gaussian_first_derivative_rows_2d_torch_batch(residual_n, grad_sigma);
      auto grad_t = gaussian_first_derivative_cols_2d_torch_batch(residual_n, grad_sigma);
      auto j_ff = gaussian_blur_2d_torch_batch(grad_f * grad_f, integ_sigma, integ_sigma);
      auto j_ft = gaussian_blur_2d_torch_batch(grad_f * grad_t, integ_sigma, integ_sigma);
      auto j_tt = gaussian_blur_2d_torch_batch(grad_t * grad_t, integ_sigma, integ_sigma);

      auto delta = torch::sqrt(torch::clamp_min((j_ff - j_tt) * (j_ff - j_tt) + 4.0f * (j_ft * j_ft), 0.0));
      auto lambda1 = 0.5f * (j_ff + j_tt + delta);
      auto lambda2 = 0.5f * (j_ff + j_tt - delta);
      auto coherence = (lambda1 - lambda2) / torch::clamp_min(lambda1 + lambda2, 1.0e-6);
      auto energy = lambda1 + lambda2;

      auto coherence_n = normalize_map01_quantile_torch_batch(coherence, 0.05, 0.99);
      auto energy_n = normalize_map01_quantile_torch_batch(energy, 0.05, 0.99);
      auto gate_value = coherence_n * torch::sqrt(torch::clamp_min(energy_n, 0.0));
      gate_max = torch::maximum(gate_max, gate_value);
    }

    auto gate = normalize_map01_quantile_torch_batch(gate_max, 0.05, 0.99).to(torch::kFloat32);
    return torch::where(valid_mask_gpu, gate, torch::zeros_like(gate)).contiguous();
  }();

  if (!gate_tensor.defined()) {
    return {};
  }
  auto gate_cpu = gate_tensor.to(torch::kCPU, torch::kFloat32).contiguous();
  std::vector<float> output(static_cast<size_t>(gate_cpu.numel()), 0.0f);
  if (!output.empty()) {
    std::memcpy(output.data(), gate_cpu.data_ptr<float>(), output.size() * sizeof(float));
  }
  return output;
}

torch::Tensor structure_tensor_gate_gpu_batch_tensor(const float* corrected_batch_device,
                                                     int batch_size,
                                                     int rows,
                                                     int cols,
                                                     const std::vector<uint8_t>& valid_row_mask_batch,
                                                     cudaStream_t cuda_stream) {
  if (batch_size <= 0 || rows <= 0 || cols <= 0 || corrected_batch_device == nullptr ||
      valid_row_mask_batch.size() != static_cast<size_t>(batch_size) * static_cast<size_t>(rows)) {
    return {};
  }

  torch::InferenceMode inference_mode_guard(true);
  const c10::Device compute_device(torch::kCUDA, 0);
  const auto torch_stream = cuda_stream
                                ? c10::cuda::getStreamFromExternal(cuda_stream, compute_device.index())
                                : c10::cuda::getDefaultCUDAStream(compute_device.index());
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto float_options = torch::TensorOptions().dtype(torch::kFloat32).device(compute_device);
  auto corrected = torch::from_blob(const_cast<float*>(corrected_batch_device),
                                    {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                    float_options)
                       .contiguous();
  auto valid_mask_gpu = torch::from_blob(const_cast<uint8_t*>(valid_row_mask_batch.data()),
                                         {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), 1},
                                         torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                            .to(compute_device, torch::kBool)
                            .expand({static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)});

  const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
  const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
  auto background = uniform_filter_2d_nearest_torch_batch(corrected,
                                                          std::max(1, bg_freq),
                                                          std::max(1, bg_time));
  auto residual_db = torch::clamp_min(corrected - background, 0.0);
  auto residual_n = normalize_map01_quantile_torch_batch(residual_db, 0.05, 0.99);

  const std::array<double, 3> scales = {0.8, 1.6, 3.2};
  auto gate_max = torch::zeros_like(corrected, float_options);
  for (double grad_sigma : scales) {
    const double integ_sigma = std::max(1.0, 1.8 * grad_sigma);
    auto grad_f = gaussian_first_derivative_rows_2d_torch_batch(residual_n, grad_sigma);
    auto grad_t = gaussian_first_derivative_cols_2d_torch_batch(residual_n, grad_sigma);
    auto j_ff = gaussian_blur_2d_torch_batch(grad_f * grad_f, integ_sigma, integ_sigma);
    auto j_ft = gaussian_blur_2d_torch_batch(grad_f * grad_t, integ_sigma, integ_sigma);
    auto j_tt = gaussian_blur_2d_torch_batch(grad_t * grad_t, integ_sigma, integ_sigma);

    auto delta = torch::sqrt(torch::clamp_min((j_ff - j_tt) * (j_ff - j_tt) + 4.0f * (j_ft * j_ft), 0.0));
    auto lambda1 = 0.5f * (j_ff + j_tt + delta);
    auto lambda2 = 0.5f * (j_ff + j_tt - delta);
    auto coherence = (lambda1 - lambda2) / torch::clamp_min(lambda1 + lambda2, 1.0e-6);
    auto energy = lambda1 + lambda2;

    auto coherence_n = normalize_map01_quantile_torch_batch(coherence, 0.05, 0.99);
    auto energy_n = normalize_map01_quantile_torch_batch(energy, 0.05, 0.99);
    auto gate_value = coherence_n * torch::sqrt(torch::clamp_min(energy_n, 0.0));
    gate_max = torch::maximum(gate_max, gate_value);
  }

  auto gate = normalize_map01_quantile_torch_batch(gate_max, 0.05, 0.99).to(torch::kFloat32);
  return torch::where(valid_mask_gpu, gate, torch::zeros_like(gate)).contiguous();
}

torch::Tensor project_aligned_map_to_output_torch_batch(const torch::Tensor& aligned_batch,
                                                        int source_rows,
                                                        int source_cols,
                                                        int output_rows,
                                                        int output_cols,
                                                        bool resized_full_chunk) {
  if (!aligned_batch.defined() || aligned_batch.dim() != 3) {
    return {};
  }
  auto source_canvas = aligned_batch;
  if (resized_full_chunk) {
    source_canvas = torch::nn::functional::interpolate(
                       aligned_batch.unsqueeze(1),
                       torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>{static_cast<int64_t>(source_rows), static_cast<int64_t>(source_cols)})
                           .mode(torch::kBilinear)
                           .align_corners(false))
                       .squeeze(1)
                       .contiguous();
  }
  if (source_canvas.size(1) == output_rows && source_canvas.size(2) == output_cols) {
    return source_canvas.contiguous();
  }
  return torch::nn::functional::interpolate(
             source_canvas.unsqueeze(1),
             torch::nn::functional::InterpolateFuncOptions()
                 .size(std::vector<int64_t>{static_cast<int64_t>(output_rows), static_cast<int64_t>(output_cols)})
                 .mode(torch::kBilinear)
                 .align_corners(false))
      .squeeze(1)
      .contiguous();
}

std::vector<HybridPostprocessResult> run_residual_veto_hybrid_gpu_batch_device_inputs(const torch::Tensor& dino_score_source_batch,
                                                                                       const torch::Tensor& coherence_source_batch,
                                                                                       const std::vector<uint8_t>& valid_mask_batch,
                                                                                       int batch_size,
                                                                                       int rows,
                                                                                       int cols,
                                                                                       bool use_fp16) {
  const size_t sample_elements = static_cast<size_t>(std::max(rows, 0)) * static_cast<size_t>(std::max(cols, 0));
  if (batch_size <= 0 || rows <= 0 || cols <= 0 || !dino_score_source_batch.defined() || !coherence_source_batch.defined() ||
      dino_score_source_batch.dim() != 3 || coherence_source_batch.dim() != 3 ||
      dino_score_source_batch.size(0) != batch_size || coherence_source_batch.size(0) != batch_size ||
      dino_score_source_batch.size(1) != rows || coherence_source_batch.size(1) != rows ||
      dino_score_source_batch.size(2) != cols || coherence_source_batch.size(2) != cols ||
      valid_mask_batch.size() != static_cast<size_t>(batch_size) * sample_elements) {
    throw std::runtime_error("invalid GPU batch hybrid postprocess inputs");
  }
  if (!torch::cuda::is_available()) {
    throw std::runtime_error("CUDA is unavailable for GPU batch hybrid postprocess");
  }

  try {
    std::vector<HybridPostprocessResult> results(static_cast<size_t>(batch_size));
    for (auto& result : results) {
      result.mask.assign(sample_elements, 0);
    }
    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto contrib_dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
    auto dino_source = dino_score_source_batch.to(compute_device, contrib_dtype).contiguous();
    auto coherence_source = coherence_source_batch.to(compute_device, contrib_dtype).contiguous();
    auto valid = torch::from_blob(const_cast<uint8_t*>(valid_mask_batch.data()),
                                  {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                     .clone()
                     .to(compute_device, torch::kBool);

    auto dino_norm = normalize_map01_quantile_torch_batch(dino_source, 0.05, 0.95);
    auto coherence_norm = normalize_map01_quantile_torch_batch(coherence_source, 0.05, 0.99);
    auto contrib = (dino_norm * coherence_norm).contiguous();

    auto base_norm = normalize_map01_masked_minmax_torch_batch(contrib, valid);
    auto envelope_map = normalize_map01_masked_minmax_torch_batch(gaussian_blur_2d_torch_batch(base_norm, 6.0, 1.4), valid);
    auto base_blur = gaussian_blur_2d_torch_batch(base_norm, 4.0, 1.0);
    auto residual_penalty = normalize_map01_masked_minmax_torch_batch(gaussian_blur_2d_torch_batch(torch::abs(base_norm - base_blur), 2.0, 0.8), valid);
    auto freq_curvature_penalty = normalize_map01_masked_minmax_torch_batch(torch::abs(gaussian_second_derivative_rows_2d_torch_batch(base_norm, 0.8)), valid);

    auto keep_freq = normalize_map01_masked_minmax_torch_batch(envelope_map - 0.90 * freq_curvature_penalty, valid);
    auto keep_res = normalize_map01_masked_minmax_torch_batch(envelope_map - 1.00 * residual_penalty, valid);
    auto residual_veto_gate = torch::clamp((keep_res - 0.30) / 0.70, 0.0, 1.0);
    auto combined_input = keep_freq * (0.35 + 0.65 * residual_veto_gate);
    auto combined_score = normalize_map01_masked_minmax_torch_batch(combined_input, valid);

    std::vector<torch::Tensor> final_masks;
    final_masks.reserve(static_cast<size_t>(batch_size));
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      auto sample_valid = valid[sample_index];
      auto active_freq = keep_freq[sample_index].masked_select(sample_valid);
      auto active_res = keep_res[sample_index].masked_select(sample_valid);
      auto active_combined = combined_score[sample_index].masked_select(sample_valid);
      auto& result = results[static_cast<size_t>(sample_index)];
      result.seed_freq_threshold = active_freq.numel() > 0 ? scalar_quantile_tensor_torch(active_freq, 0.90).item<float>() : 1.0f;
      result.seed_res_threshold = active_res.numel() > 0 ? scalar_quantile_tensor_torch(active_res, 0.82).item<float>() : 1.0f;
      result.grow_freq_threshold = result.seed_freq_threshold;
      result.grow_res_threshold = result.seed_res_threshold;
      result.combined_threshold = active_combined.numel() > 0 ? scalar_quantile_tensor_torch(active_combined, 0.78).item<float>() : 1.0f;
      auto seed_mask = torch::logical_and(sample_valid,
                                          torch::logical_and(keep_freq[sample_index] >= result.seed_freq_threshold,
                                                             keep_res[sample_index] >= result.seed_res_threshold));
      auto final_mask = torch::logical_and(seed_mask,
                                           torch::logical_and(sample_valid,
                                                              combined_score[sample_index] >= static_cast<double>(result.combined_threshold) * 0.85));
      final_masks.push_back(final_mask.to(torch::kUInt8));
    }

    auto combined_score_cpu = combined_score.to(torch::kCPU, torch::kFloat32).contiguous();
    auto final_mask_batch_cpu = torch::stack(final_masks, 0).to(torch::kCPU, torch::kUInt8).contiguous();
    const float* combined_score_ptr = combined_score_cpu.data_ptr<float>();
    const uint8_t* final_mask_ptr = final_mask_batch_cpu.data_ptr<uint8_t>();
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t max_parallel = std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(batch_size), static_cast<size_t>(hw_threads)));
    auto process_sample = [&](int sample_index) {
      auto& result = results[static_cast<size_t>(sample_index)];
      const size_t offset = static_cast<size_t>(sample_index) * sample_elements;
      result.combined_score.assign(combined_score_ptr + static_cast<std::ptrdiff_t>(offset),
                                   combined_score_ptr + static_cast<std::ptrdiff_t>(offset + sample_elements));
      std::vector<uint8_t> final_mask_vector(final_mask_ptr + static_cast<std::ptrdiff_t>(offset),
                                             final_mask_ptr + static_cast<std::ptrdiff_t>(offset + sample_elements));
      final_mask_vector = binary_closing_rect(final_mask_vector, rows, cols, 7, 3);
      final_mask_vector = binary_fill_holes(final_mask_vector, rows, cols);
      const auto valid_begin = valid_mask_batch.begin() + static_cast<std::ptrdiff_t>(offset);
      const std::vector<uint8_t> sample_valid_mask(valid_begin,
                                                   valid_begin + static_cast<std::ptrdiff_t>(sample_elements));
      for (size_t index = 0; index < sample_elements; ++index) {
        final_mask_vector[index] = (final_mask_vector[index] && sample_valid_mask[index]) ? 1 : 0;
      }
      final_mask_vector = keep_large_components(final_mask_vector, rows, cols, 24, &result.component_count);
      result.final_fraction = mean_mask_value(final_mask_vector);
      result.connected_fraction = connected_fraction(final_mask_vector, sample_valid_mask);
      result.mask = std::move(final_mask_vector);
    };

    if (max_parallel == 1) {
      for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
        process_sample(sample_index);
      }
    } else {
      std::vector<std::future<void>> pending;
      pending.reserve(max_parallel);
      for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
        pending.push_back(std::async(std::launch::async, process_sample, sample_index));
        if (pending.size() >= max_parallel) {
          pending.front().get();
          pending.erase(pending.begin());
        }
      }
      for (auto& task : pending) {
        task.get();
      }
    }
    return results;
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("GPU batch hybrid postprocess failed: ") + error.what());
  }
}

HybridPostprocessResult run_residual_veto_hybrid_gpu(const std::vector<float>& hybrid_dino_contrib,
                                                     const std::vector<uint8_t>& valid_mask,
                                                     int rows,
                                                     int cols,
                                                     bool use_fp16) {
  HybridPostprocessResult result;
  result.mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  if (hybrid_dino_contrib.size() != result.mask.size() || valid_mask.size() != result.mask.size()) {
    throw std::runtime_error("invalid GPU hybrid postprocess inputs");
  }
  if (!torch::cuda::is_available()) {
    throw std::runtime_error("CUDA is unavailable for GPU hybrid postprocess");
  }

  try {
    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto contrib_dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
    auto contrib = torch::from_blob(const_cast<float*>(hybrid_dino_contrib.data()),
                                    {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                       .clone()
                       .to(compute_device, contrib_dtype);
    auto valid = torch::from_blob(const_cast<uint8_t*>(valid_mask.data()),
                                  {static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                     .clone()
                     .to(compute_device, torch::kBool);

    auto base_norm = normalize_map01_masked_minmax_torch(contrib, valid);
    auto envelope_map = normalize_map01_masked_minmax_torch(gaussian_blur_2d_torch(base_norm, 6.0, 1.4), valid);
    auto base_blur = gaussian_blur_2d_torch(base_norm, 4.0, 1.0);
    auto residual_penalty = normalize_map01_masked_minmax_torch(gaussian_blur_2d_torch(torch::abs(base_norm - base_blur), 2.0, 0.8), valid);
    auto freq_curvature_penalty = normalize_map01_masked_minmax_torch(torch::abs(gaussian_second_derivative_rows_2d_torch(base_norm, 0.8)), valid);

    auto keep_freq = normalize_map01_masked_minmax_torch(envelope_map - 0.90 * freq_curvature_penalty, valid);
    auto keep_res = normalize_map01_masked_minmax_torch(envelope_map - 1.00 * residual_penalty, valid);
    auto residual_veto_gate = torch::clamp((keep_res - 0.30) / 0.70, 0.0, 1.0);
    auto combined_input = keep_freq * (0.35 + 0.65 * residual_veto_gate);
    auto combined_score = normalize_map01_masked_minmax_torch(combined_input, valid);
    result.combined_score = tensor_to_vector_float(combined_score);

    auto active_freq = keep_freq.masked_select(valid);
    auto active_res = keep_res.masked_select(valid);
    auto active_combined = combined_score.masked_select(valid);
    result.seed_freq_threshold = active_freq.numel() > 0 ? scalar_quantile_tensor_torch(active_freq, 0.90).item<float>() : 1.0f;
    result.seed_res_threshold = active_res.numel() > 0 ? scalar_quantile_tensor_torch(active_res, 0.82).item<float>() : 1.0f;
    result.grow_freq_threshold = result.seed_freq_threshold;
    result.grow_res_threshold = result.seed_res_threshold;
    result.combined_threshold = active_combined.numel() > 0 ? scalar_quantile_tensor_torch(active_combined, 0.78).item<float>() : 1.0f;

    auto seed_mask = torch::logical_and(valid,
                                        torch::logical_and(keep_freq >= result.seed_freq_threshold,
                                                           keep_res >= result.seed_res_threshold));
    auto final_mask = torch::logical_and(seed_mask,
                                         torch::logical_and(valid,
                                                            combined_score >= static_cast<double>(result.combined_threshold) * 0.85));

    auto final_mask_cpu = final_mask.to(torch::kCPU, torch::kUInt8).contiguous();
    std::vector<uint8_t> final_mask_vector(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
    if (!final_mask_vector.empty()) {
      std::memcpy(final_mask_vector.data(), final_mask_cpu.data_ptr<uint8_t>(), final_mask_vector.size() * sizeof(uint8_t));
    }

    final_mask_vector = binary_closing_rect(final_mask_vector, rows, cols, 7, 3);
    final_mask_vector = binary_fill_holes(final_mask_vector, rows, cols);
    for (size_t index = 0; index < final_mask_vector.size(); ++index) {
      final_mask_vector[index] = (final_mask_vector[index] && valid_mask[index]) ? 1 : 0;
    }
    final_mask_vector = keep_large_components(final_mask_vector, rows, cols, 24, &result.component_count);
    result.final_fraction = mean_mask_value(final_mask_vector);
    result.connected_fraction = connected_fraction(final_mask_vector, valid_mask);
    result.mask = std::move(final_mask_vector);
    return result;
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("GPU hybrid postprocess failed: ") + error.what());
  }
}

std::vector<HybridPostprocessResult> run_residual_veto_hybrid_gpu_batch(const std::vector<float>& hybrid_dino_contrib_batch,
                                                                        const std::vector<uint8_t>& valid_mask_batch,
                                                                        int batch_size,
                                                                        int rows,
                                                                        int cols,
                                                                        bool use_fp16) {
  std::vector<HybridPostprocessResult> results(static_cast<size_t>(std::max(batch_size, 0)));
  const size_t sample_elements = static_cast<size_t>(std::max(rows, 0)) * static_cast<size_t>(std::max(cols, 0));
  for (auto& result : results) {
    result.mask.assign(sample_elements, 0);
  }
  if (batch_size <= 0 || rows <= 0 || cols <= 0 ||
      hybrid_dino_contrib_batch.size() != static_cast<size_t>(batch_size) * sample_elements ||
      valid_mask_batch.size() != static_cast<size_t>(batch_size) * sample_elements) {
    return results;
  }
  if (!torch::cuda::is_available()) {
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      const auto contrib_begin = hybrid_dino_contrib_batch.begin() + static_cast<std::ptrdiff_t>(sample_index * sample_elements);
      const auto mask_begin = valid_mask_batch.begin() + static_cast<std::ptrdiff_t>(sample_index * sample_elements);
      results[static_cast<size_t>(sample_index)] = run_residual_veto_hybrid_gpu(
          std::vector<float>(contrib_begin, contrib_begin + static_cast<std::ptrdiff_t>(sample_elements)),
          std::vector<uint8_t>(mask_begin, mask_begin + static_cast<std::ptrdiff_t>(sample_elements)),
          rows,
          cols,
          use_fp16);
    }
    return results;
  }

  try {
    torch::InferenceMode inference_mode_guard(true);
    const c10::Device compute_device(torch::kCUDA, 0);
    const auto contrib_dtype = use_fp16 ? torch::kFloat16 : torch::kFloat32;
    auto contrib = torch::from_blob(const_cast<float*>(hybrid_dino_contrib_batch.data()),
                                    {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                       .clone()
                       .to(compute_device, contrib_dtype);
    auto valid = torch::from_blob(const_cast<uint8_t*>(valid_mask_batch.data()),
                                  {static_cast<int64_t>(batch_size), static_cast<int64_t>(rows), static_cast<int64_t>(cols)},
                                  torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU))
                     .clone()
                     .to(compute_device, torch::kBool);

    auto base_norm = normalize_map01_masked_minmax_torch_batch(contrib, valid);
    auto envelope_map = normalize_map01_masked_minmax_torch_batch(gaussian_blur_2d_torch_batch(base_norm, 6.0, 1.4), valid);
    auto base_blur = gaussian_blur_2d_torch_batch(base_norm, 4.0, 1.0);
    auto residual_penalty = normalize_map01_masked_minmax_torch_batch(gaussian_blur_2d_torch_batch(torch::abs(base_norm - base_blur), 2.0, 0.8), valid);
    auto freq_curvature_penalty = normalize_map01_masked_minmax_torch_batch(torch::abs(gaussian_second_derivative_rows_2d_torch_batch(base_norm, 0.8)), valid);

    auto keep_freq = normalize_map01_masked_minmax_torch_batch(envelope_map - 0.90 * freq_curvature_penalty, valid);
    auto keep_res = normalize_map01_masked_minmax_torch_batch(envelope_map - 1.00 * residual_penalty, valid);
    auto residual_veto_gate = torch::clamp((keep_res - 0.30) / 0.70, 0.0, 1.0);
    auto combined_input = keep_freq * (0.35 + 0.65 * residual_veto_gate);
    auto combined_score = normalize_map01_masked_minmax_torch_batch(combined_input, valid);

    std::vector<torch::Tensor> final_masks;
    final_masks.reserve(static_cast<size_t>(batch_size));
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      auto sample_valid = valid[sample_index];
      auto active_freq = keep_freq[sample_index].masked_select(sample_valid);
      auto active_res = keep_res[sample_index].masked_select(sample_valid);
      auto active_combined = combined_score[sample_index].masked_select(sample_valid);
      auto& result = results[static_cast<size_t>(sample_index)];
      result.seed_freq_threshold = active_freq.numel() > 0 ? scalar_quantile_tensor_torch(active_freq, 0.90).item<float>() : 1.0f;
      result.seed_res_threshold = active_res.numel() > 0 ? scalar_quantile_tensor_torch(active_res, 0.82).item<float>() : 1.0f;
      result.grow_freq_threshold = result.seed_freq_threshold;
      result.grow_res_threshold = result.seed_res_threshold;
      result.combined_threshold = active_combined.numel() > 0 ? scalar_quantile_tensor_torch(active_combined, 0.78).item<float>() : 1.0f;
      auto seed_mask = torch::logical_and(sample_valid,
                                          torch::logical_and(keep_freq[sample_index] >= result.seed_freq_threshold,
                                                             keep_res[sample_index] >= result.seed_res_threshold));
      auto final_mask = torch::logical_and(seed_mask,
                                           torch::logical_and(sample_valid,
                                                              combined_score[sample_index] >= static_cast<double>(result.combined_threshold) * 0.85));
      final_masks.push_back(final_mask.to(torch::kUInt8));
    }

    auto combined_score_cpu = combined_score.to(torch::kCPU, torch::kFloat32).contiguous();
    auto final_mask_batch_cpu = torch::stack(final_masks, 0).to(torch::kCPU, torch::kUInt8).contiguous();
    const float* combined_score_ptr = combined_score_cpu.data_ptr<float>();
    const uint8_t* final_mask_ptr = final_mask_batch_cpu.data_ptr<uint8_t>();
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      auto& result = results[static_cast<size_t>(sample_index)];
      const size_t offset = static_cast<size_t>(sample_index) * sample_elements;
      result.combined_score.assign(combined_score_ptr + static_cast<std::ptrdiff_t>(offset),
                                   combined_score_ptr + static_cast<std::ptrdiff_t>(offset + sample_elements));
      std::vector<uint8_t> final_mask_vector(final_mask_ptr + static_cast<std::ptrdiff_t>(offset),
                                             final_mask_ptr + static_cast<std::ptrdiff_t>(offset + sample_elements));
      final_mask_vector = binary_closing_rect(final_mask_vector, rows, cols, 7, 3);
      final_mask_vector = binary_fill_holes(final_mask_vector, rows, cols);
      const auto valid_begin = valid_mask_batch.begin() + static_cast<std::ptrdiff_t>(offset);
      for (size_t index = 0; index < sample_elements; ++index) {
        final_mask_vector[index] = (final_mask_vector[index] && valid_begin[static_cast<std::ptrdiff_t>(index)]) ? 1 : 0;
      }
      final_mask_vector = keep_large_components(final_mask_vector, rows, cols, 24, &result.component_count);
      result.final_fraction = mean_mask_value(final_mask_vector);
      result.connected_fraction = connected_fraction(final_mask_vector,
                                                     std::vector<uint8_t>(valid_begin, valid_begin + static_cast<std::ptrdiff_t>(sample_elements)));
      result.mask = std::move(final_mask_vector);
    }
    return results;
  } catch (const std::exception&) {
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      const auto contrib_begin = hybrid_dino_contrib_batch.begin() + static_cast<std::ptrdiff_t>(sample_index * sample_elements);
      const auto mask_begin = valid_mask_batch.begin() + static_cast<std::ptrdiff_t>(sample_index * sample_elements);
      results[static_cast<size_t>(sample_index)] = run_residual_veto_hybrid(
          std::vector<float>(contrib_begin, contrib_begin + static_cast<std::ptrdiff_t>(sample_elements)),
          std::vector<uint8_t>(mask_begin, mask_begin + static_cast<std::ptrdiff_t>(sample_elements)),
          rows,
          cols);
    }
    return results;
  }
}

void precompute_retry_chunk_hybrid_batch(std::vector<ChunkInferenceArtifacts>& artifacts_batch,
                                         const ValidatorConfig& config,
                                         StageProfiler* profiler,
                                         bool verbose) {
  if (artifacts_batch.empty()) {
    return;
  }
  const int batch_size = static_cast<int>(artifacts_batch.size());
  const int rows = artifacts_batch.front().result.src_rows;
  const int cols = artifacts_batch.front().result.src_cols;
  if (rows <= 0 || cols <= 0) {
    return;
  }
  if (std::all_of(artifacts_batch.begin(), artifacts_batch.end(), [](const ChunkInferenceArtifacts& artifacts) {
        return artifacts.has_precomputed_hybrid_result;
      })) {
    return;
  }

  const size_t sample_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
  std::vector<float> hybrid_contrib_batch(static_cast<size_t>(batch_size) * sample_elements, 0.0f);
  std::vector<uint8_t> valid_mask_batch(static_cast<size_t>(batch_size) * sample_elements, 0);
  {
    const size_t estimated_bytes = static_cast<size_t>(batch_size) * sample_elements * (sizeof(float) * 3 + sizeof(uint8_t));
    ScopedStageProfile stage(profiler, "chunk_hybrid_support_batch", "run", -1, estimated_bytes, verbose);
    const auto norm_start = std::chrono::steady_clock::now();
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      auto& artifacts = artifacts_batch[static_cast<size_t>(sample_index)];
      if (artifacts.keep_debug_artifacts || artifacts.result.src_rows != rows || artifacts.result.src_cols != cols ||
          artifacts.grouped_dino_score_source.size() != sample_elements || artifacts.source_chunk_coherence_gate.size() != sample_elements ||
          artifacts.source_chunk_valid_mask.size() != sample_elements) {
        return;
      }
      const size_t offset = static_cast<size_t>(sample_index) * sample_elements;
      auto* hybrid_contrib_output = hybrid_contrib_batch.data() + static_cast<std::ptrdiff_t>(offset);
      const auto& hybrid_dino_source = artifacts.grouped_dino_score_source;
      const auto [dino_low, dino_high] = quantile_bounds_from_input(hybrid_dino_source, 5.0, 95.0);
      normalize01_quantile_into(hybrid_dino_source,
                                dino_low,
                                dino_high,
                                hybrid_contrib_output);
      const auto [coherence_low, coherence_high] = quantile_bounds_from_input(artifacts.source_chunk_coherence_gate, 5.0, 99.0);
      multiply_normalized_quantile_into(artifacts.source_chunk_coherence_gate,
                                        coherence_low,
                                        coherence_high,
                                        hybrid_contrib_output);
      std::copy(artifacts.source_chunk_valid_mask.begin(),
                artifacts.source_chunk_valid_mask.end(),
                valid_mask_batch.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    record_timed_stage(profiler,
                       "chunk_hybrid_norm_batch_cpu",
                       "run",
                       -1,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - norm_start).count());

    const auto residual_veto_start = std::chrono::steady_clock::now();
    auto hybrid_results = run_residual_veto_hybrid_gpu_batch(hybrid_contrib_batch,
                                                             valid_mask_batch,
                                                             batch_size,
                                                             rows,
                                                             cols,
                                                             use_fp16_precision(config.hybrid_torch_dtype));
    record_timed_stage(profiler,
                       "chunk_hybrid_residual_veto_batch",
                       "run",
                       -1,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - residual_veto_start).count());
    for (int sample_index = 0; sample_index < batch_size; ++sample_index) {
      artifacts_batch[static_cast<size_t>(sample_index)].precomputed_hybrid_result_source = std::move(hybrid_results[static_cast<size_t>(sample_index)]);
      artifacts_batch[static_cast<size_t>(sample_index)].has_precomputed_hybrid_result = true;
    }
  }
}

std::vector<float> patch_mean_map(const std::vector<float>& input,
                                  int src_rows,
                                  int src_cols,
                                  int patch_rows,
                                  int patch_cols) {
  std::vector<float> output(static_cast<size_t>(patch_rows) * static_cast<size_t>(patch_cols), 0.0f);
  if (src_rows <= 0 || src_cols <= 0 || patch_rows <= 0 || patch_cols <= 0) {
    return output;
  }
  const int block_rows = std::max(1, src_rows / patch_rows);
  const int block_cols = std::max(1, src_cols / patch_cols);
  const int use_rows = std::min(src_rows, patch_rows * block_rows);
  const int use_cols = std::min(src_cols, patch_cols * block_cols);
  for (int patch_row = 0; patch_row < patch_rows; ++patch_row) {
    for (int patch_col = 0; patch_col < patch_cols; ++patch_col) {
      float sum = 0.0f;
      int count = 0;
      for (int row = patch_row * block_rows; row < std::min(use_rows, (patch_row + 1) * block_rows); ++row) {
        for (int col = patch_col * block_cols; col < std::min(use_cols, (patch_col + 1) * block_cols); ++col) {
          sum += input[flat_index(src_cols, row, col)];
          ++count;
        }
      }
      output[flat_index(patch_cols, patch_row, patch_col)] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }
  }
  return output;
}

struct DinoSeedPatchComponents {
  std::vector<float> seed_patch;
  std::vector<float> persistence_patch;
  std::vector<float> contrast_patch;
};

DinoSeedPatchComponents dino_seed_patch_components(const std::vector<float>& spectrogram_db,
                                                   int src_rows,
                                                   int src_cols,
                                                   int runtime_rows,
                                                   int runtime_cols,
                                                   int patch_rows,
                                                   int patch_cols) {
  std::vector<float> rel_db(static_cast<size_t>(runtime_rows) * static_cast<size_t>(runtime_cols), 0.0f);
  std::vector<float> linear_values;
  linear_values.reserve(rel_db.size());
  for (int row = 0; row < runtime_rows; ++row) {
    for (int col = 0; col < runtime_cols; ++col) {
      const float db = spectrogram_db[flat_index(src_cols, row, col)];
      linear_values.push_back(std::pow(10.0f, db / 10.0f));
    }
  }
  const float p_floor = std::max(quantile_from_values(linear_values, 0.30, 1.0e-20f), 1.0e-20f);
  for (int row = 0; row < runtime_rows; ++row) {
    for (int col = 0; col < runtime_cols; ++col) {
      const float db = spectrogram_db[flat_index(src_cols, row, col)];
      const float p_lin = std::pow(10.0f, db / 10.0f);
      const float value = 10.0f * std::log10(std::max(p_lin, 1.0e-20f) / p_floor);
      rel_db[flat_index(runtime_cols, row, col)] = clamp_value(value, -10.0f, 25.0f);
    }
  }
  const auto persistence_px = box_mean_cols(rel_db, runtime_rows, runtime_cols, 3);
  const auto local_avg = box_mean_2d(rel_db, runtime_rows, runtime_cols, 2, 2);
  std::vector<float> contrast_px(rel_db.size(), 0.0f);
  for (size_t index = 0; index < contrast_px.size(); ++index) {
    contrast_px[index] = rel_db[index] - local_avg[index];
  }
  const auto persistence_n = normalize01_quantile(persistence_px, 5.0, 95.0);
  const auto contrast_n = normalize01_quantile(contrast_px, 5.0, 95.0);
  std::vector<float> seed_px(rel_db.size(), 0.0f);
  for (size_t index = 0; index < seed_px.size(); ++index) {
    seed_px[index] = 0.65f * persistence_n[index] + 0.35f * contrast_n[index];
  }
  DinoSeedPatchComponents result;
  result.seed_patch = patch_mean_map(seed_px, runtime_rows, runtime_cols, patch_rows, patch_cols);
  result.persistence_patch = patch_mean_map(persistence_n, runtime_rows, runtime_cols, patch_rows, patch_cols);
  result.contrast_patch = patch_mean_map(contrast_n, runtime_rows, runtime_cols, patch_rows, patch_cols);
  return result;
}

std::vector<float> dino_seed_patch_map(const std::vector<float>& spectrogram_db,
                                       int src_rows,
                                       int src_cols,
                                       int runtime_rows,
                                       int runtime_cols,
                                       int patch_rows,
                                       int patch_cols) {
  return dino_seed_patch_components(
      spectrogram_db,
      src_rows,
      src_cols,
      runtime_rows,
      runtime_cols,
      patch_rows,
      patch_cols)
      .seed_patch;
}

std::vector<float> raw_feature_energy_score_patch(const std::vector<float>& patch_features,
                                                  int patch_rows,
                                                  int patch_cols,
                                                  int feature_dim,
                                                  float positional_suppression = 0.0f) {
  const int patch_count = patch_rows * patch_cols;
  std::vector<float> raw_patch_score(static_cast<size_t>(patch_count), 0.0f);
  if (patch_count <= 0 || feature_dim <= 0 ||
      patch_features.size() != static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim)) {
    return raw_patch_score;
  }
  const auto energy_features = suppress_raw_dino_positional_features(
      patch_features,
      patch_rows,
      patch_cols,
      feature_dim,
      positional_suppression);
  for (int patch_index = 0; patch_index < patch_count; ++patch_index) {
    float mean_sq = 0.0f;
    for (int feature_index = 0; feature_index < feature_dim; ++feature_index) {
      const float value = energy_features[flat_index(feature_dim, patch_index, feature_index)];
      mean_sq += value * value;
    }
    mean_sq /= static_cast<float>(feature_dim);
    raw_patch_score[static_cast<size_t>(patch_index)] = std::sqrt(std::max(mean_sq, 1.0e-6f));
  }
  return normalize01_quantile(raw_patch_score, 5.0, 95.0);
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
  const int safe_row_offset = clamp_value(row_offset, 0, std::max(0, source_rows - 1));
  const int safe_col_offset = clamp_value(col_offset, 0, std::max(0, source_cols - 1));
  const int copy_rows = std::min(aligned_rows, std::max(0, source_rows - safe_row_offset));
  const int copy_cols = std::min(aligned_cols, std::max(0, source_cols - safe_col_offset));
  for (int row = 0; row < copy_rows; ++row) {
    const int dst_row = safe_row_offset + row;
    for (int col = 0; col < copy_cols; ++col) {
      const int dst_col = safe_col_offset + col;
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
                                                 int output_cols,
                                                 bool resized_full_chunk = false) {
  const auto source_canvas = resized_full_chunk
                                 ? resize_bilinear(aligned_map, aligned_rows, aligned_cols, source_rows, source_cols)
                                 : embed_aligned_map_in_source_canvas(
                                       aligned_map,
                                       aligned_rows,
                                       aligned_cols,
                                       source_rows,
                                       source_cols,
                                       row_offset,
                                       col_offset);
  return resize_bilinear(source_canvas, source_rows, source_cols, output_rows, output_cols);
}

std::vector<float> project_patch_map_to_output(const std::vector<float>& patch_map,
                                               int patch_rows,
                                               int patch_cols,
                                               int aligned_rows,
                                               int aligned_cols,
                                               int source_rows,
                                               int source_cols,
                                               int row_offset,
                                               int col_offset,
                                               int output_rows,
                                               int output_cols,
                                               bool resized_full_chunk = false) {
  if (patch_rows <= 0 || patch_cols <= 0 ||
      patch_map.size() != static_cast<size_t>(patch_rows) * static_cast<size_t>(patch_cols)) {
    return std::vector<float>(static_cast<size_t>(std::max(output_rows, 0)) * static_cast<size_t>(std::max(output_cols, 0)), 0.0f);
  }
  const auto aligned_map = resize_bilinear(patch_map, patch_rows, patch_cols, aligned_rows, aligned_cols);
  return project_aligned_map_to_output(
      aligned_map,
      aligned_rows,
      aligned_cols,
      source_rows,
      source_cols,
      row_offset,
      col_offset,
      output_rows,
      output_cols,
      resized_full_chunk);
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

struct LinearResizeSample {
  int index0 = 0;
  int index1 = 0;
  float t = 0.0f;
};

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

std::vector<float> positional_design_matrix(int patch_rows, int patch_cols) {
  constexpr float kPi = 3.14159265358979323846f;
  const int n = patch_rows * patch_cols;
  const int basis_dim = 16;
  std::vector<float> design(static_cast<size_t>(n) * static_cast<size_t>(basis_dim), 0.0f);
  for (int row = 0; row < patch_rows; ++row) {
    const float row_coord = patch_rows > 1 ? -1.0f + 2.0f * static_cast<float>(row) / static_cast<float>(patch_rows - 1) : 0.0f;
    for (int col = 0; col < patch_cols; ++col) {
      const float col_coord = patch_cols > 1 ? -1.0f + 2.0f * static_cast<float>(col) / static_cast<float>(patch_cols - 1) : 0.0f;
      const size_t base = static_cast<size_t>(flat_index(patch_cols, row, col)) * static_cast<size_t>(basis_dim);
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

std::vector<float> pca_project_features(const std::vector<float>& patch_features,
                                        int patch_count,
                                        int feature_dim) {
  if (patch_count <= 1 || feature_dim <= 0 || patch_features.size() != static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim)) {
    return patch_features;
  }
  const int out_dim = std::min({12, feature_dim, patch_count - 1});
  if (out_dim < 1) {
    return patch_features;
  }
  const auto x = vector_to_tensor_2d(patch_features, patch_count, feature_dim);
  const auto centered = x - x.mean(0, true);
  const auto svd = torch::linalg_svd(centered, false);
  const auto u = std::get<0>(svd).narrow(1, 0, out_dim);
  const auto s = std::get<1>(svd).narrow(0, 0, out_dim).unsqueeze(0);
  return tensor_to_vector_float(u * s);
}

std::vector<float> remove_positional_trend(const std::vector<float>& features,
                                           int patch_count,
                                           int feature_dim,
                                           int patch_rows,
                                           int patch_cols) {
  if (patch_count <= 0 || feature_dim <= 0) {
    return features;
  }
  const auto design = vector_to_tensor_2d(positional_design_matrix(patch_rows, patch_cols), patch_count, 16);
  const auto x = vector_to_tensor_2d(features, patch_count, feature_dim);
  const auto xtx = design.transpose(0, 1).matmul(design);
  const auto ridge = 1.0e-3f * torch::eye(xtx.size(0), torch::TensorOptions().dtype(torch::kFloat32));
  const auto beta = torch::linalg_solve(xtx + ridge, design.transpose(0, 1).matmul(x));
  return tensor_to_vector_float(x - design.matmul(beta));
}

std::vector<float> remove_position_correlated_components(const std::vector<float>& features,
                                                         int patch_count,
                                                         int feature_dim,
                                                         int patch_rows,
                                                         int patch_cols) {
  if (patch_count <= 1 || feature_dim <= 1) {
    return features;
  }
  const auto design = positional_design_matrix(patch_rows, patch_cols);
  std::vector<float> basis(static_cast<size_t>(patch_count) * 15U, 0.0f);
  for (int row = 0; row < patch_count; ++row) {
    const size_t src_base = static_cast<size_t>(row) * 16U;
    const size_t dst_base = static_cast<size_t>(row) * 15U;
    for (int col = 0; col < 15; ++col) {
      basis[dst_base + static_cast<size_t>(col)] = design[src_base + static_cast<size_t>(col + 1)];
    }
  }
  const auto x = vector_to_tensor_2d(features, patch_count, feature_dim);
  const auto centered = x - x.mean(0, true);
  const auto svd = torch::linalg_svd(centered, false);
  auto scores = std::get<0>(svd) * std::get<1>(svd).unsqueeze(0);
  const auto vh = std::get<2>(svd);
  const int components = static_cast<int>(scores.size(1));
  if (components <= 0) {
    return features;
  }
  const auto basis_values = basis;
  auto score_values = tensor_to_vector_float(scores);
  std::vector<uint8_t> keep(static_cast<size_t>(components), 1);
  std::vector<float> correlations(static_cast<size_t>(components), 0.0f);
  for (int comp = 0; comp < components; ++comp) {
    std::vector<float> y(static_cast<size_t>(patch_count), 0.0f);
    float mean = 0.0f;
    for (int row = 0; row < patch_count; ++row) {
      const float value = score_values[static_cast<size_t>(row) * static_cast<size_t>(components) + static_cast<size_t>(comp)];
      y[static_cast<size_t>(row)] = value;
      mean += value;
    }
    mean /= static_cast<float>(patch_count);
    float y_norm = 0.0f;
    for (float& value : y) {
      value -= mean;
      y_norm += value * value;
    }
    y_norm = std::sqrt(y_norm);
    if (y_norm < 1.0e-8f) {
      continue;
    }
    float max_corr = 0.0f;
    for (int basis_col = 0; basis_col < 15; ++basis_col) {
      float basis_mean = 0.0f;
      for (int row = 0; row < patch_count; ++row) {
        basis_mean += basis_values[static_cast<size_t>(row) * 15U + static_cast<size_t>(basis_col)];
      }
      basis_mean /= static_cast<float>(patch_count);
      float dot = 0.0f;
      float basis_norm = 0.0f;
      for (int row = 0; row < patch_count; ++row) {
        const float centered_basis = basis_values[static_cast<size_t>(row) * 15U + static_cast<size_t>(basis_col)] - basis_mean;
        dot += y[static_cast<size_t>(row)] * centered_basis;
        basis_norm += centered_basis * centered_basis;
      }
      basis_norm = std::sqrt(basis_norm);
      if (basis_norm < 1.0e-8f) {
        continue;
      }
      max_corr = std::max(max_corr, std::fabs(dot / std::max(y_norm * basis_norm, 1.0e-8f)));
    }
    correlations[static_cast<size_t>(comp)] = max_corr;
    if (max_corr >= 0.30f) {
      keep[static_cast<size_t>(comp)] = 0;
    }
  }
  if (std::none_of(keep.begin(), keep.end(), [](uint8_t value) { return value != 0; })) {
    const auto min_iter = std::min_element(correlations.begin(), correlations.end());
    if (min_iter != correlations.end()) {
      keep[static_cast<size_t>(std::distance(correlations.begin(), min_iter))] = 1;
    }
  }
  for (int comp = 0; comp < components; ++comp) {
    if (keep[static_cast<size_t>(comp)] == 0) {
      scores.index_put_({torch::indexing::Slice(), comp}, 0.0f);
    }
  }
  return tensor_to_vector_float(scores.matmul(vh));
}
constexpr float kHybridRawDinoPositionalDeweight = 0.75f;

std::vector<float> suppress_raw_dino_positional_features(const std::vector<float>& patch_features,
                                                         int patch_rows,
                                                         int patch_cols,
                                                         int feature_dim,
                                                         float suppression) {
  const int patch_count = patch_rows * patch_cols;
  if (suppression <= 0.0f || patch_count <= 0 || feature_dim <= 0 ||
      patch_features.size() != static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim)) {
    return patch_features;
  }
  const float clamped = clamp_value(suppression, 0.0f, 1.0f);
  auto suppressed = remove_positional_trend(patch_features, patch_count, feature_dim, patch_rows, patch_cols);
  suppressed = remove_position_correlated_components(suppressed, patch_count, feature_dim, patch_rows, patch_cols);
  if (clamped >= 1.0f) {
    return suppressed;
  }
  std::vector<float> blended(patch_features.size(), 0.0f);
  for (size_t index = 0; index < patch_features.size(); ++index) {
    blended[index] = (1.0f - clamped) * patch_features[index] + clamped * suppressed[index];
  }
  return blended;
}

std::vector<float> row_normalize_square_matrix(const std::vector<float>& input, int size) {
  std::vector<float> output(input.size(), 0.0f);
  for (int row = 0; row < size; ++row) {
    float row_sum = 0.0f;
    for (int col = 0; col < size; ++col) {
      row_sum += input[flat_index(size, row, col)];
    }
    row_sum = std::max(row_sum, 1.0e-6f);
    for (int col = 0; col < size; ++col) {
      output[flat_index(size, row, col)] = input[flat_index(size, row, col)] / row_sum;
    }
  }
  return output;
}

std::vector<float> feature_affinity_matrix(const std::vector<float>& features,
                                           int patch_count,
                                           int feature_dim,
                                           int k) {
  std::vector<float> normalized(features.size(), 0.0f);
  for (int row = 0; row < patch_count; ++row) {
    float norm = 0.0f;
    for (int col = 0; col < feature_dim; ++col) {
      const float value = features[flat_index(feature_dim, row, col)];
      norm += value * value;
    }
    norm = std::sqrt(std::max(norm, 1.0e-12f));
    for (int col = 0; col < feature_dim; ++col) {
      normalized[flat_index(feature_dim, row, col)] = features[flat_index(feature_dim, row, col)] / norm;
    }
  }
  const int neighbors = std::min(std::max(1, k), std::max(1, patch_count - 1));
  std::vector<std::vector<std::pair<float, int>>> top_neighbors(static_cast<size_t>(patch_count));
  std::vector<float> positive_distances;
  for (int row = 0; row < patch_count; ++row) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(static_cast<size_t>(patch_count - 1));
    for (int col = 0; col < patch_count; ++col) {
      if (col == row) {
        continue;
      }
      float dot = 0.0f;
      for (int feat = 0; feat < feature_dim; ++feat) {
        dot += normalized[flat_index(feature_dim, row, feat)] * normalized[flat_index(feature_dim, col, feat)];
      }
      const float distance = std::max(0.0f, 1.0f - dot);
      ranked.emplace_back(distance, col);
    }
    std::partial_sort(ranked.begin(), ranked.begin() + neighbors, ranked.end());
    top_neighbors[static_cast<size_t>(row)].assign(ranked.begin(), ranked.begin() + neighbors);
    for (int idx = 0; idx < neighbors; ++idx) {
      if (ranked[static_cast<size_t>(idx)].first > 0.0f) {
        positive_distances.push_back(ranked[static_cast<size_t>(idx)].first);
      }
    }
  }
  const float sigma = std::max(quantile_from_values(positive_distances, 0.50, 1.0f), 1.0e-3f);
  std::vector<float> affinity(static_cast<size_t>(patch_count) * static_cast<size_t>(patch_count), 0.0f);
  for (int row = 0; row < patch_count; ++row) {
    affinity[flat_index(patch_count, row, row)] = 1.0f;
    for (const auto& neighbor : top_neighbors[static_cast<size_t>(row)]) {
      const int col = neighbor.second;
      const float distance = neighbor.first;
      const float weight = std::exp(-((distance * distance) / (2.0f * sigma * sigma)));
      affinity[flat_index(patch_count, row, col)] = std::max(affinity[flat_index(patch_count, row, col)], weight);
      affinity[flat_index(patch_count, col, row)] = std::max(affinity[flat_index(patch_count, col, row)], weight);
    }
  }
  return affinity;
}

std::vector<float> mutual_knn_affinity(const std::vector<float>& affinity, int patch_count, int top_k, float keep_q) {
  const int clipped_top_k = clamp_value(top_k, 1, std::max(1, patch_count - 1));
  std::vector<uint8_t> knn_mask(static_cast<size_t>(patch_count) * static_cast<size_t>(patch_count), 0);
  for (int row = 0; row < patch_count; ++row) {
    std::vector<std::pair<float, int>> ranked;
    ranked.reserve(static_cast<size_t>(patch_count - 1));
    for (int col = 0; col < patch_count; ++col) {
      if (col == row) {
        continue;
      }
      ranked.emplace_back(affinity[flat_index(patch_count, row, col)], col);
    }
    std::partial_sort(ranked.begin(), ranked.begin() + clipped_top_k, ranked.end(), std::greater<>());
    for (int idx = 0; idx < clipped_top_k; ++idx) {
      knn_mask[flat_index(patch_count, row, ranked[static_cast<size_t>(idx)].second)] = 1;
    }
  }
  std::vector<float> values;
  for (int row = 0; row < patch_count; ++row) {
    for (int col = 0; col < patch_count; ++col) {
      if (row == col) {
        continue;
      }
      if (knn_mask[flat_index(patch_count, row, col)] != 0 && knn_mask[flat_index(patch_count, col, row)] != 0) {
        const float value = affinity[flat_index(patch_count, row, col)];
        if (value > 0.0f) {
          values.push_back(value);
        }
      }
    }
  }
  if (values.empty()) {
    auto output = affinity;
    for (int idx = 0; idx < patch_count; ++idx) {
      output[flat_index(patch_count, idx, idx)] = 1.0f;
    }
    return output;
  }
  const float keep_threshold = quantile_from_values(values, clamp_value(static_cast<double>(keep_q), 0.0, 0.95), 0.0f);
  std::vector<float> output(affinity.size(), 0.0f);
  for (int row = 0; row < patch_count; ++row) {
    output[flat_index(patch_count, row, row)] = 1.0f;
    for (int col = 0; col < patch_count; ++col) {
      if (row == col) {
        continue;
      }
      const bool mutual = knn_mask[flat_index(patch_count, row, col)] != 0 && knn_mask[flat_index(patch_count, col, row)] != 0;
      const float value = affinity[flat_index(patch_count, row, col)];
      output[flat_index(patch_count, row, col)] = (mutual && value >= keep_threshold) ? value : 0.0f;
    }
  }
  return output;
}

std::vector<float> inject_spatial_shortcuts(const std::vector<float>& local_aff,
                                            const std::vector<float>& full_aff,
                                            int patch_rows,
                                            int patch_cols,
                                            float spatial_weight) {
  auto output = local_aff;
  const int patch_count = patch_rows * patch_cols;
  for (int row = 0; row < patch_rows; ++row) {
    for (int col = 0; col < patch_cols; ++col) {
      const int idx0 = flat_index(patch_cols, row, col);
      for (int rr = std::max(0, row - 1); rr < std::min(patch_rows, row + 2); ++rr) {
        for (int cc = std::max(0, col - 1); cc < std::min(patch_cols, col + 2); ++cc) {
          if (rr == row && cc == col) {
            continue;
          }
          const int idx1 = flat_index(patch_cols, rr, cc);
          const float base_weight = full_aff[flat_index(patch_count, idx0, idx1)];
          if (base_weight <= 0.0f) {
            continue;
          }
          const float shortcut = spatial_weight * base_weight;
          output[flat_index(patch_count, idx0, idx1)] = std::max(output[flat_index(patch_count, idx0, idx1)], shortcut);
          output[flat_index(patch_count, idx1, idx0)] = std::max(output[flat_index(patch_count, idx1, idx0)], shortcut);
        }
      }
    }
  }
  for (int idx = 0; idx < patch_count; ++idx) {
    output[flat_index(patch_count, idx, idx)] = 1.0f;
  }
  return output;
}

std::vector<float> local_affinity_score_map(const std::vector<float>& local_aff, int patch_rows, int patch_cols) {
  const int patch_count = patch_rows * patch_cols;
  struct SparseEdge {
    int col = 0;
    float weight = 0.0f;
  };
  std::vector<std::vector<SparseEdge>> trans_rows(static_cast<size_t>(patch_count));
  std::vector<float> weighted_degree(static_cast<size_t>(patch_count), 0.0f);
  std::vector<float> two_hop_return(static_cast<size_t>(patch_count), 0.0f);
  std::vector<float> three_hop_return(static_cast<size_t>(patch_count), 0.0f);
  std::vector<float> spatial_strength(static_cast<size_t>(patch_count), 0.0f);
  for (int idx = 0; idx < patch_count; ++idx) {
    float degree = 0.0f;
    for (int col = 0; col < patch_count; ++col) {
      const float value = local_aff[flat_index(patch_count, idx, col)];
      degree += value;
      if (value > 0.0f) {
        trans_rows[static_cast<size_t>(idx)].push_back({col, value});
      }
    }
    degree = std::max(degree, 1.0e-6f);
    weighted_degree[static_cast<size_t>(idx)] = degree - 1.0f;
    for (auto& edge : trans_rows[static_cast<size_t>(idx)]) {
      edge.weight /= degree;
    }
  }
  auto edge_weight = [&trans_rows](int row, int target_col) {
    for (const auto& edge : trans_rows[static_cast<size_t>(row)]) {
      if (edge.col == target_col) {
        return edge.weight;
      }
    }
    return 0.0f;
  };
  for (int idx = 0; idx < patch_count; ++idx) {
    float two_hop = 0.0f;
    float three_hop = 0.0f;
    for (const auto& edge1 : trans_rows[static_cast<size_t>(idx)]) {
      const float back_to_idx = edge_weight(edge1.col, idx);
      if (back_to_idx > 0.0f) {
        two_hop += edge1.weight * back_to_idx;
      }
      for (const auto& edge2 : trans_rows[static_cast<size_t>(edge1.col)]) {
        const float close_cycle = edge_weight(edge2.col, idx);
        if (close_cycle > 0.0f) {
          three_hop += edge1.weight * edge2.weight * close_cycle;
        }
      }
    }
    two_hop_return[static_cast<size_t>(idx)] = two_hop;
    three_hop_return[static_cast<size_t>(idx)] = three_hop;
  }
  for (int row = 0; row < patch_rows; ++row) {
    for (int col = 0; col < patch_cols; ++col) {
      const int idx0 = flat_index(patch_cols, row, col);
      float sum = 0.0f;
      int count = 0;
      for (int rr = std::max(0, row - 1); rr < std::min(patch_rows, row + 2); ++rr) {
        for (int cc = std::max(0, col - 1); cc < std::min(patch_cols, col + 2); ++cc) {
          if (rr == row && cc == col) {
            continue;
          }
          sum += local_aff[flat_index(patch_count, idx0, flat_index(patch_cols, rr, cc))];
          ++count;
        }
      }
      spatial_strength[static_cast<size_t>(idx0)] = count > 0 ? sum / static_cast<float>(count) : 0.0f;
    }
  }
  const auto degree_n = normalize_vector01(weighted_degree);
  const auto two_hop_n = normalize_vector01(two_hop_return);
  const auto three_hop_n = normalize_vector01(three_hop_return);
  const auto spatial_n = normalize_vector01(spatial_strength);
  std::vector<float> score(static_cast<size_t>(patch_count), 0.0f);
  for (int idx = 0; idx < patch_count; ++idx) {
    score[static_cast<size_t>(idx)] = 0.35f * degree_n[static_cast<size_t>(idx)] +
                                      0.30f * two_hop_n[static_cast<size_t>(idx)] +
                                      0.20f * three_hop_n[static_cast<size_t>(idx)] +
                                      0.15f * spatial_n[static_cast<size_t>(idx)];
  }
  return normalize01_quantile(score, 5.0, 95.0);
}

std::pair<std::vector<int>, std::vector<uint8_t>> connected_affinity_components(const std::vector<float>& local_aff,
                                                                                 const std::vector<float>& support_map,
                                                                                 int patch_rows,
                                                                                 int patch_cols) {
  const int patch_count = patch_rows * patch_cols;
  std::vector<float> positive;
  for (int row = 0; row < patch_count; ++row) {
    for (int col = 0; col < patch_count; ++col) {
      if (row == col) {
        continue;
      }
      const float value = local_aff[flat_index(patch_count, row, col)];
      if (value > 0.0f) {
        positive.push_back(value);
      }
    }
  }
  const float edge_threshold = positive.empty() ? 0.0f : quantile_from_values(positive, 0.55, 0.0f);
  const float seed_threshold = quantile_from_values(support_map, 0.72, 1.0f);
  const float grow_threshold = quantile_from_values(support_map, 0.58, 1.0f);
  std::vector<uint8_t> active(static_cast<size_t>(patch_count), 0);
  std::vector<uint8_t> eligible(static_cast<size_t>(patch_count), 0);
  for (int idx = 0; idx < patch_count; ++idx) {
    active[static_cast<size_t>(idx)] = support_map[static_cast<size_t>(idx)] >= seed_threshold ? 1 : 0;
    eligible[static_cast<size_t>(idx)] = support_map[static_cast<size_t>(idx)] >= grow_threshold ? 1 : 0;
  }
  for (int iter = 0; iter < 2; ++iter) {
    auto updated = active;
    bool changed = false;
    for (int row = 0; row < patch_rows; ++row) {
      for (int col = 0; col < patch_cols; ++col) {
        const int idx0 = flat_index(patch_cols, row, col);
        if (active[static_cast<size_t>(idx0)] != 0 || eligible[static_cast<size_t>(idx0)] == 0) {
          continue;
        }
        bool linked = false;
        for (int rr = std::max(0, row - 1); rr < std::min(patch_rows, row + 2) && !linked; ++rr) {
          for (int cc = std::max(0, col - 1); cc < std::min(patch_cols, col + 2); ++cc) {
            if (rr == row && cc == col) {
              continue;
            }
            const int idx1 = flat_index(patch_cols, rr, cc);
            if (active[static_cast<size_t>(idx1)] != 0 && local_aff[flat_index(patch_count, idx0, idx1)] >= edge_threshold) {
              linked = true;
              break;
            }
          }
        }
        if (linked) {
          updated[static_cast<size_t>(idx0)] = 1;
          changed = true;
        }
      }
    }
    active = std::move(updated);
    if (!changed) {
      break;
    }
  }
  return {label_components(active, patch_rows, patch_cols).labels, active};
}

float component_boundary_affinity(const std::vector<float>& local_aff,
                                  const std::vector<uint8_t>& component_mask,
                                  int patch_rows,
                                  int patch_cols) {
  const int patch_count = patch_rows * patch_cols;
  std::vector<float> values;
  for (int row = 0; row < patch_rows; ++row) {
    for (int col = 0; col < patch_cols; ++col) {
      const int idx0 = flat_index(patch_cols, row, col);
      if (component_mask[static_cast<size_t>(idx0)] == 0) {
        continue;
      }
      for (int rr = std::max(0, row - 1); rr < std::min(patch_rows, row + 2); ++rr) {
        for (int cc = std::max(0, col - 1); cc < std::min(patch_cols, col + 2); ++cc) {
          if (rr == row && cc == col) {
            continue;
          }
          const int idx1 = flat_index(patch_cols, rr, cc);
          if (component_mask[static_cast<size_t>(idx1)] == 0) {
            values.push_back(local_aff[flat_index(patch_count, idx0, idx1)]);
          }
        }
      }
    }
  }
  return values.empty() ? 0.0f : std::accumulate(values.begin(), values.end(), 0.0f) / static_cast<float>(values.size());
}

float mask_smoothness(const std::vector<uint8_t>& mask, int patch_rows, int patch_cols) {
  float v_disagree = 0.0f;
  float h_disagree = 0.0f;
  int v_count = 0;
  int h_count = 0;
  for (int row = 1; row < patch_rows; ++row) {
    for (int col = 0; col < patch_cols; ++col) {
      v_disagree += mask[flat_index(patch_cols, row, col)] != mask[flat_index(patch_cols, row - 1, col)] ? 1.0f : 0.0f;
      ++v_count;
    }
  }
  for (int row = 0; row < patch_rows; ++row) {
    for (int col = 1; col < patch_cols; ++col) {
      h_disagree += mask[flat_index(patch_cols, row, col)] != mask[flat_index(patch_cols, row, col - 1)] ? 1.0f : 0.0f;
      ++h_count;
    }
  }
  const float edge_disagreement = 0.5f * ((v_count > 0 ? v_disagree / static_cast<float>(v_count) : 0.0f) +
                                          (h_count > 0 ? h_disagree / static_cast<float>(h_count) : 0.0f));
  return 1.0f - edge_disagreement;
}

std::vector<DinoComponentSummaryRow> component_summary_table(const std::vector<float>& local_aff,
                                                             const std::vector<int>& component_map,
                                                             const std::vector<float>& support_map,
                                                             const std::vector<float>& seed_norm,
                                                             int patch_rows,
                                                             int patch_cols) {
  constexpr float kGroupedComponentSeedWeight = 0.0f;
  const int patch_count = patch_rows * patch_cols;
  int max_label = 0;
  for (int label : component_map) {
    max_label = std::max(max_label, label);
  }
  std::vector<DinoComponentSummaryRow> rows;
  for (int label = 1; label <= max_label; ++label) {
    std::vector<int> indices;
    std::vector<uint8_t> mask(static_cast<size_t>(patch_count), 0);
    for (int idx = 0; idx < patch_count; ++idx) {
      if (component_map[static_cast<size_t>(idx)] == label) {
        indices.push_back(idx);
        mask[static_cast<size_t>(idx)] = 1;
      }
    }
    if (indices.empty()) {
      continue;
    }
    float internal_sum = 0.0f;
    int internal_count = 0;
    for (size_t ii = 0; ii < indices.size(); ++ii) {
      for (size_t jj = ii + 1; jj < indices.size(); ++jj) {
        internal_sum += local_aff[flat_index(patch_count, indices[ii], indices[jj])];
        ++internal_count;
      }
    }
    std::vector<float> support_values;
    float support_mean = 0.0f;
    float seed_mean = 0.0f;
    for (int idx : indices) {
      support_values.push_back(support_map[static_cast<size_t>(idx)]);
      support_mean += support_map[static_cast<size_t>(idx)];
      seed_mean += seed_norm[static_cast<size_t>(idx)];
    }
    DinoComponentSummaryRow row;
    row.cluster = label;
    row.size_fraction = static_cast<float>(indices.size()) / static_cast<float>(patch_count);
    row.support_mean = support_mean / static_cast<float>(indices.size());
    row.support_peak = quantile_from_values(support_values, 0.90, 0.0f);
    row.internal_aff = internal_count > 0 ? internal_sum / static_cast<float>(internal_count) : 0.0f;
    row.boundary_aff = component_boundary_affinity(local_aff, mask, patch_rows, patch_cols);
    row.seed_mean = seed_mean / static_cast<float>(indices.size());
    row.smoothness = mask_smoothness(mask, patch_rows, patch_cols);
    rows.push_back(row);
  }
  if (rows.empty()) {
    return rows;
  }
  std::vector<float> support_mean_vals;
  std::vector<float> support_peak_vals;
  std::vector<float> internal_vals;
  std::vector<float> boundary_gap_vals;
  std::vector<float> seed_vals;
  std::vector<float> smooth_vals;
  std::vector<float> size_vals;
  for (const auto& row : rows) {
    support_mean_vals.push_back(row.support_mean);
    support_peak_vals.push_back(row.support_peak);
    internal_vals.push_back(row.internal_aff);
    boundary_gap_vals.push_back(row.internal_aff - row.boundary_aff);
    seed_vals.push_back(row.seed_mean);
    smooth_vals.push_back(row.smoothness);
    size_vals.push_back(row.size_fraction);
  }
  const auto support_mean_n = normalize_vector01(support_mean_vals);
  const auto support_peak_n = normalize_vector01(support_peak_vals);
  const auto internal_n = normalize_vector01(internal_vals);
  const auto boundary_gap_n = normalize_vector01(boundary_gap_vals);
  const auto seed_n = normalize_vector01(seed_vals);
  const auto smooth_n = normalize_vector01(smooth_vals);
  for (size_t index = 0; index < rows.size(); ++index) {
    const float size_penalty = clamp_value((size_vals[index] - 0.30f) / 0.20f, 0.0f, 1.0f);
    rows[index].size_penalty = size_penalty;
    rows[index].combined_score = 0.35f * support_mean_n[index] +
                                 0.20f * support_peak_n[index] +
                                 0.20f * internal_n[index] +
                                 0.15f * boundary_gap_n[index] +
                                 kGroupedComponentSeedWeight * seed_n[index] +
                                 0.05f * smooth_n[index] -
                                 0.10f * size_penalty;
  }
  std::sort(rows.begin(), rows.end(), [](const DinoComponentSummaryRow& lhs, const DinoComponentSummaryRow& rhs) {
    return lhs.combined_score > rhs.combined_score;
  });
  return rows;
}

std::vector<int> select_signal_components(const std::vector<DinoComponentSummaryRow>& component_rows) {
  if (component_rows.empty()) {
    return {};
  }
  const float best_score = component_rows.front().combined_score;
  const float score_floor = std::max(0.35f, 0.72f * best_score);
  std::vector<int> selected;
  for (const auto& row : component_rows) {
    if (row.combined_score < score_floor) {
      continue;
    }
    if (row.size_fraction > 0.45f && row.combined_score < 0.95f * best_score) {
      continue;
    }
    selected.push_back(row.cluster);
    if (selected.size() >= 3) {
      break;
    }
  }
  if (selected.empty()) {
    selected.push_back(component_rows.front().cluster);
  }
  return selected;
}

std::vector<uint8_t> smooth_binary_label_map(const std::vector<uint8_t>& label_map,
                                             int patch_rows,
                                             int patch_cols,
                                             int iters,
                                             int min_component_size) {
  auto output = label_map;
  for (int iter = 0; iter < std::max(0, iters); ++iter) {
    std::vector<float> float_map(output.begin(), output.end());
    const auto avg = box_mean_2d(float_map, patch_rows, patch_cols, 1, 1);
    for (size_t index = 0; index < output.size(); ++index) {
      output[index] = avg[index] >= 0.5f ? 1 : 0;
    }
  }
  const auto labelled = label_components(output, patch_rows, patch_cols);
  if (!labelled.sizes.empty()) {
    std::vector<uint8_t> small_mask(output.size(), 0);
    for (size_t index = 0; index < labelled.labels.size(); ++index) {
      const int label = labelled.labels[index];
      if (label > 0 && labelled.sizes[static_cast<size_t>(label - 1)] < std::max(1, min_component_size)) {
        small_mask[index] = 1;
      }
    }
    std::vector<float> float_map(output.begin(), output.end());
    const auto neigh = box_mean_2d(float_map, patch_rows, patch_cols, 1, 1);
    for (size_t index = 0; index < output.size(); ++index) {
      if (small_mask[index] != 0) {
        output[index] = neigh[index] >= 0.5f ? 1 : 0;
      }
    }
  }
  return output;
}

GroupedDinoPatchResult grouped_dino_from_patch_features(const std::vector<float>& patch_features,
                                                        int patch_rows,
                                                        int patch_cols,
                                                        int feature_dim,
                                                        const std::vector<float>& seed_patch,
                                                        const ValidatorConfig& config,
                                                        bool verbose,
                                                        const char* debug_label) {
  constexpr float kGroupedScoreSeedWeight = 0.0f;
  GroupedDinoPatchResult result;
  const int patch_count = patch_rows * patch_cols;
  result.mask_patch.assign(static_cast<size_t>(patch_count), 0);
  result.score_patch.assign(static_cast<size_t>(patch_count), 0.0f);
  result.seed_norm_map.assign(static_cast<size_t>(patch_count), 0.0f);
  result.label_map_patch.assign(static_cast<size_t>(patch_count), 0);
  result.cluster_map.assign(static_cast<size_t>(patch_count), 0);
  result.support_map.assign(static_cast<size_t>(patch_count), 0.0f);
  result.cluster_quality_map.assign(static_cast<size_t>(patch_count), 0.0f);
  result.selected_support_map.assign(static_cast<size_t>(patch_count), 0.0f);
  if (patch_count <= 0 || feature_dim <= 0 || patch_features.size() != static_cast<size_t>(patch_count) * static_cast<size_t>(feature_dim)) {
    return result;
  }

  log_grouped_patch_memory(verbose, "start", debug_label, patch_rows, patch_cols, feature_dim);

  auto x = pca_project_features(patch_features, patch_count, feature_dim);
  const int reduced_dim = std::max(1, static_cast<int>(x.size()) / std::max(1, patch_count));
  log_grouped_patch_memory(verbose, "after_pca", debug_label, patch_rows, patch_cols, reduced_dim);
  x = remove_positional_trend(x, patch_count, reduced_dim, patch_rows, patch_cols);
  x = remove_position_correlated_components(x, patch_count, reduced_dim, patch_rows, patch_cols);
  std::vector<float> feature_mean(static_cast<size_t>(reduced_dim), 0.0f);
  for (int row = 0; row < patch_count; ++row) {
    for (int col = 0; col < reduced_dim; ++col) {
      feature_mean[static_cast<size_t>(col)] += x[flat_index(reduced_dim, row, col)];
    }
  }
  for (float& value : feature_mean) {
    value /= static_cast<float>(patch_count);
  }
  for (int row = 0; row < patch_count; ++row) {
    float norm = 0.0f;
    for (int col = 0; col < reduced_dim; ++col) {
      float& value = x[flat_index(reduced_dim, row, col)];
      value -= feature_mean[static_cast<size_t>(col)];
      norm += value * value;
    }
    norm = std::sqrt(std::max(norm, 1.0e-6f));
    for (int col = 0; col < reduced_dim; ++col) {
      x[flat_index(reduced_dim, row, col)] /= norm;
    }
  }

  const auto full_aff = feature_affinity_matrix(x, patch_count, reduced_dim, config.dino_group_k);
  log_grouped_patch_memory(verbose, "after_full_aff", debug_label, patch_rows, patch_cols, reduced_dim);
  auto local_aff = mutual_knn_affinity(full_aff, patch_count, config.dino_group_k, 0.40f);
  local_aff = inject_spatial_shortcuts(local_aff, full_aff, patch_rows, patch_cols, static_cast<float>(config.dino_group_spatial_weight));
  log_grouped_patch_memory(verbose, "after_local_aff", debug_label, patch_rows, patch_cols, reduced_dim);
  const auto seed_norm = normalize01_quantile(seed_patch, 5.0, 95.0);
  result.seed_norm_map = seed_norm;
  result.seed_persistence_map = seed_patch;
  result.seed_contrast_map = seed_patch;
  result.support_map = local_affinity_score_map(local_aff, patch_rows, patch_cols);
  log_grouped_patch_memory(verbose, "after_support_map", debug_label, patch_rows, patch_cols, reduced_dim);
  const auto components = connected_affinity_components(local_aff, result.support_map, patch_rows, patch_cols);
  result.cluster_map = components.first;
  result.active_mask_patch = components.second;
  const auto component_rows = component_summary_table(local_aff, result.cluster_map, result.support_map, seed_norm, patch_rows, patch_cols);
  const auto selected_components = select_signal_components(component_rows);
  if (!selected_components.empty()) {
    for (size_t index = 0; index < result.label_map_patch.size(); ++index) {
      result.label_map_patch[index] = std::find(selected_components.begin(), selected_components.end(), result.cluster_map[index]) != selected_components.end() ? 1 : 0;
    }
  } else {
    const float fallback_threshold = quantile_from_values(result.support_map, 0.80, 1.0f);
    for (size_t index = 0; index < result.label_map_patch.size(); ++index) {
      result.label_map_patch[index] = result.support_map[index] >= fallback_threshold ? 1 : 0;
      result.cluster_map[index] = result.label_map_patch[index] != 0 ? 1 : 0;
    }
  }
  result.label_map_pre_smooth_patch = result.label_map_patch;
  result.label_map_patch = smooth_binary_label_map(result.label_map_patch, patch_rows, patch_cols, 2, config.min_component_size);
  std::vector<float> support_selected(result.support_map.size(), 0.0f);
  for (size_t index = 0; index < support_selected.size(); ++index) {
    support_selected[index] = result.support_map[index] * static_cast<float>(result.label_map_patch[index]);
  }
  result.support_selected_raw_map = support_selected;
  result.selected_support_map = normalize01_quantile(box_mean_2d(support_selected, patch_rows, patch_cols, 1, 1), 5.0, 95.0);
  if (!component_rows.empty()) {
    std::vector<float> component_scores;
    for (const auto& row : component_rows) {
      component_scores.push_back(std::max(0.0f, row.combined_score));
    }
    const auto component_scores_n = normalize_vector01(component_scores);
    for (size_t row_index = 0; row_index < component_rows.size(); ++row_index) {
      for (size_t index = 0; index < result.cluster_map.size(); ++index) {
        if (result.cluster_map[index] == component_rows[row_index].cluster) {
          result.cluster_quality_map[index] = component_scores_n[row_index];
        }
      }
    }
  }
  for (size_t index = 0; index < result.score_patch.size(); ++index) {
    result.score_patch[index] = 0.70f * result.selected_support_map[index] +
                                0.20f * result.cluster_quality_map[index] +
                                kGroupedScoreSeedWeight * seed_norm[index];
  }
  result.score_patch = normalize01_quantile(result.score_patch, 5.0, 95.0);

  std::vector<float> candidate_scores;
  for (size_t index = 0; index < result.score_patch.size(); ++index) {
    if (result.label_map_patch[index] != 0) {
      candidate_scores.push_back(result.score_patch[index]);
    }
  }
  const double score_q = clamp_value(config.dino_group_score_q, 0.50, 0.95);
  if (candidate_scores.size() >= 4) {
    result.threshold = quantile_from_values(candidate_scores, score_q, 1.0f);
    for (size_t index = 0; index < result.mask_patch.size(); ++index) {
      result.mask_patch[index] = (result.label_map_patch[index] != 0 && result.score_patch[index] >= result.threshold) ? 1 : 0;
    }
    const float fraction = result.mask_patch.empty() ? 0.0f : static_cast<float>(std::count(result.mask_patch.begin(), result.mask_patch.end(), 1)) / static_cast<float>(result.mask_patch.size());
    if (fraction < 0.02f) {
      const float fallback_threshold = quantile_from_values(result.selected_support_map, 0.75, 1.0f);
      std::vector<uint8_t> fallback_mask(result.mask_patch.size(), 0);
      for (size_t index = 0; index < fallback_mask.size(); ++index) {
        fallback_mask[index] = result.selected_support_map[index] >= fallback_threshold ? 1 : 0;
      }
      std::vector<float> fallback_scores;
      for (size_t index = 0; index < fallback_mask.size(); ++index) {
        if (fallback_mask[index] != 0) {
          fallback_scores.push_back(result.score_patch[index]);
        }
      }
      result.threshold = fallback_scores.empty() ? quantile_from_values(result.score_patch, score_q, 1.0f)
                                                 : quantile_from_values(fallback_scores, std::min(score_q, 0.80), 1.0f);
      result.mask_patch = std::move(fallback_mask);
    }
  } else {
    result.threshold = quantile_from_values(result.score_patch, score_q, 1.0f);
    for (size_t index = 0; index < result.mask_patch.size(); ++index) {
      result.mask_patch[index] = result.score_patch[index] >= result.threshold ? 1 : 0;
    }
  }
  result.mask_patch = smooth_binary_label_map(result.mask_patch, patch_rows, patch_cols, 1, std::max(2, config.min_component_size / 2));
  log_grouped_patch_memory(verbose, "done", debug_label, patch_rows, patch_cols, reduced_dim);
  return result;
}

std::vector<float> structure_tensor_gate(const std::vector<float>& corrected_resized,
                                         int rows,
                                         int cols,
                                         const std::vector<uint8_t>& valid_mask) {
  const int bg_freq = std::max(9, 2 * std::max(1, rows / 24) + 1);
  const int bg_time = std::max(9, 2 * std::max(1, cols / 24) + 1);
  const auto background = box_mean_2d(corrected_resized,
                                      rows,
                                      cols,
                                      std::max(1, bg_freq / 2),
                                      std::max(1, bg_time / 2));

  std::vector<float> residual_db(corrected_resized.size(), 0.0f);
  for (size_t index = 0; index < residual_db.size(); ++index) {
    residual_db[index] = std::max(corrected_resized[index] - background[index], 0.0f);
  }
  const auto residual_n = normalize01_quantile(residual_db, 5.0, 99.0);

  const std::array<double, 3> scales = {0.8, 1.6, 3.2};
  std::vector<float> gate_max(corrected_resized.size(), 0.0f);
  for (double grad_sigma : scales) {
    const double integ_sigma = std::max(1.0, 1.8 * grad_sigma);
    const auto grad_f = gaussian_first_derivative_rows(residual_n, rows, cols, grad_sigma);
    const auto grad_t = gaussian_first_derivative_cols(residual_n, rows, cols, grad_sigma);

    std::vector<float> grad_ff(corrected_resized.size(), 0.0f);
    std::vector<float> grad_ft(corrected_resized.size(), 0.0f);
    std::vector<float> grad_tt(corrected_resized.size(), 0.0f);
    for (size_t index = 0; index < corrected_resized.size(); ++index) {
      grad_ff[index] = grad_f[index] * grad_f[index];
      grad_ft[index] = grad_f[index] * grad_t[index];
      grad_tt[index] = grad_t[index] * grad_t[index];
    }

    const auto j_ff = gaussian_blur(grad_ff, rows, cols, integ_sigma, integ_sigma);
    const auto j_ft = gaussian_blur(grad_ft, rows, cols, integ_sigma, integ_sigma);
    const auto j_tt = gaussian_blur(grad_tt, rows, cols, integ_sigma, integ_sigma);

    std::vector<float> coherence(corrected_resized.size(), 0.0f);
    std::vector<float> energy(corrected_resized.size(), 0.0f);
    for (size_t index = 0; index < corrected_resized.size(); ++index) {
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
    if (!valid_mask[index]) {
      gate_px[index] = 0.0f;
    }
  }
  return gate_px;
}

ComponentLabelling label_components(const std::vector<uint8_t>& mask, int rows, int cols) {
  ComponentLabelling result;
  result.labels.assign(mask.size(), 0);
  if (rows <= 0 || cols <= 0 || mask.empty()) {
    return result;
  }

  std::vector<int> parent(mask.size() + 1, 0);
  std::vector<int> rank(mask.size() + 1, 0);
  int next_label = 0;

  auto find_root = [&](int label) {
    int root = label;
    while (parent[static_cast<size_t>(root)] != root) {
      root = parent[static_cast<size_t>(root)];
    }
    while (parent[static_cast<size_t>(label)] != label) {
      const int next = parent[static_cast<size_t>(label)];
      parent[static_cast<size_t>(label)] = root;
      label = next;
    }
    return root;
  };

  auto unite = [&](int lhs, int rhs) {
    int root_lhs = find_root(lhs);
    int root_rhs = find_root(rhs);
    if (root_lhs == root_rhs) {
      return root_lhs;
    }
    if (rank[static_cast<size_t>(root_lhs)] < rank[static_cast<size_t>(root_rhs)]) {
      std::swap(root_lhs, root_rhs);
    }
    parent[static_cast<size_t>(root_rhs)] = root_lhs;
    if (rank[static_cast<size_t>(root_lhs)] == rank[static_cast<size_t>(root_rhs)]) {
      ++rank[static_cast<size_t>(root_lhs)];
    }
    return root_lhs;
  };

  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const size_t flat = flat_index(cols, row, col);
      if (!mask[flat]) {
        continue;
      }

      std::array<int, 4> neighbors = {0, 0, 0, 0};
      size_t neighbor_count = 0;
      auto maybe_add_neighbor = [&](int neighbor_row, int neighbor_col) {
        if (neighbor_row < 0 || neighbor_row >= rows || neighbor_col < 0 || neighbor_col >= cols) {
          return;
        }
        const int label = result.labels[flat_index(cols, neighbor_row, neighbor_col)];
        if (label > 0) {
          neighbors[neighbor_count++] = label;
        }
      };
      maybe_add_neighbor(row, col - 1);
      maybe_add_neighbor(row - 1, col - 1);
      maybe_add_neighbor(row - 1, col);
      maybe_add_neighbor(row - 1, col + 1);

      if (neighbor_count == 0) {
        ++next_label;
        parent[static_cast<size_t>(next_label)] = next_label;
        rank[static_cast<size_t>(next_label)] = 0;
        result.labels[flat] = next_label;
      } else {
        int assigned = neighbors[0];
        for (size_t index = 1; index < neighbor_count; ++index) {
          assigned = unite(assigned, neighbors[index]);
        }
        result.labels[flat] = assigned;
      }
    }
  }

  std::vector<int> root_to_compact(static_cast<size_t>(next_label + 1), 0);
  for (size_t flat = 0; flat < result.labels.size(); ++flat) {
    const int label = result.labels[flat];
    if (label <= 0) {
      continue;
    }
    const int root = find_root(label);
    int compact = root_to_compact[static_cast<size_t>(root)];
    if (compact == 0) {
      compact = static_cast<int>(result.sizes.size()) + 1;
      root_to_compact[static_cast<size_t>(root)] = compact;
      result.sizes.push_back(0);
    }
    result.labels[flat] = compact;
    ++result.sizes[static_cast<size_t>(compact - 1)];
  }
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

std::vector<uint8_t> keep_large_components(const std::vector<uint8_t>& mask,
                                           int rows,
                                           int cols,
                                           int min_size,
                                           int* kept_component_count = nullptr) {
  const auto labelled = label_components(mask, rows, cols);
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
    for (int size : labelled.sizes) {
      if (size >= std::max(1, min_size)) {
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

int component_envelope_area(const std::vector<uint8_t>& mask, int rows, int cols) {
  int area = 0;
  for (int col = 0; col < cols; ++col) {
    int min_row = rows;
    int max_row = -1;
    for (int row = 0; row < rows; ++row) {
      if (!mask[flat_index(cols, row, col)]) {
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
  for (size_t label_index = 0; label_index < stats.size(); ++label_index) {
    const auto& component = stats[label_index];
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
    box.parent_component_id = static_cast<int>(label_index) + 1;
    box.parent_component_ids = {box.parent_component_id};
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
    result.component_labels.assign(labelled.labels.begin(), labelled.labels.end());
    result.grouped_mask = result.seed_mask;
    const auto component_stats = build_component_stats(labelled);
    for (size_t label_index = 0; label_index < component_stats.size(); ++label_index) {
      const int component_id = static_cast<int>(label_index) + 1;
      const auto& component = component_stats[label_index];
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
      box.parent_component_id = component_id;
      box.parent_component_ids = {component_id};
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
  result.component_labels.assign(labelled.labels.begin(), labelled.labels.end());
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
    const int component_id = static_cast<int>(label_index) + 1;
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
    box.parent_component_id = component_id;
    box.parent_component_ids = {component_id};
    result.boxes.push_back(std::move(box));
  }

  for (size_t flat = 0; flat < labelled.labels.size(); ++flat) {
    const int label = labelled.labels[flat];
    if (label <= 0) {
      continue;
    }
    if (component_stats[static_cast<size_t>(label - 1)].keep) {
      result.grouped_mask[flat] = 1;
    }
  }
  return result;
}

void group_chunk_mask_regions(ChunkRetryResult& chunk,
                              bool filter_detection_mask,
                              int bridge_freq_px,
                              int bridge_time_px,
                              int min_component_size,
                              int min_freq_span_px,
                              int min_time_span_px,
                              float min_density,
                              float time_continuity_ratio) {
  if (!chunk.combined_score.empty() && !chunk.final_mask.empty() && !chunk.grouped_mask.empty() && !chunk.bridged_mask.empty()) {
    chunk.grouped_box_count = static_cast<int>(chunk.grouped_boxes.size());
    return;
  }
  const auto grouping = group_mask_regions(chunk.final_mask,
                                           chunk.combined_score,
                                           chunk.valid_mask,
                                           chunk.dst_rows,
                                           chunk.dst_cols,
                                           filter_detection_mask,
                                           bridge_freq_px,
                                           bridge_time_px,
                                           min_component_size,
                                           min_freq_span_px,
                                           min_time_span_px,
                                           min_density,
                                           time_continuity_ratio);
  chunk.bridged_mask = grouping.bridged_mask;
  chunk.grouped_mask = grouping.grouped_mask;
  chunk.grouped_boxes = grouping.boxes;
  chunk.grouped_box_count = static_cast<int>(chunk.grouped_boxes.size());
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
  if (!boxes_overlap(box_a, box_b)) {
    return false;
  }
  if (boxes_share_source_chunk(box_a, box_b)) {
    return false;
  }
  const bool carrier_burst_pair =
      (box_a.split_role == "persistent_carrier" && box_b.split_role == "transient_wideband_burst") ||
      (box_b.split_role == "persistent_carrier" && box_a.split_role == "transient_wideband_burst");
  return !carrier_burst_pair;
}

struct ChunkOwnershipRange {
  int row_start = 0;
  int row_stop = 0;
};

std::vector<ChunkOwnershipRange> compute_chunk_row_ownership_ranges(const std::vector<ChunkRetryResult>& chunk_results) {
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

DetectionBox merge_box_cluster(const std::vector<DetectionBox>& cluster) {
  DetectionBox merged;
  merged.freq_start = cluster.front().freq_start;
  merged.freq_stop = cluster.front().freq_stop;
  merged.time_start = cluster.front().time_start;
  merged.time_stop = cluster.front().time_stop;
  int weighted_area = 0;
  float weighted_score_sum = 0.0f;
  std::vector<std::string> split_roles;
  std::vector<int> source_chunk_indices;
  std::vector<int> parent_component_ids;
  for (const auto& box : cluster) {
    merged.freq_start = std::min(merged.freq_start, box.freq_start);
    merged.freq_stop = std::max(merged.freq_stop, box.freq_stop);
    merged.time_start = std::min(merged.time_start, box.time_start);
    merged.time_stop = std::max(merged.time_stop, box.time_stop);
    merged.filled_area += box.filled_area;
    weighted_area += std::max(1, box.filled_area);
    weighted_score_sum += box.score_mean * static_cast<float>(std::max(1, box.filled_area));
    merged.score_peak = std::max(merged.score_peak, box.score_peak);
    merged.split_applied = merged.split_applied || box.split_applied;
    split_roles.push_back(box.split_role);
    source_chunk_indices.insert(source_chunk_indices.end(), box.source_chunk_indices.begin(), box.source_chunk_indices.end());
    parent_component_ids.insert(parent_component_ids.end(), box.parent_component_ids.begin(), box.parent_component_ids.end());
  }
  const int bbox_area = std::max(1, (merged.freq_stop - merged.freq_start) * (merged.time_stop - merged.time_start));
  merged.density = static_cast<float>(merged.filled_area) / static_cast<float>(bbox_area);
  merged.score_mean = weighted_area > 0 ? weighted_score_sum / static_cast<float>(weighted_area) : 0.0f;
  std::sort(split_roles.begin(), split_roles.end());
  split_roles.erase(std::unique(split_roles.begin(), split_roles.end()), split_roles.end());
  merged.split_role = split_roles.size() == 1 ? split_roles.front() : "mixed";
  std::sort(source_chunk_indices.begin(), source_chunk_indices.end());
  source_chunk_indices.erase(std::unique(source_chunk_indices.begin(), source_chunk_indices.end()), source_chunk_indices.end());
  std::sort(parent_component_ids.begin(), parent_component_ids.end());
  parent_component_ids.erase(std::unique(parent_component_ids.begin(), parent_component_ids.end()), parent_component_ids.end());
  merged.source_chunk_indices = std::move(source_chunk_indices);
  merged.parent_component_ids = std::move(parent_component_ids);
  merged.source_box_count = static_cast<int>(cluster.size());
  return merged;
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

void rasterize_box_score_max(std::vector<float>& score_map,
                             int rows,
                             int cols,
                             const DetectionBox& box,
                             float value,
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
    auto* row_begin = score_map.data() + static_cast<std::ptrdiff_t>(flat_index(cols, row, time_start));
    auto* row_end = row_begin + static_cast<std::ptrdiff_t>(time_stop - time_start);
    for (auto* cursor = row_begin; cursor != row_end; ++cursor) {
      *cursor = std::max(*cursor, value);
    }
  }
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

std::vector<DetectionBox> merge_projected_boxes(const std::vector<DetectionBox>& projected_boxes) {
  if (projected_boxes.empty()) {
    return {};
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

GlobalMergedResult build_global_merged_result(const std::vector<ChunkRetryResult>& chunk_results,
                                              const ValidatorConfig& config,
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
    if (!chunk.grouped_mask_source.empty() && !chunk.combined_score.empty() && chunk.src_rows > 0 && chunk.src_cols > 0 && chunk.dst_rows > 0 && chunk.dst_cols > 0 && projected_rows > 0 && owned_projected_row_stop > owned_projected_row_start) {
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
    if (deferred_group_cols > 0 && chunk.grouped_boxes.empty() && !chunk.final_mask.empty() && !chunk.combined_score.empty() && chunk.dst_rows > 0 && chunk.dst_cols > 0 && projected_rows > 0 && owned_projected_row_stop > owned_projected_row_start) {
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
    for (size_t box_index = 0; box_index < chunk.grouped_boxes.size(); ++box_index) {
      const auto& box = chunk.grouped_boxes[box_index];
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
    auto* deferred_mask_row = deferred_group_mask.data() + static_cast<std::ptrdiff_t>(flat_index(deferred_group_cols, row, 0));
    std::fill(deferred_mask_row, deferred_mask_row + deferred_group_cols, static_cast<uint8_t>(0));
    auto* deferred_score_row = deferred_group_score.data() + static_cast<std::ptrdiff_t>(flat_index(deferred_group_cols, row, 0));
    std::fill(deferred_score_row, deferred_score_row + deferred_group_cols, 0.0f);
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
    const auto global_valid_mask = resize_row_valid_mask(source_valid_row_mask, global_rows, deferred_group_cols);
    const auto deferred_grouping = group_mask_regions(deferred_group_mask,
                                                      deferred_group_score,
                                                      global_valid_mask,
                                                      global_rows,
                                                      deferred_group_cols,
                                                      config.filter_detection_mask,
                                                      config.grouping_bridge_freq_px,
                                                      config.grouping_bridge_time_px,
                                                      config.grouping_min_component_size,
                                                      config.grouping_min_freq_span_px,
                                                      config.grouping_min_time_span_px,
                                                      static_cast<float>(config.grouping_min_density),
                                                      static_cast<float>(config.grouping_time_continuity_ratio));
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

HybridPostprocessResult run_residual_veto_hybrid(const std::vector<float>& hybrid_dino_contrib,
                                                 const std::vector<uint8_t>& valid_mask,
                                                 int rows,
                                                 int cols) {
  HybridPostprocessResult result;
  result.mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  if (hybrid_dino_contrib.size() != result.mask.size() || valid_mask.size() != result.mask.size()) {
    return result;
  }

  const auto& base_map = hybrid_dino_contrib;
  const auto base_norm = normalize01_masked_minmax(base_map, valid_mask);
  const auto envelope_map = normalize01_masked_minmax(gaussian_blur(base_norm, rows, cols, 6.0, 1.4), valid_mask);
  const auto base_blur = gaussian_blur(base_norm, rows, cols, 4.0, 1.0);

  std::vector<float> residual_abs(base_norm.size(), 0.0f);
  for (size_t index = 0; index < residual_abs.size(); ++index) {
    residual_abs[index] = std::fabs(base_norm[index] - base_blur[index]);
  }
  const auto residual_penalty = normalize01_masked_minmax(gaussian_blur(residual_abs, rows, cols, 2.0, 0.8), valid_mask);
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
    seed_mask[index] = (valid_mask[index] && keep_freq[index] >= result.seed_freq_threshold && keep_res[index] >= result.seed_res_threshold) ? 1 : 0;
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

ChunkInferenceArtifacts run_retry_chunk_inference(holoscan::ops::DinoTorchRuntime& runtime,
                                                  const holoscan::ops::DinoTorchRuntimeConfig& runtime_config,
                                                  const ValidatorConfig& config,
                                                  const ChunkPlanEntry& chunk,
                                                  const std::vector<float>& power_db,
                                                  const std::vector<float>& corrected_db,
                                                  const float* corrected_full_frame_device,
                                                  int full_rows,
                                                  int full_cols,
                                                  double resolution_hz,
                                                  const std::vector<uint8_t>& source_valid_row_mask,
                                                  ChunkGpuWorkspace& gpu_workspace,
                                                  StageProfiler* profiler,
                                                  bool keep_debug_artifacts,
                                                  bool verbose) {
  ChunkInferenceArtifacts artifacts;
  auto& result = artifacts.result;
  artifacts.keep_debug_artifacts = keep_debug_artifacts;
  result.chunk_index = chunk.chunk_index;
  result.row_start = chunk.row_start;
  result.row_stop = chunk.row_stop;
  result.src_rows = std::max(0, chunk.row_stop - chunk.row_start);
  result.src_cols = full_cols;
  result.dst_rows = config.input_height;
  result.dst_cols = config.input_width;
  result.freq_start_hz = chunk.freq_start_hz;
  result.freq_stop_hz = chunk.freq_stop_hz;
  result.span_hz = std::max(0.0, chunk.freq_stop_hz - chunk.freq_start_hz);

  if (result.src_rows <= 0 || result.src_cols <= 0) {
    return artifacts;
  }

  std::vector<float> power_chunk;
  std::vector<float> corrected_chunk;
  std::vector<uint8_t> source_chunk_valid_rows;
  const bool need_power_chunk = runtime_config.compute_power_score;
  {
    const size_t estimated_bytes = static_cast<size_t>(result.src_rows) * static_cast<size_t>(result.src_cols) * sizeof(float) * (need_power_chunk ? 2 : 1) +
                                   static_cast<size_t>(result.src_rows) * sizeof(uint8_t);
    ScopedStageProfile stage(profiler, "chunk_extract_preprocess", "chunk", result.chunk_index, estimated_bytes, verbose);
    if (need_power_chunk) {
      power_chunk = slice_rows(power_db, full_rows, full_cols, chunk.row_start, chunk.row_stop);
    }
    if (keep_debug_artifacts || corrected_full_frame_device == nullptr) {
      corrected_chunk = slice_rows(corrected_db, full_rows, full_cols, chunk.row_start, chunk.row_stop);
    }
    source_chunk_valid_rows.assign(static_cast<size_t>(result.src_rows), 1);
    for (int row = 0; row < result.src_rows; ++row) {
      const int src_row = chunk.row_start + row;
      if (src_row >= 0 && src_row < static_cast<int>(source_valid_row_mask.size())) {
        source_chunk_valid_rows[static_cast<size_t>(row)] = source_valid_row_mask[static_cast<size_t>(src_row)];
      }
    }
  }

  {
    const size_t estimated_bytes = static_cast<size_t>(result.dst_rows) * static_cast<size_t>(result.dst_cols) * sizeof(uint8_t);
    ScopedStageProfile stage(profiler, "chunk_valid_mask_resize", "chunk", result.chunk_index, estimated_bytes, verbose);
    if (keep_debug_artifacts) {
      result.valid_mask = resize_row_valid_mask(source_chunk_valid_rows, result.dst_rows, result.dst_cols);
    }
  }
  const int runtime_rows = std::max(config.patch_size,
                                    (std::max(config.patch_size, config.input_height) / std::max(1, config.patch_size)) * std::max(1, config.patch_size));
  const int runtime_cols = std::max(config.patch_size,
                                    (std::max(config.patch_size, config.input_width) / std::max(1, config.patch_size)) * std::max(1, config.patch_size));
  result.ignore_bins_per_side = 0;

  const size_t chunk_elements = static_cast<size_t>(result.src_rows) * static_cast<size_t>(result.src_cols);
  const size_t chunk_bytes = chunk_elements * sizeof(float);
  {
    const size_t estimated_bytes = chunk_bytes * (need_power_chunk ? 2 : 1);
    ScopedStageProfile stage(profiler, "chunk_gpu_upload", "chunk", result.chunk_index, estimated_bytes, verbose);
    gpu_workspace.ensure_capacity(chunk_elements);
    cudaError_t corrected_copy_status = cudaSuccess;
    if (corrected_full_frame_device == nullptr) {
      corrected_copy_status = cudaMemcpyAsync(gpu_workspace.corrected_chunk_device,
                                              corrected_chunk.data(),
                                              chunk_bytes,
                                              cudaMemcpyHostToDevice,
                                              gpu_workspace.stream);
    } else {
      copy_rows_device_to_device(corrected_full_frame_device,
                                 full_cols,
                                 chunk.row_start,
                                 result.src_rows,
                                 gpu_workspace.corrected_chunk_device,
                                 gpu_workspace.stream);
    }
    if ((need_power_chunk &&
         cudaMemcpyAsync(gpu_workspace.power_chunk_device, power_chunk.data(), chunk_bytes, cudaMemcpyHostToDevice, gpu_workspace.stream) != cudaSuccess) ||
        corrected_copy_status != cudaSuccess) {
      throw std::runtime_error("failed to upload chunk tensors for offline DINO validator");
    }
  }

  holoscan::ops::DinoTorchRuntimeInput runtime_input;
  runtime_input.src_rows = result.src_rows;
  runtime_input.src_cols = result.src_cols;
  runtime_input.dst_rows = runtime_rows;
  runtime_input.dst_cols = runtime_cols;
  runtime_input.patch_size = config.patch_size;
  runtime_input.cuda_stream = gpu_workspace.stream;
  runtime_input.resolution_hz = resolution_hz;
  runtime_input.span_hz = result.span_hz;
  runtime_input.power_db_device = need_power_chunk ? gpu_workspace.power_chunk_device : nullptr;
  runtime_input.corrected_db_device = gpu_workspace.corrected_chunk_device;

  auto chunk_runtime_config = runtime_config;
  chunk_runtime_config.ignore_sideband_hz = 0.0;
  chunk_runtime_config.return_pre_model_gray = keep_debug_artifacts;
  chunk_runtime_config.return_patch_features = keep_debug_artifacts;
  holoscan::ops::DinoTorchRuntimeResult runtime_result;
  {
    ScopedStageProfile stage(profiler, "chunk_torch_runtime", "chunk", result.chunk_index, chunk_bytes * 2, verbose);
    runtime_result = runtime.run(chunk_runtime_config, runtime_input);
  }
  record_timed_stage(profiler, "chunk_model_prep", "chunk", result.chunk_index, runtime_result.timing.model_prep_ms);
  record_timed_stage(profiler, "chunk_torch_forward", "chunk", result.chunk_index, runtime_result.timing.torch_forward_ms);
  record_timed_stage(profiler, "chunk_dino_score", "chunk", result.chunk_index, runtime_result.timing.dino_score_ms);
  record_timed_stage(profiler, "chunk_score_to_cpu", "chunk", result.chunk_index, runtime_result.timing.score_to_cpu_ms);

  if (!runtime_result.success) {
    throw std::runtime_error("chunk DINO runtime failed at " + runtime_result.error_stage + ": " + runtime_result.error_message + " (" + runtime_result.error_detail + ")");
  }
  if (runtime_result.score_map.size() != static_cast<size_t>(runtime_rows) * static_cast<size_t>(runtime_cols)) {
    throw std::runtime_error("unexpected chunk DINO score map size returned from runtime");
  }
  if (!runtime_result.pre_model_gray.empty() &&
      runtime_result.pre_model_gray.size() != static_cast<size_t>(runtime_result.aligned_rows) * static_cast<size_t>(runtime_result.aligned_cols)) {
    throw std::runtime_error("unexpected pre-model grayscale size returned from runtime");
  }

  result.dino_threshold = runtime_result.dino_threshold;
  result.runtime_final_threshold = runtime_result.final_threshold;
  if (keep_debug_artifacts && !runtime_result.pre_model_gray.empty()) {
    result.runtime_input_gray_rows = runtime_result.aligned_rows;
    result.runtime_input_gray_cols = runtime_result.aligned_cols;
    result.runtime_input_gray = runtime_result.pre_model_gray;
  }
  result.patch_rows = runtime_result.patch_rows;
  result.patch_cols = runtime_result.patch_cols;
  result.feature_dim = runtime_result.feature_dim;
  if (keep_debug_artifacts) {
    result.patch_features = runtime_result.patch_features;
  }
  const int runtime_row_offset = 0;
  const int runtime_col_offset = 0;
  std::vector<float> raw_aligned_score;
  std::vector<float> raw_dino_score_map;
  std::vector<float> deweighted_raw_dino_score_map;
  std::vector<float> deweighted_raw_dino_score_source;
  {
    const size_t estimated_bytes = static_cast<size_t>(result.dst_rows) * static_cast<size_t>(result.dst_cols) * sizeof(float) * 3;
    ScopedStageProfile stage(profiler, "chunk_score_projection", "chunk", result.chunk_index, estimated_bytes, verbose);
    if (runtime_rows == runtime_result.aligned_rows && runtime_cols == runtime_result.aligned_cols) {
      raw_aligned_score = std::move(runtime_result.score_map);
    } else {
      raw_aligned_score = resize_bilinear(runtime_result.score_map,
                                          runtime_rows,
                                          runtime_cols,
                                          runtime_result.aligned_rows,
                                          runtime_result.aligned_cols);
    }
    if (!runtime_result.patch_features.empty() && result.patch_rows > 0 && result.patch_cols > 0 && result.feature_dim > 0) {
      if (keep_debug_artifacts) {
        const auto raw_patch_score = raw_feature_energy_score_patch(runtime_result.patch_features,
                                                                    result.patch_rows,
                                                                    result.patch_cols,
                                                                    result.feature_dim);
        raw_dino_score_map = project_patch_map_to_output(raw_patch_score,
                                                         result.patch_rows,
                                                         result.patch_cols,
                                                         runtime_result.aligned_rows,
                                                         runtime_result.aligned_cols,
                                                         result.src_rows,
                                                         result.src_cols,
                                                         runtime_row_offset,
                                                         runtime_col_offset,
                                                         result.dst_rows,
                                                         result.dst_cols,
                                                         runtime_result.input_resized_to_target);
      }
      const auto deweighted_raw_patch_score = raw_feature_energy_score_patch(runtime_result.patch_features,
                                                                             result.patch_rows,
                                                                             result.patch_cols,
                                                                             result.feature_dim,
                                                                             kHybridRawDinoPositionalDeweight);
      deweighted_raw_dino_score_source = project_patch_map_to_output(deweighted_raw_patch_score,
                                                                     result.patch_rows,
                                                                     result.patch_cols,
                                                                     runtime_result.aligned_rows,
                                                                     runtime_result.aligned_cols,
                                                                     result.src_rows,
                                                                     result.src_cols,
                                                                     runtime_row_offset,
                                                                     runtime_col_offset,
                                                                     result.src_rows,
                                                                     result.src_cols,
                                                                     runtime_result.input_resized_to_target);
      if (keep_debug_artifacts) {
        deweighted_raw_dino_score_map = project_patch_map_to_output(deweighted_raw_patch_score,
                                                                    result.patch_rows,
                                                                    result.patch_cols,
                                                                    runtime_result.aligned_rows,
                                                                    runtime_result.aligned_cols,
                                                                    result.src_rows,
                                                                    result.src_cols,
                                                                    runtime_row_offset,
                                                                    runtime_col_offset,
                                                                    result.dst_rows,
                                                                    result.dst_cols,
                                                                    runtime_result.input_resized_to_target);
      }
    } else {
      if (keep_debug_artifacts) {
        raw_dino_score_map = project_aligned_map_to_output(raw_aligned_score,
                                                           runtime_result.aligned_rows,
                                                           runtime_result.aligned_cols,
                                                           result.src_rows,
                                                           result.src_cols,
                                                           runtime_row_offset,
                                                           runtime_col_offset,
                                                           result.dst_rows,
                                                           result.dst_cols,
                                                           runtime_result.input_resized_to_target);
      }
      deweighted_raw_dino_score_source = project_aligned_map_to_output(raw_aligned_score,
                                                                       runtime_result.aligned_rows,
                                                                       runtime_result.aligned_cols,
                                                                       result.src_rows,
                                                                       result.src_cols,
                                                                       runtime_row_offset,
                                                                       runtime_col_offset,
                                                                       result.src_rows,
                                                                       result.src_cols,
                                                                       runtime_result.input_resized_to_target);
      if (keep_debug_artifacts) {
        deweighted_raw_dino_score_map = project_aligned_map_to_output(raw_aligned_score,
                                                                      runtime_result.aligned_rows,
                                                                      runtime_result.aligned_cols,
                                                                      result.src_rows,
                                                                      result.src_cols,
                                                                      runtime_row_offset,
                                                                      runtime_col_offset,
                                                                      result.dst_rows,
                                                                      result.dst_cols,
                                                                      runtime_result.input_resized_to_target);
      }
    }
  }
  if (keep_debug_artifacts) {
    const size_t dst_elements = static_cast<size_t>(result.dst_rows) * static_cast<size_t>(result.dst_cols);
    const size_t patch_elements = static_cast<size_t>(std::max(result.patch_rows, 0)) * static_cast<size_t>(std::max(result.patch_cols, 0));
    result.raw_dino_score_map = raw_dino_score_map;
    result.raw_dino_score_deweighted_map = deweighted_raw_dino_score_map;
    result.dino_score_map = deweighted_raw_dino_score_map;
    result.grouped_seed_score_map.assign(dst_elements, 0.0f);
    result.grouped_seed_persistence_map.assign(dst_elements, 0.0f);
    result.grouped_seed_contrast_map.assign(dst_elements, 0.0f);
    result.grouped_support_map_exact.assign(patch_elements, 0.0f);
    result.grouped_active_mask_exact.assign(patch_elements, 0.0f);
    result.grouped_cluster_labels_exact.assign(patch_elements, 0.0f);
    result.grouped_selected_mask_pre_smooth_exact.assign(patch_elements, 0.0f);
    result.grouped_selected_mask_exact.assign(patch_elements, 0.0f);
    result.grouped_support_selected_raw_exact.assign(patch_elements, 0.0f);
    result.grouped_selected_support_map.assign(dst_elements, 0.0f);
    result.grouped_cluster_quality_map.assign(dst_elements, 0.0f);
    result.corrected_resized = resize_bilinear(corrected_chunk, result.src_rows, result.src_cols, result.dst_rows, result.dst_cols);
  }
  std::vector<uint8_t> source_chunk_valid_mask;
  std::vector<float> source_chunk_coherence_gate;
  std::vector<float> coherence_gate;
  {
    const size_t estimated_bytes = static_cast<size_t>(result.src_rows) * static_cast<size_t>(result.src_cols) * (sizeof(uint8_t) + sizeof(float)) +
                                   static_cast<size_t>(result.dst_rows) * static_cast<size_t>(result.dst_cols) * sizeof(float);
    ScopedStageProfile stage(profiler, "chunk_coherence", "chunk", result.chunk_index, estimated_bytes, verbose);
    source_chunk_coherence_gate = structure_tensor_gate_gpu(gpu_workspace.corrected_chunk_device,
                                                            result.src_rows,
                                                            result.src_cols,
                                                            source_chunk_valid_rows,
                                                            gpu_workspace.stream);
    source_chunk_valid_mask = resize_row_valid_mask(source_chunk_valid_rows, result.src_rows, result.src_cols);
    if (keep_debug_artifacts) {
      coherence_gate = resize_bilinear(source_chunk_coherence_gate,
                                       result.src_rows,
                                       result.src_cols,
                                       result.dst_rows,
                                       result.dst_cols);
    }
  }
  if (keep_debug_artifacts) {
    result.coherence_gate = coherence_gate;
  }

  artifacts.source_chunk_valid_mask = std::move(source_chunk_valid_mask);
  artifacts.grouped_dino_score_source = std::move(deweighted_raw_dino_score_source);
  artifacts.source_chunk_coherence_gate = std::move(source_chunk_coherence_gate);
  return artifacts;
}

std::vector<ChunkInferenceArtifacts> run_retry_chunk_inference_batch(holoscan::ops::DinoTorchRuntime& runtime,
                                                                     const holoscan::ops::DinoTorchRuntimeConfig& runtime_config,
                                                                     const ValidatorConfig& config,
                                                                     const std::vector<ChunkPlanEntry>& chunks,
                                                                     const std::vector<float>& power_db,
                                                                     const std::vector<float>& corrected_db,
                                                                     const float* corrected_full_frame_device,
                                                                     int full_rows,
                                                                     int full_cols,
                                                                     double resolution_hz,
                                                                     const std::vector<uint8_t>& source_valid_row_mask,
                                                                     ChunkGpuWorkspace& gpu_workspace,
                                                                     StageProfiler* profiler,
                                                                     bool verbose) {
  std::vector<ChunkInferenceArtifacts> artifacts_batch(chunks.size());
  if (chunks.empty()) {
    return artifacts_batch;
  }
  if (runtime_config.compute_power_score) {
    throw std::runtime_error("batched chunk inference currently requires compute_power_score=false");
  }

  const int uniform_rows = chunk_row_count(chunks.front());
  const int src_cols = full_cols;
  if (uniform_rows <= 0 || src_cols <= 0) {
    return artifacts_batch;
  }
  for (const auto& chunk : chunks) {
    if (chunk_row_count(chunk) != uniform_rows) {
      throw std::runtime_error("batched chunk inference requires uniform chunk row counts");
    }
  }

  const size_t batch_size = chunks.size();
  const size_t chunk_elements = static_cast<size_t>(uniform_rows) * static_cast<size_t>(src_cols);
  const size_t total_elements = batch_size * chunk_elements;
  std::vector<float> corrected_batch;
  if (corrected_full_frame_device == nullptr) {
    corrected_batch.assign(total_elements, 0.0f);
  }
  std::vector<std::vector<uint8_t>> source_chunk_valid_rows(batch_size, std::vector<uint8_t>(static_cast<size_t>(uniform_rows), 1));

  {
    const size_t estimated_bytes = total_elements * sizeof(float) + batch_size * static_cast<size_t>(uniform_rows) * sizeof(uint8_t);
    ScopedStageProfile stage(profiler, "chunk_extract_preprocess_batch", "run", -1, estimated_bytes, verbose);
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      auto& artifacts = artifacts_batch[batch_index];
      auto& result = artifacts.result;
      const auto& chunk = chunks[batch_index];
      artifacts.keep_debug_artifacts = false;
      result.chunk_index = chunk.chunk_index;
      result.row_start = chunk.row_start;
      result.row_stop = chunk.row_stop;
      result.src_rows = uniform_rows;
      result.src_cols = src_cols;
      result.dst_rows = config.input_height;
      result.dst_cols = config.input_width;
      result.freq_start_hz = chunk.freq_start_hz;
      result.freq_stop_hz = chunk.freq_stop_hz;
      result.span_hz = std::max(0.0, chunk.freq_stop_hz - chunk.freq_start_hz);

      if (corrected_full_frame_device == nullptr) {
        auto corrected_chunk = slice_rows(corrected_db, full_rows, full_cols, chunk.row_start, chunk.row_stop);
        std::copy(corrected_chunk.begin(),
                  corrected_chunk.end(),
                  corrected_batch.begin() + static_cast<std::ptrdiff_t>(batch_index * chunk_elements));
      }
      for (int row = 0; row < uniform_rows; ++row) {
        const int src_row = chunk.row_start + row;
        if (src_row >= 0 && src_row < static_cast<int>(source_valid_row_mask.size())) {
          source_chunk_valid_rows[batch_index][static_cast<size_t>(row)] = source_valid_row_mask[static_cast<size_t>(src_row)];
        }
      }
    }
  }

  const int runtime_rows = std::max(config.patch_size,
                                    (std::max(config.patch_size, config.input_height) / std::max(1, config.patch_size)) * std::max(1, config.patch_size));
  const int runtime_cols = std::max(config.patch_size,
                                    (std::max(config.patch_size, config.input_width) / std::max(1, config.patch_size)) * std::max(1, config.patch_size));

  {
    const size_t estimated_bytes = total_elements * sizeof(float);
    ScopedStageProfile stage(profiler, "chunk_gpu_upload_batch", "run", -1, estimated_bytes, verbose);
    gpu_workspace.ensure_capacity(total_elements);
    cudaError_t corrected_copy_status = cudaSuccess;
    if (corrected_full_frame_device == nullptr) {
      corrected_copy_status = cudaMemcpyAsync(gpu_workspace.corrected_chunk_device,
                                              corrected_batch.data(),
                                              total_elements * sizeof(float),
                                              cudaMemcpyHostToDevice,
                                              gpu_workspace.stream);
    } else {
      for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
        copy_rows_device_to_device(corrected_full_frame_device,
                                   full_cols,
                                   chunks[batch_index].row_start,
                                   uniform_rows,
                                   gpu_workspace.corrected_chunk_device + batch_index * chunk_elements,
                                   gpu_workspace.stream);
      }
    }
    if (corrected_copy_status != cudaSuccess) {
      throw std::runtime_error("failed to upload corrected batch tensors for offline DINO validator");
    }
  }

  holoscan::ops::DinoTorchRuntimeBatchInput runtime_input;
  runtime_input.batch_size = static_cast<int>(batch_size);
  runtime_input.src_rows = uniform_rows;
  runtime_input.src_cols = src_cols;
  runtime_input.dst_rows = runtime_rows;
  runtime_input.dst_cols = runtime_cols;
  runtime_input.patch_size = config.patch_size;
  runtime_input.cuda_stream = gpu_workspace.stream;
  runtime_input.resolution_hz = resolution_hz;
  runtime_input.span_hz = static_cast<double>(uniform_rows) * resolution_hz;
  runtime_input.corrected_db_batch_device = gpu_workspace.corrected_chunk_device;

  auto chunk_runtime_config = runtime_config;
  chunk_runtime_config.ignore_sideband_hz = 0.0;
  chunk_runtime_config.return_pre_model_gray = false;
  chunk_runtime_config.return_patch_features = true;
  chunk_runtime_config.return_final_mask_device = true;
  holoscan::ops::DinoTorchRuntimeBatchResult runtime_result;
  {
    ScopedStageProfile stage(profiler, "chunk_torch_runtime_batch", "run", -1, total_elements * sizeof(float), verbose);
    runtime_result = runtime.run_batch(chunk_runtime_config, runtime_input);
  }
  record_timed_stage(profiler, "chunk_model_prep_batch", "run", -1, runtime_result.timing.model_prep_ms);
  record_timed_stage(profiler, "chunk_torch_forward_batch", "run", -1, runtime_result.timing.torch_forward_ms);
  record_timed_stage(profiler, "chunk_dino_score_batch", "run", -1, runtime_result.timing.dino_score_ms);
  if (!runtime_result.success) {
    throw std::runtime_error("batched chunk DINO runtime failed at " + runtime_result.error_stage + ": " + runtime_result.error_message + " (" + runtime_result.error_detail + ")");
  }

  const size_t runtime_elements = static_cast<size_t>(runtime_rows) * static_cast<size_t>(runtime_cols);
  std::vector<uint8_t> source_valid_row_mask_batch(batch_size * static_cast<size_t>(uniform_rows), 1);
  for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    std::copy(source_chunk_valid_rows[batch_index].begin(),
              source_chunk_valid_rows[batch_index].end(),
              source_valid_row_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_index * static_cast<size_t>(uniform_rows)));
  }
  std::vector<uint8_t> source_valid_mask_batch(total_elements, 0);
  for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    for (int row = 0; row < uniform_rows; ++row) {
      const uint8_t row_valid = source_chunk_valid_rows[batch_index][static_cast<size_t>(row)];
      std::fill_n(source_valid_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_index * chunk_elements + static_cast<size_t>(row) * static_cast<size_t>(src_cols)),
                  src_cols,
                  row_valid);
    }
  }

  std::vector<float> source_coherence_batch;
  torch::Tensor source_coherence_batch_tensor;
  {
    const size_t estimated_bytes = total_elements * (sizeof(uint8_t) + sizeof(float));
    const auto coherence_start = std::chrono::steady_clock::now();
    source_coherence_batch_tensor = structure_tensor_gate_gpu_batch_tensor(gpu_workspace.corrected_chunk_device,
                                                                           static_cast<int>(batch_size),
                                                                           uniform_rows,
                                                                           src_cols,
                                                                           source_valid_row_mask_batch,
                                                                           gpu_workspace.stream);
    if (!source_coherence_batch_tensor.defined()) {
      source_coherence_batch = structure_tensor_gate_gpu_batch(gpu_workspace.corrected_chunk_device,
                                                               static_cast<int>(batch_size),
                                                               uniform_rows,
                                                               src_cols,
                                                               source_valid_row_mask_batch,
                                                               gpu_workspace.stream);
    }
    const double coherence_elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - coherence_start).count();
    const double per_chunk_elapsed_ms = batch_size > 0 ? coherence_elapsed_ms / static_cast<double>(batch_size) : 0.0;
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      record_timed_stage(profiler,
                         "chunk_coherence",
                         "chunk",
                         chunks[batch_index].chunk_index,
                         per_chunk_elapsed_ms);
    }
  }
  if (!source_coherence_batch_tensor.defined() && source_coherence_batch.size() != total_elements) {
    throw std::runtime_error("unexpected batched coherence map size returned from GPU helper");
  }

  std::vector<float> dino_score_source_batch;
  if (source_coherence_batch_tensor.defined()) {
    dino_score_source_batch.assign(batch_size * chunk_elements, 0.0f);
  }

  for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
    auto& artifacts = artifacts_batch[batch_index];
    auto& result = artifacts.result;
    result.ignore_bins_per_side = 0;
    result.dino_threshold = batch_index < runtime_result.dino_thresholds.size()
                                ? runtime_result.dino_thresholds[batch_index]
                                : chunk_runtime_config.pipeline_final_threshold;
    result.runtime_final_threshold = batch_index < runtime_result.final_thresholds.size()
                                         ? runtime_result.final_thresholds[batch_index]
                                         : result.dino_threshold;

    const int runtime_row_offset = 0;
    const int runtime_col_offset = 0;
    {
      const size_t estimated_bytes = static_cast<size_t>(result.src_rows) * static_cast<size_t>(result.src_cols) * sizeof(float);
      ScopedStageProfile stage(profiler, "chunk_score_projection", "chunk", result.chunk_index, estimated_bytes, verbose);
      if (runtime_result.patch_features_batch.empty() || runtime_result.patch_rows <= 0 || runtime_result.patch_cols <= 0 || runtime_result.feature_dim <= 0) {
        throw std::runtime_error("batched grouped score requires patch features");
      }
      const size_t patch_count = static_cast<size_t>(runtime_result.patch_rows) * static_cast<size_t>(runtime_result.patch_cols);
      const size_t feature_dim = static_cast<size_t>(runtime_result.feature_dim);
      const size_t sample_offset = batch_index * patch_count * feature_dim;
      if (sample_offset + patch_count * feature_dim > runtime_result.patch_features_batch.size()) {
        throw std::runtime_error("batched grouped score patch feature slice is out of range");
      }
      const std::vector<float> patch_features_sample(
          runtime_result.patch_features_batch.begin() + static_cast<std::ptrdiff_t>(sample_offset),
          runtime_result.patch_features_batch.begin() + static_cast<std::ptrdiff_t>(sample_offset + patch_count * feature_dim));
      const auto deweighted_raw_patch_score = raw_feature_energy_score_patch(patch_features_sample,
                                                                             runtime_result.patch_rows,
                                                                             runtime_result.patch_cols,
                                                                             runtime_result.feature_dim,
                                                                             kHybridRawDinoPositionalDeweight);
      auto deweighted_raw_score_source = project_patch_map_to_output(deweighted_raw_patch_score,
                                                                     runtime_result.patch_rows,
                                                                     runtime_result.patch_cols,
                                                                     runtime_result.aligned_rows,
                                                                     runtime_result.aligned_cols,
                                                                     result.src_rows,
                                                                     result.src_cols,
                                                                     runtime_row_offset,
                                                                     runtime_col_offset,
                                                                     result.src_rows,
                                                                     result.src_cols,
                                                                     runtime_result.input_resized_to_target);
      if (!dino_score_source_batch.empty()) {
        std::copy(deweighted_raw_score_source.begin(),
                  deweighted_raw_score_source.end(),
                  dino_score_source_batch.begin() + static_cast<std::ptrdiff_t>(batch_index * chunk_elements));
      } else {
        artifacts.grouped_dino_score_source = std::move(deweighted_raw_score_source);
      }
    }

    artifacts.source_chunk_valid_mask.assign(
        source_valid_mask_batch.begin() + static_cast<std::ptrdiff_t>(batch_index * chunk_elements),
        source_valid_mask_batch.begin() + static_cast<std::ptrdiff_t>((batch_index + 1) * chunk_elements));
    if (!source_coherence_batch.empty()) {
      artifacts.source_chunk_coherence_gate.assign(
        source_coherence_batch.begin() + static_cast<std::ptrdiff_t>(batch_index * chunk_elements),
        source_coherence_batch.begin() + static_cast<std::ptrdiff_t>((batch_index + 1) * chunk_elements));
    }
  }

  if (source_coherence_batch_tensor.defined()) {
    const size_t estimated_bytes = static_cast<size_t>(batch_size) * chunk_elements * (sizeof(float) * 2 + sizeof(uint8_t));
    ScopedStageProfile stage(profiler, "chunk_hybrid_support_batch", "run", -1, estimated_bytes, verbose);
    const auto pack_start = std::chrono::steady_clock::now();
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      if (artifacts_batch[batch_index].source_chunk_valid_mask.size() != chunk_elements) {
        throw std::runtime_error("batched hybrid device-input precompute requires dense grouped score and valid mask inputs");
      }
    }
    record_timed_stage(profiler,
                       "chunk_hybrid_norm_batch_cpu",
                       "run",
                       -1,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pack_start).count());

    const auto residual_veto_start = std::chrono::steady_clock::now();
    auto dino_score_source_tensor = torch::from_blob(dino_score_source_batch.data(),
                                                     {static_cast<int64_t>(batch_size), static_cast<int64_t>(uniform_rows), static_cast<int64_t>(src_cols)},
                                                     torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    auto hybrid_results = run_residual_veto_hybrid_gpu_batch_device_inputs(dino_score_source_tensor,
                                                                           source_coherence_batch_tensor,
                                                                           source_valid_mask_batch,
                                                                           static_cast<int>(batch_size),
                                                                           uniform_rows,
                                                                           src_cols,
                                                                           use_fp16_precision(config.hybrid_torch_dtype));
    record_timed_stage(profiler,
                       "chunk_hybrid_residual_veto_batch",
                       "run",
                       -1,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - residual_veto_start).count());
    for (size_t batch_index = 0; batch_index < batch_size; ++batch_index) {
      artifacts_batch[batch_index].precomputed_hybrid_result_source = std::move(hybrid_results[batch_index]);
      artifacts_batch[batch_index].has_precomputed_hybrid_result = true;
    }
  }
  return artifacts_batch;
}

ChunkRetryResult finalize_retry_chunk_postprocess(const ValidatorConfig& config,
                                                  ChunkInferenceArtifacts artifacts,
                                                  StageProfiler* profiler,
                                                  bool verbose) {
  auto& result = artifacts.result;
  std::vector<float> hybrid_contrib_source;
  std::vector<float> hybrid_contrib;
  HybridPostprocessResult hybrid_result_source;
  if (artifacts.has_precomputed_hybrid_result && !artifacts.keep_debug_artifacts) {
    hybrid_result_source = std::move(artifacts.precomputed_hybrid_result_source);
  } else {
    const size_t estimated_bytes = static_cast<size_t>(result.src_rows) * static_cast<size_t>(result.src_cols) * sizeof(float) * 4 +
                                   (artifacts.keep_debug_artifacts ? static_cast<size_t>(result.dst_rows) * static_cast<size_t>(result.dst_cols) * sizeof(float) : 0);
    ScopedStageProfile stage(profiler, "chunk_hybrid_support", "chunk", result.chunk_index, estimated_bytes, verbose);
    const auto dino_norm_start = std::chrono::steady_clock::now();
    const auto& hybrid_dino_source = artifacts.grouped_dino_score_source;
    auto dino_score_norm_source = normalize01_quantile(hybrid_dino_source, 5.0, 95.0);
    record_timed_stage(profiler,
                       "chunk_hybrid_dino_norm",
                       "chunk",
                       result.chunk_index,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dino_norm_start).count());

    const auto coherence_norm_start = std::chrono::steady_clock::now();
    const auto coherence_gate_norm_source = normalize01_quantile(artifacts.source_chunk_coherence_gate, 5.0, 99.0);
    record_timed_stage(profiler,
                       "chunk_hybrid_coherence_norm",
                       "chunk",
                       result.chunk_index,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - coherence_norm_start).count());

    const auto multiply_start = std::chrono::steady_clock::now();
    hybrid_contrib_source.assign(hybrid_dino_source.size(), 0.0f);
    for (size_t index = 0; index < hybrid_contrib_source.size(); ++index) {
      hybrid_contrib_source[index] = dino_score_norm_source[index] * coherence_gate_norm_source[index];
    }
    record_timed_stage(profiler,
                       "chunk_hybrid_contrib_multiply",
                       "chunk",
                       result.chunk_index,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - multiply_start).count());

    if (artifacts.keep_debug_artifacts) {
      hybrid_contrib = resize_bilinear(hybrid_contrib_source,
                                       result.src_rows,
                                       result.src_cols,
                                       result.dst_rows,
                                       result.dst_cols);
    }

    const auto residual_veto_start = std::chrono::steady_clock::now();
    hybrid_result_source = run_residual_veto_hybrid_gpu(hybrid_contrib_source,
                                                        artifacts.source_chunk_valid_mask,
                                                        result.src_rows,
                                                        result.src_cols,
                                                        use_fp16_precision(config.hybrid_torch_dtype));
    record_timed_stage(profiler,
                       "chunk_hybrid_residual_veto",
                       "chunk",
                       result.chunk_index,
                       std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - residual_veto_start).count());
  }
  if (artifacts.keep_debug_artifacts) {
    result.hybrid_contrib = hybrid_contrib;
  }

  result.seed_freq_threshold = hybrid_result_source.seed_freq_threshold;
  result.seed_res_threshold = hybrid_result_source.seed_res_threshold;
  result.combined_threshold = hybrid_result_source.combined_threshold;
  result.final_fraction = hybrid_result_source.final_fraction;
  result.connected_fraction = hybrid_result_source.connected_fraction;
  result.component_count = hybrid_result_source.component_count;
  result.final_mask_source = hybrid_result_source.mask;
  result.final_mask = resize_mask_nearest(hybrid_result_source.mask,
                                          result.src_rows,
                                          result.src_cols,
                                          result.dst_rows,
                                          result.dst_cols);
  result.combined_score = resize_bilinear(hybrid_result_source.combined_score,
                                          result.src_rows,
                                          result.src_cols,
                                          result.dst_rows,
                                          result.dst_cols);
  if (!artifacts.keep_debug_artifacts) {
    {
      const size_t estimated_bytes = static_cast<size_t>(result.src_rows) * static_cast<size_t>(result.src_cols) * (sizeof(uint8_t) + sizeof(int)) +
                                     sizeof(DetectionBox) * 8;
      ScopedStageProfile stage(profiler, "chunk_group_boxes_fast", "chunk", result.chunk_index, estimated_bytes, verbose);
      result.grouped_boxes = group_boxes_fast_only(hybrid_result_source.mask,
                                                   hybrid_result_source.combined_score,
                                                   artifacts.source_chunk_valid_mask,
                                                   result.src_rows,
                                                   result.src_cols,
                                                   config.grouping_bridge_freq_px,
                                                   config.grouping_bridge_time_px,
                                                   config.grouping_min_component_size,
                                                   config.grouping_min_freq_span_px,
                                                   config.grouping_min_time_span_px,
                                                   static_cast<float>(config.grouping_min_density));
    }
    for (auto& box : result.grouped_boxes) {
      box = scale_box_to_shape(box,
                               result.src_rows,
                               result.src_cols,
                               result.dst_rows,
                               result.dst_cols);
    }
    result.grouped_box_count = static_cast<int>(result.grouped_boxes.size());
    result.grouped_mask_source = hybrid_result_source.mask;
    result.grouped_mask = resize_mask_nearest(hybrid_result_source.mask,
                                              result.src_rows,
                                              result.src_cols,
                                              result.dst_rows,
                                              result.dst_cols);
    result.bridged_mask.clear();
    return result;
  }
  GroupingResult grouping_source;
  {
    const size_t estimated_bytes = static_cast<size_t>(result.src_rows) * static_cast<size_t>(result.src_cols) * (sizeof(uint8_t) * 3 + sizeof(int)) +
                                   sizeof(DetectionBox) * 8;
    ScopedStageProfile stage(profiler, "chunk_group_mask_regions", "chunk", result.chunk_index, estimated_bytes, verbose);
    grouping_source = group_mask_regions(hybrid_result_source.mask,
                                         hybrid_result_source.combined_score,
                                         artifacts.source_chunk_valid_mask,
                                         result.src_rows,
                                         result.src_cols,
                                         config.filter_detection_mask,
                                         config.grouping_bridge_freq_px,
                                         config.grouping_bridge_time_px,
                                         config.grouping_min_component_size,
                                         config.grouping_min_freq_span_px,
                                         config.grouping_min_time_span_px,
                                         static_cast<float>(config.grouping_min_density),
                                         static_cast<float>(config.grouping_time_continuity_ratio));
  }
  result.grouped_mask_source = grouping_source.grouped_mask;
  result.grouped_mask = resize_mask_nearest(grouping_source.grouped_mask,
                                            result.src_rows,
                                            result.src_cols,
                                            result.dst_rows,
                                            result.dst_cols);
  if (artifacts.keep_debug_artifacts) {
    result.bridged_mask = resize_mask_nearest(grouping_source.bridged_mask,
                                              result.src_rows,
                                              result.src_cols,
                                              result.dst_rows,
                                              result.dst_cols);
    result.grouped_boxes.clear();
    result.grouped_boxes.reserve(grouping_source.boxes.size());
    for (const auto& source_box : grouping_source.boxes) {
      result.grouped_boxes.push_back(scale_box_to_shape(source_box,
                                                        result.src_rows,
                                                        result.src_cols,
                                                        result.dst_rows,
                                                        result.dst_cols));
    }
    result.grouped_box_count = static_cast<int>(result.grouped_boxes.size());
  } else {
    result.grouped_boxes.clear();
    result.grouped_boxes.reserve(grouping_source.boxes.size());
    for (const auto& source_box : grouping_source.boxes) {
      result.grouped_boxes.push_back(scale_box_to_shape(source_box,
                                                        result.src_rows,
                                                        result.src_cols,
                                                        result.dst_rows,
                                                        result.dst_cols));
    }
    result.grouped_box_count = static_cast<int>(result.grouped_boxes.size());
  }
  return result;
}

void discard_non_debug_chunk_payload(ChunkRetryResult& chunk_result) {
  chunk_result.runtime_input_gray.clear();
  chunk_result.runtime_input_gray.shrink_to_fit();
  chunk_result.patch_features.clear();
  chunk_result.patch_features.shrink_to_fit();
  chunk_result.corrected_resized.clear();
  chunk_result.corrected_resized.shrink_to_fit();
  chunk_result.raw_dino_score_map.clear();
  chunk_result.raw_dino_score_map.shrink_to_fit();
  chunk_result.dino_score_map.clear();
  chunk_result.dino_score_map.shrink_to_fit();
  chunk_result.coherence_gate.clear();
  chunk_result.coherence_gate.shrink_to_fit();
  chunk_result.hybrid_contrib.clear();
  chunk_result.hybrid_contrib.shrink_to_fit();
  chunk_result.valid_mask.clear();
  chunk_result.valid_mask.shrink_to_fit();
  chunk_result.bridged_mask.clear();
  chunk_result.bridged_mask.shrink_to_fit();
}

std::vector<uint8_t> mask_to_u8(const std::vector<uint8_t>& mask) {
  std::vector<uint8_t> image(mask.size(), 0);
  for (size_t index = 0; index < mask.size(); ++index) {
    image[index] = mask[index] ? 255 : 0;
  }
  return image;
}

std::string json_escape(const std::string& text) {
  std::string escaped;
  escaped.reserve(text.size());
  for (char ch : text) {
    switch (ch) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      default: escaped += ch; break;
    }
  }
  return escaped;
}

size_t parse_status_kb_value(std::string_view line, std::string_view key) {
  if (line.size() < key.size() || line.substr(0, key.size()) != key) {
    return 0;
  }
  size_t value = 0;
  bool seen_digit = false;
  for (char ch : line) {
    if (ch >= '0' && ch <= '9') {
      value = value * 10 + static_cast<size_t>(ch - '0');
      seen_digit = true;
    } else if (seen_digit) {
      break;
    }
  }
  return value;
}

MemorySnapshot capture_memory_snapshot() {
  MemorySnapshot snapshot;
  std::ifstream status("/proc/self/status", std::ios::binary);
  std::string line;
  while (std::getline(status, line)) {
    if (snapshot.rss_kb == 0) {
      snapshot.rss_kb = parse_status_kb_value(line, "VmRSS:");
    }
    if (snapshot.hwm_kb == 0) {
      snapshot.hwm_kb = parse_status_kb_value(line, "VmHWM:");
    }
    if (snapshot.rss_kb != 0 && snapshot.hwm_kb != 0) {
      break;
    }
  }
  return snapshot;
}

MemorySnapshot ScopedStageProfile::capture_memory_snapshot() {
  return ::capture_memory_snapshot();
}

std::string format_bytes_json(size_t bytes) {
  return std::to_string(bytes);
}

bool is_debug_phase_reassigned_stage(const StageProfileEntry& entry) {
  return entry.scope == "run" &&
         (entry.stage == "load_tensor" || entry.stage == "artifact_serialization");
}

std::vector<StageProfileEntry> collect_phase_entries(const StageProfiler& full_pass_profiler,
                                                     const StageProfiler& debug_rerun_profiler,
                                                     bool debug_phase) {
  std::vector<StageProfileEntry> phase_entries;
  const auto& full_entries = full_pass_profiler.entries();
  const auto& debug_entries = debug_rerun_profiler.entries();
  phase_entries.reserve(full_entries.size() + debug_entries.size());
  for (const auto& entry : full_entries) {
    const bool reassign_to_debug = is_debug_phase_reassigned_stage(entry);
    if ((debug_phase && reassign_to_debug) || (!debug_phase && !reassign_to_debug)) {
      phase_entries.push_back(entry);
    }
  }
  if (debug_phase) {
    phase_entries.insert(phase_entries.end(), debug_entries.begin(), debug_entries.end());
  }
  return phase_entries;
}

void write_stage_profile_object_json(std::ostream& out,
                                     const std::vector<StageProfileEntry>& entries,
                                     const std::vector<StageAggregateEntry>& aggregates,
                                     const std::string& indent,
                                     const std::string& entry_phase = std::string()) {
  out << indent << "\"entry_count\": " << entries.size() << ",\n";
  out << indent << "\"entries\": [\n";
  for (size_t index = 0; index < entries.size(); ++index) {
    const auto& entry = entries[index];
    out << indent << "  {\"stage\": \"" << json_escape(entry.stage)
        << "\", \"scope\": \"" << json_escape(entry.scope)
        << "\", \"chunk_index\": " << entry.chunk_index
        << ", \"elapsed_ms\": " << entry.elapsed_ms
        << ", \"rss_before_kb\": " << entry.rss_before_kb
        << ", \"rss_after_kb\": " << entry.rss_after_kb
        << ", \"rss_delta_kb\": " << entry.rss_delta_kb
        << ", \"hwm_before_kb\": " << entry.hwm_before_kb
        << ", \"hwm_after_kb\": " << entry.hwm_after_kb
        << ", \"hwm_delta_kb\": " << entry.hwm_delta_kb
        << ", \"component_estimated_bytes\": " << format_bytes_json(entry.component_estimated_bytes)
        << ", \"failed\": " << (entry.failed ? "true" : "false");
    if (!entry_phase.empty()) {
      out << ", \"phase\": \"" << json_escape(entry_phase) << "\"";
    }
    out << "}";
    if (index + 1 != entries.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << indent << "],\n";
  out << indent << "\"aggregates\": [\n";
  for (size_t index = 0; index < aggregates.size(); ++index) {
    const auto& entry = aggregates[index];
    out << indent << "  {\"stage\": \"" << json_escape(entry.stage)
        << "\", \"scope\": \"" << json_escape(entry.scope)
        << "\", \"count\": " << entry.count
        << ", \"failure_count\": " << entry.failure_count
        << ", \"total_ms\": " << entry.total_ms
        << ", \"mean_ms\": " << entry.mean_ms
        << ", \"max_ms\": " << entry.max_ms
        << ", \"max_rss_after_kb\": " << entry.max_rss_after_kb
        << ", \"max_hwm_after_kb\": " << entry.max_hwm_after_kb
        << ", \"max_component_estimated_bytes\": " << format_bytes_json(entry.max_component_estimated_bytes);
    if (!entry_phase.empty()) {
      out << ", \"phase\": \"" << json_escape(entry_phase) << "\"";
    }
    out << "}";
    if (index + 1 != aggregates.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << indent << "]\n";
}

void write_stage_profile_json(const std::filesystem::path& path,
                              const StageProfiler& full_pass_profiler,
                              const StageProfiler& debug_rerun_profiler) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open stage profile output");
  }
  const auto& full_entries = full_pass_profiler.entries();
  const auto& debug_entries = debug_rerun_profiler.entries();
  const auto full_phase_entries = collect_phase_entries(full_pass_profiler, debug_rerun_profiler, false);
  const auto debug_phase_entries = collect_phase_entries(full_pass_profiler, debug_rerun_profiler, true);
  std::vector<StageProfileEntry> merged_entries;
  merged_entries.reserve(full_entries.size() + debug_entries.size());
  merged_entries.insert(merged_entries.end(), full_entries.begin(), full_entries.end());
  merged_entries.insert(merged_entries.end(), debug_entries.begin(), debug_entries.end());
  const auto merged_aggregates = aggregate_stage_entries(merged_entries);
  const auto full_aggregates = aggregate_stage_entries(full_phase_entries);
  const auto debug_aggregates = aggregate_stage_entries(debug_phase_entries);
  out << std::fixed << std::setprecision(6);
  out << "{\n";
  write_stage_profile_object_json(out, merged_entries, merged_aggregates, "  ");
  out << ",\n  \"phases\": {\n";
  out << "    \"full_pass\": {\n";
  write_stage_profile_object_json(out, full_phase_entries, full_aggregates, "      ", "full_pass");
  out << "    },\n";
  out << "    \"debug_rerun\": {\n";
  write_stage_profile_object_json(out, debug_phase_entries, debug_aggregates, "      ", "debug_rerun");
  out << "    }\n";
  out << "  }\n";
  out << "}\n";
}

void print_stage_hotspots_from_aggregates(const std::vector<StageAggregateEntry>& aggregates,
                                          std::ostream& out,
                                          size_t limit,
                                          std::string_view indent) {
  const size_t count = std::min(limit, aggregates.size());
  for (size_t index = 0; index < count; ++index) {
    const auto& entry = aggregates[index];
    out << indent << "[" << (index + 1) << "] scope=" << entry.scope
        << " stage=" << entry.stage
        << " total_ms=" << std::fixed << std::setprecision(3) << entry.total_ms
        << " mean_ms=" << entry.mean_ms
        << " max_ms=" << entry.max_ms
        << " max_hwm_kb=" << entry.max_hwm_after_kb
        << " max_component_estimated_bytes=" << entry.max_component_estimated_bytes
        << " count=" << entry.count;
    if (entry.failure_count > 0) {
      out << " failures=" << entry.failure_count;
    }
    out << "\n";
  }
}

void print_stage_phase_summary(const std::vector<StageProfileEntry>& phase_entries,
                               std::ostream& out,
                               size_t limit,
                               std::string_view label) {
  const auto aggregates = aggregate_stage_entries(phase_entries);
  out << "    " << label << ":\n";
  if (aggregates.empty()) {
    out << "      (no entries)\n";
    return;
  }
  for (const std::string_view scope_name : {std::string_view("run"), std::string_view("chunk")}) {
    std::vector<StageAggregateEntry> scoped;
    for (const auto& aggregate : aggregates) {
      if (aggregate.scope == scope_name) {
        scoped.push_back(aggregate);
      }
    }
    if (scoped.empty()) {
      continue;
    }
    out << "      " << scope_name << " stages:\n";
    print_stage_hotspots_from_aggregates(scoped, out, limit, "        ");
  }
}

MaskComparison compare_masks(const std::vector<uint8_t>& offline_mask,
                             const std::vector<uint8_t>& live_mask) {
  MaskComparison comparison;
  if (offline_mask.size() != live_mask.size() || offline_mask.empty()) {
    return comparison;
  }
  size_t agree = 0;
  size_t intersection = 0;
  size_t union_count = 0;
  size_t offline_foreground = 0;
  size_t live_foreground = 0;
  for (size_t index = 0; index < offline_mask.size(); ++index) {
    const bool off = offline_mask[index] != 0;
    const bool live = live_mask[index] != 0;
    agree += off == live ? 1U : 0U;
    intersection += (off && live) ? 1U : 0U;
    union_count += (off || live) ? 1U : 0U;
    offline_foreground += off ? 1U : 0U;
    live_foreground += live ? 1U : 0U;
  }
  comparison.available = true;
  comparison.pixel_agreement = static_cast<double>(agree) / static_cast<double>(offline_mask.size());
  comparison.intersection_over_union = union_count > 0 ? static_cast<double>(intersection) / static_cast<double>(union_count) : 1.0;
  comparison.offline_foreground_fraction = static_cast<double>(offline_foreground) / static_cast<double>(offline_mask.size());
  comparison.live_foreground_fraction = static_cast<double>(live_foreground) / static_cast<double>(live_mask.size());
  return comparison;
}

ValidatorOptions parse_arguments(int argc, char** argv) {
  ValidatorOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--tensor-npy" && index + 1 < argc) {
      options.tensor_path = argv[++index];
    } else if (arg == "--config" && index + 1 < argc) {
      options.config_path = argv[++index];
    } else if (arg == "--live-mask" && index + 1 < argc) {
      options.live_mask_path = std::filesystem::path(argv[++index]);
    } else if (arg == "--output-dir" && index + 1 < argc) {
      options.output_dir = argv[++index];
    } else if (arg == "--debug-chunk-index" && index + 1 < argc) {
      options.debug_chunk_index = parse_int_or_throw(argv[++index], "--debug-chunk-index");
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " --tensor-npy PATH --config FILE [--live-mask PATH] [--output-dir DIR] [--debug-chunk-index N] [--verbose]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unrecognized argument: " + arg);
    }
  }
  if (options.tensor_path.empty()) {
    throw std::runtime_error("--tensor-npy is required");
  }
  if (options.config_path.empty()) {
    throw std::runtime_error("--config is required");
  }
  if (options.output_dir.empty()) {
    options.output_dir = options.tensor_path.parent_path() / "dino_validator_artifacts" / options.tensor_path.stem();
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ValidatorOptions options = parse_arguments(argc, argv);
    StageProfiler profiler;
    ValidatorConfig config;
    CanonicalTensor tensor;
    {
      ScopedStageProfile stage(&profiler, "load_config", "run", -1, 0, options.verbose);
      config = load_config(options.config_path);
    }
    {
      ScopedStageProfile stage(&profiler,
                               "load_tensor",
                               "run",
                               -1,
                               sizeof(dino_complex) * tensor.values.capacity(),
                               options.verbose);
      tensor = load_canonical_tensor(options.tensor_path);
    }

    {
      ScopedStageProfile stage(&profiler, "create_output_dir", "run", -1, 0, options.verbose);
      std::filesystem::create_directories(options.output_dir);
    }

    std::vector<float> power_db;
    float frontend_reference_level = 0.0f;
    std::vector<float> corrected_db;
    {
      const size_t estimated_bytes = static_cast<size_t>(tensor.rows) * static_cast<size_t>(tensor.cols) * sizeof(float) * 2;
      ScopedStageProfile stage(&profiler, "frontend_correction", "run", -1, estimated_bytes, options.verbose);
      {
        ScopedStageProfile power_stage(&profiler, "power_db_from_tensor", "run", -1, estimated_bytes / 2, options.verbose);
        power_db = power_db_from_tensor(tensor);
      }
      {
        ScopedStageProfile correction_stage(&profiler, "frontend_correction_apply", "run", -1, estimated_bytes, options.verbose);
        corrected_db = frontend_corrected_db(power_db, tensor.rows, tensor.cols, config, frontend_reference_level);
      }
    }

    double resolution_hz = config.resolution_hz;
    if ((!std::isfinite(resolution_hz) || resolution_hz <= 0.0) && config.span_hz > 0.0 && tensor.rows > 0) {
      resolution_hz = config.span_hz / static_cast<double>(tensor.rows);
    }
    const double chunk_bin_hz = (std::isfinite(resolution_hz) && resolution_hz > 0.0) ? resolution_hz : 1.0;
    IgnoreSidebandInfo source_ignore_info;
    std::vector<double> source_freq_axis_hz;
    std::vector<ChunkPlanEntry> chunk_plan;
    {
      const size_t estimated_bytes = static_cast<size_t>(tensor.rows) * (sizeof(uint8_t) + sizeof(double)) + sizeof(ChunkPlanEntry) * 128;
      ScopedStageProfile stage(&profiler, "chunk_planning", "run", -1, estimated_bytes, options.verbose);
      source_freq_axis_hz = build_frequency_axis_hz(tensor.rows, resolution_hz);
      const auto planned_selection = select_uniform_chunk_plan_with_minimal_sideband_trim(
          tensor.rows,
          chunk_bin_hz,
          0.0,
          16,
          (std::isfinite(resolution_hz) && resolution_hz > 0.0 && config.ignore_sideband_hz > 0.0)
              ? std::optional<double>(config.ignore_sideband_hz)
          : std::nullopt,
        source_freq_axis_hz,
        config.chunk_bandwidth_hz,
        config.chunk_overlap_hz,
        16,
        config.uncalibrated_chunk_fraction,
        config.uncalibrated_overlap_fraction);
      source_ignore_info = planned_selection.ignore_info;
      chunk_plan = planned_selection.chunk_plan;
    }
    const int ignore_bins_per_side = source_ignore_info.applied_bins;

    holoscan::ops::DinoTorchRuntime runtime;
    holoscan::ops::DinoTorchRuntimeConfig runtime_config;
    runtime_config.inference_backend = config.inference_backend;
    runtime_config.model_script_path = config.model_script_path;
    runtime_config.torchscript_init_mode = config.torchscript_init_mode;
    runtime_config.torch_dtype = config.torch_dtype;
    runtime_config.imagenet_mean = {0.485, 0.456, 0.406};
    runtime_config.imagenet_std = {0.229, 0.224, 0.225};
    runtime_config.return_final_mask = true;
    runtime_config.return_final_mask_device = false;
    runtime_config.return_pre_model_gray = false;
    runtime_config.return_patch_features = false;
    runtime_config.compute_dino_threshold = true;
    runtime_config.compute_power_score = false;
    runtime_config.ignore_sideband_hz = config.ignore_sideband_hz;
    runtime_config.frontend_correction_enable = config.frontend_correction_enable;
    runtime_config.frontend_correction_row_q = config.frontend_correction_row_q;
    runtime_config.frontend_correction_smooth_sigma = config.frontend_correction_smooth_sigma;
    runtime_config.frontend_correction_reference_q = config.frontend_correction_reference_q;
    runtime_config.frontend_correction_max_boost_db = config.frontend_correction_max_boost_db;
    runtime_config.frontend_correction_soft_knee_db = config.frontend_correction_soft_knee_db;
    runtime_config.frontend_correction_edge_taper_fraction = config.frontend_correction_edge_taper_fraction;
    runtime_config.frontend_correction_edge_taper_sigma = config.frontend_correction_edge_taper_sigma;
    runtime_config.frontend_correction_edge_target_drop_db = config.frontend_correction_edge_target_drop_db;
    runtime_config.power_q = config.power_q;
    runtime_config.dino_group_k = config.dino_group_k;
    runtime_config.dino_group_spatial_weight = config.dino_group_spatial_weight;
    runtime_config.dino_group_score_q = config.dino_group_score_q;
    runtime_config.pipeline_final_threshold = config.pipeline_final_threshold;
    runtime_config.pipeline_gap_floor = config.pipeline_gap_floor;
    runtime_config.pipeline_power_rescue_floor = config.pipeline_power_rescue_floor;
    runtime_config.pipeline_power_rescue_gain = config.pipeline_power_rescue_gain;
    runtime_config.legacy_fast_gray_preprocess = config.legacy_fast_gray_preprocess;
    const size_t debug_chunk_index = static_cast<size_t>(clamp_value(options.debug_chunk_index, 0, static_cast<int>(chunk_plan.size() - 1)));
    const size_t max_runtime_batch_size = static_cast<size_t>(std::max(1, config.runtime_batch_size));
    ChunkGpuWorkspace gpu_workspace;
    {
      const size_t corrected_elements = static_cast<size_t>(tensor.rows) * static_cast<size_t>(tensor.cols);
      const size_t estimated_bytes = corrected_elements * sizeof(float);
      ScopedStageProfile stage(&profiler, "corrected_full_frame_upload", "run", -1, estimated_bytes, options.verbose);
      gpu_workspace.ensure_full_frame_capacity(corrected_elements);
      if (cudaMemcpyAsync(gpu_workspace.corrected_full_frame_device,
                          corrected_db.data(),
                          estimated_bytes,
                          cudaMemcpyHostToDevice,
                          gpu_workspace.stream) != cudaSuccess) {
        throw std::runtime_error("failed to upload full corrected tensor for offline DINO validator");
      }
    }
    {
      ScopedStageProfile stage(&profiler, "runtime_warmup", "run", -1, 0, options.verbose);
      std::vector<std::pair<int, int>> warmup_shapes;
      auto add_warmup_shape = [&](int rows, int batch_size) {
        const int clamped_rows = std::max(1, rows);
        const int clamped_batch = std::max(1, batch_size);
        const std::pair<int, int> shape {clamped_rows, clamped_batch};
        if (std::find(warmup_shapes.begin(), warmup_shapes.end(), shape) == warmup_shapes.end()) {
          warmup_shapes.push_back(shape);
        }
      };

      std::vector<ChunkPlanEntry> pending_warmup_batch;
      auto flush_warmup_batch = [&]() {
        if (pending_warmup_batch.empty()) {
          return;
        }
        add_warmup_shape(chunk_row_count(pending_warmup_batch.front()), static_cast<int>(pending_warmup_batch.size()));
        pending_warmup_batch.clear();
      };

      for (size_t chunk_index = 0; chunk_index < chunk_plan.size(); ++chunk_index) {
        const auto& chunk = chunk_plan[chunk_index];
        if (!pending_warmup_batch.empty() &&
            chunk_row_count(pending_warmup_batch.front()) != chunk_row_count(chunk)) {
          flush_warmup_batch();
        }
        pending_warmup_batch.push_back(chunk);
        if (pending_warmup_batch.size() >= max_runtime_batch_size) {
          flush_warmup_batch();
        }
      }
      flush_warmup_batch();
      if (!chunk_plan.empty()) {
        add_warmup_shape(chunk_row_count(chunk_plan[debug_chunk_index]), 1);
      }

      if (warmup_shapes.empty()) {
        add_warmup_shape(config.input_height, 1);
      }
      for (const auto& warmup_shape : warmup_shapes) {
        runtime.warmup(runtime_config,
                       warmup_shape.first,
                       tensor.cols,
                       config.input_height,
                       config.input_width,
                       config.patch_size,
                       warmup_shape.second,
                       gpu_workspace.stream);
      }
    }
    std::vector<ChunkRetryResult> chunk_results(chunk_plan.size());
    StageProfiler debug_profiler;
    std::optional<ChunkRetryResult> debug_chunk_result;
    std::vector<std::future<std::vector<std::pair<size_t, ChunkRetryResult>>>> pending_postprocess;
    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const size_t max_inflight_postprocess = std::max<size_t>(1, std::min<size_t>(4, static_cast<size_t>(hw_threads > 1 ? hw_threads - 1 : 1)));
    {
      ScopedStageProfile stage(&profiler, "chunk_loop_total", "run", -1, 0, options.verbose);
      std::vector<ChunkPlanEntry> pending_runtime_batch;
      std::vector<size_t> pending_runtime_indices;
      auto flush_runtime_batch = [&]() {
        if (pending_runtime_batch.empty()) {
          return;
        }
        auto batched_inference = run_retry_chunk_inference_batch(runtime,
                                                                 runtime_config,
                                                                 config,
                                                                 pending_runtime_batch,
                                                                 power_db,
                                                                 corrected_db,
                                                                 gpu_workspace.corrected_full_frame_device,
                                                                 tensor.rows,
                                                                 tensor.cols,
                                                                 resolution_hz,
                                                                 source_ignore_info.valid_row_mask,
                                                                 gpu_workspace,
                                                                 &profiler,
                                                                 options.verbose);
        precompute_retry_chunk_hybrid_batch(batched_inference, config, &profiler, options.verbose);
        auto mapped_indices = pending_runtime_indices;
        pending_postprocess.push_back(std::async(std::launch::async,
                                                 [&, mapped_indices = std::move(mapped_indices), batched_inference = std::move(batched_inference)]() mutable {
                                                   std::vector<std::pair<size_t, ChunkRetryResult>> completed_results;
                                                   completed_results.reserve(batched_inference.size());
                                                   for (size_t batch_index = 0; batch_index < batched_inference.size(); ++batch_index) {
                                                     auto chunk_result = finalize_retry_chunk_postprocess(config,
                                                                                                         std::move(batched_inference[batch_index]),
                                                                                                         &profiler,
                                                                                                         options.verbose);
                                                     ScopedStageProfile chunk_stage(&profiler, "chunk_drop_non_debug_payload", "chunk", chunk_result.chunk_index, 0, options.verbose);
                                                     discard_non_debug_chunk_payload(chunk_result);
                                                     completed_results.emplace_back(mapped_indices[batch_index], std::move(chunk_result));
                                                   }
                                                   return completed_results;
                                                 }));
        if (pending_postprocess.size() >= max_inflight_postprocess) {
          auto completed_batch = pending_postprocess.front().get();
          pending_postprocess.erase(pending_postprocess.begin());
          for (auto& completed : completed_batch) {
            chunk_results[completed.first] = std::move(completed.second);
          }
        }
        pending_runtime_batch.clear();
        pending_runtime_indices.clear();
      };

      for (size_t chunk_index = 0; chunk_index < chunk_plan.size(); ++chunk_index) {
        const auto& chunk = chunk_plan[chunk_index];
        if (!pending_runtime_batch.empty() &&
            chunk_row_count(pending_runtime_batch.front()) != chunk_row_count(chunk)) {
          flush_runtime_batch();
        }
        pending_runtime_batch.push_back(chunk);
        pending_runtime_indices.push_back(chunk_index);
        if (pending_runtime_batch.size() >= max_runtime_batch_size) {
          flush_runtime_batch();
        }
      }
      flush_runtime_batch();
      while (!pending_postprocess.empty()) {
        auto completed_batch = pending_postprocess.front().get();
        pending_postprocess.erase(pending_postprocess.begin());
        for (auto& completed : completed_batch) {
          chunk_results[completed.first] = std::move(completed.second);
        }
      }
    }

    if (!chunk_plan.empty()) {
      ScopedStageProfile stage(&debug_profiler, "debug_chunk_rerun_total", "run", -1, 0, options.verbose);
      auto debug_inference = run_retry_chunk_inference(runtime,
                                                       runtime_config,
                                                       config,
                                                       chunk_plan[debug_chunk_index],
                                                       power_db,
                                                       corrected_db,
                                                       gpu_workspace.corrected_full_frame_device,
                                                       tensor.rows,
                                                       tensor.cols,
                                                       resolution_hz,
                                                       source_ignore_info.valid_row_mask,
                                                       gpu_workspace,
                                                       &debug_profiler,
                                                       true,
                                                       options.verbose);
      debug_chunk_result = finalize_retry_chunk_postprocess(config,
                                                            std::move(debug_inference),
                                                            &debug_profiler,
                                                            options.verbose);
    }

    GlobalMergedResult global_merged;
    {
      const size_t estimated_bytes = static_cast<size_t>(tensor.rows) * static_cast<size_t>(tensor.cols) * (sizeof(uint8_t) * 2 + sizeof(float));
      ScopedStageProfile stage(&profiler, "global_merge", "run", -1, estimated_bytes, options.verbose);
      global_merged = build_global_merged_result(
          chunk_results,
          config,
          tensor.rows,
          tensor.cols,
          source_ignore_info.valid_row_mask);
    }
    std::vector<float> corrected_resized;
    {
      const size_t estimated_bytes = static_cast<size_t>(config.input_height) * static_cast<size_t>(config.input_width) * sizeof(float);
      ScopedStageProfile stage(&profiler, "full_frame_corrected_resize", "run", -1, estimated_bytes, options.verbose);
      corrected_resized = resize_bilinear(corrected_db, tensor.rows, tensor.cols, config.input_height, config.input_width);
    }

    const auto corrected_resized_path = options.output_dir / "offline_corrected_resized.npy";
    const auto final_mask_path = options.output_dir / "offline_final_mask.npy";
    const auto final_mask_pgm = options.output_dir / "offline_final_mask.pgm";
    const auto chunk_plan_path = options.output_dir / "offline_chunk_plan.json";
    const auto chunk_results_path = options.output_dir / "offline_chunk_results.json";
    const auto chunk_debug_dir = options.output_dir / "chunk_debug";
    const auto projected_grouped_mask_path = options.output_dir / "offline_projected_grouped_mask.npy";
    const auto projected_grouped_mask_pgm = options.output_dir / "offline_projected_grouped_mask.pgm";
    const auto projected_grouped_score_path = options.output_dir / "offline_projected_grouped_score.npy";
    const auto merged_box_mask_path = options.output_dir / "offline_merged_box_mask.npy";
    const auto merged_box_mask_pgm = options.output_dir / "offline_merged_box_mask.pgm";
    const auto projected_boxes_path = options.output_dir / "offline_projected_boxes.json";
    const auto merged_boxes_path = options.output_dir / "offline_merged_boxes.json";
    const auto summary_path = options.output_dir / "offline_validation_summary.json";
    const auto stage_profile_path = options.output_dir / "offline_stage_profile.json";

    {
      const size_t estimated_bytes = static_cast<size_t>(tensor.rows) * static_cast<size_t>(tensor.cols) * (sizeof(float) + sizeof(uint8_t)) +
                                     static_cast<size_t>(config.input_height) * static_cast<size_t>(config.input_width) * sizeof(float);
      ScopedStageProfile stage(&profiler, "artifact_serialization", "run", -1, estimated_bytes, options.verbose);
      write_npy_2d(corrected_resized_path, corrected_resized.data(), corrected_resized.size() * sizeof(float), config.input_height, config.input_width, "<f4");
      const auto& final_output_mask = global_merged.stitched_final_mask;
      std::vector<float> final_mask_float(final_output_mask.size(), 0.0f);
      for (size_t index = 0; index < final_output_mask.size(); ++index) {
        final_mask_float[index] = final_output_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(final_mask_path, final_mask_float.data(), final_mask_float.size() * sizeof(float), tensor.rows, tensor.cols, "<f4");
      write_pgm(final_mask_pgm, mask_to_u8(final_output_mask), tensor.cols, tensor.rows);
      std::vector<float> projected_grouped_mask_float(global_merged.projected_grouped_mask.size(), 0.0f);
      for (size_t index = 0; index < global_merged.projected_grouped_mask.size(); ++index) {
        projected_grouped_mask_float[index] = global_merged.projected_grouped_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(projected_grouped_mask_path,
                   projected_grouped_mask_float.data(),
                   projected_grouped_mask_float.size() * sizeof(float),
                   tensor.rows,
                   tensor.cols,
                   "<f4");
      write_pgm(projected_grouped_mask_pgm, mask_to_u8(global_merged.projected_grouped_mask), tensor.cols, tensor.rows);
      write_npy_2d(projected_grouped_score_path,
             global_merged.projected_grouped_score.data(),
             global_merged.projected_grouped_score.size() * sizeof(float),
             tensor.rows,
             tensor.cols,
             "<f4");

      std::vector<float> merged_box_mask_float(global_merged.merged_box_mask.size(), 0.0f);
      for (size_t index = 0; index < global_merged.merged_box_mask.size(); ++index) {
        merged_box_mask_float[index] = global_merged.merged_box_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(merged_box_mask_path,
                   merged_box_mask_float.data(),
                   merged_box_mask_float.size() * sizeof(float),
                   tensor.rows,
                   tensor.cols,
                   "<f4");
      write_pgm(merged_box_mask_pgm, mask_to_u8(global_merged.merged_box_mask), tensor.cols, tensor.rows);

      std::filesystem::create_directories(chunk_debug_dir);
      if (!chunk_results.empty()) {
      const size_t debug_chunk_index = static_cast<size_t>(clamp_value(options.debug_chunk_index, 0, static_cast<int>(chunk_results.size() - 1)));
      const auto& debug_chunk = chunk_results[debug_chunk_index];
      const auto debug_chunk_summary = chunk_debug_dir / "chunk_debug_summary.json";
      const auto debug_corrected_path = chunk_debug_dir / "chunk_corrected_resized.npy";
      const auto debug_runtime_input_gray_path = chunk_debug_dir / "chunk_runtime_input_gray.npy";
      const auto debug_dino_score_path = chunk_debug_dir / "chunk_dino_score.npy";
      const auto debug_raw_dino_score_path = chunk_debug_dir / "chunk_dino_score_raw.npy";
      const auto debug_raw_dino_score_deweighted_path = chunk_debug_dir / "chunk_dino_score_raw_deweighted.npy";
      const auto debug_coherence_gate_path = chunk_debug_dir / "chunk_coherence_gate.npy";
      const auto debug_hybrid_contrib_path = chunk_debug_dir / "chunk_hybrid_contrib.npy";
      const auto debug_combined_score_path = chunk_debug_dir / "chunk_combined_score.npy";
      const auto debug_grouped_seed_score_path = chunk_debug_dir / "chunk_grouped_seed_score.npy";
      const auto debug_grouped_seed_persistence_path = chunk_debug_dir / "chunk_grouped_seed_persistence.npy";
      const auto debug_grouped_seed_contrast_path = chunk_debug_dir / "chunk_grouped_seed_contrast.npy";
      const auto debug_grouped_support_exact_path = chunk_debug_dir / "chunk_grouped_support_exact_patch.npy";
      const auto debug_grouped_active_mask_exact_path = chunk_debug_dir / "chunk_grouped_active_mask_exact_patch.npy";
      const auto debug_grouped_cluster_labels_exact_path = chunk_debug_dir / "chunk_grouped_cluster_labels_exact_patch.npy";
      const auto debug_grouped_selected_mask_pre_smooth_exact_path = chunk_debug_dir / "chunk_grouped_selected_mask_pre_smooth_exact_patch.npy";
      const auto debug_grouped_selected_mask_exact_path = chunk_debug_dir / "chunk_grouped_selected_mask_exact_patch.npy";
      const auto debug_grouped_support_selected_raw_exact_path = chunk_debug_dir / "chunk_grouped_support_selected_raw_exact_patch.npy";
      const auto debug_grouped_selected_support_path = chunk_debug_dir / "chunk_grouped_selected_support.npy";
      const auto debug_grouped_cluster_quality_path = chunk_debug_dir / "chunk_grouped_cluster_quality.npy";
      const auto debug_patch_features_path = chunk_debug_dir / "chunk_patch_features.npy";
      const auto debug_valid_mask_path = chunk_debug_dir / "chunk_valid_mask.npy";
      const auto debug_bridged_mask_path = chunk_debug_dir / "chunk_bridged_mask.npy";
      const auto debug_grouped_mask_path = chunk_debug_dir / "chunk_grouped_mask.npy";
      const auto debug_grouped_mask_pgm = chunk_debug_dir / "chunk_grouped_mask.pgm";
      const auto debug_final_mask_path = chunk_debug_dir / "chunk_final_mask.npy";
      const auto debug_final_mask_pgm = chunk_debug_dir / "chunk_final_mask.pgm";
      const auto debug_final_mask_source_path = chunk_debug_dir / "chunk_final_mask_source.npy";
      const auto debug_final_mask_source_pgm = chunk_debug_dir / "chunk_final_mask_source.pgm";
      const auto debug_final_mask_projected_path = chunk_debug_dir / "chunk_final_mask_projected.npy";
      const auto debug_final_mask_projected_pgm = chunk_debug_dir / "chunk_final_mask_projected.pgm";
      const auto debug_grouped_boxes_path = chunk_debug_dir / "chunk_grouped_boxes.json";

      write_npy_2d(debug_corrected_path,
                   debug_chunk.corrected_resized.data(),
                   debug_chunk.corrected_resized.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_runtime_input_gray_path,
           debug_chunk.runtime_input_gray.data(),
           debug_chunk.runtime_input_gray.size() * sizeof(float),
         std::max(1, debug_chunk.runtime_input_gray_rows),
         std::max(1, debug_chunk.runtime_input_gray_cols),
           "<f4");
      write_npy_2d(debug_dino_score_path,
                   debug_chunk.dino_score_map.data(),
                   debug_chunk.dino_score_map.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_raw_dino_score_path,
           debug_chunk.raw_dino_score_map.data(),
           debug_chunk.raw_dino_score_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
       write_npy_2d(debug_raw_dino_score_deweighted_path,
         debug_chunk.raw_dino_score_deweighted_map.data(),
         debug_chunk.raw_dino_score_deweighted_map.size() * sizeof(float),
         debug_chunk.dst_rows,
         debug_chunk.dst_cols,
         "<f4");
      write_npy_2d(debug_coherence_gate_path,
                   debug_chunk.coherence_gate.data(),
                   debug_chunk.coherence_gate.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_hybrid_contrib_path,
                   debug_chunk.hybrid_contrib.data(),
                   debug_chunk.hybrid_contrib.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_combined_score_path,
                   debug_chunk.combined_score.data(),
                   debug_chunk.combined_score.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_grouped_seed_score_path,
           debug_chunk.grouped_seed_score_map.data(),
           debug_chunk.grouped_seed_score_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
       write_npy_2d(debug_grouped_seed_persistence_path,
         debug_chunk.grouped_seed_persistence_map.data(),
         debug_chunk.grouped_seed_persistence_map.size() * sizeof(float),
         debug_chunk.dst_rows,
         debug_chunk.dst_cols,
         "<f4");
       write_npy_2d(debug_grouped_seed_contrast_path,
         debug_chunk.grouped_seed_contrast_map.data(),
         debug_chunk.grouped_seed_contrast_map.size() * sizeof(float),
         debug_chunk.dst_rows,
         debug_chunk.dst_cols,
         "<f4");
       write_npy_2d(debug_grouped_support_exact_path,
         debug_chunk.grouped_support_map_exact.data(),
         debug_chunk.grouped_support_map_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_active_mask_exact_path,
         debug_chunk.grouped_active_mask_exact.data(),
         debug_chunk.grouped_active_mask_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_cluster_labels_exact_path,
         debug_chunk.grouped_cluster_labels_exact.data(),
         debug_chunk.grouped_cluster_labels_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_selected_mask_pre_smooth_exact_path,
         debug_chunk.grouped_selected_mask_pre_smooth_exact.data(),
         debug_chunk.grouped_selected_mask_pre_smooth_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_selected_mask_exact_path,
         debug_chunk.grouped_selected_mask_exact.data(),
         debug_chunk.grouped_selected_mask_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_support_selected_raw_exact_path,
         debug_chunk.grouped_support_selected_raw_exact.data(),
         debug_chunk.grouped_support_selected_raw_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
      write_npy_2d(debug_grouped_selected_support_path,
           debug_chunk.grouped_selected_support_map.data(),
           debug_chunk.grouped_selected_support_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
      write_npy_2d(debug_grouped_cluster_quality_path,
           debug_chunk.grouped_cluster_quality_map.data(),
           debug_chunk.grouped_cluster_quality_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
      if (!debug_chunk.patch_features.empty() && debug_chunk.patch_rows > 0 && debug_chunk.patch_cols > 0 && debug_chunk.feature_dim > 0) {
        write_npy_2d(debug_patch_features_path,
                     debug_chunk.patch_features.data(),
                     debug_chunk.patch_features.size() * sizeof(float),
                     debug_chunk.patch_rows * debug_chunk.patch_cols,
                     debug_chunk.feature_dim,
                     "<f4");
      }
      std::vector<float> debug_valid_mask_float(debug_chunk.valid_mask.size(), 0.0f);
      for (size_t index = 0; index < debug_chunk.valid_mask.size(); ++index) {
        debug_valid_mask_float[index] = debug_chunk.valid_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_valid_mask_path,
                   debug_valid_mask_float.data(),
                   debug_valid_mask_float.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      std::vector<float> debug_bridged_mask_float(debug_chunk.bridged_mask.size(), 0.0f);
      for (size_t index = 0; index < debug_chunk.bridged_mask.size(); ++index) {
        debug_bridged_mask_float[index] = debug_chunk.bridged_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_bridged_mask_path,
                   debug_bridged_mask_float.data(),
                   debug_bridged_mask_float.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      std::vector<float> debug_grouped_mask_float(debug_chunk.grouped_mask.size(), 0.0f);
      for (size_t index = 0; index < debug_chunk.grouped_mask.size(); ++index) {
        debug_grouped_mask_float[index] = debug_chunk.grouped_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_grouped_mask_path,
                   debug_grouped_mask_float.data(),
                   debug_grouped_mask_float.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_pgm(debug_grouped_mask_pgm, mask_to_u8(debug_chunk.grouped_mask), debug_chunk.dst_cols, debug_chunk.dst_rows);
      std::vector<float> debug_final_mask_float(debug_chunk.final_mask.size(), 0.0f);
      for (size_t index = 0; index < debug_chunk.final_mask.size(); ++index) {
        debug_final_mask_float[index] = debug_chunk.final_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_final_mask_path,
                   debug_final_mask_float.data(),
                   debug_final_mask_float.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_pgm(debug_final_mask_pgm, mask_to_u8(debug_chunk.final_mask), debug_chunk.dst_cols, debug_chunk.dst_rows);

      {
        std::ofstream debug_grouped_boxes_out(debug_grouped_boxes_path, std::ios::binary);
        if (!debug_grouped_boxes_out.is_open()) {
          throw std::runtime_error("failed to open chunk grouped boxes output");
        }
        debug_grouped_boxes_out << std::fixed << std::setprecision(6);
        debug_grouped_boxes_out << "{\n  \"box_count\": " << debug_chunk.grouped_boxes.size() << ",\n  \"boxes\": [\n";
        for (size_t index = 0; index < debug_chunk.grouped_boxes.size(); ++index) {
          const auto& box = debug_chunk.grouped_boxes[index];
          debug_grouped_boxes_out << "    {\"freq_start\": " << box.freq_start
                                  << ", \"freq_stop\": " << box.freq_stop
                                  << ", \"time_start\": " << box.time_start
                                  << ", \"time_stop\": " << box.time_stop
                                  << ", \"filled_area\": " << box.filled_area
                                  << ", \"density\": " << box.density
                                  << ", \"score_mean\": " << box.score_mean
                                  << ", \"score_peak\": " << box.score_peak
                                  << ", \"split_role\": \"" << json_escape(box.split_role) << "\""
                                  << ", \"split_applied\": " << (box.split_applied ? "true" : "false")
                                  << ", \"parent_component_id\": " << box.parent_component_id
                                  << "}";
          if (index + 1 != debug_chunk.grouped_boxes.size()) {
            debug_grouped_boxes_out << ",";
          }
          debug_grouped_boxes_out << "\n";
        }
        debug_grouped_boxes_out << "  ]\n}\n";
      }

      std::ofstream debug_summary_out(debug_chunk_summary, std::ios::binary);
      if (!debug_summary_out.is_open()) {
        throw std::runtime_error("failed to open chunk debug summary output");
      }
      debug_summary_out << std::fixed << std::setprecision(6);
      debug_summary_out << "{\n";
      debug_summary_out << "  \"chunk_index\": " << debug_chunk.chunk_index << ",\n";
      debug_summary_out << "  \"row_start\": " << debug_chunk.row_start << ",\n";
      debug_summary_out << "  \"row_stop\": " << debug_chunk.row_stop << ",\n";
      debug_summary_out << "  \"src_rows\": " << debug_chunk.src_rows << ",\n";
      debug_summary_out << "  \"src_cols\": " << debug_chunk.src_cols << ",\n";
      debug_summary_out << "  \"freq_start_hz\": " << debug_chunk.freq_start_hz << ",\n";
      debug_summary_out << "  \"freq_stop_hz\": " << debug_chunk.freq_stop_hz << ",\n";
      debug_summary_out << "  \"ignore_bins_per_side\": " << debug_chunk.ignore_bins_per_side << ",\n";
      debug_summary_out << "  \"dino_threshold\": " << debug_chunk.dino_threshold << ",\n";
      debug_summary_out << "  \"runtime_final_threshold\": " << debug_chunk.runtime_final_threshold << ",\n";
      debug_summary_out << "  \"seed_freq_threshold\": " << debug_chunk.seed_freq_threshold << ",\n";
      debug_summary_out << "  \"seed_res_threshold\": " << debug_chunk.seed_res_threshold << ",\n";
      debug_summary_out << "  \"combined_threshold\": " << debug_chunk.combined_threshold << ",\n";
      debug_summary_out << "  \"final_fraction\": " << debug_chunk.final_fraction << ",\n";
      debug_summary_out << "  \"connected_fraction\": " << debug_chunk.connected_fraction << ",\n";
      debug_summary_out << "  \"component_count\": " << debug_chunk.component_count << ",\n";
      debug_summary_out << "  \"grouped_box_count\": " << debug_chunk.grouped_box_count << ",\n";
      debug_summary_out << "  \"runtime_input_gray_rows\": " << debug_chunk.runtime_input_gray_rows << ",\n";
      debug_summary_out << "  \"runtime_input_gray_cols\": " << debug_chunk.runtime_input_gray_cols << ",\n";
      debug_summary_out << "  \"patch_rows\": " << debug_chunk.patch_rows << ",\n";
      debug_summary_out << "  \"patch_cols\": " << debug_chunk.patch_cols << ",\n";
      debug_summary_out << "  \"feature_dim\": " << debug_chunk.feature_dim << ",\n";
      debug_summary_out << "  \"grouped_seed_prior_enabled\": false,\n";
      debug_summary_out << "  \"grouped_component_seed_weight\": 0.0,\n";
      debug_summary_out << "  \"grouped_score_seed_weight\": 0.0,\n";
      debug_summary_out << "  \"grouped_path_enabled\": false,\n";
      debug_summary_out << "  \"artifact_contract\": \"chunk_fixed_detector_grid_v1\",\n";
      debug_summary_out << "  \"corrected_resized_npy\": \"" << json_escape(debug_corrected_path.string()) << "\",\n";
      debug_summary_out << "  \"runtime_input_gray_npy\": \"" << json_escape(debug_runtime_input_gray_path.string()) << "\",\n";
      debug_summary_out << "  \"dino_score_npy\": \"" << json_escape(debug_dino_score_path.string()) << "\",\n";
      debug_summary_out << "  \"dino_score_raw_npy\": \"" << json_escape(debug_raw_dino_score_path.string()) << "\",\n";
      debug_summary_out << "  \"dino_score_raw_deweighted_npy\": \"" << json_escape(debug_raw_dino_score_deweighted_path.string()) << "\",\n";
      debug_summary_out << "  \"hybrid_dino_source_mode\": \"deweighted_raw_dino_energy\",\n";
      debug_summary_out << "  \"coherence_gate_npy\": \"" << json_escape(debug_coherence_gate_path.string()) << "\",\n";
      debug_summary_out << "  \"hybrid_contrib_npy\": \"" << json_escape(debug_hybrid_contrib_path.string()) << "\",\n";
      debug_summary_out << "  \"combined_score_npy\": \"" << json_escape(debug_combined_score_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_seed_score_npy\": \"" << json_escape(debug_grouped_seed_score_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_seed_persistence_npy\": \"" << json_escape(debug_grouped_seed_persistence_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_seed_contrast_npy\": \"" << json_escape(debug_grouped_seed_contrast_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_support_exact_patch_npy\": \"" << json_escape(debug_grouped_support_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_active_mask_exact_patch_npy\": \"" << json_escape(debug_grouped_active_mask_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_cluster_labels_exact_patch_npy\": \"" << json_escape(debug_grouped_cluster_labels_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_selected_mask_pre_smooth_exact_patch_npy\": \"" << json_escape(debug_grouped_selected_mask_pre_smooth_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_selected_mask_exact_patch_npy\": \"" << json_escape(debug_grouped_selected_mask_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_support_selected_raw_exact_patch_npy\": \"" << json_escape(debug_grouped_support_selected_raw_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_selected_support_npy\": \"" << json_escape(debug_grouped_selected_support_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_cluster_quality_npy\": \"" << json_escape(debug_grouped_cluster_quality_path.string()) << "\",\n";
      debug_summary_out << "  \"patch_features_npy\": \"" << json_escape(debug_patch_features_path.string()) << "\",\n";
      debug_summary_out << "  \"valid_mask_npy\": \"" << json_escape(debug_valid_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"bridged_mask_npy\": \"" << json_escape(debug_bridged_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_mask_npy\": \"" << json_escape(debug_grouped_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_mask_pgm\": \"" << json_escape(debug_grouped_mask_pgm.string()) << "\",\n";
      debug_summary_out << "  \"grouped_boxes_json\": \"" << json_escape(debug_grouped_boxes_path.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_npy\": \"" << json_escape(debug_final_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_pgm\": \"" << json_escape(debug_final_mask_pgm.string()) << "\"\n";
      debug_summary_out << "}\n";
      }

      std::ofstream chunk_plan_out(chunk_plan_path, std::ios::binary);
      if (!chunk_plan_out.is_open()) {
        throw std::runtime_error("failed to open chunk plan output");
      }
      chunk_plan_out << std::fixed << std::setprecision(6);
      chunk_plan_out << "{\n";
      chunk_plan_out << "  \"chunk_count\": " << chunk_plan.size() << ",\n";
      chunk_plan_out << "  \"rows\": " << tensor.rows << ",\n";
      chunk_plan_out << "  \"cols\": " << tensor.cols << ",\n";
      chunk_plan_out << "  \"resolution_hz\": " << resolution_hz << ",\n";
      chunk_plan_out << "  \"ignore_bins_per_side\": " << ignore_bins_per_side << ",\n";
      chunk_plan_out << "  \"chunk_bandwidth_hz\": " << config.chunk_bandwidth_hz << ",\n";
      chunk_plan_out << "  \"chunk_overlap_hz\": " << config.chunk_overlap_hz << ",\n";
      chunk_plan_out << "  \"uncalibrated_chunk_fraction\": " << config.uncalibrated_chunk_fraction << ",\n";
      chunk_plan_out << "  \"uncalibrated_overlap_fraction\": " << config.uncalibrated_overlap_fraction << ",\n";
      chunk_plan_out << "  \"chunks\": [\n";
      for (size_t index = 0; index < chunk_plan.size(); ++index) {
        const auto& chunk = chunk_plan[index];
        chunk_plan_out << "    {\"chunk_index\": " << chunk.chunk_index
                       << ", \"row_start\": " << chunk.row_start
                       << ", \"row_stop\": " << chunk.row_stop
                       << ", \"freq_start_hz\": " << chunk.freq_start_hz
                       << ", \"freq_stop_hz\": " << chunk.freq_stop_hz << "}";
        if (index + 1 != chunk_plan.size()) {
          chunk_plan_out << ",";
        }
        chunk_plan_out << "\n";
      }
      chunk_plan_out << "  ]\n";
      chunk_plan_out << "}\n";

      std::ofstream chunk_results_out(chunk_results_path, std::ios::binary);
      if (!chunk_results_out.is_open()) {
        throw std::runtime_error("failed to open chunk results output");
      }
      chunk_results_out << std::fixed << std::setprecision(6);
      chunk_results_out << "{\n";
      chunk_results_out << "  \"chunk_count\": " << chunk_results.size() << ",\n";
      chunk_results_out << "  \"chunks\": [\n";
      for (size_t index = 0; index < chunk_results.size(); ++index) {
        const auto& chunk = chunk_results[index];
        chunk_results_out << "    {"
                          << "\"chunk_index\": " << chunk.chunk_index
                          << ", \"row_start\": " << chunk.row_start
                          << ", \"row_stop\": " << chunk.row_stop
                          << ", \"src_rows\": " << chunk.src_rows
                          << ", \"src_cols\": " << chunk.src_cols
                          << ", \"freq_start_hz\": " << chunk.freq_start_hz
                          << ", \"freq_stop_hz\": " << chunk.freq_stop_hz
                          << ", \"ignore_bins_per_side\": " << chunk.ignore_bins_per_side
                          << ", \"dino_threshold\": " << chunk.dino_threshold
                          << ", \"runtime_final_threshold\": " << chunk.runtime_final_threshold
                          << ", \"seed_freq_threshold\": " << chunk.seed_freq_threshold
                          << ", \"seed_res_threshold\": " << chunk.seed_res_threshold
                          << ", \"combined_threshold\": " << chunk.combined_threshold
                          << ", \"final_fraction\": " << chunk.final_fraction
                          << ", \"connected_fraction\": " << chunk.connected_fraction
                          << ", \"component_count\": " << chunk.component_count
                          << ", \"grouped_box_count\": " << chunk.grouped_box_count
                          << ", \"grouped_boxes\": [";
        for (size_t box_index = 0; box_index < chunk.grouped_boxes.size(); ++box_index) {
          const auto& box = chunk.grouped_boxes[box_index];
          chunk_results_out << "{\"freq_start\": " << box.freq_start
                            << ", \"freq_stop\": " << box.freq_stop
                            << ", \"time_start\": " << box.time_start
                            << ", \"time_stop\": " << box.time_stop
                            << ", \"filled_area\": " << box.filled_area
                            << ", \"density\": " << box.density
                            << ", \"score_mean\": " << box.score_mean
                            << ", \"score_peak\": " << box.score_peak
                            << ", \"split_role\": \"" << json_escape(box.split_role) << "\""
                            << ", \"split_applied\": " << (box.split_applied ? "true" : "false")
                            << ", \"parent_component_id\": " << box.parent_component_id
                            << "}";
          if (box_index + 1 != chunk.grouped_boxes.size()) {
            chunk_results_out << ", ";
          }
        }
        chunk_results_out << "]"
                          << "}";
        if (index + 1 != chunk_results.size()) {
          chunk_results_out << ",";
        }
        chunk_results_out << "\n";
      }
      chunk_results_out << "  ]\n";
      chunk_results_out << "}\n";

      std::ofstream projected_boxes_out(projected_boxes_path, std::ios::binary);
      if (!projected_boxes_out.is_open()) {
        throw std::runtime_error("failed to open projected boxes output");
      }
      projected_boxes_out << "{\n  \"box_count\": " << global_merged.projected_boxes.size() << ",\n  \"boxes\": [\n";
      for (size_t index = 0; index < global_merged.projected_boxes.size(); ++index) {
        const auto& box = global_merged.projected_boxes[index];
        projected_boxes_out << "    {\"freq_start\": " << box.freq_start
                            << ", \"freq_stop\": " << box.freq_stop
                            << ", \"time_start\": " << box.time_start
                            << ", \"time_stop\": " << box.time_stop
                            << ", \"filled_area\": " << box.filled_area
                            << ", \"density\": " << box.density
                            << ", \"score_mean\": " << box.score_mean
                            << ", \"score_peak\": " << box.score_peak
                            << ", \"split_role\": \"" << json_escape(box.split_role) << "\""
                            << ", \"split_applied\": " << (box.split_applied ? "true" : "false")
                            << ", \"source_box_count\": " << box.source_box_count
                            << ", \"source_chunk_indices\": [";
        for (size_t chunk_index = 0; chunk_index < box.source_chunk_indices.size(); ++chunk_index) {
          projected_boxes_out << box.source_chunk_indices[chunk_index];
          if (chunk_index + 1 != box.source_chunk_indices.size()) {
            projected_boxes_out << ", ";
          }
        }
        projected_boxes_out << "]}";
        if (index + 1 != global_merged.projected_boxes.size()) {
          projected_boxes_out << ",";
        }
        projected_boxes_out << "\n";
      }
      projected_boxes_out << "  ]\n}\n";

      std::ofstream merged_boxes_out(merged_boxes_path, std::ios::binary);
      if (!merged_boxes_out.is_open()) {
        throw std::runtime_error("failed to open merged boxes output");
      }
      merged_boxes_out << "{\n  \"box_count\": " << global_merged.merged_boxes.size() << ",\n  \"boxes\": [\n";
      for (size_t index = 0; index < global_merged.merged_boxes.size(); ++index) {
        const auto& box = global_merged.merged_boxes[index];
        merged_boxes_out << "    {\"freq_start\": " << box.freq_start
                         << ", \"freq_stop\": " << box.freq_stop
                         << ", \"time_start\": " << box.time_start
                         << ", \"time_stop\": " << box.time_stop
                         << ", \"filled_area\": " << box.filled_area
                         << ", \"density\": " << box.density
                         << ", \"score_mean\": " << box.score_mean
                         << ", \"score_peak\": " << box.score_peak
                         << ", \"split_role\": \"" << json_escape(box.split_role) << "\""
                         << ", \"split_applied\": " << (box.split_applied ? "true" : "false")
                         << ", \"source_box_count\": " << box.source_box_count
                         << ", \"source_chunk_indices\": [";
        for (size_t chunk_index = 0; chunk_index < box.source_chunk_indices.size(); ++chunk_index) {
          merged_boxes_out << box.source_chunk_indices[chunk_index];
          if (chunk_index + 1 != box.source_chunk_indices.size()) {
            merged_boxes_out << ", ";
          }
        }
        merged_boxes_out << "]}";
        if (index + 1 != global_merged.merged_boxes.size()) {
          merged_boxes_out << ",";
        }
        merged_boxes_out << "\n";
      }
      merged_boxes_out << "  ]\n}\n";
    }

    const auto debug_chunk_summary_path = chunk_debug_dir / "chunk_debug_summary.json";
    std::filesystem::create_directories(chunk_debug_dir);
    if (debug_chunk_result.has_value()) {
      const auto& debug_chunk = *debug_chunk_result;
      const ChunkRetryResult& stitched_debug_chunk =
          (debug_chunk_index < chunk_results.size()) ? chunk_results[debug_chunk_index] : debug_chunk;
      const auto debug_corrected_path = chunk_debug_dir / "chunk_corrected_resized.npy";
      const auto debug_runtime_input_gray_path = chunk_debug_dir / "chunk_runtime_input_gray.npy";
      const auto debug_dino_score_path = chunk_debug_dir / "chunk_dino_score.npy";
      const auto debug_raw_dino_score_path = chunk_debug_dir / "chunk_dino_score_raw.npy";
      const auto debug_raw_dino_score_deweighted_path = chunk_debug_dir / "chunk_dino_score_raw_deweighted.npy";
      const auto debug_coherence_gate_path = chunk_debug_dir / "chunk_coherence_gate.npy";
      const auto debug_hybrid_contrib_path = chunk_debug_dir / "chunk_hybrid_contrib.npy";
      const auto debug_combined_score_path = chunk_debug_dir / "chunk_combined_score.npy";
      const auto debug_grouped_seed_score_path = chunk_debug_dir / "chunk_grouped_seed_score.npy";
      const auto debug_grouped_seed_persistence_path = chunk_debug_dir / "chunk_grouped_seed_persistence.npy";
      const auto debug_grouped_seed_contrast_path = chunk_debug_dir / "chunk_grouped_seed_contrast.npy";
      const auto debug_grouped_support_exact_path = chunk_debug_dir / "chunk_grouped_support_exact_patch.npy";
      const auto debug_grouped_active_mask_exact_path = chunk_debug_dir / "chunk_grouped_active_mask_exact_patch.npy";
      const auto debug_grouped_cluster_labels_exact_path = chunk_debug_dir / "chunk_grouped_cluster_labels_exact_patch.npy";
      const auto debug_grouped_selected_mask_pre_smooth_exact_path = chunk_debug_dir / "chunk_grouped_selected_mask_pre_smooth_exact_patch.npy";
      const auto debug_grouped_selected_mask_exact_path = chunk_debug_dir / "chunk_grouped_selected_mask_exact_patch.npy";
      const auto debug_grouped_support_selected_raw_exact_path = chunk_debug_dir / "chunk_grouped_support_selected_raw_exact_patch.npy";
      const auto debug_grouped_selected_support_path = chunk_debug_dir / "chunk_grouped_selected_support.npy";
      const auto debug_grouped_cluster_quality_path = chunk_debug_dir / "chunk_grouped_cluster_quality.npy";
      const auto debug_patch_features_path = chunk_debug_dir / "chunk_patch_features.npy";
      const auto debug_valid_mask_path = chunk_debug_dir / "chunk_valid_mask.npy";
      const auto debug_bridged_mask_path = chunk_debug_dir / "chunk_bridged_mask.npy";
      const auto debug_grouped_mask_path = chunk_debug_dir / "chunk_grouped_mask.npy";
      const auto debug_grouped_mask_pgm = chunk_debug_dir / "chunk_grouped_mask.pgm";
      const auto debug_final_mask_path = chunk_debug_dir / "chunk_final_mask.npy";
      const auto debug_final_mask_pgm = chunk_debug_dir / "chunk_final_mask.pgm";
      const auto debug_final_mask_source_path = chunk_debug_dir / "chunk_final_mask_source.npy";
      const auto debug_final_mask_source_pgm = chunk_debug_dir / "chunk_final_mask_source.pgm";
      const auto debug_final_mask_projected_path = chunk_debug_dir / "chunk_final_mask_projected.npy";
      const auto debug_final_mask_projected_pgm = chunk_debug_dir / "chunk_final_mask_projected.pgm";
      const auto debug_grouped_boxes_path = chunk_debug_dir / "chunk_grouped_boxes.json";

      const size_t estimated_bytes = static_cast<size_t>(debug_chunk.dst_rows) * static_cast<size_t>(debug_chunk.dst_cols) * (sizeof(float) * 8 + sizeof(uint8_t) * 3);
      ScopedStageProfile stage(&debug_profiler, "debug_artifact_serialization", "run", -1, estimated_bytes, options.verbose);

      write_npy_2d(debug_corrected_path,
                   debug_chunk.corrected_resized.data(),
                   debug_chunk.corrected_resized.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_runtime_input_gray_path,
                   debug_chunk.runtime_input_gray.data(),
                   debug_chunk.runtime_input_gray.size() * sizeof(float),
                   std::max(1, debug_chunk.runtime_input_gray_rows),
                   std::max(1, debug_chunk.runtime_input_gray_cols),
                   "<f4");
      write_npy_2d(debug_dino_score_path,
                   debug_chunk.dino_score_map.data(),
                   debug_chunk.dino_score_map.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_raw_dino_score_path,
                   debug_chunk.raw_dino_score_map.data(),
                   debug_chunk.raw_dino_score_map.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_raw_dino_score_deweighted_path,
           debug_chunk.raw_dino_score_deweighted_map.data(),
           debug_chunk.raw_dino_score_deweighted_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
      write_npy_2d(debug_coherence_gate_path,
                   debug_chunk.coherence_gate.data(),
                   debug_chunk.coherence_gate.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_hybrid_contrib_path,
                   debug_chunk.hybrid_contrib.data(),
                   debug_chunk.hybrid_contrib.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_combined_score_path,
                   debug_chunk.combined_score.data(),
                   debug_chunk.combined_score.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      write_npy_2d(debug_grouped_seed_score_path,
           debug_chunk.grouped_seed_score_map.data(),
           debug_chunk.grouped_seed_score_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
       write_npy_2d(debug_grouped_seed_persistence_path,
         debug_chunk.grouped_seed_persistence_map.data(),
         debug_chunk.grouped_seed_persistence_map.size() * sizeof(float),
         debug_chunk.dst_rows,
         debug_chunk.dst_cols,
         "<f4");
       write_npy_2d(debug_grouped_seed_contrast_path,
         debug_chunk.grouped_seed_contrast_map.data(),
         debug_chunk.grouped_seed_contrast_map.size() * sizeof(float),
         debug_chunk.dst_rows,
         debug_chunk.dst_cols,
         "<f4");
       write_npy_2d(debug_grouped_support_exact_path,
         debug_chunk.grouped_support_map_exact.data(),
         debug_chunk.grouped_support_map_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_active_mask_exact_path,
         debug_chunk.grouped_active_mask_exact.data(),
         debug_chunk.grouped_active_mask_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_cluster_labels_exact_path,
         debug_chunk.grouped_cluster_labels_exact.data(),
         debug_chunk.grouped_cluster_labels_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_selected_mask_pre_smooth_exact_path,
         debug_chunk.grouped_selected_mask_pre_smooth_exact.data(),
         debug_chunk.grouped_selected_mask_pre_smooth_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_selected_mask_exact_path,
         debug_chunk.grouped_selected_mask_exact.data(),
         debug_chunk.grouped_selected_mask_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
       write_npy_2d(debug_grouped_support_selected_raw_exact_path,
         debug_chunk.grouped_support_selected_raw_exact.data(),
         debug_chunk.grouped_support_selected_raw_exact.size() * sizeof(float),
         debug_chunk.patch_rows,
         debug_chunk.patch_cols,
         "<f4");
      write_npy_2d(debug_grouped_selected_support_path,
           debug_chunk.grouped_selected_support_map.data(),
           debug_chunk.grouped_selected_support_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
      write_npy_2d(debug_grouped_cluster_quality_path,
           debug_chunk.grouped_cluster_quality_map.data(),
           debug_chunk.grouped_cluster_quality_map.size() * sizeof(float),
           debug_chunk.dst_rows,
           debug_chunk.dst_cols,
           "<f4");
      if (!debug_chunk.patch_features.empty() && debug_chunk.patch_rows > 0 && debug_chunk.patch_cols > 0 && debug_chunk.feature_dim > 0) {
        write_npy_2d(debug_patch_features_path,
                     debug_chunk.patch_features.data(),
                     debug_chunk.patch_features.size() * sizeof(float),
                     debug_chunk.patch_rows * debug_chunk.patch_cols,
                     debug_chunk.feature_dim,
                     "<f4");
      }

      std::vector<float> debug_valid_mask_float(debug_chunk.valid_mask.size(), 0.0f);
      for (size_t index = 0; index < debug_chunk.valid_mask.size(); ++index) {
        debug_valid_mask_float[index] = debug_chunk.valid_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_valid_mask_path,
                   debug_valid_mask_float.data(),
                   debug_valid_mask_float.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      std::vector<float> debug_bridged_mask_float(debug_chunk.bridged_mask.size(), 0.0f);
      for (size_t index = 0; index < debug_chunk.bridged_mask.size(); ++index) {
        debug_bridged_mask_float[index] = debug_chunk.bridged_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_bridged_mask_path,
                   debug_bridged_mask_float.data(),
                   debug_bridged_mask_float.size() * sizeof(float),
                   debug_chunk.dst_rows,
                   debug_chunk.dst_cols,
                   "<f4");
      std::vector<float> debug_grouped_mask_float(stitched_debug_chunk.grouped_mask.size(), 0.0f);
      for (size_t index = 0; index < stitched_debug_chunk.grouped_mask.size(); ++index) {
        debug_grouped_mask_float[index] = stitched_debug_chunk.grouped_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_grouped_mask_path,
                   debug_grouped_mask_float.data(),
                   debug_grouped_mask_float.size() * sizeof(float),
                   stitched_debug_chunk.dst_rows,
                   stitched_debug_chunk.dst_cols,
                   "<f4");
      write_pgm(debug_grouped_mask_pgm,
                mask_to_u8(stitched_debug_chunk.grouped_mask),
                stitched_debug_chunk.dst_cols,
                stitched_debug_chunk.dst_rows);
      std::vector<float> debug_final_mask_float(stitched_debug_chunk.final_mask.size(), 0.0f);
      for (size_t index = 0; index < stitched_debug_chunk.final_mask.size(); ++index) {
        debug_final_mask_float[index] = stitched_debug_chunk.final_mask[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_final_mask_path,
                   debug_final_mask_float.data(),
                   debug_final_mask_float.size() * sizeof(float),
                   stitched_debug_chunk.dst_rows,
                   stitched_debug_chunk.dst_cols,
                   "<f4");
      write_pgm(debug_final_mask_pgm,
                mask_to_u8(stitched_debug_chunk.final_mask),
                stitched_debug_chunk.dst_cols,
                stitched_debug_chunk.dst_rows);
      std::vector<float> debug_final_mask_source_float(debug_chunk.final_mask_source.size(), 0.0f);
      for (size_t index = 0; index < debug_chunk.final_mask_source.size(); ++index) {
        debug_final_mask_source_float[index] = debug_chunk.final_mask_source[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_final_mask_source_path,
                   debug_final_mask_source_float.data(),
                   debug_final_mask_source_float.size() * sizeof(float),
                   debug_chunk.src_rows,
                   debug_chunk.src_cols,
                   "<f4");
      write_pgm(debug_final_mask_source_pgm,
                mask_to_u8(debug_chunk.final_mask_source),
                debug_chunk.src_cols,
                debug_chunk.src_rows);
      std::vector<uint8_t> debug_final_mask_projected(static_cast<size_t>(debug_chunk.src_rows) * static_cast<size_t>(tensor.cols), 0);
      if (!stitched_debug_chunk.final_mask.empty() && stitched_debug_chunk.dst_rows > 0 && stitched_debug_chunk.dst_cols > 0) {
        const auto projected_row_nearest = build_nearest_resize_indices(stitched_debug_chunk.dst_rows, debug_chunk.src_rows);
        const auto projected_col_nearest = build_nearest_resize_indices(stitched_debug_chunk.dst_cols, tensor.cols);
        for (int projected_row = 0; projected_row < debug_chunk.src_rows; ++projected_row) {
          const int src_row_nearest = projected_row_nearest[static_cast<size_t>(projected_row)];
          for (int col = 0; col < tensor.cols; ++col) {
            const int src_col_nearest = projected_col_nearest[static_cast<size_t>(col)];
            debug_final_mask_projected[flat_index(tensor.cols, projected_row, col)] =
                stitched_debug_chunk.final_mask[flat_index(stitched_debug_chunk.dst_cols, src_row_nearest, src_col_nearest)] ? 1 : 0;
          }
        }
      }
      std::vector<float> debug_final_mask_projected_float(debug_final_mask_projected.size(), 0.0f);
      for (size_t index = 0; index < debug_final_mask_projected.size(); ++index) {
        debug_final_mask_projected_float[index] = debug_final_mask_projected[index] ? 1.0f : 0.0f;
      }
      write_npy_2d(debug_final_mask_projected_path,
                   debug_final_mask_projected_float.data(),
                   debug_final_mask_projected_float.size() * sizeof(float),
                   debug_chunk.src_rows,
                   tensor.cols,
                   "<f4");
      write_pgm(debug_final_mask_projected_pgm,
                mask_to_u8(debug_final_mask_projected),
                tensor.cols,
                debug_chunk.src_rows);

      {
        std::ofstream debug_grouped_boxes_out(debug_grouped_boxes_path, std::ios::binary);
        if (!debug_grouped_boxes_out.is_open()) {
          throw std::runtime_error("failed to open chunk grouped boxes output");
        }
        debug_grouped_boxes_out << std::fixed << std::setprecision(6);
        debug_grouped_boxes_out << "{\n  \"box_count\": " << stitched_debug_chunk.grouped_boxes.size() << ",\n  \"boxes\": [\n";
        for (size_t index = 0; index < stitched_debug_chunk.grouped_boxes.size(); ++index) {
          const auto& box = stitched_debug_chunk.grouped_boxes[index];
          debug_grouped_boxes_out << "    {\"freq_start\": " << box.freq_start
                                  << ", \"freq_stop\": " << box.freq_stop
                                  << ", \"time_start\": " << box.time_start
                                  << ", \"time_stop\": " << box.time_stop
                                  << ", \"filled_area\": " << box.filled_area
                                  << ", \"density\": " << box.density
                                  << ", \"score_mean\": " << box.score_mean
                                  << ", \"score_peak\": " << box.score_peak
                                  << ", \"split_role\": \"" << json_escape(box.split_role) << "\""
                                  << ", \"split_applied\": " << (box.split_applied ? "true" : "false")
                                  << ", \"parent_component_id\": " << box.parent_component_id
                                  << "}";
          if (index + 1 != stitched_debug_chunk.grouped_boxes.size()) {
            debug_grouped_boxes_out << ",";
          }
          debug_grouped_boxes_out << "\n";
        }
        debug_grouped_boxes_out << "  ]\n}\n";
      }

      std::ofstream debug_summary_out(debug_chunk_summary_path, std::ios::binary);
      if (!debug_summary_out.is_open()) {
        throw std::runtime_error("failed to open chunk debug summary output");
      }
      debug_summary_out << std::fixed << std::setprecision(6);
      debug_summary_out << "{\n";
      debug_summary_out << "  \"chunk_index\": " << stitched_debug_chunk.chunk_index << ",\n";
      debug_summary_out << "  \"row_start\": " << stitched_debug_chunk.row_start << ",\n";
      debug_summary_out << "  \"row_stop\": " << stitched_debug_chunk.row_stop << ",\n";
      debug_summary_out << "  \"src_rows\": " << stitched_debug_chunk.src_rows << ",\n";
      debug_summary_out << "  \"src_cols\": " << stitched_debug_chunk.src_cols << ",\n";
      debug_summary_out << "  \"freq_start_hz\": " << stitched_debug_chunk.freq_start_hz << ",\n";
      debug_summary_out << "  \"freq_stop_hz\": " << stitched_debug_chunk.freq_stop_hz << ",\n";
      debug_summary_out << "  \"ignore_bins_per_side\": " << stitched_debug_chunk.ignore_bins_per_side << ",\n";
      debug_summary_out << "  \"dino_threshold\": " << stitched_debug_chunk.dino_threshold << ",\n";
      debug_summary_out << "  \"runtime_final_threshold\": " << stitched_debug_chunk.runtime_final_threshold << ",\n";
      debug_summary_out << "  \"seed_freq_threshold\": " << stitched_debug_chunk.seed_freq_threshold << ",\n";
      debug_summary_out << "  \"seed_res_threshold\": " << stitched_debug_chunk.seed_res_threshold << ",\n";
      debug_summary_out << "  \"combined_threshold\": " << stitched_debug_chunk.combined_threshold << ",\n";
      debug_summary_out << "  \"final_fraction\": " << stitched_debug_chunk.final_fraction << ",\n";
      debug_summary_out << "  \"connected_fraction\": " << stitched_debug_chunk.connected_fraction << ",\n";
      debug_summary_out << "  \"component_count\": " << stitched_debug_chunk.component_count << ",\n";
      debug_summary_out << "  \"grouped_box_count\": " << stitched_debug_chunk.grouped_box_count << ",\n";
      debug_summary_out << "  \"runtime_input_gray_rows\": " << debug_chunk.runtime_input_gray_rows << ",\n";
      debug_summary_out << "  \"runtime_input_gray_cols\": " << debug_chunk.runtime_input_gray_cols << ",\n";
      debug_summary_out << "  \"patch_rows\": " << debug_chunk.patch_rows << ",\n";
      debug_summary_out << "  \"patch_cols\": " << debug_chunk.patch_cols << ",\n";
      debug_summary_out << "  \"feature_dim\": " << debug_chunk.feature_dim << ",\n";
      debug_summary_out << "  \"grouped_seed_prior_enabled\": false,\n";
      debug_summary_out << "  \"grouped_component_seed_weight\": 0.0,\n";
      debug_summary_out << "  \"grouped_score_seed_weight\": 0.0,\n";
      debug_summary_out << "  \"grouped_path_enabled\": false,\n";
      debug_summary_out << "  \"legacy_fast_gray_preprocess\": " << (config.legacy_fast_gray_preprocess ? "true" : "false") << ",\n";
      debug_summary_out << "  \"artifact_contract\": \"chunk_fixed_detector_grid_v1\",\n";
      debug_summary_out << "  \"mask_source\": \"stitched_chunk_result\",\n";
      debug_summary_out << "  \"diagnostic_tensor_source\": \"debug_chunk_rerun\",\n";
      debug_summary_out << "  \"corrected_resized_npy\": \"" << json_escape(debug_corrected_path.string()) << "\",\n";
      debug_summary_out << "  \"runtime_input_gray_npy\": \"" << json_escape(debug_runtime_input_gray_path.string()) << "\",\n";
      debug_summary_out << "  \"dino_score_npy\": \"" << json_escape(debug_dino_score_path.string()) << "\",\n";
      debug_summary_out << "  \"dino_score_raw_npy\": \"" << json_escape(debug_raw_dino_score_path.string()) << "\",\n";
      debug_summary_out << "  \"dino_score_raw_deweighted_npy\": \"" << json_escape(debug_raw_dino_score_deweighted_path.string()) << "\",\n";
      debug_summary_out << "  \"hybrid_dino_source_mode\": \"deweighted_raw_dino_energy\",\n";
      debug_summary_out << "  \"coherence_gate_npy\": \"" << json_escape(debug_coherence_gate_path.string()) << "\",\n";
      debug_summary_out << "  \"hybrid_contrib_npy\": \"" << json_escape(debug_hybrid_contrib_path.string()) << "\",\n";
      debug_summary_out << "  \"combined_score_npy\": \"" << json_escape(debug_combined_score_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_seed_score_npy\": \"" << json_escape(debug_grouped_seed_score_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_seed_persistence_npy\": \"" << json_escape(debug_grouped_seed_persistence_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_seed_contrast_npy\": \"" << json_escape(debug_grouped_seed_contrast_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_support_exact_patch_npy\": \"" << json_escape(debug_grouped_support_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_active_mask_exact_patch_npy\": \"" << json_escape(debug_grouped_active_mask_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_cluster_labels_exact_patch_npy\": \"" << json_escape(debug_grouped_cluster_labels_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_selected_mask_pre_smooth_exact_patch_npy\": \"" << json_escape(debug_grouped_selected_mask_pre_smooth_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_selected_mask_exact_patch_npy\": \"" << json_escape(debug_grouped_selected_mask_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_support_selected_raw_exact_patch_npy\": \"" << json_escape(debug_grouped_support_selected_raw_exact_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_selected_support_npy\": \"" << json_escape(debug_grouped_selected_support_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_cluster_quality_npy\": \"" << json_escape(debug_grouped_cluster_quality_path.string()) << "\",\n";
      debug_summary_out << "  \"patch_features_npy\": \"" << json_escape(debug_patch_features_path.string()) << "\",\n";
      debug_summary_out << "  \"valid_mask_npy\": \"" << json_escape(debug_valid_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"bridged_mask_npy\": \"" << json_escape(debug_bridged_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_mask_npy\": \"" << json_escape(debug_grouped_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"grouped_mask_pgm\": \"" << json_escape(debug_grouped_mask_pgm.string()) << "\",\n";
      debug_summary_out << "  \"grouped_boxes_json\": \"" << json_escape(debug_grouped_boxes_path.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_npy\": \"" << json_escape(debug_final_mask_path.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_pgm\": \"" << json_escape(debug_final_mask_pgm.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_source_npy\": \"" << json_escape(debug_final_mask_source_path.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_source_pgm\": \"" << json_escape(debug_final_mask_source_pgm.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_projected_npy\": \"" << json_escape(debug_final_mask_projected_path.string()) << "\",\n";
      debug_summary_out << "  \"final_mask_projected_pgm\": \"" << json_escape(debug_final_mask_projected_pgm.string()) << "\"\n";
      debug_summary_out << "}\n";
    }

    write_stage_profile_json(stage_profile_path, profiler, debug_profiler);

    MaskComparison live_comparison;
    if (options.live_mask_path.has_value()) {
      int live_rows = 0;
      int live_cols = 0;
      const auto live_mask = load_pgm(*options.live_mask_path, live_rows, live_cols);
      if (live_rows == tensor.rows && live_cols == tensor.cols) {
        std::vector<uint8_t> live_binary(live_mask.size(), 0);
        for (size_t index = 0; index < live_mask.size(); ++index) {
          live_binary[index] = live_mask[index] >= 128 ? 1 : 0;
        }
        live_comparison = compare_masks(global_merged.merged_box_mask, live_binary);
      }
    }

    std::ofstream summary(summary_path, std::ios::binary);
    if (!summary.is_open()) {
      throw std::runtime_error("failed to open offline DINO summary output");
    }
    summary << std::fixed << std::setprecision(6);
    summary << "{\n";
    summary << "  \"tensor_path\": \"" << json_escape(options.tensor_path.string()) << "\",\n";
    summary << "  \"config_path\": \"" << json_escape(options.config_path.string()) << "\",\n";
    summary << "  \"input_rows\": " << tensor.input_rows << ",\n";
    summary << "  \"input_cols\": " << tensor.input_cols << ",\n";
    summary << "  \"canonical_rows\": " << tensor.rows << ",\n";
    summary << "  \"canonical_cols\": " << tensor.cols << ",\n";
    summary << "  \"transposed_to_frequency_time\": " << (tensor.transposed ? "true" : "false") << ",\n";
    summary << "  \"output_rows\": " << config.input_height << ",\n";
    summary << "  \"output_cols\": " << config.input_width << ",\n";
    summary << "  \"resolution_hz\": " << resolution_hz << ",\n";
    summary << "  \"span_hz\": " << config.span_hz << ",\n";
    summary << "  \"ignore_bins_per_side\": " << ignore_bins_per_side << ",\n";
    summary << "  \"chunk_count\": " << chunk_plan.size() << ",\n";
    summary << "  \"chunk_plan_json\": \"" << json_escape(chunk_plan_path.string()) << "\",\n";
    summary << "  \"chunk_results_json\": \"" << json_escape(chunk_results_path.string()) << "\",\n";
    summary << "  \"chunk_debug_dir\": \"" << json_escape(chunk_debug_dir.string()) << "\",\n";
    summary << "  \"projected_boxes_json\": \"" << json_escape(projected_boxes_path.string()) << "\",\n";
    summary << "  \"merged_boxes_json\": \"" << json_escape(merged_boxes_path.string()) << "\",\n";
    summary << "  \"chunk_bandwidth_hz\": " << config.chunk_bandwidth_hz << ",\n";
    summary << "  \"chunk_overlap_hz\": " << config.chunk_overlap_hz << ",\n";
    summary << "  \"frontend_reference_level\": " << frontend_reference_level << ",\n";
    summary << "  \"runtime_backend_used\": \"" << json_escape(runtime_config.inference_backend) << "\",\n";
    summary << "  \"legacy_fast_gray_preprocess\": " << (config.legacy_fast_gray_preprocess ? "true" : "false") << ",\n";
    summary << "  \"debug_chunk_rerun_enabled\": true,\n";
    summary << "  \"debug_chunk_summary_json\": \"" << json_escape(debug_chunk_summary_path.string()) << "\",\n";
    if (debug_chunk_result.has_value()) {
      const auto& debug_chunk = *debug_chunk_result;
      summary << "  \"debug_chunk_index\": " << debug_chunk.chunk_index << ",\n";
      summary << "  \"debug_chunk_dino_threshold\": " << debug_chunk.dino_threshold << ",\n";
      summary << "  \"debug_chunk_runtime_final_threshold\": " << debug_chunk.runtime_final_threshold << ",\n";
      summary << "  \"debug_chunk_seed_freq_threshold\": " << debug_chunk.seed_freq_threshold << ",\n";
      summary << "  \"debug_chunk_seed_res_threshold\": " << debug_chunk.seed_res_threshold << ",\n";
      summary << "  \"debug_chunk_combined_threshold\": " << debug_chunk.combined_threshold << ",\n";
      summary << "  \"debug_chunk_final_fraction\": " << debug_chunk.final_fraction << ",\n";
      summary << "  \"debug_chunk_component_count\": " << debug_chunk.component_count << ",\n";
      summary << "  \"debug_chunk_grouped_box_count\": " << debug_chunk.grouped_box_count << ",\n";
    }
    summary << "  \"projected_grouped_box_count\": " << global_merged.projected_boxes.size() << ",\n";
    summary << "  \"merged_grouped_box_count\": " << global_merged.merged_boxes.size() << ",\n";
    summary << "  \"stage_profile_json\": \"" << json_escape(stage_profile_path.string()) << "\",\n";
    summary << "  \"corrected_resized_npy\": \"" << json_escape(corrected_resized_path.string()) << "\",\n";
    summary << "  \"projected_grouped_mask_npy\": \"" << json_escape(projected_grouped_mask_path.string()) << "\",\n";
    summary << "  \"projected_grouped_mask_pgm\": \"" << json_escape(projected_grouped_mask_pgm.string()) << "\",\n";
    summary << "  \"projected_grouped_score_npy\": \"" << json_escape(projected_grouped_score_path.string()) << "\",\n";
    summary << "  \"merged_box_mask_npy\": \"" << json_escape(merged_box_mask_path.string()) << "\",\n";
    summary << "  \"merged_box_mask_pgm\": \"" << json_escape(merged_box_mask_pgm.string()) << "\",\n";
    summary << "  \"final_mask_npy\": \"" << json_escape(final_mask_path.string()) << "\",\n";
    summary << "  \"final_mask_pgm\": \"" << json_escape(final_mask_pgm.string()) << "\"";
    if (options.live_mask_path.has_value()) {
      summary << ",\n  \"live_mask_path\": \"" << json_escape(options.live_mask_path->string()) << "\"";
    }
    if (live_comparison.available) {
      summary << ",\n  \"live_mask_pixel_agreement\": " << live_comparison.pixel_agreement;
      summary << ",\n  \"live_mask_iou\": " << live_comparison.intersection_over_union;
      summary << ",\n  \"offline_foreground_fraction\": " << live_comparison.offline_foreground_fraction;
      summary << ",\n  \"live_foreground_fraction\": " << live_comparison.live_foreground_fraction;
    }
    summary << "\n}\n";

    if (options.verbose) {
      std::cout << "Offline DINO validation\n";
      std::cout << "  tensor: " << options.tensor_path << "\n";
      std::cout << "  config: " << options.config_path << "\n";
      std::cout << "  runtime backend: " << runtime_config.inference_backend << "\n";
      std::cout << "  ignore bins/side: " << ignore_bins_per_side << "\n";
      std::cout << "  chunk count: " << chunk_plan.size() << "\n";
      std::cout << "  legacy fast gray preprocess: " << (config.legacy_fast_gray_preprocess ? "true" : "false") << "\n";
      std::cout << "  projected grouped boxes: " << global_merged.projected_boxes.size() << "\n";
      std::cout << "  merged grouped boxes: " << global_merged.merged_boxes.size() << "\n";
      if (debug_chunk_result.has_value()) {
        const auto& debug_chunk = *debug_chunk_result;
        std::cout << "  debug chunk: " << debug_chunk.chunk_index
                  << " rows=(" << debug_chunk.row_start << ", " << debug_chunk.row_stop << ")\n";
        std::cout << "  debug chunk thresholds: freq=" << debug_chunk.seed_freq_threshold
                  << " res=" << debug_chunk.seed_res_threshold
                  << " combined=" << debug_chunk.combined_threshold << "\n";
        std::cout << "  debug chunk dino threshold: " << debug_chunk.dino_threshold
                  << " runtime_final=" << debug_chunk.runtime_final_threshold << "\n";
        std::cout << "  debug chunk grouped boxes: " << debug_chunk.grouped_box_count << "\n";
      }
      if (live_comparison.available) {
        std::cout << "  live mask agreement: " << live_comparison.pixel_agreement
                  << " IoU=" << live_comparison.intersection_over_union << "\n";
      }
      std::cout << "  timing summary:\n";
      print_stage_phase_summary(collect_phase_entries(profiler, debug_profiler, false), std::cout, 10, "full pass");
      print_stage_phase_summary(collect_phase_entries(profiler, debug_profiler, true), std::cout, 10, "debug rerun only");
      std::cout << "  stage profile: " << stage_profile_path << "\n";
      std::cout << "  summary: " << summary_path << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "offline_dino_validator_performance failed: " << error.what() << "\n";
    return 1;
  }
}
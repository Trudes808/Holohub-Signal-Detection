// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "coherent_power_signal_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace {

enum TimingStageIndex : size_t {
  kInputStage = 0,
  kPowerDbStage,
  kHostPipelineStage,
  kDeviceCopyStage,
  kMaskSaveStage,
  kTotalStage,
};

constexpr std::array<const char*, holoscan::ops::CoherentPowerSignalDetector::kTimingStageCount>
    kTimingStageNames = {
        "input_ms",
        "power_db_ms",
        "host_pipeline_ms",
        "device_copy_ms",
        "mask_save_ms",
        "total_ms",
    };

struct ChunkPlanEntry {
  int chunk_index = 0;
  int row_start = 0;
  int row_stop = 0;
  double freq_start_hz = 0.0;
  double freq_stop_hz = 0.0;
};

struct DetectionChunkResult {
  ChunkPlanEntry chunk;
  std::vector<float> coherence_px;
  std::vector<float> power_px;
  std::vector<float> score_px;
  std::vector<uint8_t> support_px;
  std::vector<uint8_t> mask_px;
  std::vector<uint8_t> grouped_mask;
  std::vector<uint8_t> valid_row_mask;
  std::vector<uint8_t> valid_score_mask;
  std::vector<std::array<int, 4>> grouped_boxes;
  float support_threshold = 0.0f;
  float score_threshold = 0.0f;
};

struct GroupingResult {
  std::vector<uint8_t> grouped_mask;
  std::vector<std::array<int, 4>> boxes;
  float peak_score_floor = 0.0f;
};

struct PipelineSummary {
  std::vector<float> final_mask;
  int subsection_count = 0;
  int grouped_box_count = 0;
  int ignore_bins_per_side = 0;
  float merged_threshold = 0.0f;
  float seed_threshold = 0.0f;
};

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

constexpr float kPi = 3.14159265358979323846f;

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

float clamp_float(float value, float low, float high) {
  return std::max(low, std::min(high, value));
}

int clamp_int(int value, int low, int high) {
  return std::max(low, std::min(high, value));
}

size_t flat_index(int rows, int cols, int row, int col) {
  return static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col);
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

GroupingResult group_signal_mask_regions(const std::vector<uint8_t>& mask,
                                         const std::vector<float>& score_map,
                                         int rows,
                                         int cols,
                                         const std::vector<uint8_t>& valid_row_mask,
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

  std::vector<uint8_t> bridged_mask = working_mask;
  binary_close_freq(bridged_mask, rows, cols, bridge_freq_px);
  fill_nearly_continuous_time_gaps(bridged_mask, rows, cols, bridge_time_px, time_continuity_ratio);

  std::vector<uint8_t> grouped_mask(mask.size(), 0);
  std::vector<uint8_t> visited(mask.size(), 0);
  std::vector<std::array<int, 4>> boxes;
  const std::array<std::pair<int, int>, 4> neighbors{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
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
      float score_peak = -std::numeric_limits<float>::infinity();
      while (!queue.empty()) {
        const auto [current_row, current_col] = queue.front();
        queue.pop();
        const size_t current_index = flat_index(rows, cols, current_row, current_col);
        component.push_back(current_index);
        min_row = std::min(min_row, current_row);
        max_row = std::max(max_row, current_row);
        min_col = std::min(min_col, current_col);
        max_col = std::max(max_col, current_col);
        score_peak = std::max(score_peak, score_map[current_index]);
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

      const int freq_start = min_row;
      const int freq_stop = max_row + 1;
      const int time_start = min_col;
      const int time_stop = max_col + 1;
      const int freq_span = freq_stop - freq_start;
      const int time_span = time_stop - time_start;
      const int filled_area = static_cast<int>(component.size());
      const int bbox_area = std::max(freq_span * time_span, 1);
      const float density = static_cast<float>(filled_area) / static_cast<float>(bbox_area);
      const bool keep_component = filled_area >= min_component_size &&
                                  freq_span >= min_freq_span_px &&
                                  time_span >= min_time_span_px &&
                                  density >= min_density &&
                                  score_peak >= peak_score_floor;
      if (!keep_component) {
        continue;
      }
      for (const size_t index : component) {
        grouped_mask[index] = 1;
      }
      boxes.push_back({freq_start, freq_stop, time_start, time_stop});
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
                                   const std::vector<std::array<int, 4>>& boxes,
                                   const std::vector<uint8_t>& valid_row_mask) {
  std::vector<uint8_t> mask(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  for (const auto& box : boxes) {
    const int freq_start = clamp_int(box[0], 0, rows);
    const int freq_stop = clamp_int(box[1], freq_start, rows);
    const int time_start = clamp_int(box[2], 0, cols);
    const int time_stop = clamp_int(box[3], time_start, cols);
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

std::vector<float> apply_frontend_correction(const std::vector<float>& power_db,
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
  std::vector<float> corrected(power_db.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    boost_db_out[static_cast<size_t>(row)] = clamp_float(reference_db - response[static_cast<size_t>(row)], 0.0f, max_boost_db);
    for (int col = 0; col < cols; ++col) {
      corrected[flat_index(rows, cols, row, col)] =
          power_db[flat_index(rows, cols, row, col)] + boost_db_out[static_cast<size_t>(row)];
    }
  }
  return corrected;
}

DetectionChunkResult detect_chunk_coherent_power(const std::vector<float>& corrected_chunk,
                                                 int rows,
                                                 int cols,
                                                 const ChunkPlanEntry& chunk,
                                                 const std::vector<uint8_t>& chunk_valid_row_mask,
                                                 double coherence_weight,
                                                 double power_weight,
                                                 double support_q,
                                                 double final_q,
                                                 int min_component_size,
                                                 int grouping_bridge_freq_px,
                                                 int grouping_bridge_time_px,
                                                 int grouping_min_component_size,
                                                 int grouping_min_freq_span_px,
                                                 int grouping_min_time_span_px,
                                                 double grouping_min_density) {
  DetectionChunkResult result;
  result.chunk = chunk;
  result.valid_row_mask = chunk_valid_row_mask;
  result.valid_score_mask = valid_row_mask_to_full_mask(chunk_valid_row_mask, cols);

  auto [coherence_px, energy_px, gate_px] = multi_scale_structure_tensor_gate(corrected_chunk, rows, cols);
  (void)energy_px;
  (void)gate_px;
  auto power_support = local_relative_power_support_map(corrected_chunk,
                                                        rows,
                                                        cols,
                                                        chunk_valid_row_mask,
                                                        30.0f,
                                                        9,
                                                        33);
  coherence_px = normalize_map01_local(coherence_px, 5.0f, 99.0f);
  power_support = normalize_map01_local(power_support, 5.0f, 95.0f);
  result.coherence_px = coherence_px;
  result.power_px = power_support;

  result.score_px.assign(corrected_chunk.size(), 0.0f);
  for (size_t index = 0; index < corrected_chunk.size(); ++index) {
    result.score_px[index] = static_cast<float>(coherence_weight) * coherence_px[index] +
                             static_cast<float>(power_weight) * power_support[index];
  }
  result.score_px = normalize_map01_local(result.score_px, 5.0f, 95.0f);
  for (size_t index = 0; index < result.valid_score_mask.size(); ++index) {
    if (result.valid_score_mask[index]) {
      continue;
    }
    result.coherence_px[index] = 0.0f;
    result.power_px[index] = 0.0f;
    result.score_px[index] = 0.0f;
  }

  const auto valid_scores = collect_masked_values(result.score_px, result.valid_score_mask);
  result.support_threshold = robust_high_quantile_threshold(valid_scores, static_cast<float>(support_q));
  result.support_px.assign(result.score_px.size(), 0);
  for (size_t index = 0; index < result.score_px.size(); ++index) {
    result.support_px[index] = (result.valid_score_mask[index] && result.score_px[index] >= result.support_threshold) ? 1 : 0;
  }
  result.support_px = smooth_binary_label_map(result.support_px,
                                              rows,
                                              cols,
                                              1,
                                              std::max(3, min_component_size / 2));

  std::vector<uint8_t> final_mask_source(result.score_px.size(), 0);
  for (size_t index = 0; index < result.score_px.size(); ++index) {
    final_mask_source[index] = (result.valid_score_mask[index] && result.support_px[index]) ? 1 : 0;
  }
  const auto final_scores = collect_masked_values(result.score_px, final_mask_source);
  result.score_threshold = final_scores.empty() ? result.support_threshold :
                           robust_high_quantile_threshold(final_scores, static_cast<float>(final_q));
  result.mask_px.assign(result.score_px.size(), 0);
  for (size_t index = 0; index < result.score_px.size(); ++index) {
    result.mask_px[index] = (result.valid_score_mask[index] && result.support_px[index] &&
                             result.score_px[index] >= result.score_threshold) ? 1 : 0;
  }
  result.mask_px = smooth_binary_label_map(result.mask_px, rows, cols, 1, min_component_size);

  const auto grouping = group_signal_mask_regions(result.mask_px,
                                                  result.score_px,
                                                  rows,
                                                  cols,
                                                  chunk_valid_row_mask,
                                                  grouping_bridge_freq_px,
                                                  grouping_bridge_time_px,
                                                  std::max(grouping_min_component_size, min_component_size),
                                                  grouping_min_freq_span_px,
                                                  grouping_min_time_span_px,
                                                  static_cast<float>(grouping_min_density));
  result.grouped_mask = grouping.grouped_mask;
  result.grouped_boxes = grouping.boxes;
  return result;
}

std::array<int, 4> scale_box(const std::array<int, 4>& box,
                             int src_rows,
                             int src_cols,
                             int dst_rows,
                             int dst_cols) {
  if (src_rows == dst_rows && src_cols == dst_cols) {
    return box;
  }
  const float row_scale = static_cast<float>(dst_rows) / static_cast<float>(std::max(src_rows, 1));
  const float col_scale = static_cast<float>(dst_cols) / static_cast<float>(std::max(src_cols, 1));
  const int freq_start = clamp_int(static_cast<int>(std::floor(static_cast<float>(box[0]) * row_scale)), 0, dst_rows);
  const int freq_stop = clamp_int(static_cast<int>(std::ceil(static_cast<float>(box[1]) * row_scale)), freq_start, dst_rows);
  const int time_start = clamp_int(static_cast<int>(std::floor(static_cast<float>(box[2]) * col_scale)), 0, dst_cols);
  const int time_stop = clamp_int(static_cast<int>(std::ceil(static_cast<float>(box[3]) * col_scale)), time_start, dst_cols);
  return {freq_start, freq_stop, time_start, time_stop};
}

PipelineSummary run_reference_pipeline(const std::vector<float>& power_db,
                                       int src_rows,
                                       int src_cols,
                                       int dst_rows,
                                       int dst_cols,
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
                                       double coherence_power_support_q,
                                       double coherence_power_q,
                                       int min_component_size,
                                       double grouping_seed_score_q,
                                       int grouping_bridge_freq_px,
                                       int grouping_bridge_time_px,
                                       int grouping_min_component_size,
                                       int grouping_min_freq_span_px,
                                       int grouping_min_time_span_px,
                                       double grouping_min_density) {
  PipelineSummary summary;
  if (resolution_hz <= 0.0 && ignore_bins_per_side == 0 && ignore_sideband_percent > 0.0) {
    ignore_bins_per_side = static_cast<int>(std::floor(static_cast<double>(src_rows) * ignore_sideband_percent / 100.0));
    ignore_bins_per_side = std::clamp(ignore_bins_per_side, 0, std::max(0, (src_rows - 16) / 2));
  }
  summary.ignore_bins_per_side = ignore_bins_per_side;

  std::vector<uint8_t> valid_row_mask(static_cast<size_t>(src_rows), 1);
  for (int row = 0; row < ignore_bins_per_side; ++row) {
    valid_row_mask[static_cast<size_t>(row)] = 0;
    valid_row_mask[static_cast<size_t>(src_rows - 1 - row)] = 0;
  }

  std::vector<float> boost_db;
  const auto corrected_sxx_db = apply_frontend_correction(power_db,
                                                          src_rows,
                                                          src_cols,
                                                          valid_row_mask,
                                                          static_cast<float>(frontend_row_q),
                                                          static_cast<float>(frontend_reference_q),
                                                          static_cast<float>(frontend_smooth_sigma),
                                                          static_cast<float>(frontend_max_boost_db),
                                                          boost_db);
  (void)boost_db;

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
  for (const auto& chunk : chunk_plan) {
    const int chunk_rows = chunk.row_stop - chunk.row_start;
    std::vector<float> corrected_chunk(static_cast<size_t>(chunk_rows) * static_cast<size_t>(src_cols), 0.0f);
    std::vector<uint8_t> chunk_valid_row_mask(static_cast<size_t>(chunk_rows), 0);
    for (int row = 0; row < chunk_rows; ++row) {
      chunk_valid_row_mask[static_cast<size_t>(row)] = valid_row_mask[static_cast<size_t>(chunk.row_start + row)];
      for (int col = 0; col < src_cols; ++col) {
        corrected_chunk[flat_index(chunk_rows, src_cols, row, col)] =
            corrected_sxx_db[flat_index(src_rows, src_cols, chunk.row_start + row, col)];
      }
    }
    chunk_results.push_back(detect_chunk_coherent_power(corrected_chunk,
                                                        chunk_rows,
                                                        src_cols,
                                                        chunk,
                                                        chunk_valid_row_mask,
                                                        coherence_weight,
                                                        power_weight,
                                                        coherence_power_support_q,
                                                        coherence_power_q,
                                                        min_component_size,
                                                        grouping_bridge_freq_px,
                                                        grouping_bridge_time_px,
                                                        grouping_min_component_size,
                                                        grouping_min_freq_span_px,
                                                        grouping_min_time_span_px,
                                                        grouping_min_density));
  }

  std::vector<float> merged_coherence_sum(power_db.size(), 0.0f);
  std::vector<float> merged_power_sum(power_db.size(), 0.0f);
  std::vector<float> merged_weight(power_db.size(), 0.0f);
  std::vector<uint8_t> merged_support(power_db.size(), 0);
  for (const auto& chunk : chunk_results) {
    const int chunk_rows = chunk.chunk.row_stop - chunk.chunk.row_start;
    std::vector<float> chunk_weights(static_cast<size_t>(chunk_rows), 1.0f);
    if (chunk_rows > 2) {
      for (int row = 0; row < chunk_rows; ++row) {
        const float phase = static_cast<float>(row) / static_cast<float>(chunk_rows - 1);
        const float base = 0.5f - 0.5f * std::cos(2.0f * kPi * phase);
        chunk_weights[static_cast<size_t>(row)] = 0.2f + 0.8f * base;
      }
    }
    for (int row = 0; row < chunk_rows; ++row) {
      for (int col = 0; col < src_cols; ++col) {
        const size_t chunk_index = flat_index(chunk_rows, src_cols, row, col);
        const size_t merged_index = flat_index(src_rows, src_cols, chunk.chunk.row_start + row, col);
        const float weight = chunk.valid_score_mask[chunk_index] ? chunk_weights[static_cast<size_t>(row)] : 0.0f;
        merged_coherence_sum[merged_index] += chunk.coherence_px[chunk_index] * weight;
        merged_power_sum[merged_index] += chunk.power_px[chunk_index] * weight;
        merged_weight[merged_index] += weight;
        merged_support[merged_index] = merged_support[merged_index] || chunk.support_px[chunk_index];
      }
    }
  }

  std::vector<float> merged_coherence(power_db.size(), 0.0f);
  std::vector<float> merged_power(power_db.size(), 0.0f);
  std::vector<float> combined_score(power_db.size(), 0.0f);
  std::vector<uint8_t> overlap_mask(power_db.size(), 0);
  for (size_t index = 0; index < power_db.size(); ++index) {
    if (merged_weight[index] > 0.0f) {
      merged_coherence[index] = merged_coherence_sum[index] / merged_weight[index];
      merged_power[index] = merged_power_sum[index] / merged_weight[index];
      overlap_mask[index] = 1;
    }
    combined_score[index] = static_cast<float>(coherence_weight) * merged_coherence[index] +
                            static_cast<float>(power_weight) * merged_power[index];
  }
  auto valid_score_mask = valid_row_mask_to_full_mask(valid_row_mask, src_cols);
  for (size_t index = 0; index < valid_score_mask.size(); ++index) {
    valid_score_mask[index] = (valid_score_mask[index] && overlap_mask[index]) ? 1 : 0;
  }
  auto merged_score = normalize_map01_masked(combined_score, valid_score_mask, 5.0f, 95.0f);
  std::vector<float> threshold_values;
  threshold_values.reserve(merged_score.size());
  for (size_t index = 0; index < merged_score.size(); ++index) {
    if (valid_score_mask[index] && merged_support[index]) {
      threshold_values.push_back(merged_score[index]);
    }
  }
  summary.merged_threshold = threshold_values.empty() ? 1.0f :
                             robust_high_quantile_threshold(threshold_values, static_cast<float>(coherence_power_q));

  std::vector<uint8_t> raw_merged_mask(merged_score.size(), 0);
  for (size_t index = 0; index < merged_score.size(); ++index) {
    raw_merged_mask[index] = (valid_score_mask[index] && merged_support[index] && merged_score[index] >= summary.merged_threshold) ? 1 : 0;
  }
  raw_merged_mask = smooth_binary_label_map(raw_merged_mask, src_rows, src_cols, 1, min_component_size);

  std::vector<float> seed_values;
  seed_values.reserve(merged_score.size());
  for (size_t index = 0; index < merged_score.size(); ++index) {
    if (valid_score_mask[index] && merged_support[index]) {
      seed_values.push_back(merged_score[index]);
    }
  }
  summary.seed_threshold = seed_values.empty() ? 1.0f :
                           robust_high_quantile_threshold(seed_values, static_cast<float>(grouping_seed_score_q));
  std::vector<uint8_t> seed_mask(merged_score.size(), 0);
  for (size_t index = 0; index < merged_score.size(); ++index) {
    const bool seed = raw_merged_mask[index] || (merged_support[index] && merged_score[index] >= summary.seed_threshold);
    seed_mask[index] = seed ? 1 : 0;
  }
  const auto merged_grouping = group_signal_mask_regions(seed_mask,
                                                         merged_score,
                                                         src_rows,
                                                         src_cols,
                                                         valid_row_mask,
                                                         grouping_bridge_freq_px,
                                                         grouping_bridge_time_px,
                                                         std::max(grouping_min_component_size, min_component_size),
                                                         grouping_min_freq_span_px,
                                                         grouping_min_time_span_px,
                                                         static_cast<float>(grouping_min_density));
  summary.grouped_box_count = static_cast<int>(merged_grouping.boxes.size());

  std::vector<std::array<int, 4>> scaled_boxes;
  scaled_boxes.reserve(merged_grouping.boxes.size());
  for (const auto& box : merged_grouping.boxes) {
    scaled_boxes.push_back(scale_box(box, src_rows, src_cols, dst_rows, dst_cols));
  }
  std::vector<uint8_t> dst_valid_row_mask(static_cast<size_t>(dst_rows), 1);
  for (int row = 0; row < dst_rows; ++row) {
    const int src_row = clamp_int(static_cast<int>(std::floor(static_cast<float>(row) * static_cast<float>(src_rows) /
                                                              static_cast<float>(std::max(dst_rows, 1)))),
                                  0,
                                  src_rows - 1);
    dst_valid_row_mask[static_cast<size_t>(row)] = valid_row_mask[static_cast<size_t>(src_row)];
  }
  const auto final_mask_u8 = boxes_to_mask(dst_rows, dst_cols, scaled_boxes, dst_valid_row_mask);
  summary.final_mask.assign(final_mask_u8.size(), 0.0f);
  for (size_t index = 0; index < final_mask_u8.size(); ++index) {
    summary.final_mask[index] = final_mask_u8[index] ? 1.0f : 0.0f;
  }
  return summary;
}

}  // namespace

namespace holoscan::ops {

CoherentPowerSignalDetector::~CoherentPowerSignalDetector() {
  for (float*& buffer : power_db_device_buffers_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  for (float*& buffer : power_db_host_buffers_) {
    if (buffer != nullptr) {
      cudaFreeHost(buffer);
      buffer = nullptr;
    }
  }
}

void CoherentPowerSignalDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<coherent_power_in_t>("in");

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(input_height_, "input_height", "Input height", "Detector output height.", 256);
  spec.param(input_width_, "input_width", "Input width", "Detector output width.", 512);
  spec.param(emit_stride_, "emit_stride", "Emit stride", "Emit one output every N input frames per channel.", 1);
  spec.param(log_detections_, "log_detections", "Log detections", "If true, logs detector execution details.", false);
  spec.param(enable_mask_save_, "enable_mask_save", "Enable mask save", "Enable writing detector masks to disk for debug runs.", false);
  spec.param(save_every_n_frames_, "save_every_n_frames", "Save stride", "Save one detector mask every N frames per channel.", 1);
  spec.param(max_masks_per_channel_, "max_masks_per_channel", "Max masks per channel", "Maximum number of detector masks to save per channel for a run.", 5);
  spec.param(output_dir_, "output_dir", "Output directory", "Directory where detector masks are written.", std::string("/workspace/coherent_power_masks"));
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
  spec.param(coherence_power_support_q_, "coherence_power_support_q", "Support quantile", "Notebook-derived support quantile.", 0.82);
  spec.param(coherence_power_q_, "coherence_power_q", "Final quantile", "Notebook-derived final score quantile.", 0.92);
  spec.param(min_component_size_, "min_component_size", "Minimum component size", "Notebook-derived minimum component size.", 6);
  spec.param(grouping_seed_score_q_, "grouping_seed_score_q", "Grouping seed score quantile", "Notebook-derived grouping seed quantile.", 0.72);
  spec.param(grouping_bridge_freq_px_, "grouping_bridge_freq_px", "Grouping bridge frequency", "Notebook-derived grouping bridge size in frequency bins.", 33);
  spec.param(grouping_bridge_time_px_, "grouping_bridge_time_px", "Grouping bridge time", "Notebook-derived grouping bridge size in time bins.", 5);
  spec.param(grouping_min_component_size_, "grouping_min_component_size", "Grouping minimum component size", "Notebook-derived grouping minimum component size.", 24);
  spec.param(grouping_min_freq_span_px_, "grouping_min_freq_span_px", "Grouping minimum frequency span", "Notebook-derived grouping minimum frequency span.", 18);
  spec.param(grouping_min_time_span_px_, "grouping_min_time_span_px", "Grouping minimum time span", "Notebook-derived grouping minimum time span.", 2);
  spec.param(grouping_min_density_, "grouping_min_density", "Grouping minimum density", "Notebook-derived grouping minimum density.", 0.06);
  spec.param(timing_summary_enable_, "timing_summary_enable", "Timing summary enable", "Enable per-stage timing summaries.", true);
  spec.param(timing_summary_every_n_, "timing_summary_every_n", "Timing summary every N", "Emit timing summaries every N emitted frames per channel.", 16);
  spec.param(timing_summary_window_, "timing_summary_window", "Timing summary window", "Maximum number of emitted frames to accumulate before reset.", 16);
}

void CoherentPowerSignalDetector::initialize() {
  holoscan::Operator::initialize();

  frame_count_.assign(num_channels_.get(), 0);
  masks_saved_.assign(num_channels_.get(), 0);
  timing_stats_.assign(num_channels_.get(), ChannelTimingStats {});
  power_db_device_buffers_.assign(num_channels_.get(), nullptr);
  power_db_device_buffer_sizes_.assign(num_channels_.get(), 0);
  power_db_host_buffers_.assign(num_channels_.get(), nullptr);
  power_db_host_buffer_sizes_.assign(num_channels_.get(), 0);

  if (enable_mask_save_.get()) {
    std::filesystem::create_directories(output_dir_.get());
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

  const uint64_t frame_number = ++frame_count_[channel_number];
  const int emit_stride = std::max(1, emit_stride_.get());
  if ((frame_number % static_cast<uint64_t>(emit_stride)) != 0) {
    return;
  }

  const int src_rows = static_cast<int>(fft_tensor.Size(0));
  const int src_cols = static_cast<int>(fft_tensor.Size(1));
  const int dst_rows = std::max(1, input_height_.get());
  const int dst_cols = std::max(1, input_width_.get());
  if (src_rows <= 0 || src_cols <= 0) {
    HOLOSCAN_LOG_WARN("Coherent power detector received empty tensor on channel {}", channel_number);
    return;
  }

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
  const double resolution_hz = static_cast<double>(meta->get<uint64_t>("resolution", 0));
  if (resolution_hz > 0.0 && ignore_sideband_hz_.get() > 0.0) {
    ignore_bins_per_side = static_cast<int>(std::ceil(ignore_sideband_hz_.get() / resolution_hz));
    ignore_bins_per_side = std::clamp(ignore_bins_per_side, 0, std::max(0, (src_rows - 16) / 2));
  } else if (resolution_hz <= 0.0 && ignore_sideband_percent_.get() > 0.0) {
    ignore_bins_per_side = static_cast<int>(std::floor(static_cast<double>(src_rows) * ignore_sideband_percent_.get() / 100.0));
    ignore_bins_per_side = std::clamp(ignore_bins_per_side, 0, std::max(0, (src_rows - 16) / 2));
  }

  const int total_bins = src_rows * src_cols;
  const size_t power_db_bytes = static_cast<size_t>(total_bins) * sizeof(float);

  time_step_ms(kInputStage, [&] {
    if (power_db_host_buffer_sizes_[channel_number] != static_cast<size_t>(total_bins)) {
      if (power_db_host_buffers_[channel_number] != nullptr) {
        cudaFreeHost(power_db_host_buffers_[channel_number]);
        power_db_host_buffers_[channel_number] = nullptr;
      }
      auto alloc_result = cudaMallocHost(reinterpret_cast<void**>(&power_db_host_buffers_[channel_number]), power_db_bytes);
      if (alloc_result != cudaSuccess) {
        throw std::runtime_error(std::string("power_db host buffer allocation failed: ") + cudaGetErrorString(alloc_result));
      }
      power_db_host_buffer_sizes_[channel_number] = static_cast<size_t>(total_bins);
    }
  });

  if (power_db_device_buffer_sizes_[channel_number] != static_cast<size_t>(total_bins)) {
    if (power_db_device_buffers_[channel_number] != nullptr) {
      cudaFree(power_db_device_buffers_[channel_number]);
      power_db_device_buffers_[channel_number] = nullptr;
    }
    auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&power_db_device_buffers_[channel_number]), power_db_bytes);
    if (alloc_result != cudaSuccess) {
      throw std::runtime_error(std::string("power_db device buffer allocation failed: ") + cudaGetErrorString(alloc_result));
    }
    power_db_device_buffer_sizes_[channel_number] = static_cast<size_t>(total_bins);
  }

  time_step_ms(kPowerDbStage, [&] {
    constexpr int threads = 256;
    const int blocks = (total_bins + threads - 1) / threads;
    coherent_power_power_db_kernel<<<blocks, threads, 0, stream>>>(fft_tensor.Data(),
                                                                    src_rows,
                                                                    src_cols,
                                                                    power_db_device_buffers_[channel_number]);
    auto kernel_result = cudaGetLastError();
    if (kernel_result != cudaSuccess) {
      throw std::runtime_error(std::string("power_db kernel launch failed: ") + cudaGetErrorString(kernel_result));
    }
  });

  PipelineSummary pipeline_summary;
  time_step_ms(kHostPipelineStage, [&] {
    auto copy_result = cudaMemcpyAsync(power_db_host_buffers_[channel_number],
                                       power_db_device_buffers_[channel_number],
                                       power_db_bytes,
                                       cudaMemcpyDeviceToHost,
                                       stream);
    if (copy_result != cudaSuccess) {
      throw std::runtime_error(std::string("power_db device-to-host copy failed: ") + cudaGetErrorString(copy_result));
    }
    auto sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
      throw std::runtime_error(std::string("power_db synchronization failed: ") + cudaGetErrorString(sync_result));
    }
    std::vector<float> host_power_db(power_db_host_buffers_[channel_number],
                                     power_db_host_buffers_[channel_number] + static_cast<size_t>(total_bins));
    pipeline_summary = run_reference_pipeline(host_power_db,
                                              src_rows,
                                              src_cols,
                                              dst_rows,
                                              dst_cols,
                                              ignore_bins_per_side,
                                              resolution_hz,
                                              chunk_bandwidth_hz_.get(),
                                              chunk_overlap_hz_.get(),
                                              uncalibrated_chunk_fraction_.get(),
                                              uncalibrated_overlap_fraction_.get(),
                                              ignore_sideband_percent_.get(),
                                              frontend_row_q_.get(),
                                              frontend_reference_q_.get(),
                                              frontend_smooth_sigma_.get(),
                                              frontend_max_boost_db_.get(),
                                              coherence_weight_.get(),
                                              power_weight_.get(),
                                              coherence_power_support_q_.get(),
                                              coherence_power_q_.get(),
                                              min_component_size_.get(),
                                              grouping_seed_score_q_.get(),
                                              grouping_bridge_freq_px_.get(),
                                              grouping_bridge_time_px_.get(),
                                              grouping_min_component_size_.get(),
                                              grouping_min_freq_span_px_.get(),
                                              grouping_min_time_span_px_.get(),
                                              grouping_min_density_.get());
  });

  stage_ms[kDeviceCopyStage] = 0.0;

  auto maybe_save_mask = [&] {
    if (!enable_mask_save_.get()) {
      return;
    }
    const int save_stride = std::max(1, save_every_n_frames_.get());
    if ((frame_number % static_cast<uint64_t>(save_stride)) != 0) {
      return;
    }
    if (masks_saved_[channel_number] >= max_masks_per_channel_.get()) {
      return;
    }

    std::vector<uint8_t> image(pipeline_summary.final_mask.size(), 0);
    for (size_t idx = 0; idx < pipeline_summary.final_mask.size(); ++idx) {
      image[idx] = pipeline_summary.final_mask[idx] > 0.5f ? 255 : 0;
    }

    const auto path = make_mask_output_path(output_dir_.get(), channel_number, frame_number, dst_rows, dst_cols);
    if (!write_pgm(path, image, dst_cols, dst_rows)) {
      HOLOSCAN_LOG_ERROR("Failed to write coherent power mask image: {}", path);
      return;
    }

    ++masks_saved_[channel_number];
    if (log_detections_.get()) {
      HOLOSCAN_LOG_INFO("Saved coherent power mask for channel {} frame {} to {}",
                        channel_number,
                        frame_number,
                        path);
    }
  };

  time_step_ms(kMaskSaveStage, maybe_save_mask);

  stage_ms[kTotalStage] =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();

  meta->set("coherent_frame_number", frame_number);
  meta->set("coherent_mask_height", static_cast<uint32_t>(dst_rows));
  meta->set("coherent_mask_width", static_cast<uint32_t>(dst_cols));
  meta->set("coherent_backend", std::string("coherent_power_reference_v1"));
  meta->set("coherent_chunk_count", static_cast<uint32_t>(pipeline_summary.subsection_count));
  meta->set("coherent_grouped_box_count", static_cast<uint32_t>(pipeline_summary.grouped_box_count));
  meta->set("coherent_ignore_bins_per_side", pipeline_summary.ignore_bins_per_side);
  meta->set("coherent_merged_threshold", pipeline_summary.merged_threshold);
  meta->set("coherent_seed_threshold", pipeline_summary.seed_threshold);
  meta->set("coherent_pipeline_variant", std::string("frontend_chunked_grouped_box_mask_v1"));
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
  HOLOSCAN_LOG_INFO("{}", oss.str());
  stats = ChannelTimingStats {};
}

}  // namespace holoscan::ops
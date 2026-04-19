// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "dinov3_torch_runtime.hpp"

#include <cuda/std/complex>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using dino_complex = cuda::std::complex<float>;

struct ValidatorOptions {
  std::filesystem::path tensor_path;
  std::filesystem::path config_path;
  std::optional<std::filesystem::path> live_mask_path;
  std::filesystem::path output_dir;
  bool verbose = false;
};

struct ValidatorConfig {
  int input_height = 256;
  int input_width = 512;
  int patch_size = 16;
  double resolution_hz = 0.0;
  double span_hz = 0.0;
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
  double dino_group_score_q = 0.60;
  double pipeline_final_threshold = 0.20;
  double pipeline_gap_floor = 0.10;
  double pipeline_power_rescue_floor = 0.10;
  double pipeline_power_rescue_gain = 2.0;
  std::string inference_backend = "torchscript";
  std::string model_script_path = "/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts";
  std::string torchscript_init_mode = "load_cuda_eval";
  std::string torch_dtype = "fp32";
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
  float seed_freq_threshold = 1.0f;
  float seed_res_threshold = 1.0f;
  float grow_freq_threshold = 1.0f;
  float grow_res_threshold = 1.0f;
  float combined_threshold = 1.0f;
  float final_fraction = 0.0f;
  float connected_fraction = 0.0f;
  int component_count = 0;
};

template <typename T>
T clamp_value(T value, T low, T high) {
  return value < low ? low : (value > high ? high : value);
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

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

ValidatorConfig load_config(const std::filesystem::path& path) {
  const std::string text = read_text_file(path);
  ValidatorConfig config;
  config.input_height = extract_number<int>(text, "input_height").value_or(config.input_height);
  config.input_width = extract_number<int>(text, "input_width").value_or(config.input_width);
  config.patch_size = extract_number<int>(text, "patch_size").value_or(config.patch_size);
  config.resolution_hz = extract_number<double>(text, "resolution").value_or(config.resolution_hz);
  config.span_hz = extract_number<double>(text, "span").value_or(config.span_hz);
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
  config.dino_group_score_q = extract_number<double>(text, "dino_group_score_q").value_or(config.dino_group_score_q);
  config.pipeline_final_threshold = extract_number<double>(text, "pipeline_final_threshold").value_or(config.pipeline_final_threshold);
  config.pipeline_gap_floor = extract_number<double>(text, "pipeline_gap_floor").value_or(config.pipeline_gap_floor);
  config.pipeline_power_rescue_floor = extract_number<double>(text, "pipeline_power_rescue_floor").value_or(config.pipeline_power_rescue_floor);
  config.pipeline_power_rescue_gain = extract_number<double>(text, "pipeline_power_rescue_gain").value_or(config.pipeline_power_rescue_gain);
  config.inference_backend = extract_yaml_string(text, "inference_backend").value_or(config.inference_backend);
  config.model_script_path = extract_yaml_string(text, "model_script_path").value_or(config.model_script_path);
  config.torchscript_init_mode = extract_yaml_string(text, "torchscript_init_mode").value_or(config.torchscript_init_mode);
  config.torch_dtype = extract_yaml_string(text, "torch_dtype").value_or(config.torch_dtype);
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
  array.rows = std::stoi(shape_match[1].str());
  array.cols = std::stoi(shape_match[2].str());
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
  std::ofstream out(path, std::ios::binary);
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
  return out.good();
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
  for (size_t index = 0; index < tensor.values.size(); ++index) {
    const float real = tensor.values[index].real();
    const float imag = tensor.values[index].imag();
    power_db[index] = 10.0f * std::log10(real * real + imag * imag + 1.0e-12f);
  }
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
  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += power_db[flat_index(cols, row, col)];
    }
    row_mean[static_cast<size_t>(row)] = sum / static_cast<float>(std::max(cols, 1));
  }
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

  std::vector<float> corrected(power_db.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    const float boost = std::min(std::max(reference_level_out - row_smooth[static_cast<size_t>(row)], 0.0f),
                                 static_cast<float>(config.frontend_correction_max_boost_db));
    for (int col = 0; col < cols; ++col) {
      corrected[flat_index(cols, row, col)] = power_db[flat_index(cols, row, col)] + boost;
    }
  }
  return corrected;
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
  const size_t index = static_cast<size_t>(std::llround(q * static_cast<double>(values.size() - 1)));
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(index), values.end());
  return values[index];
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

struct ComponentLabelling {
  std::vector<int> labels;
  std::vector<int> sizes;
};

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
        const int flat = pending[pending_head++];
        const int cur_row = flat / cols;
        const int cur_col = flat % cols;
        ++component_size;
        for (size_t neighbor = 0; neighbor < d_row.size(); ++neighbor) {
          const int next_row = cur_row + d_row[neighbor];
          const int next_col = cur_col + d_col[neighbor];
          if (next_row < 0 || next_row >= rows || next_col < 0 || next_col >= cols) {
            continue;
          }
          const size_t next_flat = flat_index(cols, next_row, next_col);
          if (!mask[next_flat] || result.labels[next_flat] != 0) {
            continue;
          }
          result.labels[next_flat] = next_label;
          pending.push_back(static_cast<int>(next_flat));
        }
      }
      result.sizes.push_back(component_size);
    }
  }
  return result;
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
                                                 const std::vector<float>& coherence_gate_map,
                                                 const std::vector<uint8_t>& valid_mask,
                                                 int rows,
                                                 int cols) {
  HybridPostprocessResult result;
  result.mask.assign(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0);
  if (dino_score.size() != result.mask.size() ||
      coherence_gate_map.size() != result.mask.size() ||
      valid_mask.size() != result.mask.size()) {
    return result;
  }

  std::vector<float> base_map(result.mask.size(), 0.0f);
  for (size_t index = 0; index < base_map.size(); ++index) {
    base_map[index] = dino_score[index] * coherence_gate_map[index];
  }
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
  auto final_mask = keep_large_components(seed_mask, rows, cols, 8);
  for (size_t index = 0; index < final_mask.size(); ++index) {
    final_mask[index] = (final_mask[index] && valid_mask[index]) ? 1 : 0;
  }
  final_mask = keep_large_components(final_mask, rows, cols, 8, &result.component_count);
  result.final_fraction = mean_mask_value(final_mask);
  result.connected_fraction = connected_fraction(final_mask, valid_mask);
  result.mask = std::move(final_mask);
  return result;
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
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--help") {
      std::cout << "Usage: " << argv[0] << " --tensor-npy PATH --config FILE [--live-mask PATH] [--output-dir DIR] [--verbose]\n";
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
    const ValidatorConfig config = load_config(options.config_path);
    const CanonicalTensor tensor = load_canonical_tensor(options.tensor_path);

    std::filesystem::create_directories(options.output_dir);

    const auto power_db = power_db_from_tensor(tensor);
    float frontend_reference_level = 0.0f;
    const auto corrected_db = frontend_corrected_db(power_db, tensor.rows, tensor.cols, config, frontend_reference_level);

    double resolution_hz = config.resolution_hz;
    if ((!std::isfinite(resolution_hz) || resolution_hz <= 0.0) && config.span_hz > 0.0 && tensor.rows > 0) {
      resolution_hz = config.span_hz / static_cast<double>(tensor.rows);
    }
    int ignore_bins_per_side = 0;
    if (resolution_hz > 0.0 && config.ignore_sideband_hz > 0.0) {
      ignore_bins_per_side = static_cast<int>(std::ceil(config.ignore_sideband_hz / resolution_hz));
      ignore_bins_per_side = std::clamp(ignore_bins_per_side, 0, std::max(0, (tensor.rows - config.patch_size) / 2));
    }

    const auto coherence_gate_full = coherence_gate(corrected_db, tensor.rows, tensor.cols, ignore_bins_per_side, config);
    const auto coherence_gate_resized = resize_bilinear(coherence_gate_full, tensor.rows, tensor.cols, config.input_height, config.input_width);
    const auto valid_mask = resize_valid_row_mask(tensor.rows, config.input_height, config.input_width, ignore_bins_per_side);

    float* power_db_device = nullptr;
    float* corrected_db_device = nullptr;
    const size_t source_bytes = power_db.size() * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&power_db_device), source_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&corrected_db_device), source_bytes) != cudaSuccess) {
      throw std::runtime_error("failed to allocate GPU buffers for offline DINO validator");
    }
    if (cudaMemcpy(power_db_device, power_db.data(), source_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(corrected_db_device, corrected_db.data(), source_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      throw std::runtime_error("failed to upload offline DINO validator tensors");
    }

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
    runtime_config.dino_group_score_q = config.dino_group_score_q;
    runtime_config.pipeline_final_threshold = config.pipeline_final_threshold;
    runtime_config.pipeline_gap_floor = config.pipeline_gap_floor;
    runtime_config.pipeline_power_rescue_floor = config.pipeline_power_rescue_floor;
    runtime_config.pipeline_power_rescue_gain = config.pipeline_power_rescue_gain;

    holoscan::ops::DinoTorchRuntimeInput runtime_input;
    runtime_input.src_rows = tensor.rows;
    runtime_input.src_cols = tensor.cols;
    runtime_input.dst_rows = config.input_height;
    runtime_input.dst_cols = config.input_width;
    runtime_input.patch_size = config.patch_size;
    runtime_input.cuda_stream = nullptr;
    runtime_input.resolution_hz = resolution_hz;
    runtime_input.span_hz = config.span_hz;
    runtime_input.power_db_device = power_db_device;
    runtime_input.corrected_db_device = corrected_db_device;

    const auto runtime_result = runtime.run(runtime_config, runtime_input);
    cudaFree(power_db_device);
    cudaFree(corrected_db_device);

    if (!runtime_result.success) {
      throw std::runtime_error("offline DINO runtime failed at " + runtime_result.error_stage + ": " + runtime_result.error_message + " (" + runtime_result.error_detail + ")");
    }
    if (runtime_result.final_mask.size() != static_cast<size_t>(config.input_height) * static_cast<size_t>(config.input_width)) {
      throw std::runtime_error("unexpected DINO score map size returned from runtime");
    }

    const auto hybrid_result = run_residual_veto_hybrid(runtime_result.final_mask,
                                                        coherence_gate_resized,
                                                        valid_mask,
                                                        config.input_height,
                                                        config.input_width);

    const auto corrected_resized = resize_bilinear(corrected_db, tensor.rows, tensor.cols, config.input_height, config.input_width);
    std::vector<float> hybrid_contrib(runtime_result.final_mask.size(), 0.0f);
    for (size_t index = 0; index < hybrid_contrib.size(); ++index) {
      hybrid_contrib[index] = runtime_result.final_mask[index] * coherence_gate_resized[index];
    }

    const auto power_db_path = options.output_dir / "offline_power_db.npy";
    const auto corrected_path = options.output_dir / "offline_corrected_db.npy";
    const auto corrected_resized_path = options.output_dir / "offline_corrected_resized.npy";
    const auto dino_score_path = options.output_dir / "offline_dino_score.npy";
    const auto coherence_gate_path = options.output_dir / "offline_coherence_gate.npy";
    const auto hybrid_contrib_path = options.output_dir / "offline_hybrid_contrib.npy";
    const auto final_mask_path = options.output_dir / "offline_final_mask.npy";
    const auto final_mask_pgm = options.output_dir / "offline_final_mask.pgm";
    const auto summary_path = options.output_dir / "offline_validation_summary.json";

    write_npy_2d(power_db_path, power_db.data(), power_db.size() * sizeof(float), tensor.rows, tensor.cols, "<f4");
    write_npy_2d(corrected_path, corrected_db.data(), corrected_db.size() * sizeof(float), tensor.rows, tensor.cols, "<f4");
    write_npy_2d(corrected_resized_path, corrected_resized.data(), corrected_resized.size() * sizeof(float), config.input_height, config.input_width, "<f4");
    write_npy_2d(dino_score_path, runtime_result.final_mask.data(), runtime_result.final_mask.size() * sizeof(float), config.input_height, config.input_width, "<f4");
    write_npy_2d(coherence_gate_path, coherence_gate_resized.data(), coherence_gate_resized.size() * sizeof(float), config.input_height, config.input_width, "<f4");
    write_npy_2d(hybrid_contrib_path, hybrid_contrib.data(), hybrid_contrib.size() * sizeof(float), config.input_height, config.input_width, "<f4");
    std::vector<float> final_mask_float(hybrid_result.mask.size(), 0.0f);
    for (size_t index = 0; index < hybrid_result.mask.size(); ++index) {
      final_mask_float[index] = hybrid_result.mask[index] ? 1.0f : 0.0f;
    }
    write_npy_2d(final_mask_path, final_mask_float.data(), final_mask_float.size() * sizeof(float), config.input_height, config.input_width, "<f4");
    write_pgm(final_mask_pgm, mask_to_u8(hybrid_result.mask), config.input_width, config.input_height);

    MaskComparison live_comparison;
    if (options.live_mask_path.has_value()) {
      int live_rows = 0;
      int live_cols = 0;
      const auto live_mask = load_pgm(*options.live_mask_path, live_rows, live_cols);
      if (live_rows == config.input_height && live_cols == config.input_width) {
        std::vector<uint8_t> live_binary(live_mask.size(), 0);
        for (size_t index = 0; index < live_mask.size(); ++index) {
          live_binary[index] = live_mask[index] >= 128 ? 1 : 0;
        }
        live_comparison = compare_masks(hybrid_result.mask, live_binary);
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
    summary << "  \"frontend_reference_level\": " << frontend_reference_level << ",\n";
    summary << "  \"runtime_backend_used\": \"" << json_escape(runtime_result.backend_used) << "\",\n";
    summary << "  \"runtime_dino_threshold\": " << runtime_result.dino_threshold << ",\n";
    summary << "  \"runtime_final_threshold\": " << runtime_result.final_threshold << ",\n";
    summary << "  \"hybrid_seed_freq_threshold\": " << hybrid_result.seed_freq_threshold << ",\n";
    summary << "  \"hybrid_seed_res_threshold\": " << hybrid_result.seed_res_threshold << ",\n";
    summary << "  \"hybrid_combined_threshold\": " << hybrid_result.combined_threshold << ",\n";
    summary << "  \"hybrid_final_fraction\": " << hybrid_result.final_fraction << ",\n";
    summary << "  \"hybrid_connected_fraction\": " << hybrid_result.connected_fraction << ",\n";
    summary << "  \"hybrid_component_count\": " << hybrid_result.component_count << ",\n";
    summary << "  \"power_db_npy\": \"" << json_escape(power_db_path.string()) << "\",\n";
    summary << "  \"corrected_db_npy\": \"" << json_escape(corrected_path.string()) << "\",\n";
    summary << "  \"corrected_resized_npy\": \"" << json_escape(corrected_resized_path.string()) << "\",\n";
    summary << "  \"dino_score_npy\": \"" << json_escape(dino_score_path.string()) << "\",\n";
    summary << "  \"coherence_gate_npy\": \"" << json_escape(coherence_gate_path.string()) << "\",\n";
    summary << "  \"hybrid_contrib_npy\": \"" << json_escape(hybrid_contrib_path.string()) << "\",\n";
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
      std::cout << "  runtime backend: " << runtime_result.backend_used << "\n";
      std::cout << "  ignore bins/side: " << ignore_bins_per_side << "\n";
      std::cout << "  dino threshold: " << runtime_result.dino_threshold << "\n";
      std::cout << "  hybrid seed thresholds: freq=" << hybrid_result.seed_freq_threshold
                << " res=" << hybrid_result.seed_res_threshold << "\n";
      std::cout << "  hybrid component count: " << hybrid_result.component_count << "\n";
      if (live_comparison.available) {
        std::cout << "  live mask agreement: " << live_comparison.pixel_agreement
                  << " IoU=" << live_comparison.intersection_over_union << "\n";
      }
      std::cout << "  summary: " << summary_path << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "offline_dino_validator failed: " << error.what() << "\n";
    return 1;
  }
}
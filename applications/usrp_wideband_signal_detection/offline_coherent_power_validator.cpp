// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "coherent_power_signal_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string json_escape_string(const std::string& value) {
  std::ostringstream escaped;
  for (const char ch : value) {
    switch (ch) {
      case '\\': escaped << "\\\\"; break;
      case '"': escaped << "\\\""; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default: escaped << ch; break;
    }
  }
  return escaped.str();
}

struct ValidatorOptions {
  std::filesystem::path metadata_path;
  std::filesystem::path output_dir;
  bool verbose = false;
};

struct SnapshotMetadata {
  int rows = 0;
  int cols = 0;
  int input_height = 256;
  int input_width = 512;
  double resolution_hz = 0.0;
  double sample_rate_hz = 0.0;
  double span_hz = 0.0;
  std::string tensor_axis_order = "";
  std::filesystem::path tensor_snapshot_path;
  std::optional<std::filesystem::path> power_db_snapshot_path;
  holoscan::ops::CoherentPowerReferenceConfig config;
};

std::string normalize_tensor_axis_order(std::string axis_order) {
  std::transform(axis_order.begin(), axis_order.end(), axis_order.begin(), [](unsigned char value) {
    return static_cast<char>(std::tolower(value));
  });
  if (axis_order == "time_frequency" || axis_order == "frequency_time") {
    return axis_order;
  }
  return "frequency_time";
}

struct ValidationWorkEstimate {
  int ignore_bins_per_side = 0;
  int valid_rows = 0;
  int chunk_rows = 0;
  int overlap_rows = 0;
  int chunk_count = 0;
  int first_chunk_rows = 0;
  int background_kernel_rows = 0;
  int background_kernel_cols = 0;
  int local_power_kernel_rows = 9;
  int local_power_kernel_cols = 33;
  double background_box_samples = 0.0;
};

template <typename T>
std::optional<T> extract_number(const std::string& text, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return std::nullopt;
  }
  std::istringstream stream(match[1].str());
  T value {};
  stream >> value;
  if (!stream.fail()) {
    return value;
  }
  return std::nullopt;
}

std::optional<std::string> extract_string(const std::string& text, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return std::nullopt;
  }
  return match[1].str();
}

std::optional<bool> extract_bool(const std::string& text, const std::string& key) {
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(true|false)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    return std::nullopt;
  }
  return match[1].str() == "true";
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

SnapshotMetadata load_snapshot_metadata(const std::filesystem::path& metadata_path) {
  const std::string text = read_text_file(metadata_path);
  SnapshotMetadata metadata;
  metadata.rows = extract_number<int>(text, "rows").value_or(0);
  metadata.cols = extract_number<int>(text, "cols").value_or(0);
  metadata.input_height = extract_number<int>(text, "input_height").value_or(256);
  metadata.input_width = extract_number<int>(text, "input_width").value_or(512);
  metadata.resolution_hz = extract_number<double>(text, "resolution_hz").value_or(0.0);
  metadata.sample_rate_hz = extract_number<double>(text, "sample_rate_hz").value_or(0.0);
  metadata.span_hz = extract_number<double>(text, "span_hz").value_or(0.0);
  metadata.tensor_axis_order = normalize_tensor_axis_order(extract_string(text, "tensor_axis_order").value_or(""));

  const auto tensor_path = extract_string(text, "tensor_snapshot_path");
  if (!tensor_path) {
    throw std::runtime_error("snapshot metadata missing tensor_snapshot_path");
  }
  metadata.tensor_snapshot_path = *tensor_path;

  const auto power_db_path = extract_string(text, "power_db_snapshot_path");
  if (power_db_path) {
    metadata.power_db_snapshot_path = std::filesystem::path(*power_db_path);
  }

  metadata.config.input_height = metadata.input_height;
  metadata.config.input_width = metadata.input_width;
  metadata.config.chunk_bandwidth_hz = extract_number<double>(text, "chunk_bandwidth_hz").value_or(metadata.config.chunk_bandwidth_hz);
  metadata.config.chunk_overlap_hz = extract_number<double>(text, "chunk_overlap_hz").value_or(metadata.config.chunk_overlap_hz);
  metadata.config.uncalibrated_chunk_fraction = extract_number<double>(text, "uncalibrated_chunk_fraction").value_or(metadata.config.uncalibrated_chunk_fraction);
  metadata.config.uncalibrated_overlap_fraction = extract_number<double>(text, "uncalibrated_overlap_fraction").value_or(metadata.config.uncalibrated_overlap_fraction);
  metadata.config.ignore_sideband_percent = extract_number<double>(text, "ignore_sideband_percent").value_or(metadata.config.ignore_sideband_percent);
  metadata.config.ignore_sideband_hz = extract_number<double>(text, "ignore_sideband_hz").value_or(metadata.config.ignore_sideband_hz);
  metadata.config.frontend_row_q = extract_number<double>(text, "frontend_row_q").value_or(metadata.config.frontend_row_q);
  metadata.config.frontend_reference_q = extract_number<double>(text, "frontend_reference_q").value_or(metadata.config.frontend_reference_q);
  metadata.config.frontend_smooth_sigma = extract_number<double>(text, "frontend_smooth_sigma").value_or(metadata.config.frontend_smooth_sigma);
  metadata.config.frontend_max_boost_db = extract_number<double>(text, "frontend_max_boost_db").value_or(metadata.config.frontend_max_boost_db);
  metadata.config.frontend_signal_cap_db = extract_number<double>(text, "frontend_signal_cap_db").value_or(metadata.config.frontend_signal_cap_db);
  metadata.config.coherence_weight = extract_number<double>(text, "coherence_weight").value_or(metadata.config.coherence_weight);
  metadata.config.power_weight = extract_number<double>(text, "power_weight").value_or(metadata.config.power_weight);
  metadata.config.power_assist_mode = extract_string(text, "power_assist_mode").value_or(metadata.config.power_assist_mode);
  metadata.config.power_floor_time_q = extract_number<double>(text, "power_floor_time_q").value_or(metadata.config.power_floor_time_q);
  metadata.config.power_floor_global_q = extract_number<double>(text, "power_floor_global_q").value_or(metadata.config.power_floor_global_q);
  metadata.config.power_excess_start_db = extract_number<double>(text, "power_excess_start_db").value_or(metadata.config.power_excess_start_db);
  metadata.config.power_excess_full_db = extract_number<double>(text, "power_excess_full_db").value_or(metadata.config.power_excess_full_db);
  metadata.config.power_local_blend = extract_number<double>(text, "power_local_blend").value_or(metadata.config.power_local_blend);
  metadata.config.coherence_source_mode = extract_string(text, "coherence_source_mode").value_or(metadata.config.coherence_source_mode);
  metadata.config.coherence_gate_start = extract_number<double>(text, "coherence_gate_start").value_or(metadata.config.coherence_gate_start);
  metadata.config.coherence_gate_full = extract_number<double>(text, "coherence_gate_full").value_or(metadata.config.coherence_gate_full);
  metadata.config.coherence_bridge_bias = extract_number<double>(text, "coherence_bridge_bias").value_or(metadata.config.coherence_bridge_bias);
  metadata.config.coherence_power_joint_weight = extract_number<double>(text, "coherence_power_joint_weight").value_or(metadata.config.coherence_power_joint_weight);
  metadata.config.score_threshold_mode = extract_string(text, "score_threshold_mode").value_or(metadata.config.score_threshold_mode);
  metadata.config.fixed_score_threshold = extract_number<double>(text, "fixed_score_threshold").value_or(metadata.config.fixed_score_threshold);
  metadata.config.coherence_power_support_q = extract_number<double>(text, "coherence_power_support_q").value_or(metadata.config.coherence_power_support_q);
  metadata.config.coherence_power_q = extract_number<double>(text, "coherence_power_q").value_or(metadata.config.coherence_power_q);
  metadata.config.min_component_size = extract_number<int>(text, "min_component_size").value_or(metadata.config.min_component_size);
  metadata.config.filter_detection_mask = extract_bool(text, "filter_detection_mask").value_or(metadata.config.filter_detection_mask);
  metadata.config.grouping_seed_score_q = extract_number<double>(text, "grouping_seed_score_q").value_or(metadata.config.grouping_seed_score_q);
  metadata.config.grouping_bridge_freq_px = extract_number<int>(text, "grouping_bridge_freq_px").value_or(metadata.config.grouping_bridge_freq_px);
  metadata.config.grouping_bridge_time_px = extract_number<int>(text, "grouping_bridge_time_px").value_or(metadata.config.grouping_bridge_time_px);
  metadata.config.grouping_min_component_size = extract_number<int>(text, "grouping_min_component_size").value_or(metadata.config.grouping_min_component_size);
  metadata.config.grouping_min_freq_span_px = extract_number<int>(text, "grouping_min_freq_span_px").value_or(metadata.config.grouping_min_freq_span_px);
  metadata.config.grouping_min_time_span_px = extract_number<int>(text, "grouping_min_time_span_px").value_or(metadata.config.grouping_min_time_span_px);
  metadata.config.grouping_min_density = extract_number<double>(text, "grouping_min_density").value_or(metadata.config.grouping_min_density);
  metadata.config.grouping_time_continuity_ratio = extract_number<double>(text, "grouping_time_continuity_ratio").value_or(metadata.config.grouping_time_continuity_ratio);
  return metadata;
}

struct NpyArray2D {
  std::string descr;
  int rows = 0;
  int cols = 0;
  std::vector<uint8_t> payload;
};

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
  const size_t payload_bytes = static_cast<size_t>(array.rows) * static_cast<size_t>(array.cols) *
                               (array.descr == "<c8" ? sizeof(float) * 2 : sizeof(float));
  array.payload.resize(payload_bytes);
  in.read(reinterpret_cast<char*>(array.payload.data()), static_cast<std::streamsize>(array.payload.size()));
  if (!in) {
    throw std::runtime_error("truncated npy payload in: " + path.string());
  }
  return array;
}

std::vector<holoscan::ops::coherent_power_complex> to_complex_tensor(const NpyArray2D& array) {
  if (array.descr != "<c8") {
    throw std::runtime_error("expected complex64 npy tensor");
  }
  std::vector<holoscan::ops::coherent_power_complex> output(static_cast<size_t>(array.rows) * static_cast<size_t>(array.cols));
  for (size_t index = 0; index < output.size(); ++index) {
    float components[2] {};
    std::memcpy(components, array.payload.data() + index * sizeof(float) * 2, sizeof(components));
    output[index] = holoscan::ops::coherent_power_complex(components[0], components[1]);
  }
  return output;
}

std::vector<holoscan::ops::coherent_power_complex> transpose_complex_tensor(
    const std::vector<holoscan::ops::coherent_power_complex>& input,
    int rows,
    int cols) {
  std::vector<holoscan::ops::coherent_power_complex> output(static_cast<size_t>(rows) * static_cast<size_t>(cols));
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      output[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] =
          input[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    }
  }
  return output;
}

std::vector<float> transpose_float_matrix(const std::vector<float>& input, int rows, int cols) {
  std::vector<float> output(static_cast<size_t>(rows) * static_cast<size_t>(cols), 0.0f);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      output[static_cast<size_t>(col) * static_cast<size_t>(rows) + static_cast<size_t>(row)] =
          input[static_cast<size_t>(row) * static_cast<size_t>(cols) + static_cast<size_t>(col)];
    }
  }
  return output;
}

std::vector<float> to_float_matrix(const NpyArray2D& array) {
  if (array.descr != "<f4") {
    throw std::runtime_error("expected float32 npy matrix");
  }
  std::vector<float> output(static_cast<size_t>(array.rows) * static_cast<size_t>(array.cols), 0.0f);
  std::memcpy(output.data(), array.payload.data(), array.payload.size());
  return output;
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

ValidationWorkEstimate estimate_validation_work(const SnapshotMetadata& metadata) {
  ValidationWorkEstimate estimate;
  if (metadata.rows <= 0 || metadata.cols <= 0) {
    return estimate;
  }

  if (metadata.config.ignore_sideband_percent > 0.0) {
    estimate.ignore_bins_per_side = static_cast<int>(std::floor(static_cast<double>(metadata.rows) * metadata.config.ignore_sideband_percent / 100.0));
    estimate.ignore_bins_per_side = std::clamp(estimate.ignore_bins_per_side, 0, std::max(0, (metadata.rows - 16) / 2));
  } else if (metadata.resolution_hz > 0.0 && metadata.config.ignore_sideband_hz > 0.0) {
    estimate.ignore_bins_per_side = static_cast<int>(std::ceil(metadata.config.ignore_sideband_hz / metadata.resolution_hz));
    estimate.ignore_bins_per_side = std::clamp(estimate.ignore_bins_per_side, 0, std::max(0, (metadata.rows - 16) / 2));
  }

  estimate.valid_rows = std::max(0, metadata.rows - 2 * estimate.ignore_bins_per_side);
  if (estimate.valid_rows <= 0) {
    return estimate;
  }

  if (metadata.resolution_hz > 0.0 && metadata.config.chunk_bandwidth_hz > 0.0) {
    estimate.chunk_rows = std::clamp(
        static_cast<int>(std::llround(metadata.config.chunk_bandwidth_hz / metadata.resolution_hz)),
        16,
        metadata.rows);
    estimate.overlap_rows = std::clamp(
        static_cast<int>(std::llround(metadata.config.chunk_overlap_hz / metadata.resolution_hz)),
        0,
        std::max(0, estimate.chunk_rows - 1));
  } else {
    estimate.chunk_rows = std::clamp(
        static_cast<int>(std::llround(static_cast<double>(estimate.valid_rows) * metadata.config.uncalibrated_chunk_fraction)),
        16,
        estimate.valid_rows);
    estimate.overlap_rows = std::clamp(
        static_cast<int>(std::llround(static_cast<double>(estimate.chunk_rows) * metadata.config.uncalibrated_overlap_fraction)),
        0,
        std::max(0, estimate.chunk_rows - 1));
  }

  const int step_rows = std::max(1, estimate.chunk_rows - estimate.overlap_rows);
  for (int start_index = 0; start_index < estimate.valid_rows; start_index += step_rows) {
    const int stop_index = std::min(start_index + estimate.chunk_rows, estimate.valid_rows);
    if (stop_index - start_index < 16) {
      if (estimate.chunk_count > 0) {
        break;
      }
      continue;
    }
    ++estimate.chunk_count;
    if (estimate.first_chunk_rows == 0) {
      estimate.first_chunk_rows = stop_index - start_index;
    }
    if (stop_index >= estimate.valid_rows) {
      break;
    }
  }

  estimate.background_kernel_rows = std::max(9, 2 * std::max(1, estimate.first_chunk_rows / 24) + 1);
  estimate.background_kernel_cols = std::max(9, 2 * std::max(1, metadata.cols / 24) + 1);
  estimate.background_box_samples = static_cast<double>(estimate.first_chunk_rows) *
                                    static_cast<double>(metadata.cols) *
                                    static_cast<double>(estimate.background_kernel_rows) *
                                    static_cast<double>(estimate.background_kernel_cols);
  return estimate;
}

std::string format_large_count(double value) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(1);
  if (value >= 1.0e9) {
    out << (value / 1.0e9) << "e9";
  } else if (value >= 1.0e6) {
    out << (value / 1.0e6) << "e6";
  } else if (value >= 1.0e3) {
    out << (value / 1.0e3) << "e3";
  } else {
    out << value;
  }
  return out.str();
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
  std::ostringstream header_builder;
  header_builder << "{'descr': '" << dtype_descr << "', 'fortran_order': False, 'shape': ("
                 << rows << ", " << cols << "), }";
  std::string header = header_builder.str();
  const size_t preamble_size = 10;
  const size_t padding = (16 - ((preamble_size + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');
  const uint16_t header_len = static_cast<uint16_t>(header.size());
  out.write("\x93NUMPY", 6);
  const unsigned char version[2] = {1, 0};
  out.write(reinterpret_cast<const char*>(version), 2);
  const unsigned char header_len_le[2] = {
      static_cast<unsigned char>(header_len & 0xFF),
      static_cast<unsigned char>((header_len >> 8) & 0xFF),
  };
  out.write(reinterpret_cast<const char*>(header_len_le), 2);
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(payload), static_cast<std::streamsize>(payload_bytes));
  return out.good();
}

std::vector<uint8_t> normalize_to_u8(const std::vector<float>& values) {
  std::vector<uint8_t> image(values.size(), 0);
  if (values.empty()) {
    return image;
  }
  const auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  const float min_value = *min_it;
  const float max_value = *max_it;
  const float span = std::max(max_value - min_value, 1e-6f);
  for (size_t index = 0; index < values.size(); ++index) {
    const float normalized = (values[index] - min_value) / span;
    image[index] = static_cast<uint8_t>(std::clamp(normalized, 0.0f, 1.0f) * 255.0f);
  }
  return image;
}

std::vector<uint8_t> overlay_mask_on_grayscale(const std::vector<float>& base, const std::vector<float>& mask) {
  auto image = normalize_to_u8(base);
  for (size_t index = 0; index < image.size() && index < mask.size(); ++index) {
    if (mask[index] > 0.5f) {
      image[index] = 255;
    }
  }
  return image;
}

ValidatorOptions parse_arguments(int argc, char** argv) {
  ValidatorOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--snapshot-json" && index + 1 < argc) {
      options.metadata_path = argv[++index];
    } else if (arg == "--output-dir" && index + 1 < argc) {
      options.output_dir = argv[++index];
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: " << argv[0] << " --snapshot-json PATH [--output-dir DIR] [--verbose]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  if (options.metadata_path.empty()) {
    throw std::runtime_error("--snapshot-json is required");
  }
  if (options.output_dir.empty()) {
    options.output_dir = options.metadata_path.parent_path() / "validator_artifacts";
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto run_start = std::chrono::steady_clock::now();
    auto stage_start = run_start;
    auto log_stage = [&](const std::string& message) {
      std::cout << message << std::endl;
    };
    auto log_stage_done = [&](const std::string& stage_name) {
      const auto now = std::chrono::steady_clock::now();
      const double stage_ms = std::chrono::duration<double, std::milli>(now - stage_start).count();
      const double total_ms = std::chrono::duration<double, std::milli>(now - run_start).count();
      std::cout << "[offline_coherent_power_validator] " << stage_name << " completed in " << std::fixed
                << std::setprecision(1) << stage_ms << " ms"
                << " (total " << total_ms << " ms)" << std::endl;
      stage_start = now;
    };

    const ValidatorOptions options = parse_arguments(argc, argv);
    if (options.verbose) {
      log_stage("[offline_coherent_power_validator] loading snapshot metadata...");
    }
    const SnapshotMetadata metadata = load_snapshot_metadata(options.metadata_path);
    if (options.verbose) {
      log_stage_done("snapshot metadata load");
      log_stage("[offline_coherent_power_validator] creating output directory...");
    }
    std::filesystem::create_directories(options.output_dir);
    if (options.verbose) {
      log_stage_done("output directory setup");
      log_stage("[offline_coherent_power_validator] loading tensor snapshot...");
    }

    const NpyArray2D tensor_array = load_npy_2d(metadata.tensor_snapshot_path);
    if (tensor_array.rows != metadata.rows || tensor_array.cols != metadata.cols) {
      throw std::runtime_error("snapshot tensor shape does not match metadata sidecar");
    }
    auto tensor = to_complex_tensor(tensor_array);
    int validator_rows = metadata.rows;
    int validator_cols = metadata.cols;
    if (metadata.tensor_axis_order == "time_frequency") {
      tensor = transpose_complex_tensor(tensor, metadata.rows, metadata.cols);
      validator_rows = metadata.cols;
      validator_cols = metadata.rows;
    }
    if (options.verbose) {
      log_stage_done("tensor snapshot load");
      const ValidationWorkEstimate estimate = estimate_validation_work(metadata);
      std::cout << "[offline_coherent_power_validator] tensor shape "
                << metadata.rows << "x" << metadata.cols
                << " (axis order " << metadata.tensor_axis_order << ", validator grid "
                << validator_rows << "x" << validator_cols << ")"
                << ", valid_rows=" << estimate.valid_rows
                << ", ignore_bins_per_side=" << estimate.ignore_bins_per_side
                << ", chunk_count=" << estimate.chunk_count
                << ", first_chunk_rows=" << estimate.first_chunk_rows
                << ", chunk_overlap_rows=" << estimate.overlap_rows
                << std::endl;
      std::cout << "[offline_coherent_power_validator] first-chunk background box filter "
                << estimate.background_kernel_rows << "x" << estimate.background_kernel_cols
                << ", local-power box filter "
                << estimate.local_power_kernel_rows << "x" << estimate.local_power_kernel_cols
                << ", estimated background samples=" << format_large_count(estimate.background_box_samples)
                << std::endl;
      log_stage("[offline_coherent_power_validator] running coherent validator pipeline (operator_live)...");
    }

    const auto result = holoscan::ops::run_coherent_power_live_validation(
        tensor,
        validator_rows,
        validator_cols,
        metadata.resolution_hz,
        metadata.config);
    if (options.verbose) {
      log_stage_done("coherent validator pipeline");
      log_stage("[offline_coherent_power_validator] writing validator artifacts...");
    }

    const auto power_db_npy = options.output_dir / "offline_power_db.npy";
    const auto corrected_npy = options.output_dir / "offline_corrected_sxx_db.npy";
    const auto merged_coherence_npy = options.output_dir / "offline_merged_coherence.npy";
    const auto merged_power_npy = options.output_dir / "offline_merged_power.npy";
    const auto merged_score_npy = options.output_dir / "offline_merged_score.npy";
    const auto raw_projected_mask_npy = options.output_dir / "offline_raw_projected_mask.npy";
    const auto mask_npy = options.output_dir / "offline_final_mask.npy";
    const auto power_db_pgm = options.output_dir / "offline_power_db_preview.pgm";
    const auto corrected_pgm = options.output_dir / "offline_corrected_preview.pgm";
    const auto raw_projected_mask_pgm = options.output_dir / "offline_raw_projected_mask.pgm";
    const auto mask_pgm = options.output_dir / "offline_final_mask.pgm";
    const auto overlay_pgm = options.output_dir / "offline_mask_overlay.pgm";
    const auto summary_json = options.output_dir / "offline_validation_summary.json";

    write_npy_2d(power_db_npy, result.power_db.data(), result.power_db.size() * sizeof(float), result.src_rows, result.src_cols, "<f4");
    write_npy_2d(corrected_npy, result.corrected_sxx_db.data(), result.corrected_sxx_db.size() * sizeof(float), result.src_rows, result.src_cols, "<f4");
    if (!result.merged_coherence.empty()) {
      write_npy_2d(merged_coherence_npy,
                   result.merged_coherence.data(),
                   result.merged_coherence.size() * sizeof(float),
                   result.src_rows,
                   result.src_cols,
                   "<f4");
    }
    if (!result.merged_power.empty()) {
      write_npy_2d(merged_power_npy,
                   result.merged_power.data(),
                   result.merged_power.size() * sizeof(float),
                   result.src_rows,
                   result.src_cols,
                   "<f4");
    }
    if (!result.merged_score.empty()) {
      write_npy_2d(merged_score_npy,
                   result.merged_score.data(),
                   result.merged_score.size() * sizeof(float),
                   result.src_rows,
                   result.src_cols,
                   "<f4");
    }
    if (!result.raw_projected_mask.empty()) {
      write_npy_2d(raw_projected_mask_npy,
                   result.raw_projected_mask.data(),
                   result.raw_projected_mask.size() * sizeof(float),
                   result.src_rows,
                   result.src_cols,
                   "<f4");
    }
    write_npy_2d(mask_npy, result.final_mask.data(), result.final_mask.size() * sizeof(float), result.dst_rows, result.dst_cols, "<f4");

    const auto power_db_preview = normalize_to_u8(result.power_db);
    const auto corrected_preview = normalize_to_u8(result.corrected_sxx_db);
    const auto raw_projected_mask_preview = normalize_to_u8(result.raw_projected_mask);
    const auto final_mask_preview = normalize_to_u8(result.final_mask);
    const auto overlay_preview = overlay_mask_on_grayscale(result.corrected_sxx_db, result.final_mask.size() == result.corrected_sxx_db.size() ? result.final_mask : std::vector<float>{});
    const size_t final_mask_nonzero_pixels = static_cast<size_t>(std::count_if(
      result.final_mask.begin(), result.final_mask.end(), [](float value) { return value > 0.5f; }));
    const int valid_rows = std::max(0, result.src_rows - (2 * result.ignore_bins_per_side));
    const size_t valid_mask_pixels = static_cast<size_t>(std::max(0, valid_rows)) * static_cast<size_t>(std::max(0, result.src_cols));
    const double final_mask_fill_ratio = valid_mask_pixels > 0
        ? static_cast<double>(final_mask_nonzero_pixels) / static_cast<double>(valid_mask_pixels)
        : 0.0;
    std::vector<size_t> grouped_box_order(result.grouped_boxes.size());
    std::iota(grouped_box_order.begin(), grouped_box_order.end(), static_cast<size_t>(0));
    std::sort(grouped_box_order.begin(), grouped_box_order.end(), [&](size_t lhs, size_t rhs) {
      return result.grouped_boxes[lhs].filled_area > result.grouped_boxes[rhs].filled_area;
    });
    const size_t grouped_box_debug_count = std::min<size_t>(5, grouped_box_order.size());
    write_pgm(power_db_pgm, power_db_preview, result.src_cols, result.src_rows);
    write_pgm(corrected_pgm, corrected_preview, result.src_cols, result.src_rows);
    if (!raw_projected_mask_preview.empty()) {
      write_pgm(raw_projected_mask_pgm, raw_projected_mask_preview, result.src_cols, result.src_rows);
    }
    write_pgm(mask_pgm, final_mask_preview, result.dst_cols, result.dst_rows);
    if (!overlay_preview.empty() && result.final_mask.size() == result.corrected_sxx_db.size()) {
      write_pgm(overlay_pgm, overlay_preview, result.src_cols, result.src_rows);
    }

    std::ofstream summary(summary_json, std::ios::binary);
    summary << "{\n";
    summary << "  \"metadata_path\": \"" << options.metadata_path.string() << "\",\n";
    summary << "  \"pipeline_mode_effective\": \"operator_live\",\n";
    summary << "  \"fast_performance_from_config\": " << (extract_string(read_text_file(options.metadata_path), "fast_performance").value_or("false")) << ",\n";
    summary << "  \"tensor_snapshot_path\": \"" << metadata.tensor_snapshot_path.string() << "\",\n";
    summary << "  \"rows\": " << result.src_rows << ",\n";
    summary << "  \"cols\": " << result.src_cols << ",\n";
    summary << "  \"input_height\": " << result.dst_rows << ",\n";
    summary << "  \"input_width\": " << result.dst_cols << ",\n";
        summary << "  \"tensor_axis_order\": \""
          << (metadata.tensor_axis_order.empty() ? "unspecified" : metadata.tensor_axis_order)
          << "\",\n";
    summary << "  \"frequency_axis_calibrated\": " << (result.frequency_axis_calibrated ? "true" : "false") << ",\n";
        summary << "  \"resolution_hz\": " << metadata.resolution_hz << ",\n";
        summary << "  \"sample_rate_hz\": " << result.sample_rate_hz << ",\n";
        summary << "  \"span_hz\": " << result.span_hz << ",\n";
    summary << "  \"ignore_bins_per_side\": " << result.ignore_bins_per_side << ",\n";
        summary << "  \"grouped_box_count\": " << result.grouped_box_count << ",\n";
    summary << "  \"final_mask_nonzero_pixels\": " << final_mask_nonzero_pixels << ",\n";
        summary << "  \"final_mask_fill_ratio\": " << final_mask_fill_ratio << ",\n";
    summary << "  \"merged_threshold\": " << result.merged_threshold << ",\n";
    summary << "  \"seed_threshold\": " << result.seed_threshold << ",\n";
        summary << "  \"power_db_npy\": \"" << power_db_npy.string() << "\",\n";
        if (!result.merged_coherence.empty()) {
          summary << "  \"merged_coherence_npy\": \"" << merged_coherence_npy.string() << "\",\n";
        } else {
          summary << "  \"merged_coherence_npy\": null,\n";
        }
        if (!result.merged_power.empty()) {
          summary << "  \"merged_power_npy\": \"" << merged_power_npy.string() << "\",\n";
        } else {
          summary << "  \"merged_power_npy\": null,\n";
        }
        if (!result.merged_score.empty()) {
          summary << "  \"merged_score_npy\": \"" << merged_score_npy.string() << "\",\n";
        } else {
          summary << "  \"merged_score_npy\": null,\n";
        }
        if (!result.raw_projected_mask.empty()) {
          summary << "  \"raw_projected_mask_npy\": \"" << raw_projected_mask_npy.string() << "\",\n";
          summary << "  \"raw_projected_mask_pgm\": \"" << raw_projected_mask_pgm.string() << "\",\n";
        } else {
          summary << "  \"raw_projected_mask_npy\": null,\n";
          summary << "  \"raw_projected_mask_pgm\": null,\n";
        }
        summary << "  \"largest_grouped_boxes\": [\n";
        for (size_t index = 0; index < grouped_box_debug_count; ++index) {
          const auto& box = result.grouped_boxes[grouped_box_order[index]];
          summary << "    {\"freq_start\": " << box.freq_start
                  << ", \"freq_stop\": " << box.freq_stop
                  << ", \"time_start\": " << box.time_start
                  << ", \"time_stop\": " << box.time_stop
                  << ", \"freq_span\": " << box.freq_span
                  << ", \"time_span\": " << box.time_span
                  << ", \"filled_area\": " << box.filled_area
                  << ", \"density\": " << box.density
                  << ", \"bbox_density\": " << box.bbox_density
                  << ", \"envelope_density\": " << box.envelope_density
                  << ", \"score_mean\": " << box.score_mean
                  << ", \"score_peak\": " << box.score_peak
                  << ", \"split_role\": \"" << json_escape_string(box.split_role) << "\"}"
                  << (index + 1 < grouped_box_debug_count ? "," : "") << "\n";
        }
        summary << "  ],\n";
    summary << "  \"corrected_sxx_db_npy\": \"" << corrected_npy.string() << "\",\n";
    summary << "  \"final_mask_npy\": \"" << mask_npy.string() << "\",\n";
    summary << "  \"power_db_preview_pgm\": \"" << power_db_pgm.string() << "\",\n";
    summary << "  \"corrected_preview_pgm\": \"" << corrected_pgm.string() << "\",\n";
    summary << "  \"final_mask_pgm\": \"" << mask_pgm.string() << "\",\n";
    summary << "  \"overlay_pgm\": \"" << overlay_pgm.string() << "\"\n";
    summary << "}\n";
    if (options.verbose) {
      log_stage_done("artifact write");
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Coherent power offline validation\n";
    std::cout << "  pipeline mode effective: operator_live\n";
    std::cout << "  fast_performance from config: "
          << extract_string(read_text_file(options.metadata_path), "fast_performance").value_or("false")
          << "\n";
    std::cout << "  metadata: " << options.metadata_path << "\n";
    std::cout << "  tensor snapshot: " << metadata.tensor_snapshot_path << "\n";
    std::cout << "  output dir: " << options.output_dir << "\n";
    std::cout << "  rows x cols: " << result.src_rows << " x " << result.src_cols << "\n";
    std::cout << "  tensor axis order: "
          << (metadata.tensor_axis_order.empty() ? "unspecified" : metadata.tensor_axis_order)
          << "\n";
    std::cout << "  resolution hz: " << metadata.resolution_hz << "\n";
    std::cout << "  sample rate hz: " << result.sample_rate_hz << "\n";
    std::cout << "  span hz: " << result.span_hz << "\n";
    std::cout << "  ignore bins per side: " << result.ignore_bins_per_side << "\n";
    std::cout << "  grouped box count: " << result.grouped_box_count << "\n";
    std::cout << "  final mask nonzero pixels: " << final_mask_nonzero_pixels << "\n";
    std::cout << "  final mask fill ratio: " << final_mask_fill_ratio << "\n";
    std::cout << "  merged threshold: " << result.merged_threshold << "\n";
    std::cout << "  seed threshold: " << result.seed_threshold << "\n";
    for (size_t index = 0; index < grouped_box_debug_count; ++index) {
      const auto& box = result.grouped_boxes[grouped_box_order[index]];
      std::cout << "  grouped box[" << index << "]: freq_span=" << box.freq_span
                << " time_span=" << box.time_span
                << " filled_area=" << box.filled_area
                << " density=" << box.density
                << " score_peak=" << box.score_peak
                << " split_role=" << box.split_role << "\n";
    }
    if (options.verbose) {
      std::cout << "  wrote: " << summary_json << "\n";
      std::cout << "  wrote: " << corrected_pgm << "\n";
      if (!raw_projected_mask_preview.empty()) {
        std::cout << "  wrote: " << raw_projected_mask_pgm << "\n";
      }
      std::cout << "  wrote: " << mask_pgm << "\n";
      if (!overlay_preview.empty() && result.final_mask.size() == result.corrected_sxx_db.size()) {
        std::cout << "  wrote: " << overlay_pgm << "\n";
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "offline_coherent_power_validator failed: " << error.what() << "\n";
    return 1;
  }
}
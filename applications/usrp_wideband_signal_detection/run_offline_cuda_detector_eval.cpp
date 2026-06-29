// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <cuda/std/__algorithm/max.h>

#include "fft_runtime_config.hpp"
#include "spectrogram_visualization.hpp"

#include <cuda_dino_detector.hpp>
#include <fft.hpp>
#include <holoscan/holoscan.hpp>
#include <spectrogram.hpp>

#include <cuda_runtime.h>

#include <getopt.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using Complex = cuda::std::complex<float>;
using FftInputMessage = std::tuple<matx::tensor_t<Complex, 2>, cudaStream_t>;

enum class InputScalarKind {
  kSignedInteger,
  kUnsignedInteger,
  kFloat,
};

struct OfflineInputFormat {
  std::string datatype = "ci16_le";
  double sample_rate_hz = 0.0;
  std::filesystem::path sigmf_meta_path;
  uint64_t capture_sample_start = 0;
  double center_frequency_hz = 0.0;
  bool has_center_frequency_hz = false;
  int num_channels = 1;
  bool little_endian = true;
  InputScalarKind scalar_kind = InputScalarKind::kSignedInteger;
  int scalar_bits = 16;
  size_t bytes_per_scalar = sizeof(int16_t);
  size_t bytes_per_complex = sizeof(int16_t) * 2U;
  double integer_midpoint = 0.0;
  double integer_scale = 1.0 / 32767.0;
};

struct SigmfAnnotation {
  uint64_t sample_start = 0;
  uint64_t sample_count = 0;
  double freq_lower_hz = 0.0;
  double freq_upper_hz = 0.0;
  std::string label = "UNLABELED";
  std::string kind = "annotation";
};

struct GroundTruthItem {
  std::string label = "UNLABELED";
  std::string kind = "annotation";
  double x_ms = 0.0;
  double width_ms = 0.0;
  double y_mhz = 0.0;
  double height_mhz = 0.0;
  uint64_t sample_start = 0;
  uint64_t sample_count = 0;
  uint64_t overlap_sample_start = 0;
  uint64_t overlap_sample_stop = 0;
  double freq_lower_hz = 0.0;
  double freq_upper_hz = 0.0;
  int row_start = 0;
  int row_stop = 0;
  int col_start = 0;
  int col_stop = 0;
};

bool string_ends_with(const std::string& value, const std::string& suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::filesystem::path infer_sigmf_meta_path(const std::filesystem::path& input_file_path) {
  auto meta_path = input_file_path;
  meta_path.replace_extension(".sigmf-meta");
  return meta_path;
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("failed to open file: " + path.string());
  }
  std::stringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::optional<std::string> extract_json_string_field(const std::string& text, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  const auto key_pos = text.find(token);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon_pos = text.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto quote_pos = text.find('"', colon_pos + 1);
  if (quote_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto end_quote_pos = text.find('"', quote_pos + 1);
  if (end_quote_pos == std::string::npos) {
    return std::nullopt;
  }
  return text.substr(quote_pos + 1, end_quote_pos - quote_pos - 1);
}

std::optional<double> extract_json_number_field(const std::string& text, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  const auto key_pos = text.find(token);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon_pos = text.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto number_start = text.find_first_of("-+0123456789.", colon_pos + 1);
  if (number_start == std::string::npos) {
    return std::nullopt;
  }
  const auto number_end = text.find_first_not_of("0123456789eE+.-", number_start);
  try {
    return std::stod(text.substr(number_start, number_end - number_start));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::optional<uint64_t> extract_json_uint64_field(const std::string& text, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  const auto key_pos = text.find(token);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon_pos = text.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto number_start = text.find_first_of("0123456789", colon_pos + 1);
  if (number_start == std::string::npos) {
    return std::nullopt;
  }
  const auto number_end = text.find_first_not_of("0123456789", number_start);
  try {
    return std::stoull(text.substr(number_start, number_end - number_start));
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

size_t skip_json_whitespace(const std::string& text, size_t position) {
  while (position < text.size() && std::isspace(static_cast<unsigned char>(text[position])) != 0) {
    ++position;
  }
  return position;
}

size_t skip_json_string(const std::string& text, size_t quote_pos) {
  if (quote_pos >= text.size() || text[quote_pos] != '"') {
    return std::string::npos;
  }

  size_t position = quote_pos + 1;
  while (position < text.size()) {
    if (text[position] == '\\') {
      position += 2;
      continue;
    }
    if (text[position] == '"') {
      return position + 1;
    }
    ++position;
  }
  return std::string::npos;
}

std::optional<size_t> find_json_value_start(const std::string& text, const std::string& key) {
  const std::string token = "\"" + key + "\"";
  const auto key_pos = text.find(token);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon_pos = text.find(':', key_pos + token.size());
  if (colon_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto value_pos = skip_json_whitespace(text, colon_pos + 1);
  if (value_pos >= text.size()) {
    return std::nullopt;
  }
  return value_pos;
}

std::optional<size_t> find_matching_json_delimiter(const std::string& text,
                                                   size_t open_pos,
                                                   char open_char,
                                                   char close_char) {
  if (open_pos >= text.size() || text[open_pos] != open_char) {
    return std::nullopt;
  }

  int depth = 0;
  for (size_t position = open_pos; position < text.size(); ++position) {
    if (text[position] == '"') {
      const auto next_position = skip_json_string(text, position);
      if (next_position == std::string::npos) {
        return std::nullopt;
      }
      position = next_position - 1;
      continue;
    }

    if (text[position] == open_char) {
      ++depth;
      continue;
    }
    if (text[position] == close_char) {
      --depth;
      if (depth == 0) {
        return position;
      }
      if (depth < 0) {
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

std::vector<std::string> extract_json_object_array(const std::string& text, const std::string& key) {
  const auto array_start = find_json_value_start(text, key);
  if (!array_start.has_value() || text[*array_start] != '[') {
    return {};
  }
  const auto array_end = find_matching_json_delimiter(text, *array_start, '[', ']');
  if (!array_end.has_value()) {
    return {};
  }

  std::vector<std::string> objects;
  size_t position = *array_start + 1;
  while (position < *array_end) {
    position = skip_json_whitespace(text, position);
    if (position >= *array_end) {
      break;
    }
    if (text[position] != '{') {
      ++position;
      continue;
    }

    const auto object_end = find_matching_json_delimiter(text, position, '{', '}');
    if (!object_end.has_value()) {
      break;
    }
    objects.push_back(text.substr(position, *object_end - position + 1));
    position = *object_end + 1;
  }
  return objects;
}

uint64_t saturated_add_u64(uint64_t lhs, uint64_t rhs) {
  const uint64_t max_value = std::numeric_limits<uint64_t>::max();
  if (max_value - lhs < rhs) {
    return max_value;
  }
  return lhs + rhs;
}

std::string json_escape_string(const std::string& value) {
  static constexpr char hex_digits[] = "0123456789abcdef";
  std::string escaped;
  escaped.reserve(value.size() + 8);

  for (unsigned char ch : value) {
    switch (ch) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\b':
        escaped += "\\b";
        break;
      case '\f':
        escaped += "\\f";
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
        if (ch < 0x20U) {
          escaped += "\\u00";
          escaped.push_back(hex_digits[(ch >> 4) & 0x0F]);
          escaped.push_back(hex_digits[ch & 0x0F]);
        } else {
          escaped.push_back(static_cast<char>(ch));
        }
        break;
    }
  }

  return escaped;
}

void write_text_file(const std::filesystem::path& path, const std::string& text) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream out(path);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open file for writing: " + path.string());
  }
  out << text;
  if (!out.good()) {
    throw std::runtime_error("failed to write file: " + path.string());
  }
}

std::vector<SigmfAnnotation> load_sigmf_annotations(const std::filesystem::path& path) {
  if (path.empty() || !std::filesystem::exists(path)) {
    return {};
  }

  const std::string meta_text = read_text_file(path);
  std::vector<SigmfAnnotation> annotations;
  for (const auto& object_text : extract_json_object_array(meta_text, "annotations")) {
    SigmfAnnotation annotation;
    annotation.sample_start = extract_json_uint64_field(object_text, "core:sample_start").value_or(0);
    annotation.sample_count = extract_json_uint64_field(object_text, "core:sample_count").value_or(0);
    annotation.freq_lower_hz = extract_json_number_field(object_text, "core:freq_lower_edge").value_or(0.0);
    annotation.freq_upper_hz = extract_json_number_field(object_text, "core:freq_upper_edge").value_or(0.0);
    annotation.label = extract_json_string_field(object_text, "core:label").value_or(std::string("UNLABELED"));
    annotation.kind = extract_json_string_field(object_text, "wfgt:kind").value_or(std::string("annotation"));
    annotations.push_back(std::move(annotation));
  }

  return annotations;
}

OfflineInputFormat make_input_format_from_datatype(const std::string& datatype) {
  OfflineInputFormat format;
  format.datatype = datatype.empty() ? "ci16_le" : datatype;

  std::string base = format.datatype;
  if (string_ends_with(base, "_le")) {
    format.little_endian = true;
    base.resize(base.size() - 3);
  } else if (string_ends_with(base, "_be")) {
    format.little_endian = false;
    base.resize(base.size() - 3);
  }

  if (base.empty() || base.front() != 'c') {
    throw std::runtime_error("offline replay only supports complex SigMF datatypes, got '" + format.datatype + "'");
  }

  const std::string scalar_spec = base.substr(1);
  if (scalar_spec.size() < 2) {
    throw std::runtime_error("invalid SigMF datatype: '" + format.datatype + "'");
  }

  const char scalar_kind = scalar_spec.front();
  int scalar_bits = 0;
  try {
    scalar_bits = std::stoi(scalar_spec.substr(1));
  } catch (const std::exception&) {
    throw std::runtime_error("invalid SigMF datatype bit-width: '" + format.datatype + "'");
  }
  if (scalar_bits <= 0 || (scalar_bits % 8) != 0) {
    throw std::runtime_error("unsupported SigMF datatype bit-width: '" + format.datatype + "'");
  }

  format.scalar_bits = scalar_bits;
  format.bytes_per_scalar = static_cast<size_t>(scalar_bits / 8);
  format.bytes_per_complex = format.bytes_per_scalar * 2U;

  switch (scalar_kind) {
    case 'i':
      format.scalar_kind = InputScalarKind::kSignedInteger;
      format.integer_midpoint = 0.0;
      format.integer_scale = 1.0 / std::max(1.0, std::ldexp(1.0, scalar_bits - 1) - 1.0);
      break;
    case 'u':
      format.scalar_kind = InputScalarKind::kUnsignedInteger;
      format.integer_midpoint = std::ldexp(1.0, scalar_bits - 1);
      format.integer_scale = 1.0 / std::max(1.0, format.integer_midpoint);
      break;
    case 'f':
      format.scalar_kind = InputScalarKind::kFloat;
      break;
    default:
      throw std::runtime_error("unsupported SigMF datatype: '" + format.datatype + "'");
  }

  return format;
}

std::optional<OfflineInputFormat> try_load_sigmf_input_format(const std::filesystem::path& input_file_path) {
  const auto meta_path = infer_sigmf_meta_path(input_file_path);
  if (!std::filesystem::exists(meta_path)) {
    return std::nullopt;
  }

  const auto meta_text = read_text_file(meta_path);
  const auto datatype = extract_json_string_field(meta_text, "core:datatype");
  const auto sample_rate_hz = extract_json_number_field(meta_text, "core:sample_rate");
  if (!datatype.has_value()) {
    throw std::runtime_error("SigMF metadata is missing core:datatype: " + meta_path.string());
  }
  if (!sample_rate_hz.has_value() || !std::isfinite(sample_rate_hz.value()) || sample_rate_hz.value() <= 0.0) {
    throw std::runtime_error("SigMF metadata is missing a valid core:sample_rate: " + meta_path.string());
  }

  auto input_format = make_input_format_from_datatype(datatype.value());
  input_format.sample_rate_hz = sample_rate_hz.value();
  input_format.sigmf_meta_path = meta_path;

  const auto num_channels = extract_json_uint64_field(meta_text, "core:num_channels").value_or(1);
  if (num_channels == 0 || num_channels > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("SigMF metadata has an invalid core:num_channels value: " + meta_path.string());
  }
  input_format.num_channels = static_cast<int>(num_channels);
  if (input_format.num_channels != 1) {
    throw std::runtime_error("offline CUDA detector eval currently supports one complex channel per SigMF data file; got core:num_channels=" +
                             std::to_string(input_format.num_channels));
  }

  const auto captures = extract_json_object_array(meta_text, "captures");
  if (!captures.empty()) {
    input_format.capture_sample_start =
        extract_json_uint64_field(captures.front(), "core:sample_start").value_or(0);
    const auto center_frequency_hz = extract_json_number_field(captures.front(), "core:frequency");
    if (center_frequency_hz.has_value() && std::isfinite(center_frequency_hz.value())) {
      input_format.center_frequency_hz = center_frequency_hz.value();
      input_format.has_center_frequency_hz = true;
    }
  }
  return input_format;
}

bool host_is_little_endian() {
  const uint16_t value = 1;
  return *reinterpret_cast<const uint8_t*>(&value) == 1;
}

template <typename T>
T decode_scalar_bytes(const uint8_t* raw_bytes, bool little_endian) {
  uint8_t storage[sizeof(T)] = {};
  if (little_endian == host_is_little_endian()) {
    std::memcpy(storage, raw_bytes, sizeof(T));
  } else {
    for (size_t index = 0; index < sizeof(T); ++index) {
      storage[index] = raw_bytes[sizeof(T) - 1U - index];
    }
  }

  T value {};
  std::memcpy(&value, storage, sizeof(T));
  return value;
}

float decode_scalar_to_float(const uint8_t* raw_bytes, const OfflineInputFormat& input_format) {
  switch (input_format.scalar_kind) {
    case InputScalarKind::kSignedInteger:
      switch (input_format.scalar_bits) {
        case 8:
          return static_cast<float>(static_cast<double>(decode_scalar_bytes<int8_t>(raw_bytes, input_format.little_endian)) *
                                    input_format.integer_scale);
        case 16:
          return static_cast<float>(static_cast<double>(decode_scalar_bytes<int16_t>(raw_bytes, input_format.little_endian)) *
                                    input_format.integer_scale);
        case 32:
          return static_cast<float>(static_cast<double>(decode_scalar_bytes<int32_t>(raw_bytes, input_format.little_endian)) *
                                    input_format.integer_scale);
        default:
          break;
      }
      break;
    case InputScalarKind::kUnsignedInteger:
      switch (input_format.scalar_bits) {
        case 8:
          return static_cast<float>((static_cast<double>(decode_scalar_bytes<uint8_t>(raw_bytes, input_format.little_endian)) -
                                     input_format.integer_midpoint) *
                                    input_format.integer_scale);
        case 16:
          return static_cast<float>((static_cast<double>(decode_scalar_bytes<uint16_t>(raw_bytes, input_format.little_endian)) -
                                     input_format.integer_midpoint) *
                                    input_format.integer_scale);
        case 32:
          return static_cast<float>((static_cast<double>(decode_scalar_bytes<uint32_t>(raw_bytes, input_format.little_endian)) -
                                     input_format.integer_midpoint) *
                                    input_format.integer_scale);
        default:
          break;
      }
      break;
    case InputScalarKind::kFloat:
      switch (input_format.scalar_bits) {
        case 32:
          return decode_scalar_bytes<float>(raw_bytes, input_format.little_endian);
        case 64:
          return static_cast<float>(decode_scalar_bytes<double>(raw_bytes, input_format.little_endian));
        default:
          break;
      }
      break;
  }

  throw std::runtime_error("unsupported SigMF datatype for offline replay decoding: '" + input_format.datatype + "'");
}

Complex decode_complex_sample(const uint8_t* raw_sample, const OfflineInputFormat& input_format) {
  return Complex(decode_scalar_to_float(raw_sample, input_format),
                 decode_scalar_to_float(raw_sample + input_format.bytes_per_scalar, input_format));
}

struct CliOptions {
  std::string config_path =
      "infocom_evals/signal_detection_experiments/config_cuda_dino_performance_single_channel_offline_eval.yaml";
  std::string input_file_path;
  std::string output_root;
  int progress_every_n_frames = -1;
};

struct EvalOverrides {
  std::filesystem::path config_path;
  std::filesystem::path input_file_path;
  std::filesystem::path output_root;
  std::filesystem::path input_sigmf_meta_path;
  bool run_offline_on_file = true;
  bool save_detector_debug_artifacts = false;
  bool save_spectrogram_preview = true;
  bool save_spectrogram_tensor = true;
  bool save_mask_preview = true;
  bool save_mask_npy = true;
  std::string input_datatype = "ci16_le";
  double input_sample_rate_hz = 0.0;
  uint64_t input_capture_sample_start = 0;
  double input_center_frequency_hz = 0.0;
  bool has_input_center_frequency_hz = false;
  int input_num_channels = 1;
  int progress_every_n_frames = 1;
  int channel_number = 0;
  int fft_num_bursts = 0;
  int fft_burst_size = 0;
  int spectrogram_output_height = 0;
  int spectrogram_output_width = 0;
  uint64_t resolution_hz = 0;
  double span_hz = 0.0;
  uint64_t samples_per_frame = 0;
  uint64_t input_total_complex_samples = 0;
  uint64_t total_complex_samples = 0;
  uint64_t dropped_tail_complex_samples = 0;
  uint64_t total_frames = 0;
  uint64_t drain_frame_count = 32;
};

struct FrameKey {
  int channel = 0;
  uint64_t frame_number = 0;

  bool operator<(const FrameKey& other) const {
    if (channel != other.channel) {
      return channel < other.channel;
    }
    return frame_number < other.frame_number;
  }
};

struct FrameArtifactRecord {
  int channel = 0;
  uint64_t frame_number = 0;
  uint64_t file_offset_complex = 0;
  uint64_t data_end_complex = 0;
  uint64_t frame_end_complex = 0;
  uint64_t complex_samples_read = 0;
  uint64_t complex_samples_padded = 0;
  bool partial_frame = false;
  int fft_rows = 0;
  int fft_cols = 0;
  int preview_rows = 0;
  int preview_cols = 0;
  std::string spectrogram_preview_path;
  std::string spectrogram_tensor_path;
  std::string mask_preview_path;
  std::string mask_npy_path;
  std::string gt_annotations_path;
  std::string gt_mask_npy_path;
  uint64_t local_file_offset_complex = 0;
  uint64_t local_data_end_complex = 0;
  uint64_t local_frame_end_complex = 0;
  uint64_t capture_sample_start = 0;
  uint64_t global_sample_start = 0;
  uint64_t global_data_end_sample = 0;
  uint64_t global_frame_end_sample = 0;
  uint64_t samples_per_row = 0;
};

std::string serialize_ground_truth_payload(const FrameArtifactRecord& record,
                                           double sample_rate_hz,
                                           double span_hz,
                                           const EvalOverrides& overrides,
                                           const std::vector<GroundTruthItem>& items) {
  std::ostringstream out;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"channel\": " << record.channel << ",\n";
  out << "  \"frame_number\": " << record.frame_number << ",\n";
  out << "  \"file_offset_complex\": " << record.file_offset_complex << ",\n";
  out << "  \"frame_start_sample\": " << record.file_offset_complex << ",\n";
  out << "  \"data_end_sample\": " << record.data_end_complex << ",\n";
  out << "  \"frame_end_sample\": " << record.frame_end_complex << ",\n";
  out << "  \"capture_sample_start\": " << record.capture_sample_start << ",\n";
  out << "  \"global_sample_start\": " << record.global_sample_start << ",\n";
  out << "  \"global_data_end_sample\": " << record.global_data_end_sample << ",\n";
  out << "  \"global_frame_end_sample\": " << record.global_frame_end_sample << ",\n";
  out << "  \"local_file_offset_complex\": " << record.local_file_offset_complex << ",\n";
  out << "  \"local_data_end_complex\": " << record.local_data_end_complex << ",\n";
  out << "  \"local_frame_end_complex\": " << record.local_frame_end_complex << ",\n";
  out << "  \"complex_samples_read\": " << record.complex_samples_read << ",\n";
  out << "  \"complex_samples_padded\": " << record.complex_samples_padded << ",\n";
  out << "  \"frame_end_complex\": " << record.frame_end_complex << ",\n";
  out << "  \"sample_rate_hz\": " << sample_rate_hz << ",\n";
  out << "  \"span_hz\": " << span_hz << ",\n";
  if (overrides.has_input_center_frequency_hz) {
    out << "  \"center_frequency_hz\": " << overrides.input_center_frequency_hz << ",\n";
  } else {
    out << "  \"center_frequency_hz\": null,\n";
  }
  out << "  \"fft_rows\": " << record.fft_rows << ",\n";
  out << "  \"fft_cols\": " << record.fft_cols << ",\n";
  out << "  \"samples_per_row\": " << record.samples_per_row << ",\n";
  out << "  \"items\": [\n";
  for (size_t index = 0; index < items.size(); ++index) {
    const auto& item = items[index];
    out << "    {\n";
    out << "      \"label\": \"" << json_escape_string(item.label) << "\",\n";
    out << "      \"kind\": \"" << json_escape_string(item.kind) << "\",\n";
    out << "      \"x_ms\": " << item.x_ms << ",\n";
    out << "      \"width_ms\": " << item.width_ms << ",\n";
    out << "      \"y_mhz\": " << item.y_mhz << ",\n";
    out << "      \"height_mhz\": " << item.height_mhz << ",\n";
    out << "      \"sample_start\": " << item.sample_start << ",\n";
    out << "      \"sample_count\": " << item.sample_count << ",\n";
    out << "      \"overlap_sample_start\": " << item.overlap_sample_start << ",\n";
    out << "      \"overlap_sample_stop\": " << item.overlap_sample_stop << ",\n";
    out << "      \"freq_lower_hz\": " << item.freq_lower_hz << ",\n";
    out << "      \"freq_upper_hz\": " << item.freq_upper_hz << ",\n";
    out << "      \"row_start\": " << item.row_start << ",\n";
    out << "      \"row_stop\": " << item.row_stop << ",\n";
    out << "      \"col_start\": " << item.col_start << ",\n";
    out << "      \"col_stop\": " << item.col_stop << "\n";
    out << "    }";
    if (index + 1 < items.size()) {
      out << ',';
    }
    out << "\n";
  }
  out << "  ]\n";
  out << "}\n";
  return out.str();
}

std::mutex& artifact_registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<FrameKey, FrameArtifactRecord>& artifact_registry() {
  static std::map<FrameKey, FrameArtifactRecord> registry;
  return registry;
}

std::filesystem::path& artifact_registry_root() {
  static std::filesystem::path root;
  return root;
}

std::filesystem::path resolve_config_path(const char* argv0, const std::string& config_arg) {
  const std::filesystem::path requested(config_arg);
  const auto binary_dir = std::filesystem::canonical(argv0).parent_path();

  if (requested.is_absolute() && std::filesystem::exists(requested)) {
    return requested;
  }
  if (std::filesystem::exists(requested)) {
    return std::filesystem::absolute(requested);
  }
  const auto from_binary_dir = binary_dir / requested;
  if (std::filesystem::exists(from_binary_dir)) {
    return from_binary_dir;
  }
  const auto from_source_dir = std::filesystem::path(USRP_WIDEBAND_APP_SOURCE_DIR) / requested;
  if (std::filesystem::exists(from_source_dir)) {
    return from_source_dir;
  }
  return from_binary_dir;
}

std::filesystem::path resolve_runtime_path(const std::filesystem::path& config_path,
                                          const std::string& raw_path) {
  if (raw_path.empty()) {
    return {};
  }
  const std::filesystem::path requested(raw_path);
  if (requested.is_absolute()) {
    return requested;
  }
  return std::filesystem::absolute(config_path.parent_path() / requested);
}

uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
  return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}

void usage(const char* argv0) {
  HOLOSCAN_LOG_INFO(
      "Usage: {} [--config FILE] [--input-file FILE.sc16] [--output-root DIR] [--progress-every N]",
      argv0);
}

CliOptions parse_arguments(int argc, char** argv) {
  CliOptions options;
  static option long_options[] = {{"config", required_argument, nullptr, 'c'},
                                  {"input-file", required_argument, nullptr, 'i'},
                                  {"output-root", required_argument, nullptr, 'o'},
                                  {"progress-every", required_argument, nullptr, 'p'},
                                  {"help", no_argument, nullptr, 'h'},
                                  {0, 0, 0, 0}};

  while (true) {
    const int opt = getopt_long(argc, argv, "c:i:o:p:h", long_options, nullptr);
    if (opt == -1) {
      break;
    }
    switch (opt) {
      case 'c':
        options.config_path = optarg;
        break;
      case 'i':
        options.input_file_path = optarg;
        break;
      case 'o':
        options.output_root = optarg;
        break;
      case 'p':
        options.progress_every_n_frames = std::stoi(optarg);
        break;
      case 'h':
        usage(argv[0]);
        std::exit(0);
      default:
        usage(argv[0]);
        std::exit(1);
    }
  }

  return options;
}

template <typename T>
void set_if_present(FrameArtifactRecord& record, const T& value, T FrameArtifactRecord::* field) {
  record.*field = value;
}

void update_record_from_metadata(FrameArtifactRecord& record,
                                 const std::shared_ptr<holoscan::MetadataDictionary>& meta) {
  if (!meta) {
    return;
  }
  record.file_offset_complex = meta->get<uint64_t>("offline_source_file_offset_complex", record.file_offset_complex);
  record.data_end_complex = meta->get<uint64_t>("offline_source_data_end_complex", record.data_end_complex);
  record.frame_end_complex = meta->get<uint64_t>("offline_source_frame_end_complex", record.frame_end_complex);
  record.local_file_offset_complex =
      meta->get<uint64_t>("offline_source_local_file_offset_complex", record.local_file_offset_complex);
  record.local_data_end_complex =
      meta->get<uint64_t>("offline_source_local_data_end_complex", record.local_data_end_complex);
  record.local_frame_end_complex =
      meta->get<uint64_t>("offline_source_local_frame_end_complex", record.local_frame_end_complex);
  record.capture_sample_start =
      meta->get<uint64_t>("offline_source_capture_sample_start", record.capture_sample_start);
  record.global_sample_start =
      meta->get<uint64_t>("offline_source_global_sample_start", record.file_offset_complex);
  record.global_data_end_sample =
      meta->get<uint64_t>("offline_source_global_data_end_sample", record.data_end_complex);
  record.global_frame_end_sample =
      meta->get<uint64_t>("offline_source_global_frame_end_sample", record.frame_end_complex);
  record.complex_samples_read =
      meta->get<uint64_t>("offline_source_complex_samples_read", record.complex_samples_read);
  record.complex_samples_padded =
      meta->get<uint64_t>("offline_source_complex_samples_padded", record.complex_samples_padded);
  record.partial_frame = meta->get<bool>("offline_source_partial_frame", record.partial_frame);
}

void update_record_from_mask_message(FrameArtifactRecord& record,
                                     const holoscan::ops::DetectorMaskMessage& mask) {
  if (mask.file_offset_complex != 0 || mask.data_end_complex != 0 || mask.frame_end_complex != 0 ||
      mask.complex_samples_read != 0 || mask.complex_samples_padded != 0) {
    record.file_offset_complex = mask.file_offset_complex;
    record.data_end_complex = mask.data_end_complex;
    record.frame_end_complex = mask.frame_end_complex;
    record.global_sample_start = mask.file_offset_complex;
    record.global_data_end_sample = mask.data_end_complex;
    record.global_frame_end_sample = mask.frame_end_complex;
    record.complex_samples_read = mask.complex_samples_read;
    record.complex_samples_padded = mask.complex_samples_padded;
    record.partial_frame = mask.complex_samples_padded > 0;
  }
}

uint64_t samples_per_row_for_record(const FrameArtifactRecord& record, const EvalOverrides& overrides) {
  if (record.samples_per_row > 0) {
    return record.samples_per_row;
  }

  const uint64_t rows = static_cast<uint64_t>(std::max(1, record.fft_rows));
  if (record.frame_end_complex > record.file_offset_complex) {
    return std::max<uint64_t>(1, ceil_div(record.frame_end_complex - record.file_offset_complex, rows));
  }
  if (overrides.samples_per_frame > 0) {
    return std::max<uint64_t>(1, ceil_div(overrides.samples_per_frame, rows));
  }
  return static_cast<uint64_t>(std::max(1, record.fft_cols));
}

std::pair<double, double> annotation_frequency_for_baseband(const SigmfAnnotation& annotation,
                                                            const EvalOverrides& overrides,
                                                            double span_hz) {
  const double freq_min_hz = -0.5 * span_hz;
  const double freq_max_hz = 0.5 * span_hz;
  auto overlaps_baseband = [&](double lower_hz, double upper_hz) {
    return std::min(freq_max_hz, upper_hz) > std::max(freq_min_hz, lower_hz);
  };

  if (overlaps_baseband(annotation.freq_lower_hz, annotation.freq_upper_hz)) {
    return {annotation.freq_lower_hz, annotation.freq_upper_hz};
  }

  if (overrides.has_input_center_frequency_hz) {
    const double shifted_lower_hz = annotation.freq_lower_hz - overrides.input_center_frequency_hz;
    const double shifted_upper_hz = annotation.freq_upper_hz - overrides.input_center_frequency_hz;
    if (overlaps_baseband(shifted_lower_hz, shifted_upper_hz)) {
      return {shifted_lower_hz, shifted_upper_hz};
    }
  }

  return {annotation.freq_lower_hz, annotation.freq_upper_hz};
}

void reset_artifact_registry(const std::filesystem::path& output_root) {
  std::fprintf(stderr,
               "[offline_cuda_detector_eval] reset_artifact_registry: begin root='%s'\n",
               output_root.string().c_str());
  std::fflush(stderr);
  std::lock_guard<std::mutex> lock(artifact_registry_mutex());
  artifact_registry().clear();
  artifact_registry_root() = output_root;
  std::fprintf(stderr, "[offline_cuda_detector_eval] reset_artifact_registry: end\n");
  std::fflush(stderr);
}

void clear_output_root(const std::filesystem::path& output_root) {
  if (output_root.empty()) {
    throw std::runtime_error("offline output root must not be empty");
  }

  const auto normalized = output_root.lexically_normal();
  if (normalized == normalized.root_path() || normalized.parent_path() == normalized.root_path()) {
    throw std::runtime_error("refusing to clear an unsafe offline output root: " + normalized.string());
  }

  if (std::filesystem::exists(normalized)) {
    const auto parent = normalized.parent_path();
    const auto stem = normalized.filename().string();
    std::filesystem::path archived_path;
    std::error_code rename_error;
    for (int suffix = 1; suffix <= 1024; ++suffix) {
      archived_path = parent / (stem + ".stale_" + std::to_string(suffix));
      if (std::filesystem::exists(archived_path)) {
        continue;
      }
      std::filesystem::rename(normalized, archived_path, rename_error);
      if (!rename_error) {
        HOLOSCAN_LOG_INFO("Archived previous offline eval output root '{}' -> '{}'",
                          normalized.string(),
                          archived_path.string());
        return;
      }
      break;
    }

    throw std::runtime_error("failed to archive offline output root '" + normalized.string() +
                             "': " + rename_error.message());
  }
}

std::string relative_to_output_root(const std::filesystem::path& absolute_path) {
  return std::filesystem::relative(absolute_path, artifact_registry_root()).generic_string();
}

FrameArtifactRecord& ensure_record(int channel, uint64_t frame_number) {
  auto& registry = artifact_registry();
  const FrameKey key {channel, frame_number};
  auto [it, inserted] = registry.try_emplace(key);
  if (inserted) {
    it->second.channel = channel;
    it->second.frame_number = frame_number;
  }
  return it->second;
}

void register_spectrogram_artifacts(int channel,
                                    uint64_t frame_number,
                                    int fft_rows,
                                    int fft_cols,
                                    int preview_rows,
                                    int preview_cols,
                                    const std::string& preview_path,
                                    const std::string& tensor_path,
                                    const std::shared_ptr<holoscan::MetadataDictionary>& meta) {
  std::lock_guard<std::mutex> lock(artifact_registry_mutex());
  auto& record = ensure_record(channel, frame_number);
  record.fft_rows = fft_rows;
  record.fft_cols = fft_cols;
  record.preview_rows = preview_rows;
  record.preview_cols = preview_cols;
  record.spectrogram_preview_path = preview_path;
  record.spectrogram_tensor_path = tensor_path;
  update_record_from_metadata(record, meta);
}

void register_mask_artifacts(int channel,
                             uint64_t frame_number,
                             int preview_rows,
                             int preview_cols,
                             const std::string& preview_path,
                             const std::string& mask_npy_path,
                             const holoscan::ops::DetectorMaskMessage& mask,
                             const std::shared_ptr<holoscan::MetadataDictionary>& meta) {
  std::lock_guard<std::mutex> lock(artifact_registry_mutex());
  auto& record = ensure_record(channel, frame_number);
  if (record.preview_rows == 0) {
    record.preview_rows = preview_rows;
  }
  if (record.preview_cols == 0) {
    record.preview_cols = preview_cols;
  }
  record.mask_preview_path = preview_path;
  record.mask_npy_path = mask_npy_path;
  update_record_from_mask_message(record, mask);
  update_record_from_metadata(record, meta);
  if (meta) {
    const auto aligned_preview_path =
        meta->get<std::string>("cuda_dino_aligned_spectrogram_preview_path", std::string {});
    if (!aligned_preview_path.empty()) {
      record.spectrogram_preview_path = relative_to_output_root(std::filesystem::path(aligned_preview_path));
      record.preview_rows =
          meta->get<int>("cuda_dino_aligned_spectrogram_preview_rows", std::max(1, record.preview_rows));
      record.preview_cols =
          meta->get<int>("cuda_dino_aligned_spectrogram_preview_cols", std::max(1, record.preview_cols));
    }

    const auto aligned_tensor_path =
        meta->get<std::string>("cuda_dino_aligned_spectrogram_tensor_path", std::string {});
    if (!aligned_tensor_path.empty()) {
      record.spectrogram_tensor_path = relative_to_output_root(std::filesystem::path(aligned_tensor_path));
      record.fft_rows = meta->get<int>("cuda_dino_aligned_spectrogram_rows", std::max(1, record.fft_rows));
      record.fft_cols = meta->get<int>("cuda_dino_aligned_spectrogram_cols", std::max(1, record.fft_cols));
    }
  }
}

bool write_pgm(const std::filesystem::path& path,
               const std::vector<uint8_t>& image,
               int width,
               int height) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(image.data()), static_cast<std::streamsize>(image.size()));
  return out.good();
}

bool write_npy_2d(const std::filesystem::path& path,
                  const void* data,
                  size_t bytes,
                  int rows,
                  int cols,
                  const std::string& descr) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    return false;
  }

  const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
  out.write(magic, sizeof(magic));
  out.put(static_cast<char>(1));
  out.put(static_cast<char>(0));

  std::ostringstream header_stream;
  header_stream << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': ("
                << rows << ", " << cols << "), }";
  std::string header = header_stream.str();
  const size_t preamble_size = 6 + 2 + 2;
  const size_t padding = (16 - ((preamble_size + header.size() + 1) % 16)) % 16;
  header.append(padding, ' ');
  header.push_back('\n');

  const uint16_t header_len = static_cast<uint16_t>(header.size());
  const char header_len_bytes[] = {static_cast<char>(header_len & 0xFF),
                                   static_cast<char>((header_len >> 8) & 0xFF)};
  out.write(header_len_bytes, sizeof(header_len_bytes));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
  return out.good();
}

std::vector<uint8_t> build_spectrogram_preview(const std::vector<Complex>& host_fft,
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

std::vector<uint8_t> build_mask_preview(const std::vector<uint8_t>& input_mask,
                                        int src_rows,
                                        int src_cols,
                                        int dst_rows,
                                        int dst_cols) {
  std::vector<uint8_t> preview(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), 0);
  for (int r = 0; r < dst_rows; ++r) {
    const int r0 = (r * src_rows) / dst_rows;
    const int r1 = ((r + 1) * src_rows) / dst_rows;
    for (int c = 0; c < dst_cols; ++c) {
      const int c0 = (c * src_cols) / dst_cols;
      const int c1 = ((c + 1) * src_cols) / dst_cols;
      bool hit = false;
      for (int rr = r0; rr < std::max(r0 + 1, r1) && !hit; ++rr) {
        for (int cc = c0; cc < std::max(c0 + 1, c1); ++cc) {
          const uint8_t value = input_mask[static_cast<size_t>(rr) * static_cast<size_t>(src_cols) +
                                           static_cast<size_t>(cc)];
          if (value != 0) {
            hit = true;
            break;
          }
        }
      }
      preview[static_cast<size_t>(r) * static_cast<size_t>(dst_cols) + static_cast<size_t>(c)] =
          hit ? static_cast<uint8_t>(255) : static_cast<uint8_t>(0);
    }
  }
  return preview;
}

std::filesystem::path make_artifact_path(const std::filesystem::path& root,
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

void write_ground_truth_artifacts(FrameArtifactRecord& record,
                                  const std::vector<SigmfAnnotation>& annotations,
                                  const EvalOverrides& overrides) {
  if (record.fft_rows <= 0 || record.fft_cols <= 0) {
    return;
  }

  const double sample_rate_hz = overrides.input_sample_rate_hz > 0.0 ? overrides.input_sample_rate_hz : overrides.span_hz;
  const double span_hz = overrides.span_hz > 0.0 ? overrides.span_hz : sample_rate_hz;
  if (!(sample_rate_hz > 0.0) || !(span_hz > 0.0)) {
    return;
  }

  const uint64_t frame_start = record.file_offset_complex;
  if (record.data_end_complex == 0 && (record.complex_samples_read > 0 || record.complex_samples_padded > 0)) {
    record.data_end_complex = frame_start + record.complex_samples_read;
  }
  if (record.frame_end_complex == 0 && (record.complex_samples_read > 0 || record.complex_samples_padded > 0)) {
    record.frame_end_complex = frame_start + record.complex_samples_read + record.complex_samples_padded;
  }
  if (record.global_sample_start == 0) {
    record.global_sample_start = record.file_offset_complex;
  }
  if (record.global_data_end_sample == 0) {
    record.global_data_end_sample = record.data_end_complex;
  }
  if (record.global_frame_end_sample == 0) {
    record.global_frame_end_sample = record.frame_end_complex;
  }
  const uint64_t data_end = record.data_end_complex;
  const uint64_t samples_per_row = samples_per_row_for_record(record, overrides);
  record.samples_per_row = samples_per_row;
  const double freq_min_hz = -0.5 * span_hz;
  const double freq_max_hz = 0.5 * span_hz;

  std::vector<GroundTruthItem> gt_items;
  gt_items.reserve(annotations.size());
  std::vector<uint8_t> gt_mask(static_cast<size_t>(record.fft_rows) * static_cast<size_t>(record.fft_cols), 0U);

  for (const auto& annotation : annotations) {
      const uint64_t annotation_start = annotation.sample_start;
      const uint64_t annotation_count = annotation.sample_count;
      const uint64_t annotation_end = saturated_add_u64(annotation_start, annotation_count);
      const uint64_t overlap_start = std::max(frame_start, annotation_start);
      const uint64_t overlap_end = std::min(data_end, annotation_end);
      if (overlap_end <= overlap_start) {
        continue;
      }

      const auto [freq_lower, freq_upper] = annotation_frequency_for_baseband(annotation, overrides, span_hz);
      const double clipped_lower = std::max(freq_min_hz, freq_lower);
      const double clipped_upper = std::min(freq_max_hz, freq_upper);
      if (!(clipped_upper > clipped_lower)) {
        continue;
      }

      const int row_start = std::clamp<int>(
          static_cast<int>((overlap_start - frame_start) / samples_per_row), 0, record.fft_rows);
      const int row_stop = std::clamp<int>(
          static_cast<int>(ceil_div(overlap_end - frame_start, samples_per_row)), 0, record.fft_rows);
      const int col_start = std::clamp<int>(
          static_cast<int>(std::floor(((clipped_lower - freq_min_hz) / span_hz) * static_cast<double>(record.fft_cols))),
          0,
          record.fft_cols);
      const int col_stop = std::clamp<int>(
          static_cast<int>(std::ceil(((clipped_upper - freq_min_hz) / span_hz) * static_cast<double>(record.fft_cols))),
          0,
          record.fft_cols);

      if (row_stop <= row_start || col_stop <= col_start) {
        continue;
      }

      for (int row = row_start; row < row_stop; ++row) {
        for (int col = col_start; col < col_stop; ++col) {
          gt_mask[static_cast<size_t>(row) * static_cast<size_t>(record.fft_cols) + static_cast<size_t>(col)] = 1U;
        }
      }

      GroundTruthItem gt_item;
      gt_item.label = annotation.label;
      gt_item.kind = annotation.kind;
      gt_item.x_ms = (static_cast<double>(overlap_start) / sample_rate_hz) * 1.0e3;
      gt_item.width_ms = (static_cast<double>(overlap_end - overlap_start) / sample_rate_hz) * 1.0e3;
      gt_item.y_mhz = clipped_lower / 1.0e6;
      gt_item.height_mhz = (clipped_upper - clipped_lower) / 1.0e6;
      gt_item.sample_start = annotation_start;
      gt_item.sample_count = annotation_count;
      gt_item.overlap_sample_start = overlap_start;
      gt_item.overlap_sample_stop = overlap_end;
      gt_item.freq_lower_hz = clipped_lower;
      gt_item.freq_upper_hz = clipped_upper;
      gt_item.row_start = row_start;
      gt_item.row_stop = row_stop;
      gt_item.col_start = col_start;
      gt_item.col_stop = col_stop;
      gt_items.push_back(std::move(gt_item));
  }

  const auto gt_annotations_path = make_artifact_path(overrides.output_root,
                                                      "gt_annotations",
                                                      "ground_truth",
                                                      record.channel,
                                                      record.frame_number,
                                                      record.fft_rows,
                                                      record.fft_cols,
                                                      ".json");
  write_text_file(gt_annotations_path,
                  serialize_ground_truth_payload(record, sample_rate_hz, span_hz, overrides, gt_items));
  record.gt_annotations_path = relative_to_output_root(gt_annotations_path);

  const auto gt_mask_path = make_artifact_path(overrides.output_root,
                                               "gt_masks",
                                               "ground_truth_mask",
                                               record.channel,
                                               record.frame_number,
                                               record.fft_rows,
                                               record.fft_cols,
                                               ".npy");
  if (!write_npy_2d(gt_mask_path,
                    gt_mask.data(),
                    gt_mask.size() * sizeof(uint8_t),
                    record.fft_rows,
                    record.fft_cols,
                    "|u1")) {
    throw std::runtime_error("failed to write ground-truth mask artifact: " + gt_mask_path.string());
  }
  record.gt_mask_npy_path = relative_to_output_root(gt_mask_path);
}

void write_manifest(const EvalOverrides& overrides) {
  std::vector<FrameArtifactRecord> records;
  {
    std::lock_guard<std::mutex> lock(artifact_registry_mutex());
    records.reserve(artifact_registry().size());
    for (const auto& [_, record] : artifact_registry()) {
      records.push_back(record);
    }
  }

  const size_t total_registry_records = records.size();
  records.erase(std::remove_if(records.begin(),
                               records.end(),
                               [&overrides](const FrameArtifactRecord& record) {
                                 return record.frame_number == 0 || record.frame_number > overrides.total_frames;
                               }),
                records.end());
  const size_t ignored_non_data_records = total_registry_records - records.size();
  if (ignored_non_data_records > 0) {
    HOLOSCAN_LOG_INFO("Ignoring {} non-data drain records beyond expected {} real frames when writing offline eval artifacts",
                      ignored_non_data_records,
                      overrides.total_frames);
  }

  std::filesystem::create_directories(overrides.output_root);

  const auto annotations = load_sigmf_annotations(overrides.input_sigmf_meta_path);

  for (auto& record : records) {
    write_ground_truth_artifacts(record, annotations, overrides);
  }

  const auto manifest_path = overrides.output_root / "frame_manifest.csv";
  std::ofstream manifest(manifest_path);
  manifest << "channel,frame_number,file_offset_complex,data_end_complex,frame_end_complex,complex_samples_read,complex_samples_padded,partial_frame,fft_rows,fft_cols,preview_rows,preview_cols,spectrogram_preview_pgm,spectrogram_tensor_npy,mask_preview_pgm,mask_npy,gt_annotations_json,gt_mask_npy,global_sample_start,global_data_end_sample,global_frame_end_sample,local_file_offset_complex,local_data_end_complex,local_frame_end_complex,capture_sample_start,samples_per_row\n";
  for (const auto& record : records) {
    manifest << record.channel << ',' << record.frame_number << ',' << record.file_offset_complex << ','
             << record.data_end_complex << ',' << record.frame_end_complex << ','
             << record.complex_samples_read << ',' << record.complex_samples_padded << ','
             << (record.partial_frame ? "true" : "false") << ',' << record.fft_rows << ','
             << record.fft_cols << ',' << record.preview_rows << ',' << record.preview_cols << ','
             << record.spectrogram_preview_path << ',' << record.spectrogram_tensor_path << ','
             << record.mask_preview_path << ',' << record.mask_npy_path << ','
             << record.gt_annotations_path << ',' << record.gt_mask_npy_path << ','
             << record.global_sample_start << ',' << record.global_data_end_sample << ','
             << record.global_frame_end_sample << ',' << record.local_file_offset_complex << ','
             << record.local_data_end_complex << ',' << record.local_frame_end_complex << ','
             << record.capture_sample_start << ',' << record.samples_per_row << '\n';
  }

  size_t spectrogram_count = 0;
  size_t spectrogram_preview_count = 0;
  size_t spectrogram_tensor_count = 0;
  size_t mask_count = 0;
  size_t mask_preview_count = 0;
  size_t mask_npy_count = 0;
  size_t gt_annotation_count = 0;
  size_t gt_mask_count = 0;
  for (const auto& record : records) {
    if (!record.spectrogram_preview_path.empty() || !record.spectrogram_tensor_path.empty()) {
      ++spectrogram_count;
    }
    if (!record.spectrogram_preview_path.empty()) {
      ++spectrogram_preview_count;
    }
    if (!record.spectrogram_tensor_path.empty()) {
      ++spectrogram_tensor_count;
    }
    if (!record.mask_preview_path.empty() || !record.mask_npy_path.empty()) {
      ++mask_count;
    }
    if (!record.mask_preview_path.empty()) {
      ++mask_preview_count;
    }
    if (!record.mask_npy_path.empty()) {
      ++mask_npy_count;
    }
    if (!record.gt_annotations_path.empty()) {
      ++gt_annotation_count;
    }
    if (!record.gt_mask_npy_path.empty()) {
      ++gt_mask_count;
    }
  }

  const auto expected_global_sample_end =
      saturated_add_u64(overrides.input_capture_sample_start, overrides.total_complex_samples);
  bool manifest_complete = records.size() == overrides.total_frames;
  if (manifest_complete && !records.empty()) {
    const auto& first_record = records.front();
    const auto& last_record = records.back();
    manifest_complete = first_record.frame_number == 1 &&
                        first_record.file_offset_complex == overrides.input_capture_sample_start &&
                        last_record.frame_number == overrides.total_frames &&
                        last_record.data_end_complex == expected_global_sample_end &&
                        last_record.frame_end_complex == expected_global_sample_end;
  }
  if (overrides.total_frames > 0 && records.empty()) {
    manifest_complete = false;
  }

  const auto summary_path = overrides.output_root / "offline_eval_summary.json";
  std::ofstream summary(summary_path);
  summary << "{\n";
  summary << "  \"input_file_path\": \"" << overrides.input_file_path.string() << "\",\n";
  summary << "  \"config_path\": \"" << overrides.config_path.string() << "\",\n";
  summary << "  \"input_sigmf_meta_path\": \"" << overrides.input_sigmf_meta_path.string() << "\",\n";
  summary << "  \"input_datatype\": \"" << overrides.input_datatype << "\",\n";
  summary << "  \"input_sample_rate_hz\": " << overrides.input_sample_rate_hz << ",\n";
  summary << "  \"input_num_channels\": " << overrides.input_num_channels << ",\n";
  summary << "  \"input_capture_sample_start\": " << overrides.input_capture_sample_start << ",\n";
  if (overrides.has_input_center_frequency_hz) {
    summary << "  \"input_center_frequency_hz\": " << overrides.input_center_frequency_hz << ",\n";
  } else {
    summary << "  \"input_center_frequency_hz\": null,\n";
  }
  summary << "  \"output_root\": \"" << overrides.output_root.string() << "\",\n";
  summary << "  \"total_frames\": " << overrides.total_frames << ",\n";
  summary << "  \"full_frame_count\": " << overrides.total_frames << ",\n";
  summary << "  \"drain_frame_count\": " << overrides.drain_frame_count << ",\n";
  summary << "  \"scheduled_frame_count\": " << overrides.total_frames + overrides.drain_frame_count << ",\n";
  summary << "  \"input_total_complex_samples\": " << overrides.input_total_complex_samples << ",\n";
  summary << "  \"total_complex_samples\": " << overrides.total_complex_samples << ",\n";
  summary << "  \"processed_complex_samples\": " << overrides.total_complex_samples << ",\n";
  summary << "  \"dropped_tail_complex_samples\": " << overrides.dropped_tail_complex_samples << ",\n";
  summary << "  \"global_sample_start\": " << overrides.input_capture_sample_start << ",\n";
  summary << "  \"global_sample_end\": " << expected_global_sample_end << ",\n";
  summary << "  \"input_global_sample_end\": " << saturated_add_u64(overrides.input_capture_sample_start, overrides.input_total_complex_samples) << ",\n";
  summary << "  \"dropped_tail_start_sample\": " << expected_global_sample_end << ",\n";
  summary << "  \"dropped_tail_end_sample\": " << saturated_add_u64(overrides.input_capture_sample_start, overrides.input_total_complex_samples) << ",\n";
  summary << "  \"samples_per_frame\": " << overrides.samples_per_frame << ",\n";
  summary << "  \"fft_num_bursts\": " << overrides.fft_num_bursts << ",\n";
  summary << "  \"fft_burst_size\": " << overrides.fft_burst_size << ",\n";
  summary << "  \"span_hz\": " << overrides.span_hz << ",\n";
  summary << "  \"resolution_hz\": " << overrides.resolution_hz << ",\n";
  summary << "  \"frames_with_saved_spectrogram\": " << spectrogram_count << ",\n";
  summary << "  \"frames_with_saved_spectrogram_preview\": " << spectrogram_preview_count << ",\n";
  summary << "  \"frames_with_saved_spectrogram_tensor\": " << spectrogram_tensor_count << ",\n";
  summary << "  \"frames_with_saved_mask\": " << mask_count << ",\n";
  summary << "  \"frames_with_saved_mask_preview\": " << mask_preview_count << ",\n";
  summary << "  \"frames_with_saved_mask_npy\": " << mask_npy_count << ",\n";
  summary << "  \"frames_with_saved_gt_annotations\": " << gt_annotation_count << ",\n";
  summary << "  \"frames_with_saved_gt_mask\": " << gt_mask_count << ",\n";
  summary << "  \"manifest_complete\": " << (manifest_complete ? "true" : "false") << ",\n";
  summary << "  \"manifest_csv\": \"frame_manifest.csv\"\n";
  summary << "}\n";

  auto throw_coverage_error = [](const std::string& message) {
    throw std::runtime_error("offline eval artifact coverage validation failed: " + message);
  };

  try {
    if (records.size() != overrides.total_frames) {
      throw_coverage_error("manifest contains " + std::to_string(records.size()) + " frame rows but expected " +
                           std::to_string(overrides.total_frames) + " complete-frame rows");
    }
    if (overrides.save_spectrogram_preview && spectrogram_preview_count != overrides.total_frames) {
      throw_coverage_error("saved spectrogram preview artifacts cover " +
                           std::to_string(spectrogram_preview_count) + " frames but expected " +
                           std::to_string(overrides.total_frames));
    }
    if (overrides.save_spectrogram_tensor && spectrogram_tensor_count != overrides.total_frames) {
      throw_coverage_error("saved spectrogram tensor artifacts cover " +
                           std::to_string(spectrogram_tensor_count) + " frames but expected " +
                           std::to_string(overrides.total_frames));
    }
    if (overrides.save_mask_preview && mask_preview_count != overrides.total_frames) {
      throw_coverage_error("saved detector mask preview artifacts cover " +
                           std::to_string(mask_preview_count) + " frames but expected " +
                           std::to_string(overrides.total_frames));
    }
    if (overrides.save_mask_npy && mask_npy_count != overrides.total_frames) {
      throw_coverage_error("saved detector mask npy artifacts cover " + std::to_string(mask_npy_count) +
                           " frames but expected " + std::to_string(overrides.total_frames));
    }
    if (gt_annotation_count != overrides.total_frames || gt_mask_count != overrides.total_frames) {
      throw_coverage_error("saved GT artifacts cover annotations=" + std::to_string(gt_annotation_count) +
                           " masks=" + std::to_string(gt_mask_count) + " frames but expected " +
                           std::to_string(overrides.total_frames));
    }

    for (size_t index = 0; index < records.size(); ++index) {
      const auto& record = records[index];
      const uint64_t expected_frame_number = static_cast<uint64_t>(index + 1);
      const uint64_t expected_frame_start = saturated_add_u64(
          overrides.input_capture_sample_start,
          static_cast<uint64_t>(index) * overrides.samples_per_frame);
      const uint64_t expected_frame_end = saturated_add_u64(expected_frame_start, overrides.samples_per_frame);

      if (record.frame_number != expected_frame_number) {
        throw_coverage_error("manifest row " + std::to_string(index + 1) + " has frame_number=" +
                             std::to_string(record.frame_number) + " but expected " +
                             std::to_string(expected_frame_number));
      }
      if (record.file_offset_complex != expected_frame_start || record.data_end_complex != expected_frame_end ||
          record.frame_end_complex != expected_frame_end) {
        throw_coverage_error("frame " + std::to_string(record.frame_number) + " covers samples [" +
                             std::to_string(record.file_offset_complex) + ", " +
                             std::to_string(record.data_end_complex) + ") with frame_end=" +
                             std::to_string(record.frame_end_complex) + " but expected [" +
                             std::to_string(expected_frame_start) + ", " +
                             std::to_string(expected_frame_end) + ")");
      }
      if (record.complex_samples_read != overrides.samples_per_frame || record.complex_samples_padded != 0 ||
          record.partial_frame) {
        throw_coverage_error("frame " + std::to_string(record.frame_number) + " read=" +
                             std::to_string(record.complex_samples_read) + " padded=" +
                             std::to_string(record.complex_samples_padded) + " partial=" +
                             (record.partial_frame ? "true" : "false") +
                             "; offline eval now emits complete frames only");
      }
    }
  } catch (...) {
    std::filesystem::remove(summary_path);
    std::filesystem::remove(manifest_path);
    throw;
  }
}

class OfflineSc16FileSourceOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(OfflineSc16FileSourceOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<FftInputMessage>("out");
    spec.param(input_file_path_, "input_file_path", "Input File Path", "Raw complex IQ input file.");
    spec.param(input_datatype_, "input_datatype", "Input Datatype", "SigMF datatype string for the input IQ file.", std::string("ci16_le"));
    spec.param(num_bursts_, "num_bursts", "Number of bursts", "Rows per FFT batch.", 512);
    spec.param(burst_size_, "burst_size", "Burst size", "Complex samples per burst.", 20480);
    spec.param(channel_number_, "channel_number", "Channel Number", "Metadata channel number.", 0);
    spec.param(span_hz_, "span_hz", "Span Hz", "Sample-rate metadata propagated to FFT.", 0.0);
    spec.param(global_sample_start_,
         "global_sample_start",
         "Global Sample Start",
         "SigMF global sample index corresponding to file sample zero.",
         static_cast<int64_t>(0));
    spec.param(total_complex_samples_,
               "total_complex_samples",
               "Total Complex Samples",
               "Complex samples available in the input file.",
               static_cast<int64_t>(0));
    spec.param(real_frame_count_,
               "real_frame_count",
               "Real Frame Count",
               "Number of complete data frames to replay before drain frames.",
               static_cast<int64_t>(0));
    spec.param(drain_frame_count_,
               "drain_frame_count",
               "Drain Frame Count",
               "Number of marked non-data frames emitted to flush downstream operators.",
               static_cast<int64_t>(0));
    spec.param(ring_size_, "ring_size", "Ring Size", "Reusable device-buffer ring size.", 4);
  }

  void initialize() override {
    Operator::initialize();

    input_.open(input_file_path_.get(), std::ios::binary);
    if (!input_.is_open()) {
      throw std::runtime_error("failed to open offline input file: " + input_file_path_.get());
    }

    input_format_ = make_input_format_from_datatype(input_datatype_.get());
    samples_per_frame_ = static_cast<uint64_t>(std::max(1, num_bursts_.get())) *
                         static_cast<uint64_t>(std::max(1, burst_size_.get()));
    real_frame_count_limit_ = static_cast<uint64_t>(std::max<int64_t>(0, real_frame_count_.get()));
    if (real_frame_count_limit_ == 0 && total_complex_samples_.get() > 0) {
      real_frame_count_limit_ = static_cast<uint64_t>(total_complex_samples_.get()) / samples_per_frame_;
    }
    host_input_bytes_.assign(static_cast<size_t>(samples_per_frame_) * input_format_.bytes_per_complex, 0U);
    host_complex_.assign(static_cast<size_t>(samples_per_frame_), Complex {0.0f, 0.0f});

    const int ring_size = std::max(1, ring_size_.get());
    slots_.resize(static_cast<size_t>(ring_size));
    for (auto& slot : slots_) {
      make_tensor(slot.device_tensor,
                  {static_cast<matx::index_t>(num_bursts_.get()),
                   static_cast<matx::index_t>(burst_size_.get())},
                  MATX_DEVICE_MEMORY);
      if (cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking) != cudaSuccess) {
        throw std::runtime_error("failed to create offline source CUDA stream");
      }
    }
  }

  void compute(holoscan::InputContext&, holoscan::OutputContext& op_output, holoscan::ExecutionContext&) override {
    const uint64_t frame_number = emitted_frames_ + 1;
    const bool drain_frame = emitted_frames_ >= real_frame_count_limit_;
    auto& slot = slots_[static_cast<size_t>(emitted_frames_ % slots_.size())];
    const auto sync_result = cudaStreamSynchronize(slot.stream);
    if (sync_result != cudaSuccess) {
      throw std::runtime_error("offline source failed to synchronize reusable CUDA stream");
    }

    size_t complex_samples_read = 0;
    if (drain_frame) {
      std::fill(host_complex_.begin(), host_complex_.end(), Complex {0.0f, 0.0f});
    } else {
      input_.read(reinterpret_cast<char*>(host_input_bytes_.data()),
                  static_cast<std::streamsize>(samples_per_frame_ * input_format_.bytes_per_complex));
      const auto bytes_read = static_cast<size_t>(input_.gcount());
      if (bytes_read == 0) {
        throw std::runtime_error("offline source reached EOF before emitting expected complete frame " +
                                 std::to_string(frame_number));
      }
      if (input_.fail() && !input_.eof()) {
        throw std::runtime_error("offline source read failure before EOF");
      }

      if ((bytes_read % input_format_.bytes_per_complex) != 0) {
        throw std::runtime_error("offline source read byte count is not aligned to the configured SigMF datatype");
      }

      complex_samples_read = bytes_read / input_format_.bytes_per_complex;
      if (complex_samples_read != samples_per_frame_) {
        throw std::runtime_error("offline source read " + std::to_string(complex_samples_read) +
                                 " complex samples for a full-frame replay step; expected " +
                                 std::to_string(samples_per_frame_) +
                                 ". Short tail samples must be dropped before scheduling frames.");
      }
      for (size_t index = 0; index < complex_samples_read; ++index) {
        const auto* raw_sample = host_input_bytes_.data() + (index * input_format_.bytes_per_complex);
        host_complex_[index] = decode_complex_sample(raw_sample, input_format_);
      }
    }

    const auto copy_result = cudaMemcpyAsync(slot.device_tensor.Data(),
                                             host_complex_.data(),
                                             host_complex_.size() * sizeof(Complex),
                                             cudaMemcpyHostToDevice,
                                             slot.stream);
    if (copy_result != cudaSuccess) {
      throw std::runtime_error("offline source failed to upload IQ batch to device");
    }

    const auto upload_sync_result = cudaStreamSynchronize(slot.stream);
    if (upload_sync_result != cudaSuccess) {
      throw std::runtime_error("offline source failed to synchronize uploaded IQ batch");
    }

    auto meta = metadata();
    if (meta) {
      meta->set("channel_number", static_cast<uint16_t>(std::max(0, channel_number_.get())));
      meta->set("sample_rate_hz", span_hz_.get());
      meta->set("offline_source_frame_number", frame_number);
      meta->set("offline_source_total_complex_samples", static_cast<uint64_t>(std::max<int64_t>(0, total_complex_samples_.get())));
      const uint64_t global_sample_start = static_cast<uint64_t>(std::max<int64_t>(0, global_sample_start_.get()));
      const uint64_t real_frame_index = std::min(emitted_frames_, real_frame_count_limit_);
      const uint64_t local_frame_start_sample = real_frame_index * samples_per_frame_;
      const uint64_t local_data_end_sample = local_frame_start_sample + static_cast<uint64_t>(complex_samples_read);
      const uint64_t local_frame_end_sample = drain_frame ? local_data_end_sample : local_frame_start_sample + samples_per_frame_;
      const uint64_t frame_start_sample = saturated_add_u64(global_sample_start, local_frame_start_sample);
      const uint64_t data_end_sample = saturated_add_u64(global_sample_start, local_data_end_sample);
      const uint64_t frame_end_sample = saturated_add_u64(global_sample_start, local_frame_end_sample);
      meta->set("offline_source_drain_frame", drain_frame);
      meta->set("offline_source_real_frame_count", real_frame_count_limit_);
      meta->set("offline_source_drain_frame_count", static_cast<uint64_t>(std::max<int64_t>(0, drain_frame_count_.get())));
      meta->set("offline_source_capture_sample_start", global_sample_start);
      meta->set("offline_source_file_offset_complex", frame_start_sample);
      meta->set("offline_source_data_end_complex", data_end_sample);
      meta->set("offline_source_frame_end_complex", frame_end_sample);
      meta->set("offline_source_global_sample_start", frame_start_sample);
      meta->set("offline_source_global_data_end_sample", data_end_sample);
      meta->set("offline_source_global_frame_end_sample", frame_end_sample);
      meta->set("offline_source_global_total_sample_start", global_sample_start);
      meta->set("offline_source_global_total_sample_end",
            saturated_add_u64(global_sample_start,
                  static_cast<uint64_t>(std::max<int64_t>(0, total_complex_samples_.get()))));
      meta->set("offline_source_local_file_offset_complex", local_frame_start_sample);
      meta->set("offline_source_local_data_end_complex", local_data_end_sample);
      meta->set("offline_source_local_frame_end_complex", local_frame_end_sample);
      meta->set("offline_source_complex_samples_read", static_cast<uint64_t>(complex_samples_read));
      meta->set("offline_source_complex_samples_padded", static_cast<uint64_t>(0));
      meta->set("offline_source_partial_frame", false);
    }

    op_output.emit(FftInputMessage {slot.device_tensor, slot.stream}, "out");
    emitted_frames_++;
  }

  void stop() override {
    for (auto& slot : slots_) {
      if (slot.stream != nullptr) {
        cudaStreamSynchronize(slot.stream);
        cudaStreamDestroy(slot.stream);
        slot.stream = nullptr;
      }
    }
    slots_.clear();
    if (input_.is_open()) {
      input_.close();
    }
    Operator::stop();
  }

 private:
  struct Slot {
    matx::tensor_t<Complex, 2> device_tensor;
    cudaStream_t stream = nullptr;
  };

  holoscan::Parameter<std::string> input_file_path_;
  holoscan::Parameter<std::string> input_datatype_;
  holoscan::Parameter<int> num_bursts_;
  holoscan::Parameter<int> burst_size_;
  holoscan::Parameter<int> channel_number_;
  holoscan::Parameter<double> span_hz_;
  holoscan::Parameter<int64_t> global_sample_start_;
  holoscan::Parameter<int64_t> total_complex_samples_;
  holoscan::Parameter<int64_t> real_frame_count_;
  holoscan::Parameter<int64_t> drain_frame_count_;
  holoscan::Parameter<int> ring_size_;

  std::ifstream input_;
  uint64_t emitted_frames_ = 0;
  uint64_t samples_per_frame_ = 0;
  uint64_t real_frame_count_limit_ = 0;
  OfflineInputFormat input_format_;
  std::vector<uint8_t> host_input_bytes_;
  std::vector<Complex> host_complex_;
  std::vector<Slot> slots_;
};

class SpectrogramArtifactSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrogramArtifactSinkOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::ops::out_t>("in");
    spec.param(output_root_, "output_root", "Output Root", "Offline evaluation artifact root.");
    spec.param(save_preview_,
               "save_preview",
               "Save Preview",
               "Save a PGM spectrogram preview for each processed frame.",
               true);
    spec.param(save_tensor_,
               "save_tensor",
               "Save Tensor",
               "Save the full detector-input spectrogram tensor for each processed frame.",
               true);
    spec.param(output_height_, "output_height", "Output Height", "Preview spectrogram height.", 256);
    spec.param(output_width_, "output_width", "Output Width", "Preview spectrogram width.", 512);
  }

  void initialize() override {
    Operator::initialize();
    std::filesystem::create_directories(std::filesystem::path(output_root_.get()) / "spectrograms");
    std::filesystem::create_directories(std::filesystem::path(output_root_.get()) / "spectrogram_tensors");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto maybe_input = op_input.receive<holoscan::ops::out_t>("in");
    if (!maybe_input) {
      return;
    }

    const auto& [tensor, stream] = *maybe_input;
    static_cast<void>(stream);
    auto meta = metadata();
    const int channel = meta ? static_cast<int>(meta->get<uint16_t>("channel_number", 0)) : 0;
    uint64_t frame_number = 0;
    if (meta && meta->has_key("fft_emitted_frame_number")) {
      frame_number = meta->get<uint64_t>("fft_emitted_frame_number", fallback_frame_number_ + 1);
    }
    if (frame_number == 0) {
      frame_number = ++fallback_frame_number_;
    } else {
      fallback_frame_number_ = std::max(fallback_frame_number_, frame_number);
    }

    const int src_rows = static_cast<int>(tensor.Size(0));
    const int src_cols = static_cast<int>(tensor.Size(1));
    const int dst_rows = std::max(1, output_height_.get());
    const int dst_cols = std::max(1, output_width_.get());

    std::string preview_rel_path;
    std::string tensor_rel_path;
    const auto output_root = std::filesystem::path(output_root_.get());
    std::vector<Complex> host_fft;

    if (save_tensor_.get() || save_preview_.get()) {
      host_fft.resize(static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols));
      const auto copy_result = cudaMemcpyAsync(host_fft.data(),
                                               tensor.Data(),
                                               host_fft.size() * sizeof(Complex),
                                               cudaMemcpyDeviceToHost,
                                               stream);
      if (copy_result != cudaSuccess) {
        throw std::runtime_error("spectrogram sink failed to copy FFT tensor to host");
      }
      const auto sync_result = cudaStreamSynchronize(stream);
      if (sync_result != cudaSuccess) {
        throw std::runtime_error("spectrogram sink failed to synchronize CUDA stream");
      }
    }

    if (save_tensor_.get()) {
      const auto tensor_path = make_artifact_path(output_root,
                                                  "spectrogram_tensors",
                                                  "spectrogram_tensor",
                                                  channel,
                                                  frame_number,
                                                  src_rows,
                                                  src_cols,
                                                  ".npy");
      if (!write_npy_2d(tensor_path,
                        host_fft.data(),
                        host_fft.size() * sizeof(Complex),
                        src_rows,
                        src_cols,
                        "<c8")) {
        throw std::runtime_error("failed to write spectrogram tensor artifact: " + tensor_path.string());
      }
      tensor_rel_path = relative_to_output_root(tensor_path);
    }

    if (save_preview_.get()) {
      const auto preview = build_spectrogram_preview(host_fft, src_rows, src_cols, dst_rows, dst_cols);
      const auto preview_path = make_artifact_path(output_root,
                                                   "spectrograms",
                                                   "spectrogram_preview",
                                                   channel,
                                                   frame_number,
                                                   dst_rows,
                                                   dst_cols,
                                                   ".pgm");
      if (!write_pgm(preview_path, preview, dst_cols, dst_rows)) {
        throw std::runtime_error("failed to write spectrogram preview artifact: " + preview_path.string());
      }
      preview_rel_path = relative_to_output_root(preview_path);
    }

    register_spectrogram_artifacts(channel,
                                   frame_number,
                                   src_rows,
                                   src_cols,
                                   dst_rows,
                                   dst_cols,
                                   preview_rel_path,
                                   tensor_rel_path,
                                   meta);

  }

 private:
  holoscan::Parameter<std::string> output_root_;
  holoscan::Parameter<bool> save_preview_;
  holoscan::Parameter<bool> save_tensor_;
  holoscan::Parameter<int> output_height_;
  holoscan::Parameter<int> output_width_;
  uint64_t fallback_frame_number_ = 0;
};

class MaskArtifactSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MaskArtifactSinkOp)

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<holoscan::ops::DetectorMaskMessage>("in");
    spec.param(output_root_, "output_root", "Output Root", "Offline evaluation artifact root.");
    spec.param(save_preview_,
               "save_preview",
               "Save Preview",
               "Save a PGM preview mask for each processed frame.",
               true);
    spec.param(save_npy_,
               "save_npy",
               "Save NPY",
               "Save the full-resolution detector mask as an NPY file for each processed frame.",
               true);
    spec.param(output_height_, "output_height", "Output Height", "Preview mask height.", 256);
    spec.param(output_width_, "output_width", "Output Width", "Preview mask width.", 512);
    spec.param(total_frames_, "total_frames", "Total Frames", "Expected total frames.", static_cast<int64_t>(0));
    spec.param(progress_every_n_frames_,
               "progress_every_n_frames",
               "Progress Every N Frames",
               "Log progress every N processed masks.",
               1);
  }

  void initialize() override {
    Operator::initialize();
    std::filesystem::create_directories(std::filesystem::path(output_root_.get()) / "mask_previews");
    std::filesystem::create_directories(std::filesystem::path(output_root_.get()) / "mask_arrays");
  }

  void compute(holoscan::InputContext& op_input,
               holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    auto maybe_mask = op_input.receive<holoscan::ops::DetectorMaskMessage>("in");
    if (!maybe_mask) {
      return;
    }

    const auto& mask = *maybe_mask;
    auto meta = metadata();
    const int64_t total_frames = std::max<int64_t>(0, total_frames_.get());
    if (total_frames > 0 && mask.frame_number > static_cast<uint64_t>(total_frames)) {
      HOLOSCAN_LOG_INFO("Ignoring offline drain mask frame {} beyond expected {} real frames",
                        mask.frame_number,
                        total_frames);
      return;
    }
    std::vector<uint8_t> host_mask(static_cast<size_t>(mask.width) * static_cast<size_t>(mask.height), 0);

    if (mask.device_pixels) {
      const auto copy_result = cudaMemcpy(host_mask.data(),
                                          mask.device_pixels.get(),
                                          host_mask.size() * sizeof(uint8_t),
                                          cudaMemcpyDeviceToHost);
      if (copy_result != cudaSuccess) {
        throw std::runtime_error("mask sink failed to copy detector mask to host");
      }
    } else if (!mask.pixels.empty()) {
      host_mask = mask.pixels;
    }

    std::string preview_rel_path;
    std::string mask_rel_path;
    const auto output_root = std::filesystem::path(output_root_.get());
    const int dst_rows = std::max(1, output_height_.get());
    const int dst_cols = std::max(1, output_width_.get());

    if (save_npy_.get()) {
      const auto mask_path = make_artifact_path(output_root,
                                                "mask_arrays",
                                                "mask",
                                                mask.channel,
                                                mask.frame_number,
                                                mask.height,
                                                mask.width,
                                                ".npy");
      if (!write_npy_2d(mask_path,
                        host_mask.data(),
                        host_mask.size() * sizeof(uint8_t),
                        mask.height,
                        mask.width,
                        "|u1")) {
        throw std::runtime_error("failed to write mask npy artifact: " + mask_path.string());
      }
      mask_rel_path = relative_to_output_root(mask_path);
    }

    if (save_preview_.get()) {
      const auto preview = build_mask_preview(host_mask, mask.height, mask.width, dst_rows, dst_cols);
      const auto preview_path = make_artifact_path(output_root,
                                                   "mask_previews",
                                                   "mask_preview",
                                                   mask.channel,
                                                   mask.frame_number,
                                                   dst_rows,
                                                   dst_cols,
                                                   ".pgm");
      if (!write_pgm(preview_path, preview, dst_cols, dst_rows)) {
        throw std::runtime_error("failed to write mask preview artifact: " + preview_path.string());
      }
      preview_rel_path = relative_to_output_root(preview_path);
    }

    register_mask_artifacts(mask.channel,
                            mask.frame_number,
                            dst_rows,
                            dst_cols,
                            preview_rel_path,
                            mask_rel_path,
                            mask,
                            meta);

    processed_frames_++;
    const uint64_t complex_samples_read =
      mask.complex_samples_read != 0
        ? mask.complex_samples_read
        : (meta ? meta->get<uint64_t>("offline_source_complex_samples_read", 0) : 0);
    processed_complex_samples_ += complex_samples_read;
    const int progress_stride = std::max(1, progress_every_n_frames_.get());
    if ((processed_frames_ % static_cast<uint64_t>(progress_stride)) == 0 ||
        (total_frames > 0 && processed_frames_ == static_cast<uint64_t>(total_frames))) {
      const double percent = total_frames > 0
                                 ? (100.0 * static_cast<double>(processed_frames_) /
                                    static_cast<double>(total_frames))
                                 : 0.0;
      HOLOSCAN_LOG_INFO(
          "Offline eval progress frame {}/{} ({:.1f}%), complex samples processed {}/{}",
          processed_frames_,
          total_frames,
          percent,
          processed_complex_samples_,
          meta ? meta->get<uint64_t>("offline_source_total_complex_samples", 0) : 0);
    }

    if (total_frames > 0 && processed_frames_ >= static_cast<uint64_t>(total_frames)) {
      auto done_condition = this->condition<holoscan::BooleanCondition>("offline_eval_artifacts_pending");
      if (done_condition) {
        HOLOSCAN_LOG_INFO("Offline eval artifact sink received all {} expected frames", total_frames);
        done_condition->disable_tick();
      }
    }
  }

  void stop() override {
    const int64_t total_frames = std::max<int64_t>(0, total_frames_.get());
    if (total_frames > 0 && processed_frames_ < static_cast<uint64_t>(total_frames)) {
      HOLOSCAN_LOG_WARN("Offline eval artifact sink stopped after {}/{} expected frames",
                        processed_frames_,
                        total_frames);
    }
    Operator::stop();
  }

 private:
  holoscan::Parameter<std::string> output_root_;
  holoscan::Parameter<bool> save_preview_;
  holoscan::Parameter<bool> save_npy_;
  holoscan::Parameter<int> output_height_;
  holoscan::Parameter<int> output_width_;
  holoscan::Parameter<int64_t> total_frames_;
  holoscan::Parameter<int> progress_every_n_frames_;
  uint64_t processed_frames_ = 0;
  uint64_t processed_complex_samples_ = 0;
};

class OfflineCudaDetectorEvalApp : public holoscan::Application {
 public:
  void set_overrides(EvalOverrides overrides) {
    overrides_ = std::move(overrides);
  }

  void compose() override {
    using namespace holoscan;

    std::fprintf(stderr,
                 "[offline_cuda_detector_eval] compose: total_frames=%llu drain_frames=%llu output_root='%s' debug_artifacts=%d\n",
                 static_cast<unsigned long long>(overrides_.total_frames),
                 static_cast<unsigned long long>(overrides_.drain_frame_count),
                 overrides_.output_root.string().c_str(),
                 overrides_.save_detector_debug_artifacts ? 1 : 0);

    auto source = make_operator<OfflineSc16FileSourceOp>(
        "offlineSc16FileSourceOp",
        make_condition<CountCondition>("offline_frame_count",
                       static_cast<int64_t>(overrides_.total_frames + overrides_.drain_frame_count)),
        Arg("input_file_path") = overrides_.input_file_path.string(),
        Arg("input_datatype") = overrides_.input_datatype,
        Arg("num_bursts") = overrides_.fft_num_bursts,
        Arg("burst_size") = overrides_.fft_burst_size,
        Arg("channel_number") = overrides_.channel_number,
        Arg("span_hz") = overrides_.span_hz,
        Arg("global_sample_start") = static_cast<int64_t>(std::min<uint64_t>(
            overrides_.input_capture_sample_start,
            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))),
        Arg("total_complex_samples") = static_cast<int64_t>(overrides_.total_complex_samples),
        Arg("real_frame_count") = static_cast<int64_t>(overrides_.total_frames),
        Arg("drain_frame_count") = static_cast<int64_t>(overrides_.drain_frame_count),
        Arg("ring_size") = static_cast<int>(std::max<uint64_t>(1, overrides_.total_frames + overrides_.drain_frame_count)));

    auto fft = make_operator<holoscan::ops::FFT>(
        "fftOpCh0",
        from_config("fft"),
        Arg("num_channels") = static_cast<uint16_t>(1),
        Arg("burst_size") = overrides_.fft_burst_size,
        Arg("num_bursts") = overrides_.fft_num_bursts,
        Arg("transform_points") = static_cast<uint32_t>(overrides_.fft_burst_size),
        Arg("window_points") = static_cast<uint32_t>(overrides_.fft_burst_size),
        Arg("resolution") = overrides_.resolution_hz,
        Arg("span") = static_cast<uint64_t>(std::llround(overrides_.span_hz)),
        Arg("f1_index") = -(overrides_.fft_burst_size / 2),
        Arg("f2_index") = (overrides_.fft_burst_size / 2) - 1,
        Arg("emit_stride") = 1);

    auto spectrogram = make_operator<holoscan::ops::Spectrogram>(
        "spectrogramOpCh0",
        from_config("spectrogram"),
        Arg("num_channels") = 1,
      Arg("enable_save") = false,
      Arg("enable_tensor_save") = false);

    auto spectrogram_sink = make_operator<SpectrogramArtifactSinkOp>(
        "spectrogramArtifactSinkOp",
        Arg("output_root") = overrides_.output_root.string(),
        Arg("save_preview") = overrides_.save_spectrogram_preview,
        Arg("save_tensor") = overrides_.save_spectrogram_tensor,
        Arg("output_height") = overrides_.spectrogram_output_height,
        Arg("output_width") = overrides_.spectrogram_output_width);

    std::fprintf(stderr,
           "[offline_cuda_detector_eval] compose: creating detector debug_artifacts=%d aligned_preview=%d aligned_tensor=%d\n",
           overrides_.save_detector_debug_artifacts ? 1 : 0,
           overrides_.save_spectrogram_preview ? 1 : 0,
           overrides_.save_spectrogram_tensor ? 1 : 0);

    auto detector = make_operator<holoscan::ops::CudaDinoDetector>(
        "cudaDinoDetectorOpCh0",
        from_config("cuda_dino_detector"),
        Arg("num_channels") = 1,
        Arg("channel_filter") = overrides_.channel_number,
        Arg("emit_stride") = 1,
      Arg("debug_mode") = overrides_.save_detector_debug_artifacts,
      Arg("enable_debug_artifact_host_copy") = overrides_.save_detector_debug_artifacts,
      Arg("debug_artifact_output_dir") =
        (overrides_.save_detector_debug_artifacts ? overrides_.output_root.string() : std::string {}),
        Arg("save_aligned_spectrogram_preview") = overrides_.save_spectrogram_preview,
        Arg("save_aligned_spectrogram_tensor") = overrides_.save_spectrogram_tensor,
        Arg("aligned_spectrogram_output_height") = overrides_.spectrogram_output_height,
        Arg("aligned_spectrogram_output_width") = overrides_.spectrogram_output_width,
        Arg("aligned_spectrogram_output_dir") = overrides_.output_root.string());

    auto mask_sink = make_operator<MaskArtifactSinkOp>(
        "maskArtifactSinkOp",
        make_condition<BooleanCondition>("offline_eval_artifacts_pending", true),
        Arg("output_root") = overrides_.output_root.string(),
        Arg("save_preview") = overrides_.save_mask_preview,
        Arg("save_npy") = overrides_.save_mask_npy,
        Arg("output_height") = overrides_.spectrogram_output_height,
        Arg("output_width") = overrides_.spectrogram_output_width,
        Arg("total_frames") = static_cast<int64_t>(overrides_.total_frames),
        Arg("progress_every_n_frames") = overrides_.progress_every_n_frames);

      std::fprintf(stderr, "[offline_cuda_detector_eval] compose: wiring flows\n");

    add_flow(source, fft);
    add_flow(fft, spectrogram);
    add_flow(spectrogram, spectrogram_sink);
    add_flow(spectrogram, detector);
    add_flow(detector, mask_sink, {{"mask_out", "in"}});
  }

 private:
  EvalOverrides overrides_;
};

EvalOverrides load_overrides(holoscan::Application& app,
                             const std::filesystem::path& config_path,
                             const CliOptions& cli_options) {
  EvalOverrides overrides;
  overrides.config_path = config_path;
  overrides.run_offline_on_file =
      usrp_wideband::from_config_or<bool>(app, "offline_eval.run_offline_on_file", true);
  overrides.input_file_path = resolve_runtime_path(
      config_path,
      cli_options.input_file_path.empty()
          ? usrp_wideband::from_config_or<std::string>(app, "offline_eval.input_file_path", std::string {})
          : cli_options.input_file_path);
  overrides.output_root = resolve_runtime_path(
      config_path,
      cli_options.output_root.empty()
          ? usrp_wideband::from_config_or<std::string>(app,
                                                       "offline_eval.output_root",
                                                       std::string("/tmp/usrp_offline_cuda_detector_eval"))
          : cli_options.output_root);
    overrides.save_detector_debug_artifacts =
      usrp_wideband::from_config_or<bool>(app, "offline_eval.save_detector_debug_artifacts", false);
  overrides.save_spectrogram_preview =
      usrp_wideband::from_config_or<bool>(app, "offline_eval.save_spectrogram_preview", true);
  overrides.save_spectrogram_tensor =
      usrp_wideband::from_config_or<bool>(app, "offline_eval.save_spectrogram_tensor", true);
  overrides.save_mask_preview =
      usrp_wideband::from_config_or<bool>(app, "offline_eval.save_mask_preview", true);
  overrides.save_mask_npy =
      usrp_wideband::from_config_or<bool>(app, "offline_eval.save_mask_npy", true);
  overrides.progress_every_n_frames =
      cli_options.progress_every_n_frames > 0
          ? cli_options.progress_every_n_frames
          : usrp_wideband::from_config_or<int>(app, "offline_eval.progress_every_n_frames", 0);
    overrides.drain_frame_count = static_cast<uint64_t>(
      std::max<int>(0, usrp_wideband::from_config_or<int>(app, "offline_eval.drain_frame_count", 32)));
  overrides.channel_number = usrp_wideband::from_config_or<int>(app, "offline_eval.channel_number", 0);
  overrides.fft_num_bursts = app.from_config("fft.num_bursts").as<int>();

  const auto sigmf_input_format = try_load_sigmf_input_format(overrides.input_file_path);
  if (sigmf_input_format.has_value()) {
    overrides.input_datatype = sigmf_input_format->datatype;
    overrides.input_sample_rate_hz = sigmf_input_format->sample_rate_hz;
    overrides.input_sigmf_meta_path = sigmf_input_format->sigmf_meta_path;
    overrides.input_capture_sample_start = sigmf_input_format->capture_sample_start;
    overrides.input_center_frequency_hz = sigmf_input_format->center_frequency_hz;
    overrides.has_input_center_frequency_hz = sigmf_input_format->has_center_frequency_hz;
    overrides.input_num_channels = sigmf_input_format->num_channels;
  }

  const auto fft_runtime = usrp_wideband::resolve_fft_runtime_config(
      app,
      overrides.input_sample_rate_hz > 0.0 ? std::optional<double> {overrides.input_sample_rate_hz} : std::nullopt);
  overrides.fft_burst_size = fft_runtime.actual_fft_size;
  overrides.span_hz = fft_runtime.active_span_hz;
  overrides.resolution_hz = fft_runtime.resolution_hz;
  if (overrides.input_sample_rate_hz <= 0.0) {
    overrides.input_sample_rate_hz = overrides.span_hz;
  }
  overrides.spectrogram_output_height = app.from_config("spectrogram.output_height").as<int>();
  overrides.spectrogram_output_width = app.from_config("spectrogram.output_width").as<int>();
  overrides.samples_per_frame = static_cast<uint64_t>(std::max(1, overrides.fft_num_bursts)) *
                                static_cast<uint64_t>(std::max(1, overrides.fft_burst_size));

  if (!overrides.run_offline_on_file) {
    throw std::runtime_error("offline_eval.run_offline_on_file must be true for run_offline_cuda_detector_eval");
  }
  if (overrides.input_file_path.empty()) {
    throw std::runtime_error("offline_eval.input_file_path is required or pass --input-file");
  }
  if (!std::filesystem::exists(overrides.input_file_path)) {
    throw std::runtime_error("offline input file does not exist: " + overrides.input_file_path.string());
  }

  const auto input_format = make_input_format_from_datatype(overrides.input_datatype);
  const uint64_t file_size_bytes = std::filesystem::file_size(overrides.input_file_path);
  if ((file_size_bytes % input_format.bytes_per_complex) != 0) {
    throw std::runtime_error("offline input file size is not an integer number of complex samples for datatype '" +
                             overrides.input_datatype + "'");
  }
  overrides.input_total_complex_samples = file_size_bytes / input_format.bytes_per_complex;
  overrides.total_frames = overrides.input_total_complex_samples / overrides.samples_per_frame;
  overrides.total_complex_samples = overrides.total_frames * overrides.samples_per_frame;
  overrides.dropped_tail_complex_samples = overrides.input_total_complex_samples - overrides.total_complex_samples;
  if (overrides.total_frames == 0) {
    throw std::runtime_error("offline input file contains " + std::to_string(overrides.input_total_complex_samples) +
                             " complex samples, which is less than one complete frame of " +
                             std::to_string(overrides.samples_per_frame) + " samples");
  }
  if (overrides.progress_every_n_frames <= 0) {
    overrides.progress_every_n_frames =
        std::max<int>(1, static_cast<int>(std::max<uint64_t>(1, overrides.total_frames / 100U)));
  }
  return overrides;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto cli_options = parse_arguments(argc, argv);
    const auto config_path = resolve_config_path(argv[0], cli_options.config_path);
    if (!std::filesystem::exists(config_path)) {
      HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist", static_cast<std::string>(config_path));
      return -1;
    }

    auto app = holoscan::make_application<OfflineCudaDetectorEvalApp>();
    app->enable_metadata(true);
    app->config(config_path);

    const auto overrides = load_overrides(*app, config_path, cli_options);
    std::fprintf(stderr, "[offline_cuda_detector_eval] main: loaded overrides\n");
    std::fflush(stderr);
    clear_output_root(overrides.output_root);
    std::fprintf(stderr, "[offline_cuda_detector_eval] main: cleared output root\n");
    std::fflush(stderr);
    reset_artifact_registry(overrides.output_root);
    std::fprintf(stderr, "[offline_cuda_detector_eval] main: applying overrides\n");
    std::fflush(stderr);
    app->set_overrides(overrides);
    std::fprintf(stderr, "[offline_cuda_detector_eval] main: creating scheduler\n");
    std::fflush(stderr);
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("event-based-scheduler",
                                      app->from_config("scheduler")));

    if (!overrides.input_sigmf_meta_path.empty()) {
      HOLOSCAN_LOG_INFO("Using SigMF metadata '{}' datatype='{}' sample_rate_hz={}",
                        overrides.input_sigmf_meta_path.string(),
                        overrides.input_datatype,
                        overrides.input_sample_rate_hz);
    } else {
      HOLOSCAN_LOG_WARN("No SigMF metadata sidecar found for input '{}'; replay is using fallback datatype='{}' sample_rate_hz={} from configuration",
                        overrides.input_file_path.string(),
                        overrides.input_datatype,
                        overrides.input_sample_rate_hz);
    }

    HOLOSCAN_LOG_INFO(
      "Starting offline CUDA detector eval input='{}' datatype='{}' sample_rate_hz={} full_frames={} input_samples={} processed_samples={} dropped_tail_samples={} samples_per_frame={} output_root='{}'",
        overrides.input_file_path.string(),
        overrides.input_datatype,
        overrides.input_sample_rate_hz,
        overrides.total_frames,
      overrides.input_total_complex_samples,
      overrides.total_complex_samples,
      overrides.dropped_tail_complex_samples,
        overrides.samples_per_frame,
        overrides.output_root.string());
      HOLOSCAN_LOG_INFO("Scheduling {} real frames plus {} drain frames to flush downstream operators",
                overrides.total_frames,
                overrides.drain_frame_count);

    std::fprintf(stderr, "[offline_cuda_detector_eval] main: entering app->run()\n");
    std::fflush(stderr);
    app->run();
    std::fprintf(stderr, "[offline_cuda_detector_eval] main: app->run() returned\n");
    std::fflush(stderr);
    write_manifest(overrides);

    HOLOSCAN_LOG_INFO("Offline CUDA detector eval complete. Wrote manifest '{}'",
                      (overrides.output_root / "frame_manifest.csv").string());
    if (overrides.dropped_tail_complex_samples > 0) {
      HOLOSCAN_LOG_WARN(
          "Dropped {} complex samples from the end of the input file because they were not enough to make a full frame of {} samples",
          overrides.dropped_tail_complex_samples,
          overrides.samples_per_frame);
    } else {
      HOLOSCAN_LOG_INFO("Dropped 0 complex samples from the end of the input file; input ended on a full-frame boundary");
    }
    return 0;
  } catch (const std::exception& exception) {
    HOLOSCAN_LOG_ERROR("run_offline_cuda_detector_eval failed: {}", exception.what());
    return -1;
  }
}
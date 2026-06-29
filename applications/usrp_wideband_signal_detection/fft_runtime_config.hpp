// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>

namespace usrp_wideband {

constexpr double kDefaultReferenceSpanHz = 500.0e6;
constexpr int kDefaultReferenceFftSize = 20480;

struct FftRuntimeConfig {
  double reference_span_hz = kDefaultReferenceSpanHz;
  int reference_fft_size = kDefaultReferenceFftSize;
  double active_span_hz = kDefaultReferenceSpanHz;
  double target_bin_size_hz = kDefaultReferenceSpanHz / static_cast<double>(kDefaultReferenceFftSize);
  double override_fft_bin_size_hz = 0.0;
  int requested_fft_size = kDefaultReferenceFftSize;
  int actual_fft_size = kDefaultReferenceFftSize;
  int packet_samples = 1024;
  int num_packets_per_fft = kDefaultReferenceFftSize / 1024;
  uint64_t resolution_hz = 0;
  int f1_index = -(kDefaultReferenceFftSize / 2);
  int f2_index = (kDefaultReferenceFftSize / 2) - 1;
  bool used_override_fft_bin_size = false;
  bool used_channel_sample_rate = false;
};

template <typename T>
T from_config_or(holoscan::Application& app, const std::string& key, T default_value) {
  try {
    return app.from_config(key).as<T>();
  } catch (const std::exception&) {
    return default_value;
  }
}

inline std::vector<double> parse_csv_doubles(const std::string& values) {
  std::vector<double> parsed_values;
  std::stringstream stream(values);
  std::string token;
  while (std::getline(stream, token, ',')) {
    const auto first = token.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
      continue;
    }
    const auto last = token.find_last_not_of(" \t\n\r");
    token = token.substr(first, last - first + 1);
    try {
      parsed_values.push_back(std::stod(token));
    } catch (const std::exception&) {
      HOLOSCAN_LOG_WARN("Ignoring unparseable channel_sample_rates_hz token '{}' while deriving FFT size", token);
      return {};
    }
  }
  return parsed_values;
}

inline std::optional<double> resolve_common_channel_sample_rate_hz(holoscan::Application& app,
                                                                   bool& conflicting_values) {
  conflicting_values = false;
  const auto configured_values =
      from_config_or<std::string>(app, "chdr_converter.channel_sample_rates_hz", std::string {});
  const auto parsed_values = parse_csv_doubles(configured_values);
  if (parsed_values.empty()) {
    return std::nullopt;
  }

  const double first_value = parsed_values.front();
  if (!std::isfinite(first_value) || first_value <= 0.0) {
    return std::nullopt;
  }

  for (size_t index = 1; index < parsed_values.size(); ++index) {
    const double value = parsed_values[index];
    if (!std::isfinite(value) || value <= 0.0) {
      conflicting_values = true;
      return std::nullopt;
    }
    const double tolerance_hz = std::max(1.0, std::max(std::abs(first_value), std::abs(value)) * 1.0e-6);
    if (std::abs(value - first_value) > tolerance_hz) {
      conflicting_values = true;
      return std::nullopt;
    }
  }

  return first_value;
}

inline FftRuntimeConfig resolve_fft_runtime_config(holoscan::Application& app,
                                                   std::optional<double> explicit_span_hz = std::nullopt) {
  FftRuntimeConfig config;

  config.reference_span_hz = std::max(
      1.0,
      from_config_or<double>(app, "fft.reference_span_hz", kDefaultReferenceSpanHz));
  config.reference_fft_size = std::max(
      1,
      from_config_or<int>(app, "fft.reference_fft_size", kDefaultReferenceFftSize));
  config.override_fft_bin_size_hz = std::max(
      0.0,
      from_config_or<double>(app, "fft.override_fft_bin_size", 0.0));
  config.packet_samples = std::max(
      1,
      from_config_or<int>(app, "chdr_converter.num_complex_samples_per_packet", 1024));

  const double configured_span_hz = std::max(
      1.0,
      from_config_or<double>(app, "fft.span", config.reference_span_hz));
  std::string span_source = "fft.span";
  if (explicit_span_hz.has_value() && std::isfinite(explicit_span_hz.value()) && explicit_span_hz.value() > 0.0) {
    config.active_span_hz = explicit_span_hz.value();
    span_source = "explicit_input_sample_rate_hz";
  } else {
    bool conflicting_channel_sample_rates = false;
    const auto common_channel_sample_rate_hz =
        resolve_common_channel_sample_rate_hz(app, conflicting_channel_sample_rates);
    if (common_channel_sample_rate_hz.has_value()) {
      config.active_span_hz = common_channel_sample_rate_hz.value();
      config.used_channel_sample_rate = true;
      span_source = "chdr_converter.channel_sample_rates_hz";
    } else {
      config.active_span_hz = configured_span_hz;
      if (conflicting_channel_sample_rates) {
        HOLOSCAN_LOG_WARN(
            "Configured channel_sample_rates_hz values differ across channels, so FFT scaling is falling back to fft.span={} Hz",
            configured_span_hz);
      }
    }
  }

  const double reference_bin_size_hz =
      config.reference_span_hz / static_cast<double>(std::max(1, config.reference_fft_size));
  config.target_bin_size_hz = reference_bin_size_hz;

  double requested_fft_size = static_cast<double>(config.reference_fft_size);
  if (config.override_fft_bin_size_hz > 0.0) {
    config.used_override_fft_bin_size = true;
    config.target_bin_size_hz = config.override_fft_bin_size_hz;
    requested_fft_size = config.active_span_hz / config.override_fft_bin_size_hz;
  } else {
    const double span_ratio = config.active_span_hz / config.reference_span_hz;
    if (std::isfinite(span_ratio) && span_ratio > 0.0) {
      const double snapped_ratio = std::exp2(std::round(std::log2(span_ratio)));
      requested_fft_size = static_cast<double>(config.reference_fft_size) * snapped_ratio;
    }
  }

  config.requested_fft_size = std::max(1, static_cast<int>(std::llround(requested_fft_size)));
  config.num_packets_per_fft = std::max(
      1,
      static_cast<int>(std::llround(static_cast<double>(config.requested_fft_size) /
                                    static_cast<double>(config.packet_samples))));
  config.actual_fft_size = std::max(config.packet_samples, config.num_packets_per_fft * config.packet_samples);
  config.f1_index = -(config.actual_fft_size / 2);
  config.f2_index = config.f1_index + config.actual_fft_size - 1;
  config.resolution_hz = static_cast<uint64_t>(std::llround(
      config.active_span_hz / static_cast<double>(std::max(1, config.actual_fft_size))));

  HOLOSCAN_LOG_INFO(
      "Derived runtime FFT config span_hz={} source={} reference_span_hz={} reference_fft_size={} requested_fft_size={} actual_fft_size={} packet_samples={} packets_per_fft={} resolution_hz={} override_fft_bin_size={}",
      config.active_span_hz,
      span_source,
      config.reference_span_hz,
      config.reference_fft_size,
      config.requested_fft_size,
      config.actual_fft_size,
      config.packet_samples,
      config.num_packets_per_fft,
      config.resolution_hz,
      config.override_fft_bin_size_hz);

  return config;
}

}  // namespace usrp_wideband
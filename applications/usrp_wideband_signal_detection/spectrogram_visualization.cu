#include "spectrogram_visualization.hpp"

#include <cuda/std/detail/libcxx/include/algorithm>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <matx.h>
#include <gxf/std/tensor.hpp>
#include <holoviz/holoviz.hpp>
#include <imgui.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>

namespace holoscan::advanced_network {
bool adv_net_shutdown_requested();
}

namespace {

bool adv_net_shutdown_requested_if_available() {
#if defined(ANO_MGR_DPDK) || defined(ANO_MGR_GPUNETIO) || defined(ANO_MGR_dpdk) || defined(ANO_MGR_gpunetio)
  return holoscan::advanced_network::adv_net_shutdown_requested();
#else
  return false;
#endif
}

using SpectrogramComplex = cuda::std::complex<float>;
using SpectrogramTensor = matx::tensor_t<SpectrogramComplex, 2>;
using SpectrogramMessage = std::tuple<SpectrogramTensor, cudaStream_t>;

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

constexpr int kLiveMaskGroupingMinComponentSize = 24;
constexpr int kLiveMaskGroupingMinFreqSpan = 18;
constexpr int kLiveMaskGroupingMinTimeSpan = 2;
constexpr float kLiveMaskGroupingMinDensity = 0.06f;

__global__ void reduce_binary_mask_to_alpha_kernel(const uint8_t* input,
                                                   int src_rows,
                                                   int src_cols,
                                                   uint8_t* output,
                                                   int dst_rows,
                                                   int dst_cols) {
  const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  if (out_col >= dst_cols || out_row >= dst_rows) {
    return;
  }

  const int row_start = (out_row * src_rows) / dst_rows;
  const int row_end = max(row_start + 1, ((out_row + 1) * src_rows) / dst_rows);
  const int col_start = (out_col * src_cols) / dst_cols;
  const int col_end = max(col_start + 1, ((out_col + 1) * src_cols) / dst_cols);

  int active = 0;
  int count = 0;
  for (int src_row = row_start; src_row < row_end; ++src_row) {
    const size_t row_offset = static_cast<size_t>(src_row) * static_cast<size_t>(src_cols);
    for (int src_col = col_start; src_col < col_end; ++src_col) {
      active += input[row_offset + static_cast<size_t>(src_col)] > 0 ? 1 : 0;
      ++count;
    }
  }

  const float occupancy = count > 0 ? static_cast<float>(active) / static_cast<float>(count) : 0.0f;
  output[static_cast<size_t>(out_row) * static_cast<size_t>(dst_cols) + static_cast<size_t>(out_col)] =
      static_cast<uint8_t>(std::lround(std::clamp(occupancy, 0.0f, 1.0f) * 255.0f));
}

__global__ void count_nonzero_u8_kernel(const uint8_t* input,
                                        int total,
                                        unsigned int* output_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  if (input[index] != 0) {
    atomicAdd(output_count, 1U);
  }
}

std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return {};
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::filesystem::path> list_pgm_frames(const std::filesystem::path& directory,
                                                   int channel_filter) {
  std::vector<std::filesystem::path> frames;

  if (!std::filesystem::exists(directory)) {
    return frames;
  }

  const std::string channel_token = channel_filter >= 0 ? "ch" + std::to_string(channel_filter) + "_" : "";
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto& path = entry.path();
    if (path.extension() != ".pgm") {
      continue;
    }
    if (!channel_token.empty() && path.filename().string().find(channel_token) == std::string::npos) {
      continue;
    }
    frames.push_back(path);
  }

  std::sort(frames.begin(), frames.end());
  return frames;
}

bool load_pgm_file(const std::filesystem::path& path,
                   std::vector<uint8_t>& pixels,
                   int& width,
                   int& height) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }

  std::string magic;
  input >> magic;
  if (magic != "P5") {
    return false;
  }

  std::string line;
  std::getline(input, line);

  auto next_token = [&input, &line]() -> std::string {
    while (std::getline(input, line)) {
      const auto comment = line.find('#');
      const auto cleaned = trim(line.substr(0, comment));
      if (!cleaned.empty()) {
        return cleaned;
      }
    }
    return {};
  };

  const auto dimensions = next_token();
  if (dimensions.empty()) {
    return false;
  }

  std::istringstream dimension_stream(dimensions);
  dimension_stream >> width >> height;
  if (!dimension_stream || width <= 0 || height <= 0) {
    return false;
  }

  const auto max_value_token = next_token();
  if (max_value_token.empty()) {
    return false;
  }

  int max_value = 0;
  std::istringstream max_value_stream(max_value_token);
  max_value_stream >> max_value;
  if (!max_value_stream || max_value <= 0 || max_value > 255) {
    return false;
  }

  pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
  input.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
  return input.good() || input.gcount() == static_cast<std::streamsize>(pixels.size());
}

std::array<uint8_t, 3> heatmap_color(float normalized) {
  normalized = std::clamp(normalized, 0.0f, 1.0f);

  // Piecewise interpolation across representative plasma colormap stops.
    static constexpr std::array<std::array<float, 4>, 10> kPlasmaStops{{
      {0.00f, 0.050f, 0.030f, 0.528f},
      {0.11f, 0.215f, 0.017f, 0.599f},
      {0.22f, 0.379f, 0.002f, 0.653f},
      {0.33f, 0.523f, 0.025f, 0.653f},
      {0.44f, 0.651f, 0.125f, 0.596f},
      {0.56f, 0.752f, 0.227f, 0.513f},
      {0.67f, 0.836f, 0.329f, 0.431f},
      {0.78f, 0.907f, 0.435f, 0.353f},
      {0.89f, 0.967f, 0.564f, 0.265f},
      {1.00f, 0.940f, 0.975f, 0.131f},
  }};

  for (size_t index = 1; index < kPlasmaStops.size(); ++index) {
    if (normalized <= kPlasmaStops[index][0]) {
      const auto& lower = kPlasmaStops[index - 1];
      const auto& upper = kPlasmaStops[index];
      const float span = std::max(0.0001f, upper[0] - lower[0]);
      const float t = (normalized - lower[0]) / span;
      return {
          static_cast<uint8_t>((lower[1] + (upper[1] - lower[1]) * t) * 255.0f),
          static_cast<uint8_t>((lower[2] + (upper[2] - lower[2]) * t) * 255.0f),
          static_cast<uint8_t>((lower[3] + (upper[3] - lower[3]) * t) * 255.0f),
      };
    }
  }

  const auto& last = kPlasmaStops.back();
  return {static_cast<uint8_t>(last[1] * 255.0f),
          static_cast<uint8_t>(last[2] * 255.0f),
          static_cast<uint8_t>(last[3] * 255.0f)};
}

std::vector<uint8_t> grayscale_to_rgb(const std::vector<uint8_t>& grayscale,
                                      float blue_limit,
                                      float red_limit) {
  std::vector<uint8_t> rgb(grayscale.size() * 3);
  const float blue = std::clamp(blue_limit, 0.0f, 1.0f);
  const float red = std::clamp(red_limit, blue + 0.01f, 1.0f);
  for (size_t index = 0; index < grayscale.size(); ++index) {
    const float normalized = static_cast<float>(grayscale[index]) / 255.0f;
    const float remapped = std::clamp((normalized - blue) / std::max(0.01f, red - blue), 0.0f, 1.0f);
    const auto color = heatmap_color(remapped);
    rgb[index * 3] = color[0];
    rgb[index * 3 + 1] = color[1];
    rgb[index * 3 + 2] = color[2];
  }
  return rgb;
}

struct RgbColor {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

constexpr int kHeaderHeight = 52;
constexpr int kFooterHeight = 40;
constexpr int kSidebarWidth = 260;
constexpr int kPsdHeight = 142;
constexpr int kPanelPadding = 28;
constexpr size_t kBytesPerMegabyte = 1024 * 1024;

std::atomic<bool>& global_overlay_enabled() {
  static std::atomic<bool> enabled{true};
  return enabled;
}

std::atomic<bool>& global_full_ui_enabled() {
  static std::atomic<bool> enabled{false};
  return enabled;
}

std::mutex& visualization_ui_state_mutex() {
  static std::mutex mutex;
  return mutex;
}

holoscan::ops::VisualizationUiState& visualization_ui_state_storage() {
  static holoscan::ops::VisualizationUiState state;
  return state;
}

holoscan::ops::VisualizationRect normalized_rect(int x,
                                                 int y,
                                                 int width,
                                                 int height,
                                                 int canvas_width,
                                                 int canvas_height) {
  const float safe_width = static_cast<float>(std::max(1, canvas_width));
  const float safe_height = static_cast<float>(std::max(1, canvas_height));
  return {static_cast<float>(x) / safe_width,
          static_cast<float>(y) / safe_height,
          static_cast<float>(width) / safe_width,
          static_cast<float>(height) / safe_height};
}

ImVec2 denormalize_point(float x, float y, const ImVec2& display_size) {
  return ImVec2(x * display_size.x, y * display_size.y);
}

ImVec2 denormalize_size(float width, float height, const ImVec2& display_size) {
  return ImVec2(width * display_size.x, height * display_size.y);
}

int clamp_history_rows_to_budget(int width, int requested_rows, int budget_mb) {
  const int safe_width = std::max(1, width);
  const int safe_rows = std::max(1, requested_rows);
  const size_t budget_bytes = static_cast<size_t>(std::max(1, budget_mb)) * kBytesPerMegabyte;
  const size_t bytes_per_row = static_cast<size_t>(safe_width) * (sizeof(uint8_t) + sizeof(uint8_t)) +
                               sizeof(int64_t) + sizeof(int);
  if (bytes_per_row == 0 || budget_bytes <= bytes_per_row) {
    return 1;
  }
  const size_t max_rows_by_budget = std::max<size_t>(1, budget_bytes / bytes_per_row);
  return std::min(safe_rows, static_cast<int>(std::min<size_t>(static_cast<size_t>(safe_rows), max_rows_by_budget)));
}

RgbColor mix(const RgbColor& a, const RgbColor& b, float t) {
  const float clamped = std::clamp(t, 0.0f, 1.0f);
  return {static_cast<uint8_t>(a.r + (b.r - a.r) * clamped),
          static_cast<uint8_t>(a.g + (b.g - a.g) * clamped),
          static_cast<uint8_t>(a.b + (b.b - a.b) * clamped)};
}

void set_pixel(std::vector<uint8_t>& canvas, int width, int height, int x, int y, const RgbColor& color) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  const size_t offset = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
  canvas[offset] = color.r;
  canvas[offset + 1] = color.g;
  canvas[offset + 2] = color.b;
}

void blend_pixel(std::vector<uint8_t>& canvas,
                 int width,
                 int height,
                 int x,
                 int y,
                 const RgbColor& color,
                 float alpha) {
  if (x < 0 || y < 0 || x >= width || y >= height) {
    return;
  }
  const float clamped = std::clamp(alpha, 0.0f, 1.0f);
  const size_t offset = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
  canvas[offset] = static_cast<uint8_t>(canvas[offset] * (1.0f - clamped) + color.r * clamped);
  canvas[offset + 1] = static_cast<uint8_t>(canvas[offset + 1] * (1.0f - clamped) + color.g * clamped);
  canvas[offset + 2] = static_cast<uint8_t>(canvas[offset + 2] * (1.0f - clamped) + color.b * clamped);
}

void fill_rect(std::vector<uint8_t>& canvas,
               int width,
               int height,
               int x,
               int y,
               int rect_width,
               int rect_height,
               const RgbColor& color) {
  for (int row = std::max(0, y); row < std::min(height, y + rect_height); ++row) {
    for (int col = std::max(0, x); col < std::min(width, x + rect_width); ++col) {
      set_pixel(canvas, width, height, col, row, color);
    }
  }
}

void draw_rect_outline(std::vector<uint8_t>& canvas,
                       int width,
                       int height,
                       int x,
                       int y,
                       int rect_width,
                       int rect_height,
                       const RgbColor& color,
                       int thickness = 1) {
  fill_rect(canvas, width, height, x, y, rect_width, thickness, color);
  fill_rect(canvas, width, height, x, y + rect_height - thickness, rect_width, thickness, color);
  fill_rect(canvas, width, height, x, y, thickness, rect_height, color);
  fill_rect(canvas, width, height, x + rect_width - thickness, y, thickness, rect_height, color);
}

void fill_vertical_gradient(std::vector<uint8_t>& canvas,
                            int width,
                            int height,
                            const RgbColor& top,
                            const RgbColor& bottom) {
  for (int row = 0; row < height; ++row) {
    const float t = height > 1 ? static_cast<float>(row) / static_cast<float>(height - 1) : 0.0f;
    const auto color = mix(top, bottom, t);
    fill_rect(canvas, width, height, 0, row, width, 1, color);
  }
}

std::array<uint8_t, 7> glyph_rows(char c) {
  switch (c) {
    case 'A': return {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'B': return {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case 'C': return {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    case 'D': return {0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C};
    case 'E': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    case 'F': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    case 'G': return {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F};
    case 'H': return {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'I': return {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E};
    case 'J': return {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C};
    case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    case 'M': return {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    case 'N': return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    case 'O': return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'P': return {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    case 'Q': return {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    case 'R': return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    case 'S': return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    case 'T': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A};
    case 'X': return {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    case 'Y': return {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    case 'Z': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
    case '0': return {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    case '1': return {0x04, 0x0C, 0x14, 0x04, 0x04, 0x04, 0x1F};
    case '2': return {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    case '3': return {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    case '4': return {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    case '5': return {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E};
    case '6': return {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    case '7': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    case '9': return {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E};
    case '.': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C};
    case ':': return {0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00};
    case '-': return {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    case '+': return {0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00};
    case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    default: return {0x1F, 0x11, 0x02, 0x04, 0x08, 0x00, 0x08};
  }
}

int text_pixel_width(const std::string& text, int scale) {
  auto advance_units = [](char raw_char) {
    const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw_char)));
    switch (c) {
      case 'I':
      case '1':
      case '.':
      case ':':
        return 4;
      case 'M':
      case 'W':
        return 8;
      case ' ':
        return 4;
      default:
        return 6;
    }
  };

  int width_units = 0;
  for (char c : text) {
    width_units += advance_units(c);
  }
  return width_units * scale;
}

void draw_line(std::vector<uint8_t>& canvas,
               int width,
               int height,
               int x0,
               int y0,
               int x1,
               int y1,
               const RgbColor& color,
               int thickness) {
  const int dx = std::abs(x1 - x0);
  const int sx = x0 < x1 ? 1 : -1;
  const int dy = -std::abs(y1 - y0);
  const int sy = y0 < y1 ? 1 : -1;
  int error = dx + dy;
  int x = x0;
  int y = y0;

  while (true) {
    fill_rect(canvas, width, height, x - thickness / 2, y - thickness / 2, thickness, thickness, color);
    if (x == x1 && y == y1) {
      break;
    }
    const int twice_error = 2 * error;
    if (twice_error >= dy) {
      error += dy;
      x += sx;
    }
    if (twice_error <= dx) {
      error += dx;
      y += sy;
    }
  }
}

int draw_stroke_char(std::vector<uint8_t>& canvas,
                     int width,
                     int height,
                     int x,
                     int y,
                     char raw_char,
                     const RgbColor& color,
                     int scale) {
  const char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw_char)));
  const int thickness = std::max(2, scale + (scale > 1 ? 0 : 1));
  auto line = [&](int x0, int y0, int x1, int y1) {
    draw_line(canvas,
              width,
              height,
              x + x0 * scale,
              y + y0 * scale,
              x + x1 * scale,
              y + y1 * scale,
              color,
              thickness);
  };

  switch (c) {
    case 'A': line(0, 6, 2, 0); line(2, 0, 4, 6); line(1, 3, 3, 3); return 6 * scale;
    case 'B': line(0, 0, 0, 6); line(0, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 2); line(4, 2, 3, 3); line(0, 3, 3, 3); line(3, 3, 4, 4); line(4, 4, 4, 5); line(4, 5, 3, 6); line(0, 6, 3, 6); return 6 * scale;
    case 'C': line(4, 1, 3, 0); line(3, 0, 1, 0); line(1, 0, 0, 1); line(0, 1, 0, 5); line(0, 5, 1, 6); line(1, 6, 3, 6); line(3, 6, 4, 5); return 6 * scale;
    case 'D': line(0, 0, 0, 6); line(0, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 5); line(4, 5, 3, 6); line(0, 6, 3, 6); return 6 * scale;
    case 'E': line(0, 0, 0, 6); line(0, 0, 4, 0); line(0, 3, 3, 3); line(0, 6, 4, 6); return 6 * scale;
    case 'F': line(0, 0, 0, 6); line(0, 0, 4, 0); line(0, 3, 3, 3); return 6 * scale;
    case 'G': line(4, 1, 3, 0); line(3, 0, 1, 0); line(1, 0, 0, 1); line(0, 1, 0, 5); line(0, 5, 1, 6); line(1, 6, 3, 6); line(3, 6, 4, 5); line(4, 5, 4, 4); line(4, 4, 2, 4); return 6 * scale;
    case 'H': line(0, 0, 0, 6); line(4, 0, 4, 6); line(0, 3, 4, 3); return 6 * scale;
    case 'I': line(0, 0, 2, 0); line(1, 0, 1, 6); line(0, 6, 2, 6); return 4 * scale;
    case 'J': line(4, 0, 4, 5); line(4, 5, 3, 6); line(3, 6, 1, 6); line(1, 6, 0, 5); return 6 * scale;
    case 'K': line(0, 0, 0, 6); line(4, 0, 0, 3); line(0, 3, 4, 6); return 6 * scale;
    case 'L': line(0, 0, 0, 6); line(0, 6, 4, 6); return 6 * scale;
    case 'M': line(0, 6, 0, 0); line(0, 0, 3, 3); line(3, 3, 6, 0); line(6, 0, 6, 6); return 8 * scale;
    case 'N': line(0, 6, 0, 0); line(0, 0, 4, 6); line(4, 6, 4, 0); return 6 * scale;
    case 'O': line(1, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 5); line(4, 5, 3, 6); line(3, 6, 1, 6); line(1, 6, 0, 5); line(0, 5, 0, 1); line(0, 1, 1, 0); return 6 * scale;
    case 'P': line(0, 6, 0, 0); line(0, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 2); line(4, 2, 3, 3); line(3, 3, 0, 3); return 6 * scale;
    case 'Q': line(1, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 5); line(4, 5, 3, 6); line(3, 6, 1, 6); line(1, 6, 0, 5); line(0, 5, 0, 1); line(0, 1, 1, 0); line(2, 4, 4, 6); return 6 * scale;
    case 'R': line(0, 6, 0, 0); line(0, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 2); line(4, 2, 3, 3); line(3, 3, 0, 3); line(0, 3, 4, 6); return 6 * scale;
    case 'S': line(4, 1, 3, 0); line(3, 0, 1, 0); line(1, 0, 0, 1); line(0, 1, 0, 2); line(0, 2, 1, 3); line(1, 3, 3, 3); line(3, 3, 4, 4); line(4, 4, 4, 5); line(4, 5, 3, 6); line(3, 6, 1, 6); line(1, 6, 0, 5); return 6 * scale;
    case 'T': line(0, 0, 4, 0); line(2, 0, 2, 6); return 6 * scale;
    case 'U': line(0, 0, 0, 5); line(0, 5, 1, 6); line(1, 6, 3, 6); line(3, 6, 4, 5); line(4, 5, 4, 0); return 6 * scale;
    case 'V': line(0, 0, 2, 6); line(2, 6, 4, 0); return 6 * scale;
    case 'W': line(0, 0, 1, 6); line(1, 6, 3, 2); line(3, 2, 5, 6); line(5, 6, 6, 0); return 8 * scale;
    case 'X': line(0, 0, 4, 6); line(4, 0, 0, 6); return 6 * scale;
    case 'Y': line(0, 0, 2, 3); line(4, 0, 2, 3); line(2, 3, 2, 6); return 6 * scale;
    case 'Z': line(0, 0, 4, 0); line(4, 0, 0, 6); line(0, 6, 4, 6); return 6 * scale;
    case '0': line(1, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 5); line(4, 5, 3, 6); line(3, 6, 1, 6); line(1, 6, 0, 5); line(0, 5, 0, 1); line(0, 1, 1, 0); line(0, 6, 4, 0); return 6 * scale;
    case '1': line(1, 1, 2, 0); line(2, 0, 2, 6); line(1, 6, 3, 6); return 4 * scale;
    case '2': line(0, 1, 1, 0); line(1, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 2); line(4, 2, 0, 6); line(0, 6, 4, 6); return 6 * scale;
    case '3': line(0, 0, 4, 0); line(4, 0, 2, 3); line(2, 3, 4, 6); line(0, 6, 4, 6); return 6 * scale;
    case '4': line(3, 0, 3, 6); line(0, 4, 4, 4); line(0, 4, 3, 0); return 6 * scale;
    case '5': line(4, 0, 0, 0); line(0, 0, 0, 3); line(0, 3, 3, 3); line(3, 3, 4, 4); line(4, 4, 4, 5); line(4, 5, 3, 6); line(3, 6, 0, 6); return 6 * scale;
    case '6': line(4, 1, 3, 0); line(3, 0, 1, 0); line(1, 0, 0, 2); line(0, 2, 0, 5); line(0, 5, 1, 6); line(1, 6, 3, 6); line(3, 6, 4, 5); line(4, 5, 4, 4); line(4, 4, 3, 3); line(3, 3, 0, 3); return 6 * scale;
    case '7': line(0, 0, 4, 0); line(4, 0, 1, 6); return 6 * scale;
    case '8': line(1, 0, 3, 0); line(3, 0, 4, 1); line(4, 1, 4, 2); line(4, 2, 3, 3); line(3, 3, 1, 3); line(1, 3, 0, 2); line(0, 2, 0, 1); line(0, 1, 1, 0); line(1, 3, 3, 3); line(3, 3, 4, 4); line(4, 4, 4, 5); line(4, 5, 3, 6); line(3, 6, 1, 6); line(1, 6, 0, 5); line(0, 5, 0, 4); line(0, 4, 1, 3); return 6 * scale;
    case '9': line(4, 4, 3, 6); line(3, 6, 1, 6); line(1, 6, 0, 5); line(0, 5, 0, 4); line(0, 4, 1, 3); line(1, 3, 4, 3); line(4, 3, 4, 1); line(4, 1, 3, 0); line(3, 0, 1, 0); return 6 * scale;
    case '.': fill_rect(canvas, width, height, x + 2 * scale, y + 6 * scale, thickness, thickness, color); return 4 * scale;
    case ':': fill_rect(canvas, width, height, x + 2 * scale, y + 2 * scale, thickness, thickness, color); fill_rect(canvas, width, height, x + 2 * scale, y + 5 * scale, thickness, thickness, color); return 4 * scale;
    case '-': line(0, 3, 4, 3); return 6 * scale;
    case '+': line(2, 1, 2, 5); line(0, 3, 4, 3); return 6 * scale;
    case ' ': return 4 * scale;
    default: line(0, 0, 4, 0); line(4, 0, 0, 6); line(0, 6, 4, 6); return 6 * scale;
  }
}

void draw_text(std::vector<uint8_t>& canvas,
               int width,
               int height,
               int x,
               int y,
               const std::string& text,
               const RgbColor& color,
               int scale) {
  int cursor_x = x;
  for (char raw_char : text) {
    cursor_x += draw_stroke_char(canvas, width, height, cursor_x, y, raw_char, color, scale);
  }
}

void blit_rgb_nearest(std::vector<uint8_t>& canvas,
                      int canvas_width,
                      int canvas_height,
                      int dst_x,
                      int dst_y,
                      int dst_width,
                      int dst_height,
                      const std::vector<uint8_t>& rgb,
                      int src_width,
                      int src_height) {
  for (int row = 0; row < dst_height; ++row) {
    const int src_row = std::min(src_height - 1, (row * src_height) / std::max(1, dst_height));
    for (int col = 0; col < dst_width; ++col) {
      const int src_col = std::min(src_width - 1, (col * src_width) / std::max(1, dst_width));
      const size_t src_offset = (static_cast<size_t>(src_row) * static_cast<size_t>(src_width) + static_cast<size_t>(src_col)) * 3;
      set_pixel(canvas,
                canvas_width,
                canvas_height,
                dst_x + col,
                dst_y + row,
                {rgb[src_offset], rgb[src_offset + 1], rgb[src_offset + 2]});
    }
  }
}

std::vector<uint8_t> scale_rgb_to_fit(const std::vector<uint8_t>& rgb,
                                      int src_width,
                                      int src_height,
                                      int max_width,
                                      int max_height,
                                      int& dst_width,
                                      int& dst_height) {
  dst_width = std::max(1, src_width);
  dst_height = std::max(1, src_height);
  const int clamped_max_width = std::max(1, max_width);
  const int clamped_max_height = std::max(1, max_height);
  if (dst_width <= clamped_max_width && dst_height <= clamped_max_height) {
    return rgb;
  }

  const double width_scale = static_cast<double>(clamped_max_width) / static_cast<double>(dst_width);
  const double height_scale = static_cast<double>(clamped_max_height) / static_cast<double>(dst_height);
  const double scale = std::min(width_scale, height_scale);
  dst_width = std::max(1, static_cast<int>(std::floor(static_cast<double>(dst_width) * scale)));
  dst_height = std::max(1, static_cast<int>(std::floor(static_cast<double>(dst_height) * scale)));

  std::vector<uint8_t> scaled(static_cast<size_t>(dst_width) * static_cast<size_t>(dst_height) * 3, 0);
  blit_rgb_nearest(scaled, dst_width, dst_height, 0, 0, dst_width, dst_height, rgb, src_width, src_height);
  return scaled;
}

struct ChannelPanelHeights {
  int psd = 64;
  int heat = 64;
  int mask = 40;
};

ChannelPanelHeights compute_channel_panel_heights(int canvas_height, int active_channels) {
  const int clamped_channels = std::max(1, active_channels);
  const int rows = std::max(1, (clamped_channels + 1) / 2);
  const int content_height = std::max(160, canvas_height - kHeaderHeight - kFooterHeight - kPanelPadding * 2);
  const int panel_height_budget = std::max(180,
                                           (content_height - (rows - 1) * kPanelPadding) / std::max(1, rows));
  const int channel_header_band = 28;
  const int section_gap = std::max(10, kPanelPadding / 2);
  const int mask_section_gap = section_gap + 18;
  const int confidence_gap = 0;
  const int confidence_height = 0;
  const int plot_stack_height = std::max(140,
      panel_height_budget - channel_header_band - confidence_gap - confidence_height -
          section_gap - mask_section_gap);

  ChannelPanelHeights heights{};
  heights.psd = std::clamp(static_cast<int>(std::lround(plot_stack_height * 0.22)), 64, 112);
  heights.mask = std::clamp(static_cast<int>(std::lround(plot_stack_height * 0.18)), 40, 88);
  heights.heat = plot_stack_height - heights.psd - heights.mask;
  if (heights.heat < 64) {
    int deficit = 64 - heights.heat;
    const int psd_reduction = std::min(deficit / 2 + deficit % 2, std::max(0, heights.psd - 64));
    heights.psd -= psd_reduction;
    deficit -= psd_reduction;
    const int mask_reduction = std::min(deficit, std::max(0, heights.mask - 40));
    heights.mask -= mask_reduction;
    heights.heat = plot_stack_height - heights.psd - heights.mask;
  }
  return heights;
}

uint8_t sample_ring_canvas_value(const std::vector<uint8_t>& ring,
                                 int src_width,
                                 int capacity_rows,
                                 int leading_blank_rows,
                                 int oldest_row,
                                 int src_canvas_row,
                                 int src_col) {
  if (src_canvas_row < 0 || src_canvas_row >= capacity_rows || src_canvas_row < leading_blank_rows) {
    return 0;
  }
  const int logical_row = src_canvas_row - leading_blank_rows;
  const int ring_row = (oldest_row + logical_row) % capacity_rows;
  return ring[static_cast<size_t>(ring_row) * static_cast<size_t>(src_width) + static_cast<size_t>(src_col)];
}

uint8_t max_ring_canvas_value(const std::vector<uint8_t>& ring,
                              int src_width,
                              int capacity_rows,
                              int leading_blank_rows,
                              int oldest_row,
                              int row_begin,
                              int row_end,
                              int src_col) {
  uint8_t max_value = 0;
  for (int src_canvas_row = std::max(0, row_begin); src_canvas_row < std::min(capacity_rows, row_end); ++src_canvas_row) {
    max_value = std::max(max_value,
                         sample_ring_canvas_value(ring,
                                                  src_width,
                                                  capacity_rows,
                                                  leading_blank_rows,
                                                  oldest_row,
                                                  src_canvas_row,
                                                  src_col));
  }
  return max_value;
}

RgbColor mask_overlay_color(float normalized_value) {
  const float t = std::clamp(std::sqrt(std::max(0.0f, normalized_value)), 0.0f, 1.0f);
  const auto blend_channel = [t](uint8_t low, uint8_t high) {
    return static_cast<uint8_t>(std::lround((1.0f - t) * static_cast<float>(low) +
                                            t * static_cast<float>(high)));
  };
  return {
      blend_channel(72, 255),
      blend_channel(208, 246),
      blend_channel(255, 168),
  };
}

void overlay_mask(std::vector<uint8_t>& canvas,
                  int canvas_width,
                  int canvas_height,
                  int dst_x,
                  int dst_y,
                  int dst_width,
                  int dst_height,
                  const holoscan::ops::OfflinePgmFrame& mask_frame,
                  float overlay_alpha) {
  for (int row = 0; row < dst_height; ++row) {
    const int src_row = std::min(mask_frame.height - 1, (row * mask_frame.height) / std::max(1, dst_height));
    for (int col = 0; col < dst_width; ++col) {
      const int src_col = std::min(mask_frame.width - 1, (col * mask_frame.width) / std::max(1, dst_width));
      const auto value = mask_frame.pixels[static_cast<size_t>(src_row) * static_cast<size_t>(mask_frame.width) + static_cast<size_t>(src_col)];
      if (value == 0) {
        continue;
      }
      const float normalized_value = static_cast<float>(value) / 255.0f;
      const float scaled_alpha = overlay_alpha * (0.18f + 0.82f * std::sqrt(normalized_value));
      blend_pixel(canvas,
                  canvas_width,
                  canvas_height,
                  dst_x + col,
                  dst_y + row,
                  mask_overlay_color(normalized_value),
                  scaled_alpha);
    }
  }
}

void blit_grayscale_ring_to_canvas(std::vector<uint8_t>& canvas,
                                   int canvas_width,
                                   int canvas_height,
                                   int dst_x,
                                   int dst_y,
                                   int dst_width,
                                   int dst_height,
                                   const std::vector<uint8_t>& ring,
                                   int src_width,
                                   int capacity_rows,
                                   int valid_rows,
                                   int write_row,
                                   float blue_limit,
                                   float red_limit) {
  if (ring.empty() || src_width <= 0 || capacity_rows <= 0) {
    return;
  }
  const int oldest_row = valid_rows == capacity_rows ? write_row : 0;
  const int leading_blank_rows = capacity_rows - valid_rows;
  const float blue = std::clamp(blue_limit, 0.0f, 1.0f);
  const float red = std::clamp(red_limit, blue + 0.01f, 1.0f);
  for (int row = 0; row < dst_height; ++row) {
    const float src_row = ((static_cast<float>(row) + 0.5f) * static_cast<float>(capacity_rows)) /
                              static_cast<float>(std::max(1, dst_height)) -
                          0.5f;
    const int src_row0 = std::clamp(static_cast<int>(std::floor(src_row)), 0, capacity_rows - 1);
    const int src_row1 = std::min(capacity_rows - 1, src_row0 + 1);
    const float row_mix = std::clamp(src_row - static_cast<float>(src_row0), 0.0f, 1.0f);
    for (int col = 0; col < dst_width; ++col) {
      const int src_col = std::min(src_width - 1, (col * src_width) / std::max(1, dst_width));
      const float gray0 = static_cast<float>(sample_ring_canvas_value(ring,
                                                                      src_width,
                                                                      capacity_rows,
                                                                      leading_blank_rows,
                                                                      oldest_row,
                                                                      src_row0,
                                                                      src_col));
      const float gray1 = static_cast<float>(sample_ring_canvas_value(ring,
                                                                      src_width,
                                                                      capacity_rows,
                                                                      leading_blank_rows,
                                                                      oldest_row,
                                                                      src_row1,
                                                                      src_col));
      const float gray = gray0 + (gray1 - gray0) * row_mix;
      const float normalized = gray / 255.0f;
      const float remapped = std::clamp((normalized - blue) / std::max(0.01f, red - blue), 0.0f, 1.0f);
      const auto color = heatmap_color(remapped);
      set_pixel(canvas, canvas_width, canvas_height, dst_x + col, dst_y + row, {color[0], color[1], color[2]});
    }
  }
}

void overlay_mask_ring(std::vector<uint8_t>& canvas,
                       int canvas_width,
                       int canvas_height,
                       int dst_x,
                       int dst_y,
                       int dst_width,
                       int dst_height,
                       const std::vector<uint8_t>& ring,
                       int src_width,
                       int capacity_rows,
                       int valid_rows,
                       int write_row,
                       float overlay_alpha) {
  if (ring.empty() || src_width <= 0 || capacity_rows <= 0) {
    return;
  }
  const int oldest_row = valid_rows == capacity_rows ? write_row : 0;
  const int leading_blank_rows = capacity_rows - valid_rows;
  for (int row = 0; row < dst_height; ++row) {
    const int row_begin = (row * capacity_rows) / std::max(1, dst_height);
    const int row_end = std::min(capacity_rows,
                                 std::max(row_begin + 1,
                                          ((row + 1) * capacity_rows + std::max(1, dst_height) - 1) /
                                              std::max(1, dst_height)));
    for (int col = 0; col < dst_width; ++col) {
      const int src_col = std::min(src_width - 1, (col * src_width) / std::max(1, dst_width));
      const uint8_t value = max_ring_canvas_value(ring,
                                                  src_width,
                                                  capacity_rows,
                                                  leading_blank_rows,
                                                  oldest_row,
                                                  row_begin,
                                                  row_end,
                                                  src_col);
      if (value > 0) {
        const float normalized_value = static_cast<float>(value) / 255.0f;
        const float boosted_visibility = std::pow(normalized_value, 0.65f);
        const float scaled_alpha = overlay_alpha * (0.16f + 0.84f * boosted_visibility);
        blend_pixel(canvas,
                    canvas_width,
                    canvas_height,
                    dst_x + col,
                    dst_y + row,
                    mask_overlay_color(normalized_value),
                    scaled_alpha);
      }
    }
  }
}

void blit_mask_ring_to_canvas(std::vector<uint8_t>& canvas,
                              int canvas_width,
                              int canvas_height,
                              int dst_x,
                              int dst_y,
                              int dst_width,
                              int dst_height,
                              const std::vector<uint8_t>& ring,
                              int src_width,
                              int capacity_rows,
                              int valid_rows,
                              int write_row) {
  if (ring.empty() || src_width <= 0 || capacity_rows <= 0) {
    return;
  }
  const int oldest_row = valid_rows == capacity_rows ? write_row : 0;
  const int leading_blank_rows = capacity_rows - valid_rows;
  for (int row = 0; row < dst_height; ++row) {
    const int row_begin = (row * capacity_rows) / std::max(1, dst_height);
    const int row_end = std::min(capacity_rows,
                                 std::max(row_begin + 1,
                                          ((row + 1) * capacity_rows + std::max(1, dst_height) - 1) /
                                              std::max(1, dst_height)));
    for (int col = 0; col < dst_width; ++col) {
      const int src_col = std::min(src_width - 1, (col * src_width) / std::max(1, dst_width));
      const uint8_t value = max_ring_canvas_value(ring,
                                                  src_width,
                                                  capacity_rows,
                                                  leading_blank_rows,
                                                  oldest_row,
                                                  row_begin,
                                                  row_end,
                                                  src_col);
      if (value == 0) {
        continue;
      }
      const float normalized_value = static_cast<float>(value) / 255.0f;
      const float boosted = std::pow(normalized_value, 0.35f);
      const uint8_t gray = static_cast<uint8_t>(std::clamp(boosted, 0.0f, 1.0f) * 255.0f);
      set_pixel(canvas, canvas_width, canvas_height, dst_x + col, dst_y + row, {gray, gray, gray});
    }
  }
}

void blit_mask_frame_to_canvas(std::vector<uint8_t>& canvas,
                               int canvas_width,
                               int canvas_height,
                               int dst_x,
                               int dst_y,
                               int dst_width,
                               int dst_height,
                               const holoscan::ops::OfflinePgmFrame& mask_frame) {
  if (mask_frame.pixels.empty() || mask_frame.width <= 0 || mask_frame.height <= 0) {
    return;
  }
  for (int row = 0; row < dst_height; ++row) {
    const int src_row = std::min(mask_frame.height - 1, (row * mask_frame.height) / std::max(1, dst_height));
    for (int col = 0; col < dst_width; ++col) {
      const int src_col = std::min(mask_frame.width - 1, (col * mask_frame.width) / std::max(1, dst_width));
      const uint8_t value = mask_frame.pixels[static_cast<size_t>(src_row) * static_cast<size_t>(mask_frame.width) +
                                             static_cast<size_t>(src_col)];
      if (value == 0) {
        continue;
      }
      const float normalized_value = static_cast<float>(value) / 255.0f;
      const float boosted = std::pow(normalized_value, 0.35f);
      const uint8_t gray = static_cast<uint8_t>(std::clamp(boosted, 0.0f, 1.0f) * 255.0f);
      set_pixel(canvas, canvas_width, canvas_height, dst_x + col, dst_y + row, {gray, gray, gray});
    }
  }
}

void draw_grid(std::vector<uint8_t>& canvas,
               int width,
               int height,
               int x,
               int y,
               int rect_width,
               int rect_height) {
  const RgbColor grid_color{90, 104, 128};
  for (int step = 1; step < 8; ++step) {
    const int gx = x + static_cast<int>(std::lround((static_cast<double>(rect_width - 1) * step) / 8.0));
    const int gy = y + static_cast<int>(std::lround((static_cast<double>(rect_height - 1) * step) / 8.0));
    fill_rect(canvas, width, height, gx, y, 1, rect_height, grid_color);
    fill_rect(canvas, width, height, x, gy, rect_width, 1, grid_color);
  }
}

void draw_metric_bar(std::vector<uint8_t>& canvas,
                     int width,
                     int height,
                     int x,
                     int y,
                     int bar_width,
                     float value,
                     const RgbColor& fill_color) {
  fill_rect(canvas, width, height, x, y, bar_width, 10, {44, 51, 71});
  fill_rect(canvas,
            width,
            height,
            x,
            y,
            std::max(1, static_cast<int>(std::round(bar_width * std::clamp(value, 0.0f, 1.0f)))),
            10,
            fill_color);
}

double safe_ratio(double numerator, double denominator) {
  return denominator > 0.0 ? numerator / denominator : 0.0;
}

std::string format_frequency_label(double frequency_hz) {
  if (!std::isfinite(frequency_hz)) {
    return "FREQ";
  }

  struct UnitScale {
    double divisor;
    const char* suffix;
    int precision;
  };
  static constexpr std::array<UnitScale, 4> kUnits{{
      {1.0e9, "GHZ", 3},
      {1.0e6, "MHZ", 3},
      {1.0e3, "KHZ", 1},
      {1.0, "HZ", 0},
  }};

  const double magnitude = std::abs(frequency_hz);
  const UnitScale* unit = &kUnits.back();
  for (const auto& candidate : kUnits) {
    if (magnitude >= candidate.divisor) {
      unit = &candidate;
      break;
    }
  }

  std::ostringstream os;
  os << std::fixed << std::setprecision(unit->precision) << (frequency_hz / unit->divisor) << ' ' << unit->suffix;
  return os.str();
}

std::string format_time_bin_label(double seconds_per_bin) {
  if (!std::isfinite(seconds_per_bin) || seconds_per_bin <= 0.0) {
    return "N/A";
  }

  std::ostringstream os;
  if (seconds_per_bin >= 1.0) {
    os << std::fixed << std::setprecision(3) << seconds_per_bin << " S";
  } else if (seconds_per_bin >= 1.0e-3) {
    os << std::fixed << std::setprecision(3) << (seconds_per_bin * 1.0e3) << " MS";
  } else if (seconds_per_bin >= 1.0e-6) {
    os << std::fixed << std::setprecision(2) << (seconds_per_bin * 1.0e6) << " US";
  } else {
    os << std::fixed << std::setprecision(2) << (seconds_per_bin * 1.0e9) << " NS";
  }
  return os.str();
}

std::string format_displayed_frame_ratio_label(double displayed_frame_ratio, int displayed_frame_stride) {
  if (!std::isfinite(displayed_frame_ratio) || displayed_frame_ratio <= 0.0) {
    return "N/A";
  }

  std::ostringstream os;
  const int safe_stride = std::max(1, displayed_frame_stride);
  os << "1:" << safe_stride << " (" << std::fixed << std::setprecision(1)
     << (displayed_frame_ratio * 100.0) << "%)";
  return os.str();
}

std::string format_fft_row_visualization_ratio_label(int displayed_rows, int source_rows) {
  if (displayed_rows <= 0 || source_rows <= 0) {
    return "N/A";
  }

  std::ostringstream os;
  os << displayed_rows << "/" << source_rows << " ("
     << std::fixed << std::setprecision(1)
     << (static_cast<double>(displayed_rows) * 100.0 / static_cast<double>(source_rows)) << "%)";
  return os.str();
}

double compute_display_frequency_bin_hz(double span_hz, int display_width_bins) {
  if (!std::isfinite(span_hz) || span_hz <= 0.0) {
    return 0.0;
  }
  return span_hz / static_cast<double>(std::max(1, display_width_bins));
}

double compute_display_time_bin_seconds(double span_hz, int fft_size, int rows_per_frame) {
  if (!std::isfinite(span_hz) || span_hz <= 0.0) {
    return 0.0;
  }
  const double fft_window_seconds = static_cast<double>(std::max(1, fft_size)) / span_hz;
  return fft_window_seconds / static_cast<double>(std::max(1, rows_per_frame));
}

double compute_displayed_frame_ratio(int render_every_n_frames) {
  return 1.0 / static_cast<double>(std::max(1, render_every_n_frames));
}

double resolved_display_span_hz(const holoscan::ops::ChannelVisualizationState& channel, int displayed_width) {
  if (std::isfinite(channel.info.span_hz) && channel.info.span_hz > 0.0) {
    return channel.info.span_hz;
  }
  if (std::isfinite(channel.info.resolution_hz) && channel.info.resolution_hz > 0.0) {
    return channel.info.resolution_hz * static_cast<double>(std::max(1, displayed_width));
  }
  return 0.0;
}

std::pair<int64_t, int64_t> history_frame_bounds(const holoscan::ops::ChannelVisualizationState& channel) {
  if (channel.history_valid_rows <= 0 || channel.history_capacity_rows <= 0 ||
      channel.history_row_frame_numbers.empty()) {
    return {-1, -1};
  }

  const int oldest_row = channel.history_valid_rows == channel.history_capacity_rows ? channel.history_write_row : 0;
  const int newest_row = (channel.history_write_row + channel.history_capacity_rows - 1) %
                         std::max(1, channel.history_capacity_rows);
  return {channel.history_row_frame_numbers[static_cast<size_t>(oldest_row)],
          channel.history_row_frame_numbers[static_cast<size_t>(newest_row)]};
}

std::string format_history_label(int64_t frame_number, const char* fallback_prefix) {
  if (frame_number < 0) {
    return fallback_prefix;
  }

  std::ostringstream os;
  os << fallback_prefix << ' ' << frame_number;
  return os.str();
}

void draw_toggle_button(std::vector<uint8_t>& canvas,
                        int width,
                        int height,
                        int x,
                        int y,
                        int button_width,
                        const std::string& label,
                        bool enabled,
                        const RgbColor& accent) {
  fill_rect(canvas, width, height, x, y, button_width, 30, {10, 21, 34});
  draw_rect_outline(canvas,
                    width,
                    height,
                    x,
                    y,
                    button_width,
                    30,
                    enabled ? accent : RgbColor{62, 74, 92},
                    1);
  draw_text(canvas, width, height, x + 8, y + 6, label, {200, 212, 224}, 1);

  const int pill_width = 42;
  const int pill_x = x + button_width - pill_width - 8;
  fill_rect(canvas,
            width,
            height,
            pill_x,
            y + 6,
            pill_width,
            18,
            enabled ? accent : RgbColor{34, 42, 54});
  draw_rect_outline(canvas,
                    width,
                    height,
                    pill_x,
                    y + 6,
                    pill_width,
                    18,
                    enabled ? RgbColor{232, 236, 241} : RgbColor{90, 104, 128},
                    1);
  draw_text(canvas,
            width,
            height,
            pill_x + 9,
            y + 11,
            enabled ? "ON" : "OFF",
            enabled ? RgbColor{8, 13, 20} : RgbColor{164, 179, 196},
            1);
}

void draw_trace_plot(std::vector<uint8_t>& canvas,
                     int width,
                     int height,
                     int x,
                     int y,
                     int plot_width,
                     int plot_height,
                     const std::vector<float>& trace,
                     const RgbColor& color,
                     int thickness) {
  if (trace.empty()) {
    return;
  }
  int prev_x = x;
  int prev_y = y + plot_height - 1 - static_cast<int>(std::round(std::clamp(trace.front(), 0.0f, 1.0f) * (plot_height - 1)));
  for (size_t i = 1; i < trace.size(); ++i) {
    const int curr_x = x + static_cast<int>((static_cast<double>(i) / std::max<size_t>(1, trace.size() - 1)) * (plot_width - 1));
    const int curr_y = y + plot_height - 1 - static_cast<int>(std::round(std::clamp(trace[i], 0.0f, 1.0f) * (plot_height - 1)));
    const int steps = std::max(std::abs(curr_x - prev_x), std::abs(curr_y - prev_y));
    for (int step = 0; step <= steps; ++step) {
      const float t = steps > 0 ? static_cast<float>(step) / static_cast<float>(steps) : 0.0f;
      const int draw_x = static_cast<int>(std::round(prev_x + (curr_x - prev_x) * t));
      const int draw_y = static_cast<int>(std::round(prev_y + (curr_y - prev_y) * t));
      fill_rect(canvas, width, height, draw_x - thickness / 2, draw_y - thickness / 2, thickness, thickness, color);
    }
    prev_x = curr_x;
    prev_y = curr_y;
  }
}

void draw_plot_axes(std::vector<uint8_t>& canvas,
                    int width,
                    int height,
                    int x,
                    int y,
                    int plot_width,
                    int plot_height,
                    const std::string& x0,
                    const std::string& x1,
                    const std::string& y0,
                    const std::string& y1,
                    int label_scale = 1,
                    const std::string& x_axis_title = {},
                    const std::string& y_axis_title = {}) {
  const RgbColor axis_color{142, 156, 174};
  draw_rect_outline(canvas, width, height, x, y, plot_width, plot_height, axis_color, 1);
  for (int step = 1; step < 5; ++step) {
    const int gy = y + static_cast<int>(std::lround((static_cast<double>(plot_height - 1) * step) / 5.0));
    fill_rect(canvas, width, height, x, gy, plot_width, 1, {48, 58, 76});
  }
  for (int step = 1; step < 8; ++step) {
    const int gx = x + static_cast<int>(std::lround((static_cast<double>(plot_width - 1) * step) / 8.0));
    fill_rect(canvas, width, height, gx, y, 1, plot_height, {40, 48, 64});
  }
  if (label_scale <= 0) {
    return;
  }
  const int label_y = y + plot_height + 8;
  draw_text(canvas, width, height, x, label_y, x0, {164, 179, 196}, label_scale);
  draw_text(canvas,
            width,
            height,
            x + plot_width - text_pixel_width(x1, label_scale),
            label_y,
            x1,
            {164, 179, 196},
            label_scale);
  draw_text(canvas, width, height, x - 2, y - (8 + 7 * label_scale), y1, {164, 179, 196}, label_scale);
  draw_text(canvas,
            width,
            height,
            x - 2,
            y + plot_height - std::max(0, 7 * label_scale - 2),
            y0,
            {164, 179, 196},
            label_scale);
  if (!x_axis_title.empty()) {
    const int title_x = x + (plot_width - text_pixel_width(x_axis_title, 1)) / 2;
    draw_text(canvas, width, height, title_x, label_y + 16, x_axis_title, {122, 143, 168}, 1);
  }
  if (!y_axis_title.empty()) {
    draw_text(canvas, width, height, x, y - 20, y_axis_title, {122, 143, 168}, 1);
  }
}

void draw_vertical_slider(std::vector<uint8_t>& canvas,
                          int width,
                          int height,
                          int x,
                          int y,
                          int slider_height,
                          float value,
                          const RgbColor& fill,
                          const std::string& label) {
  fill_rect(canvas, width, height, x + 8, y, 6, slider_height, {38, 45, 58});
  const int knob_y = y + slider_height - 1 - static_cast<int>(std::round(std::clamp(value, 0.0f, 1.0f) * (slider_height - 1)));
  fill_rect(canvas, width, height, x, knob_y - 4, 22, 8, fill);
  draw_rect_outline(canvas, width, height, x, knob_y - 4, 22, 8, {230, 238, 248}, 1);
  draw_text(canvas, width, height, x - 2, y + slider_height + 10, label, {164, 179, 196}, 1);
}

std::vector<uint8_t> reduce_spectrogram_to_grayscale(const SpectrogramTensor& tensor,
                                                     cudaStream_t stream,
                                                     int output_height,
                                                     int output_width) {
  const int src_rows = static_cast<int>(tensor.Size(0));
  const int src_cols = static_cast<int>(tensor.Size(1));
  const int dst_rows = std::max(1, std::min(output_height, src_rows));
  const int dst_cols = std::max(1, std::min(output_width, src_cols));

  std::vector<SpectrogramComplex> host_fft(static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols));
  const size_t bytes = host_fft.size() * sizeof(SpectrogramComplex);

  // auto copy_result = cudaMemcpyAsync(host_fft.data(), tensor.Data(), bytes, cudaMemcpyDeviceToHost, stream);
  // if (copy_result != cudaSuccess) {
  //   throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(copy_result));
  // }

  // auto sync_result = cudaStreamSynchronize(stream);
  // if (sync_result != cudaSuccess) {
  //   throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(sync_result));
  // }

  // Use vis_stream (passed in) which is separate from the pipeline stream.
  // This ensures we never block the pipeline's CUDA stream.
  auto copy_result = cudaMemcpyAsync(host_fft.data(), tensor.Data(), bytes, cudaMemcpyDeviceToHost, stream);
  if (copy_result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(copy_result));
  }

  // Sync only the vis_stream, not the pipeline stream
  auto sync_result = cudaStreamSynchronize(stream);
  if (sync_result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(sync_result));
  }

  std::vector<float> reduced(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), -120.0f);

  for (int row = 0; row < dst_rows; ++row) {
    const int row_start = (row * src_rows) / dst_rows;
    const int row_end = ((row + 1) * src_rows) / dst_rows;
    for (int col = 0; col < dst_cols; ++col) {
      const int col_start = (col * src_cols) / dst_cols;
      const int col_end = ((col + 1) * src_cols) / dst_cols;
      double accumulation = 0.0;
      int count = 0;
      for (int src_row = row_start; src_row < std::max(row_start + 1, row_end); ++src_row) {
        for (int src_col = col_start; src_col < std::max(col_start + 1, col_end); ++src_col) {
          const auto& value = host_fft[static_cast<size_t>(src_row) * static_cast<size_t>(src_cols) + static_cast<size_t>(src_col)];
          const float real = value.real();
          const float imag = value.imag();
          accumulation += 10.0 * std::log10(real * real + imag * imag + 1e-12f);
          ++count;
        }
      }
      reduced[static_cast<size_t>(row) * static_cast<size_t>(dst_cols) + static_cast<size_t>(col)] =
          static_cast<float>(accumulation / static_cast<double>(std::max(1, count)));
    }
  }

  // Fixed dB scale — Sage: disable auto-scaling, use fixed dBm limits
  // Typical noise floor ~-100 dBm, signals visible above -80 dBm
  // Tune these values based on your RF environment
  const float db_range = std::max(1.0f, 0.0f - (-100.0f));  // uses defaults, not called in live path
  std::vector<uint8_t> grayscale(reduced.size());
  for (size_t index = 0; index < reduced.size(); ++index) {
    const float normalized = (reduced[index] - (-100.0f)) / db_range;
    grayscale[index] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
  }
  return grayscale;

}

struct DeviceBufferLease {
  void* ptr = nullptr;
  size_t bytes = 0;
};

std::mutex& output_buffer_pool_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<size_t, std::vector<void*>>& output_buffer_pool() {
  static std::unordered_map<size_t, std::vector<void*>> pool;
  return pool;
}

void* acquire_output_buffer(size_t bytes) {
  std::lock_guard<std::mutex> lock(output_buffer_pool_mutex());
  auto& pool = output_buffer_pool();
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

void recycle_output_buffer(void* ptr, size_t bytes) {
  if (ptr == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(output_buffer_pool_mutex());
  output_buffer_pool()[bytes].push_back(ptr);
}

__global__ void reduce_complex_to_grayscale_kernel(const SpectrogramComplex* input,
                                                   int src_rows,
                                                   int src_cols,
                                                   bool transpose_input,
                                                   uint8_t* output,
                                                   int dst_rows,
                                                   int dst_cols,
                                                   float db_floor,
                                                   float db_ceil) {
  const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
  const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
  if (out_col >= dst_cols || out_row >= dst_rows) {
    return;
  }

  const int canonical_rows = transpose_input ? src_cols : src_rows;
  const int canonical_cols = transpose_input ? src_rows : src_cols;
  const int row_start = (out_row * canonical_rows) / dst_rows;
  const int row_end = max(row_start + 1, ((out_row + 1) * canonical_rows) / dst_rows);
  const int col_start = (out_col * canonical_cols) / dst_cols;
  const int col_end = max(col_start + 1, ((out_col + 1) * canonical_cols) / dst_cols);

  double accumulation = 0.0;
  int count = 0;
  for (int canonical_row = row_start; canonical_row < row_end; ++canonical_row) {
    for (int canonical_col = col_start; canonical_col < col_end; ++canonical_col) {
      const int src_row = transpose_input ? canonical_col : canonical_row;
      const int src_col = transpose_input ? canonical_row : canonical_col;
      const auto value = input[static_cast<size_t>(src_row) * static_cast<size_t>(src_cols) + static_cast<size_t>(src_col)];
      const float real = value.real();
      const float imag = value.imag();
      accumulation += 10.0 * log10(real * real + imag * imag + 1e-12f);
      ++count;
    }
  }

  const float reduced = static_cast<float>(accumulation / static_cast<double>(max(1, count)));
  const float db_range = max(1.0f, db_ceil - db_floor);
  const float normalized = (reduced - db_floor) / db_range;
  output[static_cast<size_t>(out_row) * static_cast<size_t>(dst_cols) + static_cast<size_t>(out_col)] =
      static_cast<uint8_t>(max(0.0f, min(normalized * 255.0f, 255.0f)));
}

void reduce_spectrogram_to_grayscale_row_gpu(const SpectrogramTensor& tensor,
                                             cudaStream_t stream,
                                             uint8_t* device_output,
                                             int output_height,
                                             int output_width,
                                             float db_floor,
                                             float db_ceil) {
  const int src_rows = static_cast<int>(tensor.Size(0));
  const int src_cols = static_cast<int>(tensor.Size(1));
  const bool transpose_input = src_rows > src_cols;
  const int canonical_rows = transpose_input ? src_cols : src_rows;
  const int canonical_cols = transpose_input ? src_rows : src_cols;
  const int dst_rows = std::max(1, std::min(output_height, canonical_rows));
  const int dst_cols = std::max(1, std::min(output_width, canonical_cols));
  const dim3 block(32, 4);
  const dim3 grid((dst_cols + block.x - 1) / block.x, (dst_rows + block.y - 1) / block.y);
  reduce_complex_to_grayscale_kernel<<<grid, block, 0, stream>>>(tensor.Data(),
                                                                 src_rows,
                                                                 src_cols,
                                                                 transpose_input,
                                                                 device_output,
                                                                 dst_rows,
                                                                 dst_cols,
                                                                 db_floor,
                                                                 db_ceil);
}

}  // namespace

namespace holoscan::ops {

namespace {

struct PreviewChannelMailbox {
  std::deque<VisualSpectrogramMessage> spectrogram_queue;
  std::deque<DetectorMaskMessage> mask_queue;
};

constexpr size_t kPreviewMailboxMaxDepth = 256;

std::mutex& preview_mailbox_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::vector<PreviewChannelMailbox>& preview_mailboxes() {
  static std::vector<PreviewChannelMailbox> mailboxes;
  return mailboxes;
}

void ensure_preview_mailbox_capacity(size_t channel_count) {
  auto& mailboxes = preview_mailboxes();
  if (mailboxes.size() < channel_count) {
    mailboxes.resize(channel_count);
  }
}

}  // namespace

std::vector<uint8_t> reduce_mask_to_history_rows_gpu(const uint8_t* device_input,
                                                     int src_height,
                                                     int src_width,
                                                     int dst_width,
                                                     int dst_rows,
                                                     cudaStream_t stream,
                                                     uint8_t* device_output,
                                                     void* pinned_output,
                                                     size_t available_bytes);

void SpectrogramPreviewOp::setup(OperatorSpec& spec) {
  auto& input_port = spec.input<in_t>("in", holoscan::IOSpec::IOSize{8});
  input_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
  spec.output<out_t>("out").condition(holoscan::ConditionType::kNone);
  spec.param(channel_index_,
             "channel_index",
             "Channel Index",
             "Channel index for this preview branch when metadata is absent or stale.",
             -1);
  spec.param(emit_every_n_, "emit_every_n", "Emit Every N", "Forward only every Nth preview frame.", 1);
  spec.param(output_width_, "output_width", "Output Width", "Preview width in bins.", 1024);
  spec.param(output_height_, "output_height", "Output Height", "Preview height in rows.", 16);
  spec.param(db_floor_, "db_floor", "dB Floor", "Fixed dB floor for preview normalization.", -22.0f);
  spec.param(db_ceil_, "db_ceil", "dB Ceiling", "Fixed dB ceiling for preview normalization.", 35.0f);
  spec.param(timing_summary_enable_,
             "timing_summary_enable",
             "Timing Summary Enable",
             "Enable live preview timing summaries.",
             true);
  spec.param(timing_summary_every_n_,
             "timing_summary_every_n",
             "Timing Summary Every N",
             "Emit a live preview timing summary every N preview frames per channel.",
             128);
}

void SpectrogramPreviewOp::initialize() {
  Operator::initialize();
  auto result = cudaStreamCreateWithFlags(&reduce_stream_, cudaStreamNonBlocking);
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to create preview reduction stream: ") + cudaGetErrorString(result));
  }
  timing_stats_.clear();
}

void SpectrogramPreviewOp::ensure_preview_capacity(size_t required_bytes) {
  if (buffer_bytes_ >= required_bytes && device_output_ != nullptr && pinned_output_ != nullptr) {
    return;
  }
  if (device_output_ != nullptr) {
    cudaFree(device_output_);
    device_output_ = nullptr;
  }
  if (pinned_output_ != nullptr) {
    cudaFreeHost(pinned_output_);
    pinned_output_ = nullptr;
  }
  auto device_result = cudaMalloc(reinterpret_cast<void**>(&device_output_), required_bytes);
  if (device_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate preview device buffer: ") + cudaGetErrorString(device_result));
  }
  auto pinned_result = cudaMallocHost(&pinned_output_, required_bytes);
  if (pinned_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate preview pinned buffer: ") + cudaGetErrorString(pinned_result));
  }
  buffer_bytes_ = required_bytes;
}

void SpectrogramPreviewOp::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   ExecutionContext&) {
  auto input = op_input.receive<in_t>("in");
  if (!input) {
    return;
  }
  auto meta = metadata();
  const int configured_channel = channel_index_.get();
  if (meta && meta->get<bool>("chdr_partial_batch", false)) {
    HOLOSCAN_LOG_WARN("Skipping partial CHDR spectrogram preview for channel {} frame {}",
                      meta->get<uint16_t>("channel_number", 0),
                      meta->get<uint64_t>("fft_emitted_frame_number", 0));
    return;
  }
  const uint64_t frame_number = meta ? meta->get<uint64_t>("fft_emitted_frame_number", ++frames_seen_) : ++frames_seen_;
  const uint64_t emit_every_n = static_cast<uint64_t>(std::max(1, emit_every_n_.get()));
  if ((frame_number % emit_every_n) != 0) {
    return;
  }

  auto spectrogram_input = std::move(input.value());
  const uint64_t preview_enter_ns = steady_time_ns();
  const auto& tensor = std::get<0>(spectrogram_input);
  const auto pipeline_stream = std::get<1>(spectrogram_input);
  const int input_rows = static_cast<int>(tensor.Size(0));
  const int input_cols = static_cast<int>(tensor.Size(1));
  const bool transpose_input = input_rows > input_cols;
  const int canonical_rows = transpose_input ? input_cols : input_rows;
  const int canonical_cols = transpose_input ? input_rows : input_cols;
  const int preview_height = std::max(1, std::min(output_height_.get(), canonical_rows));
  const int preview_width = std::max(1, std::min(output_width_.get(), canonical_cols));
  const size_t bytes = static_cast<size_t>(preview_width) * static_cast<size_t>(preview_height) * sizeof(uint8_t);
  ensure_preview_capacity(bytes);

  cudaEvent_t pipeline_done;
  cudaEventCreateWithFlags(&pipeline_done, cudaEventDisableTiming);
  cudaEventRecord(pipeline_done, pipeline_stream);
  cudaStreamWaitEvent(reduce_stream_, pipeline_done, 0);
  cudaEventDestroy(pipeline_done);

  reduce_spectrogram_to_grayscale_row_gpu(tensor,
                                          reduce_stream_,
                                          device_output_,
                                          preview_height,
                                          preview_width,
                                          db_floor_.get(),
                                          db_ceil_.get());
  auto copy_result = cudaMemcpyAsync(pinned_output_, device_output_, bytes, cudaMemcpyDeviceToHost, reduce_stream_);
  if (copy_result != cudaSuccess) {
    throw std::runtime_error(std::string("Preview spectrogram copy failed: ") + cudaGetErrorString(copy_result));
  }
  auto sync_result = cudaStreamSynchronize(reduce_stream_);
  if (sync_result != cudaSuccess) {
    throw std::runtime_error(std::string("Preview spectrogram sync failed: ") + cudaGetErrorString(sync_result));
  }
  const uint64_t preview_emit_ns = steady_time_ns();

  out_t message;
  message.pixels.resize(static_cast<size_t>(preview_width) * static_cast<size_t>(preview_height));
  std::memcpy(message.pixels.data(), pinned_output_, bytes);
  message.width = preview_width;
  message.height = preview_height;
  message.source_rows = canonical_rows;
  message.channel = configured_channel;
  if (meta) {
    const int metadata_channel = static_cast<int>(meta->get<uint16_t>("channel_number", 0));
    if (configured_channel >= 0 && metadata_channel != configured_channel) {
      HOLOSCAN_LOG_WARN("Dropping spectrogram preview on configured channel {} because metadata channel {} does not match frame {}",
                        configured_channel,
                        metadata_channel,
                        frame_number);
      return;
    }
    if (message.channel < 0) {
      message.channel = metadata_channel;
    }
    message.frame_number = meta->get<uint64_t>("fft_emitted_frame_number", frame_number);
    message.fft_emit_ts_ns = meta->get<uint64_t>("fft_emit_ts_ns", 0);
    message.preview_enter_ts_ns = preview_enter_ns;
    message.preview_emit_ts_ns = preview_emit_ns;
    message.center_frequency_hz = std::max(
        std::max(meta->get<double>("center_frequency_hz", 0.0), meta->get<double>("center_frequency", 0.0)),
        meta->get<double>("rx_center_frequency_hz", 0.0));
    message.sample_rate_hz = std::max(meta->get<double>("sample_rate_hz", 0.0),
                                      meta->get<double>("rx_sample_rate_hz", 0.0));
    message.span_hz = static_cast<double>(meta->get<uint64_t>("span", 0));
    message.resolution_hz = static_cast<double>(meta->get<uint64_t>("resolution", 0));
  } else {
    message.frame_number = frame_number;
    message.preview_enter_ts_ns = preview_enter_ns;
    message.preview_emit_ts_ns = preview_emit_ns;
  }
  if (message.channel < 0) {
    message.channel = 0;
  }

  if (timing_summary_enable_.get()) {
    const size_t channel_index = static_cast<size_t>(std::max(0, message.channel));
    if (timing_stats_.size() <= channel_index) {
      timing_stats_.resize(channel_index + 1);
    }
    auto& stats = timing_stats_[channel_index];
    const double fft_to_preview_ms = elapsed_ms(message.fft_emit_ts_ns, preview_enter_ns);
    const double preview_compute_ms = elapsed_ms(preview_enter_ns, preview_emit_ns);
    const double fft_to_preview_emit_ms = elapsed_ms(message.fft_emit_ts_ns, preview_emit_ns);
    ++stats.frames_seen;
    stats.fft_to_preview_total_ms += fft_to_preview_ms;
    stats.fft_to_preview_max_ms = std::max(stats.fft_to_preview_max_ms, fft_to_preview_ms);
    stats.preview_compute_total_ms += preview_compute_ms;
    stats.preview_compute_max_ms = std::max(stats.preview_compute_max_ms, preview_compute_ms);
    stats.fft_to_preview_emit_total_ms += fft_to_preview_emit_ms;
    stats.fft_to_preview_emit_max_ms = std::max(stats.fft_to_preview_emit_max_ms, fft_to_preview_emit_ms);
    const uint64_t summary_every = static_cast<uint64_t>(std::max(1, timing_summary_every_n_.get()));
    if (stats.frames_seen >= summary_every) {
      const double frames = static_cast<double>(std::max<uint64_t>(1, stats.frames_seen));
      HOLOSCAN_LOG_INFO(
          "Spectrogram preview timing ch={} frames={} fft_to_preview_ms(avg/max)={:.3f}/{:.3f} preview_compute_ms(avg/max)={:.3f}/{:.3f} fft_to_preview_emit_ms(avg/max)={:.3f}/{:.3f}",
          message.channel,
          stats.frames_seen,
          stats.fft_to_preview_total_ms / frames,
          stats.fft_to_preview_max_ms,
          stats.preview_compute_total_ms / frames,
          stats.preview_compute_max_ms,
          stats.fft_to_preview_emit_total_ms / frames,
          stats.fft_to_preview_emit_max_ms);
      stats = PreviewTimingStats {};
    }
  }

  op_output.emit(std::move(message), "out");
}

void SpectrogramPreviewOp::stop() {
  if (reduce_stream_ != nullptr) {
    cudaStreamSynchronize(reduce_stream_);
    cudaStreamDestroy(reduce_stream_);
    reduce_stream_ = nullptr;
  }
  if (device_output_ != nullptr) {
    cudaFree(device_output_);
    device_output_ = nullptr;
  }
  if (pinned_output_ != nullptr) {
    cudaFreeHost(pinned_output_);
    pinned_output_ = nullptr;
  }
  buffer_bytes_ = 0;
  Operator::stop();
}

void MaskPreviewOp::setup(OperatorSpec& spec) {
  auto& input_port = spec.input<in_t>("in", holoscan::IOSpec::IOSize{32});
  input_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
  spec.output<in_t>("out").condition(holoscan::ConditionType::kNone);
  spec.param(channel_index_,
             "channel_index",
             "Channel Index",
             "Channel index for this mask preview branch when metadata is absent or stale.",
             -1);
  spec.param(emit_every_n_, "emit_every_n", "Emit Every N", "Forward only every Nth reduced preview mask.", 1);
  spec.param(output_width_, "output_width", "Output Width", "Preview mask width in bins.", 1024);
  spec.param(output_height_, "output_height", "Output Height", "Preview mask height in rows.", 16);
}

void MaskPreviewOp::initialize() {
  Operator::initialize();
  auto result = cudaStreamCreateWithFlags(&reduce_stream_, cudaStreamNonBlocking);
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to create mask preview reduction stream: ") + cudaGetErrorString(result));
  }
}

void MaskPreviewOp::ensure_preview_capacity(size_t required_bytes) {
  if (buffer_bytes_ >= required_bytes && device_output_ != nullptr && pinned_output_ != nullptr) {
    return;
  }
  if (device_output_ != nullptr) {
    cudaFree(device_output_);
    device_output_ = nullptr;
  }
  if (pinned_output_ != nullptr) {
    cudaFreeHost(pinned_output_);
    pinned_output_ = nullptr;
  }
  auto device_result = cudaMalloc(reinterpret_cast<void**>(&device_output_), required_bytes);
  if (device_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate mask preview device buffer: ") + cudaGetErrorString(device_result));
  }
  auto pinned_result = cudaMallocHost(&pinned_output_, required_bytes);
  if (pinned_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate mask preview pinned buffer: ") + cudaGetErrorString(pinned_result));
  }
  buffer_bytes_ = required_bytes;
}

void MaskPreviewOp::compute(InputContext& op_input,
                            OutputContext& op_output,
                            ExecutionContext&) {
  auto input = op_input.receive<in_t>("in");
  if (!input) {
    return;
  }
  auto mask = std::move(input.value());
  const uint64_t frame_number = mask.frame_number == 0 ? ++frames_seen_ : mask.frame_number;
  const uint64_t emit_every_n = static_cast<uint64_t>(std::max(1, emit_every_n_.get()));
  if ((frame_number % emit_every_n) != 0) {
    return;
  }

  const int configured_channel = channel_index_.get();
  if (configured_channel >= 0 && mask.channel >= 0 && mask.channel != configured_channel) {
    HOLOSCAN_LOG_WARN("Dropping mask preview on configured channel {} because message channel {} does not match frame {}",
                      configured_channel,
                      mask.channel,
                      frame_number);
    return;
  }

  const int preview_width = std::max(1, output_width_.get());
  const int preview_height = std::max(1, output_height_.get());
  const size_t bytes = static_cast<size_t>(preview_width) * static_cast<size_t>(preview_height) * sizeof(uint8_t);
  ensure_preview_capacity(bytes);

  DetectorMaskMessage reduced_mask;
  reduced_mask.channel = configured_channel >= 0 ? configured_channel : mask.channel;
  reduced_mask.frame_number = frame_number;
  reduced_mask.width = preview_width;
  reduced_mask.height = preview_height;
  if (mask.device_pixels) {
    reduced_mask.pixels = reduce_mask_to_history_rows_gpu(mask.device_pixels.get(),
                                                          mask.height,
                                                          mask.width,
                                                          preview_width,
                                                          preview_height,
                                                          reduce_stream_,
                                                          device_output_,
                                                          pinned_output_,
                                                          buffer_bytes_);
  } else if (!mask.pixels.empty()) {
    OfflinePgmFrame host_mask;
    host_mask.pixels = mask.pixels;
    host_mask.width = mask.width;
    host_mask.height = mask.height;
    reduced_mask.pixels = reduce_mask_to_history_rows(host_mask, preview_width, preview_height);
  }
  op_output.emit(std::move(reduced_mask), "out");
}

void MaskPreviewOp::stop() {
  if (reduce_stream_ != nullptr) {
    cudaStreamSynchronize(reduce_stream_);
    cudaStreamDestroy(reduce_stream_);
    reduce_stream_ = nullptr;
  }
  if (device_output_ != nullptr) {
    cudaFree(device_output_);
    device_output_ = nullptr;
  }
  if (pinned_output_ != nullptr) {
    cudaFreeHost(pinned_output_);
    pinned_output_ = nullptr;
  }
  buffer_bytes_ = 0;
  Operator::stop();
}

void SpectrogramPreviewStoreOp::setup(OperatorSpec& spec) {
  auto& input_port = spec.input<in_t>("in", holoscan::IOSpec::IOSize{8});
  input_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
}

void SpectrogramPreviewStoreOp::compute(InputContext& op_input,
                                        OutputContext&,
                                        ExecutionContext&) {
  auto input = op_input.receive<in_t>("in");
  if (!input) {
    return;
  }
  auto message = std::move(input.value());
  if (message.channel < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(preview_mailbox_mutex());
  ensure_preview_mailbox_capacity(static_cast<size_t>(message.channel + 1));
  auto& queue = preview_mailboxes()[static_cast<size_t>(message.channel)].spectrogram_queue;
  queue.push_back(std::move(message));
  while (queue.size() > kPreviewMailboxMaxDepth) {
    queue.pop_front();
  }
}

void MaskPreviewStoreOp::setup(OperatorSpec& spec) {
  auto& input_port = spec.input<in_t>("in", holoscan::IOSpec::IOSize{8});
  input_port.conditions().emplace_back(
      holoscan::ConditionType::kMessageAvailable,
      std::make_shared<holoscan::MessageAvailableCondition>(size_t{1}));
}

void MaskPreviewStoreOp::compute(InputContext& op_input,
                                 OutputContext&,
                                 ExecutionContext&) {
  auto input = op_input.receive<in_t>("in");
  if (!input) {
    return;
  }
  auto message = std::move(input.value());
  if (message.channel < 0) {
    return;
  }
  std::lock_guard<std::mutex> lock(preview_mailbox_mutex());
  ensure_preview_mailbox_capacity(static_cast<size_t>(message.channel + 1));
  auto& queue = preview_mailboxes()[static_cast<size_t>(message.channel)].mask_queue;
  queue.push_back(std::move(message));
  while (queue.size() > kPreviewMailboxMaxDepth) {
    queue.pop_front();
  }
}

std::vector<uint8_t> reduce_mask_to_history_rows_gpu(const uint8_t* device_input,
                                                     int src_height,
                                                     int src_width,
                                                     int dst_width,
                                                     int dst_rows,
                                                     cudaStream_t stream,
                                                     uint8_t* device_output,
                                                     void* pinned_output,
                                                     size_t available_bytes) {
  if (device_input == nullptr || src_width <= 0 || src_height <= 0 || dst_width <= 0 || dst_rows <= 0) {
    return {};
  }

  const size_t reduced_bytes = static_cast<size_t>(dst_width) * static_cast<size_t>(dst_rows) * sizeof(uint8_t);
  if (reduced_bytes > available_bytes || device_output == nullptr || pinned_output == nullptr) {
    throw std::runtime_error("visualizer mask reduction buffers are not initialized for the requested live mask size");
  }

  const dim3 block(32, 4);
  const dim3 grid((dst_width + block.x - 1) / block.x, (dst_rows + block.y - 1) / block.y);
  reduce_binary_mask_to_alpha_kernel<<<grid, block, 0, stream>>>(device_input,
                                                                  src_height,
                                                                  src_width,
                                                                  device_output,
                                                                  dst_rows,
                                                                  dst_width);
  auto kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    throw std::runtime_error(std::string("visualizer mask reduction kernel launch failed: ") +
                             cudaGetErrorString(kernel_result));
  }

  auto copy_result = cudaMemcpyAsync(pinned_output,
                                     device_output,
                                     reduced_bytes,
                                     cudaMemcpyDeviceToHost,
                                     stream);
  if (copy_result != cudaSuccess) {
    throw std::runtime_error(std::string("visualizer reduced mask copy failed: ") + cudaGetErrorString(copy_result));
  }

  auto sync_result = cudaStreamSynchronize(stream);
  if (sync_result != cudaSuccess) {
    throw std::runtime_error(std::string("visualizer reduced mask sync failed: ") + cudaGetErrorString(sync_result));
  }

  std::vector<uint8_t> reduced(static_cast<size_t>(dst_width) * static_cast<size_t>(dst_rows), 0);
  std::memcpy(reduced.data(), pinned_output, reduced_bytes);
  return reduced;
}


holoscan::gxf::Entity create_rgb_entity(holoscan::ExecutionContext& context,
                                        const std::vector<uint8_t>& rgb,
                                        int width,
                                        int height,
                                        const std::string& tensor_name) {
  const size_t bytes = rgb.size() * sizeof(uint8_t);
  auto lease = std::shared_ptr<DeviceBufferLease>(new DeviceBufferLease{acquire_output_buffer(bytes), bytes},
                                                  [](DeviceBufferLease* buffer) {
                                                    if (buffer != nullptr) {
                                                      recycle_output_buffer(buffer->ptr, buffer->bytes);
                                                      delete buffer;
                                                    }
                                                  });

  auto copy_result = cudaMemcpyAsync(lease->ptr, rgb.data(), bytes, cudaMemcpyHostToDevice, 0);
  if (copy_result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(copy_result));
  }

  auto message = nvidia::gxf::Entity::New(context.context());
  if (!message) {
    throw std::runtime_error("Failed to create GXF entity for spectrogram visualization");
  }

  auto tensor = message.value().add<nvidia::gxf::Tensor>(tensor_name.c_str());
  if (!tensor) {
    throw std::runtime_error("Failed to create output tensor for spectrogram visualization");
  }

  const auto shape = nvidia::gxf::Shape{height, width, 3};
  const auto element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
  const auto element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

  tensor.value()->wrapMemory(shape,
                             element_type,
                             element_size,
                             nvidia::gxf::ComputeTrivialStrides(shape, element_size),
                             nvidia::gxf::MemoryStorageType::kDevice,
                             lease->ptr,
                             [buffer = lease](void*) mutable {
                               buffer.reset();
                               return nvidia::gxf::Success;
                             });

  return holoscan::gxf::Entity(message.value());
}



std::vector<uint8_t> reduce_from_pinned_buffer(const void* pinned_buffer,
                                               int src_rows,
                                               int src_cols,
                                               int output_height,
                                               int output_width,
                                               float db_floor = -100.0f,
                                               float db_ceil  =    0.0f)  {
  using Complex = cuda::std::complex<float>;
  const Complex* host_fft = static_cast<const Complex*>(pinned_buffer);

  const int dst_rows = std::max(1, std::min(output_height, src_rows));
  const int dst_cols = std::max(1, std::min(output_width, src_cols));

  std::vector<float> reduced(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), -120.0f);

  for (int row = 0; row < dst_rows; ++row) {
    const int row_start = (row * src_rows) / dst_rows;
    const int row_end = ((row + 1) * src_rows) / dst_rows;
    for (int col = 0; col < dst_cols; ++col) {
      const int col_start = (col * src_cols) / dst_cols;
      const int col_end = ((col + 1) * src_cols) / dst_cols;
      double accumulation = 0.0;
      int count = 0;
      for (int src_row = row_start; src_row < std::max(row_start + 1, row_end); ++src_row) {
        for (int src_col = col_start; src_col < std::max(col_start + 1, col_end); ++src_col) {
          const auto& value = host_fft[static_cast<size_t>(src_row) * static_cast<size_t>(src_cols) + static_cast<size_t>(src_col)];
          const float real = value.real();
          const float imag = value.imag();
          accumulation += 10.0 * std::log10(real * real + imag * imag + 1e-12f);
          ++count;
        }
      }
      reduced[static_cast<size_t>(row) * static_cast<size_t>(dst_cols) + static_cast<size_t>(col)] =
          static_cast<float>(accumulation / static_cast<double>(std::max(1, count)));
    }
  }

  const float db_range = std::max(1.0f, db_ceil - db_floor);
  // TEMP DEBUG — remove after tuning
  static int debug_count = 0;
  if (++debug_count % 50 == 0) {
    float min_val = *std::min_element(reduced.begin(), reduced.end());
    float max_val = *std::max_element(reduced.begin(), reduced.end());
    printf("DBG actual dB range: min=%.1f max=%.1f\n", min_val, max_val);
  }
  std::vector<uint8_t> grayscale(reduced.size());
  for (size_t index = 0; index < reduced.size(); ++index) {
    const float normalized = (reduced[index] - db_floor) / db_range;
    grayscale[index] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
  }
  return grayscale;
}

void initialize_visualization_overlay_state(bool enabled) {
  global_overlay_enabled().store(enabled, std::memory_order_relaxed);
}

void set_visualization_overlay_enabled(bool enabled) {
  global_overlay_enabled().store(enabled, std::memory_order_relaxed);
}

bool visualization_overlay_enabled() {
  return global_overlay_enabled().load(std::memory_order_relaxed);
}

void set_visualization_full_ui_enabled(bool enabled) {
  global_full_ui_enabled().store(enabled, std::memory_order_relaxed);
}

bool visualization_full_ui_enabled() {
  return global_full_ui_enabled().load(std::memory_order_relaxed);
}

void update_visualization_ui_state(const VisualizationUiState& state) {
  std::lock_guard<std::mutex> lock(visualization_ui_state_mutex());
  visualization_ui_state_storage() = state;
}

VisualizationUiState visualization_ui_state_snapshot() {
  std::lock_guard<std::mutex> lock(visualization_ui_state_mutex());
  return visualization_ui_state_storage();
}

void render_visualization_ui_overlay() {
  const auto state = visualization_ui_state_snapshot();
  if (!visualization_full_ui_enabled() || state.canvas_width <= 0 || state.canvas_height <= 0) {
    return;
  }

  const ImVec2 display_size = ImGui::GetIO().DisplaySize;
  if (display_size.x <= 0.0f || display_size.y <= 0.0f) {
    return;
  }

  auto* draw_list = ImGui::GetBackgroundDrawList();
  const ImU32 panel_bg = IM_COL32(11, 15, 23, 210);
  const ImU32 panel_border = IM_COL32(62, 74, 92, 255);
  const ImU32 panel_muted = IM_COL32(122, 143, 168, 255);
  const ImU32 panel_text = IM_COL32(232, 236, 241, 255);
  const ImU32 accent_orange = IM_COL32(255, 130, 48, 255);
  const ImU32 accent_green = IM_COL32(80, 200, 120, 255);
  const ImU32 accent_blue = IM_COL32(84, 196, 255, 255);

  auto rect_min = [&](const VisualizationRect& rect) {
    return denormalize_point(rect.x, rect.y, display_size);
  };
  auto rect_max = [&](const VisualizationRect& rect) {
    return ImVec2((rect.x + rect.width) * display_size.x, (rect.y + rect.height) * display_size.y);
  };
  auto resolved_span_hz = [](const VisualizationChannelUiState& channel) {
    if (std::isfinite(channel.span_hz) && channel.span_hz > 0.0) {
      return channel.span_hz;
    }
    if (std::isfinite(channel.resolution_hz) && channel.resolution_hz > 0.0) {
      return channel.resolution_hz * static_cast<double>(std::max(1, channel.fft_size));
    }
    return 0.0;
  };
  auto draw_panel = [&](const VisualizationRect& rect, float rounding = 12.0f) {
    draw_list->AddRectFilled(rect_min(rect), rect_max(rect), panel_bg, rounding);
    draw_list->AddRect(rect_min(rect), rect_max(rect), panel_border, rounding, 0, 1.0f);
  };

  draw_panel(state.header_rect, 14.0f);
  draw_panel(state.sidebar_rect, 14.0f);

  const ImVec2 header_min = rect_min(state.header_rect);
  const ImVec2 header_max = rect_max(state.header_rect);
  draw_list->AddText(ImVec2(header_min.x + 18.0f, header_min.y + 10.0f), panel_text, state.title.c_str());
  draw_list->AddText(ImVec2(header_min.x + 18.0f, header_min.y + 30.0f), accent_blue, state.subtitle.c_str());

  ImGui::SetNextWindowBgAlpha(0.82f);
  ImGui::SetNextWindowPos(ImVec2(header_max.x - 16.0f, header_min.y + 12.0f),
                          ImGuiCond_Always,
                          ImVec2(1.0f, 0.0f));
  ImGui::Begin("Display Controls",
               nullptr,
               ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove |
                   ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse |
                   ImGuiWindowFlags_NoSavedSettings);
  bool overlay_enabled = visualization_overlay_enabled();
  if (ImGui::Checkbox("Detect", &overlay_enabled)) {
    set_visualization_overlay_enabled(overlay_enabled);
  }
  ImGui::Separator();
  ImGui::TextColored(ImVec4(0.48f, 0.78f, 1.0f, 1.0f), "%s", state.detector_label.c_str());
  ImGui::End();

  const ImVec2 sidebar_min = rect_min(state.sidebar_rect);
  draw_list->AddText(ImVec2(sidebar_min.x + 16.0f, sidebar_min.y + 16.0f), panel_text, "Channel Info");

  float sidebar_text_y = sidebar_min.y + 42.0f;
  for (const auto& channel : state.channels) {
    if (!channel.active) {
      continue;
    }
    const std::string channel_name = std::string("CH-") + std::to_string(channel.channel);
    draw_list->AddText(ImVec2(sidebar_min.x + 16.0f, sidebar_text_y), panel_text, channel_name.c_str());
    sidebar_text_y += 18.0f;

    auto draw_kv = [&](const char* key, const std::string& value) {
      draw_list->AddText(ImVec2(sidebar_min.x + 16.0f, sidebar_text_y), panel_muted, key);
      draw_list->AddText(ImVec2(sidebar_min.x + 92.0f, sidebar_text_y), panel_text, value.c_str());
      sidebar_text_y += 16.0f;
    };

    draw_kv("Center", format_frequency_label(channel.center_frequency_hz));
    draw_kv("Span", format_frequency_label(channel.span_hz));
    draw_kv("Freq Bin", format_frequency_label(channel.display_frequency_bin_hz));
    draw_kv("Time Bin", format_time_bin_label(channel.seconds_per_time_bin));
    draw_kv("Frames", format_displayed_frame_ratio_label(channel.displayed_frame_ratio,
                                channel.displayed_frame_stride));
    draw_kv("Vis Ratio", format_fft_row_visualization_ratio_label(channel.displayed_fft_rows_per_frame,
                                     channel.fft_rows_per_frame));
    draw_kv("FFT", std::to_string(channel.fft_size));
    draw_kv("History", std::to_string(channel.history_rows) + " bins");
    draw_kv("Chunk",
            std::to_string(channel.dino_chunk_rows) + " x " + std::to_string(channel.dino_chunk_cols));

    draw_list->AddLine(ImVec2(sidebar_min.x + 16.0f, sidebar_text_y),
                       ImVec2(rect_max(state.sidebar_rect).x - 16.0f, sidebar_text_y),
                       IM_COL32(17, 29, 42, 255),
                       1.0f);
    sidebar_text_y += 12.0f;
  }

  for (const auto& channel : state.channels) {
    if (!channel.active) {
      continue;
    }

    draw_panel(channel.header_rect, 12.0f);
    const ImVec2 header_pos = rect_min(channel.header_rect);
    const std::string channel_title = std::string("CH-") + std::to_string(channel.channel) + "  " + format_frequency_label(channel.center_frequency_hz);
    draw_list->AddText(ImVec2(header_pos.x + 12.0f, header_pos.y + 10.0f), panel_text, channel_title.c_str());

    auto draw_frequency_grid_labels = [&](const VisualizationRect& rect) {
      constexpr int kGridDivisions = 8;
      const ImVec2 min = rect_min(rect);
      const ImVec2 max = rect_max(rect);
      const double span_hz = resolved_span_hz(channel);
      if (!std::isfinite(channel.center_frequency_hz) || span_hz <= 0.0) {
        return;
      }

      const double start_hz = channel.center_frequency_hz - (span_hz * 0.5);
      const float axis_y = max.y + 6.0f;
      for (int step = 0; step <= kGridDivisions; ++step) {
        const float t = static_cast<float>(step) / static_cast<float>(kGridDivisions);
        const double frequency_hz = start_hz + span_hz * static_cast<double>(t);
        const std::string label = format_frequency_label(frequency_hz);
        const ImVec2 text_size = ImGui::CalcTextSize(label.c_str());
        const float tick_x = min.x + (max.x - min.x) * t;
        const float label_x = std::clamp(tick_x - text_size.x * 0.5f, min.x, max.x - text_size.x);
        draw_list->AddText(ImVec2(label_x, axis_y), panel_muted, label.c_str());
      }
    };

    auto annotate_plot = [&](const VisualizationRect& rect,
                             const char* title,
                             const std::string& y0,
                             const std::string& y1) {
      const ImVec2 min = rect_min(rect);
      const ImVec2 max = rect_max(rect);
      draw_list->AddText(ImVec2(min.x + 10.0f, min.y + 8.0f), panel_text, title);
      draw_frequency_grid_labels(rect);
      draw_list->AddText(ImVec2(min.x - 36.0f, min.y - 2.0f), panel_muted, y1.c_str());
      draw_list->AddText(ImVec2(min.x - 28.0f, max.y - 14.0f), panel_muted, y0.c_str());
    };

    annotate_plot(channel.psd_rect,
                  "PSD / Max Hold",
                  "-100 dB",
                  "0 dB");
    annotate_plot(channel.waterfall_rect,
                  "Spectrogram",
                  "0",
                  std::to_string(channel.history_rows));
    const bool mask_enabled = state.overlay_enabled;
    const char* mask_title = mask_enabled ? "Signal Detection Mask Enabled"
                                          : "Signal Detection Mask Disabled";
    const ImU32 mask_title_color = mask_enabled ? accent_green : IM_COL32(228, 72, 72, 255);
    const ImVec2 mask_min = rect_min(channel.mask_rect);
    annotate_plot(channel.mask_rect,
                  "",
                  "0",
                  std::to_string(channel.history_rows));
    draw_list->AddText(ImGui::GetFont(),
                       ImGui::GetFontSize() * 2.0f,
                       ImVec2(mask_min.x + 10.0f, mask_min.y + 8.0f),
                       mask_title_color,
                       mask_title);
  }

  if (state.total_frames > 0) {
    const float live_ratio = 1.0f - static_cast<float>(state.dropped_frames) / static_cast<float>(state.total_frames);
    std::ostringstream footer_text;
    footer_text << "VIS " << static_cast<int>(std::round(live_ratio * 100.0f)) << "% LIVE";
    draw_list->AddText(ImVec2(display_size.x - 140.0f, display_size.y - 28.0f),
                       live_ratio < 0.9f ? accent_orange : accent_green,
                       footer_text.str().c_str());
  }
}

std::vector<uint8_t> cleanup_live_mask_for_display(const std::vector<uint8_t>& reduced_mask,
                                                   int width,
                                                   int height);

namespace {

void update_timing_accumulator(double value_ms, double& total_ms, double& max_ms) {
  total_ms += value_ms;
  max_ms = std::max(max_ms, value_ms);
}

int live_visualization_width(int configured_width, int panel_width) {
  return std::max(1, std::min(configured_width, panel_width));
}

int live_visualization_history_rows(int configured_rows, int canvas_height, int active_channels) {
  const auto panel_heights = compute_channel_panel_heights(canvas_height, active_channels);
  return std::max(1, std::min(configured_rows, panel_heights.heat));
}

void update_configured_display_metrics(holoscan::ops::VisualizationFrameInfo& info,
                                       double span_hz,
                                       int display_width_bins,
                                       int fft_size,
                                       int rows_per_frame,
                                       int render_every_n_frames) {
  info.display_frequency_bin_hz = compute_display_frequency_bin_hz(span_hz, display_width_bins);
  info.display_time_bin_seconds = compute_display_time_bin_seconds(span_hz, fft_size, rows_per_frame);
  info.displayed_frame_stride = std::max(1, render_every_n_frames);
  info.displayed_frame_ratio = compute_displayed_frame_ratio(render_every_n_frames);
}
}  // namespace

void SpectrogramToHolovizOp::setup(OperatorSpec& spec) {
  auto& spectrogram_input_port0 = spec.input<VisualSpectrogramMessage>("in0", holoscan::IOSpec::IOSize{64});
  auto& spectrogram_input_port1 = spec.input<VisualSpectrogramMessage>("in1", holoscan::IOSpec::IOSize{64});
  spectrogram_input_port0.condition(holoscan::ConditionType::kNone);
  spectrogram_input_port1.condition(holoscan::ConditionType::kNone);
  auto& mask_input_port0 = spec.input<DetectorMaskMessage>("mask_in0", holoscan::IOSpec::IOSize{32});
  auto& mask_input_port1 = spec.input<DetectorMaskMessage>("mask_in1", holoscan::IOSpec::IOSize{32});
  mask_input_port0.condition(holoscan::ConditionType::kNone);
  mask_input_port1.condition(holoscan::ConditionType::kNone);
  spec.output<gxf::Entity>("outputs");
  spec.param(num_channels_, "num_channels", "Num Channels", "Number of channels shown in the analyzer view.", 2);
  spec.param(history_frames_, "history_frames", "History Frames", "Number of consecutive spectrogram frames retained in the display history.", 5);
  spec.param(output_height_, "output_height", "Output Height", "Output spectrogram height.", 256);
  spec.param(output_width_, "output_width", "Output Width", "Output spectrogram width.", 512);
  spec.param(channel_filter_,
             "channel_filter",
             "Channel Filter",
             "Only render this channel index. Use -1 to render every incoming channel.",
             -1);
  spec.param(tensor_name_,
             "tensor_name",
             "Tensor Name",
             "Holoviz tensor name for the rendered spectrogram image.",
             std::string("spectrogram"));
  spec.param(blue_limit_, "blue_limit", "Blue Limit", "Lower color scale clamp in normalized units.", 0.10f);
  spec.param(red_limit_, "red_limit", "Red Limit", "Upper color scale clamp in normalized units.", 0.92f);
  spec.param(overlay_alpha_,
             "overlay_alpha",
             "Overlay Alpha",
             "Alpha used when detector overlays are present in composite rendering.",
             0.38f);
  spec.param(overlay_enable_,
             "overlay_enable",
             "Overlay Enable",
             "Enable or disable the detection overlay layer in the composed visualization.",
             true);
  spec.param(detector_label_,
             "detector_label",
             "Detector Label",
             "Human-readable detector label used in the visualization UI.",
             std::string("Dinov3"));
  spec.param(center_frequency_hz_, "center_frequency_hz", "Center Frequency", "Center frequency for display in Hz.", 0.0);
  spec.param(span_hz_, "span_hz", "Span Hz", "Frequency span shown on calibrated plot axes in Hz.", 0.0);
  spec.param(fft_size_, "fft_size", "FFT Size", "FFT size shown in analyzer readouts.", 20480);
  spec.param(dino_chunk_rows_, "dino_chunk_rows", "DINO Chunk Rows", "DINO chunk height shown in readouts.", 256);
  //spec.param(dino_chunk_cols_, "dino_chunk_cols", "DINO Chunk Cols", "DINO chunk width shown in readouts.", 512);
  spec.param(dino_chunk_cols_, "dino_chunk_cols", "DINO Chunk Cols", "DINO chunk width shown in readouts.", 512);
  spec.param(display_time_rows_, "display_time_rows", "Display Time Rows", "Number of waterfall rows retained in the live ring buffer.", 1024);
  spec.param(display_freq_bins_, "display_freq_bins", "Display Freq Bins", "Number of frequency bins retained in the live waterfall.", 2048);
  spec.param(history_memory_budget_mb_,
             "history_memory_budget_mb",
             "History Memory Budget MB",
             "Maximum host memory budget in MiB for the live waterfall history ring buffers.",
             512);
  spec.param(rows_per_frame_, "rows_per_frame", "Rows Per Frame", "Number of display rows emitted for each incoming spectrogram frame.", 1);
  spec.param(mask_frame_offset_,
             "mask_frame_offset",
             "Mask Frame Offset",
             "Shift detector masks by this many whole spectrogram frames before patching them into history.",
             0);
  spec.param(render_every_n_frames_, "render_every_n_frames", "Render Every N Frames", "Compose a UI frame every N processed spectrogram frames while still updating history on every frame.", 4);
  spec.param(timing_summary_enable_, "timing_summary_enable", "Timing Summary Enable", "Enable live visualizer timing summaries.", false);
  spec.param(timing_summary_every_n_, "timing_summary_every_n", "Timing Summary Every N", "Emit a live visualizer timing summary every N seen frames.", 128);
  spec.param(shutdown_scheduling_term_,
             "shutdown_scheduling_term",
             "Shutdown Scheduling Term",
             "Boolean scheduling term used to stop the visualization branch during shutdown.");
  spec.param(db_floor_, "db_floor", "dB Floor", "Fixed dB floor for spectrogram normalization.", -100.0f);
  spec.param(db_ceil_,  "db_ceil",  "dB Ceiling", "Fixed dB ceiling for spectrogram normalization.", 0.0f);
  spec.param(row_average_n_, "row_average_n", "Row Average N", "Frames averaged per waterfall row.", 4);
}


void SpectrogramToHolovizOp::initialize() {
  auto frag = fragment();
  auto has_shutdown_scheduling_term = std::find_if(args().begin(), args().end(), [](const auto& arg) {
    return arg.name() == "shutdown_scheduling_term";
  });
  if (has_shutdown_scheduling_term == args().end()) {
    shutdown_scheduling_term_ =
        frag->make_condition<holoscan::BooleanCondition>("shutdown_scheduling_term", true);
    add_arg(shutdown_scheduling_term_.get());
  }

  Operator::initialize();
  metadata_policy(holoscan::MetadataPolicy::kUpdate);  // ← add this
  if (auto shutdown_term = shutdown_scheduling_term_.get()) {
    shutdown_term->enable_tick();
  }
  initialize_visualization_overlay_state(overlay_enable_.get());
  render_stop_ = false;
  render_work_pending_ = false;
  pending_composed_ready_ = false;
  const size_t channel_count = static_cast<size_t>(std::max(1, num_channels_.get()));
  const int live_width = live_visualization_width(display_freq_bins_.get(), output_width_.get());
  const int live_rows =
      live_visualization_history_rows(display_time_rows_.get(), output_height_.get(), num_channels_.get());
  const int rows_per_frame = std::max(1, rows_per_frame_.get());
  const int render_every_n_frames = std::max(1, render_every_n_frames_.get());
  HOLOSCAN_LOG_INFO("Spectrogram visualizer initializing with num_channels={} display_freq_bins={} rows_per_frame={} render_every_n_frames={}",
                    channel_count,
                    live_width,
                    rows_per_frame_.get(),
                    render_every_n_frames_.get());
  HOLOSCAN_LOG_INFO("Spectrogram visualizer live history clamped to width={} rows={}",
                    live_width,
                    live_rows);
  channel_states_.assign(channel_count, {});
  for (size_t channel_index = 0; channel_index < channel_states_.size(); ++channel_index) {
    auto& state = channel_states_[channel_index];
    state.active = true;
    state.info.channel = static_cast<int>(channel_index);
    state.info.center_frequency_hz = center_frequency_hz_.get();
    state.info.span_hz = span_hz_.get();
    state.info.resolution_hz = state.info.span_hz > 0.0
        ? state.info.span_hz / static_cast<double>(std::max(1, live_width))
        : 0.0;
    state.info.fft_size = fft_size_.get();
    state.info.fft_rows_per_frame = 0;
    state.info.displayed_fft_rows_per_frame = 0;
    update_configured_display_metrics(state.info,
                      state.info.span_hz,
                      live_width,
                      state.info.fft_size,
                      rows_per_frame,
                      render_every_n_frames);
    state.info.dino_chunk_rows = dino_chunk_rows_.get();
    state.info.dino_chunk_cols = dino_chunk_cols_.get();
    state.info.detector_label = detector_label_.get();
  }
  latest_rendered_frame_numbers_.assign(channel_count, 0);
  channel_resources_.assign(channel_count, {});
  pending_channel_frames_.assign(channel_count, {});
  per_channel_timing_stats_.assign(channel_count, PerChannelVisualTimingStats {});
}

void SpectrogramToHolovizOp::ensure_channel_resource_capacity(size_t channel_index, size_t required_bytes) {
  if (channel_index >= channel_resources_.size()) {
    throw std::runtime_error("visualizer channel resource index exceeds configured num_channels");
  }
  auto& resources = channel_resources_[channel_index];
  if (resources.stream == nullptr) {
    auto result = cudaStreamCreateWithFlags(&resources.stream, cudaStreamNonBlocking);
    if (result != cudaSuccess) {
      throw std::runtime_error(std::string("Failed to create channel vis stream: ") + cudaGetErrorString(result));
    }
  }

  if (resources.grayscale_buffer_bytes >= required_bytes && resources.device_grayscale_buffer != nullptr &&
      resources.pinned_grayscale_buffer != nullptr && resources.device_mask_buffer != nullptr &&
      resources.pinned_mask_buffer != nullptr) {
    return;
  }

  if (resources.device_grayscale_buffer != nullptr) {
    cudaFree(resources.device_grayscale_buffer);
    resources.device_grayscale_buffer = nullptr;
  }
  if (resources.pinned_grayscale_buffer != nullptr) {
    cudaFreeHost(resources.pinned_grayscale_buffer);
    resources.pinned_grayscale_buffer = nullptr;
  }
  if (resources.device_mask_buffer != nullptr) {
    cudaFree(resources.device_mask_buffer);
    resources.device_mask_buffer = nullptr;
  }
  if (resources.pinned_mask_buffer != nullptr) {
    cudaFreeHost(resources.pinned_mask_buffer);
    resources.pinned_mask_buffer = nullptr;
  }

  auto device_result = cudaMalloc(reinterpret_cast<void**>(&resources.device_grayscale_buffer), required_bytes);
  if (device_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate device grayscale buffer: ") + cudaGetErrorString(device_result));
  }
  auto pinned_result = cudaMallocHost(&resources.pinned_grayscale_buffer, required_bytes);
  if (pinned_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate pinned grayscale buffer: ") + cudaGetErrorString(pinned_result));
  }
  auto mask_device_result = cudaMalloc(reinterpret_cast<void**>(&resources.device_mask_buffer), required_bytes);
  if (mask_device_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate device mask buffer: ") + cudaGetErrorString(mask_device_result));
  }
  auto mask_pinned_result = cudaMallocHost(&resources.pinned_mask_buffer, required_bytes);
  if (mask_pinned_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate pinned mask buffer: ") + cudaGetErrorString(mask_pinned_result));
  }
  resources.grayscale_buffer_bytes = required_bytes;
}

void SpectrogramToHolovizOp::stop() {
  HOLOSCAN_LOG_INFO("SpectrogramToHolovizOp::stop() begin");
  if (auto shutdown_term = shutdown_scheduling_term_.get()) {
    shutdown_term->disable_tick();
  }
  // Signal background render thread to stop and wait for it
  {
    std::unique_lock<std::mutex> lock(render_mutex_);
    render_stop_ = true;
    render_cv_.notify_one();
  }
  if (render_thread_.joinable()) {
    render_thread_.join();
  }
  for (auto& resources : channel_resources_) {
    if (resources.stream != nullptr) {
      cudaStreamSynchronize(resources.stream);
      cudaStreamDestroy(resources.stream);
      resources.stream = nullptr;
    }
    if (resources.device_grayscale_buffer != nullptr) {
      cudaFree(resources.device_grayscale_buffer);
      resources.device_grayscale_buffer = nullptr;
    }
    if (resources.pinned_grayscale_buffer != nullptr) {
      cudaFreeHost(resources.pinned_grayscale_buffer);
      resources.pinned_grayscale_buffer = nullptr;
    }
    if (resources.device_mask_buffer != nullptr) {
      cudaFree(resources.device_mask_buffer);
      resources.device_mask_buffer = nullptr;
    }
    if (resources.pinned_mask_buffer != nullptr) {
      cudaFreeHost(resources.pinned_mask_buffer);
      resources.pinned_mask_buffer = nullptr;
    }
    resources.grayscale_buffer_bytes = 0;
  }
  channel_resources_.clear();
  HOLOSCAN_LOG_INFO("SpectrogramToHolovizOp::stop() calling Operator::stop()");
  Operator::stop();
  HOLOSCAN_LOG_INFO("SpectrogramToHolovizOp::stop() complete");
}

//end newy added


void SpectrogramToHolovizOp::compute(InputContext& op_input,
                                     OutputContext& op_output,
                                     ExecutionContext& context) { 
  if (adv_net_shutdown_requested_if_available()) {
    if (auto shutdown_term = shutdown_scheduling_term_.get()) {
      shutdown_term->disable_tick();
    }
    return;
  }

  auto maybe_emit_timing_summary = [this]() {
    if (!timing_summary_enable_.get()) {
      return;
    }
    const uint64_t summary_every = static_cast<uint64_t>(std::max(1, timing_summary_every_n_.get()));
    std::lock_guard<std::mutex> lock(timing_mutex_);
    if (timing_stats_.frames_seen == 0 || (timing_stats_.frames_seen % summary_every) != 0) {
      return;
    }

     const double seen = static_cast<double>(std::max<uint64_t>(1, timing_stats_.frames_seen));
     const double processed = static_cast<double>(std::max<uint64_t>(1, timing_stats_.frames_processed));
     const double rendered = static_cast<double>(std::max<uint64_t>(1, timing_stats_.frames_rendered));
    std::ostringstream os;
    os << "Visualizer timing: frames_seen=" << timing_stats_.frames_seen
       << " processed=" << timing_stats_.frames_processed
       << " rendered=" << timing_stats_.frames_rendered
       << " drop_vis_busy=" << timing_stats_.dropped_vis_stream_busy
       << " drop_render_busy=" << timing_stats_.dropped_render_queue_busy
       << " cadence_skips=" << timing_stats_.render_skipped_by_cadence
       << " masks(received/backfilled/deferred/peak)="
       << timing_stats_.masks_received << "/"
       << timing_stats_.masks_backfilled << "/"
       << timing_stats_.masks_deferred << "/"
       << timing_stats_.masks_pending_peak
       << " sync_ms(avg/max)=" << std::fixed << std::setprecision(2)
      << (timing_stats_.sync_total_ms / processed) << "/" << timing_stats_.sync_max_ms
      << " reduce_ms(avg/max)=" << (timing_stats_.reduce_total_ms / processed) << "/" << timing_stats_.reduce_max_ms
      << " history_ms(avg/max)=" << (timing_stats_.history_total_ms / processed) << "/" << timing_stats_.history_max_ms
      << " fft_age_ms(avg/max)=" << (timing_stats_.fft_to_visualizer_total_ms / processed) << "/" << timing_stats_.fft_to_visualizer_max_ms
      << " preview_age_ms(avg/max)=" << (timing_stats_.preview_to_visualizer_total_ms / processed) << "/" << timing_stats_.preview_to_visualizer_max_ms
       << " compose_ms(avg/max)=" << (timing_stats_.compose_total_ms / rendered) << "/" << timing_stats_.compose_max_ms
      << " render_ms(avg/max)=" << (timing_stats_.render_total_ms / processed) << "/" << timing_stats_.render_max_ms
       << " drop_rate=" << ((timing_stats_.dropped_vis_stream_busy + timing_stats_.dropped_render_queue_busy) / seen);
    for (size_t channel_index = 0; channel_index < per_channel_timing_stats_.size(); ++channel_index) {
      auto& channel_stats = per_channel_timing_stats_[channel_index];
      if (channel_stats.frames_processed == 0) {
        continue;
      }
      const double channel_frames = static_cast<double>(std::max<uint64_t>(1, channel_stats.frames_processed));
      os << " ch" << channel_index
         << "_frames=" << channel_stats.frames_processed
         << " ch" << channel_index << "_last_frame=" << channel_stats.last_frame_number
         << " ch" << channel_index << "_fft_age_ms(avg/max)="
         << (channel_stats.fft_to_visualizer_total_ms / channel_frames) << "/"
         << channel_stats.fft_to_visualizer_max_ms
         << " ch" << channel_index << "_preview_age_ms(avg/max)="
         << (channel_stats.preview_to_visualizer_total_ms / channel_frames) << "/"
         << channel_stats.preview_to_visualizer_max_ms;
      channel_stats = PerChannelVisualTimingStats {};
    }
    HOLOSCAN_LOG_INFO("{}", os.str());
  };

  std::vector<PreviewChannelMailbox> snapshot_mailboxes;
  {
    std::lock_guard<std::mutex> lock(preview_mailbox_mutex());
    ensure_preview_mailbox_capacity(static_cast<size_t>(std::max(1, num_channels_.get())));
    auto& live_mailboxes = preview_mailboxes();
    snapshot_mailboxes.resize(live_mailboxes.size());
    for (size_t channel_index = 0; channel_index < live_mailboxes.size(); ++channel_index) {
      snapshot_mailboxes[channel_index].spectrogram_queue = std::move(live_mailboxes[channel_index].spectrogram_queue);
      snapshot_mailboxes[channel_index].mask_queue = std::move(live_mailboxes[channel_index].mask_queue);
    }
  }

  int effective_channel_filter = channel_filter_.get();
  if (effective_channel_filter == 0 && snapshot_mailboxes.size() > 1) {
    bool has_second_channel_data = false;
    for (size_t channel_index = 1; channel_index < snapshot_mailboxes.size(); ++channel_index) {
      const auto& mailbox = snapshot_mailboxes[channel_index];
      if (!mailbox.spectrogram_queue.empty() &&
          mailbox.spectrogram_queue.back().channel == static_cast<int>(channel_index)) {
        has_second_channel_data = true;
        break;
      }
    }
    if (has_second_channel_data) {
      effective_channel_filter = -1;
      if (!channel_filter_override_warning_emitted_.exchange(true)) {
        HOLOSCAN_LOG_WARN(
            "Visualizer received live data for multiple channels but channel_filter={} would suppress them; overriding to -1 for this run.",
            channel_filter_.get());
      }
    }
  }

  bool any_updates = false;
  std::vector<std::deque<DetectorMaskMessage>> residual_mask_queues(snapshot_mailboxes.size());
  uint64_t drained_mask_count = 0;
  uint64_t backfilled_mask_count = 0;
  uint64_t deferred_mask_count = 0;
  uint64_t pending_mask_peak = 0;
  {
    std::lock_guard<std::mutex> state_lock(channel_states_mutex_);
    if (channel_states_.size() < snapshot_mailboxes.size()) {
      channel_states_.resize(snapshot_mailboxes.size());
    }
    if (latest_rendered_frame_numbers_.size() < snapshot_mailboxes.size()) {
      latest_rendered_frame_numbers_.resize(snapshot_mailboxes.size(), 0);
    }

    for (size_t channel_index = 0; channel_index < snapshot_mailboxes.size(); ++channel_index) {
      auto& mailbox = snapshot_mailboxes[channel_index];
      pending_mask_peak = std::max<uint64_t>(pending_mask_peak, mailbox.mask_queue.size());
      if (mailbox.spectrogram_queue.empty()) {
        continue;
      }
      auto& state = channel_states_[channel_index];
      while (!mailbox.spectrogram_queue.empty() &&
             mailbox.spectrogram_queue.back().frame_number <= latest_rendered_frame_numbers_[channel_index]) {
        mailbox.spectrogram_queue.pop_back();
      }
      if (mailbox.spectrogram_queue.empty()) {
        continue;
      }

      // Coalesce backlog to the newest pending spectrogram so the live view stays current.
      auto spectrogram = std::move(mailbox.spectrogram_queue.back());
      mailbox.spectrogram_queue.clear();
        const uint64_t visualizer_enter_ns = steady_time_ns();
        if (spectrogram.channel != static_cast<int>(channel_index)) {
          HOLOSCAN_LOG_WARN("Dropping spectrogram from mailbox {} because message channel {} does not match frame {}",
                            channel_index,
                            spectrogram.channel,
                            spectrogram.frame_number);
          continue;
        }
        if (effective_channel_filter >= 0 && spectrogram.channel != effective_channel_filter) {
          continue;
        }
        if (spectrogram.channel < 0) {
          continue;
        }
        if (spectrogram.frame_number <= latest_rendered_frame_numbers_[channel_index]) {
          continue;
        }

        state.active = true;
        state.info.channel = spectrogram.channel;
        state.info.frame_number = static_cast<int64_t>(spectrogram.frame_number);
        state.info.center_frequency_hz = spectrogram.center_frequency_hz > 0.0 ? spectrogram.center_frequency_hz
                                                                               : center_frequency_hz_.get();
        const double resolved_span_hz = spectrogram.span_hz > 0.0 ? spectrogram.span_hz
            : (spectrogram.sample_rate_hz > 0.0 ? spectrogram.sample_rate_hz : span_hz_.get());
        state.info.span_hz = resolved_span_hz;
        state.info.resolution_hz = spectrogram.resolution_hz > 0.0 ? spectrogram.resolution_hz
            : (resolved_span_hz > 0.0 ? resolved_span_hz / static_cast<double>(std::max(1, spectrogram.width)) : 0.0);
        state.info.fft_size = fft_size_.get();
        state.info.fft_rows_per_frame = std::max(1, spectrogram.source_rows > 0 ? spectrogram.source_rows
                                            : spectrogram.height);
        state.info.displayed_fft_rows_per_frame = std::max(1, spectrogram.height);
        update_configured_display_metrics(state.info,
                          resolved_span_hz,
                          live_visualization_width(display_freq_bins_.get(), output_width_.get()),
                          state.info.fft_size,
                          rows_per_frame_.get(),
                          render_every_n_frames_.get());
        state.info.dino_chunk_rows = dino_chunk_rows_.get();
        state.info.dino_chunk_cols = dino_chunk_cols_.get();
        state.info.detector_label = detector_label_.get();

        bool patched_mask_history = false;
        while (!mailbox.mask_queue.empty() && mailbox.mask_queue.front().frame_number < spectrogram.frame_number) {
          auto late_mask = std::move(mailbox.mask_queue.front());
          mailbox.mask_queue.pop_front();
          ++drained_mask_count;
          if (late_mask.channel != static_cast<int>(channel_index) ||
              late_mask.channel != spectrogram.channel ||
              late_mask.pixels.empty()) {
            if (late_mask.channel != static_cast<int>(channel_index)) {
              HOLOSCAN_LOG_WARN("Dropping late mask from mailbox {} because message channel {} does not match frame {}",
                                channel_index,
                                late_mask.channel,
                                late_mask.frame_number);
            }
            continue;
          }
          state.latest_mask = {late_mask.width, late_mask.height, late_mask.pixels};
          state.latest_mask_frame_number = static_cast<int64_t>(late_mask.frame_number);
          const bool patched_late_mask = patch_history_mask_for_frame(
              state,
              static_cast<int64_t>(late_mask.frame_number) + static_cast<int64_t>(mask_frame_offset_.get()),
              late_mask.pixels,
              late_mask.width);
          backfilled_mask_count += patched_late_mask ? 1 : 0;
          patched_mask_history = patched_late_mask || patched_mask_history;
        }

        std::optional<DetectorMaskMessage> current_frame_mask;
        while (!mailbox.mask_queue.empty() && mailbox.mask_queue.front().frame_number == spectrogram.frame_number) {
          auto candidate_mask = std::move(mailbox.mask_queue.front());
          mailbox.mask_queue.pop_front();
          ++drained_mask_count;
          if (candidate_mask.channel != static_cast<int>(channel_index)) {
            HOLOSCAN_LOG_WARN("Dropping current-frame mask from mailbox {} because message channel {} does not match frame {}",
                              channel_index,
                              candidate_mask.channel,
                              candidate_mask.frame_number);
            continue;
          }
          current_frame_mask = std::move(candidate_mask);
        }

        const int snap_display_time_rows_requested =
            live_visualization_history_rows(display_time_rows_.get(), output_height_.get(), num_channels_.get());
        const int snap_display_time_rows = clamp_history_rows_to_budget(std::max(1, spectrogram.width),
                                                                        snap_display_time_rows_requested,
                                                                        history_memory_budget_mb_.get());
        append_spectrogram_history(state,
                                   spectrogram.pixels,
                                   std::max(1, spectrogram.width),
                                   std::max(1, spectrogram.height),
                                   snap_display_time_rows);
        state.current_psd_trace = compute_psd_trace(spectrogram.pixels,
                                                    std::max(1, spectrogram.width),
                                                    std::max(1, spectrogram.height));
        update_max_hold_trace(state.current_psd_trace, state.max_hold_trace);
        update_density_history(compute_density_trace_from_grayscale(spectrogram.pixels,
                                                                    std::max(1, spectrogram.width),
                                                                    std::max(1, spectrogram.height),
                                                                    red_limit_.get()),
                               state.density_trace,
                               state.density_frames_seen);

        state.overlay_available = patched_mask_history;
        if (current_frame_mask.has_value() &&
          current_frame_mask->channel == static_cast<int>(channel_index) &&
          current_frame_mask->channel == spectrogram.channel &&
            !current_frame_mask->pixels.empty()) {
          state.latest_mask = {current_frame_mask->width, current_frame_mask->height, current_frame_mask->pixels};
          state.latest_mask_frame_number = static_cast<int64_t>(current_frame_mask->frame_number);
          const bool patched_current_mask = patch_history_mask_for_frame(
              state,
              static_cast<int64_t>(current_frame_mask->frame_number) + static_cast<int64_t>(mask_frame_offset_.get()),
              current_frame_mask->pixels,
              current_frame_mask->width);
          backfilled_mask_count += patched_current_mask ? 1 : 0;
          state.overlay_available = patched_current_mask || state.overlay_available;
        }

        latest_rendered_frame_numbers_[channel_index] = spectrogram.frame_number;
        any_updates = true;
        {
          std::lock_guard<std::mutex> lock(timing_mutex_);
          ++timing_stats_.frames_seen;
          ++timing_stats_.frames_processed;
          const double fft_to_visualizer_ms = elapsed_ms(spectrogram.fft_emit_ts_ns, visualizer_enter_ns);
          const double preview_to_visualizer_ms = elapsed_ms(spectrogram.preview_emit_ts_ns, visualizer_enter_ns);
          timing_stats_.fft_to_visualizer_total_ms += fft_to_visualizer_ms;
          timing_stats_.fft_to_visualizer_max_ms = std::max(timing_stats_.fft_to_visualizer_max_ms,
                                                            fft_to_visualizer_ms);
          timing_stats_.preview_to_visualizer_total_ms += preview_to_visualizer_ms;
          timing_stats_.preview_to_visualizer_max_ms = std::max(timing_stats_.preview_to_visualizer_max_ms,
                                                                preview_to_visualizer_ms);
          if (per_channel_timing_stats_.size() <= channel_index) {
            per_channel_timing_stats_.resize(channel_index + 1);
          }
          auto& channel_stats = per_channel_timing_stats_[channel_index];
          ++channel_stats.frames_processed;
          channel_stats.last_frame_number = spectrogram.frame_number;
          channel_stats.fft_to_visualizer_total_ms += fft_to_visualizer_ms;
          channel_stats.fft_to_visualizer_max_ms = std::max(channel_stats.fft_to_visualizer_max_ms,
                                                            fft_to_visualizer_ms);
          channel_stats.preview_to_visualizer_total_ms += preview_to_visualizer_ms;
          channel_stats.preview_to_visualizer_max_ms = std::max(channel_stats.preview_to_visualizer_max_ms,
                                                                preview_to_visualizer_ms);
        }

      residual_mask_queues[channel_index] = std::move(mailbox.mask_queue);
      deferred_mask_count += residual_mask_queues[channel_index].size();
    }
  }

  {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    timing_stats_.masks_received += drained_mask_count;
    timing_stats_.masks_backfilled += backfilled_mask_count;
    timing_stats_.masks_deferred += deferred_mask_count;
    timing_stats_.masks_pending_peak = std::max<uint64_t>(timing_stats_.masks_pending_peak, pending_mask_peak);
  }

  if (!any_updates) {
    maybe_emit_timing_summary();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(preview_mailbox_mutex());
    ensure_preview_mailbox_capacity(snapshot_mailboxes.size());
    auto& live_mailboxes = preview_mailboxes();
    for (size_t channel_index = 0; channel_index < residual_mask_queues.size(); ++channel_index) {
      auto& residual_masks = residual_mask_queues[channel_index];
      if (residual_masks.empty()) {
        continue;
      }
      auto& live_masks = live_mailboxes[channel_index].mask_queue;
      while (!residual_masks.empty()) {
        live_masks.push_front(std::move(residual_masks.back()));
        residual_masks.pop_back();
      }
      while (live_masks.size() > kPreviewMailboxMaxDepth) {
        live_masks.pop_back();
      }
    }
  }

  int composite_width = 0;
  int composite_height = 0;
  const auto compose_start = std::chrono::steady_clock::now();
  std::vector<uint8_t> composed;
  {
    std::lock_guard<std::mutex> state_lock(channel_states_mutex_);
    composed = compose_visualization_rgb(channel_states_,
                                         blue_limit_.get(),
                                         red_limit_.get(),
                                         overlay_alpha_.get(),
                                         visualization_overlay_enabled(),
                                         output_width_.get(),
                                         output_height_.get(),
                                         composite_width,
                                         composite_height,
                                         dropped_frames_,
                                         total_frames_);
  }
  int emitted_width = composite_width;
  int emitted_height = composite_height;
  const auto emitted = scale_rgb_to_fit(composed,
                                        composite_width,
                                        composite_height,
                                        output_width_.get(),
                                        output_height_.get(),
                                        emitted_width,
                                        emitted_height);
  const auto compose_end = std::chrono::steady_clock::now();
  try {
    auto output_entity = create_rgb_entity(context,
                                           emitted,
                                           emitted_width,
                                           emitted_height,
                                           tensor_name_.get());
    op_output.emit(output_entity, "outputs");
  } catch (const std::exception& ex) {
    HOLOSCAN_LOG_ERROR("Visualizer failed to emit composed frame: {}", ex.what());
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Visualizer failed to emit composed frame due to unknown error");
  }
  {
    std::lock_guard<std::mutex> lock(timing_mutex_);
    update_timing_accumulator(std::chrono::duration<double, std::milli>(compose_end - compose_start).count(),
                              timing_stats_.compose_total_ms,
                              timing_stats_.compose_max_ms);
    if (any_updates) {
      ++timing_stats_.frames_rendered;
    }
  }
  maybe_emit_timing_summary();
}


void OfflinePgmReplayOp::setup(OperatorSpec& spec) {
  spec.output<gxf::Entity>("outputs");
  spec.param(num_channels_, "num_channels", "Num Channels", "Number of channels shown in the analyzer view.", 2);
  spec.param(history_frames_, "history_frames", "History Frames", "Number of consecutive spectrogram frames retained in the display history.", 5);
  spec.param(directory_, "directory", "Directory", "Directory containing spectrogram PGM frames.");
  spec.param(mask_directory_,
             "mask_directory",
             "Mask Directory",
             "Directory containing detector mask PGM frames to overlay when available.",
             std::string("/workspace/dino_masks"));
  spec.param(frame_rate_, "frame_rate", "Frame Rate", "Replay rate in frames per second.", 6.0);
  spec.param(repeat_, "repeat", "Repeat", "Loop the directory when the final frame is reached.", true);
  spec.param(channel_filter_,
             "channel_filter",
             "Channel Filter",
             "Only replay files matching ch<channel>_. Use -1 to replay every frame.",
             -1);
  spec.param(tensor_name_,
             "tensor_name",
             "Tensor Name",
             "Holoviz tensor name for the replayed spectrogram image.",
             std::string("spectrogram"));
  spec.param(blue_limit_, "blue_limit", "Blue Limit", "Lower color scale clamp in normalized units.", 0.10f);
  spec.param(red_limit_, "red_limit", "Red Limit", "Upper color scale clamp in normalized units.", 0.92f);
  spec.param(overlay_alpha_,
             "overlay_alpha",
             "Overlay Alpha",
             "Alpha used when blending the detector mask onto the spectrogram.",
             0.38f);
  spec.param(overlay_enable_,
             "overlay_enable",
             "Overlay Enable",
             "Enable or disable the detection overlay layer in the composed visualization.",
             true);
  spec.param(detector_label_,
             "detector_label",
             "Detector Label",
             "Human-readable detector label used in the visualization UI.",
             std::string("Dinov3"));
  spec.param(center_frequency_hz_, "center_frequency_hz", "Center Frequency", "Center frequency for display in Hz.", 0.0);
  spec.param(span_hz_, "span_hz", "Span Hz", "Frequency span shown on calibrated plot axes in Hz.", 0.0);
  spec.param(fft_size_, "fft_size", "FFT Size", "FFT size shown in analyzer readouts.", 20480);
  spec.param(dino_chunk_rows_, "dino_chunk_rows", "DINO Chunk Rows", "DINO chunk height shown in readouts.", 256);
  spec.param(dino_chunk_cols_, "dino_chunk_cols", "DINO Chunk Cols", "DINO chunk width shown in readouts.", 512);
}

void OfflinePgmReplayOp::initialize() {
  Operator::initialize();

  initialize_visualization_overlay_state(overlay_enable_.get());

  frames_ = list_pgm_frames(directory_.get(), channel_filter_.get());
  if (frames_.empty()) {
    throw std::runtime_error("No .pgm spectrogram frames found in " + directory_.get());
  }

  channel_states_.assign(static_cast<size_t>(std::max(1, num_channels_.get())), {});

  next_frame_index_ = 0;
  next_deadline_ = std::chrono::steady_clock::now();
}

void OfflinePgmReplayOp::compute(InputContext&, OutputContext& op_output, ExecutionContext& context) {
  if (frames_.empty()) {
    return;
  }

  if (next_frame_index_ >= frames_.size()) {
    if (!repeat_.get()) {
      auto replay_condition = condition<BooleanCondition>("replay_active");
      if (replay_condition) {
        replay_condition->disable_tick();
      }
      return;
    }
    next_frame_index_ = 0;
  }

  std::vector<uint8_t> grayscale;
  int width = 0;
  int height = 0;
  const auto frame_path = frames_[next_frame_index_++];

  if (!load_pgm_file(frame_path, grayscale, width, height)) {
    throw std::runtime_error("Failed to load spectrogram frame: " + frame_path.string());
  }

  int channel = -1;
  uint64_t frame_number = 0;
  int rows = 0;
  int cols = 0;
  parse_recorded_pgm_name(frame_path.filename().string(), "spectrogram", channel, frame_number, rows, cols);

  OfflinePgmFrame mask_frame;
  const auto mask_path = find_matching_recorded_pgm(mask_directory_.get(), "dino_mask", channel, frame_number);
  const bool has_mask = !mask_path.empty() && load_offline_pgm_frame(mask_path, mask_frame);
  if (channel >= 0 && channel_states_.size() < static_cast<size_t>(std::max(num_channels_.get(), channel + 1))) {
    channel_states_.resize(static_cast<size_t>(std::max(num_channels_.get(), channel + 1)));
  }
  const size_t state_index = static_cast<size_t>(std::max(0, channel));
  auto& state = channel_states_[state_index];
  state.info.frame_number = static_cast<int64_t>(frame_number);
  append_spectrogram_history(state, grayscale, width, height, history_frames_.get() * height);
  state.current_psd_trace = compute_psd_trace(grayscale, width, height);
  update_max_hold_trace(state.current_psd_trace, state.max_hold_trace);
  const auto current_density = has_mask ? compute_density_trace(&mask_frame)
                                        : compute_density_trace_from_grayscale(grayscale, width, height, red_limit_.get());
  update_density_history(current_density, state.density_trace, state.density_frames_seen);
  state.overlay_available = has_mask;
  state.latest_mask = has_mask ? mask_frame : OfflinePgmFrame{};
  state.latest_mask_frame_number = has_mask ? static_cast<int64_t>(frame_number) : -1;
  if (has_mask) {
    const auto reduced_mask = reduce_mask_to_history_rows(mask_frame, width, height);
    patch_history_mask_for_frame(state, static_cast<int64_t>(frame_number), reduced_mask, width);
  }
  state.active = true;
  state.info.channel = channel;
  state.info.center_frequency_hz = center_frequency_hz_.get();
  state.info.span_hz = span_hz_.get();
  state.info.resolution_hz = state.info.span_hz > 0.0 ? state.info.span_hz / static_cast<double>(std::max(1, width)) : 0.0;
  state.info.fft_size = fft_size_.get();
    state.info.fft_rows_per_frame = std::max(1, height);
    state.info.displayed_fft_rows_per_frame = std::max(1, height);
  const int replay_display_width = std::max(1, width);
  const int replay_rows_per_frame = std::max(1, height);
  update_configured_display_metrics(state.info,
                                    state.info.span_hz,
                                    replay_display_width,
                                    state.info.fft_size,
                                    replay_rows_per_frame,
                                    1);
  state.info.dino_chunk_rows = dino_chunk_rows_.get();
  state.info.dino_chunk_cols = dino_chunk_cols_.get();
  state.info.detector_label = detector_label_.get();
  state.info.overlay_available = has_mask;
  state.info.title = "USRP WIDEBAND";
  state.info.subtitle = "OFFLINE REPLAY";

  int composite_width = 0;
  int composite_height = 0;
  // auto composed = compose_visualization_rgb(channel_states_,
  //                                           blue_limit_.get(),
  //                                           red_limit_.get(),
  //                                           overlay_alpha_.get(),
  //                                           composite_width,
  //                                           composite_height);

  auto composed = compose_visualization_rgb(channel_states_,
                                            blue_limit_.get(),
                                            red_limit_.get(),
                                            overlay_alpha_.get(),
                                            visualization_overlay_enabled(),
                                            width,
                                            std::max(256, height * history_frames_.get()),
                                            composite_width,
                                            composite_height,
                                            0,
                                            0);

  auto output_entity = create_rgb_entity(context,
                                         composed,
                                         composite_width,
                                         composite_height,
                                         tensor_name_.get());
  op_output.emit(output_entity, "outputs");

  if (frame_rate_.get() > 0.0) {
    next_deadline_ += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(1.0 / frame_rate_.get()));
    std::this_thread::sleep_until(next_deadline_);
  }
}

std::vector<HolovizOp::InputSpec> make_spectrogram_input_specs(const std::string& tensor_name) {
  std::vector<HolovizOp::InputSpec> specs;
  auto& spectrogram_spec = specs.emplace_back();
  spectrogram_spec.tensor_name_ = tensor_name;
  spectrogram_spec.type_ = HolovizOp::InputType::COLOR;
  spectrogram_spec.priority_ = 0;
  return specs;
}

std::vector<std::filesystem::path> list_offline_pgm_frames(const std::filesystem::path& directory,
                                                           int channel_filter) {
  return list_pgm_frames(directory, channel_filter);
}

bool load_offline_pgm_frame(const std::filesystem::path& path, OfflinePgmFrame& frame) {
  return load_pgm_file(path, frame.pixels, frame.width, frame.height);
}

bool parse_recorded_pgm_name(const std::string& filename,
                             const std::string& prefix,
                             int& channel,
                             uint64_t& frame_number,
                             int& rows,
                             int& cols) {
  unsigned long long parsed_frame = 0;
  unsigned long long timestamp_ms = 0;
  const std::string format = prefix + "_ch%d_f%llu_%llu_%dx%d.pgm";
  const int parsed = std::sscanf(filename.c_str(),
                                 format.c_str(),
                                 &channel,
                                 &parsed_frame,
                                 &timestamp_ms,
                                 &rows,
                                 &cols);
  if (parsed != 5) {
    channel = -1;
    frame_number = 0;
    rows = 0;
    cols = 0;
    return false;
  }
  frame_number = static_cast<uint64_t>(parsed_frame);
  return true;
}

std::filesystem::path find_matching_recorded_pgm(const std::filesystem::path& directory,
                                                 const std::string& prefix,
                                                 int channel,
                                                 uint64_t frame_number) {
  if (channel < 0 || !std::filesystem::exists(directory)) {
    return {};
  }

  const auto candidates = list_pgm_frames(directory, channel);
  for (const auto& path : candidates) {
    int candidate_channel = -1;
    uint64_t candidate_frame = 0;
    int rows = 0;
    int cols = 0;
    if (!parse_recorded_pgm_name(path.filename().string(), prefix, candidate_channel, candidate_frame, rows, cols)) {
      continue;
    }
    if (candidate_channel == channel && candidate_frame == frame_number) {
      return path;
    }
  }

  return {};
}

std::vector<uint8_t> colorize_grayscale_spectrogram(const std::vector<uint8_t>& grayscale,
                                                    float blue_limit,
                                                    float red_limit) {
  return grayscale_to_rgb(grayscale, blue_limit, red_limit);
}

std::vector<float> compute_psd_trace(const std::vector<uint8_t>& grayscale, int width, int height) {
  std::vector<float> trace(static_cast<size_t>(width), 0.0f);
  for (int col = 0; col < width; ++col) {
    float sum = 0.0f;
    for (int row = 0; row < height; ++row) {
      sum += static_cast<float>(grayscale[static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)]) / 255.0f;
    }
    trace[static_cast<size_t>(col)] = std::clamp(sum / std::max(1, height), 0.0f, 1.0f);
  }
  return trace;
}

std::vector<float> compute_density_trace(const OfflinePgmFrame* mask_frame) {
  if (mask_frame == nullptr || mask_frame->width <= 0 || mask_frame->height <= 0) {
    return {};
  }
  std::vector<float> density(static_cast<size_t>(mask_frame->width), 0.0f);
  for (int col = 0; col < mask_frame->width; ++col) {
    int active = 0;
    for (int row = 0; row < mask_frame->height; ++row) {
      active += mask_frame->pixels[static_cast<size_t>(row) * static_cast<size_t>(mask_frame->width) + static_cast<size_t>(col)] > 0 ? 1 : 0;
    }
    density[static_cast<size_t>(col)] = static_cast<float>(active) / static_cast<float>(mask_frame->height);
  }
  return density;
}

std::vector<float> compute_density_trace_from_grayscale(const std::vector<uint8_t>& grayscale,
                                                        int width,
                                                        int height,
                                                        float threshold) {
  std::vector<float> density(static_cast<size_t>(width), 0.0f);
  const uint8_t threshold_value = static_cast<uint8_t>(std::clamp(threshold, 0.0f, 1.0f) * 255.0f);
  for (int col = 0; col < width; ++col) {
    int active = 0;
    for (int row = 0; row < height; ++row) {
      active += grayscale[static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col)] >= threshold_value ? 1 : 0;
    }
    density[static_cast<size_t>(col)] = static_cast<float>(active) / static_cast<float>(std::max(1, height));
  }
  return density;
}

void update_max_hold_trace(const std::vector<float>& current_trace, std::vector<float>& max_hold_trace) {
  if (max_hold_trace.size() != current_trace.size()) {
    max_hold_trace.assign(current_trace.size(), 0.0f);
  }
  for (size_t i = 0; i < current_trace.size(); ++i) {
    max_hold_trace[i] = std::max(max_hold_trace[i], current_trace[i]);
  }
}

void update_density_history(const std::vector<float>& current_density,
                            std::vector<float>& density_history,
                            size_t& density_frames_seen) {
  if (current_density.empty()) {
    return;
  }
  if (density_history.size() != current_density.size()) {
    density_history.assign(current_density.size(), 0.0f);
    density_frames_seen = 0;
  }
  for (size_t i = 0; i < current_density.size(); ++i) {
    density_history[i] = (density_history[i] * static_cast<float>(density_frames_seen) + current_density[i]) /
                         static_cast<float>(density_frames_seen + 1);
  }
  ++density_frames_seen;
}

void ensure_history_capacity(ChannelVisualizationState& state, int width, int max_rows) {
  const int clamped_rows = std::max(1, max_rows);
  if (state.history_width == width && state.history_capacity_rows == clamped_rows) {
    return;
  }

  state.history_width = width;
  state.history_capacity_rows = clamped_rows;
  state.history_valid_rows = 0;
  state.history_write_row = 0;
  state.history_grayscale.assign(static_cast<size_t>(width) * static_cast<size_t>(clamped_rows), 0);
  state.history_mask.assign(static_cast<size_t>(width) * static_cast<size_t>(clamped_rows), 0);
  state.history_row_frame_numbers.assign(static_cast<size_t>(clamped_rows), -1);
  state.history_row_indices_within_frame.assign(static_cast<size_t>(clamped_rows), -1);
}

void write_history_rows(ChannelVisualizationState& state,
                        const std::vector<uint8_t>& rows,
                        int width,
                        int row_count,
                        int64_t frame_number) {
  if (row_count <= 0 || width <= 0) {
    return;
  }

  ensure_history_capacity(state, width, state.history_capacity_rows > 0 ? state.history_capacity_rows : row_count);
  const size_t row_bytes = static_cast<size_t>(width);
  for (int row = 0; row < row_count; ++row) {
    const int dst_row = state.history_write_row;
    const size_t dst_offset = static_cast<size_t>(dst_row) * row_bytes;
    const size_t src_offset = static_cast<size_t>(row) * row_bytes;
    std::copy(rows.begin() + static_cast<std::ptrdiff_t>(src_offset),
              rows.begin() + static_cast<std::ptrdiff_t>(src_offset + row_bytes),
              state.history_grayscale.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    std::fill(state.history_mask.begin() + static_cast<std::ptrdiff_t>(dst_offset),
              state.history_mask.begin() + static_cast<std::ptrdiff_t>(dst_offset + row_bytes),
              0);
    state.history_row_frame_numbers[static_cast<size_t>(dst_row)] = frame_number;
    state.history_row_indices_within_frame[static_cast<size_t>(dst_row)] = row;
    state.history_write_row = (state.history_write_row + 1) % state.history_capacity_rows;
    state.history_valid_rows = std::min(state.history_valid_rows + 1, state.history_capacity_rows);
  }
}

std::vector<uint8_t> materialize_history_rows(const std::vector<uint8_t>& ring,
                                              int width,
                                              int capacity_rows,
                                              int valid_rows,
                                              int write_row) {
  const int output_rows = std::max(1, capacity_rows);
  std::vector<uint8_t> materialized(static_cast<size_t>(width) * static_cast<size_t>(output_rows), 0);
  if (width <= 0 || capacity_rows <= 0 || valid_rows <= 0) {
    return materialized;
  }

  const size_t row_bytes = static_cast<size_t>(width);
  const int leading_blank_rows = output_rows - valid_rows;
  const int oldest_row = valid_rows == capacity_rows ? write_row : 0;
  for (int row = 0; row < valid_rows; ++row) {
    const int src_row = (oldest_row + row) % capacity_rows;
    const size_t src_offset = static_cast<size_t>(src_row) * row_bytes;
    const size_t dst_offset = static_cast<size_t>(leading_blank_rows + row) * row_bytes;
    std::copy(ring.begin() + static_cast<std::ptrdiff_t>(src_offset),
              ring.begin() + static_cast<std::ptrdiff_t>(src_offset + row_bytes),
              materialized.begin() + static_cast<std::ptrdiff_t>(dst_offset));
  }
  return materialized;
}

std::vector<uint8_t> reduce_mask_to_history_rows(const holoscan::ops::OfflinePgmFrame& mask_frame,
                                                 int dst_width,
                                                 int dst_rows) {
  if (mask_frame.width <= 0 || mask_frame.height <= 0 || mask_frame.pixels.empty() ||
      dst_width <= 0 || dst_rows <= 0) {
    return {};
  }

  std::vector<uint8_t> reduced(static_cast<size_t>(dst_width) * static_cast<size_t>(dst_rows), 0);
  for (int row = 0; row < dst_rows; ++row) {
    const int src_row_start = (row * mask_frame.height) / dst_rows;
    const int src_row_end = std::max(src_row_start + 1, ((row + 1) * mask_frame.height) / dst_rows);
    for (int col = 0; col < dst_width; ++col) {
      const int src_col_start = (col * mask_frame.width) / dst_width;
      const int src_col_end = std::max(src_col_start + 1, ((col + 1) * mask_frame.width) / dst_width);
      int active = 0;
      int count = 0;
      for (int src_row = src_row_start; src_row < src_row_end; ++src_row) {
        for (int src_col = src_col_start; src_col < src_col_end; ++src_col) {
          active += mask_frame.pixels[static_cast<size_t>(src_row) * static_cast<size_t>(mask_frame.width) +
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

std::vector<uint8_t> cleanup_live_mask_for_display(const std::vector<uint8_t>& reduced_mask,
                                                   int width,
                                                   int height) {
  if (width <= 0 || height <= 0 || reduced_mask.size() != static_cast<size_t>(width) * static_cast<size_t>(height)) {
    return reduced_mask;
  }

  struct ComponentStats {
    int min_row;
    int max_row;
    int min_col;
    int max_col;
    int filled = 0;
    uint8_t peak_value = 0;
  };

  std::vector<uint8_t> visited(reduced_mask.size(), 0);
  std::vector<uint8_t> cleaned(reduced_mask.size(), 0);
  const std::array<std::pair<int, int>, 4> neighbors{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};

  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const size_t seed_index = static_cast<size_t>(row) * static_cast<size_t>(width) + static_cast<size_t>(col);
      if (visited[seed_index] || reduced_mask[seed_index] == 0) {
        continue;
      }

      ComponentStats stats{row, row, col, col, 0, 0};
      std::queue<std::pair<int, int>> pending;
      pending.push({row, col});
      visited[seed_index] = 1;

      while (!pending.empty()) {
        const auto [current_row, current_col] = pending.front();
        pending.pop();
        const size_t current_index = static_cast<size_t>(current_row) * static_cast<size_t>(width) +
                                     static_cast<size_t>(current_col);
        stats.min_row = std::min(stats.min_row, current_row);
        stats.max_row = std::max(stats.max_row, current_row);
        stats.min_col = std::min(stats.min_col, current_col);
        stats.max_col = std::max(stats.max_col, current_col);
        stats.peak_value = std::max(stats.peak_value, reduced_mask[current_index]);
        ++stats.filled;

        for (const auto& [delta_row, delta_col] : neighbors) {
          const int next_row = current_row + delta_row;
          const int next_col = current_col + delta_col;
          if (next_row < 0 || next_row >= height || next_col < 0 || next_col >= width) {
            continue;
          }
          const size_t next_index = static_cast<size_t>(next_row) * static_cast<size_t>(width) +
                                    static_cast<size_t>(next_col);
          if (visited[next_index] || reduced_mask[next_index] == 0) {
            continue;
          }
          visited[next_index] = 1;
          pending.push({next_row, next_col});
        }
      }

      const int freq_span = stats.max_col - stats.min_col + 1;
      const int time_span = stats.max_row - stats.min_row + 1;
      const int bbox_area = freq_span * time_span;
      const float density = bbox_area > 0 ? static_cast<float>(stats.filled) / static_cast<float>(bbox_area) : 0.0f;
      if (stats.filled < kLiveMaskGroupingMinComponentSize ||
          freq_span < kLiveMaskGroupingMinFreqSpan ||
          time_span < kLiveMaskGroupingMinTimeSpan ||
          density < kLiveMaskGroupingMinDensity) {
        continue;
      }

      for (int fill_row = stats.min_row; fill_row <= stats.max_row; ++fill_row) {
        for (int fill_col = stats.min_col; fill_col <= stats.max_col; ++fill_col) {
          cleaned[static_cast<size_t>(fill_row) * static_cast<size_t>(width) + static_cast<size_t>(fill_col)] =
              stats.peak_value;
        }
      }
    }
  }

  return cleaned;
}

bool patch_history_mask_for_frame(ChannelVisualizationState& state,
                                  int64_t frame_number,
                                  const std::vector<uint8_t>& mask_rows,
                                  int width) {
  if (frame_number < 0 || width <= 0 || state.history_width != width || mask_rows.size() < static_cast<size_t>(width)) {
    return false;
  }

  const size_t row_bytes = static_cast<size_t>(width);
  const int mask_row_count = static_cast<int>(mask_rows.size() / row_bytes);
  bool patched_any_row = false;
  for (int row = 0; row < state.history_capacity_rows; ++row) {
    if (state.history_row_frame_numbers[static_cast<size_t>(row)] != frame_number) {
      continue;
    }
    const int frame_row_index = state.history_row_indices_within_frame[static_cast<size_t>(row)];
    if (frame_row_index < 0 || frame_row_index >= mask_row_count) {
      continue;
    }
    const size_t src_offset = static_cast<size_t>(frame_row_index) * row_bytes;
    const size_t dst_offset = static_cast<size_t>(row) * row_bytes;
    std::copy(mask_rows.begin() + static_cast<std::ptrdiff_t>(src_offset),
              mask_rows.begin() + static_cast<std::ptrdiff_t>(src_offset + row_bytes),
              state.history_mask.begin() + static_cast<std::ptrdiff_t>(dst_offset));
    patched_any_row = true;
  }
  return patched_any_row;
}

void append_spectrogram_history(ChannelVisualizationState& state,
                                const std::vector<uint8_t>& grayscale,
                                int width,
                                int height,
                                int max_rows) {
  state.latest_frame_height = height;
  ensure_history_capacity(state, width, max_rows);
  write_history_rows(state, grayscale, width, height, state.info.frame_number);
}

std::vector<uint8_t> compose_visualization_rgb(const std::vector<ChannelVisualizationState>& channels,
                                               float blue_limit,
                                               float red_limit,
                                               float overlay_alpha,
                                               bool overlay_enabled,
                                               int panel_width,
                                               int panel_height,
                                               int& output_width,
                                               int& output_height,
                                               uint64_t dropped_frames,
                                               uint64_t total_frames) {

  // Channel accent colors — blue for CH0, purple for CH1
  const RgbColor kCh0Accent{84, 196, 255};
  const RgbColor kCh1Accent{200, 124, 255};
  const RgbColor kAccents[2] = {kCh0Accent, kCh1Accent};

  // Badge/status colors
  const RgbColor kGreen{80, 200, 120};
  const RgbColor kOrange{255, 130, 48};
  const RgbColor kYellow{255, 212, 89};
  const RgbColor kDimBlue{74, 96, 128};
  const bool render_canvas_chrome = !visualization_full_ui_enabled();

  std::vector<const ChannelVisualizationState*> active_channel_states;
  active_channel_states.reserve(channels.size());
  for (const auto& channel : channels) {
    if (channel.active) {
      active_channel_states.push_back(&channel);
    }
  }

  const bool has_active_channels = !active_channel_states.empty();
  const int active_channels = std::max(1, static_cast<int>(active_channel_states.size()));
  output_width = std::max(1, panel_width);
  output_height = std::max(1, panel_height);

  const int columns = std::min(2, active_channels);
  const int rows = std::max(1, (active_channels + columns - 1) / columns);
  const int content_x = kPanelPadding;
  const int content_y = kHeaderHeight + kPanelPadding;
  const int content_width = std::max(128, output_width - kSidebarWidth - kPanelPadding * 3);
  const int content_height = std::max(160, output_height - kHeaderHeight - kFooterHeight - kPanelPadding * 2);
  const int main_width = std::max(128,
                                  (content_width - (columns - 1) * kPanelPadding) / std::max(1, columns));
  const int panel_height_budget = std::max(180,
                                           (content_height - (rows - 1) * kPanelPadding) / std::max(1, rows));
  const int grid_width = columns * main_width + (columns - 1) * kPanelPadding;
  const int grid_height = rows * panel_height_budget + (rows - 1) * kPanelPadding;
  const int grid_x = content_x + std::max(0, (content_width - grid_width) / 2);
  const int grid_y = content_y + std::max(0, (content_height - grid_height) / 2);
  const int sidebar_x = output_width - kSidebarWidth - kPanelPadding;
  const int sidebar_y = kHeaderHeight + kPanelPadding;
  const int sidebar_height = std::max(80, output_height - sidebar_y - kFooterHeight - kPanelPadding);

  const int channel_header_band = 28;
  const int section_gap = std::max(10, kPanelPadding / 2);
  const int mask_section_gap = section_gap + 18;
  const int confidence_gap = 0;
  const int confidence_height = 0;
  const auto panel_heights = compute_channel_panel_heights(output_height, active_channels);
  const int channel_psd_height = panel_heights.psd;
  const int channel_mask_height = panel_heights.mask;
  const int channel_heat_height = panel_heights.heat;

  VisualizationUiState ui_state;
  ui_state.canvas_width = output_width;
  ui_state.canvas_height = output_height;
  ui_state.overlay_enabled = overlay_enabled;
  ui_state.blue_limit = blue_limit;
  ui_state.red_limit = red_limit;
  ui_state.dropped_frames = dropped_frames;
  ui_state.total_frames = total_frames;
  ui_state.title = "USRP WIDEBAND";
  ui_state.subtitle = "REAL TIME SIGNAL DETECTION";
  ui_state.header_rect = normalized_rect(0, 0, output_width, kHeaderHeight, output_width, output_height);
  ui_state.content_rect = normalized_rect(content_x,
                                          content_y,
                                          content_width,
                                          content_height,
                                          output_width,
                                          output_height);
  ui_state.sidebar_rect = normalized_rect(sidebar_x,
                                          sidebar_y,
                                          kSidebarWidth,
                                          sidebar_height,
                                          output_width,
                                          output_height);

  std::vector<uint8_t> canvas(static_cast<size_t>(output_width) * static_cast<size_t>(output_height) * 3, 0);
  fill_vertical_gradient(canvas, output_width, output_height, {7, 10, 18}, {14, 19, 29});

  // fill_rect(canvas, output_width, output_height, 0, 0, output_width, kHeaderHeight, {8, 12, 18});
  // fill_rect(canvas, output_width, output_height, 0, kHeaderHeight - 2, output_width, 2, {124, 198, 255});
  // draw_text(canvas, output_width, output_height, 18, 10, "USRP WIDEBAND", {232, 236, 241}, 2);
  // draw_text(canvas, output_width, output_height, 18, 24, "TWO CHANNEL ANALYZER", {156, 173, 192}, 1);

  // Header background
  fill_rect(canvas, output_width, output_height, 0, 0, output_width, kHeaderHeight, {8, 13, 20});
  fill_rect(canvas, output_width, output_height, 0, kHeaderHeight - 2, output_width, 2, {124, 198, 255});

  // Title + subtitle
  if (render_canvas_chrome) {
    draw_text(canvas, output_width, output_height, 18, 8, "USRP WIDEBAND", {232, 236, 241}, 2);
    draw_text(canvas, output_width, output_height, 18, 26, "REAL TIME SIGNAL DETECTION", {124, 198, 255}, 1);
  }

  // Divider after title
  fill_rect(canvas, output_width, output_height, 210, 10, 1, 30, {30, 45, 62});

  // Stat boxes — CH-0 RATE, CH-1 RATE, CF, SPAN, FFT
  auto draw_stat_box = [&](int x, int y, const std::string& label, const std::string& value, const RgbColor& val_color) {
    fill_rect(canvas, output_width, output_height, x, y, 88, 32, {10, 21, 34});
    draw_rect_outline(canvas, output_width, output_height, x, y, 88, 32, {26, 42, 58}, 1);
    draw_text(canvas, output_width, output_height, x + 4, y + 4, label, {122, 143, 168}, 1);
    draw_text(canvas, output_width, output_height, x + 4, y + 18, value, val_color, 1);
  };

  // Get rates from channel states if available
  std::string ch0_rate = "500.0 MSps";
  std::string ch1_rate = "500.0 MSps";

  if (render_canvas_chrome) {
    draw_stat_box(218, 10, "CH-0 RATE", ch0_rate, kGreen);
    draw_stat_box(314, 10, "CH-1 RATE", ch1_rate, kGreen);
    draw_stat_box(410, 10, "CENTER FREQ", "2400.0 MHZ", kYellow);
    draw_stat_box(506, 10, "SPAN", "500 MHZ", {232, 236, 241});
    draw_stat_box(602, 10, "FFT SIZE", "20480", {232, 236, 241});
  }

  std::string active_detector_label = "Dinov3";
  for (const auto* channel : active_channel_states) {
    if (!channel->info.detector_label.empty()) {
      active_detector_label = channel->info.detector_label;
      break;
    }
  }

  // Badges — LIVE, DUAL CH, detector
  auto draw_badge = [&](int x, int y, const std::string& text, const RgbColor& col) {
    const int w = text_pixel_width(text, 1) + 12;
    fill_rect(canvas, output_width, output_height, x, y, w, 16, {10, 21, 34});
    draw_rect_outline(canvas, output_width, output_height, x, y, w, 16, col, 1);
    draw_text(canvas, output_width, output_height, x + 6, y + 4, text, col, 1);
  };

  const int badge_y = 18;
  const int badge_x_start = output_width - kSidebarWidth - kPanelPadding - 180;
  if (render_canvas_chrome) {
    draw_badge(badge_x_start,       badge_y, "LIVE",    kGreen);
    draw_badge(badge_x_start + 48,  badge_y, "DUAL CH", kCh0Accent);
    draw_badge(badge_x_start + 108, badge_y, active_detector_label, kYellow);
  }
  ui_state.detector_label = active_detector_label;

  for (int channel_index = 0; channel_index < active_channels; ++channel_index) {
    const auto& channel = has_active_channels
        ? *active_channel_states[static_cast<size_t>(channel_index)]
        : channels.front();
    const int column_index = channel_index % columns;
    const int row_index = channel_index / columns;
    const int panel_x = grid_x + column_index * (main_width + kPanelPadding);
    const int panel_y = grid_y + row_index * (panel_height_budget + kPanelPadding);
    const int psd_y = panel_y + channel_header_band;
    const int heatmap_y = psd_y + channel_psd_height + section_gap;
    const int mask_y = heatmap_y + channel_heat_height + mask_section_gap;


    // Channel accent color
    const RgbColor& accent = kAccents[std::min(channel_index, 1)];

    // Panel header bar with channel label, freq, rate
    fill_rect(canvas, output_width, output_height, panel_x - 8, panel_y, main_width + 16, 20, {10, 18, 32});
    draw_rect_outline(canvas, output_width, output_height, panel_x - 8, panel_y, main_width + 16, 20, {26, 42, 58}, 1);
    // Left accent stripe
    fill_rect(canvas, output_width, output_height, panel_x - 8, panel_y, 3, 20, accent);
    // CH label
    if (render_canvas_chrome) {
      draw_text(canvas, output_width, output_height, panel_x, panel_y + 6, std::string("CH-") + std::to_string(channel.info.channel >= 0 ? channel.info.channel : channel_index), accent, 1);
    }
    // Freq label
    std::ostringstream freq_ss;
    freq_ss << std::fixed << std::setprecision(3) << (channel.info.center_frequency_hz / 1.0e6) << " MHZ";
    if (render_canvas_chrome) {
      draw_text(canvas, output_width, output_height, panel_x + 60, panel_y + 6, freq_ss.str(), {122, 143, 168}, 1);
    }

    const double span_hz = resolved_display_span_hz(channel, main_width);
    const std::string freq_min_label = span_hz > 0.0
                         ? format_frequency_label(channel.info.center_frequency_hz - span_hz * 0.5)
                         : std::string("LOW");
    const std::string freq_max_label = span_hz > 0.0
                         ? format_frequency_label(channel.info.center_frequency_hz + span_hz * 0.5)
                         : std::string("HIGH");
    const int history_width = channel.history_width > 0 ? channel.history_width : main_width;
    const int history_rows = std::max(1,
              channel.history_capacity_rows > 0 ? channel.history_capacity_rows
                       : channel.history_valid_rows);
    const std::string oldest_time_label = std::to_string(history_rows);
    const std::string newest_time_label = "0";

    VisualizationChannelUiState channel_ui;
    channel_ui.active = channel.active;
    channel_ui.channel = channel.info.channel >= 0 ? channel.info.channel : channel_index;
    channel_ui.header_rect = normalized_rect(panel_x - 8, panel_y, main_width + 16, 20, output_width, output_height);
    channel_ui.psd_rect = normalized_rect(panel_x, psd_y, main_width, channel_psd_height, output_width, output_height);
    channel_ui.waterfall_rect = normalized_rect(panel_x, heatmap_y, main_width, channel_heat_height, output_width, output_height);
    channel_ui.mask_rect = normalized_rect(panel_x, mask_y, main_width, channel_mask_height, output_width, output_height);
    channel_ui.confidence_rect = normalized_rect(panel_x - 8, mask_y + channel_mask_height, main_width + 16, 0, output_width, output_height);
    channel_ui.center_frequency_hz = channel.info.center_frequency_hz;
    channel_ui.span_hz = span_hz;
    channel_ui.resolution_hz = channel.info.resolution_hz;
    channel_ui.display_frequency_bin_hz = channel.info.display_frequency_bin_hz;
    channel_ui.seconds_per_time_bin = channel.info.display_time_bin_seconds;
    channel_ui.displayed_frame_ratio = channel.info.displayed_frame_ratio;
    channel_ui.displayed_frame_stride = channel.info.displayed_frame_stride;
    channel_ui.fft_rows_per_frame = channel.info.fft_rows_per_frame;
    channel_ui.displayed_fft_rows_per_frame = channel.info.displayed_fft_rows_per_frame;
    channel_ui.fft_size = channel.info.fft_size;
    channel_ui.history_rows = history_rows;
    channel_ui.dino_chunk_rows = channel.info.dino_chunk_rows;
    channel_ui.dino_chunk_cols = channel.info.dino_chunk_cols;
    channel_ui.overlay_available = channel.overlay_available;
    channel_ui.detector_label = channel.info.detector_label;

    fill_rect(canvas, output_width, output_height, panel_x - 8, psd_y - 8, main_width + 16, channel_psd_height + 16, {14, 18, 28});
    if (!channel.density_trace.empty()) {
      for (int col = 0; col < main_width; ++col) {
        const size_t density_index = std::min(static_cast<size_t>(col) * channel.density_trace.size() / std::max(1, main_width), channel.density_trace.size() - 1);
        const float density_value = std::clamp(channel.density_trace[density_index], 0.0f, 1.0f);
        const auto color = heatmap_color(density_value);
        fill_rect(canvas, output_width, output_height, panel_x + col, psd_y, 1, channel_psd_height, {static_cast<uint8_t>(color[0] * 0.35f), static_cast<uint8_t>(color[1] * 0.35f), static_cast<uint8_t>(color[2] * 0.35f)});
      }
    }
    draw_plot_axes(canvas,
             output_width,
             output_height,
             panel_x,
             psd_y,
             main_width,
             channel_psd_height,
             freq_min_label,
             freq_max_label,
             "-100",
             "0",
                   render_canvas_chrome ? 1 : 0,
             "FREQ",
             "POWER");
    draw_trace_plot(canvas, output_width, output_height, panel_x, psd_y, main_width, channel_psd_height, channel.current_psd_trace, accent, 2);
    draw_trace_plot(canvas, output_width, output_height, panel_x, psd_y, main_width, channel_psd_height, channel.max_hold_trace, {255, 212, 89}, 1);
    if (render_canvas_chrome) {
      draw_text(canvas, output_width, output_height, panel_x + 8, psd_y + 8, std::string("CH") + std::to_string(channel.info.channel), {232, 236, 241}, 1);
      draw_text(canvas, output_width, output_height, panel_x + 42, psd_y + 8, "PSD", {156, 173, 192}, 1);
      draw_text(canvas, output_width, output_height, panel_x + 72, psd_y + 8, "MAX HOLD", {255, 212, 89}, 1);
    }

    fill_rect(canvas, output_width, output_height, panel_x - 8, heatmap_y - 8, main_width + 16, channel_heat_height + 16, {14, 18, 28});
    fill_rect(canvas, output_width, output_height, panel_x - 8, heatmap_y - 8, 3, channel_heat_height + 16, accent);
    draw_plot_axes(canvas,
             output_width,
             output_height,
             panel_x,
             heatmap_y,
             main_width,
             channel_heat_height,
             freq_min_label,
             freq_max_label,
             newest_time_label,
             oldest_time_label,
                   render_canvas_chrome ? 1 : 0,
                 "FREQ",
                 "TIME BIN");
    blit_grayscale_ring_to_canvas(canvas,
                                  output_width,
                                  output_height,
                                  panel_x,
                                  heatmap_y,
                                  main_width,
                                  channel_heat_height,
                                  channel.history_grayscale,
                                  history_width,
                                  history_rows,
                                  channel.history_valid_rows,
                                  channel.history_write_row,
                                  blue_limit,
                                  red_limit);
    draw_grid(canvas, output_width, output_height, panel_x, heatmap_y, main_width, channel_heat_height);
    if (overlay_enabled) {
      overlay_mask_ring(canvas,
                        output_width,
                        output_height,
                        panel_x,
                        heatmap_y,
                        main_width,
                        channel_heat_height,
                        channel.history_mask,
                        history_width,
                        history_rows,
                        channel.history_valid_rows,
                        channel.history_write_row,
                        overlay_alpha);
    }

              fill_rect(canvas, output_width, output_height, panel_x - 8, mask_y - 8, main_width + 16, channel_mask_height + 16, {14, 18, 28});
              fill_rect(canvas, output_width, output_height, panel_x - 8, mask_y - 8, 3, channel_mask_height + 16, {232, 236, 241});
              size_t mask_nonzero = 0;
              uint8_t mask_max = 0;
              for (uint8_t value : channel.history_mask) {
                if (value > 0) {
                  ++mask_nonzero;
                  mask_max = std::max(mask_max, value);
                }
              }
              size_t latest_mask_nonzero = 0;
              uint8_t latest_mask_max = 0;
              for (uint8_t value : channel.latest_mask.pixels) {
                if (value > 0) {
                  ++latest_mask_nonzero;
                  latest_mask_max = std::max(latest_mask_max, value);
                }
              }
              if (mask_nonzero > 0) {
                draw_rect_outline(canvas, output_width, output_height, panel_x - 8, mask_y - 8, main_width + 16, channel_mask_height + 16, {80, 200, 120}, 1);
              }
              draw_plot_axes(canvas,
                             output_width,
                             output_height,
                             panel_x,
                             mask_y,
                             main_width,
                             channel_mask_height,
                             freq_min_label,
                             freq_max_label,
                             newest_time_label,
                             oldest_time_label,
                             render_canvas_chrome ? 1 : 0,
                             "FREQ",
                             "TIME BIN");
              bool showing_latest_fallback = false;
              if (overlay_enabled) {
                if (mask_nonzero > 0) {
                  blit_mask_ring_to_canvas(canvas,
                                           output_width,
                                           output_height,
                                           panel_x,
                                           mask_y,
                                           main_width,
                                           channel_mask_height,
                                           channel.history_mask,
                                           history_width,
                                           history_rows,
                                           channel.history_valid_rows,
                                           channel.history_write_row);
                } else if (!channel.latest_mask.pixels.empty() && channel.latest_mask.width > 0 && channel.latest_mask.height > 0) {
                  blit_mask_frame_to_canvas(canvas,
                                            output_width,
                                            output_height,
                                            panel_x,
                                            mask_y,
                                            main_width,
                                            channel_mask_height,
                                            channel.latest_mask);
                  showing_latest_fallback = true;
                  draw_rect_outline(canvas, output_width, output_height, panel_x - 8, mask_y - 8, main_width + 16, channel_mask_height + 16, {255, 212, 89}, 1);
                }
                draw_grid(canvas, output_width, output_height, panel_x, mask_y, main_width, channel_mask_height);
              }
              if (render_canvas_chrome) {
                const std::string mask_title = overlay_enabled
                    ? "SIGNAL DETECTION MASK ENABLED"
                    : "SIGNAL DETECTION MASK DISABLED";
                const RgbColor mask_title_color = overlay_enabled ? kGreen : RgbColor{228, 72, 72};
                draw_text(canvas,
                          output_width,
                          output_height,
                          panel_x + 8,
                          mask_y + 8,
                          mask_title,
                          mask_title_color,
                          2);
              }

    // Confidence fill — use density trace average as proxy
    float confidence = 0.0f;
    if (!channel.density_trace.empty()) {
      for (float v : channel.density_trace) confidence += v;
      confidence /= static_cast<float>(channel.density_trace.size());
      confidence = std::clamp(confidence * 4.0f, 0.0f, 1.0f);
    }
    channel_ui.confidence = confidence;
    ui_state.channels.push_back(channel_ui);
  }

  fill_rect(canvas, output_width, output_height, sidebar_x, sidebar_y, kSidebarWidth, sidebar_height, {11, 15, 23});
  draw_rect_outline(canvas, output_width, output_height, sidebar_x, sidebar_y, kSidebarWidth, sidebar_height, {62, 74, 92}, 1);
  if (render_canvas_chrome) {
    draw_text(canvas, output_width, output_height, sidebar_x + 16, sidebar_y + 12, "DISPLAY", {232, 236, 241}, 2);
    draw_vertical_slider(canvas, output_width, output_height, sidebar_x + 24, sidebar_y + 34, 104, blue_limit, {90, 148, 255}, "BLUE");
    draw_vertical_slider(canvas, output_width, output_height, sidebar_x + 74, sidebar_y + 34, 104, red_limit, {255, 98, 62}, "RED");
    for (int i = 0; i < 64; ++i) {
      const auto color = heatmap_color(static_cast<float>(63 - i) / 63.0f);
      fill_rect(canvas, output_width, output_height, sidebar_x + 122, sidebar_y + 34 + i * 2, 20, 2, {color[0], color[1], color[2]});
    }
    draw_toggle_button(canvas,
                       output_width,
                       output_height,
                       sidebar_x + 152,
                       sidebar_y + 34,
                       kSidebarWidth - 168,
                       "OVERLAY",
                       overlay_enabled,
                       kOrange);
  }

  int text_y = sidebar_y + 156;

  // Channel info section
  if (render_canvas_chrome) {
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "CHANNEL INFO", {74, 96, 128}, 2);
    text_y += 18;
    fill_rect(canvas, output_width, output_height, sidebar_x + 16, text_y, kSidebarWidth - 32, 1, {17, 29, 42});
    text_y += 6;
  }

  for (int channel_index = 0; channel_index < active_channels; ++channel_index) {
    const auto& channel = has_active_channels
        ? *active_channel_states[static_cast<size_t>(channel_index)]
        : channels.front();
    const RgbColor& accent = kAccents[std::min(channel_index, 1)];
    const int ch_num = channel.info.channel >= 0 ? channel.info.channel : channel_index;

    if (!render_canvas_chrome) {
      continue;
    }
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y,
              std::string("CH-") + std::to_string(ch_num), accent, 1);
    text_y += 14;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "CF", {122, 143, 168}, 1);
    std::ostringstream cf_text;
    cf_text << std::fixed << std::setprecision(3) << (channel.info.center_frequency_hz / 1.0e6) << "MHZ";
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, cf_text.str(), {200, 212, 224}, 1);
    text_y += 12;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "FFT", {122, 143, 168}, 1);
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y,
              std::to_string(channel.info.fft_size), {200, 212, 224}, 1);
    text_y += 12;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "FREQ BIN", {122, 143, 168}, 1);
    std::ostringstream res_text;
        res_text << format_frequency_label(channel.info.display_frequency_bin_hz);
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, res_text.str(), {200, 212, 224}, 1);
    text_y += 12;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "TIME BIN", {122, 143, 168}, 1);
    draw_text(canvas,
          output_width,
          output_height,
          sidebar_x + 60,
          text_y,
          format_time_bin_label(channel.info.display_time_bin_seconds),
          {200, 212, 224},
          1);
        text_y += 12;
        draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "FRAMES", {122, 143, 168}, 1);
        draw_text(canvas,
          output_width,
          output_height,
          sidebar_x + 60,
          text_y,
          format_displayed_frame_ratio_label(channel.info.displayed_frame_ratio,
                     channel.info.displayed_frame_stride),
          {200, 212, 224},
          1);
    text_y += 12;
            draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "VIS RATIO", {122, 143, 168}, 1);
            draw_text(canvas,
              output_width,
              output_height,
              sidebar_x + 60,
              text_y,
              format_fft_row_visualization_ratio_label(channel.info.displayed_fft_rows_per_frame,
                           channel.info.fft_rows_per_frame),
              {200, 212, 224},
              1);
            text_y += 12;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "CHUNK", {122, 143, 168}, 1);
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y,
              std::to_string(channel.info.dino_chunk_rows) + "X" + std::to_string(channel.info.dino_chunk_cols),
              {200, 212, 224}, 1);
    text_y += 18;
    fill_rect(canvas, output_width, output_height, sidebar_x + 16, text_y, kSidebarWidth - 32, 1, {17, 29, 42});
    text_y += 8;
  }

  // Signal activity section
  if (render_canvas_chrome) {
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "SIGNAL ACTIVITY", {74, 96, 128}, 2);
    text_y += 18;
  }
  for (int channel_index = 0; channel_index < active_channels; ++channel_index) {
    if (!render_canvas_chrome) {
      break;
    }
    const auto& channel = has_active_channels
        ? *active_channel_states[static_cast<size_t>(channel_index)]
        : channels.front();
    const RgbColor& accent = kAccents[std::min(channel_index, 1)];
    const int ch_num = channel.info.channel >= 0 ? channel.info.channel : channel_index;

    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y,
              std::string("CH-") + std::to_string(ch_num), accent, 1);

    float activity = 0.0f;
    if (!channel.density_trace.empty()) {
      for (float v : channel.density_trace) activity += v;
      activity = std::clamp((activity / channel.density_trace.size()) * 4.0f, 0.0f, 1.0f);
    }
    const int act_x = sidebar_x + 50;
    const int act_w = kSidebarWidth - 80;
    fill_rect(canvas, output_width, output_height, act_x, text_y, act_w, 8, {13, 21, 32});
    if (activity > 0.0f) {
      fill_rect(canvas, output_width, output_height, act_x, text_y,
                static_cast<int>(act_w * activity), 8, kOrange);
    }
    std::ostringstream act_ss;
    act_ss << static_cast<int>(activity * 100.0f) << "%";
    draw_text(canvas, output_width, output_height, act_x + act_w + 4, text_y, act_ss.str(), kOrange, 1);
    text_y += 14;
  }
  if (render_canvas_chrome) {
    text_y += 4;
    fill_rect(canvas, output_width, output_height, sidebar_x + 16, text_y, kSidebarWidth - 32, 1, {17, 29, 42});
    text_y += 8;
  }

  // System metrics section
  if (render_canvas_chrome) {
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "SYSTEM", {74, 96, 128}, 2);
    text_y += 18;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "GPU", {122, 143, 168}, 1);
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, "RTX 4000 ADA", kGreen, 1);
    text_y += 12;
    draw_text(canvas,
              output_width,
              output_height,
              sidebar_x + 16,
              text_y,
              active_detector_label + " LAT",
              {122, 143, 168},
              1);
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, "42 MS", kOrange, 1);
    text_y += 12;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "VIS FPS", {122, 143, 168}, 1);
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, "~6 FPS", kGreen, 1);
  }

  const int footer_y = output_height - kFooterHeight + 12;
  // Colored dots
  if (render_canvas_chrome) {
    fill_rect(canvas, output_width, output_height, grid_x, footer_y + 2, 5, 5, kGreen);
    draw_text(canvas, output_width, output_height, grid_x + 9, footer_y, "CLASSIC ANALYZER VIEW", {156, 173, 192}, 1);

    fill_rect(canvas, output_width, output_height, grid_x + 170, footer_y + 2, 5, 5, kDimBlue);
    draw_text(canvas, output_width, output_height, grid_x + 179, footer_y, "DENSITY HEAT UNDER MAX HOLD", {116, 132, 150}, 1);

    fill_rect(canvas, output_width, output_height, grid_x + 360, footer_y + 2, 5, 5, kOrange);
    draw_text(canvas,
              output_width,
              output_height,
              grid_x + 369,
              footer_y,
              overlay_enabled ? (active_detector_label + " DETECTION OVERLAY")
                              : (active_detector_label + " DETECTION OVERLAY OFF"),
              overlay_enabled ? RgbColor{116, 132, 150} : RgbColor{90, 104, 128},
              1);
  }
  
  // Visual drop indicator — shows frame drop rate in footer
  // Orange warning color when dropping, green when keeping up
  
  if (render_canvas_chrome && total_frames > 0) {
    const float drop_rate = static_cast<float>(dropped_frames) / static_cast<float>(total_frames);
    const uint64_t live_frames = total_frames - dropped_frames;
    std::ostringstream drop_text;
    const int live_pct = static_cast<int>(std::round((1.0f - drop_rate) * 100.0f));
    drop_text << "VIS " << live_pct << "% LIVE";
    const RgbColor drop_color = drop_rate > 0.1f ? RgbColor{255, 165, 0} : RgbColor{80, 200, 120};
    draw_text(canvas, output_width, output_height, grid_x + 420, footer_y, drop_text.str(), drop_color, 1);
  }
  // Live clock
  if (render_canvas_chrome) {
    auto now = std::chrono::system_clock::now();
    std::time_t now_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    localtime_r(&now_t, &tm_buf);
    std::ostringstream clock_ss;
    clock_ss << std::setfill('0')
             << std::setw(2) << tm_buf.tm_mon + 1 << "/"
             << std::setw(2) << tm_buf.tm_mday << "/"
             << (tm_buf.tm_year + 1900) << "  "
             << std::setw(2) << tm_buf.tm_hour << ":"
             << std::setw(2) << tm_buf.tm_min << ":"
             << std::setw(2) << tm_buf.tm_sec;
    const std::string clock_str = clock_ss.str();
    const int clock_x = output_width - kSidebarWidth - kPanelPadding
                        - text_pixel_width(clock_str, 1) - 8;
    draw_text(canvas, output_width, output_height, clock_x, footer_y, clock_str, {58, 80, 96}, 1);
  }

  update_visualization_ui_state(ui_state);

  return canvas;
}

}  // namespace holoscan::ops
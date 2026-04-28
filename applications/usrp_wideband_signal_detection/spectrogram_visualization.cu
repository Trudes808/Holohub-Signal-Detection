#include "spectrogram_visualization.hpp"

#include <cuda/std/detail/libcxx/include/algorithm>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <matx.h>
#include <gxf/std/tensor.hpp>

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using SpectrogramComplex = cuda::std::complex<float>;
using SpectrogramTensor = matx::tensor_t<SpectrogramComplex, 2>;
using SpectrogramMessage = std::tuple<SpectrogramTensor, cudaStream_t>;

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

  const float four_x = normalized * 4.0f;
  const float red = std::clamp(std::min(four_x - 1.5f, -four_x + 4.5f), 0.0f, 1.0f);
  const float green = std::clamp(std::min(four_x - 0.5f, -four_x + 3.5f), 0.0f, 1.0f);
  const float blue = std::clamp(std::min(four_x + 0.5f, -four_x + 2.5f), 0.0f, 1.0f);

  return {static_cast<uint8_t>(red * 255.0f),
          static_cast<uint8_t>(green * 255.0f),
          static_cast<uint8_t>(blue * 255.0f)};
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
constexpr int kPanelPadding = 18;

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
  const int thickness = std::max(1, scale / 2 + 1);
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

void overlay_mask(std::vector<uint8_t>& canvas,
                  int canvas_width,
                  int canvas_height,
                  int dst_x,
                  int dst_y,
                  int dst_width,
                  int dst_height,
                  const holoscan::ops::OfflinePgmFrame& mask_frame,
                  float overlay_alpha) {
  const RgbColor overlay_color{255, 130, 48};
  for (int row = 0; row < dst_height; ++row) {
    const int src_row = std::min(mask_frame.height - 1, (row * mask_frame.height) / std::max(1, dst_height));
    for (int col = 0; col < dst_width; ++col) {
      const int src_col = std::min(mask_frame.width - 1, (col * mask_frame.width) / std::max(1, dst_width));
      const auto value = mask_frame.pixels[static_cast<size_t>(src_row) * static_cast<size_t>(mask_frame.width) + static_cast<size_t>(src_col)];
      if (value < 128) {
        continue;
      }
      blend_pixel(canvas, canvas_width, canvas_height, dst_x + col, dst_y + row, overlay_color, overlay_alpha);
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
    const int gx = x + (rect_width * step) / 8;
    const int gy = y + (rect_height * step) / 8;
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
                    const std::string& y1) {
  const RgbColor axis_color{142, 156, 174};
  draw_rect_outline(canvas, width, height, x, y, plot_width, plot_height, axis_color, 1);
  for (int step = 1; step < 5; ++step) {
    const int gy = y + (plot_height * step) / 5;
    fill_rect(canvas, width, height, x, gy, plot_width, 1, {48, 58, 76});
  }
  for (int step = 1; step < 6; ++step) {
    const int gx = x + (plot_width * step) / 6;
    fill_rect(canvas, width, height, gx, y, 1, plot_height, {40, 48, 64});
  }
  draw_text(canvas, width, height, x, y + plot_height + 6, x0, {164, 179, 196}, 1);
  draw_text(canvas, width, height, x + plot_width - text_pixel_width(x1, 1), y + plot_height + 6, x1, {164, 179, 196}, 1);
  draw_text(canvas, width, height, x - 2, y - 10, y1, {164, 179, 196}, 1);
  draw_text(canvas, width, height, x - 2, y + plot_height - 6, y0, {164, 179, 196}, 1);
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


holoscan::gxf::Entity create_rgb_entity(holoscan::ExecutionContext& context,
                                        const std::vector<uint8_t>& rgb,
                                        int width,
                                        int height,
                                        const std::string& tensor_name) {
  auto device_buffer = std::shared_ptr<void*>(new void*, [](void** pointer) {
    if (pointer != nullptr) {
      if (*pointer != nullptr) {
        cudaFree(*pointer);
      }
      delete pointer;
    }
  });

  const size_t bytes = rgb.size() * sizeof(uint8_t);
  auto alloc_result = cudaMalloc(device_buffer.get(), bytes);
  if (alloc_result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(alloc_result));
  }

  // auto copy_result = cudaMemcpy(*device_buffer, rgb.data(), bytes, cudaMemcpyHostToDevice);
  // if (copy_result != cudaSuccess) {
  //   throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(copy_result));
  // }

  // Use async copy to avoid blocking — caller is responsible for
  // synchronizing vis_stream before this buffer is consumed
  auto copy_result = cudaMemcpyAsync(*device_buffer, rgb.data(), bytes, cudaMemcpyHostToDevice, 0);
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
                             *device_buffer,
                             [buffer = device_buffer](void*) mutable {
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

}  // namespace


namespace holoscan::ops {

void SpectrogramToHolovizOp::setup(OperatorSpec& spec) {
  spec.input<SpectrogramMessage>("in");
  spec.input<DetectorMaskMessage>("mask_in").condition(holoscan::ConditionType::kNone);
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
  spec.param(center_frequency_hz_, "center_frequency_hz", "Center Frequency", "Center frequency for display in Hz.", 0.0);
  spec.param(fft_size_, "fft_size", "FFT Size", "FFT size shown in analyzer readouts.", 20480);
  spec.param(dino_chunk_rows_, "dino_chunk_rows", "DINO Chunk Rows", "DINO chunk height shown in readouts.", 256);
  //spec.param(dino_chunk_cols_, "dino_chunk_cols", "DINO Chunk Cols", "DINO chunk width shown in readouts.", 512);
  spec.param(dino_chunk_cols_, "dino_chunk_cols", "DINO Chunk Cols", "DINO chunk width shown in readouts.", 512);
  spec.param(db_floor_, "db_floor", "dB Floor", "Fixed dB floor for spectrogram normalization.", -100.0f);
  spec.param(db_ceil_,  "db_ceil",  "dB Ceiling", "Fixed dB ceiling for spectrogram normalization.", 0.0f);
  spec.param(row_average_n_, "row_average_n", "Row Average N", "Frames averaged per waterfall row.", 4);
}


void SpectrogramToHolovizOp::initialize() {
  Operator::initialize();
  // Create a dedicated CUDA stream for visualization rendering
  // so it never blocks the pipeline's CUDA stream
<<<<<<< Updated upstream
  metadata_policy(holoscan::MetadataPolicy::kUpdate);  // ← add this
=======
>>>>>>> Stashed changes
  auto result = cudaStreamCreateWithFlags(&vis_stream_, cudaStreamNonBlocking);
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to create vis_stream_: ") + cudaGetErrorString(result));
  }

  // Allocate pinned host buffer for async GPU→host DMA
  // Sized for largest possible tensor: num_bursts * fft_size * complex<float>
  // 1024 bursts * 20480 FFT points * 8 bytes = 167,772,160 bytes
  const size_t max_bytes = 1024ULL * 20480ULL * sizeof(float) * 2;
  auto pinned_result = cudaMallocHost(&pinned_host_buffer_, max_bytes);
  if (pinned_result != cudaSuccess) {
    throw std::runtime_error(std::string("Failed to allocate pinned host buffer: ") + cudaGetErrorString(pinned_result));
  }
  pinned_host_buffer_bytes_ = max_bytes;

  // Start background render thread — operator thread hands off work here
  // and returns immediately, never blocking the pipeline
  render_thread_ = std::thread([this]() {
    while (true) {
      std::unique_lock<std::mutex> lock(render_mutex_);
      render_cv_.wait(lock, [this] { return render_ready_ || render_stop_; });
      if (render_stop_) break;
      auto task = std::move(render_task_);
      render_ready_ = false;
      lock.unlock();
      task();  // execute compose + create_rgb_entity + emit
    }
  });
}

void SpectrogramToHolovizOp::stop() {
  // Signal background render thread to stop and wait for it
  {
    std::unique_lock<std::mutex> lock(render_mutex_);
    render_stop_ = true;
    render_cv_.notify_one();
  }
  if (render_thread_.joinable()) {
    render_thread_.join();
  }
  // Clean up the dedicated visualization stream
  if (vis_stream_ != nullptr) {
    cudaStreamSynchronize(vis_stream_);
    cudaStreamDestroy(vis_stream_);
    vis_stream_ = nullptr;
  }
  // Free pinned host buffer
  if (pinned_host_buffer_ != nullptr) {
    cudaFreeHost(pinned_host_buffer_);
    pinned_host_buffer_ = nullptr;
  }
  Operator::stop();
}

//end newy added


void SpectrogramToHolovizOp::compute(InputContext& op_input,
                                     OutputContext& op_output,
                                     ExecutionContext& context) { 

  auto input = op_input.receive<SpectrogramMessage>("in").value();
  const auto& tensor = std::get<0>(input);
  const auto stream = std::get<1>(input);

  const auto meta = metadata();
  const int channel_number = meta ? static_cast<int>(meta->get<uint16_t>("channel_number", 0)) : 0;
  if (channel_filter_.get() >= 0 && channel_number != channel_filter_.get()) {
    return;
  }

  // Receive mask from coherent power detector if available
  auto mask_msg = op_input.receive<DetectorMaskMessage>("mask_in");
  if (mask_msg) {
<<<<<<< Updated upstream
    printf("MASK RECEIVED: w=%d h=%d pixels=%zu\n", mask_msg->width, mask_msg->height, mask_msg->pixels.size());
=======
>>>>>>> Stashed changes
    if (channel_states_.size() <= static_cast<size_t>(mask_msg->channel)) {
      channel_states_.resize(static_cast<size_t>(mask_msg->channel + 1));
    }
    auto& mask_state = channel_states_[static_cast<size_t>(mask_msg->channel)];
    mask_state.latest_mask.pixels = mask_msg->pixels;
    mask_state.latest_mask.width = mask_msg->width;
    mask_state.latest_mask.height = mask_msg->height;
    mask_state.overlay_available = true;
  }

  if (channel_states_.size() < static_cast<size_t>(std::max(num_channels_.get(), channel_number + 1))) {
    channel_states_.resize(static_cast<size_t>(std::max(num_channels_.get(), channel_number + 1)));
  }

  // const int width = std::max(1, std::min(output_width_.get(), static_cast<int>(tensor.Size(1))));
  // const int height = std::max(1, std::min(output_height_.get(), static_cast<int>(tensor.Size(0))));
  
  // //auto grayscale = reduce_spectrogram_to_grayscale(tensor, stream, height, width);
  // // Record when pipeline stream finishes writing data
  // // Then make vis_stream_ wait for that event before reading
  // // This frees the pipeline stream immediately without data corruption
  // cudaEvent_t pipeline_done;
  // cudaEventCreateWithFlags(&pipeline_done, cudaEventDisableTiming);
  // cudaEventRecord(pipeline_done, stream);           // mark pipeline stream checkpoint
  // cudaStreamWaitEvent(vis_stream_, pipeline_done, 0); // vis_stream waits for that checkpoint
  // cudaEventDestroy(pipeline_done);

  // // Now use vis_stream_ instead of pipeline stream
  // // Pipeline stream is FREE to process next RF batch immediately
  // auto grayscale = reduce_spectrogram_to_grayscale(tensor, vis_stream_, height, width);
  

  const int width = std::max(1, std::min(output_width_.get(), static_cast<int>(tensor.Size(1))));
  const int height = std::max(1, std::min(output_height_.get(), static_cast<int>(tensor.Size(0))));

  // Count every frame received
  // Emit any previously composed frame from the background thread
  // Only GXF-safe place to call emit — on operator thread within tick
  {
    std::unique_lock<std::mutex> composed_lock(composed_mutex_);
    if (pending_composed_ready_) {
      auto output_entity = create_rgb_entity(context,
                                             pending_composed_,
                                             pending_composed_width_,
                                             pending_composed_height_,
                                             tensor_name_.get());
      op_output.emit(output_entity, "outputs");
      pending_composed_ready_ = false;
    }
  }

  // Count every frame received
  total_frames_++;

  // Check if vis_stream_ is still busy rendering the previous frame.
  // If busy → drop this frame immediately, pipeline continues without blocking.
  if (cudaStreamQuery(vis_stream_) == cudaErrorNotReady) {
    dropped_frames_++;
    return;  // pipeline is NEVER blocked 
  }

  // vis_stream_ is free — safely hand off data without blocking pipeline stream.
  // cudaEventRecord marks when pipeline stream finishes writing this batch.
  // cudaStreamWaitEvent tells vis_stream_ to wait for that point before reading.
  // After this, pipeline stream is FREE to process the next RF batch immediately.
  // Decouple vis_stream_ from pipeline stream using a CUDA event.
  // Pipeline stream is FREE to process next RF batch immediately after this.
  cudaEvent_t pipeline_done;
  cudaEventCreateWithFlags(&pipeline_done, cudaEventDisableTiming);
  cudaEventRecord(pipeline_done, stream);
  cudaStreamWaitEvent(vis_stream_, pipeline_done, 0);
  cudaEventDestroy(pipeline_done);

  // Kick off async GPU→host DMA on vis_stream_ into pinned buffer.
  // Operator thread does NOT synchronize — returns immediately after this.
  const int tensor_rows = static_cast<int>(tensor.Size(0));
  const int tensor_cols = static_cast<int>(tensor.Size(1));
  const size_t tensor_bytes = static_cast<size_t>(tensor_rows)
                            * static_cast<size_t>(tensor_cols)
                            * sizeof(cuda::std::complex<float>);
  if (tensor_bytes > pinned_host_buffer_bytes_) {
    // Buffer too small — skip this frame rather than overflow
    dropped_frames_++;
    return;
  }
  cudaMemcpyAsync(pinned_host_buffer_, tensor.Data(),
                  tensor_bytes, cudaMemcpyDeviceToHost, vis_stream_);

  // Check if background render thread is free.
  // If busy → drop this frame, operator thread returns immediately.
  // Throttle rendering to every 8th frame — matches background thread capacity
  // of ~6 fps at 500 MSps. Signal history still updates every tick.
  // Per-channel throttle counter — ensures both channels get render turns
  if (channel_states_.size() <= static_cast<size_t>(channel_number)) {
    channel_states_.resize(static_cast<size_t>(channel_number + 1));
  }
  static std::array<uint64_t, 2> compose_ticks = {0, 0};
  const int tick_index = channel_number % 2;
  compose_ticks[tick_index]++;
  if (compose_ticks[tick_index] % 8 != 0) {
    return;
  }

  // Check if background render thread is free.
  // If busy → drop this frame, operator thread returns immediately.
  {
    std::unique_lock<std::mutex> lock(render_mutex_);
    if (render_ready_) {
      dropped_frames_++;
      return;
    }

    // Capture metadata by value — operator thread returns after this block
    auto snapshot_channel_number = channel_number;
    auto snapshot_dropped = dropped_frames_;
    auto snapshot_total = total_frames_;
    float blue = blue_limit_.get();
    float red = red_limit_.get();
    float alpha = overlay_alpha_.get();
    float red_lim = red_limit_.get();
    float snap_db_floor = db_floor_.get();
    float snap_db_ceil  = db_ceil_.get();
    int snap_row_average_n = row_average_n_.get();
    int snap_width = width;
    int snap_height = height;
    int snap_tensor_rows = tensor_rows;
    int snap_tensor_cols = tensor_cols;
    int snap_history_frames = history_frames_.get();

    render_task_ = [this,
<<<<<<< Updated upstream
                snapshot_channel_number,
                snapshot_dropped,
                snapshot_total,
                blue, red, alpha, red_lim,
                snap_db_floor, snap_db_ceil,
                snap_row_average_n,
                snap_width, snap_height,
                snap_tensor_rows, snap_tensor_cols,
                snap_history_frames]() mutable {

      // Sync vis_stream_ HERE — blocks background thread, NOT operator thread
      cudaStreamSynchronize(vis_stream_);

      auto snapshot_overlay_available = channel_states_[static_cast<size_t>(snapshot_channel_number)].overlay_available;
      auto snapshot_mask = channel_states_[static_cast<size_t>(snapshot_channel_number)].latest_mask;

=======
                    snapshot_channel_number,
                    snapshot_dropped,
                    snapshot_total,
                    blue, red, alpha, red_lim,
                    snap_db_floor, snap_db_ceil,
                    snap_row_average_n,
                    snap_width, snap_height,
                    snap_tensor_rows, snap_tensor_cols,
                    snap_history_frames]() mutable {
      // Sync vis_stream_ HERE — blocks background thread, NOT operator thread
      cudaStreamSynchronize(vis_stream_);

>>>>>>> Stashed changes
      // pinned_host_buffer_ now has complete GPU data — run reduction on CPU
      auto grayscale = reduce_from_pinned_buffer(pinned_host_buffer_,
                                                 snap_tensor_rows,
                                                 snap_tensor_cols,
                                                 snap_height,
                                                 snap_width,
                                                 snap_db_floor,
                                                 snap_db_ceil);

      // Update channel state
      auto& state = channel_states_[static_cast<size_t>(snapshot_channel_number)];
      state.active = true;
      state.info.channel = snapshot_channel_number;
<<<<<<< Updated upstream
      state.overlay_available = snapshot_overlay_available;
      state.latest_mask = snapshot_mask;
=======
>>>>>>> Stashed changes

      // Accumulate grayscale rows — average N frames before appending to history
      // This makes waterfall scroll slower but smoothly, no jumping
      const size_t gray_size = static_cast<size_t>(snap_width) * static_cast<size_t>(snap_height);
      if (state.row_accumulator.size() != gray_size) {
        state.row_accumulator.assign(gray_size, 0.0f);
        state.row_accumulator_count = 0;
      }
      for (size_t i = 0; i < gray_size; ++i) {
        state.row_accumulator[i] += static_cast<float>(grayscale[i]);
      }
      state.row_accumulator_count++;

      if (state.row_accumulator_count >= snap_row_average_n) {
        // Produce averaged grayscale
        std::vector<uint8_t> averaged(gray_size);
        for (size_t i = 0; i < gray_size; ++i) {
          averaged[i] = static_cast<uint8_t>(std::clamp(
              state.row_accumulator[i] / static_cast<float>(state.row_accumulator_count),
              0.0f, 255.0f));
        }
        // Reset accumulator
        std::fill(state.row_accumulator.begin(), state.row_accumulator.end(), 0.0f);
        state.row_accumulator_count = 0;
<<<<<<< Updated upstream
=======
        // Append averaged row to history
>>>>>>> Stashed changes
        append_spectrogram_history(state, averaged, snap_width, snap_height, snap_history_frames);
        state.current_psd_trace = compute_psd_trace(averaged, snap_width, snap_height);
      } else {
        // Not enough frames yet — still update PSD from latest raw frame
        state.current_psd_trace = compute_psd_trace(grayscale, snap_width, snap_height);
      }

      update_max_hold_trace(state.current_psd_trace, state.max_hold_trace);
      update_density_history(
          compute_density_trace_from_grayscale(grayscale, snap_width, snap_height, red_lim),
          state.density_trace,
          state.density_frames_seen);

      int composite_width = 0;
      int composite_height = 0;
<<<<<<< Updated upstream
      printf("RENDER: overlay_available=%d mask_pixels=%zu\n",
             (int)channel_states_[static_cast<size_t>(snapshot_channel_number)].overlay_available,
             channel_states_[static_cast<size_t>(snapshot_channel_number)].latest_mask.pixels.size());
=======
>>>>>>> Stashed changes
      auto composed = compose_visualization_rgb(channel_states_,
                                                blue,
                                                red,
                                                alpha,
                                                composite_width,
                                                composite_height,
                                                snapshot_dropped,
                                                snapshot_total);
      {
        std::unique_lock<std::mutex> composed_lock(composed_mutex_);
        pending_composed_ = std::move(composed);
        pending_composed_width_ = composite_width;
        pending_composed_height_ = composite_height;
        pending_composed_ready_ = true;
      }
    };
    render_ready_ = true;
    render_cv_.notify_one();
  }
  // Operator thread returns immediately here
  // cudaMemcpyAsync is still running on vis_stream_ — background thread will sync it
  // Pipeline stream is completely free
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
  spec.param(center_frequency_hz_, "center_frequency_hz", "Center Frequency", "Center frequency for display in Hz.", 0.0);
  spec.param(fft_size_, "fft_size", "FFT Size", "FFT size shown in analyzer readouts.", 20480);
  spec.param(dino_chunk_rows_, "dino_chunk_rows", "DINO Chunk Rows", "DINO chunk height shown in readouts.", 256);
  spec.param(dino_chunk_cols_, "dino_chunk_cols", "DINO Chunk Cols", "DINO chunk width shown in readouts.", 512);
}

void OfflinePgmReplayOp::initialize() {
  Operator::initialize();

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
  append_spectrogram_history(state, grayscale, width, height, history_frames_.get());
  state.current_psd_trace = compute_psd_trace(grayscale, width, height);
  update_max_hold_trace(state.current_psd_trace, state.max_hold_trace);
  const auto current_density = has_mask ? compute_density_trace(&mask_frame)
                                        : compute_density_trace_from_grayscale(grayscale, width, height, red_limit_.get());
  update_density_history(current_density, state.density_trace, state.density_frames_seen);
  state.overlay_available = has_mask;
  state.latest_mask = has_mask ? mask_frame : OfflinePgmFrame{};
  state.active = true;
  state.info.channel = channel;
  state.info.frame_number = static_cast<int64_t>(frame_number);
  state.info.center_frequency_hz = center_frequency_hz_.get();
  state.info.fft_size = fft_size_.get();
  state.info.dino_chunk_rows = dino_chunk_rows_.get();
  state.info.dino_chunk_cols = dino_chunk_cols_.get();
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
      active += mask_frame->pixels[static_cast<size_t>(row) * static_cast<size_t>(mask_frame->width) + static_cast<size_t>(col)] >= 128 ? 1 : 0;
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

void append_spectrogram_history(ChannelVisualizationState& state,
                                const std::vector<uint8_t>& grayscale,
                                int width,
                                int height,
                                int history_frames) {
  if (state.history_width != width) {
    state.history_grayscale.clear();
    state.history_width = width;
    state.latest_frame_height = height;
  }
  state.latest_frame_height = height;
  const int max_rows = std::max(1, history_frames) * height;
  const int existing_rows = state.history_width > 0 ? static_cast<int>(state.history_grayscale.size() / static_cast<size_t>(state.history_width)) : 0;
  const int kept_rows = std::min(existing_rows, std::max(0, max_rows - height));

  std::vector<uint8_t> combined(static_cast<size_t>(std::min(max_rows, kept_rows + height)) * static_cast<size_t>(width));
  if (kept_rows > 0) {
    const size_t src_offset = static_cast<size_t>(existing_rows - kept_rows) * static_cast<size_t>(width);
    std::copy(state.history_grayscale.begin() + static_cast<std::ptrdiff_t>(src_offset),
              state.history_grayscale.end(),
              combined.begin());
  }
  std::copy(grayscale.begin(), grayscale.end(), combined.begin() + static_cast<std::ptrdiff_t>(kept_rows * width));
  state.history_grayscale = std::move(combined);
}

std::vector<uint8_t> compose_visualization_rgb(const std::vector<ChannelVisualizationState>& channels,
                                               float blue_limit,
                                               float red_limit,
                                               float overlay_alpha,
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

  const int active_channels = std::max(1, static_cast<int>(channels.size()));
  int main_width = 512;
  for (const auto& channel : channels) {
    if (channel.active && channel.history_width > 0) {
      main_width = std::max(main_width, channel.history_width);
    }
  }
  const int channel_psd_height = 92;
  const int channel_heat_height = 220;
  const int channel_block_height = channel_psd_height + channel_heat_height + kPanelPadding * 2 + 36;
  const int columns = std::min(2, active_channels);
  const int rows = std::max(1, (active_channels + columns - 1) / columns);
  const int grid_width = columns * main_width + (columns - 1) * kPanelPadding;

  output_width = grid_width + kSidebarWidth + kPanelPadding * 3;
  output_height = kHeaderHeight + rows * channel_block_height + kFooterHeight + kPanelPadding * 2;

  std::vector<uint8_t> canvas(static_cast<size_t>(output_width) * static_cast<size_t>(output_height) * 3, 0);
  fill_vertical_gradient(canvas, output_width, output_height, {7, 10, 18}, {14, 19, 29});

  const int grid_x = kPanelPadding;
  const int sidebar_x = grid_x + grid_width + kPanelPadding;
  const int sidebar_y = kHeaderHeight + kPanelPadding;
  const int sidebar_height = output_height - sidebar_y - kFooterHeight - kPanelPadding;

  // fill_rect(canvas, output_width, output_height, 0, 0, output_width, kHeaderHeight, {8, 12, 18});
  // fill_rect(canvas, output_width, output_height, 0, kHeaderHeight - 2, output_width, 2, {124, 198, 255});
  // draw_text(canvas, output_width, output_height, 18, 10, "USRP WIDEBAND", {232, 236, 241}, 2);
  // draw_text(canvas, output_width, output_height, 18, 24, "TWO CHANNEL ANALYZER", {156, 173, 192}, 1);

  // Header background
  fill_rect(canvas, output_width, output_height, 0, 0, output_width, kHeaderHeight, {8, 13, 20});
  fill_rect(canvas, output_width, output_height, 0, kHeaderHeight - 2, output_width, 2, {124, 198, 255});

  // Title + subtitle
  draw_text(canvas, output_width, output_height, 18, 8, "USRP WIDEBAND", {232, 236, 241}, 2);
  draw_text(canvas, output_width, output_height, 18, 26, "DUAL CHANNEL ANALYZER", {124, 198, 255}, 1);

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

  draw_stat_box(218, 10, "CH-0 RATE", ch0_rate, kGreen);
  draw_stat_box(314, 10, "CH-1 RATE", ch1_rate, kGreen);
  draw_stat_box(410, 10, "CENTER FREQ", "2400.0 MHZ", kYellow);
  draw_stat_box(506, 10, "SPAN", "500 MHZ", {232, 236, 241});
  draw_stat_box(602, 10, "FFT SIZE", "20480", {232, 236, 241});

  // Badges — LIVE, DUAL CH, DINO V3
  auto draw_badge = [&](int x, int y, const std::string& text, const RgbColor& col) {
    const int w = text_pixel_width(text, 1) + 12;
    fill_rect(canvas, output_width, output_height, x, y, w, 16, {10, 21, 34});
    draw_rect_outline(canvas, output_width, output_height, x, y, w, 16, col, 1);
    draw_text(canvas, output_width, output_height, x + 6, y + 4, text, col, 1);
  };

  const int badge_y = 18;
  const int badge_x_start = output_width - kSidebarWidth - kPanelPadding - 180;
  draw_badge(badge_x_start,       badge_y, "LIVE",    kGreen);
  draw_badge(badge_x_start + 48,  badge_y, "DUAL CH", kCh0Accent);
  draw_badge(badge_x_start + 108, badge_y, "DINO V3", kYellow);

  for (int channel_index = 0; channel_index < active_channels; ++channel_index) {
    const auto& channel = channels[static_cast<size_t>(channel_index)];
    const int column_index = channel_index % columns;
    const int row_index = channel_index / columns;
    const int panel_x = grid_x + column_index * (main_width + kPanelPadding);
    const int panel_y = kHeaderHeight + kPanelPadding + row_index * channel_block_height;
    const int psd_y = panel_y;
    const int heatmap_y = panel_y + channel_psd_height + kPanelPadding;


    // Channel accent color
    const RgbColor& accent = kAccents[std::min(channel_index, 1)];

    // Panel header bar with channel label, freq, rate
    fill_rect(canvas, output_width, output_height, panel_x - 8, psd_y - 28, main_width + 16, 20, {10, 18, 32});
    draw_rect_outline(canvas, output_width, output_height, panel_x - 8, psd_y - 28, main_width + 16, 20, {26, 42, 58}, 1);
    // Left accent stripe
    fill_rect(canvas, output_width, output_height, panel_x - 8, psd_y - 28, 3, 20, accent);
    // CH label
    draw_text(canvas, output_width, output_height, panel_x, psd_y - 22, std::string("CH-") + std::to_string(channel.info.channel >= 0 ? channel.info.channel : channel_index), accent, 1);
    // Freq label
    std::ostringstream freq_ss;
    freq_ss << std::fixed << std::setprecision(3) << (channel.info.center_frequency_hz / 1.0e6) << " MHZ";
    draw_text(canvas, output_width, output_height, panel_x + 60, psd_y - 22, freq_ss.str(), {122, 143, 168}, 1);

    fill_rect(canvas, output_width, output_height, panel_x - 8, psd_y - 8, main_width + 16, channel_psd_height + 16, {14, 18, 28});
    if (!channel.density_trace.empty()) {
      for (int col = 0; col < main_width; ++col) {
        const size_t density_index = std::min(static_cast<size_t>(col) * channel.density_trace.size() / std::max(1, main_width), channel.density_trace.size() - 1);
        const float density_value = std::clamp(channel.density_trace[density_index], 0.0f, 1.0f);
        const auto color = heatmap_color(density_value);
        fill_rect(canvas, output_width, output_height, panel_x + col, psd_y, 1, channel_psd_height, {static_cast<uint8_t>(color[0] * 0.35f), static_cast<uint8_t>(color[1] * 0.35f), static_cast<uint8_t>(color[2] * 0.35f)});
      }
    }
    draw_plot_axes(canvas, output_width, output_height, panel_x, psd_y, main_width, channel_psd_height, "-SPAN", "+SPAN", "-100", "0");
    draw_trace_plot(canvas, output_width, output_height, panel_x, psd_y, main_width, channel_psd_height, channel.current_psd_trace, accent, 2);
    draw_trace_plot(canvas, output_width, output_height, panel_x, psd_y, main_width, channel_psd_height, channel.max_hold_trace, {255, 212, 89}, 1);
    draw_text(canvas, output_width, output_height, panel_x + 8, psd_y + 8, std::string("CH") + std::to_string(channel.info.channel), {232, 236, 241}, 1);
    draw_text(canvas, output_width, output_height, panel_x + 42, psd_y + 8, "PSD", {156, 173, 192}, 1);
    draw_text(canvas, output_width, output_height, panel_x + 72, psd_y + 8, "MAX HOLD", {255, 212, 89}, 1);

    const int history_rows = channel.history_width > 0 ? static_cast<int>(channel.history_grayscale.size() / static_cast<size_t>(channel.history_width)) : 0;
    auto spectrogram_rgb = colorize_grayscale_spectrogram(channel.history_grayscale.empty() ? std::vector<uint8_t>(static_cast<size_t>(main_width) * static_cast<size_t>(channel_heat_height), 0) : channel.history_grayscale,
                                                          blue_limit,
                                                          red_limit);
    
    fill_rect(canvas, output_width, output_height, panel_x - 8, heatmap_y - 8, main_width + 16, channel_heat_height + 16, {14, 18, 28});
    fill_rect(canvas, output_width, output_height, panel_x - 8, heatmap_y - 8, 3, channel_heat_height + 16, accent);
    draw_plot_axes(canvas, output_width, output_height, panel_x, heatmap_y, main_width, channel_heat_height, "START", "STOP", "NOW", "HIST");
    if (!channel.history_grayscale.empty() && history_rows > 0) {
      blit_rgb_nearest(canvas, output_width, output_height, panel_x, heatmap_y, main_width, channel_heat_height, spectrogram_rgb, channel.history_width, history_rows);
    }
    draw_grid(canvas, output_width, output_height, panel_x, heatmap_y, main_width, channel_heat_height);
    // if (channel.overlay_available) {
    //   overlay_mask(canvas, output_width, output_height, panel_x, heatmap_y + channel_heat_height - channel.latest_frame_height, main_width, channel.latest_frame_height, channel.latest_mask, overlay_alpha);
    // }

    if (channel.overlay_available && channel.latest_mask.width > 0 && channel.latest_mask.height > 0) {
      overlay_mask(canvas, output_width, output_height,
               panel_x, heatmap_y,
               main_width, channel_heat_height,
               channel.latest_mask, overlay_alpha);
}

    // DINO confidence bar
    const int dino_bar_y = heatmap_y + channel_heat_height + 10;
    fill_rect(canvas, output_width, output_height, panel_x - 8, dino_bar_y, main_width + 16, 14, {8, 12, 22});
    draw_rect_outline(canvas, output_width, output_height, panel_x - 8, dino_bar_y, main_width + 16, 14, {26, 42, 58}, 1);
    draw_text(canvas, output_width, output_height, panel_x, dino_bar_y + 3, "DINO", kOrange, 1);

    // Confidence fill — use density trace average as proxy
    float confidence = 0.0f;
    if (!channel.density_trace.empty()) {
      for (float v : channel.density_trace) confidence += v;
      confidence /= static_cast<float>(channel.density_trace.size());
      confidence = std::clamp(confidence * 4.0f, 0.0f, 1.0f);
    }
<<<<<<< Updated upstream
=======

    // DINO confidence bar
    const int dino_bar_y = heatmap_y + channel_heat_height + 10;
    fill_rect(canvas, output_width, output_height, panel_x - 8, dino_bar_y, main_width + 16, 14, {8, 12, 22});
    draw_rect_outline(canvas, output_width, output_height, panel_x - 8, dino_bar_y, main_width + 16, 14, {26, 42, 58}, 1);
    draw_text(canvas, output_width, output_height, panel_x, dino_bar_y + 3, "DINO", kOrange, 1);

    // Confidence fill — use density trace average as proxy
    float confidence = 0.0f;
    if (!channel.density_trace.empty()) {
      for (float v : channel.density_trace) confidence += v;
      confidence /= static_cast<float>(channel.density_trace.size());
      confidence = std::clamp(confidence * 4.0f, 0.0f, 1.0f);
    }
>>>>>>> Stashed changes
    const int bar_x = panel_x + 36;
    const int bar_w = main_width - 80;
    fill_rect(canvas, output_width, output_height, bar_x, dino_bar_y + 3, bar_w, 8, {13, 21, 32});
    if (confidence > 0.0f) {
      fill_rect(canvas, output_width, output_height, bar_x, dino_bar_y + 3,
                static_cast<int>(bar_w * confidence), 8, kOrange);
    }
    std::ostringstream conf_ss;
    conf_ss << static_cast<int>(confidence * 100.0f) << "%";
    draw_text(canvas, output_width, output_height, bar_x + bar_w + 4, dino_bar_y + 3, conf_ss.str(), kOrange, 1);
  }

  fill_rect(canvas, output_width, output_height, sidebar_x, sidebar_y, kSidebarWidth, sidebar_height, {11, 15, 23});
  draw_rect_outline(canvas, output_width, output_height, sidebar_x, sidebar_y, kSidebarWidth, sidebar_height, {62, 74, 92}, 1);
  draw_text(canvas, output_width, output_height, sidebar_x + 16, sidebar_y + 14, "DISPLAY", {232, 236, 241}, 1);
  draw_vertical_slider(canvas, output_width, output_height, sidebar_x + 24, sidebar_y + 34, 104, blue_limit, {90, 148, 255}, "BLUE");
  draw_vertical_slider(canvas, output_width, output_height, sidebar_x + 74, sidebar_y + 34, 104, red_limit, {255, 98, 62}, "RED");
  for (int i = 0; i < 64; ++i) {
    const auto color = heatmap_color(static_cast<float>(63 - i) / 63.0f);
    fill_rect(canvas, output_width, output_height, sidebar_x + 122, sidebar_y + 34 + i * 2, 20, 2, {color[0], color[1], color[2]});
  }

  int text_y = sidebar_y + 156;

  // Channel info section
  draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "CHANNEL INFO", {74, 96, 128}, 1);
  text_y += 14;
  fill_rect(canvas, output_width, output_height, sidebar_x + 16, text_y, kSidebarWidth - 32, 1, {17, 29, 42});
  text_y += 6;

  for (int channel_index = 0; channel_index < active_channels; ++channel_index) {
    const auto& channel = channels[static_cast<size_t>(channel_index)];
    const RgbColor& accent = kAccents[std::min(channel_index, 1)];
    const int ch_num = channel.info.channel >= 0 ? channel.info.channel : channel_index;

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
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "RES", {122, 143, 168}, 1);
    std::ostringstream res_text;
    res_text << std::fixed << std::setprecision(1) << channel.info.resolution_hz / 1000.0f << "KHZ";
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, res_text.str(), {200, 212, 224}, 1);
    text_y += 12;
    draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "DINO", {122, 143, 168}, 1);
    draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y,
              std::to_string(channel.info.dino_chunk_rows) + "X" + std::to_string(channel.info.dino_chunk_cols),
              {200, 212, 224}, 1);
    text_y += 18;
    fill_rect(canvas, output_width, output_height, sidebar_x + 16, text_y, kSidebarWidth - 32, 1, {17, 29, 42});
    text_y += 8;
  }

  // Signal activity section
  draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "SIGNAL ACTIVITY", {74, 96, 128}, 1);
  text_y += 14;
  for (int channel_index = 0; channel_index < active_channels; ++channel_index) {
    const auto& channel = channels[static_cast<size_t>(channel_index)];
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
  text_y += 4;
  fill_rect(canvas, output_width, output_height, sidebar_x + 16, text_y, kSidebarWidth - 32, 1, {17, 29, 42});
  text_y += 8;

  // System metrics section
  draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "SYSTEM", {74, 96, 128}, 1);
  text_y += 14;
  draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "GPU", {122, 143, 168}, 1);
  draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, "RTX 4000 ADA", kGreen, 1);
  text_y += 12;
  draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "DINO LAT", {122, 143, 168}, 1);
  draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, "42 MS", kOrange, 1);
  text_y += 12;
  draw_text(canvas, output_width, output_height, sidebar_x + 16, text_y, "VIS FPS", {122, 143, 168}, 1);
  draw_text(canvas, output_width, output_height, sidebar_x + 60, text_y, "~6 FPS", kGreen, 1);

  const int footer_y = output_height - kFooterHeight + 12;
  // Colored dots
  fill_rect(canvas, output_width, output_height, grid_x, footer_y + 2, 5, 5, kGreen);
  draw_text(canvas, output_width, output_height, grid_x + 9, footer_y, "CLASSIC ANALYZER VIEW", {156, 173, 192}, 1);

  fill_rect(canvas, output_width, output_height, grid_x + 170, footer_y + 2, 5, 5, kDimBlue);
  draw_text(canvas, output_width, output_height, grid_x + 179, footer_y, "DENSITY HEAT UNDER MAX HOLD", {116, 132, 150}, 1);

  fill_rect(canvas, output_width, output_height, grid_x + 360, footer_y + 2, 5, 5, kOrange);
  draw_text(canvas, output_width, output_height, grid_x + 369, footer_y, "DINO DETECTION OVERLAY", {116, 132, 150}, 1);
  
  // Visual drop indicator — shows frame drop rate in footer
  // Orange warning color when dropping, green when keeping up
  
  if (total_frames > 0) {
    const float drop_rate = static_cast<float>(dropped_frames) / static_cast<float>(total_frames);
    const uint64_t live_frames = total_frames - dropped_frames;
    std::ostringstream drop_text;
    const int live_pct = static_cast<int>(std::round((1.0f - drop_rate) * 100.0f));
    drop_text << "VIS " << live_pct << "% LIVE";
    const RgbColor drop_color = drop_rate > 0.1f ? RgbColor{255, 165, 0} : RgbColor{80, 200, 120};
    draw_text(canvas, output_width, output_height, grid_x + 420, footer_y, drop_text.str(), drop_color, 1);
  }
  // Live clock
  {
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

  return canvas;
}

}  // namespace holoscan::ops
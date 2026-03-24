// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "spectrogram.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string make_output_path(const std::string& output_dir,
                             uint16_t channel,
                             uint64_t frame_number,
                             int rows,
                             int cols) {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  std::ostringstream oss;
  oss << output_dir
      << "/spectrogram_ch" << channel
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

}  // namespace

namespace holoscan::ops {

void Spectrogram::setup(holoscan::OperatorSpec& spec) {
  spec.input<in_t>("in");
  spec.output<out_t>("out");

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(enable_save_, "enable_save", "Enable save", "Enable writing spectrogram images to disk.", true);
  spec.param(save_every_n_frames_,
             "save_every_n_frames",
             "Save stride",
             "Save one image every N frames per channel.",
             50);
  spec.param(max_images_per_channel_,
             "max_images_per_channel",
             "Max images per channel",
             "Maximum number of images to save per channel for a run.",
             20);
  spec.param(output_height_,
             "output_height",
             "Output height",
             "Saved spectrogram image height (time axis).",
             256);
  spec.param(output_width_,
             "output_width",
             "Output width",
             "Saved spectrogram image width (frequency axis).",
             512);
  spec.param(output_dir_,
             "output_dir",
             "Output directory",
             "Directory where spectrogram images are written.",
             std::string("/workspace/spectrograms"));
}

void Spectrogram::initialize() {
  holoscan::Operator::initialize();
  frame_count_.assign(num_channels_.get(), 0);
  images_saved_.assign(num_channels_.get(), 0);

  if (enable_save_.get()) {
    fs::create_directories(output_dir_.get());
    HOLOSCAN_LOG_INFO("Spectrogram save enabled. Output dir: {}", output_dir_.get());
  } else {
    HOLOSCAN_LOG_INFO("Spectrogram save disabled.");
  }
}

void Spectrogram::compute(holoscan::InputContext& op_input,
                          holoscan::OutputContext& op_output,
                          holoscan::ExecutionContext&) {
  auto input = op_input.receive<in_t>("in").value();
  auto& tensor = std::get<0>(input);
  auto stream = std::get<1>(input);

  auto meta = metadata();
  uint16_t channel_number = meta->get<uint16_t>("channel_number", 0);

  if (channel_number >= frame_count_.size()) {
    HOLOSCAN_LOG_WARN("Received out-of-range channel {} (configured channels: {}).",
                      channel_number,
                      frame_count_.size());
    return;
  }

  const uint64_t frame_number = ++frame_count_[channel_number];

  op_output.emit(out_t {tensor, stream}, "out");

  if (!enable_save_.get()) {
    return;
  }

  const int save_stride = std::max(1, save_every_n_frames_.get());
  if ((frame_number % static_cast<uint64_t>(save_stride)) != 0) {
    return;
  }

  if (images_saved_[channel_number] >= max_images_per_channel_.get()) {
    return;
  }

  const int src_rows = static_cast<int>(tensor.Size(0));
  const int src_cols = static_cast<int>(tensor.Size(1));
  if (src_rows <= 0 || src_cols <= 0) {
    return;
  }

  const int dst_rows = std::max(1, std::min(output_height_.get(), src_rows));
  const int dst_cols = std::max(1, std::min(output_width_.get(), src_cols));

  std::vector<complex> host_fft(static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols));
  const size_t bytes = host_fft.size() * sizeof(complex);

  auto copy_result = cudaMemcpyAsync(host_fft.data(), tensor.Data(), bytes, cudaMemcpyDeviceToHost, stream);
  if (copy_result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Spectrogram cudaMemcpyAsync failed: {}", cudaGetErrorString(copy_result));
    return;
  }

  auto sync_result = cudaStreamSynchronize(stream);
  if (sync_result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("Spectrogram cudaStreamSynchronize failed: {}", cudaGetErrorString(sync_result));
    return;
  }

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
          const auto& v = host_fft[static_cast<size_t>(rr) * static_cast<size_t>(src_cols) + static_cast<size_t>(cc)];
          const float re = v.real();
          const float im = v.imag();
          const float power = re * re + im * im + 1e-12f;
          accum += 10.0 * std::log10(power);
          ++count;
        }
      }

      reduced[static_cast<size_t>(r) * static_cast<size_t>(dst_cols) + static_cast<size_t>(c)] =
          static_cast<float>(accum / static_cast<double>(std::max(1, count)));
    }
  }

  float min_v = std::numeric_limits<float>::infinity();
  float max_v = -std::numeric_limits<float>::infinity();
  for (float v : reduced) {
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
  }

  const float denom = std::max(1e-6f, max_v - min_v);
  std::vector<uint8_t> image(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols));
  for (size_t i = 0; i < reduced.size(); ++i) {
    const float normalized = (reduced[i] - min_v) / denom;
    image[i] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
  }

  const auto path = make_output_path(output_dir_.get(), channel_number, frame_number, dst_rows, dst_cols);
  if (!write_pgm(path, image, dst_cols, dst_rows)) {
    HOLOSCAN_LOG_ERROR("Failed to write spectrogram image: {}", path);
    return;
  }

  ++images_saved_[channel_number];
  HOLOSCAN_LOG_INFO("Saved spectrogram image for channel {} to {}", channel_number, path);
}

}  // namespace holoscan::ops

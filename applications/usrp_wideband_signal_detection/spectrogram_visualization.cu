#include "spectrogram_visualization.hpp"

#include <cuda/std/detail/libcxx/include/algorithm>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <matx.h>
#include <gxf/std/tensor.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
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

std::vector<uint8_t> grayscale_to_rgb(const std::vector<uint8_t>& grayscale) {
  std::vector<uint8_t> rgb(grayscale.size() * 3);
  for (size_t index = 0; index < grayscale.size(); ++index) {
    const auto color = heatmap_color(static_cast<float>(grayscale[index]) / 255.0f);
    rgb[index * 3] = color[0];
    rgb[index * 3 + 1] = color[1];
    rgb[index * 3 + 2] = color[2];
  }
  return rgb;
}

std::vector<uint8_t> reduce_spectrogram_to_rgb(const SpectrogramTensor& tensor,
                                               cudaStream_t stream,
                                               int output_height,
                                               int output_width) {
  const int src_rows = static_cast<int>(tensor.Size(0));
  const int src_cols = static_cast<int>(tensor.Size(1));
  const int dst_rows = std::max(1, std::min(output_height, src_rows));
  const int dst_cols = std::max(1, std::min(output_width, src_cols));

  std::vector<SpectrogramComplex> host_fft(static_cast<size_t>(src_rows) * static_cast<size_t>(src_cols));
  const size_t bytes = host_fft.size() * sizeof(SpectrogramComplex);

  auto copy_result = cudaMemcpyAsync(host_fft.data(), tensor.Data(), bytes, cudaMemcpyDeviceToHost, stream);
  if (copy_result != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync failed: ") + cudaGetErrorString(copy_result));
  }

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
          const float power = real * real + imag * imag + 1e-12f;
          accumulation += 10.0 * std::log10(power);
          ++count;
        }
      }

      reduced[static_cast<size_t>(row) * static_cast<size_t>(dst_cols) + static_cast<size_t>(col)] =
          static_cast<float>(accumulation / static_cast<double>(std::max(1, count)));
    }
  }

  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  for (float value : reduced) {
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
  }

  const float denominator = std::max(1e-6f, max_value - min_value);
  std::vector<uint8_t> grayscale(reduced.size());
  for (size_t index = 0; index < reduced.size(); ++index) {
    const float normalized = (reduced[index] - min_value) / denominator;
    grayscale[index] = static_cast<uint8_t>(std::clamp(normalized * 255.0f, 0.0f, 255.0f));
  }

  return grayscale_to_rgb(grayscale);
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

  auto copy_result = cudaMemcpy(*device_buffer, rgb.data(), bytes, cudaMemcpyHostToDevice);
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

}  // namespace

namespace holoscan::ops {

void SpectrogramToHolovizOp::setup(OperatorSpec& spec) {
  spec.input<SpectrogramMessage>("in");
  spec.output<gxf::Entity>("outputs");
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
}

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

  const int width = std::max(1, std::min(output_width_.get(), static_cast<int>(tensor.Size(1))));
  const int height = std::max(1, std::min(output_height_.get(), static_cast<int>(tensor.Size(0))));
  auto rgb = reduce_spectrogram_to_rgb(tensor, stream, height, width);

  auto output_entity = create_rgb_entity(context, rgb, width, height, tensor_name_.get());
  op_output.emit(output_entity, "outputs");
}

void OfflinePgmReplayOp::setup(OperatorSpec& spec) {
  spec.output<gxf::Entity>("outputs");
  spec.param(directory_, "directory", "Directory", "Directory containing spectrogram PGM frames.");
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
}

void OfflinePgmReplayOp::initialize() {
  Operator::initialize();

  frames_ = list_pgm_frames(directory_.get(), channel_filter_.get());
  if (frames_.empty()) {
    throw std::runtime_error("No .pgm spectrogram frames found in " + directory_.get());
  }

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

  auto rgb = grayscale_to_rgb(grayscale);
  auto output_entity = create_rgb_entity(context, rgb, width, height, tensor_name_.get());
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

std::vector<uint8_t> colorize_grayscale_spectrogram(const std::vector<uint8_t>& grayscale) {
  return grayscale_to_rgb(grayscale);
}

}  // namespace holoscan::ops
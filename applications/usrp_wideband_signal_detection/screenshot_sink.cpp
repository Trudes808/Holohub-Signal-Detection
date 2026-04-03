#include "spectrogram_visualization.hpp"

#include <cuda_runtime.h>

#include <gxf/multimedia/video.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../gxf_extensions/yuan_qcap/stb/stb_image_write.h"

namespace holoscan::ops {

void RenderBufferScreenshotOp::setup(OperatorSpec& spec) {
  spec.input<gxf::Entity>("input");
  spec.param(output_path_, "output_path", "Output Path", "PNG path for the captured render buffer.");
}

void RenderBufferScreenshotOp::compute(InputContext& op_input,
                                       OutputContext&,
                                       ExecutionContext&) {
  if (saved_) {
    return;
  }

  auto input = op_input.receive<gxf::Entity>("input").value();
  const auto& buffer = static_cast<nvidia::gxf::Entity>(input).get<nvidia::gxf::VideoBuffer>();
  if (!buffer) {
    throw std::runtime_error("No render buffer attached to screenshot input");
  }

  const auto& info = buffer.value()->video_frame_info();
  if (info.color_format != nvidia::gxf::VideoFormat::GXF_VIDEO_FORMAT_RGBA) {
    throw std::runtime_error("Render buffer screenshot currently expects RGBA output");
  }

  std::vector<uint8_t> data(buffer.value()->size());
  if (buffer.value()->storage_type() == nvidia::gxf::MemoryStorageType::kHost) {
    std::memcpy(data.data(), buffer.value()->pointer(), data.size());
  } else {
    auto copy_result = cudaMemcpy(data.data(), buffer.value()->pointer(), data.size(), cudaMemcpyDeviceToHost);
    if (copy_result != cudaSuccess) {
      throw std::runtime_error(std::string("cudaMemcpy failed while saving screenshot: ") + cudaGetErrorString(copy_result));
    }
  }

  const std::filesystem::path output_path(output_path_.get());
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }

  const int stride = static_cast<int>(info.color_planes[0].stride);
  if (!stbi_write_png(output_path.string().c_str(),
                      static_cast<int>(info.width),
                      static_cast<int>(info.height),
                      4,
                      data.data(),
                      stride)) {
    throw std::runtime_error("Failed to write screenshot PNG to " + output_path.string());
  }

  saved_ = true;
  HOLOSCAN_LOG_INFO("Saved screenshot to {}", output_path.string());
}

}  // namespace holoscan::ops
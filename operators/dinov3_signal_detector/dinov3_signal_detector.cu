// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "dinov3_signal_detector.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>

#ifdef HOLOHUB_HAS_TORCH
#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace {

__global__ void power_db_mask_kernel(const cuda::std::complex<float>* input,
                                     float* output,
                                     int src_rows,
                                     int src_cols,
                                     int dst_rows,
                                     int dst_cols,
                                     float threshold_db) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = dst_rows * dst_cols;
  if (idx >= total) {
    return;
  }

  const int r = idx / dst_cols;
  const int c = idx % dst_cols;

  const int src_r = min((r * src_rows) / dst_rows, src_rows - 1);
  const int src_c = min((c * src_cols) / dst_cols, src_cols - 1);

  const auto v = input[src_r * src_cols + src_c];
  const float re = v.real();
  const float im = v.imag();
  const float power = re * re + im * im + 1e-12f;
  const float power_db = 10.0f * log10f(power);

  output[idx] = (power_db >= threshold_db) ? 1.0f : 0.0f;
}

}  // namespace

namespace holoscan::ops {

void DinoV3SignalDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<dino_in_t>("in");
  spec.output<dino_out_t>("out");

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(input_height_,
             "input_height",
             "Input height",
             "Detector input height (time bins).",
             256);
  spec.param(input_width_,
             "input_width",
             "Input width",
             "Detector input width (frequency bins).",
             512);
  spec.param(emit_stride_,
             "emit_stride",
             "Emit stride",
             "Emit one output every N input frames per channel.",
             1);
  spec.param(mask_threshold_db_,
             "mask_threshold_db",
             "Mask threshold (dB)",
             "Power threshold in dB used for baseline signal mask generation.",
             -20.0f);
  spec.param(log_detections_,
             "log_detections",
             "Log detections",
             "If true, logs detector execution details.",
             false);
  spec.param(use_pytorch_backend_,
             "use_pytorch_backend",
             "Use PyTorch backend",
             "If true, uses PyTorch tensor operations on GPU as detector path when available.",
             true);
  spec.param(model_name_,
             "model_name",
             "Model name",
             "DINOv3 model name placeholder for future integration.",
             std::string("dinov3_vitb16"));
  spec.param(model_repo_path_,
             "model_repo_path",
             "Model repo path",
             "Path to local DINOv3 repository (placeholder for future integration).",
             std::string("/workspace/models/dinov3"));
  spec.param(weights_path_,
             "weights_path",
             "Weights path",
             "Path to model weights placeholder while downloads complete.",
             std::string("/workspace/models/dinov3/weights/dinov3_vitb16_placeholder.pth"));
}

void DinoV3SignalDetector::initialize() {
  holoscan::Operator::initialize();

  frame_count_.assign(num_channels_.get(), 0);

  make_tensor(detection_masks_,
              {num_channels_.get(), input_height_.get(), input_width_.get()},
              MATX_DEVICE_MEMORY);

  pytorch_runtime_ready_ = false;

#ifdef HOLOHUB_HAS_TORCH
  if (use_pytorch_backend_.get()) {
    if (!torch::cuda::is_available()) {
      HOLOSCAN_LOG_WARN("PyTorch backend requested, but torch CUDA is not available. Falling back to CUDA kernel path.");
    } else {
      pytorch_runtime_ready_ = true;
      HOLOSCAN_LOG_INFO("PyTorch backend enabled for DINOv3 detector. model_name='{}' repo='{}' weights='{}'",
                        model_name_.get(),
                        model_repo_path_.get(),
                        weights_path_.get());

      if (!std::filesystem::exists(weights_path_.get())) {
        HOLOSCAN_LOG_WARN("weights path does not exist: {}", weights_path_.get());
      }
    }
  }
#else
  if (use_pytorch_backend_.get()) {
    HOLOSCAN_LOG_WARN("PyTorch backend requested, but operator was built without Torch. Falling back to CUDA kernel path.");
  }
#endif
}

void DinoV3SignalDetector::compute(holoscan::InputContext& op_input,
                                   holoscan::OutputContext& op_output,
                                   holoscan::ExecutionContext&) {
  auto input = op_input.receive<dino_in_t>("in").value();
  auto& fft_tensor = std::get<0>(input);
  auto stream = std::get<1>(input);

  auto meta = metadata();
  const uint16_t channel_number = meta->get<uint16_t>("channel_number", 0);

  if (channel_number >= frame_count_.size()) {
    HOLOSCAN_LOG_WARN("DINOv3 detector received out-of-range channel {} (configured channels: {}).",
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

  if (src_rows <= 0 || src_cols <= 0) {
    HOLOSCAN_LOG_WARN("DINOv3 detector received empty tensor on channel {}", channel_number);
    return;
  }

  auto out = matx::slice<2>(detection_masks_,
                            {static_cast<matx::index_t>(channel_number), 0, 0},
                            {matxDropDim, matxEnd, matxEnd});

  auto clear_result = cudaMemsetAsync(
      out.Data(), 0, static_cast<size_t>(input_height_.get()) * static_cast<size_t>(input_width_.get()) * sizeof(float), stream);
  if (clear_result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("DINOv3 detector cudaMemsetAsync failed: {}", cudaGetErrorString(clear_result));
    return;
  }

  const int dst_rows = std::max(1, input_height_.get());
  const int dst_cols = std::max(1, input_width_.get());

#ifdef HOLOHUB_HAS_TORCH
  if (use_pytorch_backend_.get() && pytorch_runtime_ready_) {
    const int64_t output_elements = static_cast<int64_t>(dst_rows) * static_cast<int64_t>(dst_cols);
    auto output_bytes = static_cast<size_t>(output_elements) * sizeof(float);

    c10::cuda::CUDAStream external_stream = c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device());
    c10::cuda::CUDAStreamGuard stream_guard(external_stream);

    auto complex_options = torch::TensorOptions().dtype(torch::kComplexFloat).device(torch::kCUDA);
    auto complex_input = torch::from_blob(fft_tensor.Data(),
                                          {static_cast<int64_t>(src_rows), static_cast<int64_t>(src_cols)},
                                          complex_options);

    auto power = torch::pow(torch::abs(complex_input), 2).add(1e-12);
    auto power_db = 10.0 * torch::log10(power);

    auto resized = torch::nn::functional::interpolate(
        power_db.unsqueeze(0).unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions()
            .size(std::vector<int64_t>{static_cast<int64_t>(dst_rows), static_cast<int64_t>(dst_cols)})
            .mode(torch::kBilinear)
            .align_corners(false));

    auto mask = resized.squeeze(0).squeeze(0).ge(mask_threshold_db_.get()).to(torch::kFloat32).contiguous();

    auto copy_result = cudaMemcpyAsync(out.Data(),
                                       mask.data_ptr<float>(),
                                       output_bytes,
                                       cudaMemcpyDeviceToDevice,
                                       stream);
    if (copy_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("DINOv3 detector torch-path memcpy failed: {}", cudaGetErrorString(copy_result));
      return;
    }

    meta->set("dino_frame_number", frame_number);
    meta->set("dino_mask_height", static_cast<uint32_t>(dst_rows));
    meta->set("dino_mask_width", static_cast<uint32_t>(dst_cols));
    meta->set("dino_mask_threshold_db", mask_threshold_db_.get());
    meta->set("dino_backend", std::string("pytorch_placeholder"));
    meta->set("dino_model_name", model_name_.get());
    meta->set("dino_weights_path", weights_path_.get());

    if (log_detections_.get()) {
      HOLOSCAN_LOG_INFO("DINOv3 detector (PyTorch path) emitted mask for channel {} frame {} shape {}x{}",
                        channel_number,
                        frame_number,
                        dst_rows,
                        dst_cols);
    }

    op_output.emit(dino_out_t {out, stream}, "out");
    return;
  }
#else
  if (use_pytorch_backend_.get() && !pytorch_warning_emitted_) {
    HOLOSCAN_LOG_WARN("DINOv3 detector use_pytorch_backend=true but Torch is unavailable at build time; using CUDA fallback path.");
    pytorch_warning_emitted_ = true;
  }
#endif

  const int total = dst_rows * dst_cols;
  const int threads = 256;
  const int blocks = (total + threads - 1) / threads;

  power_db_mask_kernel<<<blocks, threads, 0, stream>>>(fft_tensor.Data(),
                                                        out.Data(),
                                                        src_rows,
                                                        src_cols,
                                                        dst_rows,
                                                        dst_cols,
                                                        mask_threshold_db_.get());

  auto kernel_result = cudaGetLastError();
  if (kernel_result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("DINOv3 detector kernel launch failed: {}", cudaGetErrorString(kernel_result));
    return;
  }

  meta->set("dino_frame_number", frame_number);
  meta->set("dino_mask_height", static_cast<uint32_t>(dst_rows));
  meta->set("dino_mask_width", static_cast<uint32_t>(dst_cols));
  meta->set("dino_mask_threshold_db", mask_threshold_db_.get());
  meta->set("dino_backend", std::string("cuda_threshold_fallback"));
  meta->set("dino_model_name", model_name_.get());
  meta->set("dino_weights_path", weights_path_.get());

  if (log_detections_.get()) {
    HOLOSCAN_LOG_INFO("DINOv3 detector emitted mask for channel {} frame {} with shape {}x{}",
                      channel_number,
                      frame_number,
                      dst_rows,
                      dst_cols);
  }

  op_output.emit(dino_out_t {out, stream}, "out");
}

}  // namespace holoscan::ops

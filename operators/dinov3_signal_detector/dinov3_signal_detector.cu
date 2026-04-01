// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "dinov3_signal_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <vector>

#ifdef HOLOHUB_HAS_TORCH
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif

namespace {

std::string make_mask_output_path(const std::string& output_dir,
                                  uint16_t channel,
                                  uint64_t frame_number,
                                  int rows,
                                  int cols) {
  const auto now = std::chrono::system_clock::now();
  const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  std::ostringstream oss;
  oss << output_dir
      << "/dino_mask_ch" << channel
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

bool is_valid_torchscript_init_mode(const std::string& init_mode) {
  return init_mode == "load_only" ||
         init_mode == "load_cpu_eval" ||
         init_mode == "load_cuda_no_eval" ||
         init_mode == "load_cuda_eval";
}

std::string normalize_torchscript_init_mode(const std::string& init_mode) {
  if (is_valid_torchscript_init_mode(init_mode)) {
    return init_mode;
  }

  return "load_cuda_eval";
}

bool torchscript_init_moves_to_cuda(const std::string& init_mode) {
  return init_mode == "load_cuda_no_eval" || init_mode == "load_cuda_eval";
}

bool torchscript_init_runs_eval(const std::string& init_mode) {
  return init_mode == "load_cpu_eval" || init_mode == "load_cuda_eval";
}

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
  spec.param(enable_mask_save_,
             "enable_mask_save",
             "Enable mask save",
             "Enable writing detector masks to disk for debug runs.",
             false);
  spec.param(save_every_n_frames_,
             "save_every_n_frames",
             "Save stride",
             "Save one detector mask every N frames per channel.",
             1);
  spec.param(max_masks_per_channel_,
             "max_masks_per_channel",
             "Max masks per channel",
             "Maximum number of detector masks to save per channel for a run.",
             5);
  spec.param(output_dir_,
             "output_dir",
             "Output directory",
             "Directory where detector masks are written.",
             std::string("/workspace/dino_masks"));
  spec.param(use_pytorch_backend_,
             "use_pytorch_backend",
             "Use PyTorch backend",
             "If true, uses PyTorch tensor operations on GPU as detector path when available.",
             true);
  spec.param(inference_backend_,
             "inference_backend",
             "Inference backend",
             "Backend mode: torchscript, pytorch_placeholder, or cuda_threshold_fallback.",
             std::string("torchscript"));
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
  spec.param(model_script_path_,
             "model_script_path",
             "Model script path",
             "Path to TorchScript model for model-forward backend.",
             std::string("/workspace/models/dinov3/weights/dinov3_vitb16_placeholder.ts"));
  spec.param(torchscript_init_mode_,
             "torchscript_init_mode",
             "TorchScript init mode",
             "TorchScript initialization mode: load_only, load_cpu_eval, load_cuda_no_eval, or load_cuda_eval.",
             std::string("load_cuda_eval"));
  spec.param(strict_model_forward_,
             "strict_model_forward",
             "Strict model forward",
             "If true, drop frames when torchscript forward fails instead of falling back.",
             false);
}

void DinoV3SignalDetector::initialize() {
  holoscan::Operator::initialize();

  frame_count_.assign(num_channels_.get(), 0);
  masks_saved_.assign(num_channels_.get(), 0);

  make_tensor(detection_masks_,
              {num_channels_.get(), input_height_.get(), input_width_.get()},
              MATX_DEVICE_MEMORY);

  if (enable_mask_save_.get()) {
    std::filesystem::create_directories(output_dir_.get());
    HOLOSCAN_LOG_INFO("DINO mask save enabled. Output dir: {}", output_dir_.get());
  }

  pytorch_runtime_ready_ = false;
  torchscript_model_loaded_ = false;
  torchscript_forward_ready_ = false;
  torchscript_forward_warning_emitted_ = false;

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

      if (inference_backend_.get() == "torchscript") {
        if (!std::filesystem::exists(model_script_path_.get())) {
          HOLOSCAN_LOG_WARN("TorchScript model path not found (placeholder expected while downloading): {}",
                            model_script_path_.get());
        } else {
          const auto requested_init_mode = torchscript_init_mode_.get();
          const auto init_mode = normalize_torchscript_init_mode(requested_init_mode);
          if (init_mode != requested_init_mode) {
            HOLOSCAN_LOG_WARN("Unknown torchscript_init_mode='{}'. Falling back to '{}'.",
                              requested_init_mode,
                              init_mode);
          }

          try {
            HOLOSCAN_LOG_INFO("TorchScript init start: script='{}' mode='{}'",
                              model_script_path_.get(),
                              init_mode);
            HOLOSCAN_LOG_INFO("TorchScript init stage: load");
            auto loaded_module = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_script_path_.get()));
            HOLOSCAN_LOG_INFO("TorchScript init stage complete: load");

            if (torchscript_init_moves_to_cuda(init_mode)) {
              HOLOSCAN_LOG_INFO("TorchScript init stage: to_cuda");
              loaded_module->to(torch::kCUDA);
              HOLOSCAN_LOG_INFO("TorchScript init stage complete: to_cuda");
            } else {
              HOLOSCAN_LOG_INFO("TorchScript init skipping CUDA transfer for mode '{}'", init_mode);
            }

            if (torchscript_init_runs_eval(init_mode)) {
              HOLOSCAN_LOG_INFO("TorchScript init stage: eval");
              loaded_module->eval();
              HOLOSCAN_LOG_INFO("TorchScript init stage complete: eval");
            } else {
              HOLOSCAN_LOG_INFO("TorchScript init skipping eval for mode '{}'", init_mode);
            }

            torchscript_module_ = std::move(loaded_module);
            torchscript_model_loaded_ = true;
            torchscript_forward_ready_ = torchscript_init_moves_to_cuda(init_mode);
            HOLOSCAN_LOG_INFO("Loaded TorchScript model from {} (mode='{}', forward_ready={})",
                              model_script_path_.get(),
                              init_mode,
                              torchscript_forward_ready_ ? "true" : "false");
          } catch (const c10::Error& error) {
            HOLOSCAN_LOG_WARN("Failed to load TorchScript model '{}': {}",
                              model_script_path_.get(),
                              error.what());
          } catch (const std::exception& error) {
            HOLOSCAN_LOG_WARN("TorchScript init failed for '{}': {}",
                              model_script_path_.get(),
                              error.what());
          }
        }
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
                                   holoscan::OutputContext&,
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

  auto maybe_save_mask = [&](const std::string& backend_name) {
    if (!enable_mask_save_.get()) {
      return;
    }

    const int save_stride = std::max(1, save_every_n_frames_.get());
    if ((frame_number % static_cast<uint64_t>(save_stride)) != 0) {
      return;
    }

    if (masks_saved_[channel_number] >= max_masks_per_channel_.get()) {
      return;
    }

    std::vector<float> host_mask(static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols), 0.0f);
    const size_t output_bytes = host_mask.size() * sizeof(float);
    auto copy_result = cudaMemcpyAsync(host_mask.data(), out.Data(), output_bytes, cudaMemcpyDeviceToHost, stream);
    if (copy_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("DINO mask cudaMemcpyAsync failed: {}", cudaGetErrorString(copy_result));
      return;
    }

    auto sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("DINO mask cudaStreamSynchronize failed: {}", cudaGetErrorString(sync_result));
      return;
    }

    std::vector<uint8_t> image(host_mask.size(), 0);
    for (size_t idx = 0; idx < host_mask.size(); ++idx) {
      image[idx] = host_mask[idx] > 0.5f ? 255 : 0;
    }

    const auto path = make_mask_output_path(output_dir_.get(), channel_number, frame_number, dst_rows, dst_cols);
    if (!write_pgm(path, image, dst_cols, dst_rows)) {
      HOLOSCAN_LOG_ERROR("Failed to write detector mask image: {}", path);
      return;
    }

    ++masks_saved_[channel_number];
    HOLOSCAN_LOG_INFO("Saved DINO mask for channel {} frame {} ({}) to {}",
                      channel_number,
                      frame_number,
                      backend_name,
                      path);
  };

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

    auto backend = inference_backend_.get();
    torch::Tensor mask;

    if (backend == "torchscript" && torchscript_model_loaded_ && !torchscript_forward_ready_) {
      if (!torchscript_forward_warning_emitted_) {
        HOLOSCAN_LOG_WARN("TorchScript model loaded with init mode '{}' but not marked forward-ready. Falling back to pytorch_placeholder path.",
                          torchscript_init_mode_.get());
        torchscript_forward_warning_emitted_ = true;
      }
      backend = "pytorch_placeholder";
    }

    if (backend == "torchscript" && torchscript_model_loaded_ && torchscript_forward_ready_) {
      try {
        std::vector<torch::jit::IValue> model_inputs;
        model_inputs.emplace_back(resized.to(torch::kFloat32));

        auto model_output = torchscript_module_->forward(model_inputs);
        torch::Tensor logits;

        if (model_output.isTensor()) {
          logits = model_output.toTensor();
        } else if (model_output.isTuple()) {
          auto tuple_ptr = model_output.toTuple();
          if (tuple_ptr && !tuple_ptr->elements().empty() && tuple_ptr->elements()[0].isTensor()) {
            logits = tuple_ptr->elements()[0].toTensor();
          }
        }

        if (!logits.defined()) {
          throw std::runtime_error("TorchScript forward returned non-tensor output");
        }

        if (logits.dim() == 4) {
          logits = logits.squeeze(0).squeeze(0);
        } else if (logits.dim() == 3) {
          logits = logits.squeeze(0);
        }

        if (logits.sizes().size() != 2 ||
            logits.size(0) != static_cast<int64_t>(dst_rows) ||
            logits.size(1) != static_cast<int64_t>(dst_cols)) {
          logits = torch::nn::functional::interpolate(
                       logits.unsqueeze(0).unsqueeze(0),
                       torch::nn::functional::InterpolateFuncOptions()
                           .size(std::vector<int64_t>{static_cast<int64_t>(dst_rows), static_cast<int64_t>(dst_cols)})
                           .mode(torch::kBilinear)
                           .align_corners(false))
                       .squeeze(0)
                       .squeeze(0);
        }

        mask = logits.ge(mask_threshold_db_.get()).to(torch::kFloat32).contiguous();
        backend = "torchscript";
      } catch (const std::exception& error) {
        HOLOSCAN_LOG_WARN("TorchScript forward failed; {}. Falling back to pytorch_placeholder path.", error.what());
        if (strict_model_forward_.get()) {
          return;
        }
        mask = resized.squeeze(0).squeeze(0).ge(mask_threshold_db_.get()).to(torch::kFloat32).contiguous();
        backend = "pytorch_placeholder";
      }
    } else {
      mask = resized.squeeze(0).squeeze(0).ge(mask_threshold_db_.get()).to(torch::kFloat32).contiguous();
      backend = "pytorch_placeholder";
    }

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
    meta->set("dino_backend", backend);
    meta->set("dino_model_name", model_name_.get());
    meta->set("dino_weights_path", weights_path_.get());
    meta->set("dino_model_script_path", model_script_path_.get());
    meta->set("dino_torchscript_init_mode", torchscript_init_mode_.get());
    meta->set("dino_torchscript_forward_ready", torchscript_forward_ready_);

    if (log_detections_.get()) {
      HOLOSCAN_LOG_INFO("DINOv3 detector ({} path) emitted mask for channel {} frame {} shape {}x{}",
                        backend,
                        channel_number,
                        frame_number,
                        dst_rows,
                        dst_cols);
    }

    maybe_save_mask(backend);

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
  meta->set("dino_model_script_path", model_script_path_.get());

  if (log_detections_.get()) {
    HOLOSCAN_LOG_INFO("DINOv3 detector emitted mask for channel {} frame {} with shape {}x{}",
                      channel_number,
                      frame_number,
                      dst_rows,
                      dst_cols);
  }

  maybe_save_mask("cuda_threshold_fallback");

}

}  // namespace holoscan::ops

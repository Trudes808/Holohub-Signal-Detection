// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "dinov3_signal_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

enum TimingStageIndex : size_t {
  kInputStage = 0,
  kFrontendCorrectionStage,
  kCropAlignStage,
  kResizeStage,
  kModelPrepStage,
  kTorchForwardStage,
  kDinoScoreStage,
  kPowerScoreStage,
  kFusionStage,
  kDeviceCopyStage,
  kMaskSaveStage,
  kTotalStage,
};

constexpr std::array<const char*, holoscan::ops::DinoV3SignalDetector::kTimingStageCount> kTimingStageNames = {
    "input_ms",
    "frontend_correction_ms",
    "crop_align_ms",
    "resize_ms",
    "model_prep_ms",
    "torch_forward_ms",
    "dino_score_ms",
    "power_score_ms",
    "fusion_ms",
    "device_copy_ms",
    "mask_save_ms",
    "total_ms",
};

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

__global__ void complex_to_power_db_kernel(const cuda::std::complex<float>* input,
                                           float* output,
                                           int src_rows,
                                           int src_cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = src_rows * src_cols;
  if (idx >= total) {
    return;
  }

  const auto v = input[idx];
  const float re = v.real();
  const float im = v.imag();
  const float power = re * re + im * im + 1e-12f;
  output[idx] = 10.0f * log10f(power);
}

}  // namespace

namespace holoscan::ops {

void DinoV3SignalDetector::setup(holoscan::OperatorSpec& spec) {
  spec.input<dino_in_t>("in");

  spec.param(num_channels_, "num_channels", "Number of channels", "Number of channels in the stream.", 1);
  spec.param(input_height_,
             "input_height",
             "Input height",
             "Detector output height.",
             256);
  spec.param(input_width_,
             "input_width",
             "Input width",
             "Detector output width.",
             512);
  spec.param(patch_size_,
             "patch_size",
             "Patch size",
             "Patch size used for DINO-aligned input shaping.",
             16);
  spec.param(emit_stride_,
             "emit_stride",
             "Emit stride",
             "Emit one output every N input frames per channel.",
             1);
  spec.param(mask_threshold_db_,
             "mask_threshold_db",
             "Mask threshold (dB)",
             "Legacy fallback threshold in dB used for baseline signal mask generation.",
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
             "If true, uses the LibTorch-based notebook reproduction path when available.",
             true);
  spec.param(inference_backend_,
             "inference_backend",
             "Inference backend",
             "Backend mode: torchscript, pytorch_placeholder, or cuda_threshold_fallback.",
             std::string("torchscript"));
  spec.param(model_name_,
             "model_name",
             "Model name",
             "DINOv3 model name.",
             std::string("dinov3_vitb16"));
  spec.param(model_repo_path_,
             "model_repo_path",
             "Model repo path",
             "Path to local DINOv3 repository.",
             std::string("/workspace/models/dinov3"));
  spec.param(weights_path_,
             "weights_path",
             "Weights path",
             "Path to model weights.",
             std::string("/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"));
  spec.param(model_script_path_,
             "model_script_path",
             "Model script path",
             "Path to TorchScript model for model-forward backend.",
             std::string("/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts"));
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
  spec.param(imagenet_mean_,
             "imagenet_mean",
             "ImageNet mean",
             "Mean used for notebook-aligned model normalization.",
             std::vector<double>{0.485, 0.456, 0.406});
  spec.param(imagenet_std_,
             "imagenet_std",
             "ImageNet std",
             "Standard deviation used for notebook-aligned model normalization.",
             std::vector<double>{0.229, 0.224, 0.225});
  spec.param(fft_size_, "fft_size", "FFT size", "Notebook-derived FFT size constant for metadata and parity tracking.", 1024);
  spec.param(noverlap_, "noverlap", "FFT overlap", "Notebook-derived overlap constant for parity tracking.", 256);
  spec.param(ignore_sideband_hz_,
             "ignore_sideband_hz",
             "Ignore sideband Hz",
             "Frequency span to ignore on each side of the spectrum before DINO preprocessing.",
             7.0e6);
  spec.param(frontend_correction_enable_,
             "frontend_correction_enable",
             "Frontend correction enable",
             "Enable notebook-inspired frontend correction before DINO preprocessing.",
             true);
  spec.param(frontend_correction_row_q_, "frontend_correction_row_q", "Frontend correction row quantile", "Notebook-derived frontend correction row quantile.", 25.0);
  spec.param(frontend_correction_smooth_sigma_, "frontend_correction_smooth_sigma", "Frontend correction smoothing sigma", "Notebook-derived frontend correction smoothing sigma.", 12.0);
  spec.param(frontend_correction_reference_q_, "frontend_correction_reference_q", "Frontend correction reference quantile", "Notebook-derived frontend correction reference quantile.", 75.0);
  spec.param(frontend_correction_max_boost_db_, "frontend_correction_max_boost_db", "Frontend correction max boost", "Notebook-derived max frontend correction boost in dB.", 12.0);
  spec.param(frontend_correction_soft_knee_db_, "frontend_correction_soft_knee_db", "Frontend correction soft knee", "Notebook-derived frontend correction soft knee in dB.", 4.0);
  spec.param(frontend_correction_edge_taper_fraction_, "frontend_correction_edge_taper_fraction", "Frontend correction edge taper fraction", "Notebook-derived edge taper fraction.", 0.10);
  spec.param(frontend_correction_edge_taper_sigma_, "frontend_correction_edge_taper_sigma", "Frontend correction edge taper sigma", "Notebook-derived edge taper sigma.", 6.0);
  spec.param(frontend_correction_edge_target_drop_db_, "frontend_correction_edge_target_drop_db", "Frontend correction edge target drop", "Notebook-derived edge target drop in dB.", 2.5);
  spec.param(frontend_edge_guard_floor_, "frontend_edge_guard_floor", "Frontend edge guard floor", "Notebook-derived frontend edge guard floor.", 0.35);
  spec.param(dino_coherence_gate_floor_, "dino_coherence_gate_floor", "DINO coherence gate floor", "Notebook-derived DINO coherence gate floor.", 0.25);
  spec.param(texture_q_, "texture_q", "Texture quantile", "Notebook-derived texture quantile constant.", 0.90);
  spec.param(texture_k_, "texture_k", "Texture K", "Notebook-derived texture neighborhood size.", 6);
  spec.param(power_q_, "power_q", "Power quantile", "Notebook-derived power quantile constant.", 0.90);
  spec.param(dino_group_k_, "dino_group_k", "DINO grouping K", "Notebook-derived DINO grouping neighborhood size.", 8);
  spec.param(dino_group_spatial_weight_, "dino_group_spatial_weight", "DINO grouping spatial weight", "Notebook-derived DINO grouping spatial weight.", 0.35);
  spec.param(dino_group_score_q_, "dino_group_score_q", "DINO grouping score quantile", "Notebook-derived DINO grouping score quantile.", 0.60);
  spec.param(pipeline_final_threshold_, "pipeline_final_threshold", "Pipeline final threshold", "Notebook-derived final threshold when speckle cleanup is active.", 0.20);
  spec.param(pipeline_final_threshold_no_speckle_, "pipeline_final_threshold_no_speckle", "Pipeline final threshold without speckle", "Notebook-derived final threshold when speckle cleanup is inactive.", 0.10);
  spec.param(pipeline_gap_floor_, "pipeline_gap_floor", "Pipeline gap floor", "Notebook-derived gap floor constant.", 0.10);
  spec.param(pipeline_component_min_size_, "pipeline_component_min_size", "Pipeline minimum component size", "Notebook-derived minimum component size.", 5);
  spec.param(pipeline_component_min_size_no_speckle_, "pipeline_component_min_size_no_speckle", "Pipeline minimum component size without speckle", "Notebook-derived minimum component size without speckle cleanup.", 2);
  spec.param(pipeline_power_rescue_floor_, "pipeline_power_rescue_floor", "Pipeline power rescue floor", "Notebook-derived power rescue floor.", 0.10);
  spec.param(pipeline_power_rescue_gain_, "pipeline_power_rescue_gain", "Pipeline power rescue gain", "Notebook-derived power rescue gain.", 2.0);
  spec.param(pipeline_strong_speckle_min_component_, "pipeline_strong_speckle_min_component", "Pipeline strong speckle minimum component", "Notebook-derived strong speckle minimum component size.", 10);
  spec.param(pipeline_texture_speckle_clean_threshold_, "pipeline_texture_speckle_clean_threshold", "Pipeline texture speckle clean threshold", "Notebook-derived texture cleanup threshold.", 0.85);
  spec.param(pipeline_texture_speckle_strong_threshold_, "pipeline_texture_speckle_strong_threshold", "Pipeline texture strong threshold", "Notebook-derived texture strong threshold.", 0.20);
  spec.param(timing_summary_enable_,
             "timing_summary_enable",
             "Timing summary enable",
             "Enable per-stage detector timing summaries.",
             true);
  spec.param(timing_summary_every_n_,
             "timing_summary_every_n",
             "Timing summary every N",
             "Emit timing summaries every N emitted detector frames per channel.",
             16);
  spec.param(timing_summary_window_,
             "timing_summary_window",
             "Timing summary window",
             "Maximum number of emitted detector frames to accumulate before a timing summary reset.",
             16);
}

void DinoV3SignalDetector::initialize() {
  holoscan::Operator::initialize();

  frame_count_.assign(num_channels_.get(), 0);
  masks_saved_.assign(num_channels_.get(), 0);
  timing_stats_.assign(num_channels_.get(), ChannelTimingStats {});
  power_db_device_buffers_.assign(num_channels_.get(), nullptr);
  power_db_device_buffer_sizes_.assign(num_channels_.get(), 0);

  make_tensor(detection_masks_,
              {num_channels_.get(), input_height_.get(), input_width_.get()},
              MATX_DEVICE_MEMORY);

  if (enable_mask_save_.get()) {
    std::filesystem::create_directories(output_dir_.get());
    HOLOSCAN_LOG_INFO("DINO mask save enabled. Output dir: {}", output_dir_.get());
  }

  pytorch_runtime_ready_ = false;
  pytorch_warning_emitted_ = false;
  torchscript_forward_trace_emitted_ = false;

#ifdef HOLOHUB_HAS_TORCH
  torch_runtime_.reset();
  if (use_pytorch_backend_.get()) {
    pytorch_runtime_ready_ = true;
    torch_runtime_ = std::make_unique<DinoTorchRuntime>();
    HOLOSCAN_LOG_INFO("PyTorch backend enabled for DINOv3 detector. model_name='{}' repo='{}' weights='{}' script='{}'",
                      model_name_.get(),
                      model_repo_path_.get(),
                      weights_path_.get(),
                      model_script_path_.get());

    if (inference_backend_.get() == "torchscript") {
      if (!std::filesystem::exists(model_script_path_.get())) {
        HOLOSCAN_LOG_WARN("TorchScript model path not found: {}", model_script_path_.get());
      } else {
        const auto requested_init_mode = torchscript_init_mode_.get();
        const auto init_mode = normalize_torchscript_init_mode(requested_init_mode);
        if (init_mode != requested_init_mode) {
          HOLOSCAN_LOG_WARN("Unknown torchscript_init_mode='{}'. Falling back to '{}'.",
                            requested_init_mode,
                            init_mode);
      }
        HOLOSCAN_LOG_INFO("TorchScript load deferred until first compute: script='{}' mode='{}'",
                          model_script_path_.get(),
                          init_mode);
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
  const int dst_rows = std::max(1, input_height_.get());
  const int dst_cols = std::max(1, input_width_.get());
  const int patch_size = std::max(1, patch_size_.get());
  const bool timing_enabled = timing_summary_enable_.get();
  const bool need_device_mask = enable_mask_save_.get();
  const auto total_start = std::chrono::steady_clock::now();
  std::array<double, kTimingStageCount> stage_ms {};

  if (src_rows <= 0 || src_cols <= 0) {
    HOLOSCAN_LOG_WARN("DINOv3 detector received empty tensor on channel {}", channel_number);
    return;
  }

  auto time_step_ms = [&](size_t stage_index, auto&& fn) {
    if (!timing_enabled) {
      fn();
      return;
    }

    const auto stage_start = std::chrono::steady_clock::now();
    fn();
    auto sync_result = cudaStreamSynchronize(stream);
    if (sync_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("DINOv3 detector timing sync failed at {}: {}",
                         kTimingStageNames[stage_index],
                         cudaGetErrorString(sync_result));
      return;
    }

    stage_ms[stage_index] = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - stage_start)
                                .count();
  };

  auto out = matx::slice<2>(detection_masks_,
                            {static_cast<matx::index_t>(channel_number), 0, 0},
                            {matxDropDim, matxEnd, matxEnd});

  if (need_device_mask) {
    time_step_ms(kInputStage, [&] {
      auto clear_result = cudaMemsetAsync(out.Data(),
                                          0,
                                          static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols) * sizeof(float),
                                          stream);
      if (clear_result != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemsetAsync failed: ") + cudaGetErrorString(clear_result));
      }
    });
  } else {
    time_step_ms(kInputStage, [&] {});
  }

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

  std::string backend_used = "cuda_threshold_fallback";
  int ignore_bins_per_side = 0;
  double freq_bin_hz = 0.0;
  int aligned_rows = dst_rows;
  int aligned_cols = dst_cols;
  double dino_threshold = 0.0;
  double power_threshold = 0.0;
  double final_threshold = pipeline_final_threshold_.get();

  auto run_cuda_fallback = [&](const std::string& pipeline_variant) -> bool {
    const int total = dst_rows * dst_cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    time_step_ms(kTorchForwardStage, [&] {});
    time_step_ms(kDinoScoreStage, [&] {});
    time_step_ms(kPowerScoreStage, [&] {});
    time_step_ms(kFusionStage, [&] {
      if (!need_device_mask) {
        return;
      }
      power_db_mask_kernel<<<blocks, threads, 0, stream>>>(fft_tensor.Data(),
                                                            out.Data(),
                                                            src_rows,
                                                            src_cols,
                                                            dst_rows,
                                                            dst_cols,
                                                            mask_threshold_db_.get());
      auto kernel_result = cudaGetLastError();
      if (kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("kernel launch failed: ") + cudaGetErrorString(kernel_result));
      }
    });
    time_step_ms(kDeviceCopyStage, [&] {});
    time_step_ms(kMaskSaveStage, [&] { maybe_save_mask("cuda_threshold_fallback"); });

    meta->set("dino_frame_number", frame_number);
    meta->set("dino_mask_height", static_cast<uint32_t>(dst_rows));
    meta->set("dino_mask_width", static_cast<uint32_t>(dst_cols));
    meta->set("dino_mask_threshold_db", mask_threshold_db_.get());
    meta->set("dino_backend", std::string("cuda_threshold_fallback"));
    meta->set("dino_model_name", model_name_.get());
    meta->set("dino_weights_path", weights_path_.get());
    meta->set("dino_model_script_path", model_script_path_.get());
    meta->set("dino_pipeline_variant", pipeline_variant);

    backend_used = "cuda_threshold_fallback";
    return true;
  };

#ifdef HOLOHUB_HAS_TORCH
  if (use_pytorch_backend_.get() && pytorch_runtime_ready_) {
    std::string failure_stage = "torch_runtime_setup";
    std::string failure_detail;
    try {
      if (!torch_runtime_) {
        throw std::runtime_error("Torch runtime helper was not initialized");
      }

      {
        std::ostringstream oss;
        oss << "src_rows=" << src_rows
            << " src_cols=" << src_cols
            << " fft_ptr=" << static_cast<const void*>(fft_tensor.Data());
        failure_detail = oss.str();
      }

      const int total_bins = src_rows * src_cols;
      const size_t power_db_bytes = static_cast<size_t>(total_bins) * sizeof(float);
      if (power_db_device_buffer_sizes_[channel_number] != static_cast<size_t>(total_bins)) {
        if (power_db_device_buffers_[channel_number] != nullptr) {
          cudaFree(power_db_device_buffers_[channel_number]);
          power_db_device_buffers_[channel_number] = nullptr;
        }
        auto alloc_result = cudaMalloc(reinterpret_cast<void**>(&power_db_device_buffers_[channel_number]), power_db_bytes);
        if (alloc_result != cudaSuccess) {
          throw std::runtime_error(std::string("power_db device buffer allocation failed: ") + cudaGetErrorString(alloc_result));
        }
        power_db_device_buffer_sizes_[channel_number] = static_cast<size_t>(total_bins);
      }

      failure_stage = "power_db_device_compute";
      const int threads = 256;
      const int blocks = (total_bins + threads - 1) / threads;
      complex_to_power_db_kernel<<<blocks, threads, 0, stream>>>(fft_tensor.Data(),
                                                                 power_db_device_buffers_[channel_number],
                                                                 src_rows,
                                                                 src_cols);
      auto power_db_kernel_result = cudaGetLastError();
      if (power_db_kernel_result != cudaSuccess) {
        throw std::runtime_error(std::string("power_db kernel launch failed: ") + cudaGetErrorString(power_db_kernel_result));
      }

      if (inference_backend_.get() == "torchscript" && !torchscript_forward_trace_emitted_) {
        HOLOSCAN_LOG_INFO("Torch runtime trace: channel={} frame={} input={}x{} backend={} init_mode={}",
                          channel_number,
                          frame_number,
                          src_rows,
                          src_cols,
                          inference_backend_.get(),
                          torchscript_init_mode_.get());
      }

      DinoTorchRuntimeConfig runtime_config;
      runtime_config.inference_backend = inference_backend_.get();
      runtime_config.model_script_path = model_script_path_.get();
      runtime_config.torchscript_init_mode = torchscript_init_mode_.get();
      runtime_config.imagenet_mean = imagenet_mean_.get();
      runtime_config.imagenet_std = imagenet_std_.get();
      runtime_config.return_final_mask = need_device_mask;
      runtime_config.ignore_sideband_hz = ignore_sideband_hz_.get();
      runtime_config.frontend_correction_enable = frontend_correction_enable_.get();
      runtime_config.frontend_correction_row_q = frontend_correction_row_q_.get();
      runtime_config.frontend_correction_smooth_sigma = frontend_correction_smooth_sigma_.get();
      runtime_config.frontend_correction_reference_q = frontend_correction_reference_q_.get();
      runtime_config.frontend_correction_max_boost_db = frontend_correction_max_boost_db_.get();
      runtime_config.frontend_correction_soft_knee_db = frontend_correction_soft_knee_db_.get();
      runtime_config.frontend_correction_edge_taper_fraction = frontend_correction_edge_taper_fraction_.get();
      runtime_config.frontend_correction_edge_taper_sigma = frontend_correction_edge_taper_sigma_.get();
      runtime_config.frontend_correction_edge_target_drop_db = frontend_correction_edge_target_drop_db_.get();
      runtime_config.power_q = power_q_.get();
      runtime_config.dino_group_score_q = dino_group_score_q_.get();
      runtime_config.pipeline_final_threshold = pipeline_final_threshold_.get();
      runtime_config.pipeline_gap_floor = pipeline_gap_floor_.get();
      runtime_config.pipeline_power_rescue_floor = pipeline_power_rescue_floor_.get();
      runtime_config.pipeline_power_rescue_gain = pipeline_power_rescue_gain_.get();

      DinoTorchRuntimeInput runtime_input;
      runtime_input.channel_number = channel_number;
      runtime_input.frame_number = frame_number;
      runtime_input.src_rows = src_rows;
      runtime_input.src_cols = src_cols;
      runtime_input.dst_rows = dst_rows;
      runtime_input.dst_cols = dst_cols;
      runtime_input.patch_size = patch_size;
      runtime_input.cuda_stream = stream;
      failure_stage = "metadata_read";
      runtime_input.resolution_hz = static_cast<double>(meta->get<uint64_t>("resolution", 0));
      runtime_input.span_hz = static_cast<double>(meta->get<uint64_t>("span", 0));
      runtime_input.power_db_device = power_db_device_buffers_[channel_number];

      failure_stage = "torch_runtime";
      auto runtime_result = torch_runtime_->run(runtime_config, runtime_input);
      if (!runtime_result.success) {
        failure_stage = runtime_result.error_stage;
        failure_detail = runtime_result.error_detail;
        throw std::runtime_error(runtime_result.error_message);
      }

      if (inference_backend_.get() == "torchscript" &&
          strict_model_forward_.get() &&
          runtime_result.backend_used != "torchscript") {
        HOLOSCAN_LOG_WARN("TorchScript backend unavailable for channel {} frame {} and strict_model_forward=true; dropping frame.",
                          channel_number,
                          frame_number);
        return;
      }

      stage_ms[kFrontendCorrectionStage] = runtime_result.timing.frontend_correction_ms;
      stage_ms[kCropAlignStage] = runtime_result.timing.crop_align_ms;
      stage_ms[kResizeStage] = runtime_result.timing.resize_ms;
      stage_ms[kModelPrepStage] = runtime_result.timing.model_prep_ms;
      stage_ms[kTorchForwardStage] = runtime_result.timing.torch_forward_ms;
      stage_ms[kDinoScoreStage] = runtime_result.timing.dino_score_ms;
      stage_ms[kPowerScoreStage] = runtime_result.timing.power_score_ms;
      stage_ms[kFusionStage] = runtime_result.timing.fusion_ms;

      backend_used = runtime_result.backend_used;
      ignore_bins_per_side = runtime_result.ignore_bins_per_side;
      freq_bin_hz = runtime_result.freq_bin_hz;
      aligned_rows = runtime_result.aligned_rows;
      aligned_cols = runtime_result.aligned_cols;
      dino_threshold = runtime_result.dino_threshold;
      power_threshold = runtime_result.power_threshold;
      final_threshold = runtime_result.final_threshold;

      if (backend_used == "torchscript") {
        torchscript_forward_trace_emitted_ = true;
      }

      if (need_device_mask) {
        if (runtime_result.final_mask.size() != static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols)) {
          throw std::runtime_error("Torch runtime returned an unexpected final mask size");
        }

        failure_stage = "device_copy";
        time_step_ms(kDeviceCopyStage, [&] {
          const size_t output_bytes = static_cast<size_t>(dst_rows) * static_cast<size_t>(dst_cols) * sizeof(float);
          auto copy_result = cudaMemcpyAsync(out.Data(),
                                             runtime_result.final_mask.data(),
                                             output_bytes,
                                             cudaMemcpyHostToDevice,
                                             stream);
          if (copy_result != cudaSuccess) {
            throw std::runtime_error(std::string("torch-path memcpy failed: ") + cudaGetErrorString(copy_result));
          }
        });
      } else {
        time_step_ms(kDeviceCopyStage, [&] {});
      }

      meta->set("dino_frame_number", frame_number);
      meta->set("dino_mask_height", static_cast<uint32_t>(dst_rows));
      meta->set("dino_mask_width", static_cast<uint32_t>(dst_cols));
      meta->set("dino_mask_threshold_db", mask_threshold_db_.get());
      meta->set("dino_backend", backend_used);
      meta->set("dino_model_name", model_name_.get());
      meta->set("dino_weights_path", weights_path_.get());
      meta->set("dino_model_script_path", model_script_path_.get());
      meta->set("dino_torchscript_init_mode", torchscript_init_mode_.get());
      meta->set("dino_torchscript_forward_ready", runtime_result.torchscript_forward_ready);
      meta->set("dino_patch_size", patch_size);
      meta->set("dino_fft_size", fft_size_.get());
      meta->set("dino_noverlap", noverlap_.get());
      meta->set("dino_ignore_bins_per_side", ignore_bins_per_side);
      meta->set("dino_freq_bin_hz", freq_bin_hz);
      meta->set("dino_frontend_correction_enabled", frontend_correction_enable_.get());
      meta->set("dino_input_aligned_height", aligned_rows);
      meta->set("dino_input_aligned_width", aligned_cols);
      meta->set("dino_group_score_threshold", dino_threshold);
      meta->set("dino_power_score_threshold", power_threshold);
      meta->set("dino_pipeline_final_threshold", final_threshold);
      meta->set("dino_pipeline_variant", std::string("notebook_preprocess_feature_fusion_v1"));
      meta->set("dino_preprocess_color_mode", std::string("grayscale_triplicate"));

      if (log_detections_.get()) {
        HOLOSCAN_LOG_INFO("DINOv3 detector ({}) emitted notebook-aligned debug mask for channel {} frame {} shape {}x{}",
                          backend_used,
                          channel_number,
                          frame_number,
                          dst_rows,
                          dst_cols);
      }

      failure_stage = "mask_save";
      time_step_ms(kMaskSaveStage, [&] { maybe_save_mask(backend_used); });
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_WARN("DINOv3 detector failed on channel {} frame {} during {}: {} [{}]",
                        channel_number,
                        frame_number,
                        failure_stage,
                        error.what(),
                        failure_detail);
      if (strict_model_forward_.get()) {
        return;
      }
      try {
        run_cuda_fallback("cuda_threshold_fallback_after_error");
      } catch (const std::exception& fallback_error) {
        HOLOSCAN_LOG_ERROR("DINOv3 detector fallback failed on channel {} frame {}: {}",
                           channel_number,
                           frame_number,
                           fallback_error.what());
        return;
      }
    }
  } else {
    try {
      run_cuda_fallback("cuda_threshold_fallback");
    } catch (const std::exception& error) {
      HOLOSCAN_LOG_ERROR("DINOv3 detector fallback failed on channel {} frame {}: {}",
                         channel_number,
                         frame_number,
                         error.what());
      return;
    }
  }
#else
  if (use_pytorch_backend_.get() && !pytorch_warning_emitted_) {
    HOLOSCAN_LOG_WARN("DINOv3 detector use_pytorch_backend=true but Torch is unavailable at build time; using CUDA fallback path.");
    pytorch_warning_emitted_ = true;
  }

  try {
    run_cuda_fallback("cuda_threshold_fallback");
  } catch (const std::exception& error) {
    HOLOSCAN_LOG_ERROR("DINOv3 detector fallback failed on channel {} frame {}: {}",
                       channel_number,
                       frame_number,
                       error.what());
    return;
  }
#endif

  stage_ms[kTotalStage] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - total_start).count();
  meta->set("dino_timing_total_ms", stage_ms[kTotalStage]);
  meta->set("dino_timing_summary_enabled", timing_enabled);

  if (timing_enabled) {
    auto& stats = timing_stats_[channel_number];
    ++stats.window_frames;
    for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
      stats.total_ms[stage_index] += stage_ms[stage_index];
      stats.max_ms[stage_index] = std::max(stats.max_ms[stage_index], stage_ms[stage_index]);
    }

    const int summary_every = std::max(1, timing_summary_every_n_.get());
    const int summary_window = std::max(1, timing_summary_window_.get());
    const bool emit_summary = (frame_number % static_cast<uint64_t>(summary_every) == 0) ||
                              (stats.window_frames >= static_cast<uint64_t>(summary_window));
    if (emit_summary) {
      const double inv_frames = 1.0 / static_cast<double>(std::max<uint64_t>(1, stats.window_frames));
      std::ostringstream oss;
      oss << "DINO timing summary ch=" << channel_number
          << " backend=" << backend_used
          << " frames=" << stats.window_frames;
      for (size_t stage_index = 0; stage_index < kTimingStageCount; ++stage_index) {
        const double mean_ms = stats.total_ms[stage_index] * inv_frames;
        oss << ' ' << kTimingStageNames[stage_index] << "_mean=" << mean_ms
            << ' ' << kTimingStageNames[stage_index] << "_max=" << stats.max_ms[stage_index];
      }
      HOLOSCAN_LOG_INFO("{}", oss.str());
      stats = ChannelTimingStats {};
    }
  }
}

}  // namespace holoscan::ops
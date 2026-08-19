// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// TorchScript runtime for the fine-tuned DinoSegmenter. SCAFFOLD -- compiles the boundary; the
// forward() body must be finished against the in-container libtorch (see TODOs). Model the tensor/
// device/stream plumbing on operators/dinov3_signal_detector/dinov3_torch_runtime.cpp.
#include "finetuned_dino_torch_helpers.hpp"

#include <cstdio>

#if defined(HOLOHUB_HAS_TORCH)
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <torch/nn/functional.h>
#include <torch/script.h>
#include <torch/torch.h>
#endif

#include <vector>

namespace holoscan::ops {

struct FinetunedDinoTorchRuntime::Impl {
#if defined(HOLOHUB_HAS_TORCH)
  torch::jit::script::Module module;
  bool is_loaded = false;
  torch::Dtype dtype = torch::kFloat32;
#endif
};

FinetunedDinoTorchRuntime::FinetunedDinoTorchRuntime() : impl_(std::make_unique<Impl>()) {}
FinetunedDinoTorchRuntime::~FinetunedDinoTorchRuntime() = default;

bool FinetunedDinoTorchRuntime::load(const std::string& model_script_path,
                                     const std::string& torch_dtype) {
#if defined(HOLOHUB_HAS_TORCH)
  try {
    impl_->module = torch::jit::load(model_script_path, torch::kCUDA);
    impl_->module.eval();
    // Mixed precision is baked into the .ts at export time (torch.jit.trace under autocast bf16), not
    // toggled here: converting a traced module to half breaks on hardcoded dtype casts in the graph
    // (e.g. the RoPE bias_mask .to(float32)). So the runtime always feeds fp32 and lets the traced
    // graph run whatever precision it was traced with. torch_dtype is kept only for the native path.
    impl_->dtype = torch::kFloat32;
    (void)torch_dtype;
    impl_->is_loaded = true;
    std::fprintf(stderr, "[finetuned_dino_detector] loaded TorchScript %s\n", model_script_path.c_str());
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[finetuned_dino_detector] ERROR loading %s: %s\n",
                 model_script_path.c_str(), e.what());
    return false;
  }
#else
  (void)model_script_path; (void)torch_dtype;
  std::fprintf(stderr, "[finetuned_dino_detector] built without torch (HOLOHUB_HAS_TORCH undefined)\n");
  return false;
#endif
}

bool FinetunedDinoTorchRuntime::forward_downsampled(const float* normalized_wide, int rows, int wide,
                                                    int tile_rows, int nfft, float threshold,
                                                    uint8_t* out_mask_wide, cudaStream_t stream,
                                                    double* inference_ms) {
#if defined(HOLOHUB_HAS_TORCH)
  if (!impl_->is_loaded) return false;
  namespace F = torch::nn::functional;
  try {
    torch::InferenceMode inference_guard(true);
    c10::Device device(torch::kCUDA, 0);
    const auto torch_stream = stream
                                  ? c10::cuda::getStreamFromExternal(stream, device.index())
                                  : c10::cuda::getDefaultCUDAStream(device.index());
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);
    auto fopts = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    if (inference_ms) {
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      cudaEventRecord(ev0, stream);
    }

    // wide dB image [1,1,rows,wide] -> bilinear resize freq to model width -> [1,1,rows,nfft]
    auto wide_t = torch::from_blob(const_cast<float*>(normalized_wide), {1, 1, rows, wide}, fopts);
    auto resized = F::interpolate(
        wide_t, F::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{rows, nfft}).mode(torch::kBilinear).align_corners(false));

    // pad rows up to a whole number of tiles, then split into [B,1,tile_rows,nfft]
    const int batch = (rows + tile_rows - 1) / tile_rows;
    const int padded = batch * tile_rows;
    if (padded != rows) {
      resized = torch::constant_pad_nd(resized, {0, 0, 0, padded - rows}, 0);
    }
    auto tiles = resized.contiguous().view({batch, tile_rows, nfft}).unsqueeze(1);  // [B,1,tile_rows,nfft]
    torch::jit::IValue model_in = (impl_->dtype == torch::kHalf) ? tiles.to(torch::kHalf) : tiles;

    auto logits = impl_->module.forward({model_in}).toTensor().to(torch::kFloat32);
    auto mask_tiles = (torch::sigmoid(logits) >= threshold).to(torch::kFloat32);  // [B,1,tile_rows,nfft]

    // stitch tiles back to [1,1,padded,nfft], crop to rows, nearest-upsample freq nfft->wide
    auto stitched = mask_tiles.squeeze(1).contiguous().view({1, 1, padded, nfft});
    if (padded != rows) {
      stitched = stitched.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                 torch::indexing::Slice(0, rows), torch::indexing::Slice()});
    }
    auto up = F::interpolate(
        stitched, F::InterpolateFuncOptions().size(std::vector<int64_t>{rows, wide}).mode(torch::kNearest));
    auto out_u8 = (up.squeeze(0).squeeze(0) >= 0.5f).to(torch::kUInt8).contiguous();  // [rows,wide]

    cudaMemcpyAsync(out_mask_wide, out_u8.data_ptr<uint8_t>(),
                    static_cast<size_t>(rows) * wide, cudaMemcpyDeviceToDevice, stream);
    if (inference_ms) cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);  // keep out_u8 alive until the copy completes + mask valid downstream
    if (inference_ms) {
      float ms = 0.0f;
      cudaEventElapsedTime(&ms, ev0, ev1);
      *inference_ms = static_cast<double>(ms);
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[finetuned_dino_detector] forward_downsampled error: %s\n", e.what());
    return false;
  }
#else
  (void)normalized_wide; (void)rows; (void)wide; (void)tile_rows; (void)nfft; (void)threshold;
  (void)out_mask_wide; (void)stream; (void)inference_ms;
  return false;
#endif
}

void FinetunedDinoTorchRuntime::warmup(int tile_rows, int nfft, int batch, int iters) {
#if defined(HOLOHUB_HAS_TORCH)
  if (!impl_->is_loaded) return;
  try {
    torch::InferenceMode inference_guard(true);
    c10::Device device(torch::kCUDA, 0);
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto dummy = torch::zeros({std::max(1, batch), 1, tile_rows, nfft}, opts);
    for (int i = 0; i < iters; ++i) {
      (void)impl_->module.forward({dummy});
    }
    cudaDeviceSynchronize();
    std::fprintf(stderr, "[finetuned_dino_detector] warmup done (%d x [%d,1,%d,%d])\n",
                 iters, std::max(1, batch), tile_rows, nfft);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[finetuned_dino_detector] warmup skipped: %s\n", e.what());
  }
#else
  (void)tile_rows; (void)nfft; (void)batch; (void)iters;
#endif
}

bool FinetunedDinoTorchRuntime::loaded() const {
#if defined(HOLOHUB_HAS_TORCH)
  return impl_->is_loaded;
#else
  return false;
#endif
}

bool FinetunedDinoTorchRuntime::forward(const float* input_device, int batch, int tile_rows, int nfft,
                                        float* logits_device, cudaStream_t stream) {
#if defined(HOLOHUB_HAS_TORCH)
  if (!impl_->is_loaded) return false;
  try {
    torch::InferenceMode inference_guard(true);
    c10::Device device(torch::kCUDA, 0);
    // Run torch ops on the operator's processing stream so ordering with the front-end kernels holds.
    const auto torch_stream = stream
                                  ? c10::cuda::getStreamFromExternal(stream, device.index())
                                  : c10::cuda::getDefaultCUDAStream(device.index());
    c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto in = torch::from_blob(const_cast<float*>(input_device),
                               {batch, 1, tile_rows, nfft}, opts);
    torch::jit::IValue model_in = (impl_->dtype == torch::kHalf) ? in.to(torch::kHalf) : in;

    auto out = impl_->module.forward({model_in}).toTensor().to(torch::kFloat32).contiguous();
    const size_t nbytes = sizeof(float) * static_cast<size_t>(batch) * tile_rows * nfft;
    if (static_cast<size_t>(out.numel()) * sizeof(float) != nbytes) {
      std::fprintf(stderr, "[finetuned_dino_detector] forward: unexpected output numel %ld (want %zu)\n",
                   static_cast<long>(out.numel()), nbytes / sizeof(float));
      return false;
    }
    cudaMemcpyAsync(logits_device, out.data_ptr<float>(), nbytes,
                    cudaMemcpyDeviceToDevice, stream);
    // `out` frees on return; ensure the async copy has consumed it first.
    cudaStreamSynchronize(stream);
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[finetuned_dino_detector] forward error: %s\n", e.what());
    return false;
  }
#else
  (void)input_device; (void)batch; (void)tile_rows; (void)nfft; (void)logits_device; (void)stream;
  return false;
#endif
}

}  // namespace holoscan::ops

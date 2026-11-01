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
  // Persistent timing events (created once; the per-frame cudaEventCreate/Destroy pair cost ~70 us).
  cudaEvent_t ev0 = nullptr;
  cudaEvent_t ev1 = nullptr;
  bool timed = false;
};

FinetunedDinoTorchRuntime::FinetunedDinoTorchRuntime() : impl_(std::make_unique<Impl>()) {}
FinetunedDinoTorchRuntime::~FinetunedDinoTorchRuntime() {
  if (impl_) {
    if (impl_->ev0) cudaEventDestroy(impl_->ev0);
    if (impl_->ev1) cudaEventDestroy(impl_->ev1);
  }
}

double FinetunedDinoTorchRuntime::last_inference_ms() const {
  if (!impl_ || !impl_->timed || !impl_->ev0 || !impl_->ev1) return -1.0;
  float ms = 0.0f;
  return (cudaEventElapsedTime(&ms, impl_->ev0, impl_->ev1) == cudaSuccess) ? static_cast<double>(ms) : -1.0;
}

bool FinetunedDinoTorchRuntime::load(const std::string& model_script_path,
                                     const std::string& torch_dtype, bool freeze) {
#if defined(HOLOHUB_HAS_TORCH)
  try {
    impl_->module = torch::jit::load(model_script_path, torch::kCUDA);
    impl_->module.eval();
    if (freeze) {
      auto count_casts = [](const std::shared_ptr<torch::jit::Graph>& g) {
        int n = 0;
        for (auto* node : g->nodes()) n += (node->kind() == c10::Symbol::fromQualString("aten::to")) ? 1 : 0;
        return n;
      };
      const int before = count_casts(impl_->module.get_method("forward").graph());
      impl_->module = torch::jit::freeze(impl_->module);
      const int after = count_casts(impl_->module.get_method("forward").graph());
      std::fprintf(stderr, "[finetuned_dino_detector] froze TorchScript module (top-level aten::to nodes %d -> %d)\n",
                   before, after);
    }
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
                                                    uint8_t* out_mask_compact, cudaStream_t stream,
                                                    bool time_it, bool wrap_edges) {
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

    impl_->timed = false;
    if (time_it) {
      if (!impl_->ev0) cudaEventCreate(&impl_->ev0);
      if (!impl_->ev1) cudaEventCreate(&impl_->ev1);
      cudaEventRecord(impl_->ev0, stream);
    }

    // wide dB image [1,1,rows,wide] -> bilinear resize freq to model width -> [1,1,rows,nfft]
    auto wide_t = torch::from_blob(const_cast<float*>(normalized_wide), {1, 1, rows, wide}, fopts);
    torch::Tensor resized = wide_t;
    if (wide != nfft) {
      resized = F::interpolate(
          wide_t, F::InterpolateFuncOptions()
                      .size(std::vector<int64_t>{rows, nfft}).mode(torch::kBilinear).align_corners(false));
    }

    // pad rows up to a whole number of tiles, then split into [B,1,tile_rows,nfft]
    const int batch = (rows + tile_rows - 1) / tile_rows;
    const int padded = batch * tile_rows;
    if (padded != rows) {
      resized = torch::constant_pad_nd(resized, {0, 0, 0, padded - rows}, 0);
    }
    auto tiles = resized.contiguous().view({batch, tile_rows, nfft}).unsqueeze(1);  // [B,1,tile_rows,nfft]

    // Circular edge inference (see header): batch a half-band-rotated copy into the SAME forward so
    // the border-biased model never supplies the mask's outer columns. The DFT spectrum is periodic,
    // so the rotation is spectrally contiguous -- no synthetic seam is introduced.
    torch::Tensor tiles_all = tiles;
    if (wrap_edges) {
      auto rolled = torch::roll(resized, {nfft / 2}, {3});
      auto tiles2 = rolled.contiguous().view({batch, tile_rows, nfft}).unsqueeze(1);
      tiles_all = torch::cat({tiles, tiles2}, 0);  // [2B,1,tile_rows,nfft]
    }
    torch::jit::IValue model_in = (impl_->dtype == torch::kHalf) ? tiles_all.to(torch::kHalf) : tiles_all;

    auto logits = impl_->module.forward({model_in}).toTensor().to(torch::kFloat32);
    auto mask_tiles = (torch::sigmoid(logits) >= threshold).to(torch::kUInt8);  // [B or 2B,1,tile_rows,nfft]
    if (wrap_edges) {
      using torch::indexing::Slice;
      auto m1 = mask_tiles.narrow(0, 0, batch).clone();
      auto m2 = torch::roll(mask_tiles.narrow(0, batch, batch), {-nfft / 2}, {3});  // back to true columns
      // Outer quarters come from the rotated pass (those columns sat mid-image there); the rotated
      // pass's own borders land mid-band and are discarded. Every output column is center-sourced.
      const int quarter = nfft / 4;
      m1.index_put_({Slice(), Slice(), Slice(), Slice(0, quarter)},
                    m2.index({Slice(), Slice(), Slice(), Slice(0, quarter)}));
      m1.index_put_({Slice(), Slice(), Slice(), Slice(nfft - quarter, nfft)},
                    m2.index({Slice(), Slice(), Slice(), Slice(nfft - quarter, nfft)}));
      mask_tiles = m1;
    }

    // stitch tiles back to [padded,nfft], crop to rows, write the compact mask into the caller buffer
    auto stitched = mask_tiles.squeeze(1).contiguous().view({padded, nfft});
    if (padded != rows) {
      stitched = stitched.narrow(0, 0, rows);
    }
    auto out = torch::from_blob(out_mask_compact, {rows, nfft},
                                torch::TensorOptions().dtype(torch::kUInt8).device(device));
    out.copy_(stitched);
    if (time_it) {
      cudaEventRecord(impl_->ev1, stream);
      impl_->timed = true;
    }
    // No synchronize: every temporary above was allocated on `stream`, so the caching allocator only
    // hands their blocks out again in stream order; the caller syncs `stream` before emitting.
    return true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[finetuned_dino_detector] forward_downsampled error: %s\n", e.what());
    return false;
  }
#else
  (void)normalized_wide; (void)rows; (void)wide; (void)tile_rows; (void)nfft; (void)threshold;
  (void)out_mask_compact; (void)stream; (void)time_it; (void)wrap_edges;
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

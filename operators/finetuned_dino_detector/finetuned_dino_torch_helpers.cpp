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
#include <torch/script.h>
#include <torch/torch.h>
#endif

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
    impl_->dtype = (torch_dtype == "fp16") ? torch::kHalf : torch::kFloat32;
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
    torch::NoGradGuard no_grad;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    // input_device is a device pointer we own; wrap without copy. Model input = [B,1,tile,nfft] in [0,1].
    auto in = torch::from_blob(const_cast<float*>(input_device),
                               {batch, 1, tile_rows, nfft}, opts);
    torch::Tensor in_run = (impl_->dtype == torch::kHalf) ? in.to(torch::kHalf) : in;
    auto out = impl_->module.forward({in_run}).toTensor().to(torch::kFloat32).contiguous();
    const int64_t want = static_cast<int64_t>(batch) * tile_rows * nfft;
    if (out.numel() != want) {
      std::fprintf(stderr, "[finetuned_dino_detector] logits numel %ld != %ld (shape mismatch)\n",
                   static_cast<long>(out.numel()), static_cast<long>(want));
      return false;
    }
    // TODO(lab-admin): verify stream ordering. TorchScript inference runs on torch's current CUDA
    // stream (default stream unless a CUDAStreamGuard is set); a synchronous D2D copy on stream 0 is
    // ordered after it. If you set a non-default `stream`, add a c10::cuda::CUDAStreamGuard here.
    cudaMemcpy(logits_device, out.data_ptr<float>(), sizeof(float) * static_cast<size_t>(want),
               cudaMemcpyDeviceToDevice);
    (void)stream;
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

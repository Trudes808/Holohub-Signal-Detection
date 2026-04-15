// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "dinov3_torch_runtime.hpp"

#include <utility>

namespace holoscan::ops {

class DinoTorchRuntime::Impl {};

DinoTorchRuntime::DinoTorchRuntime() = default;
DinoTorchRuntime::~DinoTorchRuntime() = default;
DinoTorchRuntime::DinoTorchRuntime(DinoTorchRuntime&&) noexcept = default;
DinoTorchRuntime& DinoTorchRuntime::operator=(DinoTorchRuntime&&) noexcept = default;

DinoTorchRuntimeResult DinoTorchRuntime::run(const DinoTorchRuntimeConfig& config,
                                             const DinoTorchRuntimeInput& input) {
  DinoTorchRuntimeResult result;
  result.success = false;
  result.error_stage = "torch_unavailable";
  result.error_message = "Torch runtime is not available in this build";
  result.error_detail = "Rebuild the container with the pinned Torch stack or disable the PyTorch backend";
  result.backend_used = config.inference_backend.empty() ? "pytorch_unavailable" : config.inference_backend;
  result.torchscript_forward_ready = false;
  result.aligned_rows = input.dst_rows;
  result.aligned_cols = input.dst_cols;
  return result;
}

DinoHybridPostGpuResult DinoTorchRuntime::run_hybrid_post_gpu(const DinoHybridPostGpuInput& input) {
  DinoHybridPostGpuResult result;
  result.success = false;
  result.error_message = "Torch runtime is not available in this build";
  result.error_detail = "Rebuild the container with the pinned Torch stack or fall back to the CPU hybrid postprocess";
  result.seed_mask.assign(static_cast<size_t>(std::max(0, input.dst_rows * input.dst_cols)), 0);
  result.grow_mask.assign(static_cast<size_t>(std::max(0, input.dst_rows * input.dst_cols)), 0);
  result.combined_gate_mask.assign(static_cast<size_t>(std::max(0, input.dst_rows * input.dst_cols)), 0);
  return result;
}

}  // namespace holoscan::ops
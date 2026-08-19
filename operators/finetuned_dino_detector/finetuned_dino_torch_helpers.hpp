// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <string>

namespace holoscan::ops {

// TorchScript runtime for the fine-tuned DinoSegmenter. Kept in a .cpp (CXX, libtorch) so nvcc never
// sees torch headers -- mirrors cuda_dino_detector's cuda_dino_torch_helpers split.
class FinetunedDinoTorchRuntime {
 public:
  FinetunedDinoTorchRuntime();
  ~FinetunedDinoTorchRuntime();

  // Load the TorchScript module (dino_fine_tuning/weights/finetuned_dino_m*.ts), move to CUDA, eval().
  bool load(const std::string& model_script_path, const std::string& torch_dtype);

  // Forward: input_device is B x 1 x tile_rows x nfft float in [0,1] (device); writes
  // logits_device (same shape). Runs on `stream`. Returns false on failure.
  // TODO(lab-admin): wrap input_device as a torch CUDA tensor (torch::from_blob with the right
  //   sizes/strides + device), forward the module, copy the [B,1,tile_rows,nfft] output into
  //   logits_device. Match the dtype in torch_dtype (fp32). See dinov3_signal_detector/
  //   dinov3_torch_runtime.cpp for the device/stream/dtype plumbing to reuse.
  bool forward(const float* input_device, int batch, int tile_rows, int nfft,
               float* logits_device, cudaStream_t stream);

  // Real-time "downsample" path: normalized_wide is a rows x wide [0,1] dB spectrogram (device). The
  // frequency axis is bilinear-resized to the model width (nfft), tiled into tile_rows-row tiles, run
  // through the segmenter, thresholded (sigmoid >= threshold), stitched, and nearest-upsampled back to
  // rows x wide, written to out_mask_wide (device uint8). Everything runs in torch on `stream`.
  // Returns false on failure. Optionally reports the pure-inference (resize+forward+post) time in ms.
  bool forward_downsampled(const float* normalized_wide, int rows, int wide, int tile_rows, int nfft,
                           float threshold, uint8_t* out_mask_wide, cudaStream_t stream,
                           double* inference_ms = nullptr);

  // Prime the model (a few dummy forwards) so the first real frame doesn't pay the one-time
  // cuDNN-autotune / allocation cost (~hundreds of ms). Call once after load(), before compute().
  void warmup(int tile_rows, int nfft, int batch, int iters = 3);

  bool loaded() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::ops

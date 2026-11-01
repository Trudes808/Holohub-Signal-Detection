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
  // freeze: torch::jit::freeze the module (inline weights as constants + constant-propagate). The
  // exports are traced under bf16 autocast, so an unfrozen forward re-casts every fp32 weight to bf16
  // each call (~124 cast kernels per 2-tile forward); freezing folds those casts once. Verified
  // bit-identical logits on the M5-S export (2026-10-21), ~1 ms faster per 2-tile forward.
  bool load(const std::string& model_script_path, const std::string& torch_dtype, bool freeze = true);

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
  // through the segmenter, thresholded (sigmoid >= threshold) and stitched into the COMPACT rows x nfft
  // uint8 mask written to out_mask_compact (device, caller-owned). Everything is enqueued on `stream`
  // and NOT synchronized here: torch temporaries go back to the caching allocator in stream order, and
  // the caller synchronizes once per frame before emitting. The caller expands the compact mask to
  // the wide grid (nearest, column replication) if it needs one. If time_it, the resize+forward+post
  // span is bracketed by two persistent events; read it with last_inference_ms() AFTER that sync.
  //
  // wrap_edges: the M2_dr checkpoint has a measured BORDER bias -- on flat noise it fires on 30-70% of
  // its outermost columns and 0% mid-image (its training captures always had the receiver rolloff at
  // the image borders). An FFT spectrum is circular (-fs/2 == +fs/2), so the band edge is only where
  // we CUT the image, not a real boundary: run the model on the image AND a half-band circular
  // rotation of it (batched into one forward), then take every output column from the pass where it
  // sat in the image center. No output column ever comes from a model border. Measured on the 491.52
  // OTA capture: edge-region firing 16.5% -> 0.00% with mid-band masks unchanged.
  bool forward_downsampled(const float* normalized_wide, int rows, int wide, int tile_rows, int nfft,
                           float threshold, uint8_t* out_mask_compact, cudaStream_t stream,
                           bool time_it = false, bool wrap_edges = true);
  // ms between the two events recorded by the last timed forward_downsampled(); -1 if unavailable.
  // Only valid once the caller has synchronized the stream that forward was enqueued on.
  double last_inference_ms() const;

  // Prime the model (a few dummy forwards) so the first real frame doesn't pay the one-time
  // cuDNN-autotune / allocation cost (~hundreds of ms). Call once after load(), before compute().
  void warmup(int tile_rows, int nfft, int batch, int iters = 3);

  bool loaded() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::ops

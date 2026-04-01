<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation

SPDX-License-Identifier: Apache-2.0
-->
# DINOv3 Signal Detector Operator

## Overview

Provides a C++/CUDA baseline signal-detection stage for the wideband USRP pipeline.

This initial implementation is a high-throughput scaffold that:
- consumes FFT-domain tensors (`tensor_t<complex, 2>`),
- supports an optional GPU PyTorch tensor path (when built with Torch),
- performs baseline GPU-side power-threshold masking,
- emits a fixed-size detector mask tensor (`tensor_t<float, 2>`),
- propagates metadata for downstream consumers.

It is designed as the integration point for future TensorRT-backed DINOv3 inference and postprocessing.

## Configuration

```yaml
dinov3_signal_detector:
  num_channels: 2
  input_height: 256
  input_width: 512
  emit_stride: 1
  mask_threshold_db: -20.0
  log_detections: false
  enable_mask_save: true
  save_every_n_frames: 1
  max_masks_per_channel: 5
  output_dir: "/workspace/dino_masks"
  use_pytorch_backend: true
  inference_backend: "torchscript"
  model_name: "dinov3_vitb16"
  model_repo_path: "/workspace/models/dinov3"
  weights_path: "/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
  model_script_path: "/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts"
  torchscript_init_mode: "load_cuda_eval"
  strict_model_forward: true
```

## I/O Contract

- Input: `tuple<tensor_t<complex, 2>, cudaStream_t>`
- Output: none; this is currently a terminal-stage detector/debug operator

Metadata keys written:
- `dino_frame_number`
- `dino_mask_height`
- `dino_mask_width`
- `dino_mask_threshold_db`
- `dino_backend`
- `dino_model_name`
- `dino_weights_path`
- `dino_model_script_path`
- `dino_torchscript_init_mode`
- `dino_torchscript_forward_ready`

## Current ML status

- `use_pytorch_backend=true` activates a PyTorch GPU tensor-processing path if Torch is available at build/runtime.
- `inference_backend` controls behavior:
  - `torchscript`: attempts TorchScript model forward using `model_script_path`.
  - `pytorch_placeholder`: skips model forward and uses tensor-domain placeholder masking.
  - `cuda_threshold_fallback`: uses CUDA kernel path only.
- `torchscript_init_mode` controls how far C++ initialization proceeds before compute begins:
  - `load_only`: load the TorchScript file only.
  - `load_cpu_eval`: load and call `eval()` on CPU.
  - `load_cuda_no_eval`: load and move the module to CUDA.
  - `load_cuda_eval`: load, move to CUDA, and call `eval()`.
- If TorchScript load/forward fails and `strict_model_forward=false`, execution falls back to `pytorch_placeholder`.
- If `torchscript_init_mode` leaves the module off CUDA, the operator logs that TorchScript is not forward-ready and uses `pytorch_placeholder` during compute.
- The selected runtime weight is `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` and is expected to be staged under `/workspace/models/dinov3/weights` inside the Holohub container.
- The expected TorchScript runtime artifact path is `/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`.
- For validation runs, keep `strict_model_forward=true` so model-load or model-forward issues are surfaced immediately.
- `enable_mask_save=true` writes debug `.pgm` mask images to `output_dir` using the first `max_masks_per_channel` frames that match `save_every_n_frames`.

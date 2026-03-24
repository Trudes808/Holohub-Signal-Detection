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
  use_pytorch_backend: true
  model_name: "dinov3_vitb16"
  model_repo_path: "/workspace/models/dinov3"
  weights_path: "/workspace/models/dinov3/weights/dinov3_vitb16_placeholder.pth"
```

## I/O Contract

- Input: `tuple<tensor_t<complex, 2>, cudaStream_t>`
- Output: `tuple<tensor_t<float, 2>, cudaStream_t>`

Metadata keys written:
- `dino_frame_number`
- `dino_mask_height`
- `dino_mask_width`
- `dino_mask_threshold_db`
- `dino_backend`
- `dino_model_name`
- `dino_weights_path`

## Current ML status

- `use_pytorch_backend=true` activates a PyTorch GPU tensor-processing path if Torch is available at build/runtime.
- This path currently performs tensor-domain preprocessing and mask generation while preserving CUDA stream handoff.
- The configured `weights_path` is a placeholder for upcoming full DINOv3 model-forward integration.

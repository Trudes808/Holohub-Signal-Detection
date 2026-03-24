<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation

SPDX-License-Identifier: Apache-2.0
-->
# DINOv3 Signal Detector Operator

## Overview

Provides a C++/CUDA baseline signal-detection stage for the wideband USRP pipeline.

This initial implementation is a high-throughput scaffold that:
- consumes FFT-domain tensors (`tensor_t<complex, 2>`),
- performs GPU-side power-threshold masking,
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
```

## I/O Contract

- Input: `tuple<tensor_t<complex, 2>, cudaStream_t>`
- Output: `tuple<tensor_t<float, 2>, cudaStream_t>`

Metadata keys written:
- `dino_frame_number`
- `dino_mask_height`
- `dino_mask_width`
- `dino_mask_threshold_db`

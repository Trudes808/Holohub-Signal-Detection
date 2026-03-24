<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation

SPDX-License-Identifier: Apache-2.0
-->
# Spectrogram Operator

## Overview

Converts FFT complex tensors into downsampled spectrogram debug images and writes them to disk.

## Description

The operator:
- accepts FFT tensors (`tensor_t<complex, 2>`) with CUDA stream,
- computes log-power values,
- downsamples to configurable output dimensions,
- normalizes to grayscale,
- writes `.pgm` images to a configurable directory.

This is a debug/verification path intended for early bring-up.

## Configuration

```yaml
spectrogram:
  num_channels: 2
  enable_save: true
  save_every_n_frames: 100
  max_images_per_channel: 20
  output_height: 256
  output_width: 512
  output_dir: "/workspace/spectrograms"
```

## Metadata

The operator reads `channel_number` from message metadata for per-channel image indexing.

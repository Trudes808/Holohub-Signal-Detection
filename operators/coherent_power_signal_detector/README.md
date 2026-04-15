<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation

SPDX-License-Identifier: Apache-2.0
-->
# Coherent Power Signal Detector Operator

## Overview

Provides the new coherent-power detector stage for the USRP wideband signal detection app.

The current implementation is a phase-2 skeleton intended to establish:

- the operator packaging and build surface,
- the runtime config interface,
- the same FFT-domain input contract used by the DINO operator, and
- a CUDA-resident placeholder mask path for smoke testing the detector selection plumbing.

The full notebook-derived coherent-power algorithm is planned to replace the placeholder compute path incrementally.

Validation and production parity rule: configs used to validate or represent production mask generation must keep the same `backend_mode`. `emit_stride` may vary for throughput tuning, but the mask-generation backend must not.

## Configuration

```yaml
coherent_power_signal_detector:
  num_channels: 2
  input_height: 256
  input_width: 512
  emit_stride: 1
  log_detections: false
  backend_mode: "reference"
  enable_mask_save: false
  enable_tensor_snapshot_save: false
  save_every_n_frames: 1
  max_masks_per_channel: 5
  max_snapshots_per_channel: 2
  output_dir: "/workspace/coherent_power_masks"
  tensor_snapshot_dir: "/workspace/coherent_power_snapshots"
  save_power_db_snapshot: true
  chunk_bandwidth_hz: 25000000.0
  chunk_overlap_hz: 6250000.0
  uncalibrated_chunk_fraction: 0.40
  uncalibrated_overlap_fraction: 0.20
  ignore_sideband_percent: 0.0
  ignore_sideband_hz: 7000000.0
  frontend_row_q: 25.0
  frontend_reference_q: 75.0
  frontend_smooth_sigma: 12.0
  frontend_max_boost_db: 12.0
  coherence_weight: 0.55
  power_weight: 0.45
  coherence_power_support_q: 0.82
  coherence_power_q: 0.92
  min_component_size: 6
  grouping_seed_score_q: 0.72
  grouping_bridge_freq_px: 33
  grouping_bridge_time_px: 5
  grouping_min_component_size: 24
  grouping_min_freq_span_px: 18
  grouping_min_time_span_px: 2
  grouping_min_density: 0.06
  timing_summary_enable: true
  timing_summary_every_n: 4
  timing_summary_window: 4
```

## I/O Contract

- Input: `tuple<tensor_t<complex, 2>, cudaStream_t>`
- Output: none in the current app integration

Metadata keys written by the skeleton path:

- `coherent_frame_number`
- `coherent_mask_height`
- `coherent_mask_width`
- `coherent_backend`
- `coherent_chunk_count`
- `coherent_grouped_box_count`
- `coherent_pipeline_variant`
- `coherent_timing_total_ms`
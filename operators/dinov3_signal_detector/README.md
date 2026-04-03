<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation

SPDX-License-Identifier: Apache-2.0
-->
# DINOv3 Signal Detector Operator

## Overview

Provides a C++/CUDA signal-detection stage for the wideband USRP pipeline.

The current implementation is no longer a pure threshold scaffold. It now:
- consumes FFT-domain tensors (`tensor_t<complex, 2>`),
- applies notebook-derived GPU preprocessing using LibTorch when Torch is available,
- supports optional frontend correction and sideband-ignore cropping,
- prepares notebook-aligned model inputs using grayscale-triplicate plus ImageNet normalization,
- runs TorchScript model forward when configured and available,
- derives a debug-ready detector mask from DINO feature energy plus power rescue fusion, and
- emits timing and preprocessing metadata for parity and optimization work.

This is intended as the verification-ready bridge between the notebook pipeline and the future fully ported GPU postprocess path.

## Configuration

```yaml
dinov3_signal_detector:
  num_channels: 2
  input_height: 256
  input_width: 512
  patch_size: 16
  emit_stride: 1
  mask_threshold_db: -20.0
  log_detections: true
  enable_mask_save: false
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
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std: [0.229, 0.224, 0.225]
  fft_size: 1024
  noverlap: 256
  ignore_sideband_hz: 7000000.0
  frontend_correction_enable: true
  frontend_correction_row_q: 25.0
  frontend_correction_smooth_sigma: 12.0
  frontend_correction_reference_q: 75.0
  frontend_correction_max_boost_db: 12.0
  frontend_correction_soft_knee_db: 4.0
  frontend_correction_edge_taper_fraction: 0.10
  frontend_correction_edge_taper_sigma: 6.0
  frontend_correction_edge_target_drop_db: 2.5
  frontend_edge_guard_floor: 0.35
  dino_coherence_gate_floor: 0.25
  texture_q: 0.90
  texture_k: 6
  power_q: 0.90
  dino_group_k: 8
  dino_group_spatial_weight: 0.35
  dino_group_score_q: 0.60
  pipeline_final_threshold: 0.20
  pipeline_final_threshold_no_speckle: 0.10
  pipeline_gap_floor: 0.10
  pipeline_component_min_size: 5
  pipeline_component_min_size_no_speckle: 2
  pipeline_power_rescue_floor: 0.10
  pipeline_power_rescue_gain: 2.0
  pipeline_strong_speckle_min_component: 10
  pipeline_texture_speckle_clean_threshold: 0.85
  pipeline_texture_speckle_strong_threshold: 0.20
  timing_summary_enable: true
  timing_summary_every_n: 4
  timing_summary_window: 4
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
- `dino_ignore_bins_per_side`
- `dino_freq_bin_hz`
- `dino_input_aligned_height`
- `dino_input_aligned_width`
- `dino_group_score_threshold`
- `dino_power_score_threshold`
- `dino_pipeline_final_threshold`
- `dino_pipeline_variant`
- `dino_timing_total_ms`

## Current ML status

- `use_pytorch_backend=true` activates a PyTorch GPU tensor-processing path if Torch is available at build/runtime.
- `inference_backend` controls behavior:
  - `torchscript`: attempts TorchScript model forward using `model_script_path`, then derives a score map from the returned tensor.
  - `pytorch_placeholder`: skips model forward and uses the notebook-aligned preprocess path with a placeholder DINO score derived from the corrected power image.
  - `cuda_threshold_fallback`: uses CUDA kernel path only.
- The current Torch path ports the notebook preprocessing constants and timing checkpoints first. It does not yet implement the full notebook grouping, coherence, and connected-component cleanup stages exactly.
- `frontend_correction_enable` and `ignore_sideband_hz` are the first notebook constants promoted into the C++ hot path.
- `timing_summary_enable=true` emits mean/max timing summaries for the major detector stages every `timing_summary_every_n` emitted frames.
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

## Verification Notes

- Use `applications/usrp_wideband_signal_detection/config_torchscript_validation.yaml` for strict reproduction-oriented bring-up.
- Compare the operator metadata and timing summaries against `signal_detection_holoscanv1.ipynb` before making optimization changes.
- Treat the current detector as notebook-faithful in preprocessing and parameterization, but still partial in postprocess parity until grouping/coherence cleanup is ported fully.

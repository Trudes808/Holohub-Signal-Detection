<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation

SPDX-License-Identifier: Apache-2.0
-->
# DINOv3 Signal Detector Operator

## Overview

Provides a C++/CUDA signal-detection stage for the wideband USRP pipeline.

The current implementation is no longer a pure threshold scaffold. It now:
- consumes FFT-domain tensors (`tensor_t<complex, 2>`),
- uses the coherent-power detector's per-channel allocation and reuse model,
- computes frontend correction and a coherent-style GPU gate before DINO inference,
- uses LibTorch only to produce a DINO score map when Torch is available,
- runs the notebook-derived residual-veto hybrid postprocess in the operator, and
- emits timing and hybrid-threshold metadata for parity and optimization work.

This version is the in-place replacement for the earlier scaffold: coherent real-time shell, old DINO runtime only for model execution, and the notebook's later hybrid logic for final mask generation.

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
  backend_mode: "reference"
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
  dino_coherence_gate_span_db: 3.0
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
- `dino_backend_mode`
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
- `dino_coherence_gate_floor`
- `dino_coherence_gate_span_db`
- `dino_seed_freq_threshold`
- `dino_seed_res_threshold`
- `dino_grow_freq_threshold`
- `dino_grow_res_threshold`
- `dino_component_count`
- `dino_mask_fraction`
- `dino_connected_fraction`
- `dino_timing_total_ms`

## Current ML status

- `use_pytorch_backend=true` activates the LibTorch runtime path when Torch is available at build/runtime.
- `backend_mode` controls the operator-side postprocess path:
  - `reference`: keep the notebook-faithful coherent-shell residual-veto hybrid path.
- Validation and production parity requires `backend_mode=reference` everywhere that is intended to validate or represent production mask generation. `emit_stride` may change for throughput tuning, but `backend_mode` must not.
- `inference_backend` controls behavior:
  - `torchscript`: attempts TorchScript model forward using `model_script_path`, then returns a DINO score map to the operator.
  - `pytorch_placeholder`: retains the runtime entry point but does not provide a production DINO score.
  - `cuda_threshold_fallback`: bypasses the DINO runtime and uses the coherent-style gate as the fallback score.
- The current hot path is split intentionally:
  - coherent-style GPU preprocessing and gate generation live in the operator,
  - DINO model execution lives in the Torch runtime helper,
  - final hybrid mask generation lives in the operator.
- `frontend_correction_enable`, `ignore_sideband_hz`, `dino_coherence_gate_floor`, and `dino_coherence_gate_span_db` are the active controls for the coherent-shell hybrid path.
- `timing_summary_enable=true` emits mean/max timing summaries for the major detector stages every `timing_summary_every_n` emitted frames.
- `torchscript_init_mode` controls how far C++ initialization proceeds before compute begins:
  - `load_only`: load the TorchScript file only.
  - `load_cpu_eval`: load and call `eval()` on CPU.
  - `load_cuda_no_eval`: load and move the module to CUDA.
  - `load_cuda_eval`: load, move to CUDA, and call `eval()`.
  - If TorchScript load/forward fails and `strict_model_forward=false`, execution falls back to the coherent-style gate path for that frame.
  - The final operator mask now follows the residual-veto hybrid notebook logic rather than the earlier DINO-plus-power fusion path.
- The selected runtime weight is `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` and is expected to be staged under `/workspace/models/dinov3/weights` inside the Holohub container.
- The expected TorchScript runtime artifact path is `/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`.
- For validation runs, keep `strict_model_forward=true` so model-load or model-forward issues are surfaced immediately.
- `enable_mask_save=true` writes debug `.pgm` mask images to `output_dir` using the first `max_masks_per_channel` frames that match `save_every_n_frames`.

## Verification Notes

- Use `applications/usrp_wideband_signal_detection/config_torchscript_validation.yaml` for strict reproduction-oriented bring-up.
- Compare the operator metadata and timing summaries against `signal_detection_holoscan_retry_dino.ipynb`, especially the final residual-veto hybrid experiment.
- The active implementation is closest to: coherent-power shell plus notebook cell-14-style hybrid cleanup.

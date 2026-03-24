# Sage Step-by-Step Implementation Plan

Last updated: 2026-03-24

## 1) Purpose and Scope

This document defines a **careful, executable implementation plan** for adding:

1. A new **Spectrogram Operator**
2. A new **DINOv3 Signal Detection stage** (operator or InferenceOp integration)

for a high-throughput SDR workload targeting **2 channels at 500 Msps each**.

This is a delivery plan (not code), designed to be followed incrementally in future sessions.


## 2) Success Criteria

The project is considered successful when all of the following are true:

- The existing VITA49 PSD export path remains functional.
- A new GPU-resident ML branch is integrated and stable.
- Spectrogram tensors are generated with deterministic shape and metadata.
- DINOv3 inference consumes spectrogram tensors without host round-trips in the hot path.
- Inference cadence is explicitly controlled (stride/drop/window), avoiding overload.
- Throughput and latency are measured and documented under representative load.


## 3) Architecture Target (End State)

### 3.1 Existing branch retained

- `vitaConnectorOp -> fftOp -> highRatePsdOp -> lowRatePsdOp -> packetizerOp`

### 3.2 New ML branch added

- `fftOp -> spectrogramOp -> dinov3InferenceOp -> detectionPostprocessOp`

### 3.3 Why this split

- Packetizer branch is transport-focused and includes GPU->CPU copy.
- ML branch must stay GPU-native for throughput and latency.


## 4) Project Phases

Each phase includes: objective, tasks, file touchpoints, validation, and exit criteria.

---

## Phase 0 — Baseline and Instrumentation

### Objective

Establish performance and correctness baselines before adding new operators.

### Tasks

1. Record current pipeline behavior under representative test conditions.
2. Capture baseline metrics:
   - ingest packet rate
   - FFT batch latency
   - PSD branch end-to-end latency
   - GPU memory footprint
3. Confirm metadata keys continuity (`channel_number`, timestamps, stream_id, RF context).
4. Decide baseline test profile (short smoke + sustained run).

### File touchpoints

- `applications/psd_pipeline/main.cpp`
- `applications/psd_pipeline/config.yaml`
- `applications/psd_pipeline/advanced_network_connectors/vita49_rx.cu`

### Validation

- No dropped packets in short smoke run.
- Metadata fields present at packetizer stage.

### Exit criteria

- Baseline metrics saved in project notes.
- Baseline config committed to team workflow.

---

## Phase 1 — Spectrogram Operator Skeleton

### Objective

Add operator scaffolding and build integration without changing runtime behavior yet.

### Tasks

1. Create new operator directory and build files.
2. Define operator API:
   - input port contract
   - output port contract
   - config parameters
3. Register operator with top-level operators CMake.
4. Add README and metadata docs.

### Proposed file touchpoints

- New:
  - `operators/spectrogram/spectrogram.hpp`
  - `operators/spectrogram/spectrogram.cu`
  - `operators/spectrogram/CMakeLists.txt`
  - `operators/spectrogram/README.md`
  - `operators/spectrogram/metadata.json`
- Update:
  - `operators/CMakeLists.txt`

### Validation

- Repository builds with operator included.
- Operator can be instantiated with config.

### Exit criteria

- Clean build and no graph wiring changes required yet.

---

## Phase 2 — Spectrogram Compute Path (GPU)

### Objective

Implement core GPU spectrogram logic with per-channel state and fixed-shape outputs.

### Tasks

1. Implement per-channel state:
   - rolling/ring buffer
   - write index
   - frame-ready logic
2. Implement compute pipeline:
   - consume FFT output (`tensor_t<complex,2> + stream`)
   - optional power conversion
   - log scaling / normalization policy
   - produce fixed tensor shape for inference
3. Ensure no host memory transfers in hot path.
4. Ensure metadata propagation:
   - channel ID
   - timing context
   - optional frame index

### Required configuration parameters (initial)

- `num_channels`
- `freq_bins`
- `time_bins`
- `hop_size`
- `overlap_mode` or equivalent
- `output_dtype` (`fp16` preferred option, `fp32` fallback)
- `normalize_mode`
- `emit_stride` (for inference gating)

### Validation

- Output shapes are deterministic and match config.
- Channel separation is correct (no cross-channel contamination).
- GPU memory remains bounded during sustained run.

### Exit criteria

- Spectrogram op produces stable tensors and metadata under load.

---

## Phase 3 — Pipeline Wiring (Branching)

### Objective

Wire spectrogram branch into app graph while preserving legacy PSD branch.

### Tasks

1. Add `spectrogramOp` to app compose.
2. Branch from `fftOp` output:
   - one branch to existing PSD flow
   - one branch to spectrogram flow
3. Add config section for spectrogram parameters.
4. Verify no regressions in packetizer branch.

### File touchpoints

- `applications/psd_pipeline/main.cpp`
- `applications/psd_pipeline/config.yaml`

### Validation

- Both branches run concurrently.
- Legacy PSD UDP output remains valid.

### Exit criteria

- Spectrogram branch active and isolated from PSD branch regressions.

---

## Phase 4 — DINOv3 Inference Integration

### Objective

Attach DINOv3 inference to spectrogram outputs using the most maintainable path.

### Notebook reference for algorithm flow

- Reference implementation notebook:
   - `../Dinov3-RF-Signal-Detection/rf_spectrogram_segmentation.ipynb`
- This notebook defines the current algorithmic flow to preserve during operatorization:
   1. Spectrogram slice/window selection (`H x W`, patch-size aligned).
   2. Input normalization to image-like tensor (ImageNet mean/std style in the Python prototype).
   3. DINOv3 feature extraction (patch embeddings).
   4. Patch-space postprocess for detection mask candidates (prototype uses PCA + clustering and mask scoring).
   5. Visualization/debug outputs (prototype only; not in hot path).

### Decision gate

Choose one:

- **Path A (preferred first):** Holoscan `InferenceOp` backend=TRT with CUDA tensors.
- **Path B:** custom DINOv3 operator if preprocessing/postprocessing requirements exceed InferenceOp flexibility.

### Implementation decision for this project stage

- **Primary execution target:** C++/CUDA operator path for production throughput.
- **Role of Python notebook:** algorithm prototyping and regression reference only.
- **Recommended staging:**
   1. Use `InferenceOp` as bring-up path for engine/runtime validation.
   2. Implement `dinov3_signal_detector` custom C++ operator for fused preprocess + inference glue + postprocess once contracts are stable.

### Tasks (common)

1. Define model input tensor mapping and expected layout.
2. Define output contract:
   - logits/embedding/detections
   - metadata attached to output messages
3. Add explicit inference cadence control:
   - frame stride
   - optional queue depth cap
4. Add model config and runtime options:
   - precision mode
   - engine cache behavior
   - device assignment

### Tasks (C++ operator track: required for performance)

1. Define `dinov3_signal_detector` operator contract:
   - input: spectrogram tensor + metadata (`channel_number`, timing/frame index)
   - output: detection mask(s), confidence/score summary, passthrough metadata
2. Implement GPU preprocess in operator (no host copy):
   - patch-size alignment/cropping policy (e.g., multiple of 16)
   - dtype/layout conversion to model input layout
   - normalization policy equivalent to notebook baseline
3. Integrate TRT inference execution path:
   - engine load/init in `initialize()`
   - async enqueue on provided CUDA stream in `compute()`
4. Implement GPU postprocess (minimum viable):
   - produce deterministic binary/score mask from model outputs
   - include threshold parameters and deterministic tie-break rules
5. Add debug parity hooks (off by default):
   - optional sampled host export for parity checks vs notebook outputs
6. Add cadence/backpressure controls in operator config:
   - `emit_stride`, queue cap, frame-drop policy

### File touchpoints

- If Path A:
  - `applications/psd_pipeline/main.cpp`
  - `applications/psd_pipeline/config.yaml` (or dedicated app config)
- If Path B:
   - new `operators/dinov3_signal_detector/dinov3_signal_detector.hpp`
   - new `operators/dinov3_signal_detector/dinov3_signal_detector.cu`
   - new `operators/dinov3_signal_detector/CMakeLists.txt`
   - new `operators/dinov3_signal_detector/README.md`
   - new `operators/dinov3_signal_detector/metadata.json`
  - `operators/CMakeLists.txt`
  - app wiring/config files above

### Initial operator parameters (proposed)

- `num_channels`
- `input_height`, `input_width`
- `patch_size` (default 16)
- `input_layout` (`NCHW` default)
- `input_dtype` (`fp16` default, `fp32` fallback)
- `normalize_mode` (`imagenet`, `none`, or custom constants)
- `engine_path` / `onnx_path` / `engine_cache_dir`
- `infer_batch_size`
- `mask_threshold`
- `postprocess_mode` (`argmax`, `threshold`, future `cluster`)
- `emit_stride`
- `max_inflight`
- `debug_dump_enable`
- `debug_dump_every_n`

### Validation

- Inference receives expected tensor shape and dtype.
- Output message schema is stable.
- End-to-end branch runs at controlled cadence.

### Validation additions for notebook parity

1. Fixed-input parity test:
   - run one captured spectrogram slice through notebook reference and C++ operator path
   - compare output mask overlap metric (IoU) and summary statistics
2. Throughput validation:
   - verify no GPU->CPU transfer in hot path (except optional debug mode)
3. Determinism check:
   - same input + config yields identical mask output over repeated runs

### Exit criteria

- DINOv3 stage functional in integrated pipeline.

---

## Phase 5 — Throughput Optimization and Backpressure Control

### Objective

Harden for high-rate sustained operation.

### Tasks

1. Profile kernel and memory bandwidth hotspots.
2. Tune spectrogram parameters for compute budget.
3. Add/adjust backpressure strategy:
   - bounded buffering
   - deterministic frame dropping policy
4. Tune scheduler and thread/resource settings.
5. Verify channel fairness (both channels serviced consistently).

### Validation metrics

- sustained runtime stability
- no unbounded queue growth
- inference latency distribution
- GPU memory plateau

### Exit criteria

- Pipeline meets target operational envelope with documented tuning.

---

## Phase 6 — Verification Matrix and Acceptance Testing

### Objective

Formalize pass/fail checks before declaring production readiness.

### Test matrix

1. **Functional tests**
   - metadata continuity
   - shape/dtype correctness
   - deterministic channel routing
2. **Performance tests**
   - short burst stress
   - sustained run
   - inference cadence under load
3. **Failure-mode tests**
   - context-change behavior
   - packet burst irregularity
   - inference stall simulation

### Exit criteria

- All mandatory checks pass and results are documented.

---

## Phase 7 — Documentation and Operational Handoff

### Objective

Make the solution maintainable by future contributors.

### Tasks

1. Update operator READMEs with config and data contracts.
2. Add architecture diagram (branching flow).
3. Document tuning guide:
   - recommended defaults
   - safe ranges
   - troubleshooting notes
4. Add known limitations and roadmap items.

### Exit criteria

- Documentation is sufficient for another engineer to run and modify the pipeline safely.


## 5) Detailed Step Checklist (Execution Order)

Use this list in future sessions as the default execution sequence.

1. Baseline capture completed and archived.
2. Spectrogram operator scaffold created and build passes.
3. Spectrogram compute path implemented with deterministic output shape.
4. Spectrogram wired into app as side branch from FFT.
5. DINOv3 integration path selected (InferenceOp first, custom only if needed).
6. DINOv3 stage wired and functional with CUDA-resident tensors.
7. Cadence control and backpressure policy enabled and validated.
8. Sustained throughput tests executed and tuned.
9. Verification matrix executed and signed off.
10. Docs finalized and merged.


## 6) Key Technical Decisions to Lock Early

These should be decided before Phase 2 completion:

1. **Spectrogram output layout** (`NCHW` vs `NHWC`)
2. **Output precision** (`FP16` preferred for throughput unless model accuracy needs FP32)
3. **Inference cadence policy** (every frame vs stride/window)
4. **DINOv3 deployment path** (InferenceOp TensorRT vs custom operator)
5. **Per-channel model policy** (shared model instance vs channel-specific handling)
6. **Postprocess definition** (model-native mask head vs patch-feature clustering fallback)
7. **Parity metric** (e.g., IoU threshold against notebook reference outputs)


## 7) Risks and Mitigations

### Risk 1: Inference overload at full FFT emission rate
- **Mitigation:** enforce stride/window gating before inference.

### Risk 2: GPU memory pressure from rolling spectrogram history
- **Mitigation:** fixed-size ring buffers + preallocated tensors + bounded queues.

### Risk 3: Hidden host transfer in ML branch
- **Mitigation:** audit every operator boundary for device-resident tensor flow.

### Risk 4: Metadata drift across branches
- **Mitigation:** create a metadata contract checklist and validate each stage.

### Risk 5: Regression in legacy packetizer branch
- **Mitigation:** preserve branch separation and run legacy smoke tests after each integration step.


## 8) Completion Definition (Project Done)

The project is complete when:

- both new operators/stages are integrated,
- high-throughput tests pass with documented settings,
- branch-level behavior is deterministic,
- legacy PSD export still works,
- and documentation allows repeatable deployment and troubleshooting.


## 9) Immediate Next Session Starter (Recommended)

Start with **Phase 0** and produce a short baseline report including:

1. config profile used,
2. observed packet/FFT cadence,
3. current latency and memory stats,
4. issues to resolve before Phase 1.

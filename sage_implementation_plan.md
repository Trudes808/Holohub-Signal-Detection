# Sage Step-by-Step Implementation Plan

Last updated: 2026-04-01

## Current re-entry status (2026-04-01)

Completed since the original step-10 resume:

- container helper scripts were added for build, run, setup, and shell entry,
- the local DINOv3 repo and weights are staged into the Holohub container and exported to TorchScript in-container,
- the application builds successfully in local container mode with Torch enabled,
- a strict TorchScript runtime attempt still segfaults in the C++ operator initialization path,
- the active default config was intentionally moved to a stable debug-artifact mode,
- separate TorchScript validation and load-only diagnostic configs are now part of the application build output.

Immediate next steps:

1. run `config.yaml` to verify the first 5 spectrograms and first 5 detector masks are written as expected,
2. run `config_torchscript_load_only.yaml` to confirm whether `torch::jit::load(...)` remains stable in the live application process,
3. run `config_torchscript_validation.yaml` to identify whether the crash happens before or after the new `to_cuda` and `eval` stage logs.

## 1) Purpose and Scope

This document defines a careful, executable implementation plan for the current Holohub signal-detection integration path.

The active implementation target is the dedicated wideband detection application:

- `applications/usrp_wideband_signal_detection`

This plan covers:

1. hardening the existing Spectrogram and DinoV3 signal-detector operator path,
2. packaging the local DINOv3 repository and weights for use inside the Holohub container,
3. exporting and validating a TorchScript model-forward path, and
4. restoring sustained multi-channel testing after strict single-channel validation.

The earlier phase structure in this document remains useful as implementation history, but the current re-entry point for active development is the dedicated wideband application and the step-10 resume checklist below.

This is a delivery plan, not code, designed to be followed incrementally in future sessions.


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

### Current implementation status

The original phases below describe the intended build-out order. In the current repository state, the major scaffolding work has already moved into a dedicated application and custom operators:

- `applications/usrp_wideband_signal_detection`
- `operators/spectrogram`
- `operators/dinov3_signal_detector`

Because of that, active development should resume from the step-10 re-entry checklist rather than restarting at Phase 0 unless baseline re-validation is specifically needed.

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

Resume from section 10 using the dedicated wideband detection application and the packaged DINOv3 runtime path.

The first session back should produce a short packaging and runtime-readiness report including:

1. confirmation that the local DINOv3 repo exists at `/home/sat3737/holoscan_demo_workspace/dinov3`,
2. confirmation that the selected weight file exists at `/home/sat3737/holoscan_demo_workspace/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`,
3. confirmation that those assets have been staged into the Holohub container under `/workspace/models/dinov3`,
4. confirmation that a TorchScript export exists at the final container runtime path,
5. confirmation that the app config points to real staged artifacts rather than placeholders,
6. confirmation that spectrogram debug image saving is disabled for model-forward validation runs, and
7. any build or runtime issues that must be resolved before strict single-channel bring-up.


## 10) Resume Plan For Holohub Re-Entry

Use this section as the exact re-entry checklist for returning to Holohub development on the current wideband signal-detection path.

### Current re-entry status (2026-04-01)

Completed in the current session:

1. Step 10.0 is complete.
   - The local DINOv3 repo and selected weight were staged into `/workspace/models/dinov3` and `/workspace/models/dinov3/weights`.
2. Step 10.1 is complete.
   - TorchScript export now succeeds on GPU and writes:
   - `/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`
3. Step 10.2 is complete.
   - Runtime artifact paths exist in-container and active config points at the staged `.pth` and `.ts` artifacts.
4. Step 10.3 is complete.
   - `applications/usrp_wideband_signal_detection/config.yaml` is aligned to the staged artifacts, strict model-forward validation, and spectrogram save disabled.
5. Step 10.4 is complete.
   - `usrp_wideband_signal_detection` now configures and builds successfully in the Holohub container, including the Torch-enabled `dinov3_signal_detector` path.

Next active step:

1. Start at step 10.5.
2. Perform first runtime bring-up with reduced load and verify:
   - TorchScript loads from the staged container path
   - detector metadata reports `dino_backend=torchscript`
   - no fallback warnings are emitted
   - mask output is produced on the strict model-forward path

### 10.0 Package DINOv3 assets into the Holohub container

Treat the host workspace as the source of truth and the container path as the runtime source of truth.

Host-side source of truth:

1. Local DINOv3 repository:
   - `/home/sat3737/holoscan_demo_workspace/dinov3`
2. Selected weight artifact:
   - `/home/sat3737/holoscan_demo_workspace/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

Container-side runtime target:

1. Repository root:
   - `/workspace/models/dinov3`
2. Weights directory:
   - `/workspace/models/dinov3/weights`

Packaging requirements:

1. Stage the local DINOv3 repository into `/workspace/models/dinov3` as container-managed content.
2. Stage the exact selected weight file into `/workspace/models/dinov3/weights`.
3. Include the repository files required by the current notebook-based local load flow, especially:
   - `hubconf.py`
   - Python package sources under `dinov3/`
   - any repo metadata needed by local `torch.hub` loading
4. Record the final canonical container paths that runtime config and export tooling will use.

If packaging is incomplete, stop here and resolve container asset staging first.

### 10.1 Export the runtime model artifact inside the container

Before app bring-up, export a TorchScript artifact from the packaged container-side repo and weight set.

Requirements:

1. Run export from inside the target Holohub container environment.
2. Use the packaged repository under `/workspace/models/dinov3`.
3. Use the staged weight file `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
4. Write the exported TorchScript model into the canonical runtime tree, preferably under `/workspace/models/dinov3/weights`.
5. Record the exact exported filename that will be referenced by config.

Rationale:

- this avoids host-versus-container mismatches in PyTorch, CUDA, and dependency behavior.

Exit criteria:

- a real TorchScript artifact exists on disk inside the container runtime layout.
- status: complete on 2026-04-01 with export path `/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`.

### 10.2 Prerequisite runtime gate

Confirm all required runtime artifacts exist before code or config validation:

1. The container repo path exists and is readable:
   - `/workspace/models/dinov3`
2. The exact staged weight file exists.
3. The exported TorchScript file exists.
4. The target app config points to those exact staged artifact paths.
5. No placeholder filenames remain in active config or detector documentation.

If any artifact is missing or any placeholder remains, stop and resolve that before bring-up.

### 10.3 Config finalize step

Update app config in `applications/usrp_wideband_signal_detection/config.yaml`:

1. Set `dinov3_signal_detector.inference_backend` to `torchscript`.
2. Keep `use_pytorch_backend` enabled.
3. Replace placeholder values for:
   - `model_repo_path`
   - `weights_path`
   - `model_script_path`
4. Set `strict_model_forward: true` for validation runs.
5. Disable `spectrogram.enable_save` for bring-up and performance runs.

Reason for disabling spectrogram save during validation:

- the current spectrogram operator still performs device-to-host copies and stream synchronization when writing debug images, so it should be treated as a debug-only path during model-forward validation.

### 10.4 Build validation

Before runtime testing:

1. Reconfigure and rebuild the app inside the Holohub container.
2. Confirm the detector builds through the Torch-enabled branch in `operators/dinov3_signal_detector/CMakeLists.txt`.
3. Verify the detector is compiled with Torch support enabled before any runtime inference testing begins.

Exit criteria:

- build output confirms Torch support is active for `dinov3_signal_detector`.
- status: complete on 2026-04-01; app binary produced at `build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection/usrp_wideband_signal_detection`.

### 10.5 Operator bring-up sequence

Perform bring-up in this order:

1. Start with single channel and reduced load.
2. Verify operator logs show TorchScript loaded successfully from the staged container path.
3. Verify metadata reports `dino_backend=torchscript`.
4. Confirm no fallback warnings are emitted.
5. Confirm the detector emits masks without dropping into `pytorch_placeholder` or `cuda_threshold_fallback`.

Exit criteria:

- model-forward path runs without fallback for at least a short sustained single-channel run.

Immediate next action:

1. Launch the built app inside the refreshed container using the strict TorchScript configuration already in `applications/usrp_wideband_signal_detection/config.yaml`.
2. Keep channel count and load reduced for first runtime confirmation.
3. Capture operator logs and metadata for the first successful `torchscript` inference pass before any throughput tuning.

### 10.6 Input contract audit

Validate and document the current detector input contract before broader optimization:

1. Confirm the current spectrogram operator contract is still passthrough-oriented rather than a final image-tensor producer.
2. Confirm the detector currently derives power, resize behavior, and thresholding directly from FFT-domain input.
3. Record this as the accepted short-term contract for bring-up.
4. Defer any true spectrogram-tensor refactor until after stable TorchScript model-forward validation.

Exit criteria:

- the temporary input contract is documented and acknowledged as intentional.

### 10.7 Parity validation against notebook references

Run one fixed captured input through both the notebook and C++ paths.

Notebook references:

1. `noise_detection_dino_experiments2.ipynb`
   - source of truth for local repo loading assumptions, weight selection, and patch-size assumptions
2. `rf_spectrogram_segmentation.ipynb`
   - preprocessing and mask-parity reference

Validation layers:

1. Preprocessing parity:
   - compare the power-domain and resized model input used by the detector path against the notebook reference
2. Output parity:
   - compare final masks using IoU, foreground area fraction, and basic localization consistency

Exit criteria:

- parity metrics meet the agreed threshold and are recorded.

### 10.8 Throughput restoration and scale-up

After strict bring-up and parity pass:

1. Restore the target 2-channel profile.
2. Re-enable representative FFT and detector settings.
3. Tune `emit_stride`, logging, and scheduler settings.
4. Keep spectrogram saving disabled unless it is explicitly needed for sampled debug output.
5. Capture latency, bounded-memory behavior, and sustained-run stability metrics.

Exit criteria:

- stable sustained run with bounded memory, controlled inference cadence, and no unintended backend regressions.

### 10.9 Immediate follow-on tasks

1. Add a postprocess or sink stage after the detector so outputs are validated by more than logs and metadata.
2. Promote debug metadata into a formal downstream output schema.
3. Add explicit troubleshooting notes for model-load, export, packaging, and fallback modes.
4. Decide whether the spectrogram operator remains a debug saver or becomes the true tensor-producing preprocessing stage.

### 10.10 Session restart quick checks

Use the following session checklist after any pause in work:

1. Confirm host-side repo and selected weight still exist.
2. Confirm packaged container-side repo and weight staging still exist.
3. Confirm the exported TorchScript artifact still exists at the configured path.
4. Confirm `applications/usrp_wideband_signal_detection/config.yaml` still points to the staged runtime paths and not placeholders.
5. Rebuild and confirm Torch support is still enabled.
6. Re-run strict single-channel validation before returning to 2-channel throughput tuning.

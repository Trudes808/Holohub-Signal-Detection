# Sage Step-by-Step Implementation Plan

Last updated: 2026-04-12

## Current re-entry status (2026-04-12)

Completed since the original step-10 resume:

- container helper scripts were added for build, run, setup, and shell entry,
- the local DINOv3 repo and weights are staged into the Holohub container and exported to TorchScript in-container,
- the application builds successfully in local container mode with Torch enabled,
- the wideband signal detection application now runs against the live USRP hardware,
- RF receive and spectrogram plotting were validated independently in `notebooks/test_radio.ipynb`,
- a strict single-channel validation path and helper were added for low-risk live detector bring-up,
- the single-channel validation path now runs end-to-end on live input with spectrogram and mask artifact support,
- host-side UHD bring-up issues were resolved by using the system Python 3.12 interpreter and the correct X410 address,
- a dedicated 2-channel low-overhead performance config and launcher helper were added for throughput testing, and
- the active resume point is now performance tuning and low-drop validation on the real TorchScript detector path.

Immediate next steps:

1. add staged pipeline-isolation configs so ingress+FFT, ingress+FFT+spectrogram, and ingress+FFT+spectrogram+detector can be measured separately,
2. tune receive buffering and batching until the throughput ceiling clearly moves downstream to the detector stage,
3. define success for the pre-detector tuning pass as: FFT-only and spectrogram-only runs sustain the target 2-channel rate with minimal RX buffer drops while the detector-enabled path still shows a clear additional throughput cliff,
4. once the detector is isolated as the dominant bottleneck, redesign the detector hot path to eliminate host round-trips and blocking stream synchronizations,
5. after detector rewrite work, compare behavior and constants against `../Dinov3-RF-Signal-Detection/signal_detection_holoscanv2.ipynb`, and
6. only after functionality and low-drop throughput are stable use `../Dinov3-RF-Signal-Detection/speed_optimization_todo.md` to drive deeper optimization work.

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

- Primary reproduction notebook:
   - `../Dinov3-RF-Signal-Detection/signal_detection_holoscanv1.ipynb`
- Supporting earlier spectrogram notebook:
   - `../Dinov3-RF-Signal-Detection/rf_spectrogram_segmentation.ipynb`
- `signal_detection_holoscanv1.ipynb` is now the algorithmic source of truth for the active detector path and defines the behavior to preserve during operatorization:
   1. sideband-ignore calculation and optional frontend correction before DINO input formation,
   2. patch-size alignment, resize/crop policy, and ImageNet-style normalization,
   3. DINOv3 feature extraction through the current PyTorch model path,
   4. DINO grouping, coherence gating, texture scoring, power scoring, and multilevel final-mask fusion, and
   5. per-slice timing checkpoints that should be mirrored in the C++ debug path.

### Implementation decision for this project stage

- **Primary execution target:** custom `dinov3_signal_detector` C++/CUDA operator using LibTorch / TorchScript for model execution.
- **Role of Python notebook:** algorithmic source of truth for constants, parity, and debug expectations; not the production runtime.
- **Active staging order:**
   1. validate the current strict TorchScript model-forward path on reduced live input,
   2. copy the notebook detector constants and postprocess behavior into the C++ path while keeping tensors GPU-resident across the hot path,
   3. expose notebook-derived tuning constants in app config rather than hard-coding them in the operator, and
   4. add timing instrumentation before any optimization pass.

### Tasks (common)

1. Define model input tensor mapping and expected layout.
2. Define output contract:
   - logits/embedding/detections
   - metadata attached to output messages
3. Promote notebook-derived detector constants into config with stable names and documented defaults.
4. Add explicit inference cadence control:
   - frame stride
   - optional queue depth cap
5. Add model config and runtime options:
   - precision mode
   - TorchScript / LibTorch artifact behavior
   - device assignment

### Tasks (C++ operator track: required for performance)

1. Define `dinov3_signal_detector` operator contract:
   - input: spectrogram tensor + metadata (`channel_number`, timing/frame index)
   - output: detection mask(s), confidence/score summary, passthrough metadata
2. Reproduce the active notebook preprocessing path on GPU in operator code:
   - optional frontend correction and sideband-ignore crop behavior
   - patch-size alignment/cropping policy (multiple of `patch_size`)
   - dtype/layout conversion to model input layout
   - normalization policy equivalent to notebook baseline
3. Integrate LibTorch / TorchScript inference execution path:
   - model load/init in `initialize()`
   - async forward launch on the provided CUDA stream in `compute()`
4. Reproduce the current notebook postprocess path in C++ with GPU-first execution:
   - DINO grouping constants and affinity behavior
   - coherence gating
   - texture and power scoring
   - multilevel fusion and final mask cleanup
5. Add debug parity hooks (off by default):
   - optional sampled host export for parity checks vs notebook outputs
   - optional dump of intermediate maps for one selected frame when parity debugging is needed
6. Add debug timing instrumentation for major steps:
   - input staging / shape match
   - frontend correction and sideband crop
   - DINO preprocess and TorchScript forward
   - grouping / coherence / texture / power / final fusion
   - total operator compute time and emitted summary statistics
7. Add cadence/backpressure controls in operator config:
   - `emit_stride`, queue cap, frame-drop policy

### File touchpoints

- `applications/usrp_wideband_signal_detection/config.yaml`
- `applications/usrp_wideband_signal_detection/main.cpp`
- `operators/dinov3_signal_detector/dinov3_signal_detector.hpp`
- `operators/dinov3_signal_detector/dinov3_signal_detector.cpp` or `operators/dinov3_signal_detector/dinov3_signal_detector.cu`
- `operators/dinov3_signal_detector/CMakeLists.txt`
- `operators/dinov3_signal_detector/README.md`
- `operators/dinov3_signal_detector/metadata.json`
- `operators/CMakeLists.txt`

### Initial operator parameters (promote from notebook into config)

- `num_channels`
- `input_height`, `input_width`
- `patch_size` (default from model, currently 16)
- `input_layout` (`NCHW` default)
- `input_dtype` (`fp16` default, `fp32` fallback)
- `imagenet_mean`, `imagenet_std`
- `fft_size`
- `noverlap`
- `ignore_sideband_hz`
- `frontend_correction_enable`
- `frontend_correction_row_q`
- `frontend_correction_smooth_sigma`
- `frontend_correction_reference_q`
- `frontend_correction_max_boost_db`
- `frontend_correction_soft_knee_db`
- `frontend_correction_edge_taper_fraction`
- `frontend_correction_edge_taper_sigma`
- `frontend_correction_edge_target_drop_db`
- `frontend_edge_guard_floor`
- `dino_coherence_gate_floor`
- `texture_q`
- `texture_k`
- `power_q`
- `dino_group_k`
- `dino_group_spatial_weight`
- `dino_group_score_q`
- `pipeline_final_threshold`
- `pipeline_final_threshold_no_speckle`
- `pipeline_gap_floor`
- `pipeline_component_min_size`
- `pipeline_component_min_size_no_speckle`
- `pipeline_power_rescue_floor`
- `pipeline_power_rescue_gain`
- `pipeline_strong_speckle_min_component`
- `pipeline_texture_speckle_clean_threshold`
- `pipeline_texture_speckle_strong_threshold`
- `model_repo_path`
- `weights_path`
- `model_script_path`
- `infer_batch_size`
- `emit_stride`
- `max_inflight`
- `debug_dump_enable`
- `debug_dump_every_n`
- `timing_summary_enable`
- `timing_summary_every_n`
- `timing_summary_window`

### Validation

- Inference receives expected tensor shape and dtype.
- Output message schema is stable.
- End-to-end branch runs at controlled cadence.

### Validation additions for notebook parity

1. Fixed-input parity test:
   - run one captured spectrogram slice through `signal_detection_holoscanv1.ipynb` and the C++ operator path
   - compare output mask overlap metric (IoU), foreground fraction, and threshold-sensitive summary statistics
2. Throughput validation:
   - verify no GPU->CPU transfer in hot path (except optional debug mode)
3. Determinism check:
   - same input + config yields identical mask output over repeated runs
4. Timing visibility check:
   - operator logs emit per-stage timing summaries that can be compared against notebook timing categories

### Exit criteria

- DINOv3 stage functional in integrated pipeline.

---

## Phase 5 — Throughput Optimization and Backpressure Control

### Objective

Harden for high-rate sustained operation.

Start this phase only after the strict single-channel detector path is functional enough to prove the runtime path and the low-overhead 2-channel performance config is in place.

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
- minimal RX buffer drops under representative 2-channel load
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
4. **DINOv3 deployment path** (custom LibTorch/TorchScript operator as active target, with other backends only as contingency)
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
6. confirmation that spectrogram debug image saving is disabled for performance-oriented runs,
7. confirmation that a low-overhead 2-channel performance config exists for drop-rate testing, and
8. any build or runtime issues that must be resolved before sustained 2-channel throughput tuning.


## 10) Resume Plan For Holohub Re-Entry

Use this section as the exact re-entry checklist for returning to Holohub development on the current wideband signal-detection path.

### Current re-entry status (2026-04-12)

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
6. RF hardware validation is complete.
   - The app can now run against the live USRP without blocking on basic device bring-up.
7. Spectrogram validation is complete.
   - `notebooks/test_radio.ipynb` confirmed short IQ capture, receive power plotting, and spectrogram rendering on the target radio path.
8. Step 10.5 is partially complete.
   - A strict single-channel live-validation config and launcher were added.
   - The graph now runs end-to-end on live RF input and can emit saved spectrograms and detector masks for bring-up.
   - The main remaining concern is ingress packet loss during validation, not basic graph startup.
9. Throughput-prep work is complete.
   - A dedicated low-overhead 2-channel performance config and launcher were added.
   - The performance path disables artifact saves, detailed per-frame logs, and timing summaries while increasing RX buffer pools and in-flight batching.

Next active step:

1. Run the 2-channel low-overhead performance configuration at the expected radio rate.
2. Capture and review:
   - `RX out of buffers`
   - `rx_mbuf_allocation_errors`
   - per-queue application packet totals
   - any backend fallback warnings
3. Tune the performance config until drops are minimal enough that detector quality and throughput measurements are meaningful.
4. After that baseline is stable, return to notebook-parity review against `../Dinov3-RF-Signal-Detection/signal_detection_holoscanv2.ipynb`.
5. Keep `notebooks/test_radio.ipynb` as the RF sanity-check path if live detector output becomes ambiguous during tuning.

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

### 10.5 Strict operator bring-up sequence

Perform bring-up in this order:

1. Start with single channel and reduced load.
2. Verify operator logs show TorchScript loaded successfully from the staged container path.
3. Verify metadata reports `dino_backend=torchscript`.
4. Confirm no fallback warnings are emitted.
5. Confirm the detector emits masks without dropping into `pytorch_placeholder` or `cuda_threshold_fallback`.

Exit criteria:

- model-forward path runs without fallback for at least a short sustained single-channel run.

Status update (2026-04-12):

- Partially complete.
- The reduced-load live validation path exists and runs end-to-end.
- Debug artifact saving during validation was useful for bring-up but materially increases backpressure and packet drops.
- Treat the current single-channel validation config as a correctness and debugging config, not a throughput config.

Immediate next action:

1. Preserve the single-channel validation path as the debug config for detector bring-up.
2. Do not use it for drop-rate conclusions because spectrogram saving, mask saving, and per-frame timing/logging distort throughput behavior.
3. Move the next round of runtime testing to the dedicated 2-channel performance config.
4. If the performance path still shows material drops, tune buffer pools and batching before changing detector logic.

### 10.6 Notebook-constant promotion and C++/GPU reproduction

Use `../Dinov3-RF-Signal-Detection/signal_detection_holoscanv1.ipynb` as the immediate source of truth for the first non-placeholder detector implementation.

Tasks:

1. Copy the current detector constants into `applications/usrp_wideband_signal_detection/config.yaml` and operator params using notebook-aligned names where practical.
2. Promote at least the following notebook defaults into config:
   - `DINO_COHERENCE_GATE_FLOOR = 0.25`
   - `TEXTURE_Q = 0.90`
   - `TEXTURE_K = 6`
   - `POWER_Q = 0.90`
   - `DINO_GROUP_K = 8`
   - `DINO_GROUP_SPATIAL_WEIGHT = 0.35`
   - `DINO_GROUP_SCORE_Q = 0.60`
   - `PIPELINE_FINAL_THRESHOLD = 0.2`
   - `PIPELINE_FINAL_THRESHOLD_NO_SPECKLE = 0.10`
   - `PIPELINE_GAP_FLOOR = 0.10`
   - `PIPELINE_COMPONENT_MIN_SIZE = 5`
   - `PIPELINE_COMPONENT_MIN_SIZE_NO_SPECKLE = 2`
   - `PIPELINE_POWER_RESCUE_FLOOR = 0.10`
   - `PIPELINE_POWER_RESCUE_GAIN = 2.0`
   - `PIPELINE_STRONG_SPECKLE_MIN_COMPONENT = 10`
   - `PIPELINE_TEXTURE_SPECKLE_CLEAN_THRESHOLD = 0.85`
   - `PIPELINE_TEXTURE_SPECKLE_STRONG_THRESHOLD = 0.20`
   - `FRONTEND_CORRECTION_ROW_Q = 25.0`
   - `FRONTEND_CORRECTION_SMOOTH_SIGMA = 12.0`
   - `FRONTEND_CORRECTION_REFERENCE_Q = 75.0`
   - `FRONTEND_CORRECTION_MAX_BOOST_DB = 12.0`
   - `FRONTEND_CORRECTION_SOFT_KNEE_DB = 4.0`
   - `FRONTEND_CORRECTION_EDGE_TAPER_FRACTION = 0.10`
   - `FRONTEND_CORRECTION_EDGE_TAPER_SIGMA = 6.0`
   - `FRONTEND_CORRECTION_EDGE_TARGET_DROP_DB = 2.5`
   - `FRONTEND_EDGE_GUARD_FLOOR = 0.35`
   - `IGNORE_SIDEBAND_HZ = 7e6`
3. Reproduce the notebook pipeline behavior in C++ while keeping the hot path on GPU:
   - sideband-ignore calculation and crop
   - optional frontend correction
   - patch-aligned DINO input preparation
   - DINO grouping
   - coherence gate
   - texture and power maps
   - multilevel final fusion and mask cleanup
4. Keep the production model execution path in LibTorch / TorchScript rather than reintroducing Python into the runtime path.
5. Treat notebook parity as required before any performance-driven simplification.

Exit criteria:

- the placeholder detector constants are replaced by notebook-backed config values,
- the operator behavior matches the active notebook pipeline closely enough for fixed-input parity checks, and
- the hot path remains GPU-resident apart from optional debug exports.

### 10.7 Debug timing summary and hotspot visibility

After the notebook-faithful detector path is implemented, add timing summary instrumentation before performance optimization.

Required timing checkpoints:

1. input load / message unpack and shape match
2. frontend correction
3. sideband-ignore crop and patch alignment
4. DINO preprocess and normalization
5. TorchScript forward
6. DINO grouping
7. DINO coherence gate
8. texture scoring
9. power scoring
10. multilevel pipeline fusion
11. final mask cleanup / emit
12. total detector runtime

Reporting requirements:

1. add a config-gated debug timing summary to the operator logs,
2. report at least mean, max, and recent-window totals every `timing_summary_every_n` frames,
3. preserve a per-frame debug mode for one selected frame when parity debugging is needed, and
4. use notebook timing column names where practical so comparisons stay straightforward.

Exit criteria:

- operator logs clearly identify the dominant runtime stages on representative input,
- timing output is lightweight enough to leave enabled for short validation runs, and
- the instrumentation is in place before speed-optimization work starts.

### 10.8 Input contract audit

Validate and document the current detector input contract before broader optimization:

1. Confirm the current spectrogram operator contract is still passthrough-oriented rather than a final image-tensor producer.
2. Confirm the detector currently derives power, resize behavior, and thresholding directly from FFT-domain input.
3. Record this as the accepted short-term contract for bring-up.
4. Defer any true spectrogram-tensor refactor until after stable TorchScript model-forward validation.

Exit criteria:

- the temporary input contract is documented and acknowledged as intentional.

### 10.9 Parity validation against notebook references

Run one fixed captured input through both the notebook and C++ paths.

Notebook references:

1. `noise_detection_dino_experiments2.ipynb`
   - source of truth for local repo loading assumptions, weight selection, and patch-size assumptions
2. `signal_detection_holoscanv1.ipynb`
   - source of truth for the active detector constants, frontend correction, grouping, coherence, and final fusion path
3. `rf_spectrogram_segmentation.ipynb`
   - earlier preprocessing and mask-parity reference

Validation layers:

1. Preprocessing parity:
   - compare the power-domain and resized model input used by the detector path against the notebook reference
2. Output parity:
   - compare final masks using IoU, foreground area fraction, and basic localization consistency

Exit criteria:

- parity metrics meet the agreed threshold and are recorded.

### 10.10 Runtime optimization pass after functionality validation

Use `../Dinov3-RF-Signal-Detection/speed_optimization_todo.md` only after steps 10.5 through 10.9 are complete.

Execution order:

1. use the debug timing summary to identify the dominant detector stages on representative single-channel input,
2. prioritize optimizations in the order suggested by the optimization note:
   - eliminate avoidable GPU-to-CPU transfers,
   - move neighbor search and grouping work to GPU-friendly tensor code,
   - reduce or simplify coherence cost only if output quality remains acceptable,
   - remove residual Python-style loops from the reproduced notebook logic,
3. re-measure after each change before applying the next optimization, and
4. defer quality-risk tradeoffs such as lower `k`, fewer coherence scales, or no-PCA variants until after a parity baseline is recorded.

Exit criteria:

- optimization work is driven by measured bottlenecks rather than guesswork,
- no optimization begins before functional validation is complete, and
- each optimization pass is accompanied by parity and latency comparison notes.

### 10.11 Throughput restoration and scale-up

After strict bring-up and parity pass:

1. Restore the target 2-channel profile.
2. Re-enable representative FFT and detector settings.
3. Tune `emit_stride`, logging, and scheduler settings.
4. Keep spectrogram saving disabled unless it is explicitly needed for sampled debug output.
5. Capture latency, bounded-memory behavior, and sustained-run stability metrics.

Status update (2026-04-12):

- In progress.
- `config_torchscript_performance.yaml` and `run_torchscript_performance_test.sh` were added specifically for this step.
- The performance config keeps the real TorchScript path but disables spectrogram saves, mask saves, detailed detection logs, and timing summaries.
- It reduces DPDK queue `batch_size` and increases `num_simul_batches` to give the ingest path more headroom while keeping GPU RX pools at the known-safe `25000` buffers per channel to avoid GPUDirect BAR1 DMA-map failures.
- The next stage of this step is explicit pipeline isolation: FFT-only, spectrogram-only, and detector-enabled small-batch runs.
- Treat the pre-detector optimization pass as successful only when FFT-only and spectrogram-only sustain the target 2-channel rate with minimal drops and the detector-enabled path remains the clearly dominant throughput limiter.

Exit criteria:

- stable sustained run with bounded memory, controlled inference cadence, and no unintended backend regressions.

### 10.11.1 Bottleneck isolation sequence before detector rewrite

Run the throughput investigation in this order:

1. `config_torchscript_performance_fft_only.yaml`
   - proves whether ingress + CHDR conversion + FFT can sustain the target rate.
2. `config_torchscript_performance_spectrogram_only.yaml`
   - proves whether adding `spectrogramOp` with save disabled still sustains the target rate.
3. `config_torchscript_performance_small_batches.yaml`
   - tests whether smaller CHDR/FFT batch retention materially reduces drops before detector redesign.
4. `config_torchscript_performance.yaml`
   - keeps the full detector path for the final comparison against the isolated stages.

Quantify success as follows:

1. FFT-only and spectrogram-only runs should show near-parity in throughput and only minimal additional RX buffer-drop pressure relative to each other.
2. If FFT-only and spectrogram-only are both healthy while detector-enabled runs collapse, the bottleneck has been isolated to the detector path.
3. If FFT-only already drops heavily, continue working upstream on CHDR batching, queue sizing, and scheduling before changing detector code.
4. If spectrogram-only drops materially more than FFT-only, fix spectrogram-path behavior before changing detector code.

Once this sequence shows the detector as the dominant limiter, detector rewrite work becomes the next active implementation step.

Measured status update (2026-04-12):

- `config_torchscript_performance_fft_only.yaml`
   - initial isolation run accepted packets: `72,528`
   - initial isolation run wire packets: `721,808`
   - initial isolation run `RX out of buffers`: `649,280`
   - initial isolation accepted fraction: about `10.0%`
- `config_torchscript_performance_fft_only.yaml` after CHDR converter fixes
   - accepted packets: `1,010,336`
   - wire packets: `1,012,735`
   - `RX out of buffers`: `2,399`
   - accepted fraction: about `99.8%`
   - per-queue packets delivered to the app: `506,018` on queue 0 and `504,318` on queue 1
   - CHDR converter completed exactly `500,000` packets per channel, with the remaining packets explained by partial final batches at interruption time rather than catastrophic upstream loss
- `config_torchscript_performance_spectrogram_only.yaml`
   - initial isolation run accepted packets: `78,672`
   - initial isolation run wire packets: `1,170,450`
   - initial isolation run `RX out of buffers`: `1,091,778`
   - initial isolation accepted fraction: about `6.7%`
- reported `config_torchscript_performance_spectrogram_only.yaml` rerun after CHDR converter fixes
   - accepted packets: `2,161,980`
   - wire packets: `2,161,980`
   - `RX out of buffers`: `0`
   - accepted fraction: `100%`
   - per-queue packets delivered to the app: `1,080,986` on queue 0 and `1,080,994` on queue 1
- `config_torchscript_performance_small_batches.yaml`
   - accepted packets: `78,672`
   - wire packets: `1,387,636`
   - `RX out of buffers`: `1,308,964`
   - accepted fraction: about `5.7%`

Current conclusion:

1. The CHDR + FFT path is now close to healthy and no longer appears to be the dominant throughput limiter.
2. The large upstream failure observed earlier was caused by CHDR converter correctness bugs rather than an unavoidable FFT-side throughput ceiling.
3. `spectrogramOp` with save disabled also runs at or near line rate, so it is no longer a credible throughput bottleneck in the current path.
4. The dominant remaining throughput cliff is now downstream of spectrogram and is most likely the detector path.
5. Any remaining gap between app-delivered packets and CHDR completed packets in FFT-only is currently consistent with partial final batches when the run is interrupted, not systemic collapse.

Next action after these measurements:

1. Rerun the full detector path and treat the detector hot path as the dominant remaining optimization target.
2. Compare the full detector path directly against the now-healthy FFT-only and spectrogram-only baselines.
3. Focus detector optimization on eliminating host round-trips, blocking stream synchronizations, and unnecessary host/device copies.
4. Preserve the corrected CHDR converter behavior as the new upstream baseline for all further throughput comparisons.

Detector optimization update (2026-04-12):

1. The full detector-path rerun still collapsed badly even though FFT-only and spectrogram-only were healthy:
   - accepted packets: `68,432`
   - `RX out of buffers`: improved from `1,698,317` to `933,051` after the first detector-side copy eliminations, but a later rerun regressed to `1,172,170`
   - per-queue app totals: `39,336` and `29,096`
2. Detector hot-path optimizations implemented so far:
   - the operator no longer copies the full FFT tensor to host just to derive `power_db`
   - `complex_to_power_db_kernel` now generates `power_db` on GPU into a reusable per-channel device buffer
   - the Torch runtime can now consume that device-resident `power_db` buffer directly, skipping the previous host-to-device upload inside the runtime
   - final-mask materialization and host/device mask copies are skipped entirely when `enable_mask_save=false`
   - the detector no longer calls `cudaStreamSynchronize(stream)` before entering LibTorch; instead it passes the upstream CUDA stream into the runtime and uses LibTorch's external-stream guard so Torch work stays ordered on that producer stream
   - quantile threshold lookups in the Torch runtime now use `torch::kthvalue(...)` instead of full `torch::sort(...)` calls to avoid repeated whole-tensor sorts during frontend correction and score thresholding
3. Current conclusion:
   - ingress, FFT, and spectrogram are no longer the dominant bottlenecks
   - the remaining throughput cliff is inside detector-side preprocessing and Torch runtime work on full wideband frames
   - the external-stream change by itself did not materially improve end-to-end throughput, so the next work should focus on reducing full-frame preprocessing cost rather than stream handoff mechanics
4. The next measurement should rerun `config_torchscript_performance.yaml` after rebuild to quantify whether the `kthvalue` quantile change materially reduces the remaining detector bottleneck.

### 10.12 Immediate follow-on tasks

1. Rebuild the containerized app and rerun `config_torchscript_performance.yaml` to measure the `torch::kthvalue(...)` detector optimization.
2. If the full detector path is still badly stalled, instrument or simplify detector-side frontend correction to identify which full-frame operations still dominate runtime.
3. Consider moving more notebook-style preprocessing out of the detector hot path, or reducing its scope, so Torch sees a smaller or cheaper input representation.
4. After detector throughput is materially better, add a downstream sink or schema cleanup pass so detector outputs are validated by more than logs and metadata.

### 10.13 Session restart quick checks

Use the following session checklist after any pause in work:

1. Confirm host-side repo and selected weight still exist.
2. Confirm packaged container-side repo and weight staging still exist.
3. Confirm the exported TorchScript artifact still exists at the configured path.
4. Confirm `applications/usrp_wideband_signal_detection/config.yaml` still points to the staged runtime paths and not placeholders.
5. Rebuild and confirm Torch support is still enabled.
6. Re-run `notebooks/test_radio.ipynb` if basic RF visibility is in doubt.
7. Re-run strict single-channel detector validation before returning to 2-channel throughput tuning.
8. Re-run the dedicated performance config before drawing conclusions about packet-drop rate or throughput headroom.

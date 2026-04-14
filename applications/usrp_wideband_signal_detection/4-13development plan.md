# Coherent Power Detector Development Plan

## Objective

Add a new Holoscan operator named `coherent_power_signal_detector` to the USRP wideband signal detection app and introduce a config-driven detector selector so the app can run either:

- the existing `dinov3_signal_detector`, or
- the new coherent-power detector derived from `Dinov3-RF-Signal-Detection/coherant_power_signal_detection.ipynb`

The coherent-power path must be implemented as a GPU-first C++/CUDA operator, use MatX where it fits naturally, keep data on device as long as possible, and emit a final binary mask built from the merged bounding boxes.

## Status Update

Current implementation status as of 2026-04-13:

- Phase 1 complete: graph wiring, detector selection, build registration, and config surface are in place.
- Phase 2 complete: the new operator package is integrated, takes the live FFT input contract, emits metadata, and can save coherent-power masks.
- Phase 3 complete for functional parity: the operator now performs frontend correction, chunk planning, per-chunk coherent-power scoring, support thresholding, and chunk mask generation.
- Phase 4 complete for mask contract: the operator now groups detections into boxes, merges them, and emits the final binary mask as the rasterized union of merged boxes.
- Phase 5 still open: the current implementation computes power dB on GPU and returns the final mask to device, but the notebook-reference frontend, grouping, and merge stages still run on host and need a second pass to eliminate the full-frame device-to-host copy.

Current backend summary:

- detector selection is live through `pipeline.detector_type`
- DINO remains unchanged when selected
- coherent-power now produces a real box-derived mask instead of the earlier placeholder threshold path
- performance tuning remains a follow-on task, not a completed claim

## Source Of Truth

Primary algorithm source:

- `Dinov3-RF-Signal-Detection/coherant_power_signal_detection.ipynb`
- `Dinov3-RF-Signal-Detection/coherant_power_signal_detection_helpers.py`

Primary app integration points:

- `applications/usrp_wideband_signal_detection/main.cpp`
- `applications/usrp_wideband_signal_detection/config.yaml`
- `applications/usrp_wideband_signal_detection/config_*.yaml`
- `applications/usrp_wideband_signal_detection/CMakeLists.txt`
- `applications/usrp_wideband_signal_detection/README.md`
- `applications/usrp_wideband_signal_detection/metadata.json`

Primary operator reference implementation to mirror structurally:

- `operators/dinov3_signal_detector/dinov3_signal_detector.hpp`
- `operators/dinov3_signal_detector/dinov3_signal_detector.cu`
- `operators/dinov3_signal_detector/CMakeLists.txt`
- `operators/dinov3_signal_detector/metadata.json`

## What The Notebook Actually Does

The notebook implementation is not a simple threshold. Its coherent-power pipeline is:

1. Load or receive a spectrogram-like power view.
2. Compute a global frontend correction across frequency rows.
3. Ignore configured sideband rows near the band edges.
4. Split the frequency axis into overlapping chunks.
5. For each chunk, compute:
   - a coherence-like score from multi-scale structure tensor analysis
   - a local relative power support score
   - a weighted fused score
6. Produce a chunk mask using support and final score quantiles.
7. Group chunk detections into chunk-local bounding boxes.
8. Merge chunk scores back into a global wideband score map.
9. Group the merged detections into final bounding boxes.
10. Convert the final merged bounding boxes into the output binary mask.

The binary mask output requirement should therefore be defined as:

- final output mask = rasterized union of the merged bounding boxes
- not the raw per-pixel support map
- not the pre-grouping chunk mask

That matches the notebook helper flow in `group_signal_mask_regions`, `merge_chunk_results`, and `_boxes_to_mask`.

## High-Level Architecture Change

Current app flow:

- `chdrConverterOp -> fftOp -> spectrogramOp -> dinoV3SignalDetectorOp`

Target app flow:

- `chdrConverterOp -> fftOp -> spectrogramOp -> selectedDetectorOp`

Where `selectedDetectorOp` is created from config:

- `dinov3_signal_detector` when `pipeline.detector_type: "dinov3"`
- `coherent_power_signal_detector` when `pipeline.detector_type: "coherent_power"`

Recommended config shape:

```yaml
pipeline:
  enable_spectrogram: true
  enable_detector: true
  detector_type: "dinov3"  # or "coherent_power"
  log_from_spectrogram: false

dinov3_signal_detector:
  ...

coherent_power_signal_detector:
  ...
```

This is better than overloading the existing DINO section because:

- the two detectors have materially different parameters
- DINO-specific model fields should not pollute the coherent-power config
- the main graph can switch operators cleanly without fragile conditional parsing

## Files To Add

Add a new operator directory:

- `operators/coherent_power_signal_detector/CMakeLists.txt`
- `operators/coherent_power_signal_detector/metadata.json`
- `operators/coherent_power_signal_detector/coherent_power_signal_detector.hpp`
- `operators/coherent_power_signal_detector/coherent_power_signal_detector.cu`

Possible internal helper split if the CUDA file grows too large:

- `operators/coherent_power_signal_detector/coherent_power_kernels.cuh`
- `operators/coherent_power_signal_detector/coherent_power_postprocess.cu`

Only split if needed. Start with one `.cu` implementation file and extract later if compile time or readability becomes a problem.

## Files To Modify

### App graph and build

- `applications/usrp_wideband_signal_detection/main.cpp`
  - include the new operator header
  - parse `pipeline.detector_type`
  - instantiate exactly one detector operator when detection is enabled
  - keep the spectrogram logger and visualization branches unchanged

- `applications/usrp_wideband_signal_detection/CMakeLists.txt`
  - link `holoscan::ops::coherent_power_signal_detector`
  - keep `holoscan::ops::dinov3_signal_detector` linked for the alternate path
  - ensure the new config files are copied into the build tree

### Configs

- `applications/usrp_wideband_signal_detection/config.yaml`
- every `applications/usrp_wideband_signal_detection/config_*.yaml`
  - add `pipeline.detector_type`
  - keep the existing `dinov3_signal_detector` block intact
  - add a new `coherent_power_signal_detector` block with notebook-derived defaults

### Docs and metadata

- `applications/usrp_wideband_signal_detection/README.md`
  - document both detector modes
  - explain which configs exercise coherent-power

- `applications/usrp_wideband_signal_detection/metadata.json`
  - update tags and dependency/operator metadata to include `coherent_power_signal_detector`

- top-level or operator registration CMake files under `holohub-dev/operators`
  - ensure the new operator is discoverable by the build in the same way as `dinov3_signal_detector`

## Operator Design

### Input and output contract

Match the current detector input contract so app wiring stays simple:

- input: `std::tuple<matx::tensor_t<complex<float>, 2>, cudaStream_t>`
- one operator instance handles all channels using metadata channel tags
- no host-side intermediate FFT copies

Recommended output behavior for phase 1:

- no new output port required if the app only needs side effects and metadata parity with the current detector
- write optional debug masks to disk exactly like the DINO operator does
- attach metadata describing thresholds, chunk counts, grouped box counts, and timing

Recommended output behavior for phase 2:

- add an optional mask output port only if downstream visualization or postprocessing requires the binary mask in-graph

### Config surface for the new operator

Start from the notebook config and map it directly into Holoscan parameters:

- `num_channels`
- `input_height`
- `input_width`
- `emit_stride`
- `enable_mask_save`
- `save_every_n_frames`
- `max_masks_per_channel`
- `output_dir`
- `chunk_bandwidth_hz`
- `chunk_overlap_hz`
- `uncalibrated_chunk_fraction`
- `uncalibrated_overlap_fraction`
- `ignore_sideband_percent`
- `ignore_sideband_hz`
- `frontend_row_q`
- `frontend_reference_q`
- `frontend_smooth_sigma`
- `frontend_max_boost_db`
- `coherence_weight`
- `power_weight`
- `coherence_power_support_q`
- `coherence_power_q`
- `min_component_size`
- `grouping_seed_score_q`
- `grouping_bridge_freq_px`
- `grouping_bridge_time_px`
- `grouping_min_component_size`
- `grouping_min_freq_span_px`
- `grouping_min_time_span_px`
- `grouping_min_density`
- `timing_summary_enable`
- `timing_summary_every_n`
- `timing_summary_window`

Do not carry over DINO-only model parameters into this operator.

## GPU-First Reimplementation Strategy

The notebook is NumPy/SciPy based. Reimplementing it efficiently means reproducing the algorithm, not the Python call graph.

### Guiding rules

1. Keep the FFT tensor, derived power maps, score maps, and final mask on device.
2. Reuse the incoming Holoscan CUDA stream. Do not add unconditional stream-wide synchronizations in the hot path.
3. Use MatX for elementwise math, reductions that fit its model, tensor views, slicing, broadcasting, and staged temporary reuse.
4. Use custom CUDA kernels where MatX is not a natural fit, especially for:
   - structure tensor derivative stencils
   - connected-component style grouping
   - bounding-box extraction and rasterization
5. Avoid device-to-host copies except for optional debug saves and compact metadata summaries.

### Stage-by-stage implementation plan

#### Stage 1: Device power representation

Implement a device-resident power-dB view from the complex FFT input.

Approach:

- reuse the current DINO operator pattern of computing a power buffer directly on GPU
- keep one reusable `power_db_device_buffer` per channel
- expose row-major tensor views for downstream kernels without reformatting on host

MatX usage:

- tensor allocation and views
- elementwise power calculation if it benchmarks well enough

Fallback:

- keep a custom CUDA kernel if it remains faster or simpler than the MatX form

#### Stage 2: Frontend correction on GPU

Reimplement `apply_global_frontend_correction` on device.

Approach:

- compute the row floor statistic per frequency row
- smooth the row response with a GPU filter
- compute the reference quantile over valid rows
- produce a per-row boost clipped to `frontend_max_boost_db`
- broadcast-add the row boost into the spectrogram tensor

Efficiency notes:

- quantiles are the expensive part; avoid full sorts when possible
- use selection-based reductions or histogram-based approximations if exact quantiles are too costly
- preserve notebook behavior closely first, then tune with measured tolerances

MatX usage:

- row views and broadcast add
- reduction scaffolding where practical

Likely custom CUDA work:

- approximate or exact row quantile kernel
- 1D smoothing kernel if MatX convolution does not map well

#### Stage 3: Ignore-sideband and valid-row mask

Implement ignore-sideband handling entirely on device, but store the compact row mask metadata on host when needed.

Approach:

- compute applied ignore bins once per frame from metadata resolution/span
- maintain a compact valid-row mask buffer or encode valid row bounds directly
- zero excluded rows early so later stages can skip extra masking branches

#### Stage 4: Chunk planning

Chunk planning can remain host-side initially because it is tiny relative to the frame tensor and depends only on row counts, span, overlap, and valid band geometry.

Approach:

- compute chunk boundaries on host once per geometry change
- cache the plan keyed by `(rows, cols, resolution, span, ignore bins, chunk config)`
- use device views into the corrected tensor for each chunk

This is an acceptable host-side control-plane task. Do not copy chunk data to host.

#### Stage 5: Coherence score on GPU

This is the core of the new operator.

Notebook source:

- `multi_scale_structure_tensor_gate`
- `_structure_tensor_components`
- `residual_background_spectrogram`

Approach:

- compute a local background estimate on GPU
- derive residual energy map
- compute frequency and time gradients at multiple scales
- compute structure tensor terms and coherence score
- normalize into a `coherence_px` map

Implementation note:

- this stage should be a fused or semi-fused CUDA pipeline, not a collection of tiny host-launched transforms with intermediate copies
- prefer shared-memory tiles for neighborhood filters and derivative stencils

MatX usage:

- buffer views and simple elementwise transforms

Likely custom CUDA work:

- local mean filters
- Gaussian-like smoothing or separable filters
- gradient and tensor kernels

#### Stage 6: Local relative power support on GPU

Notebook source:

- `_local_relative_power_support_map`

Approach:

- convert corrected dB back to linear power only where necessary
- compute a floor statistic over valid rows
- build a local baseline with sliding-window filtering
- compute local support and normalize to `power_px`

Efficiency notes:

- this can likely use separable box filters or prefix-sum based sliding windows
- avoid repeated dB/linear conversions if a reusable linear-power view helps

#### Stage 7: Score fusion and thresholding on GPU

Notebook source:

- `detect_chunk_coherent_power`

Approach:

- fuse `coherence_px` and `power_px`
- compute support and final quantile thresholds over only valid pixels
- generate chunk support mask and chunk mask
- apply a small morphological cleanup

Efficiency notes:

- use kth-value or histogram selection, not full sorts
- keep thresholds scalar and copy only those scalars to host metadata if needed

#### Stage 8: Grouping and bounding boxes

Notebook source:

- `group_signal_mask_regions`
- `build_grouped_detection_regions`

This is the least MatX-like stage and needs careful design.

Recommended implementation:

- do morphology and hole filling on GPU
- run connected-component labeling on GPU
- compute per-component bounding boxes, area, density, and peak score on GPU
- compact only the small component summary table to host if the GPU implementation of final filtering becomes too complex

Recommended practical compromise for the first implementation:

- keep the full-resolution masks and score maps on device
- copy only a compact labeled-component summary or candidate-box table to host if needed
- never copy the full frame back to host just to group regions

If a clean GPU connected-components path is not already available in Holohub, consider:

- a custom CUDA union-find implementation
- NPP if it cleanly covers the operation and dependency policy permits it
- a temporary hybrid path that copies only a packed binary chunk mask to host for box extraction during early bring-up

The hybrid path is acceptable only as a short-lived milestone and should be explicitly removed once GPU grouping is stable.

#### Stage 9: Merge chunk results into global boxes and final mask

Notebook source:

- `merge_chunk_results`
- `_merge_projected_subsection_boxes`
- `_boxes_to_mask`

Approach:

- merge per-chunk score/support maps into global device buffers with weighted overlap blending
- compute a global grouped box set
- rasterize final merged boxes into the output binary mask on device

Output contract:

- final binary mask must be generated from the merged boxes
- the final rasterization should be a simple GPU box-fill kernel over the output tensor

## Proposed Implementation Phases

### Phase 1: Graph and config plumbing

Status:

- completed

Deliverables:

- detector selector in `main.cpp`
- new coherent-power config block in all app YAMLs
- CMake wiring for the new operator

Success criteria:

- app builds with both operators available
- config chooses one detector path without recompilation

### Phase 2: Minimal coherent-power operator skeleton

Status:

- completed

Deliverables:

- new operator compiles and receives the same FFT input contract as DINO
- per-channel frame accounting and mask-save plumbing copied from the DINO operator style
- initial placeholder path that emits a device mask for smoke testing

Success criteria:

- app runs with `detector_type: coherent_power`
- operator emits a saved mask and metadata without breaking the graph

### Phase 3: Frontend correction and chunked scoring parity

Status:

- completed as a functional reference implementation
- still needs GPU migration and scratch-buffer tuning for the final performance target

Deliverables:

- frontend correction on GPU
- chunk planner and per-chunk scoring on GPU
- chunk support and chunk mask generation

Success criteria:

- coherent-power score maps and chunk masks qualitatively match notebook debug outputs on saved spectrograms

### Phase 4: Bounding-box grouping and final mask contract

Status:

- completed as a functional reference implementation
- final mask semantics now match the intended box-derived contract

Deliverables:

- chunk-local grouping
- global merge grouping
- rasterized final binary mask from merged boxes

Success criteria:

- output mask semantics match notebook box-derived mask behavior
- grouped box counts and placements are close to notebook outputs on reference data

### Phase 5: Performance tuning

Status:

- not started beyond initial buffer reuse and GPU power-dB staging

Deliverables:

- remove avoidable synchronizations
- reuse scratch buffers across frames
- benchmark MatX vs custom kernels stage by stage
- keep host traffic restricted to optional saves and compact metadata

Success criteria:

- coherent-power path is materially faster than notebook Python
- no full-frame device-to-host copies in steady-state detection mode

## Validation Plan

### Functional validation

Use the notebook inputs as golden references:

- `/tmp/usrp_spectrograms/*.pgm`
- tensor snapshots under `/tmp/usrp_spectrograms/tensors`
- any saved masks already produced for comparison

Validation artifacts to compare:

- frontend corrected row-floor behavior
- chunk boundaries
- support thresholds and score thresholds
- number and placement of grouped boxes
- final binary mask overlap against notebook-generated mask

### Performance validation

Add a coherent-power performance config modeled after the existing performance configs.

Recommended new configs:

- `config_coherent_power_debug.yaml`
- `config_coherent_power_performance.yaml`
- `config_coherent_power_validation.yaml`

Track at least:

- input stage time
- frontend correction time
- chunk scoring time
- grouping time
- mask rasterization time
- total per-frame detector time

### Regression validation

Ensure the DINO path still works unchanged when `pipeline.detector_type: "dinov3"`.

That means:

- no behavioral regression in existing DINO configs
- no DINO-specific fields moved or renamed in a breaking way
- no forced Torch dependency in the coherent-power path

## Recommended Order Of Work

1. Add the new operator directory and build metadata.
2. Wire `pipeline.detector_type` into `main.cpp`.
3. Add the coherent-power config block to all app YAMLs.
4. Clone the DINO operator skeleton only for lifecycle, metadata, timing, and mask-save mechanics.
5. Replace the DINO-specific compute path with coherent-power stages from the notebook.
6. Get notebook parity on saved offline spectrogram inputs before live-stream tuning.
7. Optimize stage by stage, keeping the final box-to-mask contract intact.

## Key Design Decisions To Preserve

- The detector selector should happen at the application graph level, not inside the DINO operator.
- The coherent-power operator should be independent of Torch and DINO model files.
- Chunk planning may stay host-side initially, but chunk data and score computation should remain on GPU.
- The final delivered mask should be the rasterized union of merged bounding boxes.
- MatX should be used wherever it improves clarity and keeps computation on GPU, but custom CUDA kernels should be preferred for neighborhood-heavy or component-labeling stages where MatX is not the right tool.

## Immediate References For Implementation

Start by reading these files in this order:

1. `Dinov3-RF-Signal-Detection/coherant_power_signal_detection_helpers.py`
2. `operators/dinov3_signal_detector/dinov3_signal_detector.hpp`
3. `operators/dinov3_signal_detector/dinov3_signal_detector.cu`
4. `applications/usrp_wideband_signal_detection/main.cpp`
5. `applications/usrp_wideband_signal_detection/config.yaml`
6. `applications/usrp_wideband_signal_detection/CMakeLists.txt`

That sequence gives the algorithm first, then the operator pattern, then the app integration points.
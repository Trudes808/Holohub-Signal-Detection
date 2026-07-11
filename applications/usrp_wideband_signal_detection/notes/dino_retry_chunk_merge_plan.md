# DINO Retry Chunk-Merge Port Plan

## Objective

Port the retry-hybrid DINO subsection behavior from the Python validation flow into the C++ offline detector, while preserving the wideband subsection planning and merge behavior from the coherent-power pipeline.

The target behavior is:

1. Apply the wideband frontend correction once on the full spectrogram.
2. Split the corrected spectrogram into overlapping frequency subsections using the coherent-power chunk planner.
3. Run the retry-hybrid DINO logic independently on each subsection.
4. Project subsection-local detections back into the full spectrogram frame.
5. Merge subsection detections into one final wideband mask using the coherent-power merge and grouping rules.

This plan is written so work can continue incrementally even if the chat or runtime session crashes.

## Source Of Truth

The implementation should be anchored to these existing references.

Wideband chunking and merge behavior:

- `/home/sat3737/holoscan_demo_workspace/holohub-dev/notebooks/coherant_power_signal_detection_helpers.py`
- `build_frequency_chunks(...)`
- `merge_chunk_results(...)`
- `_merge_projected_subsection_boxes(...)`
- `group_signal_mask_regions(...)`

Per-subsection retry-hybrid DINO behavior:

- `/home/sat3737/holoscan_demo_workspace/Dinov3-RF-Signal-Detection/signal_detection_holoscan_retry_dino.ipynb`
- `/home/sat3737/holoscan_demo_workspace/holohub-dev/notebooks/torchscript_dino_signal_detector_validation.ipynb`
- `run_subsection_dino_texture_experiment(...)`
- `build_python_retry_hybrid_products(...)`
- `build_retry_frequency_support_mask(...)`

Current offline C++ implementation to evolve:

- `/home/sat3737/holoscan_demo_workspace/holohub-dev/applications/usrp_wideband_signal_detection/offline_dino_validator.cpp`
- `/home/sat3737/holoscan_demo_workspace/holohub-dev/operators/dinov3_signal_detector/dinov3_torch_runtime.cpp`
- `/home/sat3737/holoscan_demo_workspace/holohub-dev/operators/dinov3_signal_detector/dinov3_torch_runtime.hpp`

## Current Gap

The current offline validator does not implement wideband subsection planning or merging.

Today it does this:

1. Load the full tensor snapshot.
2. Compute full-frame `power_db` and frontend-corrected `corrected_db`.
3. Run one full-frame DINO inference through `DinoTorchRuntime` at `input_height x input_width`.
4. Build one full-frame coherence gate from the resized corrected spectrogram.
5. Form `hybrid_contrib = normalize(dino_score) * normalize(coherence_gate)`.
6. Run one full-frame residual-veto postprocess.
7. Write one final mask.

That is not equivalent to the Python retry behavior the user wants, because the Python target flow is chunked and merged:

1. Chunk-local DINO logic runs on subsection views, not on one global resized frame.
2. Each subsection has its own valid-row mask, coherence gate, texture policy, hybrid mask, and retry-support mask.
3. Final detections are produced after projecting subsection results back into the global frame and merging them.

## Target C++ Reference Behavior

The first C++ target should be a notebook-faithful reference path inside `offline_dino_validator.cpp`.

Reference behavior requirements:

1. Full-frame frontend correction must run once before chunking.
2. Chunk planning must match the coherent-power helper semantics for:
   - chunk width in frequency bins
   - overlap width in frequency bins
   - sideband ignore behavior
   - valid-row clipping
3. Each chunk must run the retry-hybrid DINO logic independently.
4. Each chunk must emit enough metadata to be projected and merged in global coordinates.
5. The final global mask must come from subsection merge behavior, not from naive OR over subsection masks.
6. The validator must keep exporting intermediate artifacts so parity can be measured stage by stage.

## Recommended Work Split

Do this in two layers.

### Layer 1: Validator-Faithful Reference Port

Goal: produce a correct offline C++ implementation that matches the Python reference behavior closely enough to debug differences.

Characteristics:

- correctness first
- may use host-side bookkeeping for chunk plans, grouping, and merge metadata
- reuses the existing Torch runtime for per-subsection DINO inference
- exports intermediate arrays and JSON summaries for parity analysis

### Layer 2: Efficient Runtime Port

Goal: once validator parity is stable, move the same chunked behavior into the operator/runtime path with fewer copies and better GPU residency.

Characteristics:

- keep corrected spectrogram and subsection views on device where practical
- reuse buffers across chunks
- avoid repeated allocations and model reloads
- preserve the same algorithmic steps as the reference path

Do not start with the efficient path. First freeze the validator-faithful reference behavior.

## Phase Plan

## Phase 0: Freeze The Python Reference

Before more C++ work, freeze one unambiguous Python reference implementation for the desired subsection behavior.

Tasks:

1. Promote the last-cell retry logic into stable helper functions if any steps still live only in notebook cells.
2. Define one Python reference entry point with this signature in spirit:
   - `run_retry_dino_chunk_pipeline(input_record, cfg, dino_cfg) -> pipeline_result`
3. Ensure it returns:
   - `chunk_plan`
   - `chunk_results`
   - per-chunk retry-hybrid intermediates
   - projected subsection masks
   - merged score or merged grouping inputs
   - `merged_boxes`
   - `merged_mask`
4. Save one golden artifact set for at least one known tensor snapshot.

Exit criteria:

- there is one stable Python reference pipeline to compare against
- notebook parity no longer depends on manually rerunning ad hoc cells

## Phase 1: Add Chunk Planning To The Offline Validator

Extend `offline_dino_validator.cpp` so it plans subsection slices from the full corrected spectrogram before any DINO postprocess is run.

Tasks:

1. Add a `ChunkPlanEntry` struct carrying at least:
   - `chunk_index`
   - `row_start`
   - `row_stop`
   - `freq_start_hz` if available
   - `freq_stop_hz` if available
2. Port or reimplement coherent-power chunk planning semantics in C++.
3. Build the plan from the original frequency axis resolution, not from the final DINO output height.
4. Derive a chunk-local valid-row mask from the global valid-row mask.

Important design rule:

The chunk plan must live in source spectrogram coordinates. DINO resizing happens inside each chunk evaluation, not before chunk planning.

Exit criteria:

- validator prints chunk count and row spans
- chunk plan matches the Python reference on the same input

## Phase 2: Add Per-Chunk DINO Retry Execution

Replace the current whole-frame retry-hybrid path with per-subsection execution.

Per chunk, the C++ flow should be:

1. Extract `corrected_chunk = corrected_db[row_start:row_stop, :]` in source coordinates.
2. Run DINO inference for that chunk through `DinoTorchRuntime`.
3. Resize DINO outputs back to the chunk pixel grid if the runtime operates on fixed `input_height x input_width`.
4. Build chunk-local structure-tensor coherence gate from `corrected_chunk`.
5. Reproduce the Python retry hybrid products:
   - normalized DINO score
   - coherence region threshold and mask
   - texture score and top-texture mask
   - texture passthrough policy
   - hybrid DINO mask
   - texture union mask
   - hybrid mask
   - hybrid DINO contribution
6. Reproduce the Python retry frequency-support mask:
   - `base_norm`
   - `envelope_map`
   - `residual_penalty`
   - `freq_curvature_penalty`
   - `keep_freq`
   - `keep_res`
   - `residual_veto_gate`
   - `combined_score`
   - `seed_mask`
   - `final_mask`

Recommended C++ data structure:

- `ChunkRetryResult`
  - chunk metadata
  - local valid mask
  - local DINO score map
  - local coherence gate
  - local hybrid contribution
  - local support or final score map
  - local final mask
  - local grouped mask
  - local grouped boxes
  - thresholds used
  - timing fields

Exit criteria:

- validator can emit one `ChunkRetryResult` per subsection
- local outputs match the Python reference for a chosen subsection within tolerance

## Phase 3: Add Chunk-Local Grouping

Each subsection needs local grouping before global merge so that the merged wideband path can project chunk-local boxes back into the full frame.

Tasks:

1. Port or reimplement `group_signal_mask_regions(...)` semantics in C++.
2. Group each chunk-local final mask using the chunk-local score map and valid-row mask.
3. Preserve box fields needed by the Python merge logic, including:
   - freq and time extents
   - density
   - filled area
   - score mean
   - score peak
   - split-role metadata if implemented

Recommendation:

For the first milestone, a simpler grouping path is acceptable if it preserves the same final grouped mask and projected boxes for the validation cases. Split-role bookkeeping can be added after base parity is established.

Exit criteria:

- each subsection emits grouped boxes and grouped mask
- projected local boxes can be compared against the Python subsection output

## Phase 4: Add Global Projection And Merge

After chunk-local results exist, add the coherent-power global merge behavior.

Tasks:

1. Project subsection-local grouped masks into the global frame.
2. Project subsection-local grouped boxes into global coordinates.
3. Blend overlapping subsections with the same chunk weighting policy used by the coherent-power pipeline.
4. Build global merged score support arrays as needed for the final merge stage.
5. Merge projected subsection boxes using the coherent-power merge logic.
6. Build the final global binary mask from merged grouped regions or merged boxes.

Do not simplify this into a raw OR unless the Python reference proves that OR is equivalent for the target cases.

Exit criteria:

- validator writes one merged global mask and one merged box set
- final mask shape matches the source wideband spectrogram grid or the explicitly chosen output contract

## Phase 5: Define The Output Contract Clearly

We need one explicit answer for the output space.

Recommended contract:

1. The reference validator should operate in source spectrogram coordinates for chunk planning and merge.
2. Per-subsection DINO inference may internally resize to the model input size.
3. DINO outputs must be resized back to chunk-local source coordinates before chunk-local retry logic and before global merge.
4. The final merged mask should therefore be expressed in the same coordinates as the corrected wideband spectrogram.

If a second, reduced-resolution output is still needed for compatibility, export it as a derived artifact, not as the primary reference mask.

## Phase 6: Promote Into Shared Runtime Or Operator Code

Only after the validator path is stable should we move the chunked retry pipeline into reusable C++ runtime code.

Recommended direction:

1. Extract shared helpers from `offline_dino_validator.cpp` into reusable files.
2. Keep the offline validator as the reference harness.
3. Reuse the same per-chunk retry implementation from the live operator path where possible.
4. Keep the live path configurable so the old global path and the new chunked retry path can be compared during rollout.

## Required Code Changes

## A. `offline_dino_validator.cpp`

This file should become the first reference implementation.

Changes needed:

1. Replace the single whole-frame `run_residual_veto_hybrid(...)` call with a chunked pipeline.
2. Add chunk planner helpers and chunk result structs.
3. Add per-chunk DINO runtime invocation.
4. Add chunk-local retry-hybrid reconstruction.
5. Add chunk-local grouping.
6. Add projection and merge.
7. Expand JSON summary output with chunk and merge metadata.

## B. `dinov3_torch_runtime.hpp` and `dinov3_torch_runtime.cpp`

These files likely need interface cleanup.

Important issue:

- current `DinoTorchRuntimeResult.final_mask` is used by the offline validator as a DINO score map, not as a postprocessed binary mask

Recommended cleanup:

1. Add explicit result fields for score maps versus binary masks.
2. Stop depending on the misleading `final_mask` name for raw DINO score output.
3. Make it easy to run the runtime repeatedly on chunk-local source windows without repeated setup cost.

## C. Validation notebook support

The notebook side should gain a stable apples-to-apples comparison harness.

Needed artifacts:

1. chunk plan JSON or table
2. per-chunk thresholds
3. per-chunk grouped boxes
4. projected global boxes
5. merged global mask
6. optional per-stage `.npy` outputs for one debug chunk

## Efficiency Plan For The Final C++ Implementation

The efficient scheme should preserve the validator-faithful algorithm while reducing unnecessary copies.

Recommended scheme:

1. Keep full-frame `power_db` and `corrected_db` in device memory after frontend correction.
2. Build the chunk plan on host once per frame. The plan is small.
3. For each chunk, pass a view or compact staging buffer for the chunk into the DINO runtime.
4. Reuse one set of device buffers for chunk-local resized inputs and outputs.
5. Reuse the loaded DINO model across all chunks.
6. Resize DINO score outputs back into chunk-local source coordinates immediately after inference.
7. Keep chunk-local coherence gate and retry-support operations on device where practical.
8. Move only compact metadata back to host for grouping if grouping remains host-side initially.
9. Once parity is stable, consider moving grouping and merge to GPU if runtime throughput requires it.

Practical note:

Chunk planning and box bookkeeping are not the expensive parts. DINO inference, resizing, and repeated temporary allocation are the likely hotspots. Optimize those first.

## Validation And Parity Checklist

Each phase should be closed with explicit comparisons against the Python reference.

### Per-subsection parity

For one chosen subsection, compare:

1. corrected chunk after resize mapping
2. DINO score map
3. coherence gate
4. texture score map
5. hybrid DINO contribution
6. `keep_freq`
7. `keep_res`
8. `combined_score`
9. seed mask
10. final chunk mask

### Global parity

Compare:

1. chunk count and row spans
2. projected chunk boxes
3. merged box count
4. merged mask
5. final mask pixel agreement and IoU

### Minimum acceptance target for the reference path

1. chunk plan identical to Python
2. final merged mask visually and numerically aligned with Python on frozen captures
3. any remaining differences localized to known floating-point or interpolation details

## Crash-Resistant Working Order

Follow this order in future requests so work can resume cleanly after interruption.

1. Freeze the Python reference helpers.
2. Add chunk planner structs and summary output in C++.
3. Implement one-chunk C++ retry evaluation and verify one subsection.
4. Loop over all chunks and write per-chunk artifacts.
5. Add chunk-local grouping.
6. Add global projection and merge.
7. Add final parity notebook cells and summary report.
8. Only then optimize for runtime efficiency.

## Immediate Next Tasks

These are the first concrete implementation tasks to take after this plan.

1. Decide and document the primary output coordinate system.
   - recommended answer: source spectrogram coordinates
2. Freeze the Python reference as helper functions instead of notebook-only code.
3. Add chunk-plan generation to `offline_dino_validator.cpp` and export it in the summary.
4. Refactor the current whole-frame retry path into a reusable `run_retry_chunk(...)` helper.
5. Validate one subsection end to end against the Python notebook before adding merge.

## Definition Of Done

This task is done when all of the following are true.

1. The offline C++ DINO validator subdivides the wideband spectrogram into overlapping frequency chunks.
2. Each chunk runs the retry-hybrid DINO subsection behavior that matches the Python reference.
3. Chunk-local detections are grouped, projected, and merged back into one global mask.
4. The final global mask matches the Python chunked retry reference on frozen validation captures within agreed tolerance.
5. The implementation path is structured so the same logic can later be moved into the live operator without changing the algorithm.
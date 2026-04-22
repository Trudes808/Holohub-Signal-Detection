# DINO Retry Chunk-Merge Port Plan

## Progress Update 2026-04-21T10:46:38-05:00

- The first uniform-chunk sideband-trim search was wrong for calibrated frequency axes. It searched for any fully uniform plan using the existing Hz-threshold chunk builder, and for the current 20,480-row capture that path only becomes perfectly uniform after trimming almost the entire band away.
- The calibrated uniform-chunk logic now derives chunk rows and overlap rows directly from the configured bandwidth and overlap in bins, then increases ignored sideband bins only enough to make the retained valid rows tile evenly into full-size chunks.
- For the current single-channel validation capture, that means the planner now lands near the expected operating point instead of collapsing to a tiny residual band: `ignore_bins_per_side` moves from `287` to `512`, retained rows become `19456`, chunk count becomes `25`, and every planned chunk is `1024` rows tall.
- The same calibrated uniform-chunk rule has been mirrored into the shared Python retry helper so the notebook and comparison scripts rebuild the same plan as the offline performance validator.

## Progress Update 2026-04-21T11:11:43-05:00

- The next runtime experiment is now wired into the offline performance validator: instead of feeding DINO a chunk-sized patch-aligned input, the validator now downsamples the full chunk to the fixed detector grid before the Torch forward path.
- For the current validation config, that means the DINO model input is driven at the existing detector-grid scale (`input_height x input_width`, currently `256x512`) instead of the prior near-source chunk scale (`1024x1024` for most uniform chunks).
- This experiment changes the runtime contract for the selected debug chunk as well, so the chunk-debug artifact bundle now advertises a dedicated contract string and the notebook helper reconstructs the same fixed-grid Python runtime input before comparing pre-model grayscale and runtime raw or grouped DINO maps.
- The purpose of this pass is to measure how much of the full-run bottleneck was truly tied to large DINO forward shapes, then compare the resulting validation drift against the current chunk-level notebook and comparison report before deciding whether a capped-input path is viable for the non-debug full-run route.

## Progress Update 2026-04-21T10:12:34-05:00

- The immediate focus is back on the offline validator, not the live Holoscan operator. The current goal is to push down full all-chunk runtime until the remaining algorithmic risks are clearer, then port the proven path.
- The next optimization sequence is now explicit:
   1. batch only naturally equal-sized planned spectrogram subsections so batching is efficient without rewriting the chunk plan,
   2. batch non-debug chunk Torch inference through the shared runtime while keeping the selected debug chunk on the single-item parity path,
   3. replace the current `group_mask_regions` hot path with faster primitives rather than carrying the older brute-force morphology and flood-fill style cost profile.
- The performance validator now keeps the original chunk plan intact, batches only non-debug chunks that already share the same row count, and keeps the debug chunk on the existing artifact-rich path so notebook parity remains apples-to-apples.
- The grouping hot path has also been reworked around faster separable binary morphology and a union-find connected-components pass so the next full-run measurement can show whether grouping is still a dominant CPU stage after those replacements.
- TensorRT remains a high-upside option for the Torch runtime bottleneck, but it is explicitly deferred for now. Keep it on the plan as a later route if batched LibTorch forward still leaves `chunk_torch_runtime` dominant after the current offline-validator changes.
- Hold the more aggressive fused hybrid-support plus grouping rewrite for a later pass. Keep it noted as a follow-on option if the current batch plus grouping changes are not enough.
- Hold chunk-count reduction and overlap redesign for now as well. Keep it in notes as a higher-risk recall tradeoff once the batched uniform-chunk baseline is measured.

## Notes 2026-04-21T10:21:47-05:00

Historical comparison against commit `36dd9f67` suggests the earlier fast Torch behavior was not just a "better Torch version" effect. The largest differences appear to come from how Torch was being driven and how much surrounding work was attached to each call.

Most important observations:

- The historical fast path appears to have run Torch at the detector grid or another much smaller fixed working size, while the current offline validator feeds Torch chunk-sized tensors whose `dst_rows` and `dst_cols` are derived from each chunk's source dimensions.
- The historical fast path kept the score map on device in the operator fast backend (`return_final_mask=false`, `return_final_mask_device=true`), while the current offline validator requests CPU score-map materialization (`return_final_mask=true`, `return_final_mask_device=false`) so downstream projection and validation logic can consume host vectors.
- The older fast path used a much lighter pre-model image construction step based on direct quantile normalization of the resized tensor, while the current runtime uses the heavier `signal_agnostic_dino_gray(...)` preprocessing path before model execution.
- The older path also used cheaper quantile selection behavior (`kthvalue` style semantics). That cheaper selection path has now been reintroduced in the current performance branch, so quantile work is no longer the leading runtime delta in the latest measurements; the dominant remaining costs have shifted to batched Torch forward, hybrid-support or residual-veto postprocess, and chunk coherence generation.
- The current offline validator is also solving a more validation-faithful problem than the older fast live path: chunk-local parity surfaces, chunk projection, chunk hybrid support, and CPU grouping all remain in play, so not all of the observed runtime delta belongs to Torch forward itself.

Likely causes to try next:

1. Test whether Torch can be run on a smaller fixed detector-grid input again for the offline validator path, or at least on a smaller capped chunk working size, without breaking the needed validation contract.
2. Add an experiment that keeps the runtime score map on device for the non-debug full-run path and delays or avoids CPU materialization until a later stage that truly requires host access.
3. Add a simplified pre-model image construction experiment that temporarily replaces `signal_agnostic_dino_gray(...)` with the older lightweight normalized-grayscale path, then compare both runtime and validation drift.
4. Reintroduce a cheaper quantile path for the full-run non-debug route, such as `kthvalue`-style thresholding, and measure whether the recovered speed is material relative to any parity loss.
5. Separate measured Torch cost from measured surrounding runtime cost more explicitly by breaking `chunk_torch_runtime` into at least `model_prep`, `torch_forward`, `dino_score`, and `score_to_cpu` summaries in the offline validator reports.
6. If the smaller-input and GPU-resident-score experiments recover most of the old performance, prioritize those before reopening the deferred TensorRT path.

## Progress Update 2026-04-21T07:41:24-05:00

- Added first-pass stage profiling to `offline_dino_validator_performance.cpp` for both run-level stages and per-chunk stages.
- The performance branch now records elapsed milliseconds, RSS snapshots, HWM snapshots, RSS or HWM deltas, and stage-local estimated component bytes for the major pipeline steps.
- Profiling currently covers config load, tensor load, frontend correction, chunk planning, GPU uploads, Torch runtime calls, score remap, coherence generation, hybrid postprocess, per-chunk grouping, global merge, and artifact serialization.
- The performance validator now writes a machine-readable stage profile artifact at `offline_stage_profile.json` and exposes that path in the main summary JSON.
- Verbose runs now print an aggregated hotspot table so the next optimization pass can target the worst total-time stages instead of guessing.
- The generated build tree has not yet been reconfigured, so the current `build.ninja` still does not list `offline_dino_validator_performance`; the next rebuild needs to force a CMake regenerate before the new target can be executed inside the container.

## Progress Update 2026-04-21T07:26:15-05:00

- Created an isolated performance branch for the offline validator so timing and memory instrumentation can move faster without destabilizing the current parity/debug path.
- Added the new performance source file `offline_dino_validator_performance.cpp` and a matching launcher `run_offline_dino_validator_performance.sh` so the optimized path can be built and run as a separate binary.
- The immediate branch objective is now three-stage: first instrument the major pipeline stages with timing and memory accounting, then optimize the proven hotspots while preserving debug artifact export, and only then expand execution from the current selected debug chunk to every planned subsection.
- The performance branch must keep the existing artifact contract alive after every optimization pass so each faster implementation can still be checked against the notebook and the current reference validator before it becomes the new default.

## Progress Update 2026-04-20T20:08:53-05:00

- Live-operator porting is now active. The immediate rollout target is the true DINO operator under the single-channel performance configuration before scaling back up to the 2-channel 500 MSps case.
- The first live correction is to remove the stale `fast_gpu` mask-generation split. Live runs now need to follow the same validated residual-veto hybrid contract as offline validation, while still using the existing GPU-resident runtime score map and GPU hybrid helper wherever available.
- `config_torchscript_performance_single_channel.yaml` is being aligned to `backend_mode: reference` so performance bring-up starts from the same backend contract used for validation.
- The live operator now treats any non-`reference` backend request as deprecated and falls back to the validated reference path instead of preserving a separate prototype-mode branch in operator logic.
- The active two-channel and single-channel performance configs now both advertise `backend_mode: reference` so live bring-up starts from the same mask-generation contract used by offline validation.
- The live DINO operator now has a first reference-style chunked path wired in: it copies the full-frame `power_db` into the validator-style source-coordinate chunk planner, runs per-chunk Torch runtime calls on aligned source chunks, reconstructs source-grid coherence and residual-veto masks chunk by chunk, and regroups the projected chunk outputs back into one merged global mask.
- The live reference path now saves and reports the DINO mask in source spectrogram coordinates instead of treating the old reduced detector grid as the primary output contract.
- The old live whole-frame coherence-resize plus one-pass hybrid tail has been removed from the active reference backend path. The current live `reference` backend is now structurally much closer to the offline validator than the earlier single-frame operator path.
- The preferred iteration loop for this phase is code edit -> `rebuild_demo_container_app.sh` -> rerun via the demo container run script with the single-channel performance config -> inspect live behavior and timing before expanding scope.

## Progress Update 2026-04-20T14:42:37-05:00

- Exact pre-model grayscale parity is now in place in the validator notebook path. The C++ runtime dump and the Python reference pre-model input now agree when compared against the Python-defined crop and patch truncation semantics.
- The notebook helper was corrected so the exact input panel is Python-first. It now compares the C++ runtime dump against the Python-expected pre-model grayscale instead of forcing Python to mirror the C++ artifact shape.
- Offline validator parity remains the active milestone. Live operator cleanup stays deferred until the offline validator and notebook comparison agree chunk by chunk.
- New root cause identified in the offline validator chunk path: each planned chunk was still reapplying `ignore_sideband_hz` inside `DinoTorchRuntime`, shrinking the exact pre-model DINO crop to an inner band even though the chunk planner had already excluded the global sidebands.
- That extra runtime crop was also shifting the raw and grouped DINO projections onto the wrong row span for validator debug artifacts, which explains why the coherence gate looked aligned to one region while the grayscale and raw DINO comparisons looked aligned to a narrower inner slice.
- The current patch disables chunk-local sideband re-cropping for the offline validator path and updates the notebook helper to compare against that same no-extra-crop runtime contract.
- Latest notebook rerun still showed `runtime_input_gray_rows = 448` and no `chunk_dino_score_raw.npy` or `chunk_patch_features.npy` in the artifact bundle, which means the validator command is still executing a stale binary or stale mounted build tree rather than the patched offline validator.
- The notebook helper now hard-fails on that stale artifact signature so we stop comparing legacy C++ outputs against the current Python runtime-crop path as if they were apples to apples.
- New regression found while validating the patched launcher: the parity instrumentation added `runtime_input_gray`, `raw_dino_score_map`, grouped `dino_score_map`, `coherence_gate`, `hybrid_contrib`, and especially `patch_features` to every `ChunkRetryResult`, so the validator could be SIGKILLed by OOM before writing fresh artifacts.
- Current fix: only the selected debug chunk retains the heavyweight debug arrays; non-debug chunks now keep only the data needed for grouping and global merge.
- Additional memory fix: the expensive patch-feature grouped-score path now runs only on the selected debug chunk. Non-debug chunks currently use the cheaper raw-score fallback so the offline validator can stay alive while we debug apples-to-apples parity on the chunk the notebook inspects.
- Docker and cgroup inspection now rule out a simple container memory ceiling for the current 137 failure: `HostConfig.Memory=0`, `HostConfig.MemorySwap=0`, `memory.max=max`, and idle `memory.current` is only about 21 MiB.
- The leading remaining hypothesis is validator-local peak allocation inside the selected debug chunk's dense patch-feature grouping path, especially the `full_aff`, `local_aff`, `trans`, `trans2`, and `trans3` square matrices plus Torch or BLAS workspaces.
- To make the next rerun decisive, the validator now emits verbose RSS and HWM probes around grouped patch-feature stages so we can see whether the kill happens before PCA, after the dense affinity build, or during the multi-hop transition scoring path.
- The latest verbose rerun isolated the actual killer: the selected debug chunk grouping peaks around 2.1 GiB RSS, but the separate offline full-frame grouped patch path reaches tens of GiB (`patch_count=79616`, dense matrix estimate about 24 GiB, HWM above 74 GiB) and then gets SIGKILLed.
- Subsection validation does not need that full-frame grouped patch path. The validator now supports `--subsection-only-validation`, which keeps the global correction, chunk planning, debug chunk execution, and chunk-to-full-frame merge outputs, but skips full-frame patch-feature export and skips `offline_full_frame_grouped_patch` entirely.
- `validate_offline_dino_subsection.sh` now enables subsection-only validation automatically so the parity workflow can get back to comparing the selected C++ subsection against Python instead of building an unnecessary full-frame grouped patch surface.
- After this patch, the next offline-validator check is whether the grouped patch-feature DINO surface now diverges from the raw feature-energy proxy in the same way as Python. If it still does not, the remaining work is inside the grouped-score port rather than the crop contract.
- Added a new `filter_detection_mask` config switch for the post-final-mask boxing stage. When true, the existing bridging and component filtering path remains active. When false, the detector skips that filtering and emits simple rectangular boxes around each connected mask region so notebook parity can isolate grouping effects directly.
- Offline validation has now converged far enough that the next active milestone is porting the validated chunked/offline behavior into the live running operator instead of continuing to reshape the notebook reference.

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

## Performance Validator Branch Plan

Use `offline_dino_validator_performance.cpp` as the isolated branch for timing, memory, and optimization work. Keep `offline_dino_validator.cpp` as the reference harness until the performance branch produces equivalent artifacts and mask outputs on the frozen captures.

Performance branch requirements:

1. Every major pipeline step must log wall time, cumulative process memory, and the step-local memory estimate or allocation delta where that can be measured reliably.
2. Instrumentation must write a machine-readable summary so timing and memory deltas can be compared after each code change.
3. Debug artifact export must remain available after every optimization pass, even if some heavyweight exports become gated behind a debug flag.
4. Hotspot reductions must be accepted only after checking both performance metrics and artifact parity against the current reference output.
5. The branch should not enable all-subsection execution until the per-step metrics clearly identify the dominant costs and the optimized single-subsection path still preserves the required artifacts.

Recommended execution order:

### Phase P0: Freeze The Performance Branch Entry Point

Tasks:

1. Build and run the new `offline_dino_validator_performance` binary through `run_offline_dino_validator_performance.sh`.
2. Confirm the copied branch reproduces the current single-debug-chunk behavior before any instrumentation changes.
3. Keep the current reference validator runnable in parallel so timing changes can always be compared back to the known-good artifact set.

Exit criteria:

- the new performance binary runs end to end on the current subsection workflow
- the copied launcher resolves the performance binary without replacing the existing validator flow

### Phase P1: Add Stage Timing And Memory Instrumentation

Instrument these stages first:

1. tensor load and metadata parse
2. frontend power and correction generation
3. chunk planning
4. chunk extraction and preprocessing
5. Torch runtime invocation
6. DINO score remap or resize
7. coherence computation
8. retry-hybrid support or residual-veto scoring
9. grouping
10. projection and merge
11. artifact serialization

Implementation notes:

1. Use one shared stage-profiler helper so each stage records start time, stop time, elapsed milliseconds, RSS or HWM snapshot, cumulative peak memory, and optional component-estimated bytes.
2. Emit both human-readable logs and JSON summary output in the artifact directory.
3. Keep per-chunk and whole-run timing separate so subsection scaling can be estimated before the all-chunk loop is enabled.

Exit criteria:

- one run produces a stage-by-stage timing and memory report for the current debug chunk flow
- the report is stable enough to rank hotspots instead of guessing

### Phase P2: Optimize Hotspots Without Breaking Artifact Parity

Optimization rules:

1. Change one hotspot family at a time.
2. After each hotspot change, rerun the performance validator with the same tensor and compare artifact outputs against the current reference branch.
3. If a speedup requires dropping an artifact, gate that artifact behind a debug or verbose option instead of removing it outright.

Expected hotspot areas to test first:

1. repeated chunk buffer allocations
2. unnecessary host-device or device-host copies
3. full-resolution intermediate materialization when only reduced-grid data is needed
4. dense grouped-score scratch buffers on non-debug chunks
5. repeated resize or normalization passes that can be fused or reused

Exit criteria:

- the performance branch shows measured improvement on the dominant stages
- the debug artifact set remains sufficient for notebook parity checks

### Phase P3: Expand From Debug Chunk To All Subsections

Tasks:

1. Re-enable the full subsection loop in the performance branch only after the single-subsection metrics are understood.
2. Keep per-chunk timing and memory entries so the slowest subsection can be identified quickly.
3. Preserve the selected debug chunk artifact export while allowing non-debug chunks to use lighter-weight paths where parity does not require heavyweight dumps.
4. Add whole-frame summary metrics for total runtime, peak RSS or HWM, total chunk count, and mean, median, and max chunk latency.

Exit criteria:

- the performance validator runs every planned subsection, not just the selected debug chunk
- full-run timing and memory reports identify whether the remaining bottleneck is per-chunk runtime cost, merge cost, or artifact overhead

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

Live-operator implementation note:

When this validator/reference path is ported into the live DINO operator, replace the remaining Torch-driven frontend-correction and coherence helper stages with custom CUDA implementations modeled after the coherent-power fast path. Treat the current validator work as the parity reference, but target dedicated CUDA kernels for the live operator so chunk-local preprocessing and coherence stay device-native with lower framework overhead.

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

## Progress Log

### 2026-04-20 Progress

Completed so far:

1. Ran the scripted subsection validator flow on chunk 13 with:
   - `sudo PYTHON_BIN=/home/sat3737/holoscan_demo_workspace/.venv/bin/python3 ./validate_offline_dino_subsection.sh --tensor-npy /tmp/usrp_spectrograms/tensors/spectrogram_tensor_ch0_f1_1776619958065_1024x20480.npy --config /home/sat3737/holoscan_demo_workspace/holohub-dev/applications/usrp_wideband_signal_detection/config_torchscript_validation_capture_single_channel.yaml --debug-chunk-index 13`
2. Confirmed the chunk-debug artifacts were updated under `/tmp/usrp_spectrograms/dino_validator_artifacts/spectrogram_tensor_ch0_f1_1776619958065_1024x20480/chunk_debug/`.
3. Captured the latest chunk 13 comparison report:
   - chunk plan row match: `True / True`
   - grouped mask agreement: `0.976395`
   - grouped mask IoU: `0.768413`
   - box counts cpp/python: `5 / 6`
   - exact box signature match: `False`
4. This is directionally better than the earlier chunk-local grouping baseline and is good enough to justify deeper stage-by-stage visual inspection instead of only summary metrics.
5. Added a new crash-resistant validation helper module at `/home/sat3737/holoscan_demo_workspace/holohub-dev/notebooks/torchscript_dino_signal_detector_validation_v2_helpers.py`.
6. Added a new notebook at `/home/sat3737/holoscan_demo_workspace/holohub-dev/notebooks/torchscript_dino_signal_detector_validation_v2.ipynb`.
7. The v2 notebook is intentionally thin: it loads the existing offline C++ validator artifacts, reconstructs the selected Python reference chunk through helper code, maps the Python intermediates into the C++ chunk grid, and renders per-stage comparisons for corrected spectrogram, coherence gate, DINO score, hybrid contribution, combined score, final mask, grouped masks, and grouped boxes.
8. The old notebook remains available for reference, but the new helper-backed notebook should be the primary debugging surface going forward because it minimizes notebook-only logic and keeps recovery simpler after crashes.
9. Confirmed the coherence mismatch was mostly an order-of-operations issue: Python computed structure-tensor coherence on the source-resolution chunk and then mapped it to the C++ grid, while the C++ validator had been computing coherence directly on the resized chunk.
10. Switched the validator coherence path to the Python order for both the chunk-local debug artifact and the legacy full-frame artifact export.
11. Identified a separate DINO-score parity gap: the Python notebook `dino_score_px` is a grouped affinity/seed score produced by `dino_region_grouping_mask(...)`, while the current C++ runtime `score_map` is a normalized raw feature-energy map derived in `derive_dino_score_map(...)`.
12. That means the current DINO panel is not apples-to-apples. The next parity step must compare the C++ `score_map` against a Python raw feature-energy score derived from the same patch feature tensor before trying to match the higher-level Python grouped DINO score.
13. Updated the offline validator chunk path so DINO inference runs on the source chunk resolution aligned to patch size, then resizes the resulting C++ score map back into the existing debug grid. This should eliminate the largest remaining input-resolution mismatch between the chunked C++ path and the Python reference.

### 2026-04-19 Progress

Completed so far:

1. Copied this plan into the dated working document.
2. Added source-coordinate chunk planning to `offline_dino_validator.cpp`.
3. Exported `offline_chunk_plan.json` so subsection scheduling survives crashes and can be compared against Python.
4. Added reusable per-chunk execution scaffolding with `run_retry_chunk(...)`.
5. Wired chunk-local DINO retry execution across all planned subsections.
6. Exported `offline_chunk_results.json` with per-chunk thresholds and mask statistics.
7. Exported one representative debug subsection artifact set under `chunk_debug/`.
8. Added a first C++ chunk-local grouping path for subsection masks.
9. Added projected global boxes and an initial merged global mask path.
10. Exported projected and merged global artifacts for inspection after each run.
11. Added explicit `score_map` fields to the DINO runtime result and switched the validator/operator call sites away from the misleading `final_mask` name.
12. Added a projected grouped score-map artifact in source coordinates so the next merge pass can use averaged subsection scores instead of box overlap alone.
13. Switched the validator toward score-driven global regrouping and richer box metadata so subsection and merge validation can inspect the same fields the Python path uses.
14. Added selectable debug-chunk export so one subsection can be validated repeatedly without editing the source.
15. Added a host-side subsection comparison script and a wrapper flow so rebuild, validator run, and Python grouping comparison can be driven from app-local scripts.

Current implementation state:

- full-frame legacy retry output still exists and remains the current validator-level final mask
- chunk-local retry execution now exists in C++
- chunk-local grouping now exists in a simplified C++ form
- global projection and merge now exist in an initial box-based form
- the runtime result contract now distinguishes raw DINO score output from the legacy `final_mask` alias
- the validator now exports a projected grouped score map in source coordinates for score-aware merge work
- grouped box artifacts now carry score and source-chunk metadata needed for subsection validation
- the validator runner now accepts a selected debug chunk and the app has a chunk-level Python comparison entry point
- coherence export order now matches the cleaner Python source-resolution path, but the remaining DINO score comparison still mixes raw C++ feature-energy output with a higher-level Python grouped score map
- the validator chunk DINO runtime now evaluates the full source chunk resolution before resizing the exported score map back into the debug grid, reducing the prior DINO input-resolution mismatch
- the remaining gap is to replace the simplified grouping and merge with notebook-faithful scoring, region grouping, and final mask selection

Immediate next slice:

1. rerun the chunk comparison notebook after the source-resolution DINO runtime change and check whether the C++ score map now aligns better with the Python raw feature-energy score
2. verify the TorchScript runtime still returns spatial patch features with the expected patch grid, rather than a class-token or otherwise mis-shaped output
3. decide whether to port the higher-level Python grouped DINO score semantics into C++ or treat that as a later stage that should be compared after raw score parity is established
4. refine chunk-local grouping to better match `group_signal_mask_regions(...)`
5. replace simple overlap-only global box merging with notebook-style merged-score-aware grouping
6. decide when the projected global mask becomes the primary validator final mask instead of a side artifact
7. use the new explicit runtime `score_map` contract as the base for merged-score-aware global projection and mask selection
8. use `torchscript_dino_signal_detector_validation_v2.ipynb` as the stepwise visual debugger for the current chunk 13 artifact set and the next selected chunks

## Definition Of Done

This task is done when all of the following are true.

1. The offline C++ DINO validator subdivides the wideband spectrogram into overlapping frequency chunks.
2. Each chunk runs the retry-hybrid DINO subsection behavior that matches the Python reference.
3. Chunk-local detections are grouped, projected, and merged back into one global mask.
4. The final global mask matches the Python chunked retry reference on frozen validation captures within agreed tolerance.
5. The implementation path is structured so the same logic can later be moved into the live operator without changing the algorithm.
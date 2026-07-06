# Coherence-rescue floor (wideband DINO-suppression fix)

Branch: `debug_dino_signal_detection`

## Problem

On wideband signals the coherence pathway is correct, but the DINO pathway
suppresses the detection over large regions where coherence is strong but the
DINO/residual response is weak. Narrowband works well and must not regress.

Root cause (all in `operators/cuda_dino_detector/cuda_dino_detector.cu`):
- Fusion is multiplicative with no floor: `combined = qnorm(dino) * qnorm(coherence)`
  (~L7616-7631). A near-zero DINO factor zeroes a confident coherence region.
- Residual-veto (`keep_res` = 2nd derivative ≈ 0 for smooth wideband) multiplies
  by 0.35× (`residual_veto_combined_input_value`, ~L5763-5769).
- Seed gate requires `keep_res ≥ p82` (~L6443) → smooth wideband never seeds.
- Raw DINO residual-deweight (0.75) fits away smooth signals; per-chunk quantile
  norm crushes contrast when a signal fills the chunk.

## Fix

A **coherence-rescue floor**: a monotonic OR (set-union) into the *final* mask,
computed from the pristine coherence gate, gated by **magnitude + spatial extent**
(large contiguous strong-coherence regions). It only ADDS pixels, so it can never
remove a narrowband detection. Disabled by default → byte-identical to prior
behavior. Reuses the previously-dormant `dino_coherence_gate_floor`/`_span_db`.

New config (in the `cuda_dino_detector:` block):
- `dino_coherence_rescue_enable` (default `false`)
- `dino_coherence_rescue_min_area_px` (default `256`) — spatial-extent gate
- `dino_coherence_rescue_threshold` (default `-1` → derive
  `gate_floor * 10^(gate_span_db/20)` ≈ 0.35 at the 0.25/3 dB defaults)
- `dino_coherence_rescue_floor_strength` (default `0.0` → mask-only)

New debug artifact: `chunk_debug/chunk_coherence_rescue_mask.npy` (rescue-alone
contribution; the DINO `chunk_hybrid_component_filtered_mask.npy` stays pure DINO,
`chunk_final_mask.npy` = DINO ∪ rescue). `plot_hybrid_support_components.py` shows it.

Files touched: `operators/cuda_dino_detector/{cuda_dino_detector.cu,cuda_dino_detector.hpp,
cuda_dino_torch_helpers.cpp,cuda_dino_torch_helpers.hpp}`,
`config_cuda_dino_performance_single_channel.yaml`, `plot_hybrid_support_components.py`.

## Build (user runs — container)

```bash
cd applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh    # FORCE_REBUILD=1 ./rebuild_demo_container_app.sh if the tree looks current
```

## Phase 1 — Reproduce & confirm mechanism (rescue OFF)

Pick a strong wideband capture (dB_0). Debug dump auto-enables when artifacts are saved.

```bash
cd applications/usrp_wideband_signal_detection
CAP=generated_inputs/attenuation_dB_0_samples_150391762_155634642.sigmf-data
python3 run_cuda_dino_offline_file.py "$CAP" --detector cuda_dino \
  --output-root /tmp/usrp_spectrograms/offline_cuda_dino/dbg_baseline \
  --debug-chunk-index <wideband_chunk> --progress-every 25
python3 plot_hybrid_support_components.py \
  /tmp/usrp_spectrograms/offline_cuda_dino/dbg_baseline/chunk_debug/chunk_debug_summary.json
```
Expect: strong `coherence_gate`, near-zero `keep_res`, crushed `combined_score`,
empty `coherence_rescue_mask` (rescue off), missing wideband region in `final_mask`.

## Phase 1b — Turn rescue ON

Set in `config_cuda_dino_performance_single_channel.yaml` (cuda_dino_detector block):
`dino_coherence_rescue_enable: true`. Rebuild not needed for config-only change, but
the offline wrapper reads the top-level config, so just re-run:
```bash
python3 run_cuda_dino_offline_file.py "$CAP" --detector cuda_dino \
  --output-root /tmp/usrp_spectrograms/offline_cuda_dino/dbg_rescue \
  --debug-chunk-index <wideband_chunk>
python3 plot_hybrid_support_components.py \
  /tmp/usrp_spectrograms/offline_cuda_dino/dbg_rescue/chunk_debug/chunk_debug_summary.json
```
Expect: `coherence_rescue_mask` now covers the large coherence region; `final_mask`
includes it. Narrowband regions unchanged.

## Phase 3 — Bandwidth-stratified regression (rescue OFF vs ON)

Run the batch eval both ways, then stratify `region_metrics.csv` by
`wfgt:occupied_bw_hz` (wideband ≈ 82.9 MHz QPSK; mid ≈ 20.7 MHz; narrowband = ZC).

```bash
cd applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments
# (run batch eval OFF, then flip config enable=true and run again ON)
python3 run_batch_offline_eval.py ...        # produces <root>/cuda_dino/<stem>/
python3 eval_detector_masks.py --run-dir <run> --detector cuda_dino \
  --captures-dir ../../generated_inputs --out-dir <out>
```
Pass criteria: wideband/large-region recall ↑ materially; narrowband per-region
precision/recall unchanged (rescue is a min-area-gated monotonic OR, so small
narrowband regions are ineligible). Sweep `min_area_px` / `threshold` to tune.
</content>

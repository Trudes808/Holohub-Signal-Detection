# Fusion v2: coherence-primary + DINO structure gate (+ DINO scout)

Branch: `debug_dino_signal_detection`. Supersedes Method 1 (coherence-rescue floor).

## Why

Verified from the dbg dumps: the legacy fusion multiplies `qnorm(dino) * qnorm(coherence)`,
so in strong-coherence regions a weak DINO score (~0.4) drags the product to ~41% of
coherence strength and pushes 55% of confident-coherence pixels below threshold. The
coherence-rescue floor (Method 1) mitigated this only where it fired, and its default
derived threshold was unreachable for the raw gate's ~0.1 scale.

Fusion v2 encodes the priors directly:
- **Coherence in bands = trust outright.** Threshold the raw gate, enforce time-continuity
  (1×k opening), drop small components → `coherence_band_mask`. DINO never scales it down.
- **DINO = low-SNR structure finder.** On the patch grid (16×32/chunk), per-chunk quantile
  threshold + a run-length opening (union of 1×N time and N×1 freq openings) = "≥N patches in
  a row" → `dino_structure_mask`. Self-normalizing quantile keeps false runs on pure noise ~1/chunk.
- **Final = band ∪ structure ∪ legacy** (legacy kept by default so narrowband can't regress).
- **Scout (Method 3):** run DINO only on chunks where `coherence_band_mask` covers < X% of valid
  pixels — noise speckle can't fake coverage (opening+min-area kill it), so all-noise chunks still
  get full DINO, while chunks coherence already explained skip inference.

## Config (cuda_dino_detector block)

| Param | Default | Notes |
|---|---|---|
| `hybrid_fusion_mode` | `legacy` | set `coherence_primary` to enable v2 |
| `coherence_band_threshold` | 0.05 | absolute, raw-gate ~0.1 scale |
| `coherence_band_threshold_quantile` | -1.0 | >=0 => per-chunk `max(absolute, quantile)` |
| `coherence_band_open_time_px` | 5 | time-continuity opening length |
| `coherence_band_close_freq_px` / `_time_px` | 0 | optional closing |
| `coherence_band_min_area_px` | 256 | small-component drop |
| `dino_structure_threshold_quantile` | 0.90 | patch-score quantile |
| `dino_structure_open_len` | 3 | ≥N patches in a row |
| `coherence_primary_include_legacy_mask` | true | keep legacy detections in the union |
| `coherence_primary_legacy_score` | `max` | score fed to the veto sub-pipeline: `max(qnorm coh, qnorm dino)` so keep_freq/keep_res/seeds derive from the coherence-preserving score; `product` = verbatim legacy sub-mask |
| `dino_scout_enable` | false | Method 3 (requires coherence_primary) |
| `dino_scout_coverage_threshold` | 0.65 | skip DINO when band covers ≥ this fraction |

## New debug artifacts (chunk_debug/)

`chunk_coherence_band_mask.npy`, `chunk_dino_structure_mask.npy`, `chunk_initial_product.npy`
(the score fed to the veto sub-pipeline: the qnorm product in legacy mode / with
`coherence_primary_legacy_score: product`; the qnorm max with the default `max` — the
keep_freq/keep_res/seed panels always derive from this map), plus existing per-stage masks. Visualized by
`plot_hybrid_support_components.py`. Metadata adds `cuda_dino_hybrid_fusion_mode`,
`cuda_dino_scout_chunks_total/run/skipped`.

## Build (user runs)

```bash
cd applications/usrp_wideband_signal_detection && ./rebuild_demo_container_app.sh
```

## Validate (user runs)

Set a capture and reuse the offline wrapper (debug dump auto-on unless --no-tensors):
```bash
CAP=generated_inputs/attenuation_dB_35_samples_507728988_512971868.sigmf-data

# P0 byte-identity gate: legacy mode must match pre-change masks.
#   (hybrid_fusion_mode: "legacy") -> run, hash mask_arrays vs a HEAD run.

# Method 2: set hybrid_fusion_mode: "coherence_primary" in the config, then:
python3 run_cuda_dino_offline_file.py "$CAP" --detector cuda_dino \
  --output-root /tmp/usrp_spectrograms/offline_cuda_dino/dbg_m2 --debug-chunk-index 0
python3 plot_hybrid_support_components.py \
  /tmp/usrp_spectrograms/offline_cuda_dino/dbg_m2/chunk_debug/chunk_debug_summary.json
#   Inspect: Coherence Band covers the wideband bands; DINO Structure lights only on
#   >=3-patch runs; Final Mask = their union. Initial Product shows the old suppression.

# Method 3: also set dino_scout_enable: true, re-run to dbg_m3.
#   Compare final masks m2 vs m3 (near-identical on quiet spectrum) and
#   runtime_torch_forward in offline_validation_summary.json for the inference saving.

# Regression: batch eval both modes, stratify region_metrics.csv by wfgt:occupied_bw_hz
#   (wideband ~82.9 MHz QPSK / mid ~20.7 MHz / narrowband ZC). Wideband recall up,
#   narrowband unchanged. Add an all-noise capture to watch DINO structure FP rate.
```

## Tuning order
1. `coherence_band_threshold` / `_open_time_px` so the band mask tracks true bands (check
   `chunk_coherence_band_mask.npy` vs `chunk_coherence_gate.npy`).
2. `dino_structure_threshold_quantile` / `_open_len` on a low-SNR capture (structure should
   light on real faint signals, stay near-empty on noise).
3. `dino_scout_coverage_threshold` for the compute/recall trade-off.

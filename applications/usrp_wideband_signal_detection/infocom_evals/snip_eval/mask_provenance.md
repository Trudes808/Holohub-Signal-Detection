# Coherent-power mask provenance: why "the same detector" produced different masks

## The observation

Running the new `snip_pipeline.py` demo (2026-07-24) on `attenuation_dB_45` produced **358
snippets / 43.1 GB/hr** under the 100 kHz/5 ms gate, while the staged-mask eval
(`real_snip_metrics_minsize_v2.csv`) for the same capture, same snipper build, and same gates
measured **28 snippets / ~2 GB/hr**. Mean mask coverage differed too: **0.89 %** (fresh) vs
**0.71 %** (staged). At first glance this looks like the coherent-power detector is
non-deterministic. It is not.

## The detector IS deterministic — proven, not assumed

The coherent-power path is FFT → power vs a threshold → connected components: no sampling, no
random init, no order-dependent atomics in the mask output. Empirical proof: the pipeline's mask
stage for `test_1` (live OTA, 140 frames) was run **twice** back-to-back with the same config —
every one of the 140 `mask_arrays/*.npy` files is **byte-identical** (`cmp`, 140/140).

Also ruled out: the snipper/eval side. The offline replication over the pipeline's fresh
`attenuation_dB_45` masks predicts **358 boxes / 43.2 GB/hr** vs the real pipeline's measured
**358 / 43.1** — the snip stage faithfully reflects whatever masks it is given.

## The actual cause: two different detector configs

Each mask run records its exact config in `offline_eval_summary.json`. Reading them:

| mask set | produced | config used |
|---|---|---|
| staged batch (`/tmp/usrp_spectrograms/all_detectors/coherent_power/…`, symlinked into `snip_run/`) | 2026-07-21 | `config_coherent_power_performance_single_channel.yaml` — since moved to `old_configs/` |
| fresh pipeline demo (`snip_pipeline.py`) | 2026-07-24 | `config_coherent_power_perf_perfreq_single_channel.yaml` — the driver's current per-detector default |

The `coherent_power_signal_detector` blocks differ materially (diff of the two generated configs
for `attenuation_dB_45`):

- **Fresh (perfreq)** enables the **per-frequency calibrated noise floor**:
  `per_freq_threshold_mode: "calibrated"`, `per_freq_threshold_offset_db: 2.0`,
  `per_freq_threshold_path: calibration/coherent_power_per_freq_floor.npy` (calibrated
  2026-07-16, i.e. *before* both runs — the file itself is not the difference).
- **Staged (old performance)** has none of the `per_freq_threshold_*` keys (legacy global
  threshold) and carries extra frontend/grouping params the perfreq config dropped
  (`frontend_row_q: 25.0`, `grouping_bridge_freq_px: 33`, `grouping_bridge_time_px: 5`, …).

A different threshold model is a different detector operating point: at +9 dB SNR the calibrated
per-frequency floor (2 dB over the measured floor in each bin) recovers substantially more real
signal than the legacy global threshold — more lit pixels (0.89 % vs 0.71 %), and many more
components that clear the 100 kHz × 5 ms gate (358 vs 28). Both mask sets contain the same two
persistent spur lines (cols 293–294 at −115.84 MHz, col 7120 at +48.00 MHz; each ≤ 2 columns), and
the `min_mask_bandwidth_hz` filter removes them identically in both — the fix is unaffected by
this provenance issue.

(A note for the record: when this discrepancy was first noticed it was provisionally attributed to
"calibration drift". That was wrong in the details — the calibration file predates both runs; the
actual difference is *which config* each run loaded.)

## Why the defaults diverged

`run_cuda_dino_offline_file.py` maps `--detector coherent_power` to the app's *current* documented
live config (`config_coherent_power_perf_perfreq_single_channel.yaml`). The July batch was
produced before/outside that default with the plain performance config, which the July
reorganization then retired to `old_configs/`. So "run the coherent detector" silently means
something different today than it did when the staged masks were made — classic provenance drift,
no non-determinism involved.

## Implications

1. **Within-family comparisons are all valid.** Every staged-mask number
   (`real_snip_metrics*.csv`, `fix_quantification.csv`, the before/after figures) is internally
   consistent — same masks throughout. Every fresh-pipeline number is likewise internally
   consistent.
2. **Across-family comparisons need a provenance note.** A fresh-pipeline footprint is not
   comparable to a staged-mask footprint at the same attenuation without stating the detector
   config. The gap can be large exactly where detection is marginal (mid/low SNR).
3. **The mask-filter validation is unaffected**: the spur lines are ≤ 2 columns in both families
   and the filter removes them identically; the before/after conclusions hold within the staged
   family they were measured on.

## Guardrails (adopted / recommended)

- `snip_pipeline_demo.yaml` now sets `detector_config` **explicitly** instead of relying on the
  driver default, so a pipeline run's operating point is pinned in the config file itself.
- When reproducing the *staged* family, pass
  `detector_config: old_configs/config_coherent_power_performance_single_channel.yaml`.
- `offline_eval_summary.json` already records the exact generated config per mask run — treat it
  as the mask set's birth certificate and check it before comparing runs. (Possible future
  hardening: also record a hash of the calibration `.npy` the config references.)

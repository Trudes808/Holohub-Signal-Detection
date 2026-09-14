# DINO-FT robust normalization — validation (roadmap step 1)

Validates the **density/noise-robust normalization** added to `finetuned_dino_detector`
(`adaptive_robust_floor`): a LOW-percentile (p20) histogram floor of the whole dB image, anchored to a
fixed span, with a fallback to the fixed calibrated clip when the frame has no usable dynamic-range
spread (dense/mostly-signal) or an implausibly low floor (noiseless). Motivation and plan:
`../../../notes/dino_ft_finetune_plan.md`.

## What this tests
The prior per-frame adaptive normalization (q=75 flatten-reference anchor) **collapses at density/SNR
extremes** — it over-brightens dense/noiseless frames and crushes signals on low-SNR frames. The robust
path must (a) never collapse the model input on any scene, and (b) still adapt on realistic sparse
scenes. We dump the actual `[0,1]` model input + emitted mask for a handful of frames in 3 modes
(`fixed` / `adapt` / `robust`) across 3 scenes and quantify contrast.

## Run
```
python3 gen_configs.py            # writes config_{fixed,adapt,robust}.yaml (identical detector block
                                  #   except the normalization knobs + debug dumps)
bash run_validation.sh            # 9 offline evals (3 modes x 3 scenes); dumps -> /tmp/.../robust_val/
python3 analyze.py                # stats table + results/robust_norm_compare.png + summary.json
```
Scenes: `clean` (comprehensive_ordered subset, dense + noiseless), `snr0` (same, +0 dB AWGN — dense,
low SNR), `sparse` (the real 2.4 GHz OTA capture). SNR variants come from
`../dino_ft_snr_benchmark/` (`make_snr_variants.py`).

## Result (results/robust_norm_summary.json, results/robust_norm_compare.png)
Model input `[0,1]` — std / % saturated (≥0.99) / % dead (≤0.02) / mask occupancy:

| scene  | mode   | std  | sat% | dead% | occ%  | note |
|--------|--------|------|------|-------|-------|------|
| clean  | fixed  | 0.31 | 0.5  | 31.5  | 0.00  | |
| clean  | adapt  | 0.45 | **45.1** | 30.5 | 0.00 | **over-saturates** — signal detail lost in white |
| clean  | robust | 0.35 | 7.4  | 32.5  | 0.00  | ≈ fixed, no saturation collapse |
| snr0   | fixed  | 0.09 | 0.6  | 0.0   | 8.17  | bright ⇒ current model fires |
| snr0   | adapt  | 0.12 | 0.0  | 22.4  | 0.00  | darkens signals into noise |
| snr0   | robust | 0.14 | 0.0  | 9.7   | 0.00  | **preserves signals** (best contrast) but current model doesn't fire |
| sparse | fixed  | 0.09 | 0.0  | 0.7   | 0.34  | |
| sparse | adapt  | 0.12 | 0.1  | 22.1  | 0.32  | |
| sparse | robust | 0.14 | 0.2  | 9.7   | 0.33  | best contrast, detections on par |

## Conclusions
1. **Robust normalization never collapses.** No scene exceeds 50% saturated or 90% dead; std stays
   healthy (0.14–0.35) everywhere. It specifically **fixes the adaptive over-saturation on the dense
   noiseless frame** (7% vs 45% saturated) and avoids adaptive's excessive dead pixels.
2. **The residual low-SNR detection gap is a model-distribution mismatch, not a normalization defect.**
   At snr0 robust keeps the signals clearly visible (higher input std than fixed), but the *current*
   checkpoint — trained for fixed-clip brightness — reads robust's brightness as noise and doesn't fire.
   Fixed happens to map snr0 into the model's trained band, so it fires. This is exactly why the plan
   couples robust normalization with **training the model ON it** (roadmap step 2).
3. **Deployment coupling (safety).** Because robust under-detects at low SNR with the current model, the
   5 shipped DINO-FT demo configs are pinned `adaptive_robust_floor: false` (legacy q-blend) until a
   checkpoint fine-tuned on robust normalization exists. Flip to `true` alongside that checkpoint.

Dumps under `/tmp/usrp_spectrograms/robust_val/` are ephemeral; regenerate with `run_validation.sh`.

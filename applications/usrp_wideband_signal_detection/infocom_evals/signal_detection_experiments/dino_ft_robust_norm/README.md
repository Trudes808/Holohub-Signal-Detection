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
3. **Deployment coupling (safety).** Because robust under-detects at low SNR with the *fixed-clip-trained*
   model, the demo configs were first pinned `adaptive_robust_floor: false` (legacy q-blend) until a
   checkpoint fine-tuned on robust normalization existed. **That checkpoint now exists (M3_491, below), so
   the configs are flipped to `true`.** (The pin was the correct interim state before the retrain.)

Dumps under `/tmp/usrp_spectrograms/robust_val/` are ephemeral; regenerate with `run_validation.sh`.

---

# M3_491: domain-match fine-tune on robust normalization (roadmap steps 2-5)

The robust normalization (above) changes the model-input distribution, so the fixed-clip-trained M2_dr
under-detected under it at low SNR. Fix per the locked plan: **retrain the segmenter ON the robust
front-end at the 491.52 deployment geometry**, across dense+sparse scenes and per-signal SNR.

**Pipeline** (`dino_fine_tuning/src/`, run in `.venv-ml` with `PYTHONPATH=~/Documents/dinov3:src`):
- `frontend.py` — torch replica of the deployed downsample front-end (wide FFT 20480 @491.52 -> flatten ->
  robust p20 norm -> bilinear resize freq->1024 -> tile 256). **Validated corr 1.0000 / MAE 0.0002 vs the
  CUDA operator** on the real 491.52 capture. Robust norm is scale-invariant, so cf32 composites transfer
  to the sc16 radio.
- `gen_491.py` — places real MATLAB library waveforms (9 classes, resampled 2x->491.52) on a 491.52 canvas
  with controllable density (dense/sparse) + per-signal SNR (near/far) + AWGN floor; emits SigMF+GT.
- `build_dataset_rt.py` — streams composites through the front-end into `data/dataset_491/`
  (train 2265 / val 277 / test 201 tiles) + `heldout_sigmf/` (6 composites for the operator A/B).
- `train.py --mode ft_lastN` from the base DINOv3 backbone -> **M3_491**: 18 epochs, val IoU 0.748->0.9555,
  F1 0.977, P 0.975, R 0.980.
- `export_dinov3_finetuned_torchscript.py` -> `weights/finetuned_dino_m3_491_bf16.ts` (thr 0.6, bf16;
  eager-vs-traced IoU 1.0; loads+runs in the container torch 2.10).

**Held-out per-SNR** (`results/m3_heldout/heldout_snr.png`, `eval_heldout.py`) @thr 0.6: region detection
**83-100% across every SNR bucket down to -3..3 dB**, dense and sparse. Threshold F1 is flat ~0.96 over
0.4-0.9 (robust to the decision threshold).

**Container A/B** — M3 vs M2_dr through the REAL operator on held-out composites (`compare_ab.py`,
`results/ab_summary.json`), pixel recall / region detection / IoU:

| model | scene | pix recall | region det | IoU |
|-------|-------|-----------|-----------|-----|
| M2_dr | dense  | 53% | 51% | 52% |
| M2_dr | sparse | 35% | 43% | 20% |
| **M3_491** | dense  | **79%** | **72%** | **72%** |
| **M3_491** | sparse | **86%** | **86%** | **51%** |

The sparse scene (the live failure mode) goes 43%->86% region detection in the deployment operator path.
The 5 DINO-FT demo configs now point at M3 with `adaptive_robust_floor: true`, threshold 0.6.

**Still to do**: a live radio run (config_live_v3_dino_ft.yaml) to confirm on-air.

Regenerate: `gen_configs.py`/`run_validation.sh`/`analyze.py` (robust norm) and, for the retrain,
`build_dataset_rt.py` -> `train.py` -> `export_...py` -> `eval_heldout.py` + `compare_ab.py`.

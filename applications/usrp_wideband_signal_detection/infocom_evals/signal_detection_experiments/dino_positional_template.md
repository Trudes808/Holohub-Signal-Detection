# DINO positional (RoPE) noise-template calibration

Branch: `debug_dino_signal_detection`.

## Problem

On noise-only input `chunk_dino_score_raw_deweighted` shows a bright edge ring (falling
~0.82 at the edge → ~0.175 interior on a 64×64 patch grid) that propagates into spurious
edge detections. DINOv3 has no additive positional embedding — position enters only via
2D RoPE on Q/K, leaving a sharp, structural border bias at eval that the smooth 16-term
online fit (`raw_dino_positional_deweight`) cannot remove. The bias is deterministic
(55 dB vs 45 dB noise raw-score maps correlate 0.991) and a fixed function of (model,
64×64 grid) → calibrate empirically and divide in score space.

## What was implemented

- **Score-space division** in both raw-score kernels (`raw_dino_rms_energy_batch_kernel`,
  `raw_dino_project_energy_batch_kernel`): `score /= max(1 + strength*(template[p]-1), 1e-3)`.
  Mean-1 template ⇒ scale-preserving; strength 0 / empty path ⇒ exact prior behavior.
- **Calibration dumps** (`save_raw_dino_patch_prenorm`): frame-indexed pre-qnorm per-patch
  RMS → `<out>/chunk_debug/patch_prenorm/patch_prenorm_f*.npy` + `meta.json`.
- **`calibrate_dino_positional_template.py`**: averages noise dumps → mean-1 clamped
  template npy + sidecar + diagnostics; hard-fails if cross-run correlation < 0.98.
- Config keys (cuda_dino_detector block): `save_raw_dino_patch_prenorm`,
  `save_raw_dino_patch_features`, `raw_dino_positional_template_path`,
  `raw_dino_positional_template_strength`, `raw_dino_positional_mu_path` (Stage-D hook).
- Dedicated dump config `config_cuda_dino_calibration_dump_single_channel.yaml`.

Default off (empty template path) ⇒ byte-identical to prior behavior.

## Procedure (user runs; container)

```bash
cd applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh

# 1. Calibration dumps — full-length NOISE captures, template disabled.
for CAP in <noise_55dB.sigmf-data> <noise_45dB.sigmf-data>; do
  python3 run_cuda_dino_offline_file.py "$CAP" --detector cuda_dino \
    --config config_cuda_dino_calibration_dump_single_channel.yaml \
    --output-root /tmp/usrp_spectrograms/offline_cuda_dino/noise_$(basename "$CAP" .sigmf-data)
done

# 2. Build the template (gate: cross-run correlation >= 0.98).
python3 calibrate_dino_positional_template.py \
  --run-dir /tmp/usrp_spectrograms/offline_cuda_dino/noise_<55dB stem> \
  --run-dir /tmp/usrp_spectrograms/offline_cuda_dino/noise_<45dB stem> \
  --output calibration/dino_vitb16_noise_sigma_64x64.npy \
  --expect-deweight 0.75 --plots-dir calibration/diagnostics

# 3. Enable in the detector config:
#    raw_dino_positional_template_path: /workspace/holohub/applications/usrp_wideband_signal_detection/calibration/dino_vitb16_noise_sigma_64x64.npy
#    (rebuild not needed for a config-only change; the offline wrapper re-reads the config)

# 4. Noise validation (template on): re-run replay + dino_cuda_validation.sh,
#    inspect the "DINO Raw Deweighted" panel (should be flat).
#    Accept: edge_interior_ratio in [0.85,1.2] (baseline ~3.0), ring_cv < 0.05,
#            noise final-mask positive fraction <= 10% of baseline.

# 5. Signal non-regression: run_batch_offline_eval.py over the capture sweep;
#    pixel/region metrics within +-1-2% of the template-off baseline.

# 6. Byte-identity: with template path empty, dino_cuda_validation.sh matches current branch.
```

## Notes / risks

- The template divides the *residual* energy, so it is tied to `raw_dino_positional_deweight`
  (0.75); the calibration script asserts `--expect-deweight` matches the dump meta.
- `dino_structure_threshold_quantile: 0.90` always takes the top 10% of patches; today they
  cluster in the edge hump (contiguous → survive the run-length opening). After flattening
  they scatter and `dino_structure_open_len: 3` should kill them; an absolute score floor is
  a follow-up knob if residual FPs persist.
- Reference CPU validator has no template — run parity comparisons with the template disabled.
- Scout mode: template still applies (per-patch position, valid for subsets); prenorm dumps
  are full-batch only.
- Stage D (embedding-space mean subtraction) is a declared hook (`raw_dino_positional_mu_path`)
  only — wire it only if Stage C flatness is insufficient.

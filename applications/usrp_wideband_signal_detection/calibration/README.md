# DINO positional noise templates

Empirical per-position (RoPE) noise calibration for the CUDA DINO detector.

DINOv3 has no additive positional embedding to subtract; position enters only via 2D
RoPE on Q/K, which at eval leaves a fixed, sharp edge/corner bias in the per-patch
residual RMS. On noise it shows as a bright ring toward the image edges that the smooth
16-term online fit cannot remove. Chunks are always resized to a fixed 64×64 patch grid,
so the bias is a deterministic function of (model, grid) — calibrate it once and divide.

Artifacts here (produced by `calibrate_dino_positional_template.py`):
- `dino_vitb16_noise_sigma_64x64.npy` — mean-1 float32 `(patch_rows, patch_cols)` template.
- `dino_vitb16_noise_sigma_64x64.json` — sidecar (model, grid, deweight, source runs, metrics).
- `diagnostics/` — heatmap + ring-profile PNGs.

Container path for the detector config: files here mount at
`/workspace/holohub/applications/usrp_wideband_signal_detection/calibration/`.

See `../infocom_evals/signal_detection_experiments/dino_positional_template.md` for the
full build + validation procedure.

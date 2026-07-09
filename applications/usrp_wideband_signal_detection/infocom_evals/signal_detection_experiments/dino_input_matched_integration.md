# DINO input matched-integration contrast enhancement

Branch: `debug_dino_signal_detection`.

## Goal

Make regions of similar power stand out for DINO so it detects **low-SNR** signals. Global
contrast stretching cannot help — it scales signal and noise together. The limiter is the
per-pixel noise *variance* (complex-Gaussian power → ~5.6 dB std per pixel in the dB domain),
not the level. A faint signal lifts a region's *mean* by only ~1–3 dB, which is buried under
the pixel noise.

## Approach (matched integration / energy detector)

The only separator is spatial/temporal **coherence**: a real emission is elevated
*consistently* across a contiguous region; noise is IID. So integrate over the region:
average **linear power** over a window matched to the expected signal extent, then convert
back to dB. Averaging N IID samples shrinks the noise std by ~√N while preserving the
region's mean lift — real SNR gain, not just brightening.

- Window intentionally **narrow in frequency, long in time**: default `2` freq bins ×
  `25` time bins (signals persist in time; narrowband stays sharp). N ≈ 3×25 = 75 taps →
  noise std ≈ 5.6/√75 ≈ 0.65 dB, so a +2 dB region becomes clearly separated.
- **DINO-only**: written to a separate `dino_enhanced_batch_device`; the coherence path keeps
  the un-integrated corrected batch (25-bin time integration would smear directional
  coherence). Applied to the packed **native-resolution** batch right before `run_batch`,
  so it covers both the full-batch and scout paths.

## As-built

- Kernel `cuda_dino_matched_integrate_db_batch_kernel` (dB→linear, box-mean over
  ±freq_radius × ±time_radius, →dB; edges use available taps). `radius = bins/2`.
- Config (cuda_dino_detector block): `dino_input_enhance` (bool, default **false** ⇒
  byte-identical input), `dino_input_enhance_freq_bins` (2), `dino_input_enhance_time_bins` (25).
- Debug artifact `chunk_dino_enhanced_input.npy` + plot panel **"DINO Enhanced Input"**.
  When enhancement is off the panel falls back to the plain corrected batch, so comparing
  "DINO Enhanced Input" vs "Corrected" shows the contrast gain directly.
- New config `config_cuda_dino_input_enhance_single_channel.yaml` (enable = true).

## Test

```bash
cd applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh          # code change: rebuild required

# low-SNR signal capture, enhancement ON
sudo python3 run_cuda_dino_offline_file.py generated_inputs/attenuation_dB_45_*.sigmf-data \
  --detector cuda_dino --config config_cuda_dino_input_enhance_single_channel.yaml \
  --output-root /tmp/usrp_spectrograms/offline_cuda_dino/enhance_dB45 --debug-chunk-index 0
python3 plot_hybrid_support_components.py \
  /tmp/usrp_spectrograms/offline_cuda_dino/enhance_dB45/chunk_debug/chunk_debug_summary.json \
  --output /tmp/enhance_dB45.png
```

Accept: low-SNR regions visibly brighter/flatter-noise in "DINO Enhanced Input" vs "Corrected",
"DINO Raw Deweighted" responds where it previously didn't, low-SNR recall up. Watch: over-long
windows smear narrowband/edges; rely on the DINO structure gate (≥3 patches in a row) to reject
the extra noise texture that any local enhancement raises.

## Gray-blend rebalance + turbo colormap (built)

The signal-agnostic DINO gray transform (`signal_agnostic_dino_gray[_batch]` in
`operators/dinov3_signal_detector/dinov3_torch_runtime.cpp`) is what actually reaches the ViT
— not the corrected dB. It builds `gray = 0.70*local_resid + 0.30*abs_detrended` then quantizes
to 8-bit and (previously) replicated it into 3 identical channels. The 0.70 local high-pass
weight suppresses faint spatially-broad signals (local residual ≈ 0 over a smooth elevated
region) — the same failure mode as keep_res, baked into the DINO input.

- `dino_gray_local_resid_weight` (default 0.70 = prior behavior): blend weight `w` in
  `gray = w*local_resid + (1-w)*abs_detrended`. Lower toward 0 to favor absolute level so
  faint-broad signals survive. Set to **0.35** in the enhance config. **Primary sweep knob.**
- `dino_colormap_enable` (default false): feed a true 3-channel **turbo** colormap
  (`apply_turbo_colormap`, widest RGB gamut) of the gray scalar instead of grayscale-replicate,
  so small dB deltas become large decorrelated RGB deltas that excite the pretrained ViT's
  chromatic patch-embed filters (unused when the 3 channels are identical). Set **true**.

Both are DINO-input-only; the coherence path is untouched. Threaded through
`DinoTorchRuntimeConfig` and set from the operator params at both runtime-config sites.

**Panel caveat:** the "DINO Enhanced Input" panel shows the matched-integrated *dB* (pre
gray-transform), NOT the post-transform gray / turbo RGB the model ingests. To visualize the
true model input (needed to tune `dino_gray_local_resid_weight` and see the colormap), wire the
batch `return_pre_model_gray` (currently single-frame only) into a debug artifact — follow-up.

## Follow-ups (not built)

- Local CFAR z-score `(integrated − local_p25_bg) / local_MAD` for a calibrated,
  floor-adaptive detection statistic (the principled next step beyond raw integration).
- Padding contamination: freq integration near the valid/zero-padded chunk boundary pulls in
  0-dB padding; negligible at freq_bins=2 (±1 row) but grows with the frequency window.

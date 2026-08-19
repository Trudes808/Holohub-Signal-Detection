<!--
SPDX-FileCopyrightText: 2026 National Instruments Corporation
SPDX-License-Identifier: Apache-2.0
-->
# Fine-tuned DINO Detector Operator

Native detector for the fine-tuned DINOv3 segmenter (`DinoSegmenter` = DINOv3 ViT-B/16 backbone +
trained SegHead). Outputs a signal/noise mask directly (`sigmoid(model) >= threshold`) with no
fusion/post-processing stack, and emits `holoscan::ops::DetectorMaskMessage` on `mask_out` so
`signal_snipper` can snip it.

## Geometry-matched front-end

The model is trained at a fixed per-pixel physics (`bin_hz = sample_rate/nfft`,
`row_seconds = nfft/sample_rate`). To reproduce that live, the operator **taps the raw time-domain
IQ** (upstream of the app's wide analysis FFT, exactly like `signal_snipper`) and runs its **own
dedicated `nfft`-point FFT → dB spectrogram**, matching `dino_fine_tuning/src/rfdata.frames_to_db`
+ `finetuned_infer.mask_for_iq`:

```
spec = fftshift(fft(row), dim=freq)        # no window, no FFT normalization
db   = 10*log10(|spec|^2 + 1e-12)
img  = clamp((db - db_vmin) / (db_vmax - db_vmin), 0, 1)
```

The model input shape (`tile_rows × nfft`) is **config-driven** and must match the checkpoint's
exported `*.meta.json` geometry contract.

### Per-frequency noise-floor flatten (`flatten_noise_floor`, default **off**)

> **Off by default and not recommended for the shipped checkpoints.** The training preprocessing
> (`dino_fine_tuning/src/rfdata.frames_to_db`) does **no** flattening — it is a raw dB spectrogram +
> global `vmin/vmax` clip — and the training captures were OTA, so the model was trained *with* the
> receiver envelope present. Flattening therefore imposes a shape the model never saw. The correct
> approach is to **match** the training envelope + absolute power, not remove it. This knob is kept
> only as an opt-in tool for a hypothetical future model trained on flat-floor data.

The segmenter (for a flat-floor-trained model) could otherwise mistake a receiver's filter shape
(rolloff/tilt at the band edges) for signal. Adapted from
`coherent_power`'s frontend correction (freq on the **column** axis here), the operator estimates a
smooth per-frequency floor from each frame and additively **lifts** low-floor bins up to a
data-derived reference *before* the `[0,1]` clip:

```
floor[f]   = smooth_freq( mean_time( db[:,f] ) )    # capped at reference+signal_cap on a 2nd pass
reference  = blend(mean -> max, quantile) of floor  # flatten_reference_q
db[:,f]   += clamp(reference - floor[f], 0, flatten_max_boost_db)
```

It only *raises* low bins (never pulls signals down). Fully dynamic per frame — no calibration file
and **no no-antenna calibration step** — so it reproduces on any OTA capture. Set
`flatten_noise_floor: false` to restore the plain global-`vmin` front-end.

**Bandwidth-invariant by construction** (so the same config works at any sample rate with no
per-rate tuning):
- `flatten_smooth_frac` — the smoothing sigma is a **fraction of `fft_size`**, not an absolute bin
  count, because the filter rolloff is a fixed fraction of Nyquist. `sigma = max(2, frac*fft_size)`
  auto-scales with the runtime FFT geometry.
- `flatten_reference_q` is a **percentile** and `flatten_max_boost_db` / `flatten_signal_cap_db` are
  in **dB** (the rolloff depth is a hardware property) — all three are already scale-free.

## I/O contract

- **Input** `iq_in`: `std::tuple<matx::tensor_t<cuda::std::complex<float>,2>, cudaStream_t>`
  (raw time-domain IQ + stream), the same message `signal_snipper` taps from the CHDR converter.
- **TorchScript module**: input `float[B,1,tile_rows,nfft]` in `[0,1]` → output
  `logits[B,1,tile_rows,nfft]`; post `sigmoid(logits) >= threshold`. Channel-repeat + imagenet-norm
  are inside the traced model.
- **Output** `mask_out`: `DetectorMaskMessage` (native `rows × nfft` uint8 device mask, owned per
  frame; `frame_number` = per-channel IQ arrival index, aligning to the snipper's IQ ring).

## Parameters

`model_script_path`, `threshold`, `tile_rows`, `nfft`, `db_vmin`, `db_vmax`, `num_channels`,
`channel_filter`, `emit_stride`, `torch_dtype`, `real_time_downsample`, `downsample_fft_size`, and
the flatten knobs `flatten_noise_floor` / `flatten_reference_q` / `flatten_smooth_frac` /
`flatten_max_boost_db` / `flatten_signal_cap_db` (see above). See the app's
`config_cuda_dino_finetuned_performance_single_channel.yaml` and the app README section
**Fine-tuned DINO detector** for how to export a checkpoint and set these.

## Build

Registered in `operators/CMakeLists.txt` (`add_holohub_operator(finetuned_dino_detector)`) and linked
into the app. TorchScript is loaded via the CXX-only `finetuned_dino_torch_helpers.cpp` split so nvcc
never sees libtorch headers (mirrors `cuda_dino_detector`).

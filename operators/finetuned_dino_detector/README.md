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
`channel_filter`, `emit_stride`, `torch_dtype`. See the app's
`config_cuda_dino_finetuned_performance_single_channel.yaml` and the app README section
**Fine-tuned DINO detector** for how to export a checkpoint and set these.

## Build

Registered in `operators/CMakeLists.txt` (`add_holohub_operator(finetuned_dino_detector)`) and linked
into the app. TorchScript is loaded via the CXX-only `finetuned_dino_torch_helpers.cpp` split so nvcc
never sees libtorch headers (mirrors `cuda_dino_detector`).

# Computer Vision Baseline Operator (traditional image-processing detector)

A GPU-resident **classical computer-vision** detector that operates on the
spectrogram image — the pre-ML approach to signal detection. It is a baseline to
compare against the DINO image-segmentation detectors, and is intentionally
built from textbook CV primitives so its failure modes (faint signals below the
global threshold, blobs merged or split by morphology, edges from noise texture,
area filtering that drops small-but-real emitters) are easy to demonstrate.

## Signal flow

Consumes the complex spectrogram tensor `std::tuple<matx::tensor_t<complex,2>,
cudaStream_t>` (the same input the DINO detectors consume from the FFT /
Spectrogram operators) and runs entirely on the GPU:

```
complex spectrogram
  -> dB magnitude image (fftshifted)
  -> adaptive z-score threshold          (foreground)
  -> morphological opening + closing     (despeckle / fill)
  -> Sobel gradient magnitude + z-score  (edges)
  -> connected-component labeling + area filter (blobs)
  -> combine (blob / blob_or_edge / edge) -> DetectorMaskMessage
```

Input: `("in")` complex spectrogram, `num_bursts x burst_size`.
Output: `("mask_out")` `holoscan::ops::DetectorMaskMessage` (time x frequency, `uint8` 0/255).

All thresholds are expressed in z-score units relative to per-frame image
statistics, so no per-file / per-system calibration is required.

## Detection stages

1. **dB image**: `10*log10(|X|^2)` with an fftshift along frequency.
2. **Adaptive threshold**: foreground where dB `>` image-mean `+ threshold_zscore * std`.
3. **Morphology**: `open_iterations` of opening (erode→dilate) then
   `close_iterations` of closing (dilate→erode) with a `morph_radius` square
   structuring element.
4. **Edges**: Sobel gradient magnitude, thresholded at `edge_zscore` standard
   deviations of the gradient image.
5. **Blobs**: 8-connected component labeling (label-equivalence with path
   compression), then discard components smaller than `min_blob_area` pixels.
6. **Combine**: `combine_mode` selects `blob`, `blob_or_edge` (default), or
   `edge` for the emitted mask.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_channels` | 1 | Pipeline channel count (routing validation). |
| `channel_filter` | 0 | Channel index this instance handles. |
| `threshold_zscore` | 3.0 | Foreground threshold in std devs above the image mean. |
| `morph_radius` | 1 | Structuring-element radius (1 => 3x3). |
| `open_iterations` | 1 | Opening passes (remove speckle). |
| `close_iterations` | 1 | Closing passes (fill gaps). |
| `edge_zscore` | 3.0 | Sobel gradient-magnitude threshold (std devs). |
| `min_blob_area` | 32 | Minimum connected-component size in pixels. |
| `ccl_max_iterations` | 256 | Safety cap on label-propagation sweeps. |
| `combine_mode` | `blob_or_edge` | `blob`, `blob_or_edge`, or `edge`. |
| `emit_stride` | 1 | Emit one mask every N frames. |

## Build

Registered via `add_holohub_operator(computer_vision_baseline)` in
`operators/CMakeLists.txt`. Built as part of the
`usrp_wideband_signal_detection` container app.

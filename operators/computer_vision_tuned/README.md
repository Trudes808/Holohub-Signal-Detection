# Computer Vision (Tuned) Operator — stronger classical CV detector

A GPU-resident, **still fully classical (pre-ML)** image-processing detector on
the spectrogram. Same concept as `computer_vision_baseline`
(threshold → morphology → connected components) but with the textbook-correct
choices for a spectrogram image, so it is a *fair* baseline rather than a
strawman.

## What changed vs. `computer_vision_baseline`

| Concern in the naive baseline | Fix here |
|---|---|
| **Global** image threshold — assumes a flat floor; one strong signal desensitizes the whole frame | **Local per-frequency-column adaptive threshold**: a sigma-clipped background (mean/std over time) per bin, so each frequency is judged against its own floor. |
| Single hard threshold loses faint signal | **Hysteresis (dual) threshold**: a high threshold seeds, a low threshold grows via connected components; low regions are kept only if connected to a seed. Recovers faint pixels attached to a confident core. |
| Square structuring element erodes thin carriers/bursts | **Direction-aware opening**: union of a horizontal-line opening and a vertical-line opening, so thin wideband bursts *and* thin narrowband carriers survive while speckle is removed. |
| Edges OR'd into the occupancy mask add outline speckle | **Blob-only by default**; Sobel edges are retained as an optional `combine_mode`. |
| No DC/LO handling | **DC notch** at band center. |

## Signal flow

```
complex spectrogram
  -> dB image (fftshifted)
  -> per-column sigma-clipped background  (local floor)
  -> hysteresis dual threshold            (high seeds + low candidates)
  -> direction-aware opening (open_h ∪ open_v) + separable closing
  -> connected components + seed test + area filter
  -> (optional Sobel edges) -> combine -> DetectorMaskMessage
```

Input: `("in")` complex spectrogram, `num_bursts x burst_size`.
Output: `("mask_out")` `holoscan::ops::DetectorMaskMessage` (time × frequency, `uint8` 0/255).

All thresholds are statistical (sigma vs. a local background), so no per-file dB
calibration is required. All computation stays on the GPU.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `num_channels` / `channel_filter` | 1 / 0 | Routing. |
| `z_high` | 4.0 | Hysteresis seed threshold (sigma above the local column floor). |
| `z_low` | 2.0 | Hysteresis grow threshold (sigma); kept only if connected to a seed. |
| `clip_z` | 3.0 | Sigma-clip cut for the per-column background estimate. |
| `morph_radius` | 1 | Line structuring-element radius (1 => length-3 lines). |
| `close_iterations` | 1 | Separable (square) closing passes. |
| `edge_zscore` | 4.0 | Sobel gradient threshold (only used if `combine_mode` includes edges). |
| `min_blob_area` | 24 | Minimum connected-component size (pixels). |
| `ccl_max_iterations` | 256 | Safety cap on label-propagation sweeps. |
| `min_std_db` | 0.5 | Floor on the per-column background std. |
| `dc_notch_bins` | 4 | Bins each side of band center forced to no-detect. |
| `combine_mode` | `blob` | `blob`, `blob_or_edge`, or `edge`. |
| `emit_stride` | 1 | Emit one mask every N frames. |

## Notes / future work

The iterative label-propagation CCL syncs once per sweep; for live real-time on
large connected regions, a block-based CCL (BUF/Komura) removes the per-sweep
sync. The per-emit `cudaMalloc` of the owned mask buffer could be pooled.

## Build

Registered via `add_holohub_operator(computer_vision_tuned)` in
`operators/CMakeLists.txt`. Built as part of the
`usrp_wideband_signal_detection` container app.

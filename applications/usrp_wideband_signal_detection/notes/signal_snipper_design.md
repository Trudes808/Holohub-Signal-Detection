# signal_snipper + sigmf_file_sink

New downstream branch that cuts detected signals out of the wideband stream and writes them to disk
as SigMF. Added on branch `feature/signal_snipper_and_file_sink`.

## Data flow

```
chdrConverterOp.out0 ─┬────────────► fftOp ─► … ─► detectorOp.mask_out ─┐
                      │ (raw IQ tap)                                     │ (mask)
                      └────────────► signalSnipperOp ◄───────────────────┘
                                          │ snippets_out (SnippetBatchMessage)
                                          ▼
                                     sigmfFileSinkOp
```

Same two operators wire into the offline graph (`run_offline_cuda_detector_eval.cpp`):
`source.out → signalSnipperOp.iq_in`, `detector.mask_out → signalSnipperOp.mask_in`,
`signalSnipperOp.snippets_out → sigmfFileSinkOp.in`.

## Why a raw-IQ tap

The pipeline is frequency-domain from the FFT onward; the detector mask carries no time-domain IQ.
Raw IQ exists only between the CHDR converter (or offline source) and the FFT, so the snipper taps
that port directly and buffers frames in a device ring.

## IQ ↔ mask correlation

`fftOp` sets `fft_emitted_frame_number` = its per-input counter (fft.cu), which the detectors copy
into `DetectorMaskMessage.frame_number`. That counter equals the CHDR arrival index, so the snipper
counts its own IQ arrivals (1-based) and matches masks by `frame_number` directly — robust under
`emit_stride`. Offline, the mask also carries `file_offset_complex`, used for the absolute
`sample_start` when present.

**IQ drives compute** (not the mask): the IQ stream arrives one frame at a time in order and *leads*
the mask by the detector-pipeline latency, so buffering per IQ frame keeps the ring current and a
lagging mask always finds its (older) frame. Masks are drained opportunistically each compute
(`mask_in` is `kNone`); the offline eval's drain frames keep compute firing so trailing masks are
processed after the last real IQ. The ring is pruned by frame age (drop frames ≤ the last processed
mask frame), with `ring_depth` as a memory backstop. Driving on the mask instead lets the fast IQ
race ahead and evict frames before their mask arrives — the original bug.

## Modes

- **time_only**: merge component time-ranges into intervals; emit one full-band, full-rate snippet
  per interval, annotated with every detected signal's freq edges in that interval.
- **frequency**: per component, digital down-convert (mix to baseband, windowed-sinc low-pass to the
  detected bandwidth + `oversample_percent`, decimate to the minimum rate). One fused CUDA kernel
  (`ddc_kernel` in `signal_snip_core.cu`). Integer-bin baseband offset; block-channelizer accuracy.

## Downstream contract (the ragged-batch pattern)

Snipped signals have varying lengths and (freq mode) varying rates. `SnippetBatchMessage` carries a
`std::vector<SignalSnippet>` of self-describing descriptors over pooled device buffers
(`DeviceBufferPool`) — the TensorList / GstSample precedent. The file sink writes:
- **per_signal**: one `.sigmf-data`/`.sigmf-meta` per snippet.
- **pack**: accumulate N frames, group by sample rate → one concatenated recording per rate; a
  `.sigmf-collection` ties multiple rates together (SigMF-native heterogeneous-rate set).

SigMF is `cf32_le`, hand-rolled JSON (no JSON dep in this build), `wfgt:` namespace for custom fields
— matching what the offline eval already reads.

## Files

- `signal_snip_types.hpp` — `SignalSnippet`, `SnippetBatchMessage`, `DeviceBufferPool`.
- `signal_snip_core.{hpp,cu}` — CC labeling, box→physical mapping, DDC kernel, SigMF writers (pure,
  reusable by the operator and any batch tooling).
- `signal_snipper.{hpp,cu}` — the operator (IQ ring, mask correlation, both modes).
- `sigmf_file_sink.{hpp,cpp}` — the sink (per_signal / pack / collection). Compiled as CUDA.
- `config_signal_snipper_single_channel.yaml` — coherent-power single-channel config with
  `pipeline.enable_signal_snipper: true` plus `signal_snipper:` / `sigmf_file_sink:` blocks.

## Validation

Host-logic (CC boxing, box→physical geometry, SigMF-meta JSON) verified on host; full DDC + operator
wiring builds and runs only in the container.

- Offline (preferred first): enable the snipper in the config and run
  `python3 run_cuda_dino_offline_file.py <file.sigmf-data> --detector coherent_power --config config_signal_snipper_single_channel.yaml --output-root <dir>`;
  inspect `<output-root>/snippets/*.sigmf-*`. Checks: time-only IQ length matches the detected span;
  freq-mode `core:sample_rate` ≈ bandwidth·(1+oversample%); baseband-centered; mixed-rate pack emits
  a valid `.sigmf-collection`.
- Live/replay: `sudo ./bash_scripts/rebuild_demo_container_app.sh` then run with the new config.
  Files land under `/tmp/usrp_spectrograms/snippets` (host).

## Known follow-ups

- DDC is a block channelizer (rectangular frame boundaries); overlap-add is the fidelity upgrade.
- Signals spanning multiple detector frames are emitted as consecutive per-frame snippets; stitching
  is left to pack mode / downstream.
- Device IQ ring copies whole frames (~134 MB/frame at the bucket size); tune `ring_depth`.

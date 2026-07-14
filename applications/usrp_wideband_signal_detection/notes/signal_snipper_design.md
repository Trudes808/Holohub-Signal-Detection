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

Every annotation records both its slot in the recording (`core:sample_start`/`core:sample_count`)
and its original-stream provenance so a downstream ingestor of a packed recording can tell which
chunks belong together and reassemble a signal across frames: `wfgt:frame_number`,
`wfgt:orig_sample_start`, `wfgt:orig_sample_count`, `wfgt:orig_sample_end`, `wfgt:center_frequency`.
Pack mode writes **one file per pack**: if all snippets in the pack share one `(rate, center)` (e.g.
time-only) it's a standard concatenated SigMF recording; otherwise (frequency mode, every signal a
distinct rate/center) it's a **container** — all snippets concatenated into one `.sigmf-data`, each
annotation carrying its file offset + own `wfgt:snippet_sample_rate` / `wfgt:center_frequency` /
freq edges (global `core:sample_rate` is the original stream rate as a reference; `wfgt:container:
true`, `wfgt:layout: concatenated_variable_rate`). This trades strict SigMF conformance (one global
rate) for ~one file per pack instead of one per signal, which is what makes emit_stride 1 sustainable
(the per-signal file rate, ~900/s, was the real wall — not compute). A downstream ingestor slices
each annotation `[core:sample_start, +core:sample_count)` and reads it at `wfgt:snippet_sample_rate`.

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

## Parameter flexibility (sample rate / center frequency)

The pipeline derives everything from the stream's true rate/center; nothing is pinned per-config
that can't be overridden at launch:

- **Offline**: automatic from the SigMF (`core:sample_rate` → `explicit_span_hz`, `core:frequency`).
  A new capture at a different rate/center just works.
- **Live / loopback**: set `USRP_SAMPLE_RATE_HZ` / `USRP_CENTER_FREQ_HZ` at launch (forwarded into
  the container by `run_torchscript_performance_test.sh`). When set they feed the FFT bin derivation
  (same `explicit_span_hz` path as offline) and are pushed into the CHDR converter, which stamps
  `rx_sample_rate_hz` / `rx_center_frequency_hz` metadata that the detector, visualizer and snipper
  read. Unset → falls back to `chdr_converter.channel_sample_rates_hz` then nominal `fft.span`.
- Everything downstream re-derives from the rate: FFT size/`num_packets_per_fft`/resolution/span
  (main.cpp), the dynamic per-freq floor (auto-resizes to the runtime bin count), and the snipper's
  Hz↔sample mapping. So a new rate/center reconfigures the whole pipeline with no rebuild and no
  config edit — just relaunch with the env vars.
- `bash_scripts/sigmf_stream_params.py <file.sigmf-meta>` prints the two env assignments from a SigMF,
  so cable-loopback replay of a capture is hands-free:
  `eval "$(bash_scripts/sigmf_stream_params.py cap.sigmf-meta)"; sudo env USRP_SAMPLE_RATE_HZ=... USRP_CENTER_FREQ_HZ=... ./bash_scripts/run_torchscript_performance_test.sh <cfg>`.

### Fully-automatic handoff via a sidecar

Both stream producers write `/tmp/usrp_stream_params.json` (`{sample_rate_hz, center_freq_hz}`):
`applications/usrp_freq_detection/rx_to_remote_udp.py` writes the radio's *actual*
`usrp.get_rx_rate()`/`get_rx_freq()`; `replay_rx_to_buff.py` writes the replayed SigMF's rate/center.
`run_torchscript_performance_test.sh` reads the sidecar (unless `USRP_*_HZ` is already set), forwards
it into the container, and **consumes it (deletes it)** so it is a one-shot fresh handoff that never
contaminates a later run of a different config. So: start the sender/replay, then launch the app —
the true rate/center flow automatically, no config edit. Explicit `USRP_*_HZ` env always wins;
absent both, the config's `channel_sample_rates_hz` (else nominal `fft.span`) is used.

## Sink threading

`sigmf_file_sink` runs the device->host copy + file I/O on a **background writer thread**; `compute()`
only moves the received batch onto a bounded queue (`max_queued_batches`, drop-when-full). This keeps
`compute()` near-instant so it drains `snippets_out` immediately. Under the live event-based scheduler
a slow (in-compute) sink let the fast snipper overflow the output transmitter fatally
(`GXF_EXCEEDING_PREALLOCATED_SIZE`); a deeper `IOSize` did not help because the sink simply wasn't
scheduled between the snipper's rapid emits. Each queued batch pins its device IQ until written, so
the queue bound also caps extra device memory.

## Efficiency / real-time notes

- The snipper emits **exactly one** batch per `compute()` (all masks drained in a tick are merged),
  matching Holoscan's one-message-per-tick transmitter. Emitting per-mask overflows it fatally.
- DDC does the baseband mix **once per input sample** (`mix_kernel`) then a MAC-only decimating FIR
  (`fir_decim_kernel`); the old fused kernel recomputed `sincos` inside the FIR loop
  (num_taps sincos per output sample ~ ntaps× more transcendentals). The mix scratch is pooled.
- Connected-components (`label_components`) is allocation-free: reused `CcScratch` (visited + a
  flat-index FIFO) instead of a fresh 5 MB `visited` + `std::queue` per masked frame; the snipper
  also reuses one host mask buffer. Matters at emit_stride 1 (~47 masks/s over ~5M pixels).
- On a modest GPU (RTX 4000 Ada) the FFT+detector+visualization+snipper cannot sustain 245 MSps; the
  snip-focused profile sets `visualization.enable: false` and `emit_stride: 16` so the heavy DDC +
  host staging + file writes fire ~few/sec. `emit_stride` also cuts the file rate ~N-fold. Keep
  `ring_depth >= ~2*emit_stride + lag`.
- The file sink's device->host copy + writes run on a background thread (see Sink threading); it drops
  with a loud `OVERFLOW` (signals + IQ samples + original-rate samples) if it can't keep up.

## Known follow-ups

- DDC is a block channelizer (rectangular frame boundaries); overlap-add is the fidelity upgrade.
- Snipper still copies every ingested IQ frame into the ring even when `emit_stride`>1 skips most
  masks; skipping non-emitted frames would cut ring memory/copies ~N-fold (needs stride awareness).
- Signals spanning multiple detector frames are emitted as consecutive per-frame snippets; stitching
  is left to pack mode / downstream.
- Device IQ ring copies whole frames (~134 MB/frame at the bucket size); tune `ring_depth`.

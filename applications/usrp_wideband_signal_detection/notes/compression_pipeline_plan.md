# Real-time compression on the data-saved path (compression branch)

Decisions (2026-08-31, user-confirmed):
1. **Fidelity contract: decode must survive.** Lossy codecs are bounded so PN9 decode/chBER
   stays intact. Phase-destroying representations (spectrogram tiles, deep lossy) are allowed
   only on classify-only lanes, later.
2. **ML ingest: compressed-domain inference** (Phase 2): classifiers retrain to consume the
   quantized representation (BFP mantissas + exponent) directly — no dequantization on the
   classify path. Entropy-coded formats (nvCOMP/ZFP bitstreams) always decompress first; the
   classical decoder always needs reconstructed IQ (it is DSP, not ML). The Phase-2 retrain
   folds in the deferred burst-gating augmentation + SNR 0–45 range (user approved reversing
   the earlier retrain hold for this).
3. **First codecs: sc16, bfp12, bfp8 (own CUDA kernels) + nvCOMP LZ4/Zstd and ZFP fixed-rate**
   (pending container deps, task #38).
4. **Full-rate "compress everything" lane: computed baseline only** — the dashboard/eval just
   multiplies the naive full-capture GB by the measured codec ratio; no 2 GB/s lane runs.

## Architecture

```
any detector ──mask──► signal_snipper ──SnippetBatchMessage──► SnippetCompressionOp ──► sigmf_file_sink
 (coherent_power /      (time or freq                          (snippet_compression.cu)   (compressed SigMF
  cuda_dino / DINO-FT)   mode, ragged GPU batch)                codec: none|sc16|bfp12|bfp8  container)
                                                                            │
                                                                            ▼
                                                            rt_decode_daemon (numpy dequant mirror)
```

- The stage is an app-level Holoscan operator (like the snipper/sink), inserted when
  `pipeline.enable_snippet_compression: true`, configured by the `snippet_compression:` block,
  and wired identically in the live app (main.cpp) and the offline eval binary.
- `SignalSnippet`/`HostSnippet` carry optional codec fields; `n_iq` stays LOGICAL complex
  samples so all downstream duration/rate math is codec-agnostic.
- Compressed packs: `.sigmf-data` is a byte stream (`core:datatype: u8`,
  `wfgt:compressed_container: true`); per-annotation `wfgt:compression`,
  `wfgt:comp_byte_offset/count`, `wfgt:comp_scale` (sc16), `wfgt:comp_block` (bfp).
  `core:sample_start/count` remain logical. Codec `"none"` produces byte-identical legacy packs.
- Live codec switching without app restart: the operator polls the DEMO CONTROLS json's
  `"codec"` key (`control_json_path` param). Dashboard dropdown itself is parked (task #37).

## Codec formats

- **sc16**: per-snippet peak scale (float, `wfgt:comp_scale`), interleaved int16. 2.0x, ~96 dB.
- **bfp8/bfp12**: blocks of `bfp_block` (64) complex samples; per block one int8 power-of-two
  exponent then two's-complement mantissas (int8, or 12-bit packed 2-per-3-bytes).
  x_hat = m * 2^e. bfp12 = 2.65x (~66 dB, measured 66.0 dB median vs uncompressed reference);
  bfp8 = 3.97x (~42 dB).

## Validation

`infocom_evals/signal_detection_experiments/compression_rd/run_rd_eval.py` sweeps
codec x SNR through the compiled pipeline and reports ratio / whole BER / chBER /
per-model accuracy (results.csv). Gotchas encountered: a sized Holoscan input port gets NO
default scheduling condition (must attach MessageAvailableCondition explicitly, like the sink);
offline captures are ~4 frames so eval configs need `pack_frames: 1`; container-side writes are
root-owned (harness sudo-cleans scratch).

## Parked / next

- #37 dashboard codec dropdown + ratio in the footer economics line (operator+daemon already
  export everything needed: `comp_codec`, `comp_ratio` in rt_metrics.json).
- #38 nvCOMP (batched LZ4/Zstd over the ragged batch) + ZFP fixed-rate; aarch64/CUDA-13 builds
  into the container; python-side decode via lz4/zstandard/zfpy (verify nvCOMP frame
  compatibility first).
- #40 Phase-2 compressed-domain classifier retrain (quantize-in-the-loop on BFP mantissas).
- Stretch: learned latent codec whose latent doubles as classifier input.

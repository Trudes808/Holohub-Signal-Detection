# Real-time snippet compression — pipeline status & rate–distortion results

Branch `compression`, commit `6f11cc90` (2026-08-31). Hardware: DGX Spark (GB10, sm_121,
unified memory), demo container `usrp_x410_sig_det_sat3737`, CUDA 13 / torch 2.10.

## 1. What was built

A real-time GPU compression stage on the data-saved path, detector- and snip-mode-agnostic:

```
any detector ──mask──► signal_snipper ──ragged GPU batch──► SnippetCompressionOp ──► sigmf_file_sink
(coherent_power /                                            codec: none | sc16 |      (compressed SigMF
 cuda_dino / DINO-FT)                                        bfp12 | bfp8               container)
```

- **Operator** (`snippet_compression.{hpp,cu}`, app-level like the snipper/sink): consumes
  `SnippetBatchMessage`, compresses each snippet's device-resident cf32 payload in place,
  releases the fat cf32 buffer. Codec is a config param AND live-switchable through the
  demo-control JSON's `"codec"` key (no app restart); dashboard dropdown intentionally
  deferred (task #37).
- **Codecs (v1, fixed-rate quantizers, own CUDA kernels):**
  - `sc16` — per-snippet peak-scaled int16 I/Q. Exactly 2.0×; ~96 dB quantization SNR.
    (The radio wire format is sc16 — the cf32 files were 2× inflated from the start.)
  - `bfp12` / `bfp8` — O-RAN-style block floating point: 64 complex samples per block,
    one int8 power-of-two exponent + two's-complement mantissas (12-bit packed 2-per-3-bytes,
    or int8). 2.65× / 3.97×; ~66 dB / ~42 dB quantization floor.
- **Container format**: compressed packs store a byte stream (`core:datatype: u8`,
  `wfgt:compressed_container: true`); each annotation carries `wfgt:compression`,
  `wfgt:comp_byte_offset/count`, `wfgt:comp_scale`/`wfgt:comp_block`. Logical geometry
  (`core:sample_start/count`) stays in complex samples, so all downstream duration/rate/truth
  math is codec-agnostic. `codec: "none"` output is byte-identical to the legacy format.
- **Consumers**: the AMC decode daemon reads compressed packs via a numpy dequantizer that
  mirrors the CUDA kernels, and reports `comp_codec`/`comp_ratio` in `rt_metrics.json`.
  The same operator is wired into the offline eval binary, so offline sweeps exercise the
  real compiled kernels.

Implementation gotcha worth remembering: a **sized Holoscan input port gets no default
scheduling condition** — without an explicit `MessageAvailableCondition` the operator is never
scheduled and batches queue silently (the sink sets one; now the compressor does too).

## 2. Method

`run_rd_eval.py` (this directory): for each codec × SNR, run the compiled offline eval
(coherent_power detector, demo snipper config, `pack_frames: 1`) on the burst-structured
per-signal-referenced captures `snr_single_{30,20,12,6}db.sigmf-data` (245.76 MSps, 4-class
BPSK/4FSK/16QAM/OFDM, ~4 frames each), then run the decode daemon over the emitted packs and
harvest ratio, whole/attempted PN9 BER, oracle channel BER (chBER: known branch + known symbol
rate), and 3-model classification accuracy. Gate = T-PRIME.

Independently, reconstruction fidelity was measured by matching compressed-run snips to an
uncompressed run of the same capture by original-stream position and computing error SNR.

## 3. Results (16 runs)

| codec | SNR (dB) | ratio | stored (MB) | whole BER | chBER | Δ chBER vs none | acc T-PRIME |
|---|---|---|---|---|---|---|---|
| none | 30 | 1.00× | 87.2 | 3.79e-05 | 6.77e-03 | — | 0.94 |
| none | 20 | 1.00× | 70.4 | 1.91e-04 | 6.71e-03 | — | 0.56 |
| none | 12 | 1.00× | 64.0 | 6.75e-03 | 1.16e-01 | — | 0.75 |
| none | 6 | 1.00× | 59.6 | 6.39e-02 | 2.71e-01 | — | 0.44 |
| sc16 | 30 | 2.00× | 43.6 | 3.79e-05 | 6.77e-03 | +4.5e-07 | 0.94 |
| sc16 | 20 | 2.00× | 35.2 | 1.92e-04 | 6.71e-03 | −4.5e-07 | 0.56 |
| sc16 | 12 | 2.00× | 32.0 | 6.75e-03 | 1.16e-01 | −1.4e-06 | 0.75 |
| sc16 | 6 | 2.00× | 29.8 | 6.39e-02 | 2.71e-01 | +1.6e-06 | 0.44 |
| bfp12 | 30 | 2.65× | 32.9 | 3.92e-05 | 6.77e-03 | +6.8e-07 | 0.94 |
| bfp12 | 20 | 2.65× | 26.6 | 1.92e-04 | 6.71e-03 | −2.0e-06 | 0.56 |
| bfp12 | 12 | 2.65× | 24.1 | 6.75e-03 | 1.16e-01 | +2.7e-06 | 0.75 |
| bfp12 | 6 | 2.65× | 22.5 | 6.38e-02 | 2.71e-01 | +3.2e-06 | 0.44 |
| bfp8 | 30 | 3.97× | 22.0 | 3.54e-05 | 6.78e-03 | +1.0e-05 | 0.94 |
| bfp8 | 20 | 3.97× | 17.8 | 1.90e-04 | 6.52e-03 | −2.0e-04 | 0.56 |
| bfp8 | 12 | 3.97× | 16.1 | 6.72e-03 | 1.15e-01 | −9.2e-04 | 0.75 |
| bfp8 | 6 | 3.97× | 15.0 | 6.39e-02 | 2.69e-01 | −1.7e-03 | 0.44 |

Reconstruction fidelity (bfp12, 20 dB, 16/16 boxes matched against the uncompressed
reference): **66.0 dB median error SNR, 65.2 dB min** — exactly the theoretical 12-bit
block-floating-point floor, confirming the CUDA encoder ↔ numpy decoder round trip.

## 4. Findings

1. **Every v1 codec is decode-transparent at every rung tested (6–30 dB).** wBER, chBER,
   and all three classifiers' accuracy match the uncompressed baseline within run noise
   (accuracy columns are bit-identical per SNR — the classifiers make literally the same
   decisions). Physics: channel noise dominates even bfp8's ~42 dB quantization floor by
   ≥12 dB at these signal SNRs, so the quantizer is invisible to modem and models alike.
   (bfp8's slightly *negative* chBER deltas are noise/dither, not improvement.)
   The identical accuracy columns were verified, not assumed: re-running T-PRIME on matched
   raw-vs-dequantized band pairs (16 bands, 20 dB) shows the softmax genuinely moves
   (median max|Δp| ≈ 1e-4; worst case 0.0025 for bfp12, 0.023 for bfp8) with **zero argmax
   flips** — the model consumed the codec-altered data and made the same decisions.
   NOTE: this whole eval is decompress-then-ingest — the models classify the dequantized
   reconstruction. Compressed-DOMAIN inference (mantissa-domain inputs, no dequantization)
   is Phase 2 (#40) and requires retraining.
2. **Ratios land exactly on theory and are content-independent** (fixed-rate codecs):
   2.00× / 2.65× / 3.97×.
3. **Economics**: at 20 dB the same detections shrink 70.4 → 17.8 MB (bfp8). Stacked on
   snipping's ~62× vs full-rate capture, the pipeline stores **~250× less** than naive
   capture with zero task degradation.
4. **The rate–distortion knee was not reached.** bfp8 is a safe default at demo SNRs.
   Finding the knee needs either a finer rate knob (ZFP bits/sample, task #38), captures at
   ≥40 dB (the 50 dB "clean" rung is where bfp8's floor first becomes visible), or more
   aggressive codecs.

## 4b. ZFP knee sweep (2026-09-01): the knee found

`zfp_knee_sweep.py`: the SAME detected snips (uncompressed offline runs) re-encoded with zfp
fixed-rate (planar I/Q, zfpy — bitstream-deterministic, so identical to what the CUDA backend
would produce) at 12/10/8/6/5/4/3 bits per scalar (2.67–10.7× vs cf32), across
clean(50)/30/20/12 dB captures; daemon decode+classify over the reconstructions. 28 points:
`zfp_knee_results.csv`, plot `zfp_knee.png`. (zfpy 1.0.1 crashes below rate 3 on aarch64.)

Reconstruction SNR is content-independent (~6 dB/bit): rate 12→~50 dB, 10→~38, 8→~26,
6→~14, 5→~8, 4→~2, 3→~−3 dB.

**Finding 1 — the decode knee scales with signal SNR.** Last transparent ratio → first
broken ratio (chBER vs its own uncompressed baseline):

| capture | decode transparent through | knee (first damage) | collapse |
|---|---|---|---|
| clean (50 dB) | 5.33× (rate 6) | 6.4× (chBER → 1.0) | ≥6.4× |
| 30 dB | 4.0× (rate 8) | 5.33× (6.8e-3 → 8.9e-2) | ≥6.4× |
| 20 dB | 3.2× (rate 10) | 4.0× (6.7e-3 → 8.9e-2) | ≥6.4× |
| 12 dB | 3.2× (rate 10) | 4.0× (0.116 → 0.365) | ≥10.7× |

**Finding 2 — the classification knee sits FAR beyond the decode knee.** T-PRIME accuracy is
unchanged through 5.33–6.4× at every rung where decode has already degraded or collapsed
(e.g. 20 dB: accuracy identical at 6.4× where chBER is 0.31), and only breaks toward 8–10.7×
(reconstruction ≤2 dB). Classify-only consumers can ride roughly **2× deeper compression**
than decode-grade storage — the concrete justification for the planned two-lane split
(decode-grade to disk, aggressive lane for ML ingest). Low-rate accuracy points are noisy
(classifying near-garbage is prior-biased; small N) — read trends, not single cells.

**Finding 3 — BFP beats zfp by ~16 dB at matched ratio on RF IQ.** bfp12 (2.65×) reconstructs
at 66 dB vs zfp rate-12 (2.67×) at ~50 dB; bfp8 (3.97×) at ~42 dB vs zfp rate-8 (4.0×) at
~26 dB. Consequence at 20 dB: bfp8 is transparent at 4.0× while zfp is already 13× worse in
chBER at the same ratio. zfp's decorrelating transform assumes smooth fields; near-critically
sampled IQ is noise-like, so plain block floating point is the better family per bit.

**Decision guidance:** do NOT integrate zfp CUDA as a production codec — it served its
purpose as the rate knob that located the knee. The production ladder should instead extend
BFP downward (bfp6 ≈ 5.2×/~30 dB floor, bfp4 ≈ 7.5×/~18 dB) to walk the knee with the
better-per-bit family, plus the nvCOMP entropy stage for free lossless gains. Predicted from
the recon-SNR mapping: bfp6 transparent at 20–30 dB rungs, bfp4 classification-only.

## 5. Caveats

- Accuracy columns are small-N (each capture is ~0.1 s ≈ 4 frames; e.g. 20 dB reads 0.56 while
  live runs read ~0.88) — read them codec-relative, not absolute.
- Ratios are pure quantization; the lossless entropy stage (nvCOMP LZ4/Zstd over sc16/BFP
  payloads) is not yet integrated, so total ratios should improve another ~1.1–1.5× "for free".
- Whole BER includes classifier-routing effects (T-PRIME gate); chBER is the codec-sensitive
  column by construction.
- Fixed loopback/offline channel; OTA revalidation is future work.

## 6. Next steps (tracked)

- #38 nvCOMP (batched LZ4/Zstd) + ZFP fixed-rate — container deps, then rerun this sweep.
- #40 Phase 2: compressed-domain classifier retrain (models ingest BFP mantissas directly;
  folds in the deferred burst-gating + SNR 0–45 augmentation).
- #37 (parked) dashboard codec dropdown + ratio in the footer economics line — the operator
  and daemon already export everything needed.
- Richer corpus: the original MATLAB composite (5G/802.11ax/BT/FM variety) replays at
  245.76 MSps and is a better ratio-statistics corpus; its pcap is already generated.

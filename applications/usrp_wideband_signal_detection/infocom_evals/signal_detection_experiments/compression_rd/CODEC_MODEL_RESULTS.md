# Codec-per-model experiment — compressed-domain T-PRIME on the original MATLAB composite

Branch `compression`, 2026-09-01. Code: `amc/matlab_ds.py`, `amc/train_matlab.py`
(training), `make_noisy_composites.py`, `eval_codec_models.py` (eval). Raw outputs:
`codec_model_matrix.csv` (5,312 cells), `detection_recall.csv`.

## Design

- **Models**: four architecturally identical compressed-domain T-PRIME classifiers
  (`TPrimeC`: 256 mantissa-normalized I/Q reals + 2 window-relative block exponents per
  token), trained per storage codec: `none` (float control) / `sc16` / `bfp12` / `bfp8`.
  BFP models consume the stored integers as-is — no dequantization anywhere.
- **Classes (10)**: BPSK, QPSK, 16QAM, OFDM, 5G_Downlink, 802_11ax, Bluetooth,
  Narrowband_FM, Broadband_FM, NOISE (pure AWGN windows, class-balanced).
- **Training**: PAIRED — one seeded 51k-window set from `generated_waveforms_24576`
  (windows mimic stored snips: snipper decimation law, AWGN U(−10, 30) dB in-band, phase/CFO
  jitter), featurized four ways. So codec deltas carry no sampling variance.
- **Eval**: original composite `comprehensive_ordered` + AWGN variants (nominal 30/15/0/−10 dB;
  every annotation stamped with its ACTUAL in-band SNR). Two detection lanes —
  `gt` (annotation-true boxes, sliced/mixed/decimated by the snipper's law) and
  `dino` (DINO-FT through the real pipeline, exact offline truth join). Then the full
  4×4 storage-codec × model matrix with transcode semantics and median-softmax
  aggregation over block-aligned 8,192-sample windows.

## Result 1 — the codec effect is NULL end-to-end (the experiment's primary question)

Validation (paired, in-distribution): none 0.874, sc16 0.882, bfp12 0.882, bfp8 0.880.

4×4 matrix, all variants pooled (rows = storage codec, cols = model):

| gt lane | none | sc16 | bfp12 | bfp8 |
|---|---|---|---|---|
| none | .479 | .469 | .479 | .474 |
| sc16 | .479 | .469 | .479 | .474 |
| bfp12 | .479 | .469 | .479 | .474 |
| bfp8 | .481 | .468 | .479 | .474 |

Every row is identical to three decimals: storing under sc16/bfp12/bfp8 changes *nothing*,
matched-vs-transcoded is free, and the per-codec models are interchangeable (spread across
models ≈ 0.01, within training noise). The same holds in the dino lane at its (much lower)
level. **Conclusion: through 4× quantization, compressed-domain classification needs neither
per-codec models nor dequantization — one model covers the ladder.** This extends the RD-sweep
finding (task metrics codec-invariant) from decompress-then-ingest to true mantissa-domain
ingest, now on 10-way protocol ID over standards traffic.

## Result 2 — ground-truth-box accuracy: the composite context costs ~0.3

GT-lane matched accuracy: 0.56–0.57 at clean/30/15 nominal, 0.41 at 0, 0.27 at −10 —
versus 0.88 validation. Per class at actual SNR ≥ 0 dB (matched bfp8):

| class | acc | class | acc |
|---|---|---|---|
| 802_11ax | **0.91** | OFDM | 0.58 |
| Broadband_FM | **0.94** | Bluetooth | 0.59 |
| Narrowband_FM | 1.00 (n=10) | BPSK | 0.48 |
| 5G_Downlink | 0.61 | QPSK / 16QAM | 0.46 / 0.47 |

The gap vs validation is a *context* gap, not a codec gap (codecs identical): real composite
placements bring adjacent-channel neighbors inside the keep bandwidth, per-record power
differences, and the clean variant sits ABOVE the trained SNR range (accuracy at "clean" bins
0.48 < the 0.55–0.67 of the +5…+40 dB bins). Single-carrier classes confuse among themselves
at 10-way granularity (BPSK/QPSK/16QAM each ~0.46–0.48), while the structurally distinctive
classes (11ax, FM) transfer well. Accuracy vs actual per-signal SNR rises monotonically from
0.10 at −20 dB to ~0.65 at +5…+40 (bin composition varies by class, so local dips are
mixture effects, not physics).

## Result 3 — the DINO-FT lane exposes two real deployment gaps (acc 0.135 on signal boxes)

The dino lane collapses: BPSK/QPSK/16QAM/Bluetooth ≈ 0.00, 5G 0.27, OFDM 0.45, 802_11ax 0.71.
Two mechanisms, both diagnosable and fixable:

1. **Rate mismatch (dominant for single-carrier).** Training/GT windows are decimated by the
   *true occupied bandwidth*; the real snipper decimates by the *detected* box bandwidth. The
   detected-box rate differs enough to change samples-per-symbol, and the compressed-domain
   T-PRIME — which consumes raw sample tokens and CANNOT be resampled without leaving the
   mantissa domain — is rate-brittle. Wideband classes (11ax at full rate, OFDM-family) are
   least affected, exactly as observed.
2. **The noise-class distribution gap (predicted in advance).** False-positive boxes dominate
   the dino lane — 32% of snips at clean rising to 97% at −10 nominal — and the models,
   trained on *pure AWGN* noise, recognize only 7% of these real false boxes (they contain
   signal edges, splatter, and neighbor leakage, not clean noise).

Detection recall itself (secondary result, `detection_recall.csv`): DINO-FT finds every class
at ≥0 dB — including 5G, 802.11ax, and Bluetooth it never trained on — with over-segmentation
(matched boxes > annotations); Narrowband_FM (30 kHz, sub-bin at 240 kHz/bin) is chronically
under-detected; everything collapses at −10 dB nominal (detection-limited regime).

## Remedies (next iteration, harness is one command per step)

1. **Rate-diversity augmentation**: draw the training decimation from the detector's actual
   keep-bandwidth distribution (scale true occ by ~U(1.0, 2.5)) instead of the exact occ law.
2. **Real noise-class data**: harvest DINO-FT false-positive boxes (truth-join = NOISE) from a
   composite run as noise-class training windows, replacing/augmenting pure AWGN.
3. Retrain (paired, same seeds) and rerun `eval_codec_models.py` — the codec-null result is
   expected to persist; the dino-lane floor should move toward the gt-lane ceiling.

## Caveats

- Train/eval share library records (the composite is built from them), so results measure
  pipeline/context generalization, not unseen-waveform generalization. Note this makes the
  observed domain gap a LOWER bound.
- GT lane has no NOISE-truth boxes by construction; dino-lane NOISE dominates pooled numbers —
  read signal-class rows, not pooled totals, for classifier quality.
- 4 windows per snip, median softmax; clean variant lies outside the trained SNR range.

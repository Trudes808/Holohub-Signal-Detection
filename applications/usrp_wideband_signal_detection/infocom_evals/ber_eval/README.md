# ber_eval — bit-error-rate evaluation of detector-saved waveforms

Measures **BER vs SNR** for the USRP wideband detectors by *decoding the IQ each
detector actually saved* and comparing to the known transmitted bits. The point:
show whether lossy **detect → snip** (data saving) preserves **decodability**, per
modulation class, for **ground_truth / coherent_power / finetuned_dino_m2**.

Branch: `BER_eval`. Uses MATLAB R2025b (Comms/Signal/DSP/5G/LTE/WLAN/Bluetooth +
the `helperOFDM*` example). Run headless: `matlab -batch "..."`.

> **Status (2026-07-29): SNR SWEEP COMPLETE** — 17 attenuation levels (0–80 dB,
> skipping the `*_v2` recaptures) × 3 detectors = **51/51 detector-levels**.
> Headline: **dino_ft decodes essentially as well as perfect genie extraction
> across ~50 dB of SNR (54 → +4 dB), while coherent_power runs 2–3× worse than
> genie over the same span.** Below −1 dB dino collapses (real, verified — see
> "dino's deep-noise collapse"). Full curves: `results/ber_sweep_*.{csv,png}`.

## Quickstart
```matlab
% one capture, all three detectors -> per-class CSV + figure under results/
ber_eval_all('attenuation_dB_0')

% one detector
r = ber_eval_run('attenuation_dB_0','coherent_power');

% aggregate a finished sweep -> BER-vs-SNR tables + figures
ber_sweep_combine
```
```bash
# full sweep for one detector (gen -> eval -> delete that level's snippets).
# Evals run in a bounded parallel pool; snippet gen stays serial (one GPU/container job).
LEVELS="5 10 ... 80" MAXEVAL=6 BER_THREADS=5 bash run_ber_sweep.sh coherent_power
bash ber_status.sh          # live progress dashboard (reads results/, crash-proof)
```
Snippets are the frequency-mode `--outputs iq` output of `soft_label_pipeline.py`
under `/tmp/usrp_spectrograms/ber_eval/<detector>/iq/<stem>/snippets`. Ground-truth
reads the capture directly; the detectors read only their (small) snippet files.

`/tmp` is volatile — regenerate snippets if wiped (from `../snip_eval/soft_label_pipeline/`,
conda env `dinov3`, demo container running):
```bash
python soft_label_pipeline.py --waveform-dir /home/bqn82/captures \
    --glob "attenuation_dB_0.sigmf-data" --detector coherent_power --outputs iq \
    --mode frequency --min-box-pixels 256 \
    --mask-root /tmp/usrp_spectrograms/all_detectors \
    --output-root /tmp/usrp_spectrograms/ber_eval/coherent_power
# dino: --detector dino_finetuned, drop --mask-root (regenerates masks on GPU),
#       --output-root .../ber_eval/finetuned_dino_m2
```

## Method
Per ground-truth *data* waveform (`wfgt:kind=="waveform"`, digital classes only):
1. **Match** a detector snippet where the **signal center `(freq_lo+freq_hi)/2` ∈
   detection box band** AND time overlap ≥ `TimeOverlapMin` (default 0.10). No match → `miss` (BER 1).
2. **ground_truth** = genie: extract the signal's exact slot straight from the
   capture at its known center (perfect detection, full bandwidth).
3. **Recenter** to the known center, **isolate** the occupied band (reject
   neighbors), **data-aided CFO** correct vs the known `f_sig`, decode with
   `decode_waveforms_24576.m`. Single-carrier uses a **data-aided MMSE equalizer**.
4. Aggregate **bit-weighted BER per class**; `insufficient` (see slotting) excluded.

## Decode fixes (the load-bearing details)
- **Recenter to the true signal center** `(freq_lo+freq_hi)/2`, NOT
  `wfgt:block_center_hz` (a block holds several signals side-by-side).
- **Match on signal-center ∈ box** (not box-center ∈ signal band): coherent merges
  adjacent signals into one wide box centred in the gap.
- **Data-aided CFO** vs the known TX waveform (`estimate_cfo_da`): residual carrier
  offset (~±2 kHz hardware + edge quantization) wrecks narrowband SC / OFDM / 5G /
  GFSK. Decimate the reference to a rate that **preserves the occupied bandwidth**
  (a fixed 1.92 MHz aliased ≥2 MHz signals and broke uncoded BLE le1m/le2m).
- **BER-refinement**: if a decode still fails (BER>0.05), a small BER-minimizing
  carrier search rescues it (GFSK correlation peaks are broad → the DA estimate can
  mislock). Only fires on failing signals.
- **MMSE equalizer** on single-carrier (`decode_waveforms_24576.m`, gated by
  `metadata.equalizer=="mmse"`) — the stock 1-tap gain can't undo wideband ISI.
- **Slot-exact extraction**: read exactly `[gStart,gEnd)`; reading past the slot
  pulled in adjacent slots and inflated BER.
- **Normalize snippet box freqs to baseband** (the sweep-blocking fix): the
  captures are **inconsistent** in their declared RF center —
  `captures[0].core:frequency` is **0** for {0,5,10,15,30,45,50,55,60} and
  **2 GHz** for {20,25,35,40,65,70,75,80} — while GT annotations are *always*
  baseband. `snip_annotations.py` labels each box `capture_center + offset`, so on
  the 2 GHz captures every box lands ~1.9 GHz and **nothing matches → BER 1.0,
  nDecoded 0 for the whole level**. `read_capture_center` + `index_snippets(root,
  capCF)` subtract the capture center (no-op when it's 0, so atten_0 is unchanged).
  The snippet IQ was always fine — only the labels were absolute. Verified 0/1112
  → 975/1112 matched at atten_20. **Hid because atten_0 happens to be a 0-center
  capture.** Affects any eval that matches snippet boxes to GT, not just BER.
- **Adaptive CFO refine** (the runtime fix): the old refine ran a fixed 33-point
  (±8 kHz @ 500 Hz) grid of *full standards decodes* on every failing signal.
  Fine at high SNR (little fails), catastrophic at low SNR where ~everything fails
  but nothing is recoverable — it dominated sweep runtime (one level ≈ 53 min).
  Now: 9-point coarse pass, then a ±1500 Hz @ 500 Hz local pass **only if the
  coarse best is recoverable** (BER ≤ 0.35), plus early-exit on a clean lock
  (< 0.02). Hopeless signals stop after 9 decodes. Validated faithful — overall
  BER Δ < 0.1% (L35 0.28657→0.28658, L40 0.34975→0.35006), per-class identical to
  ~4 s.f. for 6/7 classes; only Bluetooth (GFSK, the class refine actually
  rescues) shifts, 0.006→0.013 at L35. **All three detectors use the same refine**
  so the comparison has no harness confound. Coherent's 16 levels: ~58 min total.
  The per-piece fallback is likewise gated to recoverable BER (0.05, 0.40].
- **No time guard on the snippet trim** (the Bluetooth fix): waveforms tile from
  `gStart`, so `gStart` IS a packet boundary — and the sync-free ideal receivers
  (BT BR/EDR + LE) assume the waveform starts at sample 0. A ±2% "sync guard"
  prepended pre-slot content and misaligned every packet → BT ~0.8 while every
  guard-free path (genie, dev trim) decoded ~0. Exact-slot trim fixed BT
  (0.77→0.0029) and slightly improved all other classes. Classes with real
  timing recovery (SC correlation, WLAN packet-detect, 5G DM-RS, OFDM search)
  never noticed the guard — which is why it hid so long.

## Slotting & the "data lost / undecodable" accounting
Every waveform is transmitted at **6 slot durations** — 0.04, 0.2, **1, 5, 10,
20 ms** (287 of each) — and *tiles/repeats* to fill its slot (verified by a flat
power envelope). So a full slot is always decodable. Annotations **< 1 ms** (the
0.04 & 0.2 ms slots, exactly 1/3 of all signals) are capture fragments too short to
sync/decode → marked **`insufficient`** and **excluded** (data lost, undecodable by
any detector — not a decode failure). This is the expected slot-truncation loss.

## Figures
`python plot_ber_figs.py` (env `dinov3`) regenerates every figure from the result
CSVs. BER is plotted in **percent** ("BER %"), and detector colors/markers follow
the repo-wide `DETECTOR_STYLE` (coherent `#1f77b4` ○, dino_ft `#8c564b` ✚) with
ground truth as the neutral grey baseline, so these match the other evals.

**The headline claim — "snipping is minimally destructive vs what the channel
already did":**
- `fig_snip_vs_channel.png` — per detector, channel-only vs after-snipping on the
  **same signals**. The curves nearly coincide. Channel: **1.9% → 49% (+47 pts)**;
  snipping adds **at most +3.1 pts (coherent, median +0.7)** and **+2.1 pts
  (dino, median +0.3)**.
- `fig_snip_excess.png` — that cost isolated: excess BER in percentage points,
  hugging zero across the whole sweep.
- `fig_snip_vs_channel_byclass.png` — the same claim per modulation class.

These use a **matched subset**: each detector is compared against the genie
evaluated on *exactly the signals that detector saved*. That is what isolates
"what the snip did" from "what the detector missed" — without it, survivorship
bias flatters whichever detector missed the weak signals (see the selection-bias
section). Note each per-class panel therefore carries **one grey reference per
detector**; the two separate only where the detectors saved different sets.

**Full-sweep reference:** `ber_sweep_overall.png`, `ber_sweep_byclass.png` — the
overall metric, where an unsaved signal counts as 100%. Use these for the
detector comparison (coverage included), and the claim figures for snip fidelity.

## Results: BER vs SNR (the sweep)
Overall bit-weighted BER (`results/ber_sweep_overall.csv`, figure
`ber_sweep_overall.png`; SNR ≈ 54 − attenuation, per `snip_eval/snr_calibration.json`):

| atten | SNR | ground_truth | coherent_power | dino_ft_m2 |
|---|---|---|---|---|
| 0 | 54 | 0.0188 | 0.0243 | **0.0224** |
| 5 | 49 | 0.0200 | 0.0282 | **0.0301** |
| 10 | 44 | 0.0191 | 0.0409 | **0.0244** |
| 15 | 39 | 0.0214 | 0.0626 | **0.0250** |
| 20 | 34 | 0.0492 | 0.1474 | **0.0598** |
| 25 | 29 | 0.1438 | 0.2790 | **0.1514** |
| 30 | 24 | 0.2007 | 0.4233 | **0.2086** |
| 35 | 19 | 0.2866 | 0.6476 | **0.2891** |
| 40 | 14 | 0.3501 | 0.7194 | **0.3513** |
| 45 | 9 | 0.4026 | 0.7697 | **0.4049** |
| 50 | 4 | 0.4390 | 0.7915 | **0.4474** |
| 55 | −1 | 0.4668 | 0.8257 | 0.5688 |
| 60 | −6 | 0.5119 | 0.8519 | 0.7886 |
| 65 | −11 | 0.5430 | 0.8363 | 0.9994 |
| 70 | −16 | 0.5457 | 0.8473 | 0.9995 |
| 75 | −21 | 0.5445 | 0.8622 | 1.0000 |
| 80 | −26 | 0.5477 | 0.8670 | 0.9924 |

**Three regimes:**
1. **54 → +4 dB SNR (11 of 17 levels): dino ≈ genie.** Within 1–2% *relative* of
   ground truth the whole way (0.4474 vs 0.4390 at +4 dB). Data-saving costs
   essentially nothing in decodability.
2. **Same span: coherent is 2–3× worse than genie** and the gap opens as SNR
   falls (0.0626 vs 0.0214 at 39 dB; 0.6476 vs 0.2866 at 19 dB). Per class the
   overall-BER cost is large — at 39 dB **BPSK is 0.194 (coherent) vs 0.0010
   (genie/dino)** — but note **this is a miss-rate effect, not snippet quality**:
   its detect rate falls 1.00 → 0.23 while dino holds 1.00, and missed bits score
   1.0. On the signals it *does* save, coherent decodes ≈ genie (see below).
3. **Below −1 dB dino inverts and collapses** (0.999+ at −11 dB and below);
   coherent plateaus ~0.85. Real, not a harness artifact — but read the
   "nothing is genuinely decodable below ~0 dB" caveat before interpreting it.

GT itself saturates at **~0.546** (not 0.5) from −11 dB on: the floor once
nothing is decodable. Per-class detail for every level: `ber_sweep_byclass.csv`,
figure `ber_sweep_byclass.png`.

### Why coherent loses so much — it's MISSES, not snippet quality
Decompose overall BER into bits from **missed** signals (scored 1.0 by
definition) vs bits from **decoded** signals, and the mechanism is unambiguous:

| SNR | coherent overall | from misses | from decoded | miss share of bits |
|---|---|---|---|---|
| 39 dB | 0.0626 | 0.0357 | 0.0269 | 3.6% |
| 24 dB | 0.4201 | 0.2674 | 0.1527 | 26.7% |
| 9 dB | 0.7694 | **0.6366** | 0.1328 | **63.7%** |
| −26 dB | 0.8616 | 0.7289 | 0.1327 | 72.9% |

**BER on decoded signals only** (i.e. quality of the IQ each detector actually
saved) is ≈ genie at every SNR — coherent is sometimes *better*:

| SNR | GT | coherent | dino |
|---|---|---|---|
| 39 dB | 0.0214 | 0.0279 | 0.0242 |
| 24 dB | 0.2007 | 0.2084 | 0.2078 |
| 9 dB | 0.4026 | **0.3654** | 0.4042 |

So **what gets saved decodes as well as perfect extraction, for both detectors**
(the snip DDC band-limits each snippet, which can even shave out-of-band noise).
The entire detector gap is **detection coverage**: coherent's detect rate falls
1.00 → 0.23 while dino holds 1.00 down to +4 dB. Coherent's ~19×-too-wide boxes
(IoU 0.48 vs dino 0.95) therefore cost **storage**, ~19× more spectrum per
detection, rather than decodability. Do NOT attribute coherent's overall-BER gap
to noise in its wide snippets — the decoded-only column refutes that.

### Why coherent "beats" dino below −10 dB: the miss=1.0 convention, not detection
The spec scores an undetected signal as **BER 1.0** ("no saved signal → 100%
BER"). Below ~−10 dB nothing is decodable, so a *saved* signal yields coin-flip
bits (~0.49) while an *unsaved* one costs 1.0 — i.e. **saving spectrum you cannot
decode still earns 0.49 instead of 1.0.** Coherent's block-quantized ~19×
oversized boxes blanket spectrum regardless of content, incidentally covering
~27% of bits; dino correctly declines to fire and takes 1.0 on 98.5% of bits.
Re-score misses as a random guess (0.5) and **the ranking inverts**:

| SNR | miss=1.0 COH/DINO | winner | miss=0.5 COH/DINO | winner | decoded-only COH/DINO |
|---|---|---|---|---|---|
| −1 | 0.826 / 0.569 | dino | 0.484 / 0.505 | coh | 0.449 / 0.464 |
| −11 | 0.836 / 0.999 | coh | 0.506 / 0.500 | dino | 0.483 / 0.451 |
| −16 | 0.847 / 1.000 | coh | 0.513 / 0.500 | dino | 0.487 / 0.475 |
| −26 | 0.867 / 0.992 | coh | 0.517 / 0.501 | dino | 0.490 / 0.496 |

Under the guess convention they are statistically identical (0.517 vs 0.501).
**Do not read coherent's deep-noise plateau as better detection** — it is a
metric that rewards indiscriminate hoarding and penalises correct abstention.
Both conventions are defensible; the reported curves use miss=1.0 per the spec.

### Detectors sometimes score below genie — selection bias, plus real noise-shaving
Two distinct causes, both verified:
1. **At low SNR: survivorship bias** (the dominant one). A detector is graded
   only on the signals it detected — the *strong* ones. Restricting the genie to
   the **same signal subset** shows genie is still better:

   | SNR | coherent dec-only | genie on same signals | genie on all |
   |---|---|---|---|
   | 9 dB | 0.3654 | **0.3569** | 0.4026 |
   | −6 dB | 0.4704 | **0.4669** | 0.4792 |

   So **on matched subsets the genie bound is never violated** in aggregate.
2. **At atten_0 (detect rate 1.00, no selection possible) a few per-class wins
   are real**, and they split by signal type: **narrowband single-carrier gains**
   (QPSK 0.0027/0.0026 vs genie 0.0035; 16QAM 0.0138/0.0146 vs 0.0184) while
   **wideband OFDM-family loses** (OFDM 0.0275/0.0265 vs 0.0238; 5G, 802.11ax
   similar). Mechanism: the snipper's DDC lowpasses to the box bandwidth and
   decimates *at the full 245.76 MHz rate*, upstream of the harness's own
   `isolate_band`. For a narrowband signal in a wide native window that cascaded
   filtering rejects noise the genie's single-stage path retains; for OFDM the box
   hugs the occupied band so the extra transition band clips subcarrier edges.

When comparing detectors, prefer **matched-subset** comparisons (or report detect
rate alongside BER) so coverage and snippet quality stay separable.

### Caveat: below ~0 dB SNR nothing is genuinely decodable (by anyone)
`status=="decoded"` only means the decoder ran without throwing, not that it
recovered the bits. Decoded-only BER at −26 dB is **0.4935 (GT) / 0.4896
(coherent) / 0.4961 (dino)** — a coin flip. GT's own overall saturates ~0.55.
So the deep-noise end of every curve measures **how much spectrum a detector
happened to save**, not decodability; only the ≳ +9 dB region carries real
decodability information. (A third status, `decode_err` — a genuine decoder
exception, scored 1.0 — also grows at depth: 6.3% of GT bits at −6 dB, 10.7% at
−26 dB, which is why GT's overall exceeds its decoded-only.)

### dino's deep-noise collapse (verified real, ≤ −11 dB)
At −16 dB dino emits ~1012 boxes but decodes **3 / 1112** signals. Verified by
regenerating attenuation_dB_70 and measuring box geometry directly
(`dev_*`/one-off check, breadcrumbs in `results/verify_dino_deep.log`):
- Only **36/1112 (3.2%)** GT signals get *any* time+frequency overlap from a dino box.
- Boxes are ~**50 MHz wide** (median 49.9) — dino stops hugging occupied bandwidth
  and emits big blobs — but they are **sparse in time: median 0.29 ms** (vs 1–20 ms
  slots), covering **<1% of the time-frequency plane**.
- It collapses **wideband-first**: at −6 dB detect rate is 0.09 for 802.11ax and
  0.39 for 5G, while narrowband survives (Bluetooth 1.00, BPSK 0.87).
So dino genuinely saves almost nothing that deep — it fires on brief moments where
something peeks above the noise. Coherent's crude oversized blocks keep bracketing
signals by sheer coverage, so it plateaus instead of collapsing. Two hypotheses
were **ruled out**: (a) boxes clustering on the 48 MHz RX clock spur — they don't;
(b) a scoring artifact from per-piece time-overlap — aggregating overlap across
pieces (union) moves −16 dB from 3 → 4 matches, i.e. ≤0.1%.

## Performance
`ber_precompute.m` builds `wave_cache.mat` (67 MB): per-variation `md`, `txBits`,
`refd`, ... shared by every run so no run reloads the 287×15 MB `.mat` files.
`ber_eval_all` builds it once. Verified identical results (GT 0.018823).

## Running a sweep (operational notes)
- `run_ber_sweep.sh <detector>` does gen → eval → **delete that level's snippets**.
  Necessary: one level is ~7–25 GB of snippets, so 17 levels × 3 detectors would
  blow past /tmp. Only the (small) result CSVs persist. Nothing is written to /home.
- **Gen is serial** (one container/GPU job), **evals run in a bounded pool**
  (`MAXEVAL`, `BER_THREADS`) since levels are independent and eval is the CPU
  bottleneck. Each eval writes only its own per-level files, so no CSV races.
- **Idempotent resume**: a level whose snippets are already on disk skips gen and
  goes straight to eval, so an interrupted sweep resumes without redoing GPU work.
- **Launch detached** (`setsid nohup … </dev/null &`) — a plain background job dies
  with the terminal/session; `setsid` survived a mid-sweep session crash here.
- Masks: coherent + cuda_dino masks pre-exist under
  `/tmp/usrp_spectrograms/all_detectors/` for 0–70 and are reused; **75/80 are
  generated fresh**. The dino path also seeds the deterministic `cuda_dino`
  foundation from there to skip one GPU pass per level.
- `bash ber_status.sh` prints a live progress dashboard (per detector × level, with
  BERs) straight from `results/` — accurate even after a crash.

## Open items
- **Per-piece vs union time overlap** (known, ≤0.1%): `TimeOverlapMin` is applied
  to each snippet piece independently rather than to the union across pieces of the
  same signal. Only matters when a detector emits many sub-threshold fragments
  (dino at ≤ −11 dB); measured impact there is 3 → 4 matches of 1112. Fixing it
  would be more principled but requires re-running all 51 levels for a ≤0.1% change.
- **Fairness nuance** (minor): a handful of detector snippet-decode *throws* are
  excluded as `insufficient` alongside the genuinely slot-truncated fragments;
  strictly they should be tied to the genie decodable set so detector
  under-saving is penalised rather than excluded. Detect rates are ≈1.0 at high
  SNR so it doesn't move the table there.
- **dino Bluetooth** is the one class where dino trails at high SNR (0.054 vs
  coherent 0.0029 at 54 dB, 0.113 vs 0.079 at 39 dB) — its tight boxes decimate
  GFSK harder. Worth a look if BT matters for the paper.
- **`attenuation_dB_85`** was excluded (a partial 6.3 GB capture, beyond the 80 dB
  requested range), as were the `*_v2` recaptures.

### Resolved (see Decode fixes for mechanisms)
- **Bluetooth snippet corruption** — the trim's ±2% time guard misaligned the
  sync-free BT receivers; exact-slot trim took coherent BT 0.77→**0.0029**.
  Breadcrumbs in `dev_debug_bt_snippet.m`; the phase-continuous stitch,
  per-piece fallback, and wide ±8 kHz CFO refine were kept (harmless, small wins).
- **"DINO misses narrowband"** — an artifact of scoring capture fragments: the
  apparent misses were overwhelmingly the sub-1 ms slot fragments (dino draws no
  box for them; coherent's huge boxes covered them incidentally). With fragment
  exclusion, dino covers all decodable signals (miss = 0, detect ≈ 1.0), matching
  mask-eval (dino narrowband recall 98.6%).
- **"5G decodes at 0.34"** — slot-truncated fragments scored as failures; full
  slots decode at ~0.005 (GT).

## Files
- `ber_eval_run.m` — core: match, extract/decode, per-class aggregate.
- `ber_eval_all.m` — 3-detector driver → `results/ber_comparison_<stem>.{csv,png}`.
- `ber_precompute.m` — build the shared `wave_cache.mat`.
- `decode_waveforms_24576.m` — standards-aware decoder (+ MMSE eq add-on).
- **`run_ber_sweep.sh`** — sweep one detector across levels (gen → eval → cleanup;
  serial gen, pooled evals, idempotent resume).
- **`ber_sweep_one.m`** — one (level, detector) eval + its own per-level summary row.
- **`ber_sweep_combine.m`** — aggregate all levels → `ber_sweep_overall.csv`
  + `ber_sweep_byclass.csv` (the tables; figures come from `plot_ber_figs.py`).
- **`plot_ber_figs.py`** — all figures (claim + full-sweep), repo-standard styling,
  BER in %. Run this after a sweep; it reads only the result CSVs.
- **`ber_status.sh`** — live progress dashboard (detector × level grid with BERs).
- `dev_*.m` — one-off diagnostics used to root-cause the fixes above.
- `results/` — per-signal + per-class CSVs per level, sweep tables + figures, logs.
- `wave_cache.mat` — generated by `ber_precompute.m` (do not commit; rebuild anywhere).

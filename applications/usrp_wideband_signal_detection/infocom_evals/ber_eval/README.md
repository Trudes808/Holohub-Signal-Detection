# ber_eval — bit-error-rate evaluation of detector-saved waveforms

Measures **BER vs SNR** for the USRP wideband detectors by *decoding the IQ each
detector actually saved* and comparing to the known transmitted bits. The point:
show whether lossy **detect → snip** (data saving) preserves **decodability**, per
modulation class, for **ground_truth / coherent_power / finetuned_dino_m2**.

Branch: `BER_eval`. Uses MATLAB R2025b (Comms/Signal/DSP/5G/LTE/WLAN/Bluetooth +
the `helperOFDM*` example). Run headless: `matlab -batch "..."`.

> **Status (2026-07-28):** attenuation_dB_0 is COMPLETE — all 7 classes decode
> cleanly for all three detectors (results table below); every known harness
> artifact is fixed. Next step: the SNR sweep (snippets per attenuation + one
> `ber_eval_all` per level).

## Quickstart
```matlab
% one capture, all three detectors -> per-class CSV + figure under results/
ber_eval_all('attenuation_dB_0')

% one detector
r = ber_eval_run('attenuation_dB_0','coherent_power');
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

## Results @ attenuation_dB_0 (SNR ≈ 54 dB)  — overall BER: GT 0.019 / coh 0.024 / dino 0.022
| class | ground_truth | coherent | dino_ft_m2 |
|---|---|---|---|
| BPSK | 0.0010 | 0.0010 | 0.0012 |
| QPSK | 0.0035 | 0.0027 | 0.0026 |
| 16QAM | 0.018 | 0.014 | 0.015 |
| OFDM | 0.024 | 0.027 | 0.027 |
| 5G | 0.0049 | 0.018 | 0.013 |
| 802.11ax | 0.0052 | 0.0067 | 0.0056 |
| Bluetooth | 0.0035 | 0.0029 | 0.054 |

Detect rate ≈ 1.0 everywhere (coherent 5G: 1 miss). Signals that were
slot-truncated in the capture (the 0.04/0.2 ms slots) are excluded as
`insufficient` for all detectors alike.

**Headline:** for **all 7 classes coherent ≈ dino ≈ ground-truth** — the saved
snippets decode essentially as well as perfect extraction, i.e. data-saving
**preserves decodability**. Detectors occasionally edge out GT (QPSK/16QAM):
the snip DDC band-limits each snippet, shaving out-of-band noise before decode.
Residuals: dino BT 0.054 (its tight boxes decimate GFSK harder — small, worth a
look at an SNR sweep); coherent 5G 0.018 vs GT 0.0049. Detection quality itself:
mask-eval confirms **dino IoU 0.95 vs coherent 0.48** — dino detects far more
precisely; coherent's crude ~19×-wide boxes just bracket everything (and store
~19× more spectrum per detection).

## Performance
`ber_precompute.m` builds `wave_cache.mat` (67 MB): per-variation `md`, `txBits`,
`refd`, ... shared by every run so no run reloads the 287×15 MB `.mat` files.
`ber_eval_all` builds it once. Verified identical results (GT 0.018823).

## Open items
- **SNR sweep** (the next step): extend beyond attenuation_dB_0 — generate
  coherent + dino snippets per attenuation (see regen commands above), then one
  `ber_eval_all('<stem>')` per level. The cache + harness make each level cheap.
- **Fairness nuance** (minor): a handful of detector snippet-decode *throws* are
  excluded as `insufficient` alongside the genuinely slot-truncated fragments;
  strictly they should be tied to the genie decodable set so detector
  under-saving is penalised rather than excluded. At atten_0 the counts are near
  identical across detectors (detect rate ≈ 1.0), so it doesn't move the table.
- **Small residuals to watch during the sweep**: dino BT 0.054 (tight boxes
  decimate GFSK harder) and coherent 5G 0.018 vs GT 0.0049.

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
- `dev_*.m` — one-off diagnostics used to root-cause the fixes above.
- `results/` — per-signal + per-class CSVs, comparison figure, run logs.
- `wave_cache.mat` — generated by `ber_precompute.m` (do not commit; rebuild anywhere).

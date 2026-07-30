# results_v2 — detection defined as region-level mask coverage ≥ 0.1

Same sweep as `../results/` (17 attenuation levels, baseline snip settings), with the
**detection rule replaced** so it matches the other evaluations:

> A ground-truth emission counts as detected when **≥ 0.1 of its area on the FFT grid is
> covered by the detector's raw mask** (`comparison_config.yaml` → `snr.det_threshold: 0.1`).

The legacy rule (signal-center-inside-a-box **AND** ≥10% time overlap, applied per
snippet piece) is **not used at all** here.

## How it is computed
`region_detect.py` **imports `signal_detection_experiments/mask_eval_metrics.py`** and calls
its own `region_coverage` / `resize_mask_nearest`, so the criterion is bit-for-bit the
same as the mask evals rather than a re-implementation. Per level it writes
`regdet_<detector>_<stem>.csv` (one row per emission: covered_px, box_px, coverage,
detected), which `ber_eval_run.m` joins on `(sample_start, freq_lower_hz, freq_upper_hz)`
— sample_start alone is not unique because emissions share slots.

Two choices worth stating, since `mask_eval_metrics` does not specify them for BER use:
- **Emission-level aggregation.** That module emits one row per *(frame, annotation)*, but
  BER scores per emission and emissions span frames. Coverage here is
  `Σ covered_px / Σ box_px` over the emission's frames — the literal reading of "0.1 of a
  ground-truth emission is covered".
- **Frames with no detector mask** (pipeline-drain tails) are skipped, not scored as
  misses — same as `mask_eval_metrics`, so a drain artifact cannot bias detection.

Scoring: not detected → BER 1.0 (`miss`). Detected but the snipper saved no covering
snippet → BER 1.0 too, tagged `miss_nosave` so the data-saving loss stays countable
(it happens: 84 signals at −1 dB, 44 at +4 dB, 0 at high SNR).

`ground_truth` is copied from `../results/` unchanged — the genie reads the capture
directly and is never subject to detection, so no rule can change it.

## Result: DINO is insensitive to the definition, coherent is not

overall BER % (v1 = `../results/`, v2 = this folder), with signals decoded:

| SNR | COH v1 | **COH v2** | dec v1 → v2 | DINO v1 | **DINO v2** | dec v1 → v2 |
|---|---|---|---|---|---|---|
| 44 | 4.09 | **3.90** | 1064 → 1091 | 2.44 | **2.44** | 1108 → 1108 |
| 39 | 6.26 | **5.25** | 991 → 1094 | 2.50 | **2.39** | 1105 → 1106 |
| 24 | 42.33 | **38.61** | 908 → 1042 | 20.86 | **20.77** | 1102 → 1103 |
| 19 | 64.76 | **71.46** | 789 → 786 | 28.91 | **28.95** | 1105 → 1105 |
| 9 | 76.97 | **94.99** | 713 → 396 | 40.49 | **40.53** | 1104 → 1103 |
| −1 | 82.57 | **99.38** | 531 → 86 | 56.88 | **56.33** | 1009 → 1008 |
| −11 | 83.63 | **100.00** | 528 → **0** | 99.94 | **99.97** | 8 → 4 |
| −26 | 86.70 | **100.00** | 462 → **0** | 99.24 | **99.71** | 23 → 8 |

**DINO barely moves** (≤0.6 points anywhere, decode counts within a handful) — its masks
genuinely cover the emissions, so both definitions agree. **Coherent moves a lot, in both
directions:**
- **High SNR it improves** (39 dB: 6.26 → 5.25, +103 more signals decoded). The old
  center-in-band test was discarding well-covered signals, because coherent merges
  neighbours into wide boxes centred in the *gaps* between them.
- **Low SNR it collapses to exactly 100% with zero decodes** (≤ −11 dB), where the old
  rule credited it with ~460–530 decodes and an ~85% "plateau".

That plateau was never detection. This is the **third independent confirmation**: (1) the
mask geometry at −16 dB (36/1112 emissions had any overlap; boxes ~50 MHz wide but 0.29 ms
sparse), (2) the 75 kHz mask pre-filter driving coherent to zero snippets once the RX clock
spur is removed (`../results_coh_75k_1ms/`), and (3) region coverage rejecting those
detections here because they cover <10% of any emission's area.

**Use this folder for detector comparison.** It removes the metric artifact that let
coherent appear to "win" below −10 dB, and it is the same detection definition as the rest
of the evaluation suite. `../results/` remains valid for the snip-fidelity claim (which is
matched-subset and therefore rule-independent).

## Figures
Same nine as the baseline (`python ../plot_ber_figs.py . --title-suffix "..."`), including
`ber_sweep_overall_v2.png` in the presentation style.

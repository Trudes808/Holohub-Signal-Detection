# results/ — baseline sweep

The original sweep: 17 attenuation levels (0–80 dB, skipping the `*_v2` recaptures) ×
{`ground_truth`, `coherent_power`, `finetuned_dino_m2`}.

**Parameters** (defaults everywhere — this is the reference the other folders vary from):
- snip: `--mode frequency --min-box-pixels 256`, no bandwidth or duration gate
- detection: signal-center-inside-a-detection-box **AND** ≥10% time overlap, applied
  per snippet piece (`TimeOverlapMin = 0.10`)

## Use this folder for the snip-fidelity claim
"Snipping is minimally destructive compared to what the channel already did" —
`fig_snip_vs_channel.png`, `fig_snip_excess.png`, `fig_snip_vs_channel_byclass.png`.
That claim is computed on a **matched subset** (each detector against the genie on exactly
the signals that detector saved), so it is independent of the detection rule and is the
same story in any of these folders.

## For detector comparison prefer `../results_v2/`
Its detection rule (region-level mask coverage ≥ 0.1) matches the rest of the evaluation
suite, and it removes an artifact present here: the center-in-band test both discards
well-covered signals at high SNR (coherent merges neighbours into wide boxes centred in the
*gaps*) and credits coherent with ~460–530 spurious decodes below −10 dB, where it is
actually detecting the RX clock spur rather than signals. See the parent README's
"Why coherent 'beats' dino below −10 dB" and `../results_v2/README.md`.

## Contents
- `ber_<detector>_<stem>.csv` — per signal: class, variation, status, bitErrors, numBits
- `ber_<detector>_<stem>_byclass.csv` — per class: nDecoded/nMiss/detectRate/BER
- `ber_<detector>_<stem>_overall.csv` — one summary row per level
- `ber_sweep_overall.csv` / `ber_sweep_byclass.csv` — aggregated by `ber_sweep_combine.m`
- `*.png` — figures from `plot_ber_figs.py` (see parent README "Figures")
- run logs are gitignored; the CSVs and figures are the record

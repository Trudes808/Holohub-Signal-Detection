# results_coh_75k_notime — coherent_power, 75 kHz bandwidth gate, NO time minimum

Third variant. Same sweep as `../results/`, coherent_power snipping gated on
**bandwidth only**:

    --min-mask-bandwidth-hz 75000   # per-row run-length mask pre-filter (spur removal)
    --min-bandwidth-hz      75000   # post-merge bounding-box bandwidth gate
    --min-duration-s        0       # NO minimum duration (the difference vs 75k_1ms)
    --min-box-pixels        256     # unchanged

`ground_truth` and `finetuned_dino_m2` are copied verbatim from `../results/`.

Question being tested: does dropping the 1 ms duration gate recover BER? Misses
dominate the error budget, so more coverage should lower BER — but the harness applies
`TimeOverlapMin` (10%) **per snippet piece**, so a sub-1 ms piece still fails to match
a 5/10/20 ms slot and adds stored data without converting to a decode.

Three-way comparison lives in `fig_gate_tradeoff.png` (vs baseline) and the table below.

## Answer: no, dropping the 1 ms gate does NOT recover BER

| SNR | coverage base / +1ms / no-time | overall BER base / +1ms / no-time |
|---|---|---|
| 44 | 95.8% / 90.0% / **90.7%** | 4.09% / 7.16% / **7.06%** |
| 39 | 89.5% / 82.2% / **83.0%** | 6.26% / 17.51% / **17.14%** |
| 29 | 86.1% / 69.4% / **74.0%** | 27.9% / 50.6% / **45.7%** |
| 24 | 82.1% / 60.9% / **65.5%** | 42.3% / 66.0% / **60.8%** |
| 19 | 71.8% / 44.7% / **48.5%** | 64.8% / 88.7% / **87.0%** |
| 9 | 64.8% / 16.7% / **23.7%** | 77.0% / 98.8% / **98.0%** |
| ≤ −1 | 48.1% / 0.0% / **0.1%** | 82.6% / 100% / **100%** |

Removing the duration gate recovers only **1–5 points of coverage** and **0.1–5 points
of BER** — both 75 kHz variants remain far worse than the ungated baseline. The
duration gate was never the problem; **the 75 kHz bandwidth gate is**, and it is
common to both.

Why: `min_mask_bandwidth_hz` zeroes lit runs narrower than 75 kHz **per mask row**. At
low SNR a genuine signal's coherent mask is patchy and fragmented into narrow runs, so
the filter deletes real signal content along with the clock spur. Coverage therefore
collapses with SNR (65% → 0% below −1 dB) no matter what the duration gate does.

Note the snippet **count** rises above baseline (atten_0: 3091 vs 1928) because the
same per-row filter strips the spur rows that were 4-connecting separate signals into
single giant components — de-fusing them into many tighter boxes. So this variant emits
*more, smaller, better-localized* boxes than baseline while still covering fewer
signals. Count is therefore not a storage proxy across variants; use `snip_eval`'s
GB/hr metrics for volume.

Fidelity of what it kept is unchanged-to-slightly-worse (+0.25…+4.9 points vs baseline
+0.25…+3.1), so the gate is not damaging snippets — it is dropping signals.

**Takeaway:** for BER, ungated coherent snipping is best; the 75 kHz gate buys
box precision and spur immunity (its real purpose in the data-saving work) at a
substantial coverage cost. If the goal is BER with spur immunity, the gate needs to be
bandwidth-aware only for *persistent* lines (e.g. temporal-persistence detection)
rather than a blanket per-row width filter.

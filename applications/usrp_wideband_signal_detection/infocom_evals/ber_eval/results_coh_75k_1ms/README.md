# results_coh_75k_1ms — coherent_power with the 75 kHz + 1 ms snip gate

Same sweep as `../results/`, with ONE change: the **coherent_power** IQ snipping runs
the 75 kHz / 1 ms selectivity gate from `snip_eval` (the variation evaluated in the
data-saving work), i.e.

    --min-mask-bandwidth-hz 75000   # per-row run-length mask pre-filter (spur/fragment proof)
    --min-bandwidth-hz      75000   # post-merge bounding-box bandwidth gate
    --min-duration-s        0.001   # 1 ms minimum duration
    --min-box-pixels        256     # unchanged

`ground_truth` and `finetuned_dino_m2` are **copied verbatim** from `../results/`
(not re-run) so the only difference between the two folders is the coherent snipper
gate. Regenerate figures with:

    python ../plot_ber_figs.py . --title-suffix "coherent_power: 75 kHz + 1 ms snip gate"

Caveat by construction: the gate discards signals narrower than 75 kHz or shorter
than 1 ms, so those are dropped from what coherent saves. The BER harness already
excludes sub-1 ms *slots* as `insufficient`, but genuinely narrowband classes remain
in scope and will show up as misses here.

## Results — what the gate buys and what it costs

`python ../compare_variants.py ../results . --label "75 kHz + 1 ms gate"`
→ `fig_gate_tradeoff.png` (3 panels: what it discards / fidelity of what it kept /
net effect). Coherent only; GT and dino are identical in both folders by construction.

| SNR | snippets kept | coverage base→gate | overall BER base→gate | snip excess base/gate |
|---|---|---|---|---|
| 49 | 705 / 2234 = **32%** | 99.9% → 97.2% | 2.82% → 4.06% | +0.67 / +0.68 |
| 39 | 1046 / 2748 = **38%** | 89.5% → 82.2% | 6.26% → 17.51% | +0.50 / +1.12 |
| 24 | 1066 / 3316 = **32%** | 82.1% → 60.9% | 42.3% → 66.0% | +1.26 / +3.59 |
| 9 | 311 / 1873 = **17%** | 64.8% → 16.7% | 77.0% → 98.8% | +0.85 / +1.00 |
| 4 | 64 / 1638 = **4%** | 57.2% → 3.2% | 79.2% → 99.9% | +0.80 / +4.40 |
| ≤ −1 | **0** | → **0%** | → 100% | n/a (nothing saved) |

**The gate trades coverage for storage and leaves fidelity alone.** It keeps 4–38% of
the snippets (atten_0: 1928 → 547), and the BER of *what it keeps* is essentially the
baseline's (+0.25…+1.1 pts at usable SNR). Every bit of the extra BER is signals it
declined to save — visible as the miss band in `fig_error_budget.png`.

**It also confirms the deep-noise finding independently.** At ≤ −1 dB the gate emits
**zero** snippets: the 75 kHz per-row mask pre-filter removes the RX clock spur
([[streak-48mhz-is-rx-clock-spur]]), which was exactly the proposed mechanism for
coherent's apparent deep-noise "advantage" over dino in the baseline sweep. Strip the
spur and the accidental coverage disappears — coherent then declines to fire, like
dino. Two independent lines of evidence now agree that plateau was spur-driven
hoarding, not detection.

Harness note: "detector saved nothing" is a legitimate result (all-miss), not an
error. `ber_eval_run` now distinguishes a missing snippet folder (pipeline never ran
→ error) from an empty one (ran, saved nothing → scored), and `run_ber_sweep.sh`
materializes the empty `snippets/` dir when gen succeeds with zero output, since the
snipper does not create it in that case.

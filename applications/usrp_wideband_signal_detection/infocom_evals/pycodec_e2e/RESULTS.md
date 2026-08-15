# pycodec end-to-end: detect → snip → decode (offline, 2026-08-15)

First full rehearsal of the real-time decode chain, entirely in Python on the
Spark — no MATLAB anywhere:

```
comprehensive_ordered_py (pycodec composite, 245.76 MSps, 48 placements)
  -> offline detection run (coherent_power + signal_snipper + sigmf_file_sink,
     config_signal_snipper_single_channel.yaml via run_cuda_dino_offline_file.py --snippets-only)
  -> decode_snippets.py: match snippets to TX truth, mix residual, pycodec decode
```

## Result

| Metric | Value |
| --- | --- |
| Truth placements | 48 (BPSK/QPSK × RRC/RC × 20/10/5/1/0.2/0.04 ms) |
| Detected + decoded | **40/48** |
| Aggregate BER over decoded bits | **0 / 12,787,450 = 0.0** |
| Worst placement BER | 0.0 |
| Time coverage of decoded placements | 100% each |

The 8 misses are exactly the **0.04 ms duration tier** (all 8 waveforms of the
last pass): at the 245.76 MSps offline geometry one FFT row spans 41.7 µs, so a
40 µs burst is ~1 mask row — below the detector's morphology/persistence
minimum (~3 rows ≈ 125 µs). That is a detector design point (short-burst
sensitivity), not a decode limitation; the 0.2 ms tier (~5 rows) detects and
decodes flawlessly.

## How to reproduce

```bash
# 1. build the composite (see pycodec/README.md in holoscan_waveform_generation)
# 2. offline detect + snip:
python3 run_cuda_dino_offline_file.py \
  ~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_ordered_py.sigmf-data \
  --detector coherent_power --config config_signal_snipper_single_channel.yaml \
  --snippets-only --output-root /tmp/usrp_spectrograms/pycodec_e2e
# 3. decode:
python3 infocom_evals/pycodec_e2e/decode_snippets.py \
  --truth ~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_ordered_py.sigmf-meta \
  --snips /tmp/usrp_spectrograms/pycodec_e2e/snippets
```

## Notes / next

- The driver is truth-driven (uses wfgt:source_mat to pick the modulation spec):
  the next step toward live is a header/classifier-driven variant, or the framed
  mode (preamble + modulation header + CRC) so snippets self-describe.
- Snippets carried decimation_factor 1 here (noise-free composite → wide boxes);
  the driver already handles decimated snips via wfgt:snippet_sample_rate.
- Live integration target: the same decode logic consuming sigmf_file_sink
  output (or an in-process mask+IQ tap) with a PN9-BER readout in the demo.

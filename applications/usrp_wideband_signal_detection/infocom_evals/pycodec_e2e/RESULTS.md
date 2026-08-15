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

## Addendum — REAL-TIME framed decode with live BER metrics (2026-08-15)

`rt_decode_daemon.py` ran concurrently with the detection pipeline processing
the framed composite (`comprehensive_framed_py`): it consumed signal_snipper
packs as they appeared and decoded every snippet **with zero ground truth** —
sub-band channelization (merged detection boxes), blind symbol-rate estimation,
residual-CFO correction, self-describing frame headers, per-frame CRC, live
PN9 BER:

```
[16:45:48] === LIVE: frames 816 (crc_ok 812) by_mod {'BPSK': 277, 'QPSK': 539}
           PN9 BER 1.08e-04 (361/3342336) ===
```

99.5% of frames CRC-verified with BER 0; the handful of failures are frames
clipped at snippet time-boundaries (the CRC flags them — which is its job).
Metrics also stream to `rt_metrics.json` for a future HoloViz overlay.
Remaining polish for the demo: snip-boundary edge guard, decode-throughput
optimization (pure-numpy prototype runs ~100 ms per short snip), and a native
in-graph operator (or holoviz overlay feed) instead of the file-tail sidecar.

## Addendum 2 — six modulation families through the live chain (2026-08-15)

pycodec grew 16QAM, 8PSK, framed GFSK (2FSK/4FSK, noncoherent), and generic
framed OFDM (64-FFT, pilots, LTF channel est). The daemon now runs a family
cascade per channelized sub-band: constant envelope -> FSK discriminator;
else linear framed; else OFDM at the profile-snapped rate.

Concurrent detect+snip+decode over `comprehensive_framed_py` (36 placements,
6 classes x 3 durations):

```
frames_decoded 3249   crc_ok 3130 (96.3%)   PN9 BER 4.3e-04
by_mod: BPSK 274, QPSK 539, 8PSK 786, 16QAM 1018, 2FSK 20, 4FSK 39,
        OFDM-QPSK/OFDM-16QAM 573
```

Every family decodes blind in real time; CRC failures remain concentrated at
snippet time-boundary clips (plus marginal 4FSK, whose I&D filter is not
Gaussian-matched yet). This is the classifier-ready substrate: six visually
and statistically distinct classes, each with self-describing ground truth.

## Addendum 3 — LIVE DECODE dashboard panel (2026-08-15)

The HoloViz sidebar now renders a **LIVE DECODE** panel fed by the decode
daemon's `rt_metrics.json` (config: `visualization.renderer.decode_metrics_json`):
frames + CRC pass rate (color-coded), the aggregate PN9 BER as the headline
number, Mbit checked + decode throughput, an instantaneous-BER sparkline
(log-scaled, per poll interval), and per-modulation frame-count bars.

Captured live (single-channel 2.4 GHz ambient RF on the waterfall, decode
metrics from a concurrent framed-composite sweep), via the new dashboard-only
render-buffer screenshot path (`visualization.screenshot_path` +
`screenshot_after_frames`, headless-friendly — never touches the desktop):

![LIVE DECODE dashboard v1](img/hud_dashboard_v1.png)

Iteration knobs on the table: panel placement/size, a wider BER history strip
under the waterfalls, per-band decode markers drawn onto the detection panel,
frames/s + pipeline-latency readouts (from the existing timing summaries), and
a "last decoded payload" text ticker for arbitrary-payload demos.

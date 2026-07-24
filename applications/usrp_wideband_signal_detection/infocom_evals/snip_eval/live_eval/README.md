# live_eval — over-the-air 500 MSps data-saving proof

End-to-end demonstration of the coherent-power snip pipeline on **live over-the-air X410 captures**,
comparing the original save-all footprint against the snipped footprint under two snipper gate
settings. This is the "prove it works on real OTA data" companion to the wired-attenuation-sweep
`real_snip_metrics*` evals one level up.

## Captures (2026-07-24)
Two single-channel over-the-air captures on the X410 (serial 3415B2B, ZBX, FPGA `CG_400`, 100 GbE):

| center | rate | duration | datatype | size | quality |
|---|---|---|---|---|---|
| 2.4 GHz | 500 MSps | 10.0 s | cf32_le | 40.00 GB | 0 dropped / 0 overflow |
| 1.0 GHz | 500 MSps | 10.0 s | cf32_le | 40.00 GB | 0 dropped / 0 overflow |

RX0, gain 30 dB. Analog bandwidth coerces to 400 MHz (ZBX front-end), so the outer ±50 MHz of the
500 MHz digital span rolls off. Files + SigMF metas live in **`/tmp/usrp_spectrograms/ota_x410_*`**
(kept off `/home` deliberately; ephemeral — re-capture to regenerate).

### How they were captured
`rx_samples_to_file` cannot sustain 500 MSps to disk (single-threaded recv+write interleave drops
~60% even to a RAM disk), and the NVMe only sustains ~0.7 GB/s vs the 4 GB/s cf32 rate. So a small
UHD capturer (`cap_ram.cpp`, in this folder) **receives the whole 10 s into a
pre-faulted 40 GB RAM buffer with no disk I/O during the stream, then flushes to `/tmp` afterward**.
`benchmark_rate` confirmed the receive path is lossless at 500 MSps; the capturer reported exactly
5,000,000,000 samples, 0 overflows, for each capture.

## Detector + pipeline
- **Detector config**: `../config_coherent_power_500msps_1ch.yaml` — single-channel, 500 MSps,
  fft 20480 (24.4 kHz/bin), carrying the **two-channel 500 MSps config's** detector settings
  (`fast_score_threshold 0.7`, full-width 20480 masks, fast-threshold path). The 245 MHz per-frequency
  `.npy` calibration is disabled (it does not transfer to these bands).
- **Pipeline**: `../snip_pipeline.py` with `../snip_pipeline_ota.yaml` (256-pixel gate) and
  `../snip_pipeline_ota_75k1ms.yaml` (+75 kHz bandwidth filter + 1 ms min duration). Both run
  `coherent_power` with `write_iq: false` — no snippet IQ stored; sizes are reconstructed exactly
  from the decimated sample counts in each snippet meta. Masks are identical across the two gate
  settings, so the second pass reuses the first pass's masks.

## Results (`ota_live_eval.csv`, `ota_live_eval.png`)
Save-all baseline at 500 MSps cf32 = **14,400 GB/hr** (40 GB per 10 s capture).

| capture | gate | snippets | split size (10 s) | GB/hr | reduction |
|---|---|---:|---:|---:|---:|
| 2.4 GHz | original (save-all) | — | 40,000 MB | 14,400 | 1× |
| 2.4 GHz | snip · 256px | 16,716 | 893 MB | 321 | 44.8× |
| 2.4 GHz | snip · 75 kHz + 1 ms | **2,185** | **650 MB** | 234 | **61.6×** |
| 1.0 GHz | original (save-all) | — | 40,000 MB | 14,400 | 1× |
| 1.0 GHz | snip · 256px | 5,236 | 317 MB | 114 | 126× |
| 1.0 GHz | snip · 75 kHz + 1 ms | **1,829** | **254 MB** | 91 | **157.5×** |

### Two-channel system totals — 10 s (`ota_system_totals.csv`)
Both captures together = the two-channel 500 MSps system. Save-all for 10 s = **2 × 40 GB = 80 GB**.

| gate | stored (10 s) | vs 80 GB | recordings | files (data+meta) |
|---|---:|---:|---:|---:|
| snip · 256px | **1.21 GB** | **66.1× less** | 21,952 | 43,904 |
| snip · 75 kHz + 1 ms | **0.90 GB** | **88.5× less** | **4,014** | **8,028** |

So in a 10-second dual-500-MSps window the coherent-power snipper turns **80 GB of raw IQ into
~0.90 GB across 4,014 SigMF recordings** (8,028 files: one `.sigmf-data` + one `.sigmf-meta` per
detection) under the 75 kHz/1 ms gate. (These runs use `write_iq: false`, so only the `.sigmf-meta`
files are actually on disk; the byte totals are the exact sizes the `.sigmf-data` would occupy. Note
the 1.0 GHz share — 1,829 recordings / 254 MB — is the band-edge spur, see below.)

**Interpretation.** The 256-pixel gate alone leaves many tiny fragments (16.7k / 5.2k snippets).
Adding the 75 kHz bandwidth + 1 ms duration filters cuts the snippet count by **87 % / 65 %** while
shedding only **27 % / 20 %** of the bytes — i.e. it removes thousands of tiny, low-content fragments
and leaves a far more usable set of "split captures" at a slightly better reduction (44.8→61.6× and
126→157.5×). 2.4 GHz keeps more data than 1.0 GHz (more occupied spectrum: Wi-Fi / Bluetooth).

## Spectrogram + bounding-box overlays (`ota_overlay_cf{2400,1000}MHz.png`)
`render_ota_overlay.py` computes a spectrogram directly from the raw IQ (FFT each 20480-sample row of
a frame → 512×20480 dB, the detector's grid) and overlays the **actual** snipper boxes (read from the
75 kHz/1 ms soft-label meta; not re-derived). These are a visual truth-check and they reveal an
important asymmetry between the two bands:

- **2.4 GHz — genuine detections.** 2,185 boxes spread across the ISM band (offset 0→+200 MHz, i.e.
  2400–2600 MHz), bandwidths 8.5 MHz median up to 69 MHz, **99 % inside the ±200 MHz analog passband**.
  The green boxes land squarely on real Wi-Fi/Bluetooth/Zigbee bursts. This is the "it works on real
  OTA data" result.
- **1.0 GHz — dominated by a band-edge artifact.** *All* 1,829 boxes sit at offset +225…+250 MHz
  (abs 1225–1250 MHz), all ~8.5 MHz wide (median 8.50, max 9.03 — implausibly uniform), and **100 %
  beyond the 400 MHz analog passband** (in the rolled-off region near the +250 MHz Nyquist edge).
  That signature (fixed frequency, fixed width, persistent every frame, outside the passband) is a
  **receiver spur / alias**, not a real emitter — the same class of artifact as the 48 MHz RX clock
  spur documented in `../problem.md`. Meanwhile the genuine broad cellular occupancy at ~758–803 MHz
  (visible at the left of the figure) is *not* boxed: the coherent-power detector normalizes against a
  per-row frequency reference, so smooth wideband occupancy is absorbed into the floor and only
  sharp/narrow features that stand out get flagged.

**Consequence:** the 1.0 GHz `n_snippets` / size numbers above are essentially all spur, not real
signal — treat the 1.0 GHz row as a hardware-artifact demonstration, not a data-saving result. Fixes
to get a meaningful 1.0 GHz number: restrict detection to the ±200 MHz analog passband (or notch the
+240 MHz spur bin), which the current gate settings do not do. The 2.4 GHz result stands on its own.

## Files
- `render_ota_overlay.py` — spectrogram + snipper-box overlays (→ `ota_overlay_cf{2400,1000}MHz.png`).
- `make_live_eval.py` — builds the combined table + figure from the two source metrics CSVs.
- `ota_live_eval.csv` — the three-way comparison table (source of the table above).
- `ota_system_totals.csv` — two-channel system totals (save-all 80 GB vs stored, recording/file counts).
- `ota_live_eval.png` — stored GB/hr (log) per capture vs the save-all line, snippet counts + reductions.
- `ota_metrics_snip_256px.csv`, `ota_metrics_snip_75kHz_1ms.csv` — raw `snip_pipeline` per-capture metrics.

Larger regenerable artifacts (NOT committed; in `/tmp`): the 40 GB captures, the mask arrays under
`/tmp/usrp_spectrograms/ota_snip_pipeline{,_75k1ms}/masks/`, and the soft-label metas under
`.../soft_labels/` (original capture meta + one `coherent_power_detection` annotation per snippet).

## Reproduce
```bash
# 1. build + capture (needs the X410 + UHD):
#    g++ -O2 -std=c++17 live_eval/cap_ram.cpp -o /tmp/cap_ram $(pkg-config --cflags --libs uhd) -lpthread
#    /tmp/cap_ram 192.168.100.3 2400e6 500e6 30 10 0 /tmp/usrp_spectrograms/ota_x410_cf2400MHz_500Msps_cf32_10s.sigmf-data
#    cap_ram 192.168.100.3 1000e6 500e6 30 10 0 /tmp/usrp_spectrograms/ota_x410_cf1000MHz_500Msps_cf32_10s.sigmf-data
#    (then write the .sigmf-meta pair: cf32_le, core:sample_rate 500000000, core:frequency = center)
# 2. run both gate settings (from ../):
conda activate dinov3
python snip_pipeline.py snip_pipeline_ota.yaml           # 256px
python snip_pipeline.py snip_pipeline_ota_75k1ms.yaml    # +75 kHz / 1 ms (reuses masks)
# 3. rebuild this comparison:
python live_eval/make_live_eval.py
```

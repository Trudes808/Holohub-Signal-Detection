# Detector mask comparison — 1 s OTA capture @ 491.52 MSps (2.4 GHz)

**Date:** 2026-09-08 · **Branch/commit:** `live_demo` (follows the composite sanity check in
`../detector_mask_sanity/`) · **Bench:** DGX Spark GB10, X410 serial 3415B36.

Companion to the comprehensive_ordered sanity check, this run compares the four
detectors on **real over-the-air data at the actual live rate** (the composite
capture was 245.76 MSps — the DINO-FT model's trained rate; live runs at 491.52).
No ground truth exists for an OTA capture, so the comparison is masks-vs-raw
spectrogram plus cross-detector statistics.

## The capture

UHD host capture (kernel sockets over sfp0 — not the DPDK app path), then wrapped
into SigMF:

```
rx_samples_to_file --args "addr=192.168.10.2" --freq 2.4e9 --rate 491.52e6 \
  --gain 10 --channel 0 --type short --wirefmt sc16 --nsamps 491520000 \
  --spb 1000000 --file /dev/shm/ota_cap.dat
```

- **Zero overflow indications** in the UHD log (the host sustained the full
  15.7 Gbps for the second; write to `/dev/shm`, not disk, to dodge the file-write
  bottleneck).
- Stored: `~/Documents/captures/x410_ota_2g4_gain10_20260908.sigmf-{data,meta}`
  (`ci16_le`, 1.97 GB — the offline eval binary reads ci16 natively).
- Content: ambient 2.4 GHz ISM — WiFi bursts at ch 1/6 (2412/2437 MHz), scattered
  BT-class hops, thin CW lines near 2550–2560 MHz. RMS ≈ −38 dBFS, no clipping.
- Because this bypasses the DPDK/CHDR path entirely, the capture is also a
  clean-transport reference: no torn-frame/data-path-garbage artifacts anywhere
  in it, consistent with the tear being on the CHDR ingest side.

## Runs

Same four detectors/configs as the composite study (`run_ota_detectors.sh`
pattern; coherent kept the **live** `max_emit_occupancy=0.35` guard — real OTA
occupancy never came close, 0 faults in 93 frames):

| run | frames | grid | note |
|---|---|---|---|
| coherent_power (dynamic floor) | 93 × 10.7 ms | 256×20480 | live config verbatim |
| cuda_dino (zero-shot) | 46 × 21.3 ms | wide grid | live config verbatim |
| DINO-FT native | 46 × 21.3 ms | 10240×1024 | dedicated 1024-pt FFT → **480 kHz/bin, 2× the trained 240 kHz/bin** |
| DINO-FT RT (live path) | 46 × 21.3 ms | 512×20480 | auto wide FFT 20480 → 20:1 resize; flatten on; ~53 ms/frame inference |

## Results (`ota_frame_stats.csv`, panels `ota_panels_frame*.png`)

Mean mask occupancy, split into the inner 94 % of the band vs the outer 3 %
band-edge margins (where the X410 anti-alias rolloff lives):

| detector | inner-band occupancy | edge-margin occupancy |
|---|---|---|
| coherent_power | 0.56 % | 0.08 % |
| cuda_dino (zero-shot) | 13.35 % | 0.58 % |
| DINO-FT native @491.52 | 4.78 % | 2.19 % |
| **DINO-FT RT (live path)** | **0.32 %** | 17.50 % |

Findings:

1. **The live DINO-FT path works on real OTA data at the live rate.** In the
   usable band it is the *tightest* of the four (0.32 % occupancy), sitting
   exactly on the WiFi/BT bursts (frames 26/42: single-burst frames where the
   mask is the burst and nothing else).
2. **Native mode at 491.52 MSps is confirmed broken — on real data, offline.**
   Its masks are dominated by phantom wide horizontal streaks across the
   noise-floor tilt and band edges (frames 19/26/42), reproducing exactly the
   live symptom that prompted the RT fix ("detecting activity where there isn't
   any"). RT-vs-native frame IoU here is 0.06 mean — versus 0.87 median at the
   trained 245.76 rate in the composite study. The disagreement *is* the
   off-distribution failure, now demonstrated offline on real data.
3. **cuda_dino is very loose on real OTA at this rate**: 12–13 % of the band-time
   lit as speckle/blobs around (and far from) real bursts. Usable as a
   high-recall fallback, but expect noisy ROIs.
4. **coherent_power is clean** (0.56 %) and the live emit guard never fired on
   real traffic — guard-on live config is validated against real occupancy.
5. **Known residual: RT band-edge columns.** 17.5 % of the outer 3 % margins lit
   (intermittent full-height edge bands; visible in frame 19). The flatten
   doesn't fully tame the rolloff cliff at the outermost bins. Candidate fix:
   a DINO-side sideband ignore, mirroring the coherent detector's
   `ignore_sideband_percent`/`ignore_sideband_hz` params (zero mask columns in
   the outer ~3 %) — not yet implemented.

Regenerate: `python3 render_ota_masks.py [frames...]` (mask/tensor artifacts under
`/tmp/usrp_spectrograms/offline_eval/*/x410_ota_2g4_gain10_20260908/`).

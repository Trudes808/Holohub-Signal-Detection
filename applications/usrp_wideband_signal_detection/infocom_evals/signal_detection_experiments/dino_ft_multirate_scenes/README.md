# DINO-FT multi-rate / multi-scenario retrain (M6-S) — synthetic scenes calibrated on real data

Started 2026-11-02. Goal: fix the shortcomings observed with the deployed M5-S detector in extended
testing — frequency hoppers, adjacent wideband cellular channels (5 + 10 + 10 + 20 MHz LTE/5G downlinks),
ISM collisions (WiFi packets of every size colliding with BT / IoT), a drone overpowering one WiFi
channel, signals straddling the band edges, and **every X410 sample rate** (master clock 491.52 or 500
MHz, decimation 1 or any even factor, 5–500 MSps) — by generating those scenarios synthetically with
realism knobs measured on real captures, retraining the fine-tune, and scoring on held-out real
recordings.

All code lives in `dino_fine_tuning/src/` (paths below are relative to `dino_fine_tuning/`):

| file | role |
|---|---|
| `rates.py` | X410 rate set (101 rates) + the app's per-rate frame geometry (512 rows × `app_fft_size(rate)`; C++ rounding — Python `round()` picks 2048 where the app uses 3072 at 49–83 MSps; `frontend.auto_fft_size` fixed the same way) |
| `real_stats.py` → `data/real_stats/` | receiver floor templates (X410 @ 491.52 / 245.76 / 30.72 MSps), DC spur, dB-pixel spread; real 2.4 GHz / 1 GHz OTA component statistics (duration, bandwidth, level above floor) from the coherent-teacher tiles |
| `synth_mods.py` | modulators (PSK/QAM, GFSK, CP-OFDM with LTE/NR load patterns, chirps/LoRa/FMCW, FM, noise bands, CW) + the rate-agnostic `Placer` (interior: resample→mix; band edge: resample to 2R → mix → anti-alias decimate = the DDC cut-off), impairments (ramps, fading, phase noise, ADC clipping), receiver response FIR + DC spur |
| `gen_scenes.py` | scenario samplers → scenes (IQ + per-burst GT boxes), SigMF writer, preview |
| `build_dataset_scenes.py` | parallel scene synthesis + deployment front-end at each scene's rate → RFSegDataset tiles; held-out SigMF per scenario × rate bucket |
| `eval_scenes.py` | held-out evaluation at any rate (synthetic or real SigMF): region det@0.1 / det@0.5 by scenario, rate bucket, PSD-SNR, band-edge visibility, size, class; pixel P/R/IoU; overlays |
| `check_scene_levels.py` | GT sanity: realised vs intended level per emitter type |
| `scripts/run_m6.sh` | combine + train + evaluate + export, resumable |

## Scenarios (train mixture weights)

legacy_dense .10 · legacy_sparse .12 · cellular_adjacent .18 · ism_collisions .18 · drone_ism .12 ·
hoppers_focus .10 · band_edge_focus .10 · radar_pulses .02 · wideband_fill .02 · noise_only .06.
Rates: 30 % at 491.52 / 500 MSps, the rest log-uniform over the 101-rate X410 set. Levels are PSD SNR
(dB above the floor in the spectrogram): mixture 25 % 15–35 dB, 40 % 6–15 dB, 35 % −2–6 dB (real OTA
components: median 8 dB, p95 18 dB above floor). GT = filled box per burst (per hop / packet / TDD burst)
with the designed occupied bandwidth; lightly loaded LTE/NR carriers (load ≥ 0.3 / 0.2, full-width control
symbols every slot) keep the full-carrier box.

## Real statistics used (data/real_stats)

* X410 floor is flat to ±1 dB over the inner 90 % of the band at 491.52 / 245.76 / 30.72 MSps, −3 dB at the
  very edge; DC spur +9 dB × 2 wide bins at 491.52 (none at 245.76 / 30.72); dB-pixel std 5.4 dB (Gaussian).
  Templates are applied as a 513-tap FIR with ×0.6–1.8 amplitude jitter / mirroring (45 %), synthetic
  tilt + roll-off shapes (40 %), flat (15 %); DC spur 3–20 dB in 60 % of scenes.
* Real 2.4 GHz components (coherent teacher, 1680 tiles): duration p5/50/95 = 208 / 625 / 6000 µs,
  bandwidth 0.5 / 2.4 / 17.8 MHz, level 3.5 / 8.0 / 17.6 dB above floor, ~14 components per 10.7 ms tile.
  1 GHz: 417 / 1542 / 10667 µs, 0.5 / 1.0 / 3.8 MHz, 3.5 / 4.6 / 10.2 dB.

## Held-out material

* synthetic: `data/dataset_scenes/heldout_sigmf/<scenario>/` — 2 scenes per scenario × rate bucket (120).
* real, labelled: `data/hard_test_cases/` (B210 40 MSps 880 MHz cellular/LMR; X310 15.36 MSps LTE FDD and
  NR 10 MHz) — evaluated with `--real --assume-continuous` (the B210 labels mark a time window only);
  ARGUS `OOD_captures/LTE` (245.76 MSps, LTE-only, +46 / +26 / +6 / −4 dB) via `data/real_heldout/`.
* real, unlabelled (qualitative): DroneDetect_V2, T-PRIME overlapping WiFi, HCRO 915 MHz chunks.

## Dataset (built 2026-11-02, `data/dataset_scenes`, 18 GB incl. held-out SigMF)

3000 train / 300 val / 120 test scenes → **18 762 / 1 909 / 827 tiles** (256 × 1024 uint8 + GT), 12 workers,
15 min. Train tiles by scenario: ism_collisions 3517 · cellular_adjacent 3400 · drone_ism 2430 ·
legacy_sparse 2156 · legacy_dense 1976 · band_edge_focus 1815 · hoppers_focus 1727 · noise_only 1085 ·
radar_pulses 350 · wideband_fill 306. By rate bucket: 400–500 M 4321 · 60–150 M 4287 · 10–25 M 3429 ·
25–60 M 2533 · 5–10 M 2293 · 150–400 M 1899. Combined with the real OTA tiles (`data/dataset_ota`, 3360 train /
1112 val) → `data/dataset_m6`: **22 122 train / 3 021 val** (mask occupancy 17 %).

## Results

### M5-S baseline (deployed model) on the real held-out material

| set | metric | M5-S |
|---|---|---|
| hard_test_cases (B210 880 MHz cellular/LMR, X310 LTE-FDD, X310 NR; continuous labels) | pixel P / R / IoU | 0.988 / 0.330 / 0.328 |
| | region det@0.1 / det@0.5 (66 regions) | 65 % / 24 % |
| | per class det@0.1 / det@0.5 | CDMA 100 / 69 · LTE 78 / 16 · LTE-FDD 100 / 0 · NR 100 / 0 · LMR 0 / 0 · Unknown 0 / 0 |
| ARGUS LTE OOD (245.76 MSps) | LTE det@0.1 / det@0.5 | +46 dB 100 / 94 · +26 dB 100 / 100 · +6 dB 100 / 100 · −4 dB 0 / 0 |
| | pixel P / R / IoU (4 captures) | 0.781 / 0.472 / 0.417 |

Reading: on real wideband cellular downlinks M5-S finds the carrier (det@0.1) but paints fragments (det@0.5
near zero, pixel recall 0.33) — the "LTE/5G downlink" complaint; narrowband LMR is missed outright. On the
unlabelled recordings (`eval_out/unlabelled_M5S/overlays`) it misses the faint 35 MHz drone video pedestal,
drops the weaker of two overlapping WiFi transmitters, and marks the receiver DC spur as signal.

### M6-S (`checkpoints/M6S_multirate`, ViT-S ft_lastN, 18 epochs, 1.9 h, best val IoU 0.781) vs M5-S

Full tables: [`results/comparison.md`](results/comparison.md); overlays: `results/overlays/`.

**Real hard test cases** (B210 880 MHz cellular/LMR @40 MSps, X310 LTE-FDD + NR 10 MHz @15.36 MSps,
continuous labels, thr 0.6):

| metric | M5-S | **M6-S** |
|---|---|---|
| pixel precision / recall / IoU | 0.988 / 0.330 / 0.328 | 0.987 / **0.948** / **0.937** |
| region det@0.1 / det@0.5 (66 regions) | 65 % / 24 % | **88 % / 77 %** |
| LTE (32) · CDMA (16) · LTE-FDD · NR | 78/16 · 100/69 · 100/0 · 100/0 | **100/100 · 100/100 · 100/100** · 100/0 |
| LMR (8, 7 kHz = 0.2 model bins) · Unknown (8, 160 kHz) | 0 · 0 | 0 · 0 |

The 880 MHz capture is now segmented as solid carrier blocks (predicted occupancy 68.9 % vs 69.1 %
labelled) with the inter-carrier gaps left clean (`results/overlays/M6S_multirate_real_cellular_...png`).
The NR capture is a bursty TDD-like test model: M6-S follows the actual bursts, the time-continuous label
makes det@0.5 read 0 — a label artefact, not a miss (see the overlay). Remaining real misses: the sub-bin
LMR / paging carriers at 40 MSps (thin persistent lines the 1024-bin grid barely resolves).

**ARGUS LTE-only OOD** (245.76 MSps): LTE det@0.1/det@0.5 identical (100/100 at +6…+46 dB, 6/0 at −4 dB);
recall up (p06 0.72→0.96, p46 0.86→0.92) but precision down (0.78→0.63): M6-S paints ragged blobs around a
persistent spur at −116 MHz in the X410 roll-off region on quiet frames (+0.2–0.3 pp predicted occupancy per
capture). Known false-alarm mode; see "Open items".

**Legacy 491.52 held-out** (`eval_heldout.py`, the M3/M4/M5 benchmark, thr 0.6): no forgetting —
dense −3..3 / 3..10 / 10..20 / 20..32 dB = 92 / 94 / **97** / 100 % (M5-S 91 / 92 / 91 / 100), sparse
83 / 100 / 86 / 100 (same as M5-S); test-split IoU 0.920 vs 0.935.

**Synthetic held-out** (120 scenes, in-distribution for M6-S, thr 0.6):

| metric | M5-S | **M6-S** |
|---|---|---|
| pixel precision / recall / IoU | 0.930 / 0.227 / 0.223 | 0.925 / **0.671** / **0.636** |
| region det@0.1 / det@0.5 (headline, 10 015 regions) | 51.4 / 39.0 % | **85.7 / 81.8 %** |
| false-alarm occupancy, noise-only scenes | 0.114 % | **0.021 %** |
| cellular_adjacent · drone_ism · hoppers · band_edge · ism | 29/15 · 66/52 · 57/43 · 50/42 · 54/44 | **97/96 · 95/92 · 89/82 · 78/70 · 67/61** |
| 5–10 M · 10–25 M · 400–500 M rate buckets | 20/10 · 34/21 · 56/47 | **72/68 · 95/90 · 70/65** |
| PSD SNR 0–6 dB · 6–12 dB · band-edge cut 0.5–0.9 | 23/12 · 56/41 · 39/24 | **82/75 · 98/95 · 97/95** |

**Unlabelled recordings** (qualitative, `results/overlays/M6S_multirate_real_unlabelled_*`): the faint
35 MHz drone video pedestal (Mavic Air 2S) and the two ~2.5 MHz continuous drone bands (Mavic Pro, with
BT + WiFi interference) are now painted; on the two overlapping WiFi transmitters (T-PRIME, 62.5 MSps) the
weaker transmitter's non-overlapping part is partly recovered; the receiver DC spur is not flagged (M6-S was
trained with it as background; per the 2026-11-03 convention it should be flagged, which the v3 generator does).

### Open items

* Convention (user, 2026-11-03): **every emission in a capture is a signal, receiver spurs and the DC line
  included**; spur filtering happens downstream. The generator therefore labels its DC spur from v3 on
  (`add_dc_spur` returns the frequency, `make_scene` adds a `DC_spur` burst); M6-S/M7-S already mark the
  N320 DC line on the OTA captures. The Argus-LTE issue below is the noise painted *around* the spur, not
  the spur itself. Follow-up requested: a spur overlapping another emission should ideally come out as a
  **separate instance**; the binary mask merges overlaps, so this needs a second output channel
  (narrowband persistent lines, GT available from the generator's per-burst bandwidths) and instance
  splitting in the snipper — an operator-contract change, scoped separately.
* Band-edge spur false alarms (ARGUS LTE quiet frames). Raising the threshold to 0.7 does not remove them
  (ARGUS precision 0.63 → 0.64) and costs detection elsewhere (hard cases det@0.1 88 → 76 %, synthetic
  85.7 → 84.8 %), so the export keeps 0.6. Candidates: negatives with real receiver spurs in the roll-off
  region, more antenna-off captures from other radios.
* Sub-bin narrowband carriers at low rates (LMR at 40 MSps): ~0.2 model bins; needs the short-FFT /
  higher-resolution front-end or a per-bin "thin line" cue; out of scope for the segmenter grid.
* Wideband-fill scenes (carrier wider than the band): det 53 %; the robust floor estimate is the carrier
  itself, so the input looks like noise — a front-end limitation, not a model one.
* ism_collisions is the weakest synthetic scenario (67 %): dominated by weak (<6 dB) 1 MHz BT hops and
  ACK-length packets; real OTA BT at the measured 3.5–18 dB levels is covered.
* Not yet validated over the air or at the true 491.52 MSps live rate; `config_live_v3_dino_ft_m7s.yaml`
  is the drop-in A/B config (`sudo ./bash_scripts/run_live_demo.sh config_live_v3_dino_ft_m7s.yaml`).

## v2 generator (2026-11-03) → M7-S

After reviewing the M5-S / M6-S panels the user asked for three more real-world cases; the generator gained
(all in `gen_scenes.py` / `synth_mods.py`, scenario weights in `SCENARIO_WEIGHTS`):

* **`cellular_uplink`** (weight .10): FDD uplink bands with no downlink — 1–4 carriers of sporadic traffic
  from 2–12 UEs each (`ue_jobs`: PUSCH bursts of 1–8 TTIs and 1–100 PRB at random / persistent / hopping
  positions, 25 dB near-far spread, 2–40 % duty), PUCCH bursts hopping between the carrier edges, PRACH
  preambles (6 PRB, 1–3 ms), wideband single-symbol SRS, NB-IoT / LTE-M devices in the guard bands.
  The same UE model now fills the UL slots of every TDD carrier in `cellular_adjacent` (replacing the old
  single wide UL burst).
* **ISM near-far and co-channel collisions**: each AP gets, with p = 0.5, a partner on the same or an
  overlapping channel 15–40 dB weaker (or 10–30 dB stronger) with independent timing, and with p = 0.25 a
  hidden node on the same channel at 40–90 % duty. 35 % of ISM scenes are now a **5 GHz plan** (UNII
  channels, 20/40/80/160 MHz BSSs, LAA / NR-U listen-before-talk bursts, occasional DFS radar).
* **`lmr_band`** (.07) and the **`nbfh` hopper kind**: DMR / TETRA / P25 / NXDN / analog / generic FSK on
  the 6.25–25 kHz rasters, push-to-talk transmissions of 0.3–5 s, TDMA slot structure (DMR 2 × 30 ms,
  TETRA 4 × 14.17 ms), frequency hopping with 10–500 ms dwells over 4–64 channels, FLEX-class pagers,
  narrowband data, 40 dB power spread.
* **`broadcast_fm`** (.02): 8–40 stations on the 200 kHz raster, 75 kHz deviation, 40 dB spread.
* **Time-varying downlink load** (`mod_ofdm p_idle / load_jitter`): idle slots and jittered allocation
  on 40 % of LTE/NR carriers; **in-band narrowband interferers** (CW / FSK / PSK inside a carrier, p = 0.3).
* **Impairments on every emitter** (`resolve_impairments`): static 2–6 tap multipath (50 ns–1.5 µs) on
  half of the wideband content → notches / ripple; Rapp PA compression (2–8 dB backoff) on half of the
  strong emitters → spectral shoulders.

Dataset `data/dataset_scenes_v2` (seed 23, same sizes: 3000 / 300 scenes + 2 held-out per scenario × rate
bucket, now 13 scenarios → 156 held-out scenes) + the OTA tiles → `data/dataset_m7`; model
`checkpoints/M7S_scenes_v2` (same recipe as M6-S). M5-S and M6-S are re-scored on the v2 held-out set
(`eval_out/scenes2_*`) and on the real sets; tables in `results/comparison_M7S_scenes_v2.md`.

### M7-S results (`checkpoints/M7S_scenes_v2`, 18 epochs, best val IoU 0.728 on the harder v2 validation set)

Full tables: [`results/comparison_M7S_scenes_v2.md`](results/comparison_M7S_scenes_v2.md) (M5-S, M6-S and M7-S
all scored on the v2 held-out set and the same real sets).

| set | metric | M5-S | M6-S | **M7-S** |
|---|---|---|---|---|
| v2 synthetic held-out (156 scenes, 13 scenarios) | region det@0.1 / det@0.5 | 65 / 49 % | 86 / 81 % | **87 / 81 %** |
| | pixel precision / recall | 0.892 / 0.284 | 0.890 / 0.730 | **0.915** / 0.728 |
| | false-alarm occupancy, noise-only | 0.125 % | 0.009 % | **0.004 %** |
| | lmr_band · cellular_uplink · ism_collisions (det@0.1) | 49 · 62 · 68 | 56 · 77 · 84 | **64** · 77 · **85** |
| real hard test cases | region det@0.1 / det@0.5 | 65 / 24 % | 76 / 74 % | **83 / 80 %** |
| | pixel precision / recall / IoU | 0.988 / 0.330 / 0.328 | 0.987 / 0.948 / 0.937 | 0.969 / 0.950 / 0.922 |
| ARGUS LTE OOD | pixel precision / IoU | 0.781 / 0.417 | 0.625 / 0.374 | **0.764 / 0.422** |
| | region det@0.1 / det@0.5 | 82 / 71 % | 84 / 78 % | 84 / 76 % |
| legacy 491.52 held-out | dense −3..3 / 3..10 / 10..20 dB | 91 / 92 / 91 % | 92 / 94 / 97 % | 92 / 94 / **99 %** |

Reading: the new scenarios cost nothing on the old ones, M7-S repairs the band-edge false-alarm regression
M6-S showed on the Argus LTE captures (precision back to 0.76), and gains on the land-mobile band and the
real test recordings. The uplink scenario is the hardest remaining synthetic case (77 % det@0.1: sub-6 dB
single-PRB bursts). **M7-S is the candidate for the OTA A/B**; `config_live_v3_dino_ft_m7s.yaml` points at
its export.

## OTA stress test on the N320 (2026-11-03)

User request after the M7-S run: short real snippets across bands at several rates and centre offsets, raw
IQ stored with the masks, plus mixed real + synthetic UAV cases. Tooling: `bash_scripts/capture_n320_sigmf.sh`
(one snippet → ci16 SigMF with `argus:*` fields), `bash_scripts/capture_n320_matrix.sh` (the 30-capture
matrix with per-band gain probing), `dino_fine_tuning/src/run_ota_detections.py` (M5-S / M6-S / M7-S masks
as packed `.npz` + overlays + 5-row panels + `detections.json` **inside each capture folder**),
`src/overlay_synthetic_uav.py` (OcuSync-class video + control hopper added to a real 2.4 / 5 GHz capture at a
set PSD SNR above the measured floor; only the synthetic bursts are labelled), `scripts/ota_stress_pipeline.sh`.
Data: `~/Datasets/ARGUS/OOD_captures/OTA/n320_<band>_<desc>_c<MHz>M_r<MSps>M_g<gain>_ch<0|1>/`.
Radio: N320 serial 3459703 (addr 192.168.10.2), ch0 = RF0 + VERT2450, ch1 = RF1 + VERT900, gains 40–60 dB
(auto-probed to rms −40…−20 dBFS, no clipping), master clock 200 / 245.76 / 250 MHz for exact rates.

### OTA results (30 captures, 12 frames each, thr 0.6; `results/ota_n320_detections.md`)

Occupancy of the three masks tells the story (no labels on the real captures):

| capture | M5-S | M6-S | **M7-S** | what the panel shows |
|---|---|---|---|---|
| cell850 downlink, 25 MSps, centred | 2.0 % | 76.7 % | 92.5 % | band full of LTE carriers: M7-S paints whole carriers with the guard gap left clean; M5-S fragments |
| cell700 downlink, 31.25 MSps | 26 % | 67 % | 68 % | three carriers (one weak) as blocks |
| FM band, 50 MSps wide / 10 MSps partial | 2.4 / 1.9 % | 15 / 55 % | 3.4 / 13.6 % | M7-S marks the stations (slightly wide), M6-S smears between them / fills half a tile |
| cell850 **uplink**, 25 MSps, gain 60 | 0.5 % | **94.8 %** | 1.1 % | empty band with narrow lines and a few UE bursts: M6-S fills the frame, M7-S is right |
| 900 MHz ISM, 30.72 MSps, gain 60 | 0.9 % | **99.4 %** | **49.7 %** | quiet band with lines: M6-S fills everything, M7-S fills one of the two tiles |
| quiet 1350 MHz reference | 0.4 % | 4.9 % | 0.5 % | only the DC line (flagged by design) |
| 2.4 GHz ISM (6 looks, 12.5–200 MSps) | 3.7–8.3 % | 10–17 % | 7.9–13.4 % | WiFi packets, BT hops, DC line |
| 5 GHz (4 looks) | 0.7–3.5 % | 1.6–4.9 % | 2.1–5.3 % | mostly quiet; sparse WiFi |

**Mixed real + synthetic UAV** (OcuSync-class video + control hopper added to the real 2.4 / 5 GHz captures,
`results/ota_synth_uav_comparison.md`): video bursts found by M6-S and M7-S in 100 % of cases (det@0.5) from
12 dB above the floor up, control hops 60–100 %; M5-S 2–99 % and 3–82 %.

**Finding → v3:** the remaining failure of both retrains is a *tile-level fill* on quiet, high-gain frames
(M6-S often, M7-S on one tile of the 900 MHz ISM capture). Cause: the `wideband_fill` scenario labelled a band
filled edge-to-edge by a noise-like carrier as all-signal, but after per-frame robust normalisation such a
frame is indistinguishable from an empty one, so the model learned an arbitrary rule for flat frames. v3
keeps a carrier edge in view in that scenario, doubles `noise_only` (half of it quiet bands with 1–4 narrow
lines), labels the DC spur, and retrains as **M8-S** (chain running).


### M8-S (v3 generator) — outcome

`checkpoints/M8S_scenes_v3` (val IoU 0.825 on the v3 set; `results/comparison_M8S_scenes_v3.md`): best synthetic
numbers of the lineage (v3 held-out pixel IoU 0.860 vs M7-S 0.818, band-edge 76/67 vs 54/45, legacy_sparse
87/85 vs 66/60, quiet-band lines 84 % vs 38 %; real hard cases IoU 0.940) and the Argus-LTE band-edge blobs are
gone (its lower precision there, 0.66, is now the DC line it flags by convention). **But on real air it did not
cure the quiet-band fill**: 900 MHz ISM 46 % (M7-S 38 %), and a 44 % fill on the 499–531 MHz HDTV look that
M7-S handled at 14 %; FM wide 10 % vs 3 %. Mixed real + synthetic UAV: equal to M7-S (video 81–100 %, hops
31–100 %). **M7-S remains the candidate.** The four-model OTA table is `results/ota_n320_detections.md`.

Reading: the fill mode is not explained by the `wideband_fill` labels alone. The quiet N320 captures at
60 dB gain show a comb of faint narrowband emitters and receiver ripple that no synthetic floor template
reproduces; the model has no real examples of such frames with correct labels. Next step with the best odds:
fold the N320 quiet/high-gain captures into training with coherent-teacher labels (the M3→M4 recipe,
`label_ota.sh` → `build_dataset_ota.py`), plus antenna-off N320 negatives at the same gains.

### Live-pipeline note

The live configs' invalid-frame guard (`invalid_frame_min_occupancy: 0.40`, tuned for M5-S) would suppress
legitimate frames with M7-S, which paints 70–94 % of a full 850/700 MHz downlink band. `config_live_v3_dino_ft_m7s.yaml`
now sets it to 0.98 (only near-total broadband garbage trips it).

## test_1 "whole-frame fill" investigation (2026-11-05)

User report: M7-S occasionally paints nearly a whole frame on `OOD_captures/OTA/test_1` (2.45 GHz, 245.76 MSps)
and could do better at low SNR. Findings (`dino_fine_tuning/src/scan_capture.py`, `eval_out/scan_test1*`):

* **Not reproducible on the deployment path.** Python twin: M7-S occupancy median 2.1 %, max 5.8 % over all
  140 frames, 0 of 280 tiles above 50 %; same on test_2 / test_3. The C++ operator in the container (standard
  and the sf2048 time-pool front end) agrees to 0.02 pp. The fills on the other machine were at tile indices
  224 and 1710 — test_1 has 2812 tiles only under the **native 1024-point geometry**; run that way (fixed dB
  clip, no flatten/robust floor) M7-S fills exactly those tiles 100 % (454 of 2812). Front-end mismatch, not the
  weights; `src/run_ts_on_sigmf.py` is the portable correct path (README §3a).
* **Real remaining defects on test_1**: (1) row-leak false positives — strong bursts raise the probability along
  their whole time row, so streaks appear across quiet spectrum (frames 41 / 132, ~4 % of the frame, prob maps
  in `eval_out/scan_test1/frame_*.png`); (2) low-SNR recall — vs the coherent teacher (container run) M7-S misses
  ~16 % of coherent pixels per frame, median level ~7 dB above the floor, mostly fringes of bursts and thin
  weak lines; M7-S-only pixels sit at ~5–6 dB (box fill, as trained).
* **Live-ingest artefacts do cause fills**: zero-filling 30 % of the 1024-sample packets gives 24–54 % occupancy,
  a 20 dB mid-frame gain step 16–35 %; 2–10 % random loss and contiguous outages are harmless. None of these are
  in the training data; with the invalid-frame guard raised to 0.98 for M7-S they would reach the mask.

Actions: **M9S_headv2** = `SegHeadV2` (full-resolution input stem [x, x − local 129-bin mean], one 3×3 conv +
1×1 fuse, +3 ms per frame frozen) + `fp_quiet_w 2.0` (extra BCE weight on background pixels at floor level),
trained on `dataset_m8`; generator **v4** adds the ingest artefacts (random packet zero-fill, contiguous outages
with cleared GT, gain steps) for the next retrain.

### Quiet-band fill: mechanism found (2026-11-05)

Synthetic test: a noise-only band with an irregular comb of 40+ persistent CW lines (3–30 dB) at 10–30 MSps makes
M7-S paint 78–98 % of the tile and M8-S 77–92 % (ground truth 12–15 %); the same band with ~29 lines gets 3 %.
The models learned "dense comb of lines = resolved OFDM subcarriers at a low sample rate = one filled carrier",
which is exactly what the quiet, high-gain 900 MHz ISM and HDTV looks on the N320 contain (faint smart-meter /
spur combs). Real terminated-input (TERM) N320 recordings look Gaussian after the front end (identical p50/p95/
p99 to synthetic noise), so the texture itself is not the trigger; the comb is.

**v5 generator** (`gen_scenes.py`, `real_backgrounds.py`): spur combs (5–60 irregular, partly drifting / gated
lines) in 50 % of noise-only and 6 % of all other scenes, each line labelled; real TERM backgrounds (28 rates ×
3 gains × 2 channels, `bash_scripts/capture_n320_term_backgrounds.sh`) under 50 % of the scenes at ≤ 250 MSps,
with their DC line and persistent spurs labelled; plus the v4 ingest artefacts. **M10S_v5** = SegHeadV2 +
fp_quiet 2.0 on `dataset_m10` (v5 + OTA tiles), 20 epochs, queued behind M9-S.

### M9-S (SegHeadV2 + quiet-pixel penalty, dataset_m8) — outcome (2026-11-05 evening)

`checkpoints/M9S_headv2`, val IoU 0.8435 (M8-S 0.825 on the same set). Synthetic v3 held-out: pixel precision
0.949 / IoU 0.883 (M8 0.919 / 0.860), noise-only false alarms 0.023 % (M8 0.047 %), region det@0.1 84.8 % (flat),
det@0.5 below 0 dB 34 % (M8 37 %). Real: hard cases pixel P 0.986 / R 0.946; the bursty NR carrier now fully
covered; **the 7 kHz LMR lines are lost** (M7-S half-covered them by smearing); Argus LTE precision back to 0.76
with the best det@0.5 of the lineage (79 %). test_1 row-leak streaks halved (frame 131: 5.8 % → 3.1 %).
**Quiet-band fills unchanged or worse** (900 MHz ISM 69 %, HDTV 44 %; comb probe 25 % of tiles at 10/31 MSps):
the head and loss do not fix a data problem.

**Fill guard rejected.** A per-tile post-filter (mask covers ≥ 50 % while the masked pixels sit at floor level
→ keep only pixels above floor + 6 dB; `src/postfilter.py`, `--fill-guard` flags) removes the ISM fill (33 → 17 %)
but also guts the legitimate full 850 MHz downlink band (94 → 41 %) and costs 17 points of synthetic recall: after
per-frame robust normalisation a band filled edge to edge with carriers is indistinguishable from a quiet band with a
paint-over. The absolute level is gone before the model sees the image. Principled fix for a later operator change:
a **second input channel carrying the fixed-clip (absolute) dB image** (`.meta.json input_channels: 2`, operator
builds [2, rows, nfft]); until then the fill mode is addressed through data (M10-S).

### M10-S (v5 generator: real TERM backgrounds, labelled spur combs, ingest artefacts; SegHeadV2 + fp_quiet 2.0) — outcome (2026-11-05 night)

`checkpoints/M10S_v5` (dataset_m10 = v5 scenes + OTA tiles, 20 epochs, 2.7 h, best val IoU 0.737 on the harder v5
validation set; `results/comparison_M10S_v5.md`, probe + OTA + scans below). **Strongest model of the lineage on
real data and the first one without the quiet-band fill mode:**

* **Comb-fill probe** (`src/probe_comb_fill.py`, quiet band + 40–60 irregular persistent lines at 10 / 31 / 61 MSps):
  M7-S / M9-S fill 12–25 % of the tiles at 10–31 MSps; **M10-S fills none** (mean occupancy 7.4 %, line recall 47 %).
* **Real hard cases**: region det@0.1 / det@0.5 **92.4 / 87.9 %** (M7-S 83.3 / 80.3, M9-S 75.8 / 75.8), pixel
  P 0.978 R 0.940 IoU 0.921; the eight "Unknown" emitters of the 880 MHz recording are now found (100 %, were 0–12 %);
  LMR 38 / 12 % (M7-S 50 / 50: the 7 kHz lines are at the resolution limit at 40 MSps); NR det@0.5 0 is the
  time-continuous-label artefact (the model follows the bursts).
* **Argus LTE OOD**: precision 0.797 (best), det@0.5 78.3 %. **Legacy 491.52 held-out** @0.6: P 0.980 R 0.939 IoU 0.921.
* **Synthetic v5 held-out**: pixel precision 0.941 / recall 0.842 / IoU 0.800 (M7-S 0.800 / 0.888 / 0.726, M9-S
  0.849 / 0.907 / 0.782), det@0.1 80.4 % (M7-S 78.6, M9-S 80.6), noise-only occupancy 5.1 % (M7-S 11.8, M9-S 9.8;
  these v5 noise-only scenes carry labelled spur combs, so the number is paint-around-lines, not pure false alarm).
  Like-for-like on the v3 held-out: P 0.951 IoU 0.845 det@0.5 78.2 %.
* **N320 OTA** (`results/ota_n320_detections.md`, now M5/M7/M9/M10): 900 MHz ISM @30.72 g60 **18 %** (M7-S 38, M9-S 69) —
  the lower tile is now marked line by line, the upper tile (dense smart-meter / FHSS comb) is still painted whole;
  cell850 uplink narrow 9.7 % (M7-S 22.5), FM partial 5.1 % (13.9), quiet 1350 MHz 0.6 % (0.5); full downlink bands
  unchanged at 70–95 %. **HDTV 499–531 MHz 44 % is correct, not a fill**: the look holds two weak ATSC channels
  (RF 21 / 22; pilots at −2.69 and +9.3 MHz baseband) that M7-S under-detects at 14 %.
* **test_1 / 2 / 3 scans** (`eval_out/scan_test_*_m10`): test_1 max occupancy 5.3 % (M7-S 5.8), the frame-131 row-leak
  streak 3.3 % (M7-S 5.8); test_3 unchanged (max 5.1 %). test_2 max 18 % (M7-S 6.5) on frames 9–13: a faint ~75 MHz
  wide carrier with a ~5 ms TDD on/off pattern at +60…+125 MHz that M7-S mostly misses (visual check of frame 11;
  the file has no labels) — consistent with better low-SNR recall of wide carriers, not with the fill mode.
* **Mixed real + synthetic UAV** (`results/ota_synth_uav_comparison.md`, now with M10-S): det@0.1 / det@0.5 97.0 / 92.4 %
  (M7-S 97.1 / 94.7, M8-S 96.6 / 92.6); drone control hops 90 / 80 %, video 99 / 95 % — equal within noise.

**Regressions on the synthetic held-out** (same v5 set, M7-S → M10-S): cellular_adjacent det@0.1 91 → 84 % (5–10 MSps
98.6 → 80.6 %; nr_dl_tdd 100 → 49 %, 5G_Downlink det@0.5 98 → 61 %), PSD SNR < 0 dB 47 → 42 %, mostly-cut edges
70 → 64 %, pixel recall 0.888 → 0.842. Per scene the large drops are the scenes carrying ingest artefacts (packet
zero-fill, gain steps); by background kind the real-TERM scenes are unchanged (85.7 → 86.3 %) while synthetic
floors drop 94.5 → 90.7 %. **Cause: the v4/v5 gain-step artefact** (band-wide level step, GT 0) is indistinguishable
from a band-filling carrier switching on in a narrow look (a 10 MHz NR TDD carrier at 10 MSps *is* a band-wide level
step) — the same ambiguity that killed the fill guard — so it taught "level step = background". The quiet-pixel
weight (2.0) buys the precision and costs the sub-0 dB recall.

**Operating threshold for M10-S** (`eval_out/*_M10S_v5_thr05|thr04`): lowering the mask threshold from 0.6 to 0.5 / 0.4
leaves the held-out pixel IoU unchanged (0.800 / 0.801 / 0.801) and trades precision 0.941 → 0.930 → 0.918 for recall
0.842 → 0.852 → 0.862, det@0.5 73.2 → 74.3 → 75.3 %, sub-0 dB det@0.5 25.7 → 27.5 → 29.2 % (M7-S 33.0), mostly-cut edges
57.5 → 58.9 → 59.9 % (M7-S 63.6). Real sets: hard cases det@0.5 87.9 → 89.4 → 90.9 % at precision 0.978 → 0.973 → 0.968,
Argus det@0.5 78.3 → 79.3 → 79.3 % at precision 0.797 → 0.788 → 0.777; legacy 491.52 sweep F1 0.959 → 0.960 → 0.960.
**0.5 is the better operating point for this lineage** (IoU-neutral, +1 pp detection); the threshold is a per-config
field (`threshold:` in the live YAML, `--threshold` at export), so it does not change the operator.

## v6 generator (2026-11-06) → M11-S

What the remaining 900 MHz ISM fill is (`eval_out/scan_ism900_m10`, all 58 frames): M10-S paints 7 of 116 tiles
above 50 % (M7-S 34 of 116; mean occupancy 16.8 % vs 31.9 %), the filled tiles are not time-contiguous and the two
halves of such a frame have identical level statistics (median −7.6 dB in both) — a borderline tile-level decision,
not a level step. The look carries **30 persistent lines on an exact 1.00 MHz grid** (median 4 dB, up to 10 dB, a few
extra teeth): clock / reference harmonics reaching the antenna (the terminated-input recordings of the same radio
show none of them). The v5 combs are irregular, so a *regular* grid of weak lines — the resolved-subcarrier signature
— was never shown with line-by-line labels.

Changes, all in `gen_scenes.py`: (1) **gain-step artefact removed** (zero-fill and contiguous outages stay — a zero
row is unambiguous); (2) **dense FHSS networks** (`fhss_network_jobs`: 8–30 GFSK hoppers of 25–200 kHz over a
2–60 MHz span of 16–100 channels, dwell 1–50 ms, duty 20–90 %, ~40 dB level spread, every burst its own box) in 35 %
of `hoppers_focus`, 30 % of `lmr_band` and 15 % of 2.4 GHz `ism_collisions` scenes; (3) **regular spur combs**: 45 %
of the generated combs are now a grid (spacing 0.2–2.5 MHz, clipped to 4–60 teeth across the look, random offset,
10 % teeth missing, 0–3 unrelated spurs, levels 3–9 dB with a slow ripple and 1.5 dB jitter, no gating), every tooth
labelled. Training recipe = M10-S with `fp_quiet_w` 2.0 → 1.0 to give back some low-SNR recall. Chain: `data/dataset_scenes_v6` (3000 train / 300 val scenes, seed 61) → `dataset_m11` →
`checkpoints/M11S_v6` (20 epochs) → `eval_out/scenes6_*` vs M7-S / M10-S → comb probe → OTA rerun (M5 / M7 / M10 / M11)
→ test_1/2/3 scans (`eval_out/scan_test_*_m11`) → full 900 MHz ISM scan (`eval_out/scan_ism900_m11`).

### M11-S (v6 generator, SegHeadV2, fp_quiet 1.0) — outcome (2026-11-06 early morning)

`checkpoints/M11S_v6` (dataset_m11 = v6 scenes + OTA tiles, 22.5 k training tiles, 20 epochs, 2.5 h, best val IoU 0.764
at epoch 17; `results/comparison_M11S_v6.md`; export `weights/finetuned_dino_m11s_v6_bf16.ts`).

**Like-for-like on the v6 held-out (156 scenes, thr 0.6)** — M7-S / M10-S / M11-S: pixel precision 0.820 / 0.928 / **0.939**,
recall 0.885 / 0.863 / **0.889**, IoU 0.741 / 0.809 / **0.841**; region det@0.1 78.8 / 81.2 / **82.8 %**, det@0.5 72.2 / 73.4 /
**74.7 %**; scenario macro det@0.1 75.2 / 79.3 / **82.8 %**. M11-S leads every sample-rate bucket but 25–60 MSps (−0.7 pp
vs M10-S), and every SNR bucket above 0 dB (0–6 dB det@0.5 57 / 57 / **59 %**); below 0 dB it is back to 33.6 % (M10-S
32.0, M7-S 37.9). Emitters: `fhss_net` 73 / 73 / **78.5 %**, `spur_comb` 38 / 52 / **55 %**, `rx_dc` 24 / 67 / **69 %**,
`fill` 93 / 93 / **98 %**; drones, WiFi, Bluetooth, GSM, FM, TDD downlinks equal. The recall M10-S had given up is back
at M10-S's precision.

**The two remaining gaps against M7-S are one scene each.** PUSCH det@0.5 72 / 60 / 59 % and band-edge-cut det@0.5
97 / 85 / 92 % come from `cellular_uplink_5-10M_9.26M_s200046` (318 PUSCH boxes, 69 % GT occupancy: a 9.26 MSps look
filled edge to edge by dense uplink traffic, M7-S 100 % / M11-S 41 %) and `cellular_adjacent_5-10M_7.35M_s200035`
(39 / 22 %). On the first scene M7-S paints **99.95 %** of the frame (pixel precision 0.64, recall 1.00) — the
whole-band paint-over the user reported as a false-positive mode — while M11-S marks 31 % selectively. Without these two
scenes PUSCH det@0.5 is 67.8 / 66.3 / 68.6 %, cut signals 97.4 / 87.1 / 95.5 %, headline det@0.5 78.1 / 77.6 / **83.4 %**.

**Real sets**: hard cases pixel P **0.987** R **0.943** IoU **0.931** (M10-S 0.978 / 0.940 / 0.921), det@0.1 93.9 %, det@0.5
80.3 % (M10-S 87.9: the eight "Unknown" edge lines of the 880 MHz recording are found but covered < 50 % of their generous
boxes; LMR 50 / 25 %); Argus LTE P 0.779 R 0.482 IoU **0.424** (best), det@0.5 79.3 %; legacy 491.52 @0.6 P 0.960 R 0.956
IoU 0.919 (M10-S 0.980 / 0.939 / 0.921); mixed real + synthetic UAV det@0.1 / det@0.5 96.1 / 91.9 % (M10-S 97.0 / 92.4).
**Comb probe**: 0 % filled tiles, mean occupancy 10.5 %, line recall **61.9 %** (M10-S 54.4, M7-S 45.6).

**N320 OTA** (`results/ota_n320_detections.md`, five-model table; full scans in `eval_out/scan_*_m11`):

* 900 MHz ISM @30.72 g60, all 58 frames: **0 of 116 tiles above 50 %** (M10-S 7, M7-S 34), mean occupancy 8.2 %, p95 9.7 %
  — the regular 1 MHz comb is now marked line by line in every frame. Fixed.
* **New fill on the narrow 850 MHz uplink look** (10.24 MSps, g60, same 1 MHz comb plus band-wide low-duty bursts): 12 of 78
  tiles above 50 % (M7-S 13, **M10-S 0**), mean occupancy 22 %. The filled tiles hold a band-wide burst row or two; the model
  paints the floor between them. Raising the mask threshold does not cure it (11 tiles at 0.7, 8 at 0.8 — the fills are
  confident).
* **Quiet 1350 MHz reference**: occupancy 4.8 % (M5/M7/M10 0.4–0.6 %): M11-S paints the ±1 MHz phase-noise pedestal around the
  DC line in every frame (no tile above 50 %). Noise painted *around* a spur — the one spur-related defect that counts.
* HDTV 499–531: 47 % (M10-S 44 %, both channels covered, 7 tiles above 50 % vs 1). Downlink bands: M11-S covers the loaded
  parts and leaves more of the lightly loaded carriers dark (AWS 2107–2157 56 % vs M10-S 77 %, M7-S 64 %; cell1900 72.5 vs
  75.0; cell700 65.1 vs 70.6) — fewer tiles above 80 % (29 vs 43 over the 30 captures), i.e. less "full-carrier box" completion.
  Everything else (2.4 / 5 GHz ISM, FM, ADS-B, uplinks at 20–61 MSps) within ±2 pp of M10-S.
* test_1 / 2 / 3: max occupancy 5.6 / 10.0 / 6.2 % (M10-S 5.3 / 18.0 / 5.1), no tile above 50 %; the frame-131 streak 3.8 %.

**Reading.** v6 data fixed what it targeted (the comb, hopper networks, the gain-step label noise) and M11-S is the best model
on every labelled set, but lowering the quiet-pixel weight from 2.0 to 1.0 gave back the floor-level painting that M10-S had
suppressed: one new tile-fill capture and the DC pedestal. The two levers are separable, so **M12S_v6q2 = v6 data +
`fp_quiet_w 2.0`** (everything else identical) was launched at 03:10; its chain adds the full scans of the three sensitive
captures (ism900, narrow uplink, quiet 1350) to the usual evaluation. Decision rule unchanged: zero fills on the probe and on
the three sensitive captures, labelled-set numbers at M11-S level. Until then **M7-S stays the live candidate config**;
M10-S and M11-S exports are kept for A/B.

### M12-S (v6 data, fp_quiet 2.0) — outcome (2026-11-06 06:00): the loss weight is not the lever

`checkpoints/M12S_v6q2` (same dataset_m11, `fp_quiet_w` 2.0, best val IoU 0.764 = M11-S). Labelled sets identical to M11-S
within noise: v6 held-out P 0.946 R 0.885 IoU 0.842 (M11-S 0.939 / 0.889 / 0.841), det@0.1 82.5 % (82.8), hard cases
IoU 0.920 (0.931), Argus IoU 0.425 (0.424), comb probe 0 % fills / line recall 58.9 %. **But the two OTA regressions are
unchanged**: narrow 850 MHz uplink look 12 of 78 tiles above 50 % over all 39 frames (M11-S 12, M10-S 0; mean occupancy
20.2 %), quiet 1350 MHz pedestal 5.9 % (M11-S 4.8, M10-S 0.6); 900 MHz ISM stays clean (0 of 116). So the regressions come
from the v6 *data*, not from the quiet-pixel weight.

Mechanism (`eval_out/scan_cell850_uplink_825-835_m11`, per tile): the tiles M11-S fills (83–97 %) are exactly the tiles that
contain a few **band-wide burst rows** (12–21 rows more than 6 dB above the floor; M10-S paints 27–45 % of the same tiles,
tiles without such rows stay at 3–7 % for both); the two halves of the frame differ by ≤ 0.5 dB otherwise. In v5 the
gain-step artefact (band-wide level change, GT 0, 5 % of scenes) was the only thing teaching "an elevated region without
spectral structure is background", and that lesson generalised to the band-wide-burst tiles and to the DC phase-noise
pedestal. Removing it in v6 as label noise took the lesson away. The ambiguity it creates (a Gaussian-like carrier filling
a narrow look edge to edge switches on = a level step) is real but rare in practice and absent at the 491.52 MSps live
rate, where no carrier fills the band; the fill mode shows at every rate. **Policy: keep the gain-step negatives.**

## v7 generator (2026-11-06) → M13-S

v6 + (1) the **gain step restored** (p 0.05, as in v4/v5) and (2) **short band-wide bursts** (`wideband_burst_jobs`: 1–8
edge-to-edge OFDM / noise bursts of 0.3–5 ms at 3–20 dB, labelled on their rows only — the narrow-look uplink pattern that
triggered the fill — in 25 % of `cellular_uplink` scenes at ≤ 20 MSps, 8 % above, 10 % of `noise_only`, 8 % of `lmr_band`).
A short band-wide burst (paint its rows) and a persistent gain step (paint nothing) are distinguishable by duration, so the
labels stay consistent. **M13S_v7** = SegHeadV2 + `fp_quiet_w` 2.0 on `dataset_m13` (v7 + OTA tiles); chain = build
(`dataset_scenes_v7`, seed 71) → train → scenes7 eval vs M10 / M11 / M12 → probe → OTA rerun (six models) → scans of
test_1/2/3 and the three sensitive captures. Decision rule: zero fills on the probe and on ism900 / narrow uplink / quiet 1350,
labelled-set numbers at M11-S level.

### Deploy

`dino_fine_tuning/weights/finetuned_dino_m7s_scenes_v2_bf16.ts` (+ `.meta.json`, eager-vs-traced IoU 1.000,
threshold 0.6, trained_rate_range 5–500 MSps; the M6-S export `finetuned_dino_m6s_multirate_bf16.ts` is kept
for A/B). Same geometry contract and front-end as M5-S, so the only config change is `model_script_path`
(done in `config_live_v3_dino_ft_m7s.yaml` and `config_live_v3_dino_ft_m7s_two_channel.yaml`). Since 2026-11-06
the demo's DINO switch (conductor single + dual, loopback supervisor and runners) loads M7-S; M5-S is off the switch.

### Reproduce

```bash
cd dino_fine_tuning
PYTHONPATH=src .venv-ml/bin/python src/real_stats.py
PYTHONPATH=src .venv-ml/bin/python src/build_dataset_scenes.py --out-dir data/dataset_scenes --workers 12
scripts/run_m6.sh all                      # combine + train + eval (vs M5-S) + export
scripts/post_train_m6.sh                   # unlabelled overlays + results/comparison.md + copies into results/
```

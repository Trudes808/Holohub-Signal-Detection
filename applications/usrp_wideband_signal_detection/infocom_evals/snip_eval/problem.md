# Signal-snipper bounding-box over-count at low SNR

> **Update 2026-07-23 (second investigation):** the pipeline accounting was audited end-to-end and
> is correct — the footprint numbers really are what the bounding boxes imply. However, one earlier
> conclusion below is **corrected**: the persistent 48 MHz streak is **not** a real transmitted
> signal. An attenuation-sweep test (`streak_forensics.py`) proves it is a **receiver-generated CW
> clock spur** at exactly 2048 MHz (= 2^11 MHz) absolute, ~117 Hz wide, part of an exact spur
> family at 2048 ± k×20.48 MHz. See "Streak provenance — RESOLVED" and "Fix options quantified"
> below.

## TL;DR

While measuring the real per-detector storage footprint of the `signal_snipper`, `coherent_power`
appears to store a large amount of data at low SNR (~0.9 TB/hr in frequency mode, ~5 TB/hr in
time-slice mode at −16 dB), while the fine-tuned DINO detector stores ~0. Investigation shows this
gap is **not** explained by how much each detector flags — both flag ~0.5 % of the spectrogram at
low SNR. It is an artifact of the snipper representing every detection as the **bounding box of a
4-connected mask component** and storing that whole rectangle. At low SNR a persistent narrowband
component and transient wideband bursts get fused into a single connected component whose bounding
box spans ~30 MHz × 21 ms but is only ~1.4 % filled — so the snipper stores ~**51× more bytes than
the actual detected content**, in ~76 % of frames, identically.

## Background & setup

- **Pipeline under test:** `mask_replay_detector` → `signal_snipper` → `sigmf_file_sink`. Detector
  masks generated offline are replayed into the real C++ snipper, which cuts detected regions out of
  the wideband IQ. Footprint-only (`write_iq: false`), so only tiny `.sigmf-meta` files are written,
  each carrying the stored sample counts; bytes are reconstructed exactly from those.
- **Grid / units.** Masks are `512 rows (time) × 10240 cols (freq)` per frame.
  - freq resolution: `fs / cols = 245.76e6 / 10240 = 24 kHz` per column
  - time resolution: `samples_per_row / fs = 10240 / 245.76e6 = 41.67 µs` per row
  - frame = `512 × 10240 = 5,242,880` samples = `21.33 ms`
  - `fs = 245.76 MHz`, `cf32` (8 B/sample); naive save-all = 7.078 TB/hr.
- **SNR axis.** `snr_db = 54.02 − attenuation_db`. So `atten_70 = −16 dB`, `atten_65 = −11 dB`,
  `atten_60 = −6 dB`, `atten_0 = +54 dB`.
- **Snipper clustering algorithm** (`signal_snip_core.cu`):
  1. `label_components`: 4-connected connected-component labeling; keep components with
     `pixel_count ≥ min_box_pixels` (256).
  2. `merge_boxes`: coalesce component bounding boxes within `merge_gap_rows` (16) / `merge_gap_cols`
     (80), to a fixed point.
  3. (added for this study) `filter_boxes_by_size`: drop a box unless its bounding box spans
     `≥ min_bandwidth_hz` **and** `≥ min_duration_s`.
  4. `map_box_to_physical`: each surviving box → **one rectangle** = a single frequency band
     `[freq(col0) .. freq(col1)]` × a single time span `[row0 .. row1]`.
  5. Frequency mode: mix that band to baseband by its (constant) center frequency, low-pass to the
     band's bandwidth, integer-decimate; store `center_freq_hz`, `sample_rate_hz`, sample counts.
     Time-slice mode: keep the box's time rows at full bandwidth/rate.
- **Where this was found.** During a min-size-filter experiment (`min_bandwidth_hz = 100 kHz`,
  `min_duration_s = 5 ms`) run over `coherent_power` and `finetuned_dino_m2` into
  `/tmp/usrp_spectrograms/snip_eval_minsize`; results in `real_snip_metrics_minsize.csv`.

## Observation

Comparing the 100 kHz / 5 ms filtered run to the default (256-px-only) run, at low SNR:

| detector | mode | atten_60 (−6 dB) | atten_65 (−11 dB) | atten_70 (−16 dB) |
|---|---|---|---|---|
| coherent_power | frequency (snip) | 0.75 TB/hr | 0.97 TB/hr | 0.92 TB/hr |
| coherent_power | time_only | 5.12 TB/hr | 5.31 TB/hr | 4.82 TB/hr |
| finetuned_dino_m2 | frequency (snip) | 0.41 TB/hr | 0.0001 TB/hr | **0.00** TB/hr |
| finetuned_dino_m2 | time_only | 0.83 TB/hr | 0.02 TB/hr | **0.00** TB/hr |

`coherent_power` keeps storing a large footprint at the weakest SNR, while `finetuned_dino_m2`
collapses to ~0. The initial expectation was that this reflected a detection difference; it does not.

## Investigation

### Ruled out (each tested against the data, not assumed)

- **"Coherent over-detects noise at low SNR."** False. Mean mask coverage (fraction of lit pixels
  over ~40 frames): coherent `atten_70 = 0.57 %`, DINO FT `atten_70 = 0.50 %` — essentially the
  same. (For contrast, at high SNR both are dense: coherent `atten_0 = 22.4 %`, DINO `atten_0 =
  23.1 %`.) The two detectors flag nearly identical *amounts* at low SNR.
- **"It's the `merge_gap` step fusing separate boxes."** False. Applying the size gate to the raw
  connected components **without** any merge gives the identical count as **with** merge
  (250 = 250 at atten_70). The fusion is plain 4-connectivity, not the gap-merge.
- **"The persistent streak is a phantom/false detection at an empty frequency."** ~~False.~~
  **CORRECTED 2026-07-23: this bullet's original conclusion was wrong.** The streak does sit
  ~18 dB above the noise floor and is lit in all 512 rows — but above-floor power alone does not
  make it a transmitted signal, and the "1074 / 3594 annotations covering 48 MHz" are all
  **≥5 MHz-wide bursts that merely span that frequency** (the closest *narrowband* ground-truth
  signal is at 53.7 MHz — nothing narrowband exists at 48 MHz). The attenuation sweep proves the
  streak is a **receiver clock spur**: real energy in the digitized samples, so the energy
  detectors are "honestly" flagging it, but it is not a signal of interest and by ground truth it
  is a false detection. See "Streak provenance — RESOLVED" below.

### Mechanism (verified by replicating the snipper's clustering in Python on the real masks)

At low SNR each flagged frame contains:
1. A **real persistent narrowband component** at a fixed ~48 MHz — a vertical line lit across the
   entire 21 ms frame (≈72 kHz wide, i.e. ~3 columns).
2. **Real transient wideband bursts** (ZC / metadata, ~30 MHz wide, ~0.3 ms / ~7–10 rows) that sweep
   across the band and therefore **cross** the 48 MHz frequency.

Because a burst passes through the narrowband line's frequency, the line and the burst(s) **touch**
and become a **single 4-connected component**. Its bounding box is therefore:
- **wide** — from the bursts (~30 MHz), and
- **tall** — from the persistent line (the full 21.3 ms).

`map_box_to_physical` turns that one component into a single rectangle of ~30 MHz × 21.3 ms, and the
snipper stores the **entire rectangle**. Individually, neither piece would pass a
`100 kHz AND 5 ms` gate — the bursts are wide but far too short (~0.3 ms < 5 ms), and the line is
long but far too narrow (72 kHz < 100 kHz). Only their **fused bounding box** satisfies both
requirements, and only because the height comes from one signal and the width from another.

## Quantitative evidence (coherent_power @ −16 dB, 328 frames)

- **250** boxes pass the 100 kHz / 5 ms gate across the capture.
- **250 / 250** boxes are **< 15 % filled** (median **1.43 %**, max 8.3 %).
- **250 / 250** boxes span the **full 21.3 ms frame height**.
- **248 / 250** boxes have their persistent-streak column at the **same fixed frequency (48 MHz)**;
  the other 2 at −116 MHz.
- Total **stored rectangle area = 162.9 M pixels** vs **actually-lit = 3.2 M pixels** →
  **51× over-count**.
- **248 / 328 frames (76 %)** are flagged — the pattern is systematic across the capture, not a few
  outliers. `atten_65` and `atten_60` show the same behavior (all boxes < 15 % filled, full-height,
  streak at 48 MHz).

For reference, a single representative passing component (frame 100): bounding box `1283 cols × 512
rows = 30.8 MHz × 21.3 ms`, `16,847` lit pixels in a `656,896`-pixel rectangle = **2.56 % filled**,
composed of 14 wide horizontal bar-rows (bursts) + 1 full-height column (the persistence streak),
connected into one component.

### Why DINO FT reads ~0 at the same SNR

`finetuned_dino_m2` produces the same transient wideband bursts but **no persistent narrowband
streak**. Its components therefore stay wide-but-short (~0.3 ms) and honestly fail the 5 ms
requirement, so nothing is stored. The large coherent-vs-DINO footprint gap at low SNR is dominated
by this bounding-box fusion, not by a real difference in captured signal.

## Streak provenance — RESOLVED: receiver clock spur (2026-07-23)

Four independent lines of evidence (`streak_forensics.py`, `streak_mask_presence.py`; figures in
`figs_minsize/streak_*.png`, data in `streak_forensics.csv`, `streak_mask_presence.csv`):

1. **It does not pass through the attenuator.** Every transmitted signal traverses the programmable
   attenuator, so its received power must fall 1 dB per dB of attenuation. The real narrowband BPSK
   at +60 MHz does exactly that (falls ~46 dB across the sweep, then disappears into the floor).
   The 48 MHz streak falls only ~7 dB across 65 dB of added attenuation (slope ≈ −0.1 dB/dB for
   atten ≥ 20) and is still **15–20 dB above the noise floor at atten 85 (−31 dB SNR)**, where every
   real signal is long gone. (Its apparent drop over atten 0→20 is contamination: real wideband
   signals overlapping 48 MHz dominate the bin at high SNR — the peak frequency wanders 48.07–48.19
   MHz there, then locks to exactly 48.000 once they fade.)
   → `figs_minsize/streak_power_vs_attenuation.png`, `streak_zoom_across_attens.png`.
2. **It is a pure CW tone at a binary-round clock frequency.** A 2^22-point FFT resolves it to a
   **~117 Hz-wide carrier at exactly 48.000000 MHz** baseband = **2048 MHz = 2^11 MHz absolute**
   (capture cf 2 GHz), amplitude constant across attens 40/70/85. No modulation — this is not
   narrowband FM (even NBFM is ≳10 kHz wide), not a comms signal.
3. **It has an exact spur family.** At atten 85 the persistent tones sit at
   **2048 + k×20.48 MHz**: 1884.16 (k=−8, **50 dB above floor** — this is the "−116 MHz" of the 2
   outlier boxes noted above), 1904.64 (−7), 1925.12 (−6), 1966.08 (−4), 1986.56 (−3),
   2088.96 (+2), 2109.44 (+3). All binary MHz (20.48 = 2^11×10 kHz; 245.76 MS/s = 12×20.48 MHz) —
   textbook RFSoC/PLL clock-spur structure. → `figs_minsize/streak_fullband_psd.png`.
4. **Only energy-threshold detectors see it; ground truth doesn't have it.** Column-occupancy over
   all staged masks (`streak_mask_presence.csv`): `coherent_power`, `cuda_dino`, `3dB_power`, and
   `blob_detection` light the 48 MHz column at ~100% occupancy in ~100% of frames at *every*
   attenuation 40–70. `finetuned_dino`, `finetuned_dino_m2`, `yolo26s/m`, and `ground_truth` do
   not (≤2% occupancy at low SNR). The learned detectors were trained against ground truth that
   never labels the spur, so they learned to ignore it.

**Interpretation.** The spur is real energy in the digitized samples, so an energy detector is
"honest" to flag it — but it is not a signal of interest, ground truth never annotates it, and it
counts as a false detection in any GT-scored comparison. The earlier conclusion that it was "a real
persistent narrowband component" is retracted: it rested on above-floor power (which a spur also
has) plus wideband GT annotations *spanning* 48 MHz (none of which is a narrowband signal *at*
48 MHz).

This also settles attribution for the footprint gap: `coherent_power`'s low-SNR footprint is
inflated by a **hardware artifact fused to real bursts by the snipper's bounding-box clustering** —
neither a pure detector failure nor a pure snipper failure, but the interaction of (a) an energy
detector faithfully flagging a receiver spur and (b) box-of-connected-component representation
storing the fused rectangle whole.

## Nature of the problem

- The over-count is a property of **representing a detection as the rectangular bounding box of a
  connected component**. The mask content is correct; the *rectangle drawn around it* is not
  representative of what was detected when a component contains structurally different signals (a
  tall-thin persistent line and a wide-thin transient burst) that are incidentally connected.
- It is **attribution-independent**: it does not depend on whether the 48 MHz line is an intended
  transmitted signal or a hardware spur. Either way, a persistent narrowband component crossed by
  transient wideband bursts yields a full-band, full-duration, mostly-empty stored rectangle.
- It systematically **inflates `coherent_power`'s low-SNR footprint** (both frequency and time-slice
  modes), which in turn distorts any storage/reduction comparison against detectors that do not emit
  persistent narrowband detections.

## Complication: bounding-box sparsity is not, by itself, evidence of an artifact

A low bounding-box fill ratio does **not** uniquely indicate this problem, because several legitimate
signal types have naturally sparse (non-rectangular) time-frequency footprints:

- **Chirps / linear-FM (including Zadoff-Chu-like sequences):** a diagonal frequency sweep. The
  bounding box is `sweep-bandwidth × sweep-duration`, but the instantaneous signal is a thin diagonal
  → the box is genuinely mostly empty, yet the whole box is required to represent the signal.
- **Wideband / analog FM:** instantaneous frequency wanders over a wide excursion → a thin wandering
  trace inside a wide box.
- **Frequency-hopping / stepped signals:** energy at different frequencies at different times → a
  sparse box.

There is also a signal-processing constraint that bounds what "storing less" can mean. A sub-band is
stored by mixing it to baseband with a **single constant center frequency** and recording that center
plus the sample rate as the reconstruction reference; this is a linear, time-invariant, invertible
operation. Following a signal whose frequency changes with time (e.g. a chirp) would require a
**time-varying mix** whose full frequency-vs-time schedule must itself be estimated and stored as the
reference — signal-specific processing that the generic snipper does not perform. Consequently, for a
genuinely frequency-agile single signal, the full-bandwidth rectangular box is the minimal invertible
representation. This makes distinguishing "artifact bounding box" (multiple distinct signals fused)
from "legitimately sparse bounding box" (one agile signal) non-trivial.

## Pipeline verification (2026-07-23): the accounting is correct

The whole chain — mask staging → `mask_replay_detector` → `signal_snipper` → `sigmf_file_sink` →
`verify_snip.py` — was audited for anything that could inflate the footprint beyond what the masks
plus bounding-box clustering imply. Nothing was found:

- `mask_replay_detector` (`operators/mask_replay_detector/mask_replay_detector.cu`) replays the
  `.npy` masks **verbatim** — no dilation, thresholding, or resizing; geometry is passed through by
  ratio, missing masks become all-zero (no detections), and frame numbers match the snipper's
  1-based IQ arrival counter.
- In `per_signal` mode each snippet produces exactly **one** `.sigmf-meta`; every annotation in it
  shares `core:sample_start=0, core:sample_count = stored (decimated) count`, and `verify_snip.py`
  takes `max(start+count)` (not a sum), so multi-annotation snippets are **not** double-counted.
  `--snippets-only` only suppresses debug artifacts; it does not change accounting.
- Analytic cross-check: the measured numbers are exactly what the fused boxes predict. 76% of
  frames flagged with full-height boxes → time_only ≈ 0.76 × 7.08 ≈ 5.4 TB/hr (measured 4.8–5.3);
  a ~30 MHz box at 25% oversample decimates by 6 → 0.76 × 7.08/6 ≈ 0.90 TB/hr (measured 0.92).
- Known intentional inflations (not bugs): frequency mode stores bandwidth × 1.25 (oversample);
  gap-merge (16 rows / 80 cols) covers inter-fragment gaps; the base config has both physical size
  gates disabled.

## Reproducing

All scripts live in `applications/usrp_wideband_signal_detection/infocom_evals/snip_eval/` and read
the staged masks (`snip_run/<detector>/<stem>/mask_arrays/*.packed.npz`) and raw captures
(`~/captures/attenuation_dB_*.sigmf-data`). Run with an environment that has numpy/scipy/matplotlib
(e.g. `~/miniforge3/envs/dinov3/bin/python`).

- `prove_coherent_artifact.py [atten]` — prints the quantitative table above (fill %, full-height
  fraction, fixed streak frequency, over-count factor, GT coverage, streak SNR) and writes a montage
  of flagged frames → `figs_minsize/prove_coherent_artifact.png`.
- `render_spectrogram_overlay.py [atten] [frame]` — real spectrogram (computed from raw IQ) with the
  mask overlaid in high contrast and the snipper bounding boxes, plus a zoom on the persistence
  streak → `figs_minsize/debug_spectrogram_overlay.png`.
- `visualize_bbox.py [atten] [frame]` — per-frame overlay of the snipper's bounding boxes on the mask
  (green = passes the size gate / snipped, red = dropped) → `figs_minsize/debug_bbox_overlay.png`.

Key figures produced:
- `figs_minsize/prove_coherent_artifact.png` — 6 flagged frames, all the same fixed-frequency,
  full-height, ~1.4 %-filled box.
- `figs_minsize/debug_spectrogram_overlay.png` — spectrogram + mask + boxes + streak zoom.
- `figs_minsize/debug_bbox_overlay.png` — coherent (one box passes via the fused streak) vs DINO FT
  (wideband bursts only, all dropped).

## Fix options quantified (2026-07-23, diagnose-only — nothing implemented)

`quantify_fixes.py` replicates the snipper's exact clustering + decimation math offline on the
staged masks and scores candidate fixes under all three gate configs (full grid across detectors ×
attens 40–70 in `fix_quantification.csv`). Validation: the "current" strategy reproduces the real
measured pipeline (e.g. 0.748 vs measured 0.75 TB/hr, coherent @ atten 60, 100 kHz/5 ms gate).

Strategies:
- **split** — *persistent-column split*: pre-labeling, columns lit in ≥60% of the frame's rows
  (the spur line) are clustered **separately** from the rest, so 4-connectivity can no longer
  bridge the line to transient bursts. Each part then merges/gates/stores on its own merits.
  ~30 host-side lines in `signal_snip_core.cu` (the mask is already on the host in
  `process_mask`), plus one config param.
- **suppress** — split, then drop the persistent-column boxes entirely (treat as environment/spur).
- **fill10** — keep current clustering, drop final boxes <10% lit (fill-ratio gate).
- **content** — accounting-only lower bound (lit pixels × 8 B; 1 TF pixel = 1 complex sample on
  this critically-sampled grid). Not a snipper change; shows the floor.

Headline (coherent_power @ atten 70 = −16 dB SNR; save-all = 7.08 TB/hr):

| gate | mode | current | split | suppress | fill10 | finetuned_dino_m2 (current) |
|---|---|---|---|---|---|---|
| 100 kHz/5 ms | frequency | 0.92 | **0.00** | 0.00 | 0.00 | 0.00 |
| 100 kHz/5 ms | time_only | 5.35 | **0.00** | 0.00 | 0.00 | 0.00 |
| 75 kHz/1 ms | frequency | 0.92 | **0.00** | 0.00 | 0.00 | 0.01 |
| 75 kHz/1 ms | time_only | 5.72 | **0.00** | 0.00 | 1.23 | 0.01 |
| default (256 px) | frequency | 0.96 | **0.066** | 0.064 | 0.041 | 0.09 |
| default (256 px) | time_only | 7.08 | 7.08 | **0.186** | 7.04 | 0.14 |

(TB/hr. Same pattern at attens 50–65; atten 40 in the CSV.)

Key observations:
1. **Under either physical size gate, `split` alone completely removes the artifact**: the spur box
   (~72 kHz wide) honestly fails the bandwidth gate, the bursts (~0.3 ms) honestly fail the
   duration gate — coherent_power lands on exactly the same 0 as the fine-tuned DINO, because at
   these SNRs the two detectors flag nearly the same real content.
2. **With no size gate, `split` fixes frequency mode (0.96 → 0.066, ~15×) but NOT time_only**: a
   full-height spur box still forces whole-frame time slices (7.08 = every frame stored). Only
   `suppress` fixes time mode (7.08 → 0.19, ~38×). If time-slice mode matters with the default
   gate, spur handling is mandatory, not optional.
3. **`fill10` is dominated**: it kills the artifact but also drops legitimately sparse content —
   lit-pixel retention falls (e.g. 74% → 45% at atten 70 default gate; and see the chirp/hopper
   caveat in the Complication section). `split` achieves the same footprint fix with no such loss.
   Not recommended.
4. **DINO is untouched by every strategy** (identical numbers across the row) — the fixes are
   surgical: they only change behavior where a persistent line exists.
5. After the fix, the "apples-to-apples" low-SNR comparison becomes coherent 0.066 vs DINO 0.094
   TB/hr (default gate, frequency): **the detectors are actually comparable** — the entire
   0.92-vs-0.00 gap in the current tables is the spur-fusion artifact.

### Recommendation

1. **Implement the persistent-column split in the snipper** (`signal_snip_core.cu` /
   `signal_snipper.cu`): new param e.g. `persistent_col_row_frac` (0 = off, current behavior;
   0.6 = split). This is representation-honesty, not signal suppression — everything detected is
   still stored, as separate honest boxes.
2. **Add a spur policy for time-slice mode / gate-free configs — but restrict it to known spur
   bins.** The quantification rules out blanket suppression of persistent narrowband boxes: the
   waveform set genuinely contains persistent narrowband signals (the 20 ms / 324 kHz BPSKs), and
   on `ground_truth` masks the blanket `suppress` strategy cuts lit-pixel retention from ~100% to
   **37%** (real signal thrown away). Instead, statically notch the *calibrated* spur columns
   (2048 ± k×20.48 MHz, measurable from a no-TX capture) before clustering — a few columns out of
   10240 — which fixes time_only (7.08 → ~0.19 TB/hr on coherent) while leaving every real
   persistent signal alone.
3. **Keep the min-size gates** (either 100 kHz/5 ms or 75 kHz/1 ms per collection policy) — with
   the split in place they behave as intended for every detector.
4. Re-run `run_snip_all.sh` after a container rebuild to regenerate `real_snip_metrics*.csv` and
   confirm the offline prediction (needs docker permissions — next section).

## Enabling unattended container rebuilds (docker permissions)

The eval account (`bqn82@austin.utexas.edu`) has no passwordless sudo and is not in the `docker`
group, so an unattended session cannot rebuild the container or run the real snip pipeline. From
the sudo-capable account, either:

1. **Sudoers rule (recommended — the repo wrappers work unchanged):** the wrappers
   (`rebuild_demo_container_app.sh`, `run_snip_all.sh`, …) hardcode `sudo docker`, so group
   membership alone does not unblock them. A NOPASSWD rule for docker does:
   ```bash
   echo 'bqn82@austin.utexas.edu ALL=(ALL) NOPASSWD: /usr/bin/docker' | sudo tee /etc/sudoers.d/bqn82-docker
   sudo chmod 440 /etc/sudoers.d/bqn82-docker
   ```
2. **Docker group** (`sudo usermod -aG docker 'bqn82@austin.utexas.edu'`) also grants API access
   (new logins only), but the wrappers would still prompt for the `sudo` password.

Either grant is root-equivalent (standard docker caveat). Verify with `sudo -n docker ps`.

## Status

**2026-07-23, second investigation complete (diagnose-only, per user direction):**
- Pipeline accounting audited end-to-end — correct; the footprint numbers are real, caused by the
  bounding-box fusion.
- Streak provenance **resolved**: receiver clock spur at 2048 MHz (attenuation-sweep proof;
  `streak_forensics.py`, `streak_mask_presence.py`, figures in `figs_minsize/streak_*.png`). The
  earlier "real narrowband component" conclusion is retracted above.
- Fix options quantified (`quantify_fixes.py` → `fix_quantification.csv`): persistent-column
  **split** removes the artifact entirely under either min-size gate (coherent → 0.00 TB/hr,
  matching DINO) without dropping anything real; spur-bin notching additionally fixes gate-free
  time-slice mode; blanket persistent-line suppression and fill-ratio gates are ruled out by
  retention loss. **No C++/CUDA changes made yet** — see Recommendation.
- Next steps: grant docker perms (section above), implement the split + calibrated spur notch in
  the snipper, rebuild the container, re-run `run_snip_all.sh`, regenerate
  `real_snip_metrics*.csv`, and refresh the paper figures.

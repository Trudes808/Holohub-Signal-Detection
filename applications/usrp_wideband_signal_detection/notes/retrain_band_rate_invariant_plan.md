# Band- & rate-invariant fine-tuned DINO — retrain plan

**Goal.** One fine-tune that deploys at *any* center frequency and *any* valid integer decimation of a
491.52 MHz or 500 MHz master clock (sample rates ~20–500 MS/s), robust across bands: 1150 MHz (mostly
quiet), 2400 MHz (ISM), 500 MHz (loud HDTV/OFDM). Success = no false positives in quiet bands, no
regression on ISM, clean detection of wideband signals — at every rate — with **no per-deploy
calibration**.

---

## 1. Root cause (why the current model is brittle)

The current checkpoint (`M2_ft`) was fine-tuned on a **single, narrow distribution**:
- one rate (**245.76 MS/s** → RBW = 245.76e6/1024 = **240 kHz/bin**),
- one receiver envelope (the decimation-filter shape at that rate),
- one gain regime / one band (synthetic ZC/LTE transmitted OTA),
- `frames_to_db` with **no normalization** (raw dB + a global `vmin/vmax` clip),
- augmentation limited to freq-flip / time-roll / light additive noise (`dataset.py::_augment`).

So the model memorized the *absolute* look of that setup as "background." Every deployment axis that
differs is out-of-distribution:
1. **Level** (gain/antenna/band noise) — scalar; the fixed `vmin/vmax` mis-levels it. (We confirmed a
   scalar trim only helps *unevenly* → level is not the whole story.)
2. **Envelope shape** (decimation filter vs rate/hardware) — per-frequency; can't be fixed at inference
   without inventing the training envelope.
3. **Resolution / RBW** (bins-per-Hz = rate/nfft) — 240 kHz/bin in training vs 20–488 kHz/bin across
   20–500 MS/s. Signals occupy a rate-dependent pixel width; noise correlation differs.
4. **Signal type** — positives are synthetic ZC/LTE; real HDTV/OFDM at 500 MHz is a different shape.

Inference-time preprocessing cannot close 2–4 for *this* model. The durable fix is to **train the
invariance in** via domain randomization, reusing the existing labeled captures + a small amount of new
*unlabeled* data.

---

## 2. Invariance strategy (augment, don't normalize at inference)

Keep the inference contract simple and principled — the operator stays: **raw dB → FFT processing-gain
correction (`10·log10(fft/nfft)`) → global `vmin/vmax` clip**, with **flatten OFF** and **no level
trim** (the model becomes level-invariant, so neither is needed). We make the model invariant to each
axis by bracketing it in training:

| Axis | Training augmentation | Where |
| --- | --- | --- |
| Level | random per-frame dB offset (in dB domain, before the clip) | `dataset.py` on-the-fly |
| Envelope | multiply by a per-frequency curve: measured decimation envelopes ± random smooth tilt/ripple | `dataset.py` on-the-fly |
| RBW / rate | **capture-chain emulation** (random center + real LPF + decimate; upsample+paste for R>source) — see §2a | `build_dataset.py` |
| Signal type | expand the synthetic waveform zoo (bandwidths, OFDM/CW/chirp/burst) | `build_lte_batch.py` |
| Background | cut-paste synthetic signals onto **real multi-band** unlabeled backgrounds; use quiet regions as **negatives** | `build_dataset.py` + `dataset.py` |
| (existing) | freq-flip, time-roll, light additive noise | `dataset.py` (keep) |

**Key implication for materialization:** level and envelope augmentation must happen in the **dB
domain before the clip**. So `build_dataset.py` should store **float dB frames (pre-clip)** instead of
(or alongside) the `[0,1]` uint8 stacks; `dataset.py` applies level+envelope+paste, then clips with
`vmin/vmax` on the fly. Rate cannot be faked from a rendered image → it must be rendered from IQ, so
the multi-rate loop lives in `build_dataset.py`.

### 2a. Rate augmentation = faithful capture-chain emulation (not naive resample)

The labeled source is **245.76 MS/s IQ, 245.76 MHz wide**. To make a training frame look exactly as if
it had been captured at a *lower* rate `R` and a random center inside the wideband, emulate the real
receive chain **in IQ, before spectrogram creation** (this also augments frequency position for free):

1. **Random center offset** `f_c` drawn within `±(245.76e6 − R)/2` of baseband → selects which sub-band
   of the wideband becomes the new capture, and randomizes where signals land.
2. **Frequency shift** the IQ by `−f_c` (complex mix) so `f_c` moves to DC.
3. **Anti-alias low-pass** with cutoff ≈ `R/2`, shaped like the **real decimation filter measured for
   rate `R`** (from the sweep envelope, §3b.1) rather than an ideal brick wall — so the emulated band
   carries the true rolloff/ripple the hardware would have produced.
4. **Decimate** by `D = round(245.76e6 / R)` → IQ as if captured at `R`, centered at
   `source_center + f_c`.
5. **Spectrogram** (nfft-FFT) → RBW = `R/nfft`, with the correct per-rate envelope baked in.
6. **Label remap (careful):** each annotation's `freq_lower/upper_edge` shifts by `−f_c` and rescales to
   the new grid; signals fully outside `[−R/2, +R/2]` are **dropped**, partially-overlapping ones are
   **clipped** to the band edge; `sample_start/count` (time) is unchanged by decimation of a
   whole-frame block but the row index rescales with `D`. Rebuild the GT mask on the new grid.

**Target rates > 245.76 (250, 491.52, 500):** cannot be synthesized by decimation (no real bandwidth
beyond the source). For these, **upsample the labeled signal** (lossless — it's bandlimited to ≤245.76)
to `R` so it occupies the center portion of the `R`-wide band, then **composite it onto a real sweep
background captured at `R`** (which supplies the true wideband noise/envelope/spurs). Labels come from
the upsampled signal's positions. This is the §3b.3 cut-paste path, rate-matched.

Both paths run in `build_dataset.py`; multiple random `f_c` per source frame multiply the data.

---

## 3. Data plan

### 3a. Existing labeled (reuse as-is)
`/home/bqn82/captures/attenuation_dB_{0,10,15,20,25,30,30_v2,35,40,...}` — synthetic ZC/LTE at 245.76
MS/s with precise SigMF annotations (`freq_lower/upper_edge` Hz, `sample_start/count`, `wfgt:` kind).
Gives the **SNR axis** and clean labels. Annotations in Hz → remap to bins at any rate.

### 3b. New UNLABELED data — an automated deployment-range sweep (decided)
A new automated script (`data_collection/sweep_capture.py`) sweeps the USRP over the **expected
deployment envelope** in short bursts and saves a per-burst PSD (+ a subset of raw IQ). Grid: **center
50 MHz → 5900 MHz** (~100 steps) × the **rate set** (§4) × **~6 gain** levels. Occupancy is *unknown and
uncontrolled* — so this data is used for **characterization, NOT labels**.

**Two collection passes, and two FPGA phases:**
- **Terminated pass (no antenna):** clean envelope + noise-floor *lower* bound. The envelope is
  center-independent with no antenna, so sweep **rate × gain at a few centers only** (`--tag terminated
  --center-count 3`). Fast, exact, no signal contamination.
- **Antenna pass:** the full center × rate × gain sweep → real **background textures**, spur/occupancy
  diversity, and the level *upper* range (`--tag antenna`). Envelope here is a median-across-centers
  cross-check.
- **FPGA image / master clock:** usable rates = integer decimations of the master clock, and legal
  master clocks depend on the loaded X410 FPGA image — **200 MHz image → 245.76/250 clocks** (≤~245.76
  MS/s, stock), **400 MHz image → 491.52/500 clocks** (up to ~500). Sweep with the stock image first
  (default `--master-clocks-hz 245.76e6 250e6`), **reimage**, then add the high rates
  (`--master-clocks-hz 491.52e6 500e6 --rate-min-hz 260e6`). An unsupported clock is skipped with a
  message, not fatal. `sweep_stats.py` merges the run dirs. (The near-identical .76/round clock families
  are deduped so the sweep stays a clean ~5–7 rate set; training emulation covers the continuum.)

Uses of the collected data:

1. **Per-rate receiver envelope templates (robust, occupancy-agnostic).** The decimation/analog envelope
   at a given rate is a *fixed hardware shape* in baseband; across many sweep bursts at the same rate but
   *different centers*, real signals land at different baseband positions while the envelope stays put.
   So per baseband bin, take the **median (or low percentile) across many bursts** → the sparse signals
   are rejected and the receiver envelope is recovered. Do this per rate (and optionally per gain). These
   become the realistic envelope-augmentation templates.
2. **Level / floor distribution.** Measure the noise-floor dB per (band, gain) → set the **level
   augmentation range** (bracket the observed min/max ±margin). Gain mostly shifts level (scalar), not
   envelope shape.
3. **Background textures for cut-paste.** Use sweep frames as realistic backgrounds to composite *under*
   the synthetic labeled signals. Any real signal in a background becomes bounded label noise (mitigate
   by power-gating paste locations below the measured floor, or masking occupied bins with a quick
   coherent-power pass).
4. **Range characterization.** Confirms the rate/level/envelope ranges the model must span so the
   augmentation brackets (never extrapolates beyond) deployment.

**Clean negatives do NOT come from the sweep** (occupancy unknown → would teach false negatives). They
come from the **labeled captures' known-empty frames** (the synthetic annotations tell us exactly where
signals are, so signal-free frames are guaranteed-empty), then **augmented** with the sweep-derived
envelopes + levels + background texture → diverse, realistic, *correctly-labeled-empty* negatives. This
is what directly teaches "quiet band → no detection."

### 3c. Signal-type variety (decided: use the full existing labeled set)
The labeled captures already contain a **huge variety** (100s of waveform configurations across the
attenuation sweep). No new synthesis/transmission needed — instead ensure `build_dataset.py` ingests the
**full variety across all captures** (not a subset), so every waveform config appears. Verify the set
already includes wideband/OFDM-like shapes for the 500 MHz HDTV zone; if a shape is truly absent, that
is the only case that might warrant one targeted addition later.

---

## 4. Rate set (valid decimations, 20–500 MS/s)

- **491.52 MHz master:** 491.52/N → 491.52, 245.76, 163.84, 122.88, 98.304, 81.92, 61.44, 49.152,
  40.96, 30.72, 24.576, 20.48 MS/s.
- **500 MHz master:** 500/N → 500, 250, 166.67, 125, 100, 83.33, 62.5, 50, 41.67, 33.33, 25, 20.83 MS/s.
- RBW range at nfft=1024: **~20 kHz/bin (20 MS/s) → ~488 kHz/bin (500 MS/s)**, ~25× span.

**Training rate set (proposed):** a log-spaced representative subset that brackets the RBW range from
both masters, e.g. `{20.48, 30.72, 49.152, 81.92, 122.88, 245.76, 491.52}` ∪ `{25, 50, 100, 250, 500}`
(~10–12 rates). The model interpolates between; endpoints (20 & 500) must be included so deployment is
never extrapolating. Resample the 245.76 captures to each with a polyphase resampler; remap each
annotation's `freq_edges`/`sample_start/count` to the new grid (freq bins scale by rate ratio; time
rows by the inverse).

---

## 5. Training steps

1. **Expand waveform zoo** (`build_lte_batch.py`): add OFDM/wideband, CW, chirp, burst types across
   bandwidth fractions. (Re-transmit/re-capture OR synthesize into IQ — see §8 decision.)
2. **Multi-rate materialization** (`build_dataset.py`):
   - Loop the rate set; resample IQ per rate; render **float dB** frames (pre-clip) + masks; remap
     annotations to the per-rate grid.
   - Add multi-band backgrounds as negatives + cut-paste sources.
   - Keep the `db_vmin/db_vmax` calibration (global) for the clip applied downstream; also record the
     **measured floor level + envelope per rate** into a sidecar for augmentation ranges.
3. **Augmentation** (`dataset.py::_augment`): add, in dB domain before the clip —
   - level: `db += U(level_lo, level_hi)` (range from §3b stats, bracketed);
   - envelope: `db += env_curve(freq)` where `env_curve` ∈ {measured templates} ⊕ random smooth
     tilt/ripple (bounded slope + edge rolloff);
   - background paste (optional): composite a synthetic signal tile onto a real background tile;
   - then clip with `vmin/vmax`; keep existing flip/roll/noise.
   Gate each with a config knob + range in `train.yaml`.
4. **Train** (`train.py`, `configs/train.yaml`): unchanged recipe (bf16, dice+bce, pos_weight,
   unfreeze_last_n), `augment: true`. Consider a modest epoch bump for the larger/looser distribution.
   Validate on the multi-rate/multi-band val split.
5. **Export** (`export_dinov3_finetuned_torchscript.py --autocast bf16`): emit `.ts` + `.meta.json`.
   **Extend `meta.json`** to record the trained **rate range / RBW range** (so the operator can warn if
   deployed outside it) and confirm `db_vmin/db_vmax/threshold/tile_rows/nfft`.
6. **Deploy:** point the operator config at the new `.ts`/`.meta.json`; `flatten_noise_floor: false`,
   `match_training_power_level: false`, `power_level_trim_db: 0` (model is now invariant). Keep the
   FFT-gain correction (it's exact physics, needed for the downsample path).

---

## 6. Saved tests (did it help?)

Record everything under `infocom_evals/signal_detection_experiments/retrain_band_rate/` with a direct
comparison to the current `M2_ft` model. Define these as reusable saved tests:

1. **Quiet-zone false-positive rate — the headline.** Offline eval on a confirmed-quiet **1150 MHz**
   capture (no annotations → GT empty). Metric: detected-pixels / frame (should be ≈0). Report current
   vs retrained. This is the metric the whole retrain targets.
2. **ISM regression — 2400 MHz.** Re-run the existing SNR harness (detection-rate-vs-SNR + frame
   precision/recall/F1/IoU) on the labeled attenuation sweep. Retrained must **not regress** vs current.
3. **Wideband/HDTV — 500 MHz.** Offline eval on a 500 MHz HDTV capture; qualitative + coverage/IoU of
   the wideband occupancy (should be detected as solid regions, not fragmented or missed). If no GT,
   score coverage + false-area outside the known-occupied span.
4. **Rate-invariance sweep.** Run tests 1–3 at several rates (e.g. 25, 100, 245, 500 MS/s) by
   resampling each capture; confirm metrics are stable across rate (the core deliverable).
5. **Held-out synthetic SNR** (existing eval harness) as a clean regression gate.

Wire these into the existing `eval_detector_masks.py` / `build_snr_results.py` / `plot_snr_results.py`
harness; save the serialized `SnrResults` + plots so re-runs are one command.

---

## 7. Inference-code changes (small)
- Operator: no new logic needed. Keep gain correction; leave `flatten_noise_floor`/`match_training_
  power_level`/`power_level_trim_db` as opt-in escape hatches, all default off.
- Optionally: read the trained rate range from `meta.json` and log a warning if the live rate is
  outside it (deployment-outside-training-distribution guard).

---

## 8. Open decisions
1. **Waveform variety — DECIDED:** use the full existing labeled set (100s of configs); no new
   synthesis/transmission.
2. **Data collection — DECIDED:** automated unlabeled sweep, center 50 MHz–5900 MHz × rates × gains,
   short bursts, used for envelope/level/background characterization (not labels). Negatives come from
   labeled known-empty frames.
3. **Native vs downsample rendering** — render training frames as native nfft-FFT at each rate
   (clean) only, or ALSO render via the deployment downsample path (wide FFT + resize) so the model
   also sees that artifact? Recommend native-multi-rate primary + a fraction rendered via the
   downsample path for train==inference safety.
4. **Rate discretization** — train only on the exact valid decimations, or add off-grid rates so the
   model interpolates? Recommend a few off-grid rates for margin.
5. **Storage** — multi-rate float-dB materialization multiplies dataset size (~#rates × current, in
   float). Cap `max_frames_per_capture` or store fp16 dB. Confirm disk budget on the training host.

---

## 9. Concrete change checklist
- **DONE** `data_collection/sweep_capture.py` — X410 **dual-channel** sweep (ch0 antenna + ch1
  terminated simultaneously) over center(50–5900 MHz)×rate×gain; `--master-clocks-hz` (default the stock
  200 MHz-image clocks 245.76/250; an unsupported clock is skipped, not fatal) with rates auto-derived
  from `--decims` + near-duplicate dedup; `--preflight` sanity check (radio reachable, per-channel PSD
  sane, **disk estimate vs free space → aborts if it won't fit**, + saves `preflight_envelope.png` /
  `preflight_psd.png` sanity plots — one burst/rate); per-cell failure logging
  (`failures.jsonl`, prints FAILED) with `--resume` + `--retry-failed`. Appends psd.f32 + manifest.jsonl.
- **DONE** `data_collection/sweep_stats.py` — merges sweep dirs (phases) → `envelopes.npz` (per-rate,
  terminated), `floor_stats.json` (level-offset range), `backgrounds.json` (antenna IQ index). Verified
  on a synthetic run.
- **DONE** fine-tuning README "Reproduce a band/rate-invariant fine-tune (any radio)" — Step 0 preflight
  + Steps 1–5, radio-specific inputs called out (rate/master-clock set, FPGA images, gain range, envelope).
- **NEW** sweep stats/envelope extractor — per-rate median-across-bursts envelope templates + per-(band,
  gain) floor levels + background-texture bank → a sidecar the dataset builder consumes.
- **DONE** `dino_fine_tuning/src/rate_augment.py` — capture-chain emulation (§2a: freq-shift + LPF +
  decimate) + annotation remap (freq shift/clip, time rescale) + measured-envelope reshape; self-test
  verifies a tone + its label land in the right bins after emulation.
- **DONE** `dino_fine_tuning/src/build_dataset.py` — `domain_randomize` mode: emulate each planned frame
  at every `dr_rates_hz` (<=source) × `dr_centers_per_frame` random centers → **float16 dB** stacks
  (pre-clip), skip-aware exact counts, ingests the sweep `envelopes.npz`. Legacy uint8 build untouched.
  (Upsample+paste for R>source = phase-2 TODO, needs the wideband sweep IQ.)
- **DONE** `dino_fine_tuning/src/dataset.py` — dual-mode: float16_db → dB-domain level-offset (gain
  invariance, range from `floor_stats.json`) + envelope-tilt jitter + flip/roll, then clip to [0,1];
  uint8 path unchanged. `train.py` consumes it transparently. Config keys added to `configs/dataset.yaml`.
- **DONE** upsample+paste path (R > source): `rate_augment.emulate_frame_upsample_paste` (upsample the
  labeled signal onto a real wideband sweep background, IQ domain; label remap) + `build_dataset`
  `load_backgrounds` + a guarded paste block (dormant until the wideband sweep provides backgrounds).
  Synthetic test passes (tone lands in the right bin in the wide band).
- **DONE** Step 4 export: `export_dinov3_finetuned_torchscript.py` records `trained_rate_range_hz` in
  `meta.json` (from `--rate-range-hz` or the dataset's `dr_rates_hz`).
- **DONE** Step 5 harness: `infocom_evals/signal_detection_experiments/retrain_band_rate/eval_band_rate.py`
  (rate-sweep pixel-F1 on labeled bands + false-positive-rate on quiet bands, current-vs-retrained,
  emulates the capture at each rate) + a README mapping all 5 saved tests (2 & 5 reuse the existing
  SNR harness). Compiles; runs on the user's models/captures.
- **TODO (data-gated, not code):** run the sweep, build the DR dataset, train + export the model, then
  run the saved tests. The >source-rate paste path activates once the wideband-image sweep exists.
- `dino_fine_tuning/configs/dataset.yaml` — rate set, background dirs, aug ranges, storage caps.
- `dino_fine_tuning/configs/train.yaml` — augmentation range knobs; epochs.
- `export_dinov3_finetuned_torchscript.py` — record rate/RBW range in `meta.json`.
- `applications/.../infocom_evals/signal_detection_experiments/retrain_band_rate/` — the 5 saved tests
  + comparison plots.
- Operator: optional `meta.json` rate-range warning; otherwise unchanged.

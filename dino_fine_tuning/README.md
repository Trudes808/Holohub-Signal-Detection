# DINOv3 Signal-Detection Fine-Tuning

Fine-tuning DINOv3 (ViT-B/16) for **signal-vs-noise segmentation on RF spectrograms**,
and comparing it head-to-head against the deployed **coherent-power** and **zero-shot
DINOv3** detectors — with a focus on **low-SNR (high-attenuation)** performance.

---

## ⭐ Key paths (this machine)

### Fine-tuned model checkpoints
| model | what it is | absolute path |
|---|---|---|
| **M1_ft** | fine-tuned on **≤30 dB** attenuation (backbone last-4 blocks + head) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/checkpoints/M1_ft/best.pt` |
| **M2_ft** | fine-tuned on **all** attenuations (0–60 dB) — **best at low SNR** | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/checkpoints/M2_ft/best.pt` |
| M1_frozen | frozen-backbone decoder head, ≤30 dB (baseline) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/checkpoints/M1_frozen/best.pt` |
| M2_frozen | frozen-backbone decoder head, all dB (baseline) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/checkpoints/M2_frozen/best.pt` |

Backbone weights (DINOv3 ViT-B/16, pretrained):
`/home/bqn82/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`

> Each `best.pt` bundles the backbone + segmentation head + its val-tuned decision
> threshold (M1_ft = 0.45, M2_ft = 0.85). Load with `src/model.py::DinoSegmenter`
> (see `src/finetuned_infer.py` for the exact inference wrapper).

### Notebooks to look at (open with the **"Python (dinov3)"** kernel)
| notebook | absolute path |
|---|---|
| Rich comparison graphs (mirrors `batch_eval_review.ipynb`, 4 detectors) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/batch_eval_review_three_detectors.ipynb` |
| Side-by-side spectrogram panels + headline metrics | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/compare_three_detectors.ipynb` |
| **Low-SNR false-positive analysis** (M2 hallucination gallery) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/low_snr_false_positives.ipynb` |
| **Wide-bandwidth low-SNR eval** (OFDM / 5G / 802.11ax) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/wideband_lowsnr_eval.ipynb` |
| **LTE OOD eval** (unseen waveform; detection/recall/coverage/FP vs SNR) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/lte_ood_eval.ipynb` |
| **Live-data M1/M2 masks** (unlabeled; spectrogram · M1 · M2) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/live_data_masks.ipynb` |
| **Frozen vs. ft variants** (is unfreezing worth the compute?) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/finetuned_variants_eval.ipynb` |

### Reports & tables
| item | absolute path |
|---|---|
| Results writeup (figures + narrative) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/reports/report.md` |
| Pipeline / system documentation | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/reports/pipeline.md` |
| Frozen-vs-ft (compute trade-off) report | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/reports/finetuned_variants_report.md` |
| Canonical 4-detector metric tables | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/compare_tables_canonical/{region_metrics,frame_pixel_metrics}.csv` |
| Per-model eval CSVs | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/eval_out/<model>/` |
| Combined detector-mask run root (all 4 detectors) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/sweep_detectors/` |
| Materialized training dataset | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/data/dataset/` |

Source captures (SigMF, 245.76 MSps): `/home/bqn82/captures/` ·
Deployed-detector batch masks: `/tmp/usrp_spectrograms/batch_eval/sweep_20260630/`

**Extra datasets (M1/M2 run offline; deployed baselines pending a container run):**
- LTE OOD captures: `/home/bqn82/captures/lte/` → M1/M2 masks+GT in
  `notebooks/sweep_lte/`, tables in `notebooks/compare_tables_lte/`. To add the two
  deployed baselines, run the offline pipeline on the LTE captures (standard 512×10240
  framing), drop `coherent_power/` and `cuda_dino/` under `notebooks/sweep_lte/`, then
  regenerate the tables (command in the notebook's last cell) and re-run — they appear
  automatically. **OOD result:** both fine-tuned models detect the unseen LTE waveform
  perfectly to 30 dB; M2 again leads at low SNR (det 0.85/0.75/0.35 vs M1 0.69/0.14/0 at
  40/45/50 dB); both fail by 55–60 dB.
- Live unlabeled captures: `/home/bqn82/captures/live_data/sigmf_out/` → qualitative
  M1/M2 mask panels in `live_data_masks.ipynb` (no GT).

### Environment
Conda env **`dinov3`** (`/home/bqn82/miniforge3/envs/dinov3`), registered Jupyter
kernel **"Python (dinov3)"**. Rebuild with `scripts/setup_env.sh`.

---

## What was done

### Task
Binary **per-pixel signal/noise segmentation** on spectrograms. GT = filled SigMF
annotation boxes (all kinds — waveforms, ZC, METADATA — count as signal). Trained
purely binary; waveform class / bandwidth / duration are carried through to
**evaluation only**, for the failure-mode breakdowns.

### Data & geometry
14 SigMF captures `attenuation_dB_{0…60}` (+`30_v2`), 245.76 MSps, ~7 s each.
Spectrogram frame = **256 time × 1024 freq** (nfft 1024 → 240 kHz/bin, ~1.07 ms/frame)
== the DINO input grid. Global fixed dB→uint8 mapping preserves cross-SNR contrast.
Dataset: ~21k frames, temporal 70/15/15 split with a guard gap so **every dB level
appears in every split**. Details + all deviations from the original brief:
`reports/pipeline.md`.

### Models (2×2 grid + baselines)
Backbone = DINOv3 ViT-B/16. Two adaptation modes — `frozen` (decoder head only) and
`ft_lastN` (unfreeze last 4 transformer blocks + head) — trained on two data splits:
- **M1** = ≤30 dB only, **M2** = all dB.  → `M1_frozen, M1_ft, M2_frozen, M2_ft`.

Compared against the **deployed** detectors from the `sweep_20260630` batch run:
- **coherent_power** (classical power detector) and **cuda_dino** (zero-shot DINOv3).

### Evaluation (consistency)
All detectors are scored by the **same canonical tool**,
`infocom_evals/.../eval_detector_masks.py`, over the batch run — identical metrics,
coverage/box-IoU, and source-SigMF bucketing for every detector. The fine-tuned
models' masks are materialized into a batch-format run dir
(`src/gen_finetuned_run.py` → `notebooks/sweep_detectors/`) and scored by that same
code path. **Baselines are the deployed detectors' own masks, unmodified.**
Region-detection threshold matches the original notebook (coverage ≥ 0.1).

### Headline results
**Region detection rate vs attenuation (coverage ≥ 0.1):**

| detector | 40 dB | 45 dB | 50 dB | 55 dB | 60 dB |
|---|---|---|---|---|---|
| Coherent Power | 0.69 | 0.63 | 0.57 | 0.52 | 0.47 |
| Zero-shot DINOv3 | 0.72 | 0.66 | 0.56 | 0.48 | 0.46 |
| Fine-tuned **M1** (≤30 dB) | 0.98 | 0.85 | 0.70 | 0.50 | 0.46 |
| Fine-tuned **M2** (all dB) | **0.99** | **0.98** | **0.94** | **0.87** | **0.74** |

- Fine-tuning **beats both deployed detectors** across the board; **M2 degrades
  gracefully to 74% detection at 60 dB** where everything else collapses to ~46%.
- **M2 ≫ M1 at low SNR** — including low-SNR data in training is what buys low-SNR
  generalization (M1, trained only ≤30 dB, drops off a cliff after 40 dB).
- **Known trade-off (see the false-positive notebook):** M2's low-SNR sensitivity
  causes it to *hallucinate* on some pure-noise frames (~7.5% at 55 dB → ~35% at
  60 dB), whereas M1 never false-alarms on noise. Coherent power emits constant
  low-level speckle. This is the sensitivity/precision trade-off at the noise floor.
- **Wide-bandwidth waveforms (OFDM / 5G / 802.11ax) at low SNR** (see
  `notebooks/wideband_lowsnr_eval.ipynb`): coherent power & zero-shot DINOv3 collapse to
  ~0 detection by 45–50 dB; M1 falls off after ~45 dB; **M2 is the only method that
  survives** — e.g. detection at 55 / 60 dB: OFDM 0.74 / 0.43, 5G 0.67 / 0.34,
  802.11ax 0.29 / 0.02, vs ≈0 for all others. At the **widest channels** even M2 breaks:
  160 MHz 802.11ax fails by 55–60 dB (≤8% coverage), while 98 MHz 5G still partially
  recovers (~51% / 41% coverage at 55 / 60 dB).
- **Failure modes:** OFDM is hard even at high SNR; narrowband (<240 kHz) is
  sub-pixel at this grid. Per-class/bandwidth breakdowns in the notebooks.

---

## Directory layout

```
dino_fine_tuning/
├── README.md                     ← this file
├── checkpoints/<model>/best.pt   ← trained models (M1/M2 × frozen/ft)
├── notebooks/                    ← the 3 comparison notebooks + tables + sweep_detectors/
├── reports/                      ← report.md, pipeline.md, figs*/ (figures)
├── eval_out/<model>/             ← per-model metric CSVs
├── data/dataset/                 ← materialized spectrogram frames + masks + index
├── src/                          ← rfdata, build_dataset, model, train, evaluate,
│                                    finetuned_infer, gen_finetuned_run, report, …
├── configs/                      ← dataset.yaml, train.yaml
├── data_collection/              ← sweep_capture.py, sweep_stats.py (deployment-range sweep)
└── scripts/                      ← setup_env.sh, run_full.sh, run_notebooks.sh
```

## Reproduce

```bash
# environment
bash scripts/setup_env.sh

# full pipeline: dataset -> 4 models -> eval -> report  (resumable)
bash scripts/run_full.sh

# comparison tables (4 detectors) + notebooks
python src/gen_finetuned_run.py                                   # M1 -> finetuned_dino
python src/gen_finetuned_run.py --ft-ckpt checkpoints/M2_ft/best.pt \
       --detector-name finetuned_dino_m2 --ft-eval-meta eval_out/M2_ft/eval_meta.json
python <infocom_evals>/eval_detector_masks.py --batch-root notebooks/sweep_detectors \
       --out-dir notebooks/compare_tables_canonical --coverage-threshold 0.1
bash scripts/run_notebooks.sh
```

## Reproduce: a band/rate-invariant fine-tune (any radio)

Goal: one fine-tune deployable at **any center frequency** and **any valid sample rate** of your
radio, robust across bands. The M1/M2 models above were trained at a *single* rate/band and do **not**
generalize; this workflow domain-randomizes the training data so the model does. Full rationale +
design is in
[`applications/usrp_wideband_signal_detection/notes/retrain_band_rate_invariant_plan.md`](../applications/usrp_wideband_signal_detection/notes/retrain_band_rate_invariant_plan.md).

> **Radio-specific inputs** (adapt these for a different radio — everything else is generic):
> the **valid sample-rate set** (integer decimations of your master clock(s)), which rates need a
> **different FPGA image**, the **rx-gain range**, and the **receiver envelope** (measured, not assumed).

> **Environment.** The dataset/train/eval scripts need **torch** + the **DINOv3 backbone repo on
> `PYTHONPATH`** (the repo is not a pip package). Clone DINOv3 and point `PYTHONPATH` at it, e.g.
> `export PYTHONPATH=/path/to/dinov3`. On this bench the training data lives under a root-only home, so
> the commands run under `sudo` — and because `sudo` strips the environment, pass it inline:
> `sudo env PYTHONPATH=/path/to/dinov3 /path/to/venv/bin/python src/train.py …`. The sweep-capture step
> instead needs the UHD python bindings (run with the system `python3`).

**Step 0 — Sanity check (always run this first).** Verifies the radio is reachable, both channels
capture a sane spectrum (finite, non-flat PSD per channel, prints `OK`/`SUSPECT`), and there is enough
disk for the full grid (estimated usage vs free space; **aborts** if it won't fit). It captures one
burst per rate at mid gain/center and writes two sanity plots — **does not** run the full sweep:
```bash
cd dino_fine_tuning/data_collection
python3 sweep_capture.py --device-args "addr=<radio>" --out-dir <sweeps>/low --preflight
```
Then eyeball the plots in `<sweeps>/low/`:
- `preflight_envelope.png` — per-channel receiver envelope (median-subtracted) overlaid across rates.
  The **terminated** channel should be a clean bowl (edge rolloff, no spikes) and the rates should
  roughly **overlay** (rate-stable envelope). The **antenna** channel shows the same bowl plus any real
  signals as spikes.
- `preflight_psd.png` — absolute antenna-vs-terminated PSD at a mid rate (checks levels + that the
  terminated floor sits at/below the antenna floor).

**Step 1 — Characterize your radio's deployment range (one dual-channel sweep).** On the X410, capture
**channel 0 = antenna** and **channel 1 = terminated (50Ω)** *simultaneously* — one run yields both the
real backgrounds/upper-level (antenna) and the clean envelope/floor-lower-bound (terminated) under
matched conditions. The usable rates are integer decimations of the **master clock**, and which master
clocks are legal depends on the **loaded FPGA image** (X410: **200 MHz image → 245.76/250 MHz clocks**,
max ~245.76 MS/s; **400 MHz image → 491.52/500 MHz clocks**, up to ~500 MS/s). So sweep with the stock
image first (default clocks), then reimage and add the wideband rates:
```bash
# stock 200 MHz image (default --master-clocks-hz = 245.76e6 250e6):
python3 sweep_capture.py --device-args "addr=<radio>" --out-dir <sweeps>/low
# --- reimage the X410 to the 400 MHz-bandwidth FPGA image, then add only the new high rates: ---
python3 sweep_capture.py --device-args "addr=<radio>" --out-dir <sweeps>/high \
        --master-clocks-hz 491.52e6 500e6 --rate-min-hz 260e6
```
An unsupported master clock (wrong image loaded) is **skipped with a message**, not fatal. Any cell that
errors prints `FAILED` → `failures.jsonl` (run continues); repeat just the gaps with `--resume` or
`--retry-failed <dir>/failures.jsonl`. For a different radio, set `--master-clocks-hz` / `--decims` /
`--channel-roles`.

**Step 2 — Extract augmentation statistics** — merges all sweep dirs (both phases):
```bash
python3 sweep_stats.py --run-dirs <sweeps>/low <sweeps>/high --out <sweeps>/stats
```
Emits `envelopes.npz` (per-rate receiver envelope, from the terminated channel), `floor_stats.json`
(per-(rate,role) floor levels → the level-offset augmentation range), and `backgrounds.json` (antenna
IQ index for cut-paste).

**Step 3 — Build the domain-randomized dataset.** Set `domain_randomize: true` + `sweep_stats_dir` (and
the `dr_*` knobs) in `configs/dataset.yaml`, then:
```bash
python src/build_dataset.py --config configs/dataset.yaml --out-suffix _dr
```
Each planned source frame is emulated at every `dr_rates_hz` (≤ source) × `dr_centers_per_frame` random
centers via the capture-chain emulation (`src/rate_augment.py`: freq-shift + measured LPF + decimate,
with label remap + per-rate envelope reshape), stored as **float16 dB** (pre-clip). `dataset.py` then
does dB-domain **level-offset** (gain invariance) + envelope-jitter augmentation and clips to [0,1] at
train time. (Rates *above* the 245.76 source — 491.52/500 — are added via the upsample+paste path once
the wideband-image sweep exists; not yet wired.)

**Step 4 — Train + export** (`src/train.py`, `configs/train.yaml`; then
`export_dinov3_finetuned_torchscript.py --autocast bf16 --dataset-meta data/dataset_dr/dataset_meta.json`).
The exported `meta.json` records `trained_rate_range_hz` (from the dataset's `dr_rates_hz`).

**Step 5 — Validate** with the saved multi-band / multi-rate tests under
`applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/retrain_band_rate/`
(`eval_band_rate.py` = rate-sweep F1 on labeled bands + false-positive-rate on quiet bands, current vs
retrained; plus the existing SNR harness for ISM regression / held-out SNR — see that dir's README).

**Deploy:** point the operator config at the new `.ts`/`.meta.json`; `flatten_noise_floor: false`,
`match_training_power_level: false` (the model is now level/envelope/rate invariant); keep the FFT
processing-gain correction.

> **All code ready (Steps 0–5).** What remains is data-gated: run the sweep, build the DR dataset,
> train + export, run the saved tests. The >source-rate upsample+paste path is wired but dormant until
> the wideband-image sweep provides backgrounds.

## Caveats (before quoting numbers)
- GT boxes mark where a transmitter *was*, even when buried below the noise floor at
  high attenuation — so pixel recall/IoU is low there for *all* detectors.
- Pixel-IoU favors the fine-tuned models (trained to reproduce the filled-box
  convention); **region detection rate is the fair headline metric.**
- ZC/METADATA are detected ~100% at all SNRs and inflate pooled aggregates — read the
  per-class views.

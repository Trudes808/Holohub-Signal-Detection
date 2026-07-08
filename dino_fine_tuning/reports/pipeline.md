# DINOv3 Signal-Detection Fine-Tuning — Pipeline Documentation

**Author:** generated for bqn82 (Brandon Nguyen) · **Date:** 2026-07-07
**Location:** `~/Holohub-Signal-Detection/dino_fine_tuning/`

This document explains the end-to-end system that turns raw SigMF captures into a
fine-tuned DINOv3 signal-vs-noise detector and evaluates it. It is the companion
to [`report.md`](report.md), which presents the scientific results. Every place
where the implementation deviates from the original task brief is called out in
**[Deviation]** notes and collected in [§11](#11-deviations-from-the-original-brief).

---

## 1. Goal

Determine whether **fine-tuning DINOv3 for spectrogram signal detection improves
detection at low SNR**, relative to (a) a decoder trained on frozen DINO features
and (b) a classical energy detector. The deliverable is a reproducible pipeline
plus quantitative evidence sliced by attenuation (SNR proxy), waveform class,
bandwidth, and pulse length.

The task is framed as **binary per-pixel semantic segmentation** on the
spectrogram: each time-frequency pixel is *signal* (1) or *noise* (0). This
matches the deployed `cuda_dino` detector's mask output contract and lets us
reuse the existing evaluation code.

---

## 2. Repository layout

```
dino_fine_tuning/
├── configs/
│   ├── dataset.yaml         # geometry, dB calibration, frame selection, splits
│   └── train.yaml           # optimizer, loss, schedule, AMP
├── src/
│   ├── rfdata.py            # SigMF I/O, GPU spectrogram, GT-mask rasterization
│   ├── build_dataset.py     # 2-pass dataset builder (plan -> materialize)
│   ├── dataset.py           # torch Dataset over the materialized stacks
│   ├── model.py             # DINOv3 backbone + seg head + Dice/BCE loss
│   ├── train.py             # training loop (frozen / ft_lastN)
│   ├── evaluate.py          # per-frame + per-region metrics on the test split
│   └── report.py            # figures + tables aggregation
├── scripts/
│   ├── setup_env.sh         # conda env `dinov3` (torch cu124)
│   └── run_full.sh          # orchestrates the whole run (resumable)
├── data/dataset/            # materialized frames/masks + index CSVs (built)
├── checkpoints/<model>/     # best.pt / last.pt / history.json per model
├── eval_out/<model>/        # frame_metrics.csv / region_metrics.csv per model
└── reports/                 # report.md, pipeline.md, figs/, tables, STATUS.txt
```

---

## 3. Environment

Standalone conda env `dinov3` (offline PyTorch; **not** the container app):

```bash
bash scripts/setup_env.sh          # python 3.11, torch 2.6.0+cu124, torchvision,
                                   # torchmetrics, numpy/scipy/pandas/sklearn,
                                   # matplotlib/seaborn, omegaconf, einops, ...
```

- GPU: NVIDIA RTX 4000 Ada (20 GB, sm_89), driver CUDA 12.4 → cu124 wheels.
- The DINOv3 code is imported from `~/dinov3` (on `PYTHONPATH`); backbone weights
  are the local `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` (ViT-B/16, 85.7 M).

---

## 4. Input data

- 14 SigMF pairs in `~/captures`: `attenuation_dB_{0,5,10,15,20,25,30,30_v2,35,40,45,50,55,60}`.
- Format `cf32_le` (interleaved float32 I/Q = complex64), **245.76 MSps**, single
  channel, baseband centered at 0 Hz (±122.88 MHz), ~7.09 s each (~13 GB, 185 GB total).
- **[Deviation]** The deployed live app runs the USRP at 500 MSps; these offline
  captures are 245.76 MSps. All geometry below is derived from the capture's own
  sample rate, not the live 500e6 config.

### Annotations (ground truth)
Each `.sigmf-meta` carries per-emission annotations with
`core:sample_start`, `core:sample_count`, `core:freq_lower_edge`,
`core:freq_upper_edge`, `core:label`, `wfgt:kind`, `wfgt:time_group`. Across a
capture there are ~3,600 annotations spanning 11 label types (BPSK, QPSK, 16QAM,
OFDM, 5G_Downlink, 802_11ax, Bluetooth, Narrowband/Broadband FM, plus `ZC`
Zadoff-Chu sync and `METADATA` framing markers), bandwidths **8 kHz–160 MHz**,
durations **0.012–20 ms**.

- **Signal definition:** *all* annotation kinds — including ZC and METADATA — are
  treated as **signal** (any transmitted energy above the noise floor). Only the
  unannotated noise floor is *noise*. (User-confirmed.)

---

## 5. Spectrogram & label geometry

Implemented in `src/rfdata.py`; identical convention to the deployed detector and
to the existing `eval_viz.spectrogram_db_from_iq`:

- A **frame** is a `(frame_rows, nfft)` image. Row *r* is the `nfft`-point
  `fftshift(fft(...))` of the contiguous IQ block `iq[r*nfft:(r+1)*nfft]`, taken
  as `10·log10(|·|²)`. Time increases downward; frequency increases left→right,
  DC at the center column.

| Parameter | Value | Meaning |
|---|---|---|
| `nfft` | **1024** | frequency bins → 245.76e6 / 1024 = **240 kHz/bin** |
| `frame_rows` | **256** | time rows per frame |
| frame span | **1.067 ms** | `frame_rows·nfft` = 262,144 samples @ 245.76 MSps |
| row span | 4.167 µs | one `nfft` block |
| DINO input | **256 × 1024** | native grid == model input (patch 16 → 16×64 = 1024 tokens) |

Rationale: the grid matches the deployed DINO tile's time×freq aspect while
keeping a single grid for the image, the GT mask, and the prediction — **no
resampling** is needed anywhere, so training labels and eval scoring are
pixel-identical. DINOv3's RoPE position embedding handles the non-square
`256×1024` input natively.

- **[Deviation]** The brief suggested "~1-second segments" (a 10 s file → 10
  frames). At the model's native tile scale each frame is ~1 ms, so a capture
  yields thousands of frames — necessary for a real training set (~21 k frames
  total vs. ~140). We segment at the model scale, not 1 s.
- **Consequence (documented, not a bug):** narrowband emissions (< ~240 kHz, e.g.
  the 8–26 kHz FM tones) are sub-pixel in frequency and will be hard to detect
  for *any* method at this grid. This is surfaced in the bandwidth breakdown
  rather than hidden.

### GT mask rasterization (`build_frame_mask`)
For each annotation overlapping a frame:
`row = round((sample − frame_start)/nfft)`, `col = round((f_hz + Fs/2)/Fs·nfft)`,
clipped to the frame and guaranteed ≥1 px. The union of annotation rectangles is
the binary signal mask. Per-annotation boxes (with class/bandwidth/length/
time_group) are recorded separately for **evaluation-time slicing only** — they
are never used as training targets.

- **[Deviation]** GT is a filled-box segmentation mask (IoU = mask IoU), not a
  bounding-box regression target. This matches the deployed detector and the
  existing `mask_eval_metrics.py`.

**Validation performed:** at 0 dB, 97% of GT-mask pixels are genuinely bright
(> noise+6 dB) with **+28.6 dB** signal-vs-noise separation, confirming correct
frequency/time registration. At 45 dB the separation drops to +4.3 dB and only
26% of GT pixels rise above noise — the expected low-SNR regime.

### Global dB → uint8 mapping
Frames are stored as `uint8` using a **single global** `[db_vmin, db_vmax]`
linear map (1st / 99.9th percentile of dB pooled across all captures during a
calibration pass). Per-frame normalization is deliberately avoided so that a
strongly attenuated frame really does look fainter than a 0 dB frame — essential
for a low-SNR study. The FFT uses a rectangular window (no taper), matching the
deployed detector; the resulting sidelobes are a property of the data shared by
all detectors.

---

## 6. Dataset builder (`build_dataset.py`)

Two passes, resumable, GPU-batched FFT:

1. **Plan** (no FFT): scan candidate frames per capture, classify signal/noise via
   the mask, keep all signal frames + a `noise_to_signal_ratio ≈ 0.6` sample of
   noise-only frames, cap at `max_frames_per_capture = 1500`, then assign a
   **temporal** train/val/test split (0.70/0.15/0.15) with an 8-frame guard gap
   between splits to prevent adjacent-frame leakage.
2. **Materialize**: compute spectrograms on GPU, write `uint8` frame stacks and
   `uint8` mask stacks (`frames_{split}.npy`, `masks_{split}.npy`) plus
   `frames.csv` (per-frame index: id, stem, split, attenuation, is_signal,
   mem_pos) and `regions.csv` (per-annotation boxes + attributes).

Every capture contributes to all three splits, so **every attenuation level
appears in the test set** — required for per-dB metrics on all models.

Smoke and full builds share the code; the smoke build used
`--limit-captures ... --max-frames 200 --out-suffix _smoke`.

---

## 7. Model (`model.py`)

- **Backbone:** DINOv3 ViT-B/16 (local LVD-1689M weights), features pulled from
  layers `[2, 5, 8, 11]` via `get_intermediate_layers(..., reshape=True)` →
  four `(B, 768, 16, 64)` maps.
- **Head (`SegHead`):** 1×1 projection of each layer → concat → four
  Conv-GroupNorm-GELU blocks with ×2 bilinear upsampling each (16×64 → 256×1024)
  → 1×1 conv to a single logit channel. ~2 M params.
- **Input handling:** grayscale spectrogram `[B,1,H,W]∈[0,1]` → repeated to 3
  channels → ImageNet-normalized (matching the deployed detector's mean/std).
- **Adaptation modes:**
  - `frozen` — backbone frozen & in eval mode; only the head trains (~2 M params).
  - `ft_lastN` — additionally unfreeze the **last 4 transformer blocks + final
    norm** (~30 M trainable). **[Deviation/choice]** We implement backbone
    adaptation as last-N-block unfreezing rather than LoRA (simpler, robust, fits
    in 4.2 GB at batch 8); user approved "frozen-head baseline + backbone-adapt".
- **Loss:** `BCEWithLogits(pos_weight=3) + soft Dice` to handle signal/noise pixel
  imbalance.

---

## 8. Training (`train.py`)

- AdamW, separate LRs for head (`3e-4`) and unfrozen backbone (`1e-5`), cosine
  schedule with 5% warmup, gradient clipping, **bf16** autocast.
- 25 epochs, batch 16, light augmentation (frequency flip, circular time roll,
  small additive noise). Validation each epoch computes pixel IoU/F1/P/R at
  threshold 0.5; **best-val-IoU** checkpoint is saved. Fully resumable
  (`--resume` from `last.pt`).

---

## 9. Experiments

| Model | Training data | Backbone |
|---|---|---|
| `energy` | — (non-learned baseline) | adaptive dB threshold (median + 4·MAD) |
| `M1_frozen` | attenuation ≤ 30 dB | frozen |
| `M1_ft` | attenuation ≤ 30 dB | last-4 unfrozen |
| `M2_frozen` | all attenuations | frozen |
| `M2_ft` | all attenuations | last-4 unfrozen |

All learned models share the **same all-dB test split**. This 2×2 (training data
× adaptation) grid separates two questions:
- **Backbone adaptation:** `frozen` vs `ft` — does adapting DINO features to
  spectrograms help, especially at low SNR?
- **Training-data coverage:** `M1` (high-SNR only) vs `M2` (all SNRs) — does
  including low-SNR data during training improve low-SNR test performance?

- **[Deviation/addition]** The `energy` baseline was not requested but is a cheap,
  informative "no-ML" floor for the PI/collaborators.

---

## 10. Evaluation (`evaluate.py`, `report.py`)

- Decision threshold per learned model is **tuned on the val split** (max micro-F1)
  then applied to test — a fair per-model operating point.
- **Metric primitives are reused** from
  `applications/.../infocom_evals/signal_detection_experiments/mask_eval_metrics.py`
  (`pixel_metrics`, `region_coverage`, `bucket_bandwidth`, `bucket_length`) so
  numbers are on the same yardstick as the deployed detector's eval.
- **Per-frame:** IoU, precision, recall, F1 on signal frames; FP-area fraction on
  noise-only frames. **Per-region:** coverage of each GT box; a region is
  "detected" if coverage ≥ 0.30.
- Figures (`reports/figs/`): pixel metrics vs attenuation; FP-rate vs attenuation;
  region detection rate vs attenuation; detection by waveform class / bandwidth /
  pulse length; class×attenuation heatmap. Tables in `reports/metrics_tables.md`,
  machine-readable aggregates in `reports/summary.json`.

---

## 11. Deviations from the original brief

1. **Frame duration:** ~1.067 ms model-native tiles (thousands/capture), not 1 s
   segments — required for a trainable dataset and to match the DINO input.
2. **Task:** per-pixel binary segmentation with filled-box GT (IoU = mask IoU),
   not literal bounding-box regression — matches the deployed detector & eval.
3. **Signal definition:** all annotation kinds incl. ZC + METADATA = signal
   (user-confirmed).
4. **Class labels:** used for **evaluation slicing only**; training is purely
   binary signal/noise (user-confirmed).
5. **Backbone adaptation** = last-4-block unfreeze (not LoRA).
6. **Energy-threshold baseline** added for context.
7. **Split:** temporal within each capture (every dB in every split), rather than
   holding out whole attenuations.
8. **Global dB normalization** to preserve cross-SNR contrast.
9. **Frame cap** 1500/capture (subsampled from ~6,600) and noise:signal ≈ 0.6 for
   tractable storage/compute.

---

## 12. Reproduction

```bash
# one-shot (dataset -> 4 models -> 5 evals -> report), resumable
bash scripts/run_full.sh
# monitor
tail -f reports/STATUS.txt

# or step by step
PYTHONPATH=~/dinov3:src python src/build_dataset.py --config configs/dataset.yaml
PYTHONPATH=~/dinov3:src python src/train.py --config configs/train.yaml \
    --dataset data/dataset --mode ft_lastN --atten-max 30 --name M1_ft --out checkpoints/M1_ft
PYTHONPATH=~/dinov3:src python src/evaluate.py --config configs/train.yaml \
    --dataset data/dataset --ckpt checkpoints/M1_ft/best.pt --name M1_ft --out eval_out
PYTHONPATH=src python src/report.py --eval-root eval_out \
    --models energy,M1_frozen,M2_frozen,M1_ft,M2_ft --reports reports --heatmap-model M2_ft
```

---

## 13. Limitations & future work

- Narrowband (< 240 kHz) emissions are sub-pixel at this frequency grid; a
  higher-`nfft` variant (finer GT, DINO input resized) would test them fairly.
- `ft_lastN` unfreezes 4 blocks; a LoRA variant or full fine-tune could be added.
- Detection threshold for the region metric (0.30) and the operating threshold
  (val-F1-tuned) can be swept for ROC/PR-vs-dB curves.
- The test split is temporally adjacent to train within a capture (guard-gapped);
  a stricter capture-level or time-group-level hold-out would further reduce
  leakage risk.

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

### Reports & tables
| item | absolute path |
|---|---|
| Results writeup (figures + narrative) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/reports/report.md` |
| Pipeline / system documentation | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/reports/pipeline.md` |
| Canonical 4-detector metric tables | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/compare_tables_canonical/{region_metrics,frame_pixel_metrics}.csv` |
| Per-model eval CSVs | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/eval_out/<model>/` |
| Combined detector-mask run root (all 4 detectors) | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/notebooks/sweep_detectors/` |
| Materialized training dataset | `/home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/data/dataset/` |

Source captures (SigMF, 245.76 MSps): `/home/bqn82/captures/` ·
Deployed-detector batch masks: `/tmp/usrp_spectrograms/batch_eval/sweep_20260630/`

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

## Caveats (before quoting numbers)
- GT boxes mark where a transmitter *was*, even when buried below the noise floor at
  high attenuation — so pixel recall/IoU is low there for *all* detectors.
- Pixel-IoU favors the fine-tuned models (trained to reproduce the filled-box
  convention); **region detection rate is the fair headline metric.**
- ZC/METADATA are detected ~100% at all SNRs and inflate pooled aggregates — read the
  per-class views.

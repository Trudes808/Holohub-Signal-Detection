# DINOv3 fine-tuning for RF signal segmentation

Fine-tunes a DINOv3 vision transformer to do **binary per-pixel signal-vs-noise segmentation on
spectrogram tiles**, and exports the result as the TorchScript model the live
`cuda_dino_finetuned` detector runs (`operators/finetuned_dino_detector`). `DinoSegmenter`
(`src/model.py`) = DINOv3 backbone + a small multi-scale decoder head; two adaptation modes:
`frozen` (head only) and `ft_lastN` (head + the last 4 transformer blocks).

All commands run **from this directory**. Relative paths in the configs are resolved against it.

---

## 1. Environment

```bash
bash scripts/setup_env.sh        # conda env "dinov3" (torch wheels for CUDA_WHEEL_INDEX, default cu124)
conda activate dinov3
```

Any Python 3.11+ environment with `torch`, `torchvision`, `numpy`, `scipy`, `pandas`,
`scikit-learn`, `matplotlib`, `pyyaml`, `tqdm`, `einops` works instead (on the DGX Spark bench the
shared `.venv-ml` with `torch 2.13+cu130` is used).

The [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) checkout must be a
**sibling of this repository** (`<repo>/../dinov3`, the same place the demo container build looks)
or pointed to by `DINOV3_ROOT`; it is put on `sys.path` automatically. Backbone weights (gated,
Meta license form) go under its `weights/`:

| Backbone | File | Used by |
| --- | --- | --- |
| ViT-B/16 | `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | M1–M4, zero-shot `cuda_dino` |
| ViT-S/16 | `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | M5-S (deployed) |

`configs/train.yaml` points at the ViT-B file via `../../dinov3/weights/...`; edit `weights_path`
or pass `--weights-path` to use another location.

---

## 2. Reproduce the M2 fine-tune on the Argus dataset

**Dataset.** The Argus captures are the attenuation sweep `attenuation_dB_{0,5,…,60}.sigmf-{data,meta}`
(+ `attenuation_dB_30_v2`): 14 single-channel `cf32_le` recordings at 245.76 MSps, ~7 s each, with
per-emission SigMF annotations (`core:sample_start/count`, `core:freq_lower/upper_edge`,
`core:label`, `wfgt:kind`). Put them (or a symlink) at `data/captures`, or set `captures_dir` in
`configs/dataset.yaml`.

**M2** = train on **all** attenuations in `ft_lastN` mode. (M1 = the same with `--atten-max 30`.)

```bash
# 1. materialize the dataset: 256 x 1024 uint8 tiles + filled-box GT masks, 70/15/15 temporal split
python src/build_dataset.py --config configs/dataset.yaml                 # -> data/dataset (~21k tiles)

# 2. train (resumable; 18 epochs, bf16 autocast, Dice+BCE)
python src/train.py --config configs/train.yaml --dataset data/dataset \
       --mode ft_lastN --name M2_ft --out checkpoints/M2_ft --resume    # -> checkpoints/M2_ft/best.pt

# 3. evaluate on the shared test split (pixel + per-region metrics, picks the val threshold)
python src/evaluate.py --config configs/train.yaml --dataset data/dataset \
       --name M2_ft --ckpt checkpoints/M2_ft/best.pt --out eval_out         # -> eval_out/M2_ft/

# 4. export for the live operator (traces backbone + head; writes <out>.meta.json geometry contract)
python ../applications/usrp_wideband_signal_detection/export_dinov3_finetuned_torchscript.py \
       --ckpt checkpoints/M2_ft/best.pt --output weights/finetuned_dino_m2_ft_bf16.ts --autocast bf16 \
       --dataset-meta data/dataset/dataset_meta.json --eval-meta eval_out/M2_ft/eval_meta.json
```

`scripts/run_full.sh` runs steps 1–3 for all four M1/M2 × frozen/ft models plus the energy
baseline and builds `reports/` (resumable; `MODELS="M2_ft"` to train just one; `PY=<python>` to
pick the interpreter).

**Deploy.** In the app config's `finetuned_dino_detector` block set `model_script_path` to the
`.ts` (container path `/workspace/holohub/dino_fine_tuning/weights/...`) and copy `tile_rows`,
`nfft`, `fft_window`, `db_vmin`, `db_vmax`, `threshold` from the `.meta.json`. The operator runs
its own `nfft`-point FFT to reproduce the trained per-pixel physics, so a model trained at
245.76 MSps geometry is applied to a 491.52 MSps stream through the `real_time_downsample`
front end; see §3 for the models trained directly at the deployment rate.
The app also offers a short-FFT variant of that front end (`downsample_time_pool`, app README §7,
`config_live_v3_dino_ft_m5s_sf2048.yaml`). M5-S was trained on the wide-FFT front end that
`src/frontend.py` replicates, so port the pooling into `src/frontend.py` before retraining a model for
that deployment.

---

## 3. Model lineage and the deployed model

| Model | Backbone | Training data / front end | Status |
| --- | --- | --- | --- |
| M1_ft / M2_ft | ViT-B/16 | Argus attenuation sweep, ≤30 dB / all dB, raw dB tiles (§2) | the reference fine-tunes; M2 is the low-SNR winner (74% region detection at 60 dB attenuation vs ~46% for every other detector) |
| M2_dr | ViT-B/16 | M2 recipe with sample-rate domain randomisation (20.48–245.76 MSps, Hann) | rate-invariant model used by the 245.76 MSps replay demo (`config_dino_finetuned_viz_demo.yaml`) |
| M3_491 | ViT-B/16 | synthetic dense+sparse composites at **491.52 MSps** through the exact deployment front end (`src/frontend.py`: flatten + robust p20 floor normalisation) | first model trained at the deployment rate; held-out detection 83–100 % across SNR |
| M4_491_ota | ViT-B/16 | M3 data + real over-the-air captures labelled by the coherent detector (teacher) + antenna-off noise negatives | fixed the real-OTA domain gap |
| **M5S_491_ota** | **ViT-S/16** | M4 data | **deployed**: ~10 ms per tile pair, so the live detector runs on every frame at 491.52 MSps |
| M6S_multirate / **M7S_scenes_v2** | ViT-S/16 | multi-rate / multi-scenario synthetic scenes (`src/gen_scenes.py`: hoppers, adjacent cellular carriers and uplinks, ISM near-far collisions, 5 GHz, drones, land-mobile radios, broadcast FM, band-edge cut-offs, every X410 rate 5–500 MSps; realism calibrated on real captures) + the M4 OTA tiles | **M7-S = candidate successor of M5-S since 2026-11-03** (`config_live_v3_dino_ft_m7s.yaml`; M8-S / v3 scored higher on synthetic sets but filled quiet high-gain frames on real N320 captures): real cellular/LMR/NR test recordings pixel recall 0.33 → 0.95, region det@0.5 24 → 80 %; synthetic held-out detection 65 → 87 %; legacy 491.52 held-out unchanged or better; OTA A/B pending (`config_live_v3_dino_ft_m7s.yaml`). Details in [`infocom_evals/.../dino_ft_multirate_scenes/`](../applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/dino_ft_multirate_scenes/README.md) |
| M8S_scenes_v3 / M9S_headv2 / **M10S_v5** / M11S_v6 | ViT-S/16 | generator v3 (fill keeps an edge, DC spur labelled) → v4/v5 (`SegHeadV2` full-resolution input stem + quiet-pixel BCE weight; real terminated-input N320 backgrounds, spur combs labelled line by line, live-ingest artefacts) → v6 (gain-step artefact dropped as label noise, dense FHSS networks, regular spur combs) | **M10-S (2026-11-05) is the first model without the quiet-band "comb fill"** (synthetic comb probe 0 % filled tiles vs 12–25 % for M7/M9; 900 MHz ISM capture 7 of 116 tiles vs 34) and has the best real-data numbers of the lineage (hard cases det@0.5 88 % vs 80 %, pixel P 0.98; Argus P 0.80), at the cost of some cellular recall at 5–10 MSps traced to the gain-step artefact. **M11-S (v6, 2026-11-06)** is the best model on every labelled set (v6 held-out IoU 0.841 vs 0.809 / 0.741 for M10-S / M7-S, hard cases IoU 0.931) and marks the 900 MHz comb line by line, but with the quiet-pixel weight lowered to 1.0 it fills tiles on one narrow 10.24 MSps uplink look and paints the DC pedestal of the quiet reference; **M12-S (v6 data, weight 2.0) is training** to separate the two effects. The winner replaces M7-S as the candidate (operating threshold 0.5, see the experiment record). |

To retrain the deployed lineage (needs the waveform library for the synthetic composites,
`ARGUS_WAVEFORM_LIB`, and OTA captures):

```bash
python src/build_dataset_rt.py --out-dir data/dataset_491                 # synthetic composites @ 491.52 (gen_491.py)
./label_ota.sh                                                            # coherent-teacher masks for antenna-on OTA captures (container)
python src/build_dataset_ota.py --labels-dir /tmp/usrp_spectrograms/ota_labels --out-dir data/dataset_ota
python src/combine_datasets.py --inputs data/dataset_491 data/dataset_ota --out data/dataset_491_plus_ota
python src/train.py --config configs/train.yaml --dataset data/dataset_491_plus_ota --mode ft_lastN \
       --backbone vits16 --weights-path ../../dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
       --name M5S_491_ota --out checkpoints/M5S_491_ota --resume
python src/eval_heldout.py --ckpt checkpoints/M5S_491_ota/best.pt --dataset data/dataset_491   # threshold sweep + per-SNR detection
python ../applications/usrp_wideband_signal_detection/export_dinov3_finetuned_torchscript.py \
       --ckpt checkpoints/M5S_491_ota/best.pt --output weights/finetuned_dino_m5s_491_bf16.ts \
       --autocast bf16 --sample-rate-hz 491.52e6 --fft-window hann --threshold 0.6
```

**Retrain with the scenario generator (M6-S recipe).** Measures the real-data statistics, builds the
multi-rate dataset (parallel scene synthesis + the deployment front-end at each scene's own rate), merges it
with the OTA tiles, trains, evaluates against M5-S and exports; every step is resumable:

```bash
python src/real_stats.py                                                   # -> data/real_stats (floor templates, OTA stats)
python src/build_dataset_scenes.py --out-dir data/dataset_scenes --workers 12   # ~16k tiles + heldout_sigmf/ (--smoke to test)
scripts/run_m6.sh all          # combine -> train (M6S_multirate) -> eval_scenes.py / eval_heldout.py -> export .ts
python src/gen_scenes.py --scenario drone_ism --rate 61.44e6 --png /tmp/preview.png   # eyeball one scene
python src/eval_scenes.py --model checkpoints/M6S_multirate/best.pt --sigmf <dir-of-sigmf> --real --out eval_out/x
```

`eval_scenes.py` evaluates any SigMF recording (cf32 / ci16 / cf64, any rate) through the deployment
front-end and reports region detection by scenario, rate bucket, PSD SNR, band-edge visibility, size and
class, plus pixel metrics and overlays; `--real` re-bases absolute-RF annotation edges to baseband and
`--assume-continuous` extends labels over the whole recording.

The design record for this lineage (why robust normalisation, what the ablations showed, the
ingest guard) is in
[`applications/usrp_wideband_signal_detection/notes/dino_ft_finetune_plan.md`](../applications/usrp_wideband_signal_detection/notes/dino_ft_finetune_plan.md)
and the demo handoff
[`notes/live_demo_dev_state.md`](../applications/usrp_wideband_signal_detection/notes/live_demo_dev_state.md).

---

## 3a. Front-end contract (read before running a model anywhere else)

Every model from M3 on was trained behind the **deployment front end**, and only produces sensible masks
behind it. Feeding it a plain 1024-point-per-row spectrogram with a fixed dB clip (the M1/M2 geometry)
makes it paint whole tiles (verified 2026-11-05 on `OOD_captures/OTA/test_1`: 454 of 2812 native tiles at
100 % occupancy vs 0 of 280 frames through the proper front end). Per frame at sample rate R:

1. **Frame** = 512 rows × F samples, F = `rates.app_fft_size(R)` (the app's wide FFT: 20480 at 491.52/500
   MSps, 10240 at 245.76, 5120 at 122.88, 3072 at 49–83, 1024 below ~22 MSps). Row time = F/R (41.7 µs at
   245.76), not 1024/R.
2. **Per row**: Hann window scaled to unit RMS, F-point FFT, fftshift, power in dB, minus 10·log10(F/1024).
3. **Flatten**: per-frequency column mean, Gaussian-smoothed (σ = max(2, 0.005·F)), two passes with signal
   capped at reference + 6 dB; add `clamp(reference − floor, 0, 12 dB)` per column.
4. **Robust normalisation**: p20 / p95 of the flattened image; if p95 − p20 ≥ 8 dB and p20 is within 25 dB
   of the calibrated vmin: `value = clamp((dB − (p20 − 0.12·34)) / 34, 0, 1)`, else the fixed clip
   (`db_vmin` / `db_vmax` of the export's `.meta.json`).
5. **Resize** frequency F → 1024 bilinearly (align_corners = false, no antialias).
6. **Tile** the 512 rows into two 256 × 1024 tiles, float in [0, 1], shape [B, 1, 256, 1024]. The export
   returns logits; mask = sigmoid ≥ threshold (0.6).

Reference implementations: `src/frontend.py::compute_front_end` (+ `src/rates.py`; Python twin of the CUDA
operator, corr 1.0) and the C++ operator with `real_time_downsample: true`, `flatten_noise_floor: true`,
`adaptive_normalization: true`, `adaptive_robust_floor: true` (`config_live_v3_dino_ft_m7s.yaml`). To run a
TorchScript export on another machine you need only `src/run_ts_on_sigmf.py`, `src/frontend.py`,
`src/rates.py`, the `.ts` and torch:

```bash
python run_ts_on_sigmf.py --ts finetuned_dino_m7s_scenes_v2_bf16.ts --capture x.sigmf-meta --out out_dir --png 4
```

---

## 4. Weights

`weights/` and `checkpoints/` are gitignored. Release artifacts to distribute alongside the code:

| File | Needed for |
| --- | --- |
| `weights/finetuned_dino_m5s_491_bf16.ts` + `.meta.json` | the live demo and every `*_m5s*` config |
| `weights/finetuned_dino_m4_491_bf16.ts`, `..._m3_491_...`, `..._m2_dr_...` | legacy configs only (see the app README §6) |
| `checkpoints/<model>/best.pt` | retraining / re-export (each bundles backbone + head + the val threshold) |

---

## 5. Layout

```
dino_fine_tuning/
├── configs/      dataset.yaml (geometry, dB calibration, splits), train.yaml (optimizer, loss, schedule)
├── src/          rfdata (SigMF I/O, GPU spectrogram, GT rasterization) · build_dataset (Argus sweep)
│                 build_dataset_rt / gen_491 (491.52 synthetic) · build_dataset_ota + ../label_ota.sh (OTA teacher labels)
│                 rates (X410 rate set + per-rate app geometry) · real_stats (receiver / OTA statistics) · synth_mods
│                 (modulators + rate-agnostic placer + receiver model) · gen_scenes (scenario scenes, any rate) ·
│                 build_dataset_scenes (multi-rate dataset) · eval_scenes (any-rate SigMF eval) · check_scene_levels ·
│                 run_ts_on_sigmf (standalone .ts inference, correct front end) · scan_capture / run_ota_detections (per-frame scans)
│                 combine_datasets · dataset · model · train · evaluate · eval_heldout · frontend (deployment front end)
│                 finetuned_infer / finetuned_rt_infer (Python twins of the operator) · gen_finetuned_run / gen_compare_tables
│                 (materialize masks for the shared eval harness) · paths (portable path resolution)
├── scripts/      setup_env.sh · run_full.sh · run_m6.sh (scenario retrain pipeline) · run_notebooks.sh · resmon.sh
├── ab_norm_validate.py, ab_span_floor_sweep.py   ground-truth A/B harnesses for the front-end knobs
├── data/         materialized datasets (gitignored) · real_stats/ (committed statistics) · hard_test_cases/ (real test
│                 recordings: metas committed, IQ local)      checkpoints/   weights/   eval_out/   (gitignored)
└── reports/      report.md (results), pipeline.md (system description), metrics tables, figures
```

Evaluation notebooks live in [`notebooks/dino_fine_tuning_evals/`](../notebooks/dino_fine_tuning_evals/)
(detector comparison, low-SNR false positives, wideband and LTE out-of-distribution studies,
live-data galleries).

---

## 6. Results summary (Argus sweep, M1/M2)

Region detection rate vs attenuation (coverage ≥ 0.1), scored by the shared
`eval_detector_masks.py` harness:

| detector | 40 dB | 45 dB | 50 dB | 55 dB | 60 dB |
| --- | --- | --- | --- | --- | --- |
| Coherent Power | 0.69 | 0.63 | 0.57 | 0.52 | 0.47 |
| Zero-shot DINOv3 | 0.72 | 0.66 | 0.56 | 0.48 | 0.46 |
| Fine-tuned M1 (≤30 dB) | 0.98 | 0.85 | 0.70 | 0.50 | 0.46 |
| Fine-tuned **M2** (all dB) | **0.99** | **0.98** | **0.94** | **0.87** | **0.74** |

Caveats before quoting numbers: GT boxes mark where a transmitter *was*, even below the noise
floor, so pixel recall is low for every detector at high attenuation; pixel IoU favours the
fine-tuned models (trained on the filled-box convention), so **region detection rate is the fair
headline**; M2's low-SNR sensitivity costs false alarms on pure-noise frames (~7.5 % at 55 dB,
~35 % at 60 dB). Full narrative and figures: [`reports/report.md`](reports/report.md); system
description and every deviation from the original brief: [`reports/pipeline.md`](reports/pipeline.md).

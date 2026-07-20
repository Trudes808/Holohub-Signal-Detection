# Detector comparisons — six detectors on a shared SNR axis

**Six** signal detectors evaluated on the *same* frames, ground truth, and metrics,
then plotted against a physically-meaningful **SNR (dB)** axis (shared −20…+40 dB
range) so you can see how classic signal/image processing, the deployed detectors,
and the fine-tuned ML models each degrade with SNR — all fairly comparable because
every detector is scored by one identical `eval_detector_masks.py` code path.

| `detector_type`   | Family | What it is |
|-------------------|--------|------------|
| `coherent_power`  | deployed | Coherent power detector (C++/CUDA offline binary). |
| `cuda_dino`       | deployed | Zero-shot DINOv3 (C++/CUDA offline binary). |
| `3dB_power`       | baseline (non-ML) | Static single-threshold power detector. **One** threshold for the whole frame (not per bin): a scalar noise-floor reference + `threshold_db` (default 3 dB). Floor = per-frame percentile of dB power by default, or an absolute `floor_db`. |
| `blob_detection`  | baseline (non-ML) | Textbook edge-based CV, deliberately **not** tuned: Gaussian smooth → Sobel gradient (edges) → percentile threshold → morphological closing → fill enclosed regions → connected components ≥ min area. |
| `yolo`            | fine-tuned ML | Fine-tuned YOLO26 (Ultralytics); detection boxes rasterized to a binary mask. |
| `dino_finetuned`  | fine-tuned ML | Fine-tuned DINOv3 segmenter (M2_ft: last-4 blocks unfrozen, trained on all attenuations). |

Every detector emits a binary `uint8` mask on the **identical FFT grid** as the
trained detectors, dropped into the batch root as a sibling `<detector>/` dir, so the
entire downstream chain (`eval_detector_masks.py` → `build_snr_results.py` →
`plot_snr_results.py` → the notebook) is detector-agnostic and reused unchanged.
The baselines are pure Python (numpy + scipy, no GPU); the ML detectors need a GPU +
their trained weights; the deployed detectors need the demo container.

## Files

- `run_full_comparison.py` — **top-level orchestrator**: runs every stage below over
  one batch root (idempotent, resumable, per-stage flags). Start here.
- `comparison_config.yaml` — **single source of truth**: batch root, captures, all
  six detectors' params/weights/paths, and the SNR-axis knobs.
- `run_ml_detectors_offline.py` — driver for the two fine-tuned ML detectors
  (`yolo`, `dino_finetuned`); path-robust wrapper around the colleague's
  `yolo_infer` / `finetuned_infer` classes (see "Why this, not gen_*_run.py" below).
- `baseline_detectors.py` — the two non-ML detector algorithms + a registry.
- `run_baseline_offline.py` — driver for the non-ML baselines.
- `baseline_detectors_config.yaml` — standalone baseline params (still usable on its
  own; the orchestrator synthesizes an equivalent from `comparison_config.yaml`).
- `snr_measurement.py` / `build_snr_results.py` — SNR calibration + the serialized
  `SnrResults` object.
- `plot_snr_results.py` — SNR-axis figures (shared −20…+40 dB axis, consistent
  per-detector colors/markers; `cuda_dino` is labelled `zero_shot_dino` in plots).
- `baseline_eval_review.ipynb` — visual review + the SNR-axis 6-detector comparison.
- `OTA_eval.ipynb` — qualitative review on known captures + a live over-the-air (OTA)
  collection: per-detector spectrogram/mask panels (Cases 1–3) plus a compact,
  one-column composite (`compact_grid()`) for the paper.
- `save_plot_data.py` — extract the small plot-regeneration artifacts (metrics CSVs +
  the `SnrResults` object) into `saved_results/<run_id>/` so the large per-frame mask
  arrays can be deleted without losing the ability to regenerate/tweak the plots.
- `evaluation_section.tex`, `paper_sections_system_detectors.tex`, `figs/` — LaTeX for
  the paper's evaluation and system/detector sections, plus the generated figures.

## Quick start — the full 6-way sweep

```bash
cd infocom_evals/baseline_comparisons
# 1. edit comparison_config.yaml: set batch_root (a run id), captures_dir, and the
#    ml_detectors weight paths (defaults point at the /home/bqn82 trained tree).
# 2. produce the trained detectors first (needs the demo container; run_full_comparison
#    prints the exact command if they're missing), then run everything else headless:
python3 run_full_comparison.py --config comparison_config.yaml \
    --batch-root ../signal_detection_experiments/batch_runs/<run_id>
```

Stages (each idempotent; select with `--stages`, skip with `--skip`):
`preflight` (verify trained detectors present) → `baselines` (3dB_power, blob_detection)
→ `ml` (yolo, dino_finetuned; GPU) → `eval` (score all six) → `snr` (calibrate + join)
→ `plots` (SNR figures). When it finishes it prints the `BATCH_ROOT` / `RESULTS`
lines to paste into the notebook's first cell.

The GPU stages (`ml`, and the trained sweep flagged by `preflight`) run where the
weights + container live; the CPU stages (`baselines`, `eval`, `snr`, `plots`) run
anywhere. Re-invoke with `--stages ml eval snr plots` to resume after producing the
trained detectors, etc.

### Why `run_ml_detectors_offline.py` instead of the colleague's `gen_*_run.py`

`yolo_training/src/gen_yolo_run.py` and `dino_fine_tuning/src/gen_finetuned_run.py`
hardcode `/home/bqn82/Holohub-Signal-Detection` on `sys.path` for the **shared** eval
modules (`eval_viz`, `mask_eval_metrics`), so running them from this branch would
score the ML detectors with a *different* checkout's code than the baselines. The
unified driver resolves those shared modules (and `rfdata` / `model`) from **this**
branch and only reads the trained artifacts (weights, dB calibration, dinov3 backbone)
from wherever they live — one identical scoring path for all six detectors.

## How it fits together

The C++ binary `run_offline_cuda_detector_eval` already writes, per capture, a run
directory with the ground truth, a manifest, and (optionally) the spectrogram
tensors:

```
<batch_root>/<detector>/<file_stem>/
    frame_manifest.csv
    gt_masks/            ground_truth_mask_ch<c>_f<f>_<H>x<W>.npy
    gt_annotations/      ground_truth_ch<c>_f<f>_<H>x<W>.json
    spectrogram_tensors/ spectrogram_tensor_ch<c>_f<f>_<H>x<W>.npy   (only with --save-tensors)
    mask_arrays/         mask_ch<c>_f<f>_<H>x<W>.npy
```

`run_baseline_offline.py` (baselines) and `run_ml_detectors_offline.py` (ML) each
reuse that ground truth and write their detectors as **new sibling detector dirs in
the same batch root**:

```
<batch_root>/coherent_power/<file_stem>/ ...   # trained (C++ binary)
<batch_root>/cuda_dino/<file_stem>/ ...         # trained (C++ binary)
<batch_root>/3dB_power/<file_stem>/ ...         # run_baseline_offline.py
<batch_root>/blob_detection/<file_stem>/ ...    # run_baseline_offline.py
<batch_root>/yolo/<file_stem>/ ...              # run_ml_detectors_offline.py
<batch_root>/dino_finetuned/<file_stem>/ ...    # run_ml_detectors_offline.py
```

(the shared `frame_manifest.csv`, `gt_masks/`, `gt_annotations/`,
`spectrogram_tensors/` are symlinked from a reference detector, so no data is
duplicated — GT is detector-independent). A single
`eval_detector_masks.py --batch-root <batch_root>` then scores **all six** detectors
together into one set of fact tables. The ML detectors reconstruct each frame's
spectrogram from the source SigMF IQ at the model's native geometry (nfft=1024,
256-row tiles), then max-pool the mask onto the shared display/GT grid.

**Spectrogram source.** The driver prefers a saved `spectrogram_tensors/*.npy`
(`<c8`); if tensors were not saved it reconstructs the spectrogram from the source
SigMF on the exact same FFT grid (via `eval_viz`), so it works against masks-only
sweeps too.

## Run it

This folder is a sibling of `signal_detection_experiments/` (which holds the shared
eval helpers + `batch_runs/`); paths below are relative to
`applications/usrp_wideband_signal_detection/`.

### 1. Get a source batch run (with ground truth + spectrograms)

Either **reuse an existing** `batch_runs/<run_id>/` (any run has GT + manifest; the
driver reconstructs the spectrogram from SigMF if tensors are absent), **or** produce
a fresh one with tensors saved:

```bash
# from the app root; needs a real shell for sudo docker (see repo CLAUDE.md)
python3 infocom_evals/signal_detection_experiments/run_batch_offline_eval.py \
    --captures-dir /home/bqn82/captures \
    --run-id baseline_source_$(date +%Y%m%d) \
    --detectors coherent_power \
    --save-tensors            # <-- dumps spectrogram_tensors/ the baselines read
```

### 2. Run the baselines

Edit `baseline_detectors_config.yaml` so `source_batch_root` points at that run
(or pass `--source-batch-root`), then:

```bash
cd infocom_evals/baseline_comparisons
python3 run_baseline_offline.py \
    --config baseline_detectors_config.yaml \
    --source-batch-root ../signal_detection_experiments/batch_runs/<run_id>
# run only one, or override captures for SigMF reconstruction:
#   --detectors 3dB_power
#   --captures-dir /home/bqn82/captures
```

By default the baseline dirs are written **into** the source batch root (as
siblings). **If the source run is root-owned** (batch runs produced by the container
eval under `sudo` are owned by `root` and not writable), write to a new, writable
batch root instead — the driver then symlinks the trained detector dirs in so the
output root is a complete comparison root:

```bash
python3 run_baseline_offline.py \
    --source-batch-root ../signal_detection_experiments/batch_runs/<run_id> \
    --out-batch-root ../signal_detection_experiments/batch_runs/<run_id>_with_baselines \
    --captures-dir /home/bqn82/captures
```

(`<run_id>_with_baselines/` then contains `3dB_power/`, `blob_detection/`, and a
`coherent_power` symlink — point the notebook / step 3 at it.)

### 3. Score baselines + trained detectors together

Point `--batch-root`/`--out-dir` at the root that holds the baselines — the source
run if you wrote into it, or the writable `--out-batch-root` from step 2.

```bash
cd ../signal_detection_experiments    # where the eval helpers + batch_runs/ live
python3 eval_detector_masks.py \
    --batch-root batch_runs/<run_id_with_baselines> \
    --captures-dir /home/bqn82/captures \
    --out-dir batch_runs/<run_id_with_baselines>
```

This (re)writes `frame_pixel_metrics.csv` + `region_metrics.csv` covering every
detector directory present, including the two baselines.

### 4. Plot / review

```bash
# still in signal_detection_experiments/
python3 plot_eval_results.py --tables-dir batch_runs/<run_id> --det-threshold 0.1
```

or open `baseline_comparisons/baseline_eval_review.ipynb`, set `BATCH_ROOT` to the
run, and run all cells for the performance-vs-SNR curves and per-frame overlays
(GT vs each detector's mask).

### 5. SNR-axis analysis (physical SNR instead of attenuation)

`attenuation_dB_<N>` is a knob, not a physical quantity: 20 dB of attenuation puts a
wideband 5G burst and a narrowband FM tone at very different signal-to-noise ratios.
`snr_measurement.py` + `build_snr_results.py` re-express the fact tables on a real
**SNR (dB)** axis.

How SNR is defined (see `snr_measurement.py`):

- Measured **once, on the 0 dB capture**, per signal instance:
  - *peak* = mean of the top **2%** of linear FFT power **inside the signal's
    bounding box** (its time rows × its `freq_lower..freq_upper` column band), in dB.
  - *noise* = mean FFT power in the **same frequency band**, in a quiet window
    **3 → 1 ms before that burst's Zadoff-Chu preamble** (`wfgt:zc_sample`), in dB.
  - `snr0_db = peak_db − noise_db`, aggregated (median) per `(signal_class, occupied_bw_hz)`.
- Every other capture is a **physical attenuator step** on the same emitter, so the
  signal drops 1:1 with attenuation while the noise floor is unchanged:
  `snr_db(class, bw, A) = snr0_db(class, bw) − A`. This avoids measuring a peak
  buried in noise at high attenuation.

Build the serialized, reloadable results object (NPZ arrays + JSON sidecar):

```bash
cd infocom_evals/baseline_comparisons
python3 build_snr_results.py \
    --tables-dir ../signal_detection_experiments/batch_runs/<run_id> \
    --captures-dir /home/bqn82/captures \
    --out ../signal_detection_experiments/batch_runs/<run_id>/snr_results
# knobs (defaults = the method above): --peak-top-fraction, --noise-pre-zc-start-ms,
#   --noise-pre-zc-stop-ms, --fft-cols, --max-peak-rows, --max-instances-per-key
```

This writes `snr_results.npz` (region + frame columns, joined with `snr_db` /
`frame_snr_db`) and `snr_results.json` (the per-`(class, bw)` calibration table +
the params/provenance). Frame-level pixel metrics use the **per-frame mean-signal
SNR** (mean SNR of the signals present in the frame).

Plot straight from the object — **no recompute** needed to re-bin, restyle, or add
lines:

```bash
python3 plot_snr_results.py --results ../signal_detection_experiments/batch_runs/<run_id>/snr_results \
    --det-threshold 0.1 --snr-bin-width 5 --snr-range -20 40
```

Every figure shares the `--snr-range` x-axis (default **-20 to 40 dB**) and omits data
outside it, so panels and figures line up for visual comparison; widen it with e.g.
`--snr-range -30 50`. Figures: `rate_vs_snr_by_class`, `rate_vs_snr_by_bandwidth`,
`rate_vs_snr_overall`, `frame_metrics_vs_snr`. From a notebook, reload and tweak
without rerunning:

```python
import snr_measurement as sm, plot_snr_results as psr
res = sm.SnrResults.load(".../snr_results")     # cheap; no captures touched
figs = psr.make_all_figures(res, threshold=0.1, snr_bin_width=2.5,
                            snr_range=(-20, 40))  # re-bin / re-range freely
# pass snr_range=None to auto-fit each axis; or build a custom aggregate straight
# off res.region / res.frame column arrays
```

To add another detector's line, just re-run the eval + `build_snr_results.py`; the
object is rebuilt but plotting stays instant.

### 6. Qualitative review (`OTA_eval.ipynb`)

`OTA_eval.ipynb` renders per-detector **spectrogram + mask** panels for a few hand-picked
cases — a controlled signal frame, a noise-only frame, and a live OTA capture (2.4 GHz
ISM). For the OTA file the notebook loads the two container detectors from a
`live_ota` batch (if present) and fills the four in-notebook detectors live, so all six
can be shown; otherwise it shows the four Python detectors. `compact_grid()` builds the
one-column composite (ground truth on top, detectors as rows, cases as columns) used in
the paper.

### 7. Preserving results for later (`save_plot_data.py`)

The batch runs are hundreds of GB (mostly per-frame masks), but regenerating/tweaking
the aggregate plots needs only a few small files (the metrics CSVs + the `SnrResults`
NPZ/JSON). Extract them so the masks can be deleted to reclaim disk:

```bash
python3 save_plot_data.py             # the config's batch_root
python3 save_plot_data.py --all       # every run under batch_runs/ with metrics
```

This copies the artifacts into `saved_results/<run_id>/` (each with a `MANIFEST.json`
giving the exact regenerate commands). The plots then rebuild from `saved_results/`
alone; the per-frame overlay panels (this notebook / `OTA_eval` Cases 2–3) still need
the masks, so only delete those once you no longer need the visual panels.

## Tuning

All knobs are in `baseline_detectors_config.yaml`:

- **`3dB_power`** (static single global threshold): `threshold_db` (dB above the
  reference floor), `noise_percentile` (the single per-frame floor = this percentile
  of the whole frame's dB power), `floor_db` (set to an absolute value for a
  threshold that is fully fixed across frames — nulls out `noise_percentile`).
- **`blob_detection`** (generic edge-based CV — left untuned on purpose):
  `smooth_sigma` (Gaussian pre-smoothing), `edge_percentile` (keep gradient
  magnitudes above this percentile), `close_iters` (link edge fragments),
  `fill_holes` (fill regions enclosed by edges), `min_blob_area` (discard specks).

> Note: GT masks are *filled bounding boxes*, so sparse detectors read as high
> precision / low pixel-recall. Lean on region **coverage** + a tuned
> `--det-threshold` for "did it find the signal", exactly as for the trained
> detectors.

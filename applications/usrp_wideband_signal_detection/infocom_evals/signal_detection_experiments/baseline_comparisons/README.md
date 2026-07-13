# Baseline detector comparisons (non-ML)

Two classic, **no-machine-learning** signal detectors evaluated on the *same*
frames, ground truth, and metrics as the trained `coherent_power` / `cuda_dino`
detectors — so you can see how far simple signal/image processing gets, and how it
degrades vs SNR.

| `detector_type`   | What it does |
|-------------------|--------------|
| `3dB_power`       | Pure moving-average power detector. For each time row it slides a moving average along frequency and flags every bin that rises `threshold_db` (default 3 dB) above that local average — ON above, OFF below. No calibration, no learning; just a local mean. |
| `blob_detection`  | Classic image-processing blob detection: estimate a smooth background, threshold the residual, morphological open/close, connected-component label, keep blobs above a minimum area. The kept blobs are the mask. |

Both consume the `(time, freq)` spectrogram in dB power and emit a binary `uint8`
mask on the **identical FFT grid** as the trained detectors, so the entire
downstream evaluation (`eval_detector_masks.py` → `plot_eval_results.py` → the
notebook) is reused unchanged. Everything here is pure Python (numpy + scipy) — **no
container rebuild, no GPU, no Holoscan**.

## Files

- `baseline_detectors.py` — the two detector algorithms + a `detector_type` registry.
- `run_baseline_offline.py` — driver: reads an existing batch-eval tree, runs the
  baselines, writes them as sibling detector dirs.
- `baseline_detectors_config.yaml` — detector parameters + source/output paths.
- `baseline_eval_review.ipynb` — visual review + performance-vs-SNR plots (mirrors
  the parent `batch_eval_review.ipynb`).

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

`run_baseline_offline.py` reuses that ground truth + those spectrograms and writes
the baselines as **new sibling detector dirs in the same batch root**:

```
<batch_root>/3dB_power/<file_stem>/ ...
<batch_root>/blob_detection/<file_stem>/ ...
```

(the shared `gt_masks/`, `gt_annotations/`, `spectrogram_tensors/` are symlinked, so
no data is duplicated). A single `eval_detector_masks.py --batch-root <batch_root>`
then scores baselines *and* trained detectors together into one set of fact tables.

**Spectrogram source.** The driver prefers a saved `spectrogram_tensors/*.npy`
(`<c8`); if tensors were not saved it reconstructs the spectrogram from the source
SigMF on the exact same FFT grid (via `eval_viz`), so it works against masks-only
sweeps too.

## Run it

Working directory for these commands is the app root's eval folder:
`applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments`.

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
cd infocom_evals/signal_detection_experiments/baseline_comparisons
python3 run_baseline_offline.py \
    --config baseline_detectors_config.yaml \
    --source-batch-root ../batch_runs/<run_id>
# run only one, or override captures for SigMF reconstruction:
#   --detectors 3dB_power
#   --captures-dir /home/bqn82/captures
```

By default the baseline dirs are written **into** the source batch root (as
siblings). Use `--out-batch-root <dir>` to write them elsewhere.

### 3. Score baselines + trained detectors together

```bash
cd ..    # signal_detection_experiments/
python3 eval_detector_masks.py \
    --batch-root ../signal_detection_experiments/batch_runs/<run_id> \
    --captures-dir /home/bqn82/captures \
    --out-dir batch_runs/<run_id>
```

This (re)writes `frame_pixel_metrics.csv` + `region_metrics.csv` covering every
detector directory present, including the two baselines.

### 4. Plot / review

```bash
python3 plot_eval_results.py --tables-dir batch_runs/<run_id> --det-threshold 0.1
```

or open `baseline_comparisons/baseline_eval_review.ipynb`, set `BATCH_ROOT` to the
run, and run all cells for the performance-vs-SNR curves and per-frame overlays
(GT vs each detector's mask).

## Tuning

All knobs are in `baseline_detectors_config.yaml`:

- **`3dB_power`**: `window_bins` (moving-average length along frequency — make it
  wider than the widest signal so the average tracks noise, not the signal),
  `threshold_db` (dB over the average), `axis` (`freq` per-time-row baseline, or
  `time` per-frequency-column), `min_run_bins` (drop ON runs shorter than this — the
  speckle guard; raising it trades recall for far fewer noise false positives).
- **`blob_detection`**: `background_mode` (`per_freq_median` / `uniform` / `global`),
  `threshold_k` (or absolute `threshold_db`), `open_size` / `close_size`
  (morphology), `min_blob_area` (discard specks), `fill_bboxes` (fill each blob's
  bounding box for a region-style mask).

> Note: GT masks are *filled bounding boxes*, so sparse detectors read as high
> precision / low pixel-recall. Lean on region **coverage** + a tuned
> `--det-threshold` for "did it find the signal", exactly as for the trained
> detectors.

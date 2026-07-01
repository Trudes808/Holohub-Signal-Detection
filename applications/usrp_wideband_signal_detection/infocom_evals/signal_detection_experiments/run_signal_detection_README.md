# Running the Signal-Detection Evaluation — Reproduction Guide

End-to-end recipe to run every SigMF capture through both detectors offline, score the detector
masks against the SigMF ground truth, and visualize the comparison. This is the **how-to-run**
entry point; see `FORMAL_EVAL_WORKFLOW.md` (same directory) for the design, component reference,
and interpretation caveats.

- **Detectors compared:** `coherent_power` and `cuda_dino` (modular — new detectors drop in).
- **Path:** offline, SigMF → FFT → spectrogram → detector → masks, on the real GPU operators.
- **Ground truth:** the binary rasterizes SigMF annotations onto the detector FFT grid.

Directories used below:
- **APP** = `applications/usrp_wideband_signal_detection`
- **EXP** = `APP/infocom_evals/signal_detection_experiments` (this README lives here)
- **CAPTURES** = `/home/bqn82/captures` (14 files: `attenuation_dB_{0,5,10,…,60}.sigmf-{data,meta}` + `dB_30_v2`; `cf32_le`, 245.76 MSps)

> Run the container-touching commands with `sudo` (the wrappers shell into `sudo docker exec`).

---

## 0. Prerequisites (one time)

- Demo container built and the SigMF captures present at `CAPTURES`.
- GPU visible to the container (single RTX 4000 Ada / 20 GB is enough).
- The offline binary + operators are compiled into the container by the rebuild step below.

## 1. Build / rebuild the app in the container

```bash
cd APP
./rebuild_demo_container_app.sh
```
Rebuild whenever you change C++ (`run_offline_cuda_detector_eval.cpp`, operators) or a top-level `config*.yaml`.

## 2. Verify one capture per detector BEFORE the full sweep

Confirms the pipeline is healthy and the frame↔mask alignment is correct (this is the check that
caught the ring-aliasing bug — masks must align at frame offset **k = 0**).

```bash
cd APP
sudo python3 run_cuda_dino_offline_file.py CAPTURES/attenuation_dB_25.sigmf-data --detector cuda_dino      --no-tensors
sudo python3 run_cuda_dino_offline_file.py CAPTURES/attenuation_dB_25.sigmf-data --detector coherent_power --no-tensors

cd EXP
python3 check_mask_alignment.py --batch-root /tmp/usrp_spectrograms/offline_eval --file-stem attenuation_dB_25
# Expect: [PASS] ... best offset k=+0   for both detectors.
```
If it prints `FAIL` with a non-zero offset, the binary is stale/buggy — rebuild (step 1) and re-run.

## 3. Full sweep — ONE command (detectors → metrics → plots, in series)

```bash
cd EXP
sudo nohup python3 run_batch_offline_eval.py \
    --captures-dir /home/bqn82/captures \
    --run-id sweep_$(date +%Y%m%d) \
    --repack-masks \
    --progress-every 25 \
    > /tmp/sweep.log 2>&1 &
tail -f /tmp/sweep.log
```
- Processes **14 captures × 2 detectors** (~335 frames each) under the synchronous GreedyScheduler.
- **Resumable + self-validating:** re-run the same command to resume (skips finished jobs); each job
  runs the alignment check and the summary counts any `misaligned` runs.
- After the detectors finish it **auto-runs** `eval_detector_masks.py` then `plot_eval_results.py`
  (add `--no-post` to skip; `--det-threshold 0.1` tunes the plot detection cutoff).
- Log the run to `/tmp/sweep.log` (writable); the state dir is under `batch_runs/` which you *cannot*
  redirect a shell `>` into as a non-root user before `sudo` — hence `/tmp`.

Everything lands in **`EXP/batch_runs/<run_id>/`**:
```
batch_runs/<run_id>/
  <detector>/<file_stem>/   mask_arrays/ gt_masks/ gt_annotations/ frame_manifest.csv offline_eval_summary.json
  region_metrics.csv        # per-annotation: coverage, box_iou, detected, class/bw/length/power
  frame_pixel_metrics.csv   # per-frame: precision/recall/F1/IoU/false-positive-area
  eval_summary.json
  plots/*.png               # the six comparison figures
  batch_state.json
```
`batch_runs/` is git-ignored.

## 4. Visualize in the notebook

The sweep ends by printing a **"VISUALIZE IN THE NOTEBOOK"** block with the exact edits. Open
`EXP/batch_eval_review.ipynb`, **Restart Kernel**, edit the two cells it names, then **Run All**:

- **Parameters cell (first code cell):**
  ```python
  BATCH_ROOT = Path('.../batch_runs/<run_id>')
  FILE_STEM  = 'attenuation_dB_25'   # any capture stem present
  ```
- **"Aggregate performance plots" cell (near the bottom):**
  ```python
  TABLES_DIR    = Path('.../batch_runs/<run_id>')
  DET_THRESHOLD = 0.1
  ```

The notebook then renders:
- per-frame 3-panel review (spectrogram + GT boxes | coherent_power mask | cuda_dino mask), including
  a reproducible random sample of annotated + noise-only frames;
- the six aggregate figures (also saved as PNGs in `plots/`):
  detection rate **vs power** faceted by signal class / bandwidth / pulse length; detection rate /
  IoU / coverage **vs bandwidth** and **vs pulse length**; and frame-level precision / recall / F1 /
  pixel-IoU / **false-positive area vs power**.

CLI equivalents (if you prefer not to use the notebook):
```bash
cd EXP
sudo python3 eval_detector_masks.py --batch-root batch_runs/<run_id> --captures-dir /home/bqn82/captures --out-dir batch_runs/<run_id>
python3 plot_eval_results.py --tables-dir batch_runs/<run_id> --det-threshold 0.1
python3 report_eval.py       --tables-dir batch_runs/<run_id>     # optional markdown summary
```

## 5. Component reference (all under EXP unless noted)

| File | Role |
| --- | --- |
| `APP/run_offline_cuda_detector_eval.cpp` | Offline binary: SigMF → FFT → spectrogram → selected detector → masks/GT/manifest. Detector-agnostic (registry). |
| `APP/run_cuda_dino_offline_file.py` | Per-file wrapper (docker-exec). `--detector`, `--no-tensors`, `--trace-frames`. |
| `run_batch_offline_eval.py` | Orchestrator: sweep → metrics → plots; resumable; per-job alignment check. |
| `mask_eval_metrics.py` | Metrics library (pixel P/R/F1/IoU + FP-area, per-region coverage/detection, breakdown attrs). |
| `eval_detector_masks.py` | Driver → tidy `region_metrics` + `frame_pixel_metrics` tables. |
| `plot_eval_results.py` | The six comparison figures. |
| `report_eval.py` | Stdlib markdown summary. |
| `eval_viz.py` + `batch_eval_review.ipynb` | Per-frame 3-panel review + notebook driver. |
| `check_mask_alignment.py` | Frame↔mask alignment gate (PASS = offset k≈0). |

## 6. Interpretation caveats (important)

- **GT masks are filled bounding boxes.** A detector that fires only on signal energy scores high
  pixel-*precision* but low pixel-*recall*, and a strict coverage threshold reads few detections. Use
  **region detection-rate + coverage** with a tuned `--det-threshold` (0.05–0.3) for "did it find the
  signal", and pixel-precision for "are detections clean".
- **False-positive area is a per-frame (whole-grid) metric** → plotted **vs power** only; it is not
  attributable to a single signal's bandwidth/duration.
- **ZC / METADATA** annotations have no `occupied_bw_hz`, so they don't appear in the bandwidth cuts.
- **Frame↔mask alignment must be k = 0** (see §2). Any non-zero offset = stale/buggy binary; rebuild.

## 7. Adding a new detector (modularity)

1. Link its operator lib in `APP/CMakeLists.txt` (the `run_offline_cuda_detector_eval` target).
2. Add one entry to `detector_adapter_table()` in `run_offline_cuda_detector_eval.cpp`.
3. Add its base config to `DETECTOR_BASE_CONFIGS` in `run_cuda_dino_offline_file.py`.
Everything downstream (orchestrator, metrics, plots, notebook) is detector-agnostic.

## 8. Troubleshooting

- **`bash: sweep.log: Permission denied`** — the shell `>` runs before `sudo`; log to `/tmp` (as above).
- **Alignment `FAIL` (offset ≠ 0)** — stale binary; rebuild (§1) and re-run that file.
- **`double_buffer_receiver Push failed` / `greedy_scheduler … deadlock` / `Unprocessed … in queue`** —
  benign GreedyScheduler drain/shutdown noise; each job still writes its manifest.
- **Looks "hung" after the summary prints** — the batch is done (state saved); it's usually `tail -f`
  that doesn't exit. Ctrl-C after the summary is harmless.
- **Coherent power drops a few tail frames** — expected (pipeline drain); the manifest is kept and
  metrics reconstruct/ignore missing-mask frames.

## 9. What the current results show (dB sweep, both detectors)

- `cuda_dino` leads on detection/recall at high SNR (~0.9–1.0 vs coherent ~0.6–0.7), crossing near
  dB_45–55 where coherent sometimes edges ahead. ZC/METADATA ~1.0 for both.
- `coherent_power` is the "clean" detector: higher precision and far lower false-positive area
  (~0.001 by dB_25) vs `cuda_dino` flooding noise (~10× higher FP, ~0.009 at dB_60).
- Detection falls with attenuation and with bandwidth (wider signals harder to fully cover).

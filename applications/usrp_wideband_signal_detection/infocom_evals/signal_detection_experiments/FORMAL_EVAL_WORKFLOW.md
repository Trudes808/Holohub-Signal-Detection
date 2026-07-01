# Formal detector evaluation workflow (offline, mask-vs-SigMF-GT)

Step 1 of the evaluation program: run every capture through every detector offline,
save per-frame masks + ground-truth masks, and score detectors against the SigMF
annotation ground truth, broken down by signal type / bandwidth / pulse length /
power level. Built to be modular for new detectors and future studies.

See the full design in `REPLAY_INGEST_PLAN.md` and `/home/sat3737/.claude/plans/luminous-finding-graham.md`.

## Components

| File | Role |
| --- | --- |
| `../../run_offline_cuda_detector_eval.cpp` | Offline binary. SigMF → FFT → Spectrogram → **selected detector** → masks/GT/manifest. Now detector-agnostic via a small registry (`cuda_dino`, `coherent_power`). |
| `../../run_cuda_dino_offline_file.py` | Per-file wrapper (docker-exec). New flags: `--detector`, `--no-tensors`, `--trace-frames`. Picks the base config per detector. |
| `run_batch_offline_eval.py` | Orchestrator: {captures} × {detectors}, GPU-serialized, resumable, lazy staging cleanup, optional `--repack-masks`. |
| `mask_eval_metrics.py` | Metrics library: pixel P/R/F1/IoU + FP-area, per-annotation coverage/detection, breakdown attrs re-joined from source `.sigmf-meta`, `BUCKETERS` registry. CSV (+Parquet if pandas) fact tables. |
| `eval_detector_masks.py` | Driver: walk a batch tree → combined `frame_pixel_metrics` + `region_metrics` tables. |
| `report_eval.py` | Stdlib Markdown report: pixel summary + detection-rate tables per breakdown. |
| `plot_eval_results.py` | Comparison figures from the tidy tables: perf vs power (faceted by class/bandwidth/pulse-length), perf vs bandwidth, perf vs pulse length, and frame-level precision/recall/F1/pixel-IoU + false-positive area vs power. `--tables-dir <dir> [--det-threshold 0.1]`. Stdlib+matplotlib. |
| `check_mask_alignment.py` | Frame↔mask alignment gate (systematic offset via median column-profile correlation, margin-gated). PASS=k≈0. |
| `eval_viz.py` | Visualization helpers: reconstruct the spectrogram from SigMF (matches the binary's saved tensor to ~1e-5 dB), load a frame bundle (spectrogram + GT + each detector's mask), render the (N+1)-panel comparison. |
| `batch_eval_review.ipynb` | Thin notebook driver: point at a batch run, pick a file + frame, render `[GT | detector_1 | … | detector_N]` panels, show per-frame metrics. |

## Adding a new detector (modularity)

1. Link its operator lib in `CMakeLists.txt` (`run_offline_cuda_detector_eval` target).
2. Add one entry to `detector_adapter_table()` in `run_offline_cuda_detector_eval.cpp`.
3. Ensure its config block exists; add it to `DETECTOR_BASE_CONFIGS` in the wrapper.
Everything downstream (orchestrator, metrics, report) is detector-agnostic.

## Run procedure

### 0. Build (needs the container + sudo)
```
cd applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh        # rebuilds run_offline_cuda_detector_eval with both detectors
```

### 1. Verify one capture per detector (do this before the full sweep)
```
cd applications/usrp_wideband_signal_detection
python3 run_cuda_dino_offline_file.py /home/bqn82/captures/attenuation_dB_25.sigmf-data \
    --detector cuda_dino      --no-tensors --trace-frames --progress-every 25
python3 run_cuda_dino_offline_file.py /home/bqn82/captures/attenuation_dB_25.sigmf-data \
    --detector coherent_power --no-tensors --trace-frames --progress-every 25
```
Confirm each `offline_eval_summary.json` has `manifest_complete: true` and that
`mask_arrays/`, `gt_masks/`, `gt_annotations/`, `frame_manifest.csv` are populated.
(The strict coverage check fails the run if a detector does not emit one mask/frame;
the `--trace-frames` log shows per-frame mask non-zero counts for debugging.)

### 2. Full sweep (15 captures × 2 detectors)
```
cd applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments
python3 run_batch_offline_eval.py \
    --captures-dir /home/bqn82/captures \
    --run-id sweep_$(date +%Y%m%d) \
    --progress-every 25 \
    --repack-masks            # packbits-compress masks after each job (~8x disk)
```
Resumable: re-running skips jobs whose `manifest_complete` is already true.

The sweep **auto-chains** metrics + plots after the detectors finish (unless `--no-post`), all into
`batch_runs/<run_id>/` (masks, `region_metrics.csv`/`frame_pixel_metrics.csv`, `plots/`, `batch_state.json`).
It ends by printing exactly which `batch_eval_review.ipynb` cells to edit (BATCH_ROOT / FILE_STEM /
TABLES_DIR) so all visualization lives in the notebook. `--det-threshold` tunes the plot detection cutoff.

### 3. Metrics + report (only needed if you used `--no-post`, or want the markdown report)
```
python3 eval_detector_masks.py \
    --batch-root /tmp/usrp_spectrograms/batch_eval/<run_id> \
    --captures-dir /home/bqn82/captures \
    --out-dir batch_runs/<run_id>
python3 report_eval.py --tables-dir batch_runs/<run_id>
```

## Partial-coverage runs (tail-frame drain)

Some detectors (e.g. `coherent_power`) drop a few **tail frames** to pipeline drain, so they
emit < total_frames masks. Behavior:
* The binary keeps the manifest/summary (coverage shortfall is a warning;
  `manifest_complete: false`). Set `offline_eval.require_full_mask_coverage: true` to hard-fail instead.
* If a run's manifest was deleted by an older binary, the metrics/viz layer **reconstructs** it
  from the surviving `gt_annotations/*.json` automatically — no re-run needed.
* Maskless frames are recorded but **excluded from aggregates** (not scored as misses), so the
  drain artifact doesn't bias the detector comparison.
* `eval_viz.resolve_layout()` accepts a batch root, a detector dir, or a single run dir.

## Mask/spectrogram alignment (must rebuild before trusting multi-frame masks)

The offline binary runs under the **synchronous `GreedyScheduler`**. The earlier multi-threaded
`EventBasedScheduler` + reusable device-tensor ring let a source slot be overwritten while a detector
kernel on another CUDA stream still read it, producing **time-misaligned masks on full-file runs**
(GT + spectrogram stay aligned; only the masks shift, by a frame-varying amount). Any multi-frame run
produced before this fix must be re-run after `./rebuild_demo_container_app.sh`.

Root cause (found by a full multi-agent trace + empirics): the offline source emitted a live handle
into a **reused device-tensor ring** and overwrote a slot `ring_size` frames before the detector read
it, so mask "frame N" held frame **N+ring_size**'s samples (offset = ring depth = 8; identical for both
detectors; GT/metadata stayed correct). Fixed by allocating a **fresh per-frame device tensor** in
`OfflineSc16FileSourceOp` (matx tensors are reference-counted → freed after the sink drops the message),
so no memory is shared between frames; VRAM scales with in-flight frames, not total. `ring_size_` is now
ignored. Requires a rebuild; runs made before it are corrupt.

One-command alignment self-check (run after re-running):
```
python3 check_mask_alignment.py --batch-root <output-root> --file-stem attenuation_dB_25
```
It scans frame offsets `k` and reports the systematic offset via median column-profile correlation
(robust to the capture's periodicity). PASS = best offset `k=0`; a non-zero `k` (e.g. +8) means the
binary is stale. This is also run automatically per job by `run_batch_offline_eval.py` (warns + counts
`misaligned` in the final summary).

## Interpretation caveats (IMPORTANT)

* **GT masks are filled bounding boxes.** The binary rasterises each annotation's full
  freq×time rectangle as "signal". A detector that fires only on signal energy (sparse)
  scores high pixel-*precision* but low pixel-*recall* against a filled box, and a 0.5
  region-coverage threshold may report 0 detections. Tune `--coverage-threshold` (e.g.
  0.05–0.2) and lean on **region detection rate + mean coverage** for "did it find the
  signal", and pixel precision for "are detections clean". (Observed on the dB_0 DINO
  baseline: precision 0.93, recall 0.018.)
* **Offline FFT geometry is derived from the SigMF sample rate** (245.76 MSps →
  10240-pt FFT, 24 kHz bins, 512 time rows, samples_per_frame 5,242,880, ~332 frames/file).
  Both detectors share this grid, so the comparison is apples-to-apples. This differs
  from the forced 20480-pt live-replay config; reconciling offline vs live framing is
  part of the deferred replay-refactor task, not this accuracy eval.

## Future studies (schema already supports)

* **New detector** → registry entry (above).
* **Ablations** → add a `variant` column to the fact tables; orchestrator parameterises
  config overrides per job.
* **Congestion / storage** → fraction of spectrum passed by the mask per frame, derivable
  from the retained (packed) masks.
* **ML training/inference** → consume the retained masks directly; no harness change.

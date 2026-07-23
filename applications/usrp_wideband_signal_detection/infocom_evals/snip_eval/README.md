# snip_eval — real signal_snipper footprint (offline data-saving eval)

Measures the **real** per-detector storage footprint of mask-driven collection by replaying each
detector's precomputed masks through the actual C++ `mask_replay_detector` → `signal_snipper` →
`sigmf_file_sink` pipeline — the same operators the live app uses. Two snipper modes are measured:

- **`frequency`** — resample + filter each detected box to its bandwidth (mix → lowpass at
  `oversample_percent` → integer decimate). This is the "snip" strategy.
- **`time_only`** — keep the full-band, full-rate time regions that contain any detection. This is
  the "time-slice" strategy.

**Footprint-only:** the config sets `sigmf_file_sink.write_iq: false` and the run passes
`--snippets-only`, so **no IQ is stored** — each snippet writes only a tiny `.sigmf-meta` carrying the
decimated + full-rate `core:sample_count` and the decimation factor. Bytes are reconstructed exactly
from those counts. Speckle is gated by `signal_snipper.min_box_pixels` (256) so only real boxes count.

## Pipeline
```
baseline_comparisons masks (8 detectors + ground_truth)
        │  stage_snip_masks.py
        ▼
snip_run/<detector>/<stem>/mask_arrays/   (.packed.npz symlinks + packed baselines + frame_manifest)
        │  launch_snip.sh → run_snip_all.sh   (per detector × capture × mode)
        │    • materialize_npy.py     .packed.npz → .npy   (once per capture, transient)
        │    • run_cuda_dino_offline_file.py --detector mask_replay --snippets-only
        │        → mask_replay_detector → signal_snipper → sigmf_file_sink (write_iq=false)
        │    • snip_annotations.py    → waveform-detection .sigmf-meta (cross-check / labels)
        ▼
<SNIP_OUT>/<mode>/<detector>/<stem>/snippets/*.sigmf-meta
        │  verify_snip.py
        ▼
real_snip_metrics.csv   →   plot_data_saving.py  (figures in figs/)
```

## Scripts
- **`find_masks.py [ROOT ...]`** — inventory masks (detector × capture, count, format, GT/manifest,
  complete?). Run before producing anything so finished detectors aren't recomputed.
- **`create_all_masks.sh`** — produce all detectors' masks into one batch root (wraps the baseline
  `run_full_comparison.py`), reusing existing C++ `coherent_power`/`cuda_dino` masks.
- **`stage_snip_masks.py`** — stage every detector's masks under `snip_run/<det>/<stem>/mask_arrays`
  (symlink `.packed.npz`, `np.packbits` the `.npy` baselines) + copy `frame_manifest.csv`.
- **`materialize_npy.py <ROOT>`** — unpack `.packed.npz` → `.npy` so the (`.npy`-only) mask_replay
  operator can read them. Called once per capture by the runner; the `.npy` are deleted after.
- **`run_snip_all.sh`** — loop detectors × captures × `MODES` in one invocation; footprint-only;
  **resumable** (skips a (mode,detector,capture) whose snippet metas exist) and **tolerant** (a failed
  run logs and continues). `set -euo pipefail`.
- **`launch_snip.sh`** — resilient detached wrapper: re-invokes `run_snip_all.sh` until a pass makes
  no new progress, so transient kills self-heal. Launch with
  `sudo nohup setsid ./launch_snip.sh > /tmp/snip_run.log 2>&1 &` (run `sudo -v` first to cache creds).
- **`snip_annotations.py --run-dir <root>/<det>/<stem>`** — cluster masks (snipper's
  `min_box_pixels`/`merge_gap` rule) → the overall waveform-detection `.sigmf-meta` (cross-check).
- **`verify_snip.py --snip-out <SNIP_OUT>`** — read all snippet metas → `real_snip_metrics.csv`
  (decimated + full-rate TB/hr, mean decimation factor, rate stats, pct full-rate) per mode/detector.
- **`snip_data_metrics.py`** — standalone *analytic* per-box footprint proxy (container-free) that also
  writes `<stem>_detected.sigmf-meta`; used by the notebook methodology as a cross-check of the real
  measurement. Not part of the live run.

## Run it (both modes, one command)
```
cd applications/usrp_wideband_signal_detection/infocom_evals/snip_eval
sudo -v                                  # cache sudo creds (no TTY in the detached run)
sudo nohup setsid env CONTAINER_NAME=usrp_x410_sig_det_<user> \
     ./launch_snip.sh > /tmp/snip_run.log 2>&1 &
tail -f /tmp/snip_run.log                 # Ctrl-C the tail anytime; the run keeps going
```
Needs the container with `mask_replay_detector` compiled in (a short `rebuild_demo_container_app.sh`)
and `~/captures` mounted read-only. Env knobs: `MODES` (default `frequency time_only`), `BATCH_ROOT`,
`SNIP_OUT`, `CAPTURES_DIR`, `DETECTORS`, `CONFIG`.

## Outputs / git
- **`real_snip_metrics.csv`** — the results (committed).
- `snip_run/`, `detected/`, `gt_snip_run/` — large staged masks / analytic metas; **gitignored**,
  regenerable from the scripts above.

## Notebook & figures
`data_saving_eval_review.ipynb` (+ jupytext `.py`) is the self-documenting analysis notebook — data
reduction vs save-all, the fidelity (signal-retention) trade-off, and per-detector compute cost — as
hand-calcs over the attenuation sweep on a physical **SNR axis**. Kernel: **yolo** (pure analysis;
loads no models). Baseline: 245.76 MHz `cf32` → **save-all = 7.08 TB/hour** (flat, SNR-independent).

- **`build_ds_cache.py` → `ds_cache.csv`** — precomputes per-detector reduction / retention / TF-coverage
  and the analytic snipper footprint once, so the notebook re-renders instantly (`DS_REBUILD=1` forces
  a rebuild; knobs `DS_NFRAMES`, `DS_MIN_BOX_PIXELS`, `DS_SWEEP`).
- **`plot_data_saving.py` → `figs/`** — the real-snip figures (Figs 1–3 × {all, curated}) from
  `real_snip_metrics.csv`: GB/hour (log, plain-number) vs SNR. Standalone, ~2 s.
- **`snr_calibration.json`** — `snr0_ref_db` for the attenuation→SNR mapping (`snr = snr0_ref − atten`).
- **compute** — `compute_table.csv` (via `yolo_training/src/measure_compute.py`): FLOPs + measured GPU
  memory + real-time factor per detector, for the compute figure.
- **live-OTA** — `live_data_saving.csv` (via `yolo_training/src/measure_live_saving.py`) feeds the
  live-OTA figure; `instructions.md` has the container replay that produces the OTA masks it reads.

### Regenerate (run from the repo root)
```bash
cd ~/Holohub-Signal-Detection
SE=applications/usrp_wideband_signal_detection/infocom_evals/snip_eval
conda activate dinov3 && python $SE/plot_data_saving.py                    # real-snip figures -> figs/
conda activate dinov3 && python yolo_training/src/measure_compute.py       # -> snip_eval/compute_table.csv
conda activate yolo   && jupyter nbconvert --to notebook --execute --inplace $SE/data_saving_eval_review.ipynb
```
Tracked: the notebook (+`.py`), the scripts, `figs/`, `ds_cache.csv`, `real_snip_metrics.csv`,
`snr_calibration.json`, `live_data_saving.csv`. Regenerable tables (`compute_table.csv`, `*_table.csv`)
are gitignored.

## Fine-tuned DINO weights (provenance)
The `finetuned_dino` (M1) and `finetuned_dino_m2` (M2) detector masks were produced upstream by the
fine-tuned DINOv3 TorchScript models in `dino_fine_tuning/weights/` (`finetuned_dino_m1.ts`,
`finetuned_dino_m2.ts`, ~335 MB each). Those weights are **gitignored** (669 MB; not referenced by any
runtime config — the live/offline DINO configs load the base backbone
`/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`). They are a
reproducibility input only, regenerable from `dino_fine_tuning/` training.

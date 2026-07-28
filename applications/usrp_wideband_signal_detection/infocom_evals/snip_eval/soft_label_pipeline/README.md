# soft_label_pipeline — waveforms → detector → soft labels

A self-contained pipeline that takes **any folder of SigMF waveforms** and a **detector**, runs the
detector to produce per-frame masks, and emits soft labels:

- **`<stem>_snipped.sigmf-meta`** — SigMF annotations = the detector's `detected_waveform` boxes mapped
  onto the **original capture timeline** (`core:sample_start/count`, `core:freq_lower/upper_edge`).
  Always produced.
- **decimated IQ pairs** — `snippets/*.sigmf-{data,meta}`, one recording per detection, each **cut in
  time and decimated in frequency** to its bandwidth (mix → low-pass → decimate). Optional.

`--outputs {snipped_meta | iq | both}` selects one, the other, or both.

## Quickstart
```bash
# from this folder, with the demo container running (see §1) and conda env `dinov3` active:
python soft_label_pipeline.py \
    --waveform-dir /home/bqn82/captures \
    --detector coherent_power \
    --outputs both \
    --output-root /tmp/usrp_spectrograms/soft_label_run
```
Outputs land under `--output-root` (keep it on `/tmp`, not `/home`). Process a subset with
`--glob "attenuation_dB_30.sigmf-data"` or `--limit N`.

## 1. Docker container setup (required)
Mask generation and the IQ snipper run inside the demo container (the pipeline calls
`run_cuda_dino_offline_file.py`, which does `sudo docker exec …`). From the app root
(`applications/usrp_wideband_signal_detection`):
```bash
# first-time container bring-up (image already built elsewhere):
sudo env SKIP_IMAGE_BUILD=1 ./bash_scripts/build_demo_container.sh
# after any code/config change to the app:
sudo ./bash_scripts/rebuild_demo_container_app.sh
```
- Container/image identity lives in `bash_scripts/container_env.sh`; override per run with
  `--container-name` (or `CONTAINER_NAME=…`). Default here: `usrp_x410_sig_det_bqn82`.
- The container must have the `mask_replay_detector` operator compiled in (a `rebuild_demo_container_app.sh`).
- `sudo docker` must be passwordless for the account (the wrappers rely on it); run the Python
  pipeline **without** `sudo`.
- **Mounts / paths:** the container maps `/home/bqn82/captures → /workspace/captures`,
  `/tmp/usrp_spectrograms → /workspace/spectrograms`, and the repo → `/workspace/holohub`. Your
  `--waveform-dir` and `--output-root` should sit under one of those host roots so files are read/written
  in place (no copy). A `--waveform-dir` elsewhere still works but the driver stages a scratch copy of
  each capture (slow for large files).

## 2. Input format
Each waveform is a **SigMF pair** in `--waveform-dir`:
- `<name>.sigmf-data` — interleaved complex IQ, **`cf32_le`** (complex64, 8 B/sample).
- `<name>.sigmf-meta` — JSON with at least `global.core:sample_rate`, `core:datatype: "cf32_le"`, and
  `captures[0].core:frequency` (RF center, used for absolute freq labels). Existing `annotations[]`
  (ground truth) are fine — they're left untouched. Captures without a matching `.sigmf-meta` are skipped.

Any sample rate works (the container reads `core:sample_rate` from the meta). The lab captures are
245.76 MSps; the OTA captures are 500 MSps.

## 3. Detectors & prerequisites
Choose one with `--detector`. Three tiers, increasing setup:

| tier | detectors | how masks are made | needs |
|---|---|---|---|
| **trained** | `coherent_power`, `cuda_dino` | container binary, directly from IQ | demo container |
| **baselines** | `3dB_power`, `blob_detection` | `run_baseline_offline.py`, mirroring a trained run | container (for the ref run) + numpy/scipy (CPU) |
| **ml** | `yolo`, `dino_finetuned`, `dino_finetuned_m1`, `yolo26s` | `run_ml_detectors_offline.py`, mirroring a trained run | container (ref) + **GPU** + conda `dinov3` (torch, ultralytics) + model weights + dinov3 repo |

**Dependency chain:** baselines and ML detectors do *not* process raw IQ on their own — they reuse a
trained detector's frame grid, ground truth, and `frame_manifest.csv`. So when you pick a baseline/ML
detector, the pipeline first **seeds the reference trained detector** (`--ref-detector`, default
`cuda_dino`) on every capture, then runs the chosen one. That means even a `blob_detection` run needs
the container.

**ML weights** are configured in `../../baseline_comparisons/comparison_config.yaml` (`ml_detectors:`)
— `dino_finetuned` = M2, `dino_finetuned_m1` = M1, `yolo` = YOLO26-m, `yolo26s` = YOLO26-s. Point at
your dinov3 repo with `--dinov3-repo` (default from the config). These are gitignored, GPU-resident.

## 4. Usage examples
```bash
# trained detector, both outputs, one capture:
python soft_label_pipeline.py --waveform-dir /home/bqn82/captures \
    --glob "attenuation_dB_30.sigmf-data" --detector coherent_power --outputs both

# fine-tuned DINO (M2), snipped meta only, whole folder (needs GPU + weights):
python soft_label_pipeline.py --waveform-dir /home/bqn82/captures \
    --detector dino_finetuned --outputs snipped_meta

# reuse pre-computed masks (skip generation), IQ pairs only:
python soft_label_pipeline.py --waveform-dir /home/bqn82/captures \
    --glob "attenuation_dB_45.sigmf-data" --detector coherent_power \
    --mask-root /tmp/usrp_spectrograms/all_detectors --outputs iq

# tighter gate (drop tiny detections) via the snipper selectivity flags:
python soft_label_pipeline.py --waveform-dir DIR --detector cuda_dino --outputs both \
    --min-box-pixels 256 --min-bandwidth-hz 75000 --min-duration-s 0.001 --min-mask-bandwidth-hz 75000
```

## 5. Outputs (under `--output-root`)
```
masks/<detector>/<stem>/         mask_arrays/ + frame_manifest.csv + gt_masks + gt_annotations
snipped/<stem>_snipped.sigmf-meta   detected_waveform annotations on the original timeline
iq/<stem>/snippets/*.sigmf-{data,meta}   decimated IQ per detection   (--outputs iq|both)
config_snip_generated.yaml       the mask-replay/snipper config the IQ step used
```

## 6. Key options
- `--outputs {snipped_meta,iq,both}` · `--glob PATTERN` · `--limit N`
- `--mask-root DIR` — reuse existing `<DIR>/<detector>/<stem>/mask_arrays` instead of generating.
- selectivity: `--mode {frequency,time_only}`, `--min-box-pixels`, `--min-bandwidth-hz`,
  `--min-duration-s`, `--min-mask-bandwidth-hz` (snipper); `--merge-gap-rows/-cols` (annotations).
- container/wiring: `--container-name`, `--captures-root`, `--ref-detector`, `--detector-config`,
  `--config` (baseline/ML params), `--dinov3-repo`.

Note: the `<stem>_snipped.sigmf-meta` uses the size+merge clustering (`min_box_pixels` + merge gaps);
the IQ snipper additionally applies the bandwidth/duration gates. At the default (256 px, no bw/dur
gate) the two agree one-to-one.

## 7. Files here
- `soft_label_pipeline.py` — the orchestrator (this pipeline).
- `snip_annotations.py` — clusters masks → `<stem>_snipped.sigmf-meta` (copied from `snip_eval/`).
- `materialize_npy.py` — unpacks `.packed.npz` masks → `.npy` for the replay snipper (copied).

The heavy detector-generation tools live in the app (`run_cuda_dino_offline_file.py`,
`baseline_comparisons/run_{baseline,ml_detectors}_offline.py`) and are called in place.

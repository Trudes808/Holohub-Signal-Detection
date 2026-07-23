# snip_eval — feed detector masks into the signal_snipper (offline data-saving)

Turns the 8-detector masks produced by the `baseline_comparisons` pipeline into snipped SigMF +
annotations, via the C++ `mask_replay_detector` → `signal_snipper`. All offline.

## Pipeline
```
baseline_comparisons  ->  <batch_root>/<detector>/<stem>/mask_arrays/   (8 detectors, shared grid)
        |                                   |
   create_all_masks.sh                 mask_replay_detector (C++)  ->  signal_snipper  ->  sigmf_file_sink
   find_masks.py                                                              |
                                                                    <detector>_snipped/<stem>_snipped/
                                                                        N.sigmf-data / N.sigmf-meta  (per snippet)
   snip_annotations.py  ->  <stem>_snipped.sigmf-meta  (overall detections, labelled)
   plot_data_saving.py  ->  figures
```

## Scripts
- **`find_masks.py [ROOT ...]`** — inventory existing masks (detector × capture, count, format,
  GT/manifest, complete?). Run before producing anything so finished detectors aren't recomputed.
- **`create_all_masks.sh`** — produce all 8 detectors into one batch root, reusing existing. Wraps the
  collaborator's `run_full_comparison.py` (baselines + ML stages) over a root that already holds the
  C++ `coherent_power`/`cuda_dino` masks. Env knobs: `BATCH_ROOT`, `STAGES`, `SNIP_ENV`, `CONFIG`.
- **`materialize_npy.py <ROOT>`** — unpack `.packed.npz` masks -> `.npy` so the (`.npy`-only) mask_replay operator can read them. Run once over the batch root before the snip stage.
- **`snip_annotations.py --run-dir <batch_root>/<det>/<stem>`** — cluster masks (snipper's
  `min_box_pixels`/`merge_gap` rule) → the overall `<stem>_snipped.sigmf-meta` detections file.
- **`plot_data_saving.py`** — (pending) read the snipped artifacts + a summary → data-saving figures.

## The snipper stage (needs the C++ operator)
`mask_replay_detector` (in `operators/mask_replay_detector/`, drafted for lab-admin to compile) is a
drop-in "detector" that reads a detector's precomputed masks and emits them as `DetectorMaskMessage`,
so the existing `signal_snipper` + `sigmf_file_sink` produce real snipped SigMF. First `python3 materialize_npy.py <batch_root>` (op reads .npy). Then per detector:
```
sudo env CONTAINER_NAME=... python3 ../signal_detection_experiments/run_cuda_dino_offline_file.py \
    <capture> --detector mask_replay --config ../../config_mask_replay_snip_single_channel.yaml \
    --output-root <out>/<detector>/<stem>          # mask_replay_detector.mask_dir -> <batch_root>/<detector>/<stem>/mask_arrays
```
(a small `run_snip_all.sh` wrapper to loop detectors×captures comes with the operator.)

## Status
- `find_masks.py`, `snip_annotations.py` — working (tested on existing coherent/cuda_dino masks).
- `create_all_masks.sh` — ready (wraps run_full_comparison).
- `comparison_config.yaml` — extended with `dino_finetuned_m1` + `yolo26s` (8 detectors total).
- `mask_replay_detector` + its config — drafted; needs a container compile (lab-admin/collaborator).
- `plot_data_saving.py` + the per-snippet artifact reorg — after the snipper runs (they consume its output).
- Validate once: `mask_replay` on `cuda_dino`'s masks == a direct `cuda_dino`+snipper run.

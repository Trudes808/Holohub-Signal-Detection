# Eight-detector batch-eval — how to generate masks and run the notebook

This **augments** the six-detector setup (`../SIX_DETECTOR_WORKFLOW.md`) with the two
**tuned** classical baselines. The six existing detectors are reused byte-for-byte, so
every previously-reported number is unchanged; only two columns/lines are added.

| # | detector | family | input | how masks are made |
|---|----------|--------|-------|--------------------|
| 1 | `coherent_power`        | traditional        | spectrogram | container batch (already in the sweep) |
| 2 | `power_detection`       | traditional        | **raw IQ** (own FFT) | container batch (already in the sweep) |
| 3 | `power_detection_tuned` | traditional, tuned | **raw IQ** (own FFT) | **container batch (NEW — you generate)** |
| 4 | `computer_vision`       | traditional        | spectrogram | container batch (already in the sweep) |
| 5 | `computer_vision_tuned` | traditional, tuned | spectrogram | **container batch (NEW — you generate)** |
| 6 | `cuda_dino`             | learned            | spectrogram | container batch (already in the sweep) |
| 7 | `finetuned_dino`        | learned (M1: ≤30 dB) | spectrogram | offline, `src/gen_finetuned_run.py` (already materialized) |
| 8 | `finetuned_dino_m2`     | learned (M2: all dB) | spectrogram | offline, `src/gen_finetuned_run.py` (already materialized) |

The two tuned detectors' masks are produced by the **same offline batch pipeline** that
made the other container masks, so the notebook and metrics are identical (same frames,
buckets, `coverage-threshold 0.1`, `eval_detector_masks.py`).

Everything is **additive**: new detector dirs land next to the existing ones; nothing
existing is moved, overwritten, or deleted. The assembler and notebook gracefully skip
the tuned detectors until their sweep dirs exist.

---

## Prerequisite — the tuned operators must be in the checkout you build

The tuned operators (`operators/power_detection_tuned/`, `operators/computer_vision_tuned/`),
their wiring (`main.cpp`, `run_offline_cuda_detector_eval.cpp`, both `CMakeLists.txt`,
`run_cuda_dino_offline_file.py`), and the two configs live on branch
**`baseline_tuned_detectors`**. Make sure they are in your working checkout before building —
e.g. from the repo root, on `baseline_dev_branch`:

```bash
cd ~/Holohub-Signal-Detection
git merge baseline_tuned_detectors      # fast-forward; leaves your dino_fine_tuning/ edits untouched
```

They are already wired into the offline eval binary and its Python driver
(`DETECTOR_BASE_CONFIGS` has `power_detection_tuned` / `computer_vision_tuned`), so after the
rebuild the batch driver knows them by name — nothing else to hand-edit.

---

## Step 1 — rebuild the container (needs `sudo docker`, run in your shell)

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
FORCE_REBUILD=1 ./rebuild_demo_container_app.sh
```

(First time only: `./build_demo_container.sh`.) This compiles the two new operator libs and
syncs the two top-level `config_*_tuned_single_channel.yaml` files into the build tree.

## Step 2 — generate masks for the two tuned detectors over the SAME sweep

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments
python3 run_batch_offline_eval.py \
    --captures-dir ~/captures \
    --detectors power_detection_tuned computer_vision_tuned \
    --run-id sweep_20260630 \
    --output-root /tmp/usrp_spectrograms/batch_eval/sweep_20260630 \
    --repack-masks --no-post
```

- Writes `.../sweep_20260630/power_detection_tuned/<stem>/` and
  `.../sweep_20260630/computer_vision_tuned/<stem>/` next to the existing detector dirs.
- `--no-post` skips the standalone plot pipeline (we visualize in the notebook instead).
- Must run where `sudo docker` works (your shell, not a sandbox). Resumable.

Sanity check (should now list 6 container detector dirs):
```bash
ls /tmp/usrp_spectrograms/batch_eval/sweep_20260630/
```

## Step 3 — assemble the 8-detector root + regenerate the canonical tables

```bash
cd ~/Holohub-Signal-Detection/dino_fine_tuning
~/miniforge3/envs/dinov3/bin/python src/assemble_eight_detectors.py
```

Symlinks the 6 container/materialized detectors + the 2 tuned container dirs into
`notebooks/eight_detectors/sweep_detectors_eight/`, then runs
`eval_detector_masks.py --coverage-threshold 0.1` into
`notebooks/eight_detectors/compare_tables_eight/`. (Run it with the `dinov3` env python —
the base env has no numpy.)

## Step 4 — run the notebook

Open `notebooks/eight_detectors/batch_eval_review_eight_detectors.ipynb`, kernel
**Python (dinov3)**, Restart & Run All. Sections 1–2 use the same frames/examples as the
six-detector notebook (`attenuation_dB_45`, frame 100; review seed 7); sections 3–6
auto-include every detector present in the tables. Section 7 is new: a paired
tuned-vs-naive detection-rate delta by attenuation.

---

## Notes

- **Runs before the tuned masks exist too.** `assemble_eight_detectors.py` and the notebook
  gracefully skip `power_detection_tuned` / `computer_vision_tuned` if their dirs aren't in
  the sweep yet — you'll see the six current detectors until you complete Steps 1–3.
- **The six original detectors are reused verbatim** (container dirs + byte-identical
  materialized fine-tuned dirs from `notebooks/sweep_detectors/`), so all six-detector
  numbers carry over unchanged.
- If the fine-tuned dirs are ever missing, rebuild them as in `../SIX_DETECTOR_WORKFLOW.md`:
  ```bash
  ~/miniforge3/envs/dinov3/bin/python src/gen_finetuned_run.py     # M1 -> finetuned_dino
  ~/miniforge3/envs/dinov3/bin/python src/gen_finetuned_run.py \
      --ft-ckpt checkpoints/M2_ft/best.pt --detector-name finetuned_dino_m2 \
      --ft-eval-meta eval_out/M2_ft/eval_meta.json                 # M2 -> finetuned_dino_m2
  ```

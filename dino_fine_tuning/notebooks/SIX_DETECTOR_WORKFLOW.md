# Six-detector batch-eval — how to generate masks and run the notebook

This extends the 3/4-detector setup (`batch_eval_review_three_detectors.ipynb`) with the
two **new traditional baselines**:

| detector | family | input | how masks are made |
|----------|--------|-------|--------------------|
| `coherent_power`    | traditional | spectrogram | container batch (already in the sweep) |
| `cuda_dino`         | learned     | spectrogram | container batch (already in the sweep) |
| `computer_vision`   | traditional | spectrogram | **container batch (NEW — you generate)** |
| `power_detection`   | traditional | **raw IQ** (own FFT) | **container batch (NEW — you generate)** |
| `finetuned_dino`    | learned (M1: ≤30 dB) | spectrogram | offline, `src/gen_finetuned_run.py` (already materialized) |
| `finetuned_dino_m2` | learned (M2: all dB) | spectrogram | offline, `src/gen_finetuned_run.py` (already materialized) |

The two new detectors' masks are produced by the **same offline batch pipeline** that
made the `coherent_power` / `cuda_dino` masks — the notebook and metrics are therefore
identical (same frames, buckets, `coverage-threshold 0.1`, `eval_detector_masks.py`).

Everything is **additive**: new detector dirs land next to the existing ones in the sweep;
nothing existing is moved, overwritten, or deleted.

---

## One-time: the operators are already wired into the offline eval

The offline eval binary (`run_offline_cuda_detector_eval.cpp`) and its Python driver
(`run_cuda_dino_offline_file.py` → `DETECTOR_BASE_CONFIGS`) already know the two new
detectors:
- `computer_vision` taps the spectrogram (like DINO / coherent power).
- `power_detection` taps the **raw-IQ source** and runs its own FFT (`consumes_raw_iq`),
  mirroring the live `main.cpp` wiring; the source fans out to both the FFT chain and
  the detector.

So the only thing you need to do is **rebuild the container** so the binary picks up the
new operator libs, then run the batch.

---

## Step 1 — rebuild the container (needs `sudo docker`, run in your shell)

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh          # or: FORCE_REBUILD=1 ./rebuild_demo_container_app.sh
```

(First time only: `./build_demo_container.sh`.)

## Step 2 — generate masks for the two new detectors over the SAME sweep

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments
python3 run_batch_offline_eval.py \
    --captures-dir ~/captures \
    --detectors computer_vision power_detection \
    --run-id sweep_20260630 \
    --output-root /tmp/usrp_spectrograms/batch_eval/sweep_20260630 \
    --repack-masks --no-post
```

- Writes `.../sweep_20260630/computer_vision/<stem>/` and `.../power_detection/<stem>/`
  right next to the existing `coherent_power/` and `cuda_dino/` dirs.
- `--no-post` skips the standalone plot pipeline (we visualize in the notebook instead).
- Must run where `sudo docker` works (your shell, not a sandbox).
- Resumable: re-running skips already-complete `<detector>/<stem>` runs.

Sanity check (should list 4 detector dirs):
```bash
ls /tmp/usrp_spectrograms/batch_eval/sweep_20260630/
```

## Step 3 — assemble the 6-detector root + regenerate the canonical tables

```bash
cd ~/Holohub-Signal-Detection/dino_fine_tuning
~/miniforge3/envs/dinov3/bin/python src/assemble_six_detectors.py
```

This symlinks the 4 container detectors (from the sweep) + the 2 materialized fine-tuned
dirs (from `notebooks/sweep_detectors/`) into `notebooks/sweep_detectors_six/`, then runs
`eval_detector_masks.py --coverage-threshold 0.1` into `notebooks/compare_tables_six/`.
(Run it with the `dinov3` env python — the base env has no numpy.)

## Step 4 — run the notebook

Open `notebooks/batch_eval_review_six_detectors.ipynb`, kernel **Python (dinov3)**,
Restart & Run All. Sections 1–2 are the same frames/examples as the 3/4-detector notebook
(`attenuation_dB_45`, frame 100; review seed 7); sections 3–6 auto-include every detector
present in the tables.

---

## Notes

- **Runs before the new masks exist too.** `assemble_six_detectors.py` and the notebook
  gracefully skip `power_detection` / `computer_vision` if their dirs aren't in the sweep
  yet — you'll just see the 4 current detectors until you complete Steps 1–3.
- **Fine-tuned masks are reused byte-for-byte** from `notebooks/sweep_detectors/` so the
  fine-tuned numbers match the 3/4-detector notebook exactly.
- If the fine-tuned dirs are ever missing, rebuild them:
  ```bash
  ~/miniforge3/envs/dinov3/bin/python src/gen_finetuned_run.py     # M1 -> finetuned_dino
  ~/miniforge3/envs/dinov3/bin/python src/gen_finetuned_run.py \
      --ft-ckpt checkpoints/M2_ft/best.pt --detector-name finetuned_dino_m2 \
      --ft-eval-meta eval_out/M2_ft/eval_meta.json                 # M2 -> finetuned_dino_m2
  ```

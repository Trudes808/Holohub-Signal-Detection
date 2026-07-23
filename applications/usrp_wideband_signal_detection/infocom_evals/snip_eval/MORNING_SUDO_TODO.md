# Morning TODO — snip stage for 65/70 dB (needs sudo + container)

Written overnight by Claude. The mask-generation for 65 & 70 dB is **complete** for all
detectors (no sudo was needed). What remains needs the demo container, hence sudo.

## What got done overnight (no sudo)
65/70 dB masks (328 frames each) produced into their canonical locations, so `/tmp/ds_batch`
now shows a full **0–70 dB** sweep for all 6 comparison detectors:

| Detector | Producer | Location |
|---|---|---|
| coherent_power, cuda_dino | (already done earlier, container) | `/tmp/usrp_spectrograms/all_detectors` |
| 3dB_power, blob_detection | `run_baseline_offline.py` | `/tmp/ds_batch` |
| yolo26s, yolo26m | `yolo_training/src/gen_yolo_run.py` | `notebooks/yolo_evals/sweeps/sweep_all` |
| finetuned_dino (M1), finetuned_dino_m2 (M2) | `dino_fine_tuning/src/gen_finetuned_run.py` | `notebooks/dino_fine_tuning_evals/sweeps/sweep_detectors` |

Verify anytime: each `/tmp/ds_batch/<det>/attenuation_dB_{65,70}/mask_arrays` has 328 files.

## Step 1 — compile the mask_replay_detector into the container (sudo)
The operator source (`operators/mask_replay_detector/`) exists and is registered in
`operators/CMakeLists.txt`, but is **not yet compiled** into the container.

```bash
cd /home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/rebuild_demo_container_app.sh
```

## Step 2 — run the snipper over ALL masks (sudo + container)
`run_snip_all.sh` loops over every `<detector>/<stem>/mask_arrays` under `BATCH_ROOT`. Point it at
`ds_batch` and list all 8 detectors explicitly (so the glob doesn't treat `data_saving_figs/` etc.
as a detector). This snips **every detector × every attenuation (0–70, 16 captures each)** — all
possible masks:
```bash
cd .../infocom_evals/snip_eval
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 \
     BATCH_ROOT=/tmp/ds_batch \
     CAPTURES_DIR=/home/bqn82/captures \
     DETECTORS="coherent_power cuda_dino 3dB_power blob_detection yolo26s yolo26m finetuned_dino finetuned_dino_m2" \
     ./run_snip_all.sh
```
Output → `/tmp/usrp_spectrograms/snipped/<detector>/<stem>/` + `<stem>_snipped.sigmf-meta`.

The script **auto-skips** any run dir with no matching capture in `CAPTURES_DIR` (it prints
`skip ... -- stale dir?`), so the leftover `cuda_dino/*.stale_1` dirs are ignored either way — the
stale cleanup below is optional and independent of the snip.

## Step 3 — repack dB_70 masks to `.packed.npz` (sudo) — REQUIRED for the notebook
`attenuation_dB_70` for `coherent_power` + `cuda_dino` was written as unpacked `.npy` (the resumed
run skipped the repack pass that 0–65 got). The data-saving notebook globs `*.packed.npz`, so dB_70
comes back empty → `timeslice_frac`/`retention` = NaN → the validation assert fails
("time-slice exceeds save-all" is just NaN tripping the `<=` check). The masks are fine — only the
format differs. Repack them (root-owned → sudo). Repacking `cuda_dino/70/gt_masks` also fixes the 4
ML detectors, whose `gt_masks` are directory-symlinks into it:
```bash
cd /home/bqn82/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
sudo python3 repack_offline_masks.py /tmp/usrp_spectrograms/all_detectors/coherent_power/attenuation_dB_70
sudo python3 repack_offline_masks.py /tmp/usrp_spectrograms/all_detectors/cuda_dino/attenuation_dB_70
```
Then re-run the notebook (no sudo) to render 65/70 into every figure:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate dinov3
cd /home/bqn82/Holohub-Signal-Detection/notebooks/data_saving_evals
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3000 data_reduction_eval.ipynb
```

## Also fixed overnight (no sudo, already applied)
- `run_cuda_dino_offline_file.py`: registered `mask_replay` in `DETECTOR_BASE_CONFIGS` so
  `--detector mask_replay` is accepted (was throwing `invalid choice: 'mask_replay'`). The snip in
  Step 2 now passes argparse; it still needs the operator compiled (Step 1).

## (Optional) reclaim ~9.9 GB — delete superseded cuda_dino stale dirs
These are older re-run copies of 4 `cuda_dino` captures (0/5/10/30 dB), auto-renamed `.stale_1` and
superseded by the live dirs (same 328 frames, older uncompressed-`.npy` format). Nothing depends on
them; the snip already skips them. This is purely disk hygiene. They are root-owned → sudo:
```bash
sudo rm -rf /tmp/usrp_spectrograms/all_detectors/cuda_dino/*.stale_1
```
Nothing inside `/tmp/ds_batch` itself is deleted — `ds_batch/cuda_dino` is just a symlink into that
`all_detectors/cuda_dino` dir, which is where the stale copies physically live.

## Notes / caveats
- `/tmp/ds_batch/baseline_run_summary.json` was overwritten by the 65/70 baseline run, so it now
  only lists the 4 new runs. The 0–60 baseline **masks are untouched** — only the summary JSON
  reflects the last run.
- Overnight I also ran `eval_detector_masks.py` over all 8 detectors × 0–70 and refreshed
  `notebooks/yolo_evals/compare_tables/{frame_pixel_metrics,region_metrics}.csv` (collaborator's
  Jul-14 originals backed up to `compare_tables/backup_pre_refresh_20260722_140244/`). Also repointed
  `sweep_all/{coherent_power,cuda_dino}` to the 0–70 data (old targets in
  `~/.claude/jobs/9c38f9ee/tmp/sweep_all_symlinks_before.txt`) so the notebook sees 65/70 for the
  reference detectors too.
- The overnight viz/validation agent worked on `notebooks/data_saving_evals/data_reduction_eval.ipynb`
  (publication figures + time-slicing algorithm validation). Its report: `~/.claude/jobs/9c38f9ee/tmp/viz_agent_report.md`.
- Container name assumed `usrp_x410_sig_det_bqn82` (the repo default in `container_env.sh` points at
  sat3737's container, so the override above is required).

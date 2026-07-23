# Instructions — measured signal_snipper (resample+filter) for the data-saving notebook

These commands produce the **measured** resample+filter footprints that fill in:
- **Figure 3c** — live OTA snipper bars (`~/captures/live_data/sigmf_out/test_{1,2,3}`), and
- **Figure 1** + the "measured resample+filter" pivot table — the wired attenuation sweep
  (`~/captures/attenuation_dB_*`), shown as solid **diamonds** over the dashed projection.

Until you run these, both places show the raw-mask projection with "pending snipper" placeholders;
after running + re-executing the notebook, the measured values drop in automatically.

> **Which models get measured:** the snipper runs *inside the container* and consumes a container
> detector's mask, so measured values are available for **`coherent_power`** and **`cuda_dino`**
> (zero-shot DINO). The offline-only models (FT-DINO M1/M2, YOLO26s/m) make masks in Python and can't
> be snipped by the container binary, so they stay on the projection (labeled). Getting measured
> numbers for them would require running those models in the container (future work).

Steps 1–3 need docker / **lab-admin sudo**; `CONTAINER_NAME` must match the built container.

---

## 0. Build the container (once) — mounts *your* checkout

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 SKIP_IMAGE_BUILD=1 ./bash_scripts/build_demo_container.sh
```
Reuses the existing image, mounts `~/Holohub-Signal-Detection` at `/workspace/holohub`, and compiles
the snipper into the offline-eval binary. (Same container as the OTA coherent/zero-shot run.)

## 1. Live OTA snipper → Figure 3c

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
# coherent_power masks (default):
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_offline_snipper.sh
# zero-shot DINO masks:
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 DETECTOR=cuda_dino CONFIG_NAME=old_configs/config_cuda_dino_signal_snipper_single_channel.yaml ./bash_scripts/run_offline_snipper.sh
```
Default captures = the 3 live OTA files. Snippets land at
`/tmp/usrp_spectrograms/snippets_eval/<detector>/<stem>/snippets/*.sigmf-data`.

## 2. Attenuation sweep snipper → Figure 1 (+ measured table)

Run over the exact attenuation set the notebook uses (0–60 dB, incl. the 30 dB v2 capture):

```bash
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection
CAPS=(0 5 10 15 20 25 30 30_v2 35 40 45 50 55 60)
FILES=(); for a in "${CAPS[@]}"; do FILES+=("$HOME/captures/attenuation_dB_${a}.sigmf-data"); done

# coherent_power:
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 ./bash_scripts/run_offline_snipper.sh "${FILES[@]}"
# zero-shot DINO:
sudo env CONTAINER_NAME=usrp_x410_sig_det_bqn82 DETECTOR=cuda_dino CONFIG_NAME=old_configs/config_cuda_dino_signal_snipper_single_channel.yaml ./bash_scripts/run_offline_snipper.sh "${FILES[@]}"
```
Heads-up: 14 captures × 2 detectors × ~14 GB each is a long replay. To start small, run just
`coherent_power`, re-run the notebook, then add `cuda_dino`. Snippets land at
`/tmp/usrp_spectrograms/snippets_eval/<detector>/attenuation_dB_<db>/snippets/`.

## 3. Re-run the notebook (no sudo)

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate yolo
cd ~/Holohub-Signal-Detection/applications/usrp_wideband_signal_detection/infocom_evals/snip_eval
jupyter nbconvert --to notebook --execute --inplace data_saving_eval_review.ipynb
```
The notebook reads the snippet SigMF sizes and updates:
- `fig1_bytes_per_hour.png` — measured diamonds for `coherent_power` / `cuda_dino`,
- `fig3c_live_ota_resample_measured.png` — measured OTA bars,
- the printed "measured resample+filter: N/78 cells" + the measured pivot table.

---

## Notes / knobs
- Snippet root: `DS_SNIP_ROOT` (default `/tmp/usrp_spectrograms/snippets_eval`). Must match the
  runner's `SNIP_ROOT`.
- OTA stems/duration: `DS_OTA_STEMS` (default `test_1,test_2,test_3`), `DS_OTA_CAP_SEC` (default `3.0`).
- Attenuation captures dir: `DS_CAPTURES_DIR` (default `~/captures`); durations are derived from each
  capture's file size at 245.76 MS/s cf32.
- **cuda_dino** uses `old_configs/config_cuda_dino_signal_snipper_single_channel.yaml` (the perf
  cuda_dino config + the transplanted `signal_snipper`/`sigmf_file_sink` blocks) — passed via
  `CONFIG_NAME`. The default `config_signal_snipper_single_channel.yaml` has only a
  `coherent_power_signal_detector` block, so it can't drive `--detector cuda_dino`.
- The snipper reuses `config_signal_snipper_single_channel.yaml` (coherent_power detector,
  `enable_signal_snipper: true`); `run_cuda_dino_offline_file.py` injects the offline block and points
  `sigmf_file_sink.output_dir` at the per-capture host-visible dir automatically.
- Idempotent: re-running a capture overwrites its snippet dir; re-running the notebook re-reads sizes.

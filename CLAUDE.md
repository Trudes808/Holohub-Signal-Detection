# Claude Code

## Repo-wide guidance

- Keep changes narrow and local. Do not refactor unrelated parts of Holohub while working on one app.
- Prefer existing helper scripts and documented workflows over ad hoc build, container, or run commands.
- Fork-specific user documentation: repo-root `README.md` (entry points), `applications/usrp_wideband_signal_detection/README.md` (install / live demo / offline preprocessing / configs), `dino_fine_tuning/README.md` (retraining). Upstream HoloHub docs live in `doc/holohub_upstream_README.md`. Keep those three READMEs current when workflows change.

## USRP Folder Organization (KEEP THIS)

`applications/usrp_wideband_signal_detection/` is organized as follows — preserve this layout
when adding files, and update references when moving anything:

- **Top-level `config*.yaml`**: only the *current* runnable configs (the table in the app README §6;
  every one is referenced by a wrapper, the conductor, or an offline driver). These are the only
  configs `rebuild_demo_container_app.sh` syncs into the build tree.
- **`bash_scripts/`**: all shell wrappers (build/rebuild/run/enter/after_reboot) plus the shared
  `container_env.sh` (container/image identity) and `container_repo_guard.sh`. Scripts use an
  absolute `SCRIPT_DIR` and derive `APP_DIR="${SCRIPT_DIR}/.."`; keep repo-root math and helper
  `source`s relative to those.
- **`calibration/`**: calibration configs + `calibrate_*.sh`/`.py` + emitted `.npy` artifacts.
  Calibrate scripts run from the app root (they `cd "${SCRIPT_DIR}/.."`) and source
  `bash_scripts/container_env.sh`.
- **`old_configs/`**: superseded/experimental configs (runnable via `CONFIG_NAME=old_configs/<name>.yaml`).
- **`notes/`**: historical development markdowns and design plans (including the pre-fork root notes
  `sage_*`, `running_pytorch_in_holohub.md`, `USRP_X410_Holoscan_FFT_README.txt`).
- New shell scripts go in `bash_scripts/`; new dev notes go in `notes/`; non-current configs go
  in `old_configs/`. Do not scatter `.sh`/notes/stale configs back into the app root.

## USRP Signal-Detection Workflow

- The supported live workflow for `applications/usrp_wideband_signal_detection` is container-based. Do not use a host-local CMake configure/build for that app.
- Container name/image are centralized in `bash_scripts/container_env.sh`; override with `CONTAINER_NAME`/`IMAGE_NAME` env vars. Scripts run under `sudo`, which strips the shell env, so the identity must come from that file (or `sudo env VAR=...`).
- First-time container setup (run from `applications/usrp_wideband_signal_detection`):
  - `sudo env SKIP_IMAGE_BUILD=1 ./bash_scripts/build_demo_container.sh`
- After code or config changes:
  - `sudo ./bash_scripts/rebuild_demo_container_app.sh`
- Use the live wrapper scripts instead of hand-written `docker exec` commands unless debugging the wrappers themselves (all under `bash_scripts/`):
  - `./bash_scripts/run_torchscript_performance_test.sh` (generic runner; `CONFIG_NAME=<cfg> ...`)
  - `./bash_scripts/run_coherent_power_performance.sh`
- Current wrapper constraint: runnable current configs stay as top-level `config*.yaml`. `rebuild_demo_container_app.sh` syncs top-level `config*.yaml` into the build tree; the run wrappers copy `${SOURCE_APP_DIR}/${CONFIG_NAME}` (so `old_configs/<name>.yaml` also works since the build tree has an `old_configs/` dir).
- Prefer frozen-input or offline validation before repeated live SDR runs.

## Experiment Entry Points

- Live demo (current): `sudo ./bash_scripts/run_live_demo.sh v5` / `v5dual` (coherent start, DINO v3 M5-S via DEMO CONTROLS; configs `config_live_v3_*.yaml`, `config_live_v3_dino_ft_m5s*.yaml`).
- Plain live runs: `config_coherent_power_perf_dynamic_single_channel.yaml` (dynamic floor), `config_coherent_power_perf_perfreq_single_channel.yaml` (calibrated), `config_cuda_dino_performance_single_channel.yaml`, `config_cuda_dino_finetuned_performance_single_channel.yaml` (M5-S).
- No-radio loopback: `bash_scripts/run_loopback_m5s_{single,dual}.sh`, `loopback_channel_supervisor.sh` (tcpreplay over the QSFP loopback; configs `config_loopback_v3_*.yaml` for the older 245.76 composite demo).
- Offline dataset preprocessing (detector + snipper over a directory -> `<dir>_preprocessed`): `sudo python3 preprocess_dataset.py <dir> --detector {coherent_power|cuda_dino|cuda_dino_finetuned}`.
- Signal snipper live (headless): `config_signal_snipper_single_channel.yaml`
- Frozen-input capture: `config_coherent_power_capture_chdr_single_channel.yaml`
- Offline eval (all detectors, same operators as live): `python3 run_cuda_dino_offline_file.py <file.sigmf-data> --detector {coherent_power|cuda_dino|cuda_dino_finetuned} --config <cfg>`; sweeps + scoring per `infocom_evals/signal_detection_experiments/run_signal_detection_README.md`.
- Fine-tuning: `dino_fine_tuning/README.md` (M2 recipe = `build_dataset.py` + `train.py --mode ft_lastN`; deployed lineage M3 -> M4 -> M5-S).
- Calibration: `sudo ./calibration/calibrate_coherent_power_config.sh` and `sudo ./calibration/calibrate_dino_coherence_config.sh`.

## Validation Rules

- For detector-mask comparisons, keep validation and production on the same backend and vary `emit_stride`, logging, saves, or timing flags before changing the validated backend.
- Saved host-side artifacts usually appear under `/tmp/usrp_spectrograms` and `/tmp/usrp_dino_masks`.
- Record repeated experiment runs under `applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/` instead of leaving results only in chat.
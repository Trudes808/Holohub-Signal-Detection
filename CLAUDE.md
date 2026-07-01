# Claude Code

## Repo-wide guidance

- Keep changes narrow and local. Do not refactor unrelated parts of Holohub while working on one app.
- Prefer existing helper scripts and documented workflows over ad hoc build, container, or run commands.
- When the task is specific to `applications/usrp_wideband_signal_detection`, follow the path-scoped rule in `.claude/rules/usrp_wideband_signal_detection.md`.

## USRP Signal-Detection Workflow

- The supported live workflow for `applications/usrp_wideband_signal_detection` is container-based. Do not use a host-local CMake configure/build for that app.
- First-time container setup:
  - `cd applications/usrp_wideband_signal_detection`
  - `./build_demo_container.sh`
- After code or config changes:
  - `cd applications/usrp_wideband_signal_detection`
  - `./rebuild_demo_container_app.sh`
- Use the live wrapper scripts instead of hand-written `docker exec` commands unless debugging the wrappers themselves:
  - `./run_torchscript_performance_test.sh`
  - `./run_torchscript_performance_single_channel.sh`
  - `./run_coherent_power_performance.sh`
  - `./run_coherent_power_reference_timing.sh`
  - `./run_torchscript_live_timing_single_channel.sh`
- Current wrapper constraint: runnable experiment configs must stay in `applications/usrp_wideband_signal_detection/` and match `config*.yaml`. `rebuild_demo_container_app.sh` only syncs top-level `config*.yaml` files into the build tree, and the run wrappers copy `${SOURCE_APP_DIR}/${CONFIG_NAME}` from that same root.
- Prefer frozen-input or offline validation before repeated live SDR runs.

## Experiment Entry Points

- Coherent live single-channel baseline: `config_coherent_power_performance_single_channel.yaml`
- Coherent validation baseline: `config_coherent_power_validation.yaml`
- Coherent frozen-input capture baseline: `old_configs/config_coherent_power_debug_capture.yaml`
- DINO live single-channel baseline: `config_torchscript_performance_single_channel.yaml`
- DINO live two-channel baseline: `config_torchscript_performance.yaml`
- Coherent offline replay: `./run_offline_coherent_power_validator_from_tensor.sh --latest-snapshot`
- DINO offline compare: `./dino_cuda_validation.sh --tensor-npy <path>`

## Validation Rules

- For detector-mask comparisons, keep validation and production on the same backend and vary `emit_stride`, logging, saves, or timing flags before changing the validated backend.
- Saved host-side artifacts usually appear under `/tmp/usrp_spectrograms` and `/tmp/usrp_dino_masks`.
- Record repeated experiment runs under `applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/` instead of leaving results only in chat.
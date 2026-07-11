---
paths:
  - "applications/usrp_wideband_signal_detection/**"
---

# USRP Wideband Signal Detection

## Experiment workflow

- Start from an existing baseline config before creating a new one.
- Folder layout (keep it): current runnable configs are top-level `config*.yaml`; shell wrappers live in `bash_scripts/`; calibration configs+scripts in `calibration/`; superseded configs in `old_configs/`; dev notes in `notes/`. New scripts → `bash_scripts/`, new notes → `notes/`, non-current configs → `old_configs/`.
- If a new live config is meant to work with the existing container wrappers, keep it as a top-level `applications/usrp_wideband_signal_detection/config*.yaml`.
- Do not assume `generated_configs/` is runnable by the stock live wrappers. The wrappers sync only top-level `config*.yaml` files into the build tree.
- Prefer single-channel or offline validation first. Use the dual-channel live configs only when the question explicitly depends on dual-channel ingest behavior.
- Keep experiment notes in `applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/`.

## Supported run paths

Run from `applications/usrp_wideband_signal_detection/` as the working root. Container identity comes from `bash_scripts/container_env.sh` (override with `CONTAINER_NAME`/`IMAGE_NAME`).

- Live coherent runs:
  - `CONFIG_NAME=<config>.yaml sudo ./bash_scripts/run_coherent_power_performance.sh`
- Live DINO / generic runs:
  - `CONFIG_NAME=<config>.yaml sudo ./bash_scripts/run_torchscript_performance_test.sh`
- Cable-loopback replay: run live with `config_coherent_power_performance_single_channel_replay.yaml` while replaying the SigMF onto the loopback path.
- Offline eval (both detectors, same operators as live):
  - `python3 run_cuda_dino_offline_file.py <file.sigmf-data> --detector {coherent_power|cuda_dino} --config <cfg> --output-root <dir>`
  - or the notebook `infocom_evals/signal_detection_experiments/debugging_signal_detection_eval.ipynb`
- Calibration: `sudo ./calibration/calibrate_coherent_power_config.sh`, `sudo ./calibration/calibrate_dino_coherence_config.sh`.

## Config integrity

- Unless the task is explicitly about calibration mismatch, preserve the current sender and FFT alignment:
  - sender rate `500e6`
  - `fft.span: 500000000`
  - `fft.transform_points: 20480`
  - `fft.resolution: 24414`
- For coherent-power experiments, treat `emit_stride`, thresholding, persistence, grouping, and save/logging toggles as the normal sweep surface.
- For DINO experiments, prefer existing throughput knobs such as `emit_stride`, detector input size, timing toggles, and reference-vs-offline comparisons before changing unrelated pipeline structure.

## Rebuild and sync behavior

- Do not use host-local CMake for the live app.
- If code or configs changed, use `sudo ./bash_scripts/rebuild_demo_container_app.sh` or one of the wrapper scripts that already calls it.
- `FORCE_REBUILD=1 sudo ./bash_scripts/rebuild_demo_container_app.sh` is the explicit escape hatch when the container build tree looks current but needs a refresh.

## Result capture

- For coherent capture runs, preserve snapshot JSON sidecars so offline replay stays reproducible.
- For offline detector comparisons, capture the `--output-root` directory (masks + spectrogram previews) from `run_cuda_dino_offline_file.py` / the notebook.
- Summaries should include the config path, exact command, key metrics, artifact paths, and the next recommended sweep.
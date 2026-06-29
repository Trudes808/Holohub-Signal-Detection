---
paths:
  - "applications/usrp_wideband_signal_detection/**"
---

# USRP Wideband Signal Detection

## Experiment workflow

- Start from an existing baseline config before creating a new one.
- If a new live config is meant to work with the existing container wrappers, keep it at `applications/usrp_wideband_signal_detection/config*.yaml`.
- Do not assume `generated_configs/` is runnable by the stock live wrappers. Today the wrappers sync only top-level `config*.yaml` files into the build tree.
- Prefer single-channel or offline validation first. Use the dual-channel live configs only when the question explicitly depends on dual-channel ingest behavior.
- Keep experiment notes in `applications/usrp_wideband_signal_detection/infocom_evals/signal_detection_experiments/`.

## Supported run paths

- Live coherent runs:
  - `CONFIG_NAME=<config>.yaml ./run_coherent_power_performance.sh`
- Live DINO runs:
  - `CONFIG_NAME=<config>.yaml ./run_torchscript_performance_test.sh`
- Coherent frozen-input replay:
  - `./run_offline_coherent_power_validator_from_tensor.sh --latest-snapshot`
  - `./run_offline_coherent_power_validator_from_tensor.sh --snapshot-json <path>`
- DINO frozen-input compare:
  - `./dino_cuda_validation.sh --tensor-npy <path> [--cuda-config <path> --reference-config <path>]`

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
- If code or configs changed, use `./rebuild_demo_container_app.sh` or one of the wrapper scripts that already calls it.
- `FORCE_REBUILD=1 ./rebuild_demo_container_app.sh` is the explicit escape hatch when the container build tree looks current but needs a refresh.

## Result capture

- For coherent validation or capture runs, preserve snapshot JSON sidecars and validator outputs so offline replay stays reproducible.
- For DINO comparisons, capture the CUDA output directory, reference output directory, and generated plots from `dino_cuda_validation.sh`.
- Summaries should include the config path, exact command, key metrics, artifact paths, and the next recommended sweep.
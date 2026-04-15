# USRP Wideband Signal Detection

## Overview

This application mirrors the high-rate USRP ingest path and supports selectable detector stages.

Flow:

`chdrConverterOp -> fftOp -> spectrogramOp -> detectorOp`

A side logger branch is kept from `fftOp` for throughput visibility.

The app now supports pipeline-isolation modes through config:

- `pipeline.enable_spectrogram`
	- bypasses `spectrogramOp` entirely when false
- `pipeline.enable_detector`
	- bypasses the detector operator entirely when false
- `pipeline.detector_type`
	- selects `dinov3` or `coherent_power` at graph construction time
- `pipeline.log_from_spectrogram`
	- switches the throughput logger to the post-spectrogram path when true

Current detector choices:

- `dinov3`
	- existing notebook-aligned DINOv3 path with TorchScript and CUDA fallback modes
- `coherent_power`
	- new coherent-power detector scaffold derived from `coherant_power_signal_detection.ipynb`
	- current state is plumbing-complete but algorithmically still a placeholder CUDA mask path until notebook stages are ported fully

The current runtime target is the Holohub development container. The local DINOv3 source of truth lives outside the container and must be staged into the container runtime tree before model-forward validation:

- host repo: `/home/sat3737/holoscan_demo_workspace/dinov3`
- host weights: `/home/sat3737/holoscan_demo_workspace/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- container repo target: `/workspace/models/dinov3`
- container TorchScript target: `/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`

## Run

From the build directory for this application:

```bash
./usrp_wideband_signal_detection config.yaml
```

Available configs copied into the build directory:

- `config.yaml`
	- stable debug-artifact mode
	- saves the first 5 spectrograms and first 5 detector masks per channel
	- keeps `inference_backend: "pytorch_placeholder"` so the known C++ TorchScript init crash does not block runtime checks
- `config_cuda_fallback.yaml`
	- C++/CUDA fallback debug mode
	- saves the first 5 spectrograms and first 5 detector masks per channel
	- forces `use_pytorch_backend: false` and `inference_backend: "cuda_threshold_fallback"` so detector behavior stays on the non-Torch path
- `config_torchscript_cpu_eval.yaml`
	- isolates whether `eval()` is safe while the module is still on CPU
	- uses `inference_backend: "torchscript"`, `torchscript_init_mode: "load_cpu_eval"`, `strict_model_forward: false`, and the CPU-exported TorchScript artifact `dinov3_vitb16_pretrain_lvd1689m-73cec8be_cpu.ts`
- `config_torchscript_cuda_no_eval.yaml`
	- isolates whether the CUDA transfer itself is safe before `eval()` runs
	- uses `inference_backend: "torchscript"`, `torchscript_init_mode: "load_cuda_no_eval"`, and `strict_model_forward: false`
- `config_torchscript_validation.yaml`
	- strict crash-repro and validation mode
	- uses `inference_backend: "torchscript"`, `strict_model_forward: true`, and `torchscript_init_mode: "load_cuda_eval"`
- `config_torchscript_performance.yaml`
	- two-channel throughput test mode
	- disables spectrogram saves, detector mask saves, per-frame detection logging, and timing summaries to keep the data path as lean as possible
	- keeps GPU RX pools at the known-safe `25000` buffers per channel to avoid GPUDirect BAR1 DMA-map failures, while reducing queue burst size and raising `num_simul_batches` so the graph can absorb more ingress jitter before dropping packets
- `config_torchscript_performance_timing_debug.yaml`
	- debug-only hotspot profiling mode for the two-channel TorchScript path
	- re-enables detector timing summaries and raises `emit_stride` so the synchronized timing probe can print stage timings without immediately collapsing ingress
- `config_torchscript_realtime_guarded.yaml`
	- guarded two-channel realtime attempt for the notebook-faithful TorchScript reference backend
	- uses smaller ingress/FFT batches plus a conservative detector cadence to reduce packet retention and create buffer headroom before re-tightening throughput knobs
- `config_torchscript_performance_fft_only.yaml`
	- ingress and FFT isolation mode
	- bypasses both spectrogram and detector so the first throughput ceiling can be measured without downstream ML work
	- uses legacy-style large ingress batches (`12500` packets / `625` FFTs, `2` simultaneous batches) to mirror the older PSD path more closely
- `config_torchscript_performance_spectrogram_only.yaml`
	- ingress, FFT, and spectrogram isolation mode
	- bypasses the detector while logging from the post-spectrogram path to prove whether `spectrogramOp` is still throughput-safe when save is disabled
- `config_torchscript_performance_small_batches.yaml`
	- detector-enabled throughput mode with smaller CHDR/FFT batches
	- reduces `num_ffts_per_batch` and queue batch size to test whether coarse batch retention is a major source of drops before detector rewrite work begins
- `config_torchscript_load_only.yaml`
	- lower-risk TorchScript diagnostic mode
	- loads the TorchScript artifact without moving it to CUDA or attempting `eval()`, then falls back to placeholder inference during compute

Use the same external USRP stream command used by `usrp_freq_detection`.

## Stream Alignment

This app does not currently receive `sample_rate_hz` metadata from `rx_to_remote_udp.py`.
In practice, the receive side falls back to the local FFT config, so the sender and receiver settings must match.

For the current X410 flow, keep these aligned:

- sender `rx_to_remote_udp.py --rate 500e6`
- receiver `fft.span: 500000000` in the selected `config*.yaml`
- receiver `fft.transform_points: 20480`

Notes:

- `master_clock_rate=500e6` in the USRP device args does not propagate into Holoscan operator metadata.
- The FFT operator will use upstream metadata only if some earlier operator explicitly sets `sample_rate_hz` or `bandwidth_hz`.
- If no upstream rate metadata exists, the FFT operator derives downstream `span` and `resolution` from the selected config file.
- With `transform_points: 20480` and `span: 500000000`, the effective FFT bin width is about `24414 Hz`, so the fallback `resolution` values in the configs are set to `24414` for consistency.

If these values do not match, saved spectrograms, detector metadata, notebook validation, and offline replay can all appear frequency-calibrated while still being calibrated to the wrong span.

## Host Workflow

The container workflow is now organized around four primary host-side scripts:

- `build_demo_container.sh`
	- initial setup script for a new machine or fresh container
	- builds the demo image, creates the container, mounts `/tmp/usrp_spectrograms` and `/tmp/usrp_dino_masks`, stages the local DINOv3 repo and weight, installs the pinned CUDA 12.6 PyTorch stack when needed, exports the TorchScript artifact, ensures Vulkan and MatX are present, and builds the app in the container
- `run_demo_container.sh`
	- starts an already-created container if it is stopped
	- does not recreate or reprovision the container
- `rebuild_demo_container_app.sh`
	- checks whether tracked build targets are already up to date and only rebuilds when necessary
	- always syncs the latest `config*.yaml` files into the build output directory after the check
- `enter_demo_container.sh`
	- opens an interactive shell in the running container

Compatibility wrappers still exist:

- `setup_demo_container.sh`
	- deprecated wrapper that forwards to `build_demo_container.sh`
- `rebuild_and_debug.sh`
	- deprecated wrapper that forwards to `rebuild_demo_container_app.sh`

Default host-side debug outputs:

- spectrograms: `/tmp/usrp_spectrograms`
- DINO masks: `/tmp/usrp_dino_masks`

### First-Time Setup

Run this once when bringing the app up on a new machine or when recreating the container from scratch:

Clone dinoV3 
```bash
git clone https://github.com/facebookresearch/dinov3
```

```bash
cd applications/usrp_wideband_signal_detection
./build_demo_container.sh
```

If the container already exists later and is only stopped, restart it with:

```bash
cd applications/usrp_wideband_signal_detection
./run_demo_container.sh
```

### Enter And Run The App

Enter the container:

```bash
cd applications/usrp_wideband_signal_detection
./enter_demo_container.sh
```

Inside the container, run the application from the build directory. For the current non-Torch smoke-test path that saves spectrograms and detector masks to the mounted host `/tmp` directories:

```bash
cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
./usrp_wideband_signal_detection config_cuda_fallback.yaml
```

If you want the stable placeholder debug path instead:

```bash
cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
./usrp_wideband_signal_detection config.yaml
```

Start the radio stream from a second host terminal using the same external USRP command family as `usrp_freq_detection`.

### After Code Changes

When you change code or config files in the repository:

```bash
cd applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh
./enter_demo_container.sh
```

Then rerun the app inside the container:

```bash
cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
./usrp_wideband_signal_detection config_cuda_fallback.yaml
```

For the two-channel performance pass with the real TorchScript detector path and debug outputs disabled:

```bash
cd applications/usrp_wideband_signal_detection
./run_torchscript_performance_test.sh
```

For the two-channel coherent-power real-time path with the coherent detector selected and debug outputs disabled:

```bash
cd applications/usrp_wideband_signal_detection
./run_coherent_power_performance.sh
```

For the cleanest startup behavior, launch the app before starting the RF transmitter. The advanced-network DPDK workers are started during app initialization, before the Holoscan graph is fully active, so starting the transmitter first can produce a brief burst of startup-only drops even when steady-state throughput is healthy.

Validation and production parity rule: for both the coherent-power and DINO detectors, any config intended to validate or represent production mask behavior must use the same `backend_mode`. The approved throughput knob is `emit_stride`; logs, saves, and timing summaries may differ, but the mask-generation backend must not.

For a more conservative coherent real-time profile that prioritizes sustained ingest headroom while the coherent detector still contains host-side stages:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=config_coherent_power_realtime_guarded.yaml ./run_coherent_power_performance.sh
```

To run the coherent detector with an explicit config through the same sudo/docker flow:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=config_coherent_power_validation.yaml ./run_coherent_power_performance.sh
```

To capture frozen coherent-power validation artifacts for notebook and offline replay:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=config_coherent_power_debug_capture.yaml ./run_coherent_power_performance.sh
```

That debug profile keeps the coherent detector on the notebook-faithful reference backend and saves a bounded set of:

- final mask images
- complex input tensor snapshots
- optional `power_db` snapshots
- JSON sidecars containing the detector config and frame metadata

After a capture run, validate one frozen snapshot offline with the standalone C++ validator:

```bash
./offline_coherent_power_validator --snapshot-json /path/to/coherent_power_snapshot_ch0_f1_<timestamp>_<rows>x<cols>.json --verbose
```

The validator writes replay artifacts under `validator_artifacts/` next to the snapshot sidecar, including `offline_power_db.npy`, `offline_corrected_sxx_db.npy`, `offline_final_mask.npy`, preview `.pgm` images, and `offline_validation_summary.json`.

The matching notebook replay path now lives in [dinov3/notebooks/coherant_power_signal_detector_validation.ipynb](dinov3/notebooks/coherant_power_signal_detector_validation.ipynb). Use that notebook to load the same snapshot sidecar, replay the Python reference pipeline, and compare the notebook outputs against the offline C++ validator on the exact same frozen input.

For the staged bottleneck-isolation passes, reuse the same helper with `CONFIG_NAME`:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=config_torchscript_performance_fft_only.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=config_torchscript_performance_spectrogram_only.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=config_torchscript_performance_small_batches.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=config_torchscript_realtime_guarded.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=config_torchscript_performance_timing_debug.yaml ./run_torchscript_performance_test.sh
```

If you need to force a rebuild even when the targets look current:

```bash
cd applications/usrp_wideband_signal_detection
FORCE_REBUILD=1 ./rebuild_demo_container_app.sh
```

`rebuild_demo_container_app.sh` now also tracks `coherent_power_signal_detector` and `offline_coherent_power_validator` directly in its dry-run target set, and then explicitly builds those auxiliary targets after the main app configure/build step. That keeps operator-only and offline-validator edits from being skipped just because the app binary itself was already current.

The container helper scripts now also verify that the running `usrp_x410_signal_detection_demo` container is mounted from the same `holohub-dev` checkout as the script you launched. If the container was created from another clone, the scripts stop immediately and tell you to recreate it from the current checkout instead of silently using the wrong source tree.

## Visualization

The visualization path is now structured around the real C++ pipeline:

- `usrp_wideband_signal_detection` can open a HoloViz spectrogram window from the live `spectrogramOp` branch when `visualization.enable: true`
- `offline_spectrogram_visualizer` replays saved `.pgm` spectrogram frames without requiring a connected radio

### Live Spectrogram Window

Set `visualization.enable: true` in `config.yaml` to turn on the live spectrogram branch.

The current live renderer:

- branches directly from `spectrogramOp`
- converts the spectrogram tensor into a classic spectrum-analyzer style HoloViz image
- keeps detector overlay work separate so the render path stays reusable
- includes a top PSD strip, max-hold trace, analyzer-style spectrogram panel, side readouts, and color-limit slider visuals shared with offline replay

### Offline Replay

Example usage from the build directory:

```bash
./offline_spectrogram_visualizer --offline-dir /tmp/usrp_spectrograms --mask-dir /tmp/usrp_dino_masks --fps 8
```

Headless screenshot export:

```bash
./offline_spectrogram_visualizer --offline-dir /workspace/spectrograms --screenshot offline_preview.png
```

Useful flags:

- `--config <FILE>`
	- use a specific replay config file, defaulting to `config_offline_replay.yaml`
- `--offline-dir <DIR>`
	- directory containing saved `.pgm` spectrogram frames
- `--fps <FPS>`
	- playback rate for offline replay
- `--mask-dir <DIR>`
	- directory containing `dino_mask_ch*_f*_*.pgm` files; when a frame match exists, the mask is blended into the spectrogram panel and reflected in the sidebar metrics
- `--screenshot <FILE>`
	- export the first replayed frame as the full composed dashboard preview without starting HoloViz; relative paths are saved under `/workspace/spectrograms`, which maps back to the host spectrogram directory

Renderer tuning fields in the replay and live configs:

- `blue_limit`
	- lower heatmap clamp shown by the blue slider and applied to spectrogram color scaling
- `red_limit`
	- upper heatmap clamp shown by the red slider and applied to spectrogram color scaling
- `center_frequency_hz`
	- display-only center frequency readout until the live pipeline forwards that metadata directly
- `fft_size`
	- FFT size shown in the analyzer info panel
- `dino_chunk_rows`, `dino_chunk_cols`
	- DINO chunk dimensions shown in the analyzer info panel
- `--no-loop`
	- stop after the final frame instead of looping

Remote usage note:

- the viewer uses HoloViz and opens a native window on the machine running the executable
- this works in a local desktop session, a remote desktop session, or a correctly configured display-forwarded container
- this is not expected to render in a plain headless SSH session with no display server

The `--screenshot` path is the current no-desktop fallback. It writes the same composed dashboard frame used by offline replay and does not depend on HoloViz, GLFW, or Vulkan. If a matching detector mask exists in `/workspace/dino_masks` or the directory passed to `--mask-dir`, the preview includes the overlay and sidebar overlay metrics. If you pass a simple filename such as `offline_preview.png`, it will be written to `/workspace/spectrograms/offline_preview.png` in the container and show up in the mapped host directory, typically `/tmp/usrp_spectrograms/offline_preview.png`.

If you run inside the demo container and see `Failed to initialize glfw` or `Failed to detect any supported platform`, the container was started without host display forwarding. Relaunch it from a desktop-capable session with `DISPLAY` set so `run_demo_container.sh` can forward `/tmp/.X11-unix` and `XAUTHORITY` into the container.

The next visualization step is to add a detector overlay postprocessor that emits HoloViz overlay tensors and `InputSpec` metadata.

## Validation Notes

- `config.yaml` is now the stable debug run configuration. It intentionally keeps `inference_backend: "pytorch_placeholder"` while saving the first 5 spectrograms and detector masks per channel.
- Parity rule: validation and production/performance configs must keep the same detector `backend_mode` for mask creation. Use `emit_stride`, artifact saves, logging, and timing summaries as the tuning knobs; do not change the mask backend between validation and production.
- `config_coherent_power_debug_capture.yaml` is the coherent-power frozen-input capture profile. It enables tensor snapshot saves, optional `power_db` snapshot saves, and final mask saves so notebook and offline C++ parity checks can run on the exact same detector input.
- `config_cuda_fallback.yaml` is the debug configuration for the pure C++/CUDA detector path. It disables the PyTorch backend in operator logic and uses `cuda_threshold_fallback` while keeping artifact saves enabled.
- `config_torchscript_validation.yaml` is the strict TorchScript bring-up configuration. Use it when you want the C++ TorchScript path to fail loudly.
- `config_torchscript_performance.yaml` is the low-overhead throughput configuration for two-channel rate testing. It keeps the same reference mask-generation backend as validation while disabling artifact saves, detailed detection logs, and timing summaries.
- `config_torchscript_load_only.yaml` is the first diagnostic step for the C++ TorchScript path. It confirms whether `torch::jit::load(...)` itself is safe before the operator attempts CUDA transfer.
- `config_torchscript_cpu_eval.yaml` is the second diagnostic step. It tests whether `eval()` is safe while staying entirely on CPU.
- The CPU validation flow should use the CPU-exported artifact `dinov3_vitb16_pretrain_lvd1689m-73cec8be_cpu.ts`; the original `dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts` remains the CUDA-traced artifact.
- `config_torchscript_cuda_no_eval.yaml` is the third diagnostic step. It tests whether `to(torch::kCUDA)` is safe before `eval()` runs.
- Because the current executable is linked against libtorch when Torch is available at build time, the Torch runtime libraries still need to be present in the container even when you launch `config_cuda_fallback.yaml`.
- The selected runtime weight is `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
- The recommended export helper for container-side TorchScript generation is `applications/usrp_wideband_signal_detection/export_dinov3_torchscript.py`.
- The setup flow is GPU-only. It verifies `nvidia-smi`, checks `torch.cuda.is_available()`, and fails instead of silently exporting on CPU.
- The default PyTorch package source for setup is `https://download.pytorch.org/whl/cu126`, pinned to `torch==2.10.0` and `torchvision==0.25.0` to stay in the CUDA 12.x family.
- This matches the CUDA 12.6 direction already used by [applications/usrp_freq_detection/Dockerfile](applications/usrp_freq_detection/Dockerfile#L26), instead of the incompatible CUDA 13.0 wheel currently present in the notebook environment.
- The in-container build step now calls `./holohub build usrp_wideband_signal_detection --local` explicitly to avoid a nested Docker build attempt inside the container.
- The in-container build step also passes `-Dmatx_DIR=/usr/local/lib/cmake/matx`, matching the MatX install location already present in the USRP container image.
- Fresh image rebuilds should use `./build_demo_container.sh` after this change so new containers inherit the MatX-enabled base image instead of relying on setup-time bootstrapping.
- When `torchscript_init_mode` is set to `load_only` or `load_cpu_eval`, the operator logs that the module is not forward-ready and intentionally falls back to `pytorch_placeholder` during compute.

## Notes

- `spectrogramOp` can still save debug spectrogram images when explicitly re-enabled.
- `dinoV3SignalDetectorOp` currently expects container-staged model artifacts and emits a deterministic mask tensor plus metadata.
- The current TorchScript path is a model-forward bring-up target, not the final postprocessed detection architecture.

# USRP Wideband Signal Detection

## Overview

This application mirrors the high-rate USRP ingest path and adds the new DINOv3 signal detector stage.

Flow:

`chdrConverterOp -> fftOp -> spectrogramOp -> dinoV3SignalDetectorOp`

A side logger branch is kept from `fftOp` for throughput visibility.

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
- `config_torchscript_load_only.yaml`
	- lower-risk TorchScript diagnostic mode
	- loads the TorchScript artifact without moving it to CUDA or attempting `eval()`, then falls back to placeholder inference during compute

Use the same external USRP stream command used by `usrp_freq_detection`.

## Host Automation Scripts

The application directory now includes three host-side helper scripts for the container workflow:

- `build_demo_container.sh`
	- builds the Holohub image with `IMAGE_NAME` defaulting to `usrp_x410_signal_detection_demo:latest`
	- uses `applications/usrp_freq_detection/Dockerfile` by default so the image inherits the same MatX and USRP-facing dependencies already used by the working frequency-detection app
- `run_demo_container.sh`
	- starts a fresh privileged container with `CONTAINER_NAME` defaulting to `usrp_x410_signal_detection_demo`
	- by default it starts the container detached with a keepalive command so setup can target it reliably
	- exports `HOLOHUB_BUILD_LOCAL=1` so build commands executed inside the container stay in local mode
	- mounts `/tmp/usrp_spectrograms` on the host to `/workspace/spectrograms` in the container so the viewer notebook can read generated debug spectrograms after the run
	- mounts `/tmp/usrp_dino_masks` on the host to `/workspace/dino_masks` in the container for detector mask artifacts
	- host output paths can be overridden with `SPECTROGRAM_HOST_DIR=...` and `DINO_MASK_HOST_DIR=...`
- `setup_demo_container.sh`
	- stages the local DINOv3 repo and selected weight into the running container, verifies `nvidia-smi` works inside the container, installs a pinned CUDA 12.6 PyTorch stack plus the DINOv3 Python requirements when needed, exports the TorchScript artifact on GPU, and by default builds the application inside the container
	- bootstraps MatX inside the running container if `/usr/local/lib/cmake/matx/matx-config.cmake` is missing, so an older image can still complete the build without a full rebuild
	- verifies DPDK build dependencies are present before the application configure step and stops early if the container must be rebuilt from the MatX/USRP-enabled image
- `rebuild_demo_container_app.sh`
	- rebuilds only `usrp_wideband_signal_detection` inside the existing container and lists the generated app build directory
	- use this after source edits when the container already has staged models and dependencies
- `enter_demo_container.sh`
	- opens an interactive bash shell in the named container and auto-starts it if needed

Typical usage from the host:

```bash
cd applications/usrp_wideband_signal_detection
./build_demo_container.sh
./run_demo_container.sh
```

Default host-side debug outputs after container startup:

- spectrograms: `/tmp/usrp_spectrograms`
- DINO masks: `/tmp/usrp_dino_masks`

In a second terminal:

```bash
cd applications/usrp_wideband_signal_detection
./setup_demo_container.sh
```

To open a shell after the container is up:

```bash
cd applications/usrp_wideband_signal_detection
./enter_demo_container.sh
```

If you want setup to skip dependency installation or app build:

```bash
INSTALL_PYTHON_DEPS=0 BUILD_APP_IN_CONTAINER=0 ./setup_demo_container.sh
```

If the running container has a stale binary after source changes, rebuild just this app:

```bash
cd applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh
```

## Offline Visualizer

An initial offline viewer executable is now part of this application directory:

- `offline_spectrogram_visualizer`

Purpose:

- replay saved spectrogram `.pgm` frames without a connected radio
- exercise the HoloViz rendering path before live pipeline integration
- provide a usable visualization path during remote development

Example usage from the build directory:

```bash
./offline_spectrogram_visualizer --offline-dir /tmp/usrp_spectrograms --fps 8
```

Useful flags:

- `--offline-dir <DIR>`
	- directory containing saved `.pgm` spectrogram frames
- `--fps <FPS>`
	- playback rate for offline replay
- `--count <COUNT>`
	- stop after a fixed number of frames
- `--color-mode <MODE>`
	- `gray` or `heatmap`
- `--no-loop`
	- stop advancing once the final frame is reached
- `--hide-fake-overlay`
	- disable the prototype overlay labels at startup

Remote usage note:

- the viewer uses HoloViz and opens a native window on the machine running the executable
- this works well in a local desktop session, inside a remote desktop session, or with display forwarding configured correctly
- this is not expected to render in a plain headless SSH session with no display server

The current viewer is intentionally offline-only. The next step is to add overlay rendering and then connect the same viewer path to live spectrogram output.

## Validation Notes

- `config.yaml` is now the stable debug run configuration. It intentionally keeps `inference_backend: "pytorch_placeholder"` while saving the first 5 spectrograms and detector masks per channel.
- `config_cuda_fallback.yaml` is the debug configuration for the pure C++/CUDA detector path. It disables the PyTorch backend in operator logic and uses `cuda_threshold_fallback` while keeping artifact saves enabled.
- `config_torchscript_validation.yaml` is the strict TorchScript bring-up configuration. Use it when you want the C++ TorchScript path to fail loudly.
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

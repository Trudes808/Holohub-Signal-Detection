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
./usrp_wideband_signal_detection old_configs/config.yaml
```

Available configs copied into the build directory:

- `old_configs/config.yaml`
	- stable debug-artifact mode
	- saves the first 5 spectrograms and first 5 detector masks per channel
	- keeps `inference_backend: "pytorch_placeholder"` so the known C++ TorchScript init crash does not block runtime checks
- `old_configs/config_cuda_fallback.yaml`
	- C++/CUDA fallback debug mode
	- saves the first 5 spectrograms and first 5 detector masks per channel
	- forces `use_pytorch_backend: false` and `inference_backend: "cuda_threshold_fallback"` so detector behavior stays on the non-Torch path
- `old_configs/config_torchscript_cpu_eval.yaml`
	- isolates whether `eval()` is safe while the module is still on CPU
	- uses `inference_backend: "torchscript"`, `torchscript_init_mode: "load_cpu_eval"`, `strict_model_forward: false`, and the CPU-exported TorchScript artifact `dinov3_vitb16_pretrain_lvd1689m-73cec8be_cpu.ts`
- `old_configs/config_torchscript_cuda_no_eval.yaml`
	- isolates whether the CUDA transfer itself is safe before `eval()` runs
	- uses `inference_backend: "torchscript"`, `torchscript_init_mode: "load_cuda_no_eval"`, and `strict_model_forward: false`
- `old_configs/config_torchscript_validation.yaml`
	- strict crash-repro and validation mode
	- uses `inference_backend: "torchscript"`, `strict_model_forward: true`, and `torchscript_init_mode: "load_cuda_eval"`
- `config_torchscript_performance.yaml`
	- two-channel throughput test mode
	- disables spectrogram saves, detector mask saves, per-frame detection logging, and timing summaries to keep the data path as lean as possible
	- when spectrogram save, tensor save, visualization, and post-spectrogram logging are all disabled, the app now bypasses the pass-through `spectrogramOp` and feeds FFT output directly into the detector to remove one graph hop from the hot path
	- now uses `backend_mode: "reference"` with `emit_stride: 1` at the full `256x512` detector input so the default non-debug throughput path exercises the validated live mask-generation path rather than a prototype-only backend split
	- keeps GPU RX pools at the current highest known-good `26624` buffers per channel; larger values can fail during DPDK startup when mlx5 attempts to DMA-map both GPU RX regions for GPUDirect RDMA
- `config_torchscript_performance_single_channel.yaml`
	- single-channel throughput test mode for finding the highest realtime sender rate the current BAR1/topology-limited host can support with one active RF channel
	- keeps the same large `1024`-FFT batch geometry and `emit_stride: 1` validated detector settings as the main performance config, but reduces the pipeline, network queues, and operators to channel 0 only
	- uses a larger single GPU RX pool (`49152` buffers) because only one GPUDirect RX region is mapped in this mode
- `old_configs/config_torchscript_performance_timing_debug.yaml`
	- debug-only hotspot profiling mode for the two-channel TorchScript path
	- re-enables detector timing summaries and raises `emit_stride` so the synchronized timing probe can print stage timings without immediately collapsing ingress
- `old_configs/config_torchscript_performance_timing_debug_fast_post.yaml`
	- legacy debug-only throughput probe from the prototype backend split era
	- keep this only for historical comparison notes; active live-port work should stay on the validated `reference` backend
- `old_configs/config_torchscript_performance_timing_debug_small_input.yaml`
	- debug-only reduced-token-count probe
	- changes detector and spectrogram output from `256x512` to `192x384` so you can measure how much Torch runtime scales with patch count
- `old_configs/config_torchscript_performance_timing_debug_fp16.yaml`
	- debug-only lower-precision inference probe
	- keeps the same `256x512` input size but requests `torch_dtype: "fp16"` to estimate how much speedup half-precision TorchScript can provide on this GPU
- `old_configs/config_torchscript_realtime_guarded.yaml`
	- guarded two-channel realtime attempt for the notebook-faithful TorchScript reference backend
	- uses smaller ingress/FFT batches plus a conservative detector cadence to reduce packet retention and create buffer headroom before re-tightening throughput knobs
- `old_configs/config_torchscript_performance_fft_only.yaml`
	- ingress and FFT isolation mode
	- bypasses both spectrogram and detector so the first throughput ceiling can be measured without downstream ML work
	- uses legacy-style large ingress batches (`12500` packets / `625` FFTs, `2` simultaneous batches) to mirror the older PSD path more closely
- `old_configs/config_torchscript_performance_fft_only_matched.yaml`
	- ingress and FFT isolation mode using the same `2048` queue batches and `128` FFT batches as the current fast DINO performance profile
	- use this together with `old_configs/config_torchscript_performance_spectrogram_only.yaml` and `config_torchscript_performance.yaml` to pinpoint whether drops begin in FFT, spectrogram, or only once the fast DINO stage is added
- `old_configs/config_torchscript_performance_spectrogram_only.yaml`
	- ingress, FFT, and spectrogram isolation mode
	- bypasses the detector while logging from the post-spectrogram path to prove whether `spectrogramOp` is still throughput-safe when save is disabled
- `old_configs/config_torchscript_performance_small_batches.yaml`
	- detector-enabled throughput mode with smaller CHDR/FFT batches
	- reduces `num_ffts_per_batch` and queue batch size to test whether coarse batch retention is a major source of drops before detector rewrite work begins
- `old_configs/config_torchscript_load_only.yaml`
	- lower-risk TorchScript diagnostic mode
	- loads the TorchScript artifact without moving it to CUDA or attempting `eval()`, then falls back to placeholder inference during compute

Use the same external USRP stream command used by `usrp_freq_detection`.

## Platform Notes

The current host has two verified GPUDirect constraints that matter for this application:

- `sudo ./operators/advanced_network/python/tune_system.py --check bar1-size`
	- reports GPU BAR1 size `256 MiB`
	- the advanced-network guide recommends `1 GiB` or higher for GPUDirect-heavy workloads
- `sudo ./operators/advanced_network/python/tune_system.py --check topo`
	- reports the GPU-to-NIC path as non-ideal for this setup
	- `nvidia-smi topo -m` shows the active Mellanox NIC function used by the app is connected to the GPU through `SYS`, not `PIX` or `PXB`

Observed impact on `config_torchscript_performance.yaml`:

- dual-channel GPU RX pools at `26624` buffers per channel start and run successfully
- dual-channel GPU RX pools at `27648` buffers per channel fail during DPDK startup with `mlx5_common: Fail to create MR` and `Could not DMA map EXT memory`
- this failure occurs in GPUDirect DMA registration, not because of framebuffer memory exhaustion

Practical guidance:

- treat `26624` as the current best known two-channel operating point on this host unless BAR1 size or PCIe topology changes
- if the machine is reconfigured, re-check BAR1 size and `nvidia-smi topo -m` before assuming larger GPU RX pools should work

## Stream Alignment

This app does not currently receive `sample_rate_hz` metadata from `rx_to_remote_udp.py`.
In practice, the receive side derives its runtime FFT sizing from `chdr_converter.channel_sample_rates_hz` when that config is set consistently across channels, and otherwise falls back to the local FFT config, so the sender and receiver settings must match.

For the current X410 flow, keep these aligned:

- sender `rx_to_remote_udp.py --rate 500e6`
- receiver `fft.span: 500000000` in the selected `config*.yaml`
- receiver `fft.transform_points: 20480`
- receiver `fft.reference_span_hz: 500000000` and `fft.reference_fft_size: 20480` if you want to pin a different baseline explicitly

Notes:

- `master_clock_rate=500e6` in the USRP device args does not propagate into Holoscan operator metadata.
- The live app now derives a runtime FFT width from the 500 MHz / 20480 reference pair by snapping the span ratio to the nearest factor-of-two step, then quantizing to the CHDR packet width. With the current `num_complex_samples_per_packet: 1024`, live FFT sizes move in 1024-sample increments.
- Set `fft.override_fft_bin_size` to request an exact target bin width instead of the default factor-of-two scaling. The resulting live FFT size is still quantized to the packet width, so the final bin size is the nearest feasible value.
- The FFT operator will use upstream metadata only if some earlier operator explicitly sets `sample_rate_hz` or `bandwidth_hz`.
- If no upstream rate metadata exists, the FFT operator derives downstream `span` and `resolution` from the selected config file or the configured CHDR sample rate fallback.
- With `transform_points: 20480` and `span: 500000000`, the effective reference FFT bin width is about `24414 Hz`, so the fallback `resolution` values in the configs are set to `24414` for consistency.

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

Use two terminals for the active live flow.

In a host Python terminal, start the dual-channel radio stream:

```bash
python3.12 rx_to_remote_udp.py \
	--args "mgmt_addr=192.168.20.3,addr=192.168.100.3,second_addr=192.168.10.2" \
	--freq 2400e6 1000e6 \
	--rate 500e6 \
	--gain 30 \
	--channels 0 1 \
	--dest-addr 192.168.10.51 \
	--dest-port 1234 1235 \
	--adapter sfp0 \
	--dest-mac-addr E0:9D:73:E0:5B:6B \
	--spp 1024
```

In a second terminal, enter the demo container and run the application from the build directory.

Dual-channel active config:

```bash
cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
./usrp_wideband_signal_detection config_coherent_power_performance_emit_stride1_two_channel.yaml
```

Single-channel active config:

```bash
cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
./usrp_wideband_signal_detection config_coherent_power_performance_single_channel.yaml
```

For the cleanest startup behavior, launch the app before starting the RF transmitter. The advanced-network DPDK workers are started during app initialization, before the Holoscan graph is fully active, so starting the transmitter first can produce a brief burst of startup-only drops even when steady-state throughput is healthy.

### After Reboot

If the host has been rebooted since the last successful run, restore the host-side prerequisites before blaming the app config.

For the common reboot-recovery path, run this from the host checkout first:

```bash
cd applications/usrp_wideband_signal_detection
./after_reboot.sh
```

That script restores the pieces this demo is sensitive to after reboot: it reloads `nvidia-peermem`, ensures hugepages are reserved and mounted, fixes `/dev/hugepages` permissions so the non-root container shell can launch DPDK again, resets the dedicated Mellanox ports used by this workflow, starts the demo container if needed, and clears stale DPDK runtime state.

On this setup, the helper defaults to `3` hugepages if no usable persisted value is present. Override that with `HUGEPAGES_COUNT` if your host needs a different reservation.

If your host loses hugepage reservation on reboot and you have not already persisted it, provide the known-good count explicitly the first time:

```bash
cd applications/usrp_wideband_signal_detection
HUGEPAGES_COUNT=<COUNT> ./after_reboot.sh
```

To persist those host-side settings for future boots through the same helper:

```bash
cd applications/usrp_wideband_signal_detection
PERSIST_BOOT_CONFIG=1 HUGEPAGES_COUNT=<COUNT> ./after_reboot.sh
```

To make the basic host setup survive reboot, persist the pieces that are normally lost:

```bash
# Load GPUDirect peer memory at boot.
echo nvidia-peermem | sudo tee /etc/modules-load.d/nvidia-peermem.conf

# Persist the hugepage count. Replace <COUNT> with the value that is known-good on your host.
# Do not leave the literal placeholder in the file.
printf 'vm.nr_hugepages=<COUNT>\n' | sudo tee /etc/sysctl.d/90-holohub-usrp.conf
sudo sysctl --system

# Persist the hugetlbfs mount. This example matches the 1G hugepages used on the current host.
echo 'nodev /dev/hugepages hugetlbfs defaults,pagesize=1G,mode=1777 0 0' | sudo tee -a /etc/fstab
sudo mkdir -p /dev/hugepages
sudo mount /dev/hugepages
```
```bash
sudo xhost +local:root

sudo docker exec -it -u 0:0 -e DISPLAY="$DISPLAY" -e XAUTHORITY="$XAUTHORITY" -e PS1='container#' usrp_x410_signal_detection_demo bash

```
Recommended post-reboot verification:

```bash
lsmod | grep nvidia_peermem
grep -E 'HugePages_Total|HugePages_Free|Hugepagesize' /proc/meminfo
mount | grep hugetlbfs
ls -ld /dev/hugepages
```

If DPDK still fails after reboot with mlx5 `DevX create TIS failed`, `TIS allocation failure`, or `Failed to get port number for sdr_data`, the helper already resets the dedicated Mellanox ports by default. The equivalent manual host-side sequence is:

```bash
sudo ip link set ens4f0np0 down
sudo ip link set ens4f1np1 down
sudo devlink dev reload pci/0000:a2:00.0
sudo devlink dev reload pci/0000:a2:00.1
```

Then clear stale DPDK runtime state inside the demo container:

```bash
sudo docker exec -u 0:0 usrp_x410_signal_detection_demo bash -lc '
pkill -f "(^|/)usrp_wideband_signal_detection( |$)" || true
rm -rf /tmp/xdg-runtime-*/dpdk/*
rm -f /dev/hugepages/nwlrbbmqbh*
'
```

That host reload plus container cleanup was the working recovery path for the reboot-induced mlx5 bring-up failure on this setup.

If visualization fails after reboot with `Failed to initialize glfw`, the problem is separate from DPDK: the container was created without usable display forwarding for the current desktop session. In that case, recreate the demo container from the session you want to render through:

```bash
cd applications/usrp_wideband_signal_detection
SKIP_IMAGE_BUILD=1 sudo -E ./build_demo_container.sh
./run_demo_container.sh
./enter_demo_container.sh
```

This repository now assumes `ens4f0np0` and `ens4f1np1` are dedicated to this workflow on this machine, so `after_reboot.sh` performs that reset by default. If you need to skip it on a different host, run `RESET_MLX_PORTS=0 ./after_reboot.sh`.

### After Code Changes

When you change code or config files in the repository:

Use the helper scripts below rather than a host-local CMake configure/build. The rebuild happens inside the demo container and is the supported path for the live app.

```bash
cd applications/usrp_wideband_signal_detection
./rebuild_demo_container_app.sh
./enter_demo_container.sh
```

Then rerun the app inside the container:

```bash
cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
./usrp_wideband_signal_detection config_coherent_power_performance_emit_stride1_two_channel.yaml
```

For the single-channel rerun path instead:

```bash
cd /workspace/holohub/build/usrp_wideband_signal_detection/applications/usrp_wideband_signal_detection
./usrp_wideband_signal_detection config_coherent_power_performance_single_channel.yaml
```

For the two-channel performance pass with the real TorchScript detector path and debug outputs disabled:

```bash
cd applications/usrp_wideband_signal_detection
./run_torchscript_performance_test.sh
```

For the one-channel performance pass:

```bash
cd applications/usrp_wideband_signal_detection
./run_torchscript_performance_single_channel.sh
```

Use the sender with only channel 0 and only UDP destination port `1234` when exercising the one-channel config.

For the two-channel coherent-power real-time path with the coherent detector selected and debug outputs disabled:

```bash
cd applications/usrp_wideband_signal_detection
./run_coherent_power_performance.sh
```

To quantify the three remaining DINO optimization levers with the same helper script:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=old_configs/config_torchscript_performance_timing_debug_fast_post.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=old_configs/config_torchscript_performance_timing_debug_small_input.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=old_configs/config_torchscript_performance_timing_debug_fp16.yaml ./run_torchscript_performance_test.sh
```

Compare these fields across runs:

- `torch_runtime_ms`
- `hybrid_post_ms`
- `total_ms`
- `RX out of buffers`
- `rx_mbuf_allocation_errors`
- `rx_q0_errors` and `rx_q1_errors`

To bisect the current live throughput bottleneck with matched configs, run these three in order:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=old_configs/config_torchscript_performance_fft_only_matched.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=old_configs/config_torchscript_performance_spectrogram_only.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=config_torchscript_performance.yaml ./run_torchscript_performance_test.sh
```

Interpretation:

- if `fft_only_matched` already drops badly, the problem is upstream of spectrogram and DINO
- if `fft_only_matched` is clean but `spectrogram_only` drops, spectrogram is the first unstable stage
- if both are clean but `config_torchscript_performance.yaml` drops, the remaining gap is in the fast DINO stage or graph scheduling around it

Validation and production parity rule: for both the coherent-power and DINO detectors, any config intended to validate or represent production mask behavior must use the same `backend_mode`. The approved throughput knob is `emit_stride`; logs, saves, and timing summaries may differ, but the mask-generation backend must not.

For a more conservative coherent real-time profile that prioritizes sustained ingest headroom while the coherent detector still contains host-side stages:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=old_configs/config_coherent_power_realtime_guarded.yaml ./run_coherent_power_performance.sh
```

To sample stage timings from the live notebook-faithful coherent reference path without running the detector on every FFT batch:

```bash
cd applications/usrp_wideband_signal_detection
./run_coherent_power_reference_timing.sh
```

That profile keeps `backend_mode: "reference"`, enables `timing_summary_enable`, and raises coherent `emit_stride` to `32` so the timing synchronizations are sparse enough to inspect the live path without completely collapsing throughput.

To sample stage timings from the live single-channel DINO reference path with the same container launch flow:

```bash
cd applications/usrp_wideband_signal_detection
./run_torchscript_live_timing_single_channel.sh
```

That profile keeps the current single-channel live DINO settings, enables `timing_summary_enable`, and raises DINO `emit_stride` to `32` so the forced timing synchronizations are sparse enough to inspect the GPU path without completely swamping ingest. Look for these log lines:

- `DINO hybrid timing summary` for `input_ms`, `power_db_ms`, `frontend_ms`, `coherence_ms`, `torch_runtime_ms`, `hybrid_post_ms`, `mask_save_ms`, and `total_ms`
- `DINO service timing` for `frontend`, `crop_align`, `resize`, `model_prep`, `torch_forward`, `dino_score`, `fusion`, plus the reference chunk stages `host_copy`, `host_frontend`, `chunk_plan`, `chunk_upload`, `score_project`, `coherence_hybrid`, `chunk_group`, and `global_merge`

To run the coherent detector with an explicit config through the same sudo/docker flow:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=config_coherent_power_validation.yaml ./run_coherent_power_performance.sh
```

To capture frozen coherent-power validation artifacts for notebook and offline replay:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=old_configs/config_coherent_power_debug_capture.yaml ./run_coherent_power_performance.sh
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

### Coherent-Power Per-Frequency Floor Modes

The coherent detector's per-frequency fill (OR-ed into the fast mask; fires where `corrected_db` exceeds a per-row noise floor by `per_freq_threshold_offset_db`) can source that per-row floor three ways, selected in the `coherent_power_signal_detector` config block via `per_freq_threshold_mode`:

- `"calibrated"` — load the static per-row floor `.npy` produced by the offline calibration flow (`calibrate_coherent_power_config.sh`, written to `calibration/coherent_power_per_freq_floor.npy` and referenced by `per_freq_threshold_path`). Best when the live noise environment matches the capture set you calibrated against; recalibrate when the front end, gain, or band changes. See the header of [calibrate_coherent_power_config.sh](calibrate_coherent_power_config.sh) for the full calibration procedure and policy knobs.
- `"dynamic"` — learn the per-row floor live, no calibration run required. Each bin starts at a high bar (`dynamic_floor_init_db`) and only descends toward the quietest power it observes, self-calibrating to the current noise floor. An always-on signal never presents a quiet frame, so its bin stays high and is ignored by design. The floor re-seeds to the high bar on app reset and on a center-frequency change. Ready-made config: `config_coherent_power_perf_dynamic_single_channel.yaml`.
- `"static"` — disable the per-frequency fill entirely; only the global `fast_power_floor_db` / `fast_score_threshold` path runs.
- (empty) — legacy behavior: derived from `per_freq_threshold_enable` (`true` → calibrated, `false` → static), so existing configs are unchanged.

Dynamic-mode tuning knobs (config block):

| Key | Default | Effect |
| --- | --- | --- |
| `dynamic_floor_init_db` | `40.0` | High starting bar per bin, in dB; each bin only ever descends from here. |
| `dynamic_floor_std_k` | `2.0` | Per-frame per-row statistic = `mean + k*std` of `corrected_db`; `k` approximates the noise high-quantile the offline calibration measures. Raise to reduce false positives. |
| `dynamic_floor_window_slots` | `8` | Number of sub-window minima kept per bin. The floor is the min across all slots; stale lows age out once every slot rotates, bounding the slow downward creep a pure global minimum would accumulate over a long run. |
| `dynamic_floor_slot_frames` | `16` | Frames each slot accumulates before the cursor rotates. Effective sliding window = `window_slots * slot_frames` frames (default `8*16 = 128`). Shorter window = more responsive and less creep but a noisier floor; longer = smoother and closer to the calibrated floor but slower to adapt. |
| `dynamic_floor_warmup_frames` | `0` | Frames to accumulate before the learned floor feeds the fill. The high init bar already keeps early frames conservative, so `0` is usually fine. |

`per_freq_threshold_offset_db` still sets the firing margin above the floor in every mode, and in dynamic mode it also absorbs any residual bias in the learned floor.

For the staged bottleneck-isolation passes, reuse the same helper with `CONFIG_NAME`:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=old_configs/config_torchscript_performance_fft_only.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=old_configs/config_torchscript_performance_spectrogram_only.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=old_configs/config_torchscript_performance_small_batches.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=old_configs/config_torchscript_realtime_guarded.yaml ./run_torchscript_performance_test.sh
CONFIG_NAME=old_configs/config_torchscript_performance_timing_debug.yaml ./run_torchscript_performance_test.sh
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

Set `visualization.enable: true` in `old_configs/config.yaml` to turn on the live spectrogram branch.

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
	- use a specific replay config file, defaulting to `old_configs/config_offline_replay.yaml`
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

If you run inside the demo container and see `Failed to initialize glfw` or `Failed to detect any supported platform`, the container was created without usable host display forwarding. Restarting that same container is not enough because `docker start` cannot add `DISPLAY`, `XAUTHORITY`, or X11 mounts after creation.

Recreate the container from the desktop-capable session you actually want to render through, including FastX sessions. From that session, confirm `DISPLAY` is set, then run `SKIP_IMAGE_BUILD=1 sudo -E ./build_demo_container.sh`. The updated launcher forwards `DISPLAY` whenever it is present, mounts `/tmp/.X11-unix` when that socket directory exists on the host, and forwards `XAUTHORITY` when the file exists. After recreation, `./run_demo_container.sh` will refuse to start a stale container if it was created without the current display settings.

The next visualization step is to add a detector overlay postprocessor that emits HoloViz overlay tensors and `InputSpec` metadata.

## Validation Notes

- `old_configs/config.yaml` is now the stable debug run configuration. It intentionally keeps `inference_backend: "pytorch_placeholder"` while saving the first 5 spectrograms and detector masks per channel.
- Parity rule: validation, performance bring-up, and any config intended to represent production mask behavior must keep the same detector `backend_mode` for mask creation. The validated live path is `backend_mode: "reference"`; throughput tuning should come from cadence and pipeline controls such as `emit_stride`, not from a separate backend.
- `old_configs/config_coherent_power_debug_capture.yaml` is the coherent-power frozen-input capture profile. It enables tensor snapshot saves, optional `power_db` snapshot saves, and final mask saves so notebook and offline C++ parity checks can run on the exact same detector input.
- `old_configs/config_cuda_fallback.yaml` is the debug configuration for the pure C++/CUDA detector path. It disables the PyTorch backend in operator logic and uses `cuda_threshold_fallback` while keeping artifact saves enabled.
- `old_configs/config_torchscript_validation.yaml` is the strict TorchScript bring-up configuration. Use it when you want the C++ TorchScript path to fail loudly.
- `config_torchscript_performance.yaml` is the low-overhead throughput configuration for two-channel rate testing. It now stays on the validated `reference` backend while using `emit_stride` and the lean graph path for throughput work.
- `old_configs/config_torchscript_load_only.yaml` is the first diagnostic step for the C++ TorchScript path. It confirms whether `torch::jit::load(...)` itself is safe before the operator attempts CUDA transfer.
- `old_configs/config_torchscript_cpu_eval.yaml` is the second diagnostic step. It tests whether `eval()` is safe while staying entirely on CPU.
- The CPU validation flow should use the CPU-exported artifact `dinov3_vitb16_pretrain_lvd1689m-73cec8be_cpu.ts`; the original `dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts` remains the CUDA-traced artifact.
- `old_configs/config_torchscript_cuda_no_eval.yaml` is the third diagnostic step. It tests whether `to(torch::kCUDA)` is safe before `eval()` runs.
- Because the current executable is linked against libtorch when Torch is available at build time, the Torch runtime libraries still need to be present in the container even when you launch `old_configs/config_cuda_fallback.yaml`.
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

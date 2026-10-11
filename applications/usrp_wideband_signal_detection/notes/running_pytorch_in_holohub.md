# Running PyTorch In Holohub

This guide captures the working method and the main lessons learned while bringing a TorchScript-backed DINOv3 operator into a Holohub containerized application.

The specific integration target during this work was `applications/usrp_wideband_signal_detection`, but the build and runtime notes below are broadly useful for any Holohub app that needs C++ libtorch plus Python torch inside a container.

## Scope

This guide covers:

- packaging a local PyTorch model repo and weights into a Holohub container,
- exporting a TorchScript artifact inside the runtime container,
- getting CMake to discover Python-installed torch for a C++ operator build,
- resolving CUDA and `nvJitLink` linker issues,
- practical runtime checks, and
- the main failure modes observed during bring-up.

## Known-good high-level method

The method that worked best was:

1. Build a dedicated Holohub container image from an application Dockerfile that already carries the required network and GPU-side dependencies.
2. Start that container as a long-lived detached runtime container.
3. Stage the local model repository and selected weights into a canonical container runtime path.
4. Install a PyTorch build that matches the container CUDA stack and the host driver.
5. Export the TorchScript artifact inside the container, not on the host.
6. Build the Holohub application inside the same running container using `./holohub build ... --local`.
7. Validate TorchScript loading separately in Python before trusting the C++ operator path.
8. Only then attempt full application runtime bring-up.

That sequence avoided most of the host-versus-container mismatches that caused earlier failures.

## Container workflow

For this integration, the working helper scripts live under:

- `applications/usrp_wideband_signal_detection/build_demo_container.sh`
- `applications/usrp_wideband_signal_detection/run_demo_container.sh`
- `applications/usrp_wideband_signal_detection/setup_demo_container.sh`
- `applications/usrp_wideband_signal_detection/enter_demo_container.sh`

The important behavior encoded in those scripts is:

- build from `applications/usrp_freq_detection/Dockerfile` to inherit MatX and USRP-facing dependencies,
- run the container detached so follow-on setup can target a stable container name,
- export `HOLOHUB_BUILD_LOCAL=1` so Holohub builds run locally inside the container instead of trying nested Docker,
- mount GPU and device resources correctly, and
- stage model assets into `/workspace/models/dinov3`.

## Model staging pattern

The working source-of-truth split was:

- host repo: `/home/sat3737/holoscan_demo_workspace/dinov3`
- host weights: `/home/sat3737/holoscan_demo_workspace/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`
- container repo target: `/workspace/models/dinov3`
- container TorchScript target: `/workspace/models/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts`

The main lesson here is simple: treat the container path as the only runtime truth. Do not point the app at host paths, and do not export TorchScript on the host if the app will run in the container.

## PyTorch installation lessons

### 1. Python torch alone is not enough for C++ builds

The operator build initially failed to enable the Torch path because `find_package(Torch)` could not find a standalone libtorch install, even though Python `torch` was installed.

The fix was to resolve the CMake prefix from Python and prepend it before calling `find_package(Torch)`:

- see `operators/dinov3_signal_detector/CMakeLists.txt`

This pattern is reliable when you are using a Python-installed torch wheel inside the container.

### 2. CUDA version alignment mattered more than anything else

An earlier container torch install used a CUDA 13.0 wheel. That left `torch.cuda.is_available()` false on this machine even though the container had GPU access and `nvidia-smi` worked.

The working pinned stack was:

- `torch==2.10.0`
- `torchvision==0.25.0`
- `PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`

The practical lesson is:

- first validate `nvidia-smi`,
- then validate `torch.cuda.is_available()`,
- and only after both pass should you attempt TorchScript export or C++ model-forward bring-up.

### 3. Export TorchScript inside the same runtime container

The export helper added for this work is:

- `applications/usrp_wideband_signal_detection/export_dinov3_torchscript.py`

That helper runs against the container-staged repo and weights and writes the runtime artifact into the same container tree used by the Holohub app.

This eliminated host/container ABI and dependency drift during export.

## Holohub build lessons

### 1. Build locally inside the container

Running plain `./holohub build ...` inside the container can trigger a nested Docker path. The correct pattern for this workflow was:

- `export HOLOHUB_BUILD_LOCAL=1`
- `./holohub build <app> --local`

This should be treated as mandatory for in-container app compilation.

### 2. Pass the actual MatX CMake path explicitly

The build needed:

- `-Dmatx_DIR=/usr/local/lib/cmake/matx`

This mattered even when MatX was already installed in the image.

### 3. Prefer a container image that already has DPDK and related USRP dependencies

An early image variant was missing pieces that the wideband app needed. Reusing the `usrp_freq_detection` Dockerfile lineage avoided repeated dependency drift.

## CUDA `nvJitLink` lessons

This was the main build-system trap.

### What went wrong

Once the operator was built with Torch enabled, the final executable link started failing on `libcusparse.so` dependencies that required `libnvJitLink.so.12`.

Several assumptions turned out to be wrong:

- the container did not expose `libnvJitLink.so.12` under the expected `/usr/local/cuda/...` paths,
- adding `-lnvJitLink` was not enough because there was no usable unversioned `libnvJitLink.so` in the expected search path,
- CMake's `CUDA_nvJitLink_LIBRARY` resolution was not reliable in this environment.

### What worked

The actual library location inside this container was discovered with:

- `find /usr /usr/local -name 'libnvJitLink.so*' 2>/dev/null`

The result showed the usable library under the Python NVIDIA package tree:

- `/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12`

The application CMake was then changed to:

1. ask Python where `nvidia.nvjitlink` is installed,
2. append that directory to the link search paths, and
3. resolve `libnvJitLink.so.12` from the real container path.

See:

- `applications/usrp_wideband_signal_detection/CMakeLists.txt`

The lesson is that Python-packaged NVIDIA libraries may be the real source of CUDA-side link dependencies inside the container, even when `/usr/local/cuda` exists.

## Runtime lessons

### 1. Python TorchScript validation is worth doing before C++ runtime bring-up

The following check succeeded inside the container:

- import torch
- `torch.jit.load(...)`
- move module to CUDA
- call `eval()`

That proved the TorchScript artifact and Python-side libtorch runtime were healthy before continuing with the C++ operator.

This is a very good diagnostic split:

- if Python fails, fix packaging or runtime dependencies first,
- if Python passes and C++ fails, investigate the operator implementation or C++ libtorch usage.

### 2. The first C++ crash happened during operator initialization

After the build succeeded, the runtime moved further and then crashed during `DinoV3SignalDetector::initialize()` after logging that the PyTorch backend was enabled.

The Python TorchScript check still succeeded, which strongly suggested the crash lived in the C++ integration path rather than in the model artifact itself.

To reduce risk in the C++ path, the operator was changed from storing a TorchScript module by value to storing it behind `std::unique_ptr<torch::jit::script::Module>`.

See:

- `operators/dinov3_signal_detector/dinov3_signal_detector.hpp`
- `operators/dinov3_signal_detector/dinov3_signal_detector.cu`

The working hypothesis was that the value-assignment path in the embedded libtorch usage was causing the crash.

### 3. Current status of runtime validation

At the time this guide was written:

- container build was succeeding,
- Python TorchScript load and move-to-CUDA were succeeding,
- the C++ operator had been patched to store the TorchScript module behind a pointer,
- but full end-to-end runtime validation after that last patch had not yet been completely re-confirmed in this document.

So the build method is proven, the packaging method is proven, and the runtime debug path is narrowed substantially, but the final operator runtime should still be treated as an active validation step.

## Reduced-risk radio test method

Because the active radio was not the usual one and should not be reflashed casually, the reduced-risk runtime test path used:

- lower rate operation,
- the adapter identifier accepted by the current UHD setup, and
- the existing FPGA image rather than changing radio firmware.

Practical lessons from that side of the bring-up:

- use `sfp0` instead of `sfp1` for the current host/UHD setup,
- if `uhd_usrp_probe` reports `fpga=X4_200`, do not expect full `491.52e6` operation,
- use `245.76e6` as the safer test rate on that configuration.

See also:

- `USRP_X410_Holoscan_FFT_README.txt`

## Recommended checklist for future PyTorch-in-Holohub work

1. Start from a container image lineage that already has the network and math dependencies your app needs.
2. Verify `nvidia-smi` in the container before doing anything else.
3. Verify `torch.cuda.is_available()` in the container before doing anything else with models.
4. Use Python to derive Torch CMake paths instead of assuming a standalone libtorch layout.
5. Export TorchScript inside the same runtime container that will build and run the app.
6. Build with `./holohub build ... --local` inside the container.
7. Resolve CUDA-side dependencies from real container paths, including Python-packaged NVIDIA libraries when necessary.
8. Validate TorchScript loading in Python before blaming the model when the C++ operator crashes.
9. Reduce radio load and channel count before first end-to-end runtime validation.
10. Treat spectrogram saving and other debug-only paths as disabled during performance or model-forward bring-up.

## Files changed during this effort

The most relevant files for this workflow are:

- `applications/usrp_wideband_signal_detection/README.md`
- `applications/usrp_wideband_signal_detection/CMakeLists.txt`
- `applications/usrp_wideband_signal_detection/config.yaml`
- `applications/usrp_wideband_signal_detection/build_demo_container.sh`
- `applications/usrp_wideband_signal_detection/run_demo_container.sh`
- `applications/usrp_wideband_signal_detection/setup_demo_container.sh`
- `applications/usrp_wideband_signal_detection/enter_demo_container.sh`
- `applications/usrp_wideband_signal_detection/export_dinov3_torchscript.py`
- `operators/dinov3_signal_detector/CMakeLists.txt`
- `operators/dinov3_signal_detector/dinov3_signal_detector.hpp`
- `operators/dinov3_signal_detector/dinov3_signal_detector.cu`
- `USRP_X410_Holoscan_FFT_README.txt`

## Bottom line

The biggest lessons were:

- keep the whole PyTorch lifecycle inside one container,
- trust Python to tell CMake where torch and related NVIDIA libs actually live,
- expect CUDA auxiliary libraries like `nvJitLink` to come from unexpected package paths,
- and split debugging into packaging, build, Python runtime, and C++ runtime phases so each failure can be isolated quickly.
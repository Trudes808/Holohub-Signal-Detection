# Build Fixes

This document tracks build and runtime issues encountered while bringing up the visualization path for `usrp_wideband_signal_detection`.

## Resolved

### 1. Incorrect CUDA runtime target name

- Symptom:
  - CMake configure failed because `CUDAToolkit::cudart` was not defined in this environment.
- Fix:
  - Switched the app target linkage to `CUDA::cudart`.
- Status:
  - Resolved.

### 2. Old prototype viewer source left in the app directory

- Symptom:
  - The obsolete offline prototype remained alongside the new replay harness and caused confusion during iteration.
- Fix:
  - Removed the old `offline_spectrogram_visualizer.cpp` prototype from the app directory and kept the new replay harness source.
- Status:
  - Resolved.

### 3. Shared visualization code compiled as C++ instead of CUDA

- Symptom:
  - MatX-heavy visualization code failed with template and `cuda::std::max`-related errors when built as a `.cpp` file.
- Fix:
  - Moved the shared implementation to `spectrogram_visualization.cu` so it is compiled by `nvcc`.
- Status:
  - Resolved.

### 4. `OutputContext::emit(...)` rejected temporary `gxf::Entity` objects

- Symptom:
  - Build failed because `emit()` expected an lvalue reference, not a temporary entity.
- Fix:
  - Stored created entities in local variables before calling `emit()`.
- Status:
  - Resolved.

### 5. Visualization screenshot path did not support headless workflows

- Symptom:
  - HoloViz could not be used from a headless remote session, blocking iteration on visuals.
- Fix:
  - Added a headless screenshot path to `offline_spectrogram_visualizer` using HoloViz render-buffer output and PNG export.
- Status:
  - Resolved in code. Requires rebuild and runtime verification.

### 6. Vulkan runtime missing in container

- Symptom:
  - `offline_spectrogram_visualizer` failed to launch with `libvulkan.so.1: cannot open shared object file`.
- Fix:
  - Added script-side installation of `libvulkan1` in container setup and rebuild flows.
- Status:
  - Resolved for setup/rebuild automation. Not yet baked permanently into the image.

### 7. Container display forwarding missing for HoloViz window mode

- Symptom:
  - Runtime failed with GLFW platform detection errors.
- Fix:
  - Updated `run_demo_container.sh` to forward `DISPLAY`, `/tmp/.X11-unix`, and `XAUTHORITY` when launched from a desktop-capable session.
- Status:
  - Resolved in launcher logic. Still depends on the actual host session having a usable display backend.

### 8. `dinov3_signal_detector.cu` had compile-breaking TorchScript diagnostics structure

- Symptom:
  - The detector hit compile errors after TorchScript diagnostics were added because the `try`/`catch` flow and `#ifdef HOLOHUB_HAS_TORCH` structure in `dinov3_signal_detector.cu` had drifted into an invalid arrangement.
- Fix:
  - Restructured the Torch-enabled branch so the exception handling and preprocessor boundaries were well-formed again.
  - Kept the CUDA fallback path intact while isolating Torch-specific logic.
- Status:
  - Resolved.

### 9. Pip-installed LibTorch was not discoverable reliably from CMake

- Symptom:
  - `find_package(Torch)` was unreliable in the demo container because Torch came from the pinned Python package install rather than a traditional standalone LibTorch layout.
- Fix:
  - Added Python-driven Torch CMake hint discovery in `operators/dinov3_signal_detector/CMakeLists.txt` using the active package location and `torch.utils.cmake_prefix_path`.
  - Prepended the discovered Torch hint directories to `CMAKE_PREFIX_PATH` before calling `find_package(Torch)`.
- Status:
  - Resolved in CMake logic. Still worth verifying against the actual generated compile commands whenever the container Torch stack changes.

### 10. Torch-linked targets were missing robust `nvJitLink` resolution in the container

- Symptom:
  - Torch-linked builds were brittle because `nvJitLink` was not consistently resolved from the CUDA install and Python-packaged NVIDIA libraries used in the container.
- Fix:
  - Added explicit `CUDA_LINK_DIRS` search paths for standard CUDA locations.
  - Added discovery of the Python `nvidia.nvjitlink` package directory and appended it when present.
  - Wired `nvJitLink` into target link libraries and `rpath-link` handling for `dinov3_signal_detector` and the standalone sandbox target.
- Status:
  - Resolved in build-system logic.

### 11. Added standalone `dinov3_libtorch_sandbox` target for build/runtime isolation

- Symptom:
  - It was difficult to tell whether failures came from LibTorch itself, TorchScript artifacts, threading, or the NVCC-compiled operator translation unit.
- Fix:
  - Added `operators/dinov3_signal_detector/dinov3_libtorch_sandbox.cpp` as a standalone executable target.
  - Linked it with the same Torch and `nvJitLink` configuration as the detector library so CPU and CUDA TorchScript behavior could be validated independently.
- Status:
  - Resolved and useful for future bring-up/debugging.

### 12. `HOLOHUB_HAS_TORCH` caused an operator class layout mismatch across translation units

- Symptom:
  - After the pure C++ Torch runtime refactor, the app segfaulted during `DinoV3SignalDetector::initialize()` with a backtrace through `DinoTorchRuntime::~DinoTorchRuntime()`.
  - The likely cause was that `DinoV3SignalDetector` had a different private-member layout depending on whether `HOLOHUB_HAS_TORCH` was defined for a given translation unit.
- Fix:
  - Removed the conditional compilation around the `torch_runtime_` member in `dinov3_signal_detector.hpp` so the operator class layout stays identical for the app and the detector library.
- Status:
  - Resolved in source. Requires rebuild validation in the container after header changes.

### 13. `rebuild_and_debug.sh` stopped building `offline_spectrogram_visualizer`

- Symptom:
  - The build output directory no longer contained `./offline_spectrogram_visualizer`, even though the target still existed in CMake.
  - Running the documented screenshot command failed with `No such file or directory`.
- Fix:
  - Updated `rebuild_and_debug.sh` to always build both `usrp_wideband_signal_detection` and `offline_spectrogram_visualizer`.
  - This applies to both the Torch-enabled path and the Torch-skipped visualization-only path.
- Status:
  - Resolved in script. Requires one rebuild to repopulate the missing binary.

### 14. Headless screenshot mode still depended on HoloViz Vulkan startup

- Symptom:
  - `offline_spectrogram_visualizer --screenshot ...` still aborted before rendering because HoloViz tried to create a Vulkan instance and the container Vulkan ICD was missing `VK_KHR_external_memory_capabilities`.
- Fix:
  - Changed `--screenshot` to bypass HoloViz entirely and export the first replayed `.pgm` frame directly to PNG.
  - Windowed replay still uses HoloViz; screenshot export no longer depends on GLFW or Vulkan.
- Status:
  - Resolved in source. Requires rebuild validation in the container.

## Current Unresolved Issue

### Torch headers not reaching `dinov3_signal_detector`

- Symptom:
  - `rebuild_and_debug.sh` currently fails while compiling:

  ```text
  operators/dinov3_signal_detector/dinov3_torch_runtime.cpp:17:10:
  fatal error: c10/cuda/CUDAGuard.h: No such file or directory
  ```

- Current evidence:
  - `find_package(Torch)` reports Torch as found.
  - The actual compile command for `dinov3_torch_runtime.cpp` still does not include any Torch include directories.
  - This means Torch detection is succeeding for linkage metadata, but the detector target is still missing header include propagation in the build actually used by `rebuild_and_debug.sh`.

- Current working hypothesis:
  - The rebuild script’s generated build tree is not receiving the expected Torch include directories from the pip-installed Torch package layout.
  - We may need one of the following:
    - a stronger `target_include_directories(...)` fix in `operators/dinov3_signal_detector/CMakeLists.txt`
    - direct verification of the Torch include paths inside the running container
    - a fallback to disable the Torch path in the rebuild/debug flow when those headers are absent

- Next debugging step:
  - Inspect the actual Torch include directories available inside the container and compare them to the generated compile command used for `dinov3_torch_runtime.cpp`.

### Updated finding: container currently has no Python Torch installed

- Evidence collected inside the running container:

  ```json
  {
    "error": "No module named 'torch'"
  }
  ```

- Interpretation:
  - The current container is not provisioned with the Python Torch package that the DINO Torch runtime path expects.
  - That means the earlier `c10/cuda/CUDAGuard.h` failure is not only an include-propagation issue. It is also a container provisioning issue for the Torch-enabled detector/debug path.

- Practical impact:
  - Visualization work should not be blocked on Torch bring-up.
  - Rebuild/debug flows should either:
    - install the pinned Torch stack first via `setup_demo_container.sh`, or
    - skip Torch-dependent targets and validations when Torch is absent.

- Additional source-level fix applied:
  - `operators/dinov3_signal_detector/CMakeLists.txt` now excludes `dinov3_torch_runtime.cpp` from the detector library when `Torch_FOUND` is false.
  - This allows the CUDA fallback detector path to compile in Torch-less containers instead of failing on a missing Torch header translation unit.
  - Added `operators/dinov3_signal_detector/dinov3_torch_runtime_stub.cpp` so the detector still links cleanly when the real Torch runtime source is excluded.

- Status:
  - Partially addressed for visualization-focused rebuilds. Torch-enabled validation still requires the actual Torch package to be installed in the container.

## Runtime Notes

### HoloViz window mode

- Works only when the container is launched from a session with a real display backend.
- PuTTY X11 forwarding is not currently a reliable runtime path for GLFW/HoloViz.

### Headless export mode

- Intended fallback command after rebuild:

  ```bash
  ./offline_spectrogram_visualizer --offline-dir /workspace/spectrograms --screenshot offline_preview.png
  ```

- Expected host output path:

  ```text
  /tmp/usrp_spectrograms/offline_preview.png
  ```
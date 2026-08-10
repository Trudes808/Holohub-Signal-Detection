# DGX Spark (GB10 / ARM64) port notes

Record of changes made to run `usrp_wideband_signal_detection` live on this box
(`spark-4512`, NVIDIA DGX Spark: aarch64, GB10 = Blackwell **sm_121 / compute 12.1**,
driver 580 / CUDA 13). The app was originally written for an x86_64 bench (NIC at PCIe
`0000:a2:00.0`, CUDA 12.6, GPUDirect via nvidia-peermem). Started 2026-08-10.

## Summary of what changed vs the original bench

| Area | Original (x86 bench) | This Spark | Where changed |
| --- | --- | --- | --- |
| NIC / DPDK bind | `0000:a2:00.0` (`ens4f0np0`) | `enp1s0f0np0` / `0000:01:00.0` (ConnectX-7) | `bash_scripts/after_reboot.sh` (done); all top-level `config*.yaml` (done) |
| GPUDirect | `nvidia-peermem` | **dmabuf** (peermem won't load here) | `bash_scripts/after_reboot.sh` peermem = best-effort |
| Hugepages | 1 GB (runtime) | 1 GB via GRUB (`default_hugepagesz=1G hugepagesz=1G hugepages=8`) | `/etc/default/grub` (backup `grub.bak.holohub`); **needs reboot** |
| Container CUDA | 12.6 | **13** (GB10 needs sm_121; nvcc 12.6 tops out at compute_90) | `applications/usrp_freq_detection/Dockerfile` (Phase 3, pending) |

## Phase 0 — host prep (DONE)
- **nvidia-peermem**: cannot load — inbox `ib_core` lacks the legacy peer-memory API
  (`/sys/kernel/mm/memory_peers` absent). Normal on Grace/Spark; GPUDirect uses **dmabuf**.
  `after_reboot.sh:ensure_nvidia_peermem_loaded` now warns + continues (was `die`).
  TODO at live bring-up: confirm the container's DPDK/DOCA registers GPU memory via dmabuf.
- **1 GB hugepages**: `/etc/default/grub` `GRUB_CMDLINE_LINUX_DEFAULT` had `hugepages=2048`
  (2 MB); replaced with `default_hugepagesz=1G hugepagesz=1G hugepages=8`. `update-grub` run,
  verified in `/boot/grub/grub.cfg`. **Requires a reboot to activate** (deferred so it does not
  interrupt the build). Isolated CPUs are 5,7 (`isolcpus`); the app configs pin DPDK cores 9–11
  — revisit for live latency tuning.
- **after_reboot.sh retargeted**: `MLX_PORTS=enp1s0f0np0`, `MLX_DEVICES=pci/0000:01:00.0`,
  `MLX_PCI_FUNCTIONS=0000:01:00.0`, `DEFAULT_HUGEPAGES_COUNT=8`.

## Phase 1 — container image (DONE, then being rebuilt on CUDA 13)
- First build via `holohub build-container ... --docker-file applications/usrp_freq_detection/Dockerfile`
  produced an **aarch64** image with DPDK 22.11 + DOCA + MatX + Holoscan 3.4 C++ SDK — but on
  **CUDA 12.6**, whose nvcc cannot target GB10 (sm_121). holohub itself detected `CUDA_MAJOR=13`
  / `COMPUTE_CAPACITY=12.1` and offered a `holoscan:v3.7.0-cuda13` base, but the Dockerfile
  hardcoded `FROM nvcr.io/nvidia/cuda:12.6.3-base-ubuntu22.04` and ignored it.
- **Decision (user): rebuild on CUDA 13.** Plan = base the Dockerfile on
  `nvcr.io/nvidia/clara-holoscan/holoscan:v3.7.0-cuda13` (bundles CUDA 13 + Holoscan 3.7 +
  CUDA dev libs) and add only DOCA + MatX, instead of hand-bumping `libcublas-dev-12-6`→`-13-0`,
  the holoscan apt version, and the DOCA/OS pins. (Details filled in during Phase 3.)

## DINOv3 weights (for the cuda_dino detector only; not needed to build or to run coherent_power)
- Repo cloned: `git clone https://github.com/facebookresearch/dinov3 ~/Documents/dinov3` (done).
- Weights are **license-gated by Meta** — cannot be scripted without accepting the license:
  1. Open <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>, accept terms.
  2. Meta emails signed URLs for all checkpoints. Find the **ViT-B/16 pretrained on LVD-1689M**
     (filename `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`).
  3. Download with `wget` (README says do NOT use a browser) to the exact path the build expects:
     ```
     wget -O ~/Documents/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth "<SIGNED_URL>"
     ```
- NOTE: the Hugging Face repo `facebook/dinov3-vitb16-pretrain-lvd1689m` hosts `model.safetensors`
  (transformers format) — **not** the original torch.hub `.pth`, so it is NOT a drop-in for
  `export_dinov3_torchscript.py` / `build_demo_container.sh`. Use the Meta form path above.

## Reboot checklist (do this before live ingest, after the app build + offline validation)
1. Reboot (activates 1 GB hugepages).
2. `cd applications/usrp_wideband_signal_detection && sudo ./bash_scripts/after_reboot.sh`
3. Cable X410 to `enp1s0f0np0`; run sender with `--dest-mac` = that port's MAC.

## Build status (2026-08-10)
- **CUDA-13 image DONE + verified** via new `Dockerfile.spark_cuda13` (FROM
  `holoscan:v3.7.0-cuda13` + DOCA 3.1 dev + MatX). Verified inside: aarch64 / Ubuntu 24.04, nvcc
  13.0 knows **sm_121**, DPDK 22.11, DOCA 3.1.0105, MatX, Holoscan 3.7 C++ SDK. Tagged
  `usrp_x410_signal_detection_demo:{cuda13,latest}` (14.6 GB). DOCA gotcha fixed: base ships DOCA
  3.1.0105 runtime but pins the 3.0.0 apt repo → the Dockerfile realigns it to `doca/3.1.0`.
- **Docker nvidia runtime**: the daemon had only `runc` (no `nvidia` runtime, no daemon.json), so
  the scripts' `--runtime nvidia` failed with `unknown or invalid runtime name: nvidia` (exit 125).
  Fixed once with `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart
  docker` (writes /etc/docker/daemon.json; persists across reboot). `--gpus all` alone also works.
- **PyTorch resolved**: `torch==2.10.0+cu130` (up to 2.13) exists for aarch64 at
  `download.pytorch.org/whl/cu130` — native CUDA 13, no PTX-JIT gamble. Install via that index
  (torchvision to match); replaces the stock cu126 pin in `build_demo_container.sh`.
- **app CMakeLists.txt ported** (additive): added `aarch64-linux-gnu` lib dirs, `sbsa-linux` +
  `cuda-13.0` CUDA link dirs, `nvJitLink.so.13`, and CUDA arch `70;80;90` → `90;121` (CUDA 13
  dropped sm_70; 121 = GB10).
- **DINOv3 weights VALIDATED**: present at the expected path, 327 MiB, valid zip torch checkpoint
  (`data.pkl` + tensor storages, internal name `dinov3_vitb16_pretrain_lvd1689m`). Definitive
  `torch.load` key-check pending once torch is in the container.
- **TorchScript export OK**: `dinov3_vitb16_pretrain_lvd1689m-73cec8be.ts` (343 MB) created
  in-container — torch 2.10.0+cu130 loaded the weights on the GB10 GPU and traced the model, so the
  weights are definitively valid and torch runs on GB10.
- **DOCA gpunetio dev fix**: advanced_network's gpunetio manager needs pkg-config `doca-gpunetio`,
  which `Requires: doca-verbs` → added `libdoca-sdk-verbs-dev` (+ `libdoca-sdk-rdma-dev`) to
  `Dockerfile.spark_cuda13`. Now `doca-gpunetio`/`doca-verbs` resolve (3.1.0105).
- **Docker nvidia runtime fix**: registered once (see above) — was the exit-125 cause.
- **Container created + torch installed** via adapted `build_demo_container.sh`
  (`SKIP_IMAGE_BUILD=1 CAPTURES_HOST_DIR=/tmp/usrp_captures`). Script edits: `PYTORCH_INDEX_URL`
  → cu130; torch cuda check `12.6`→`13`; `ensure_nvjitlink_symlink` handles `.so.13` + sbsa/cuda-13.
- **config DPDK bind retargeted**: all top-level `config*.yaml` `0000:a2:00.{0,1}` → `0000:01:00.0`.
- **Torch compute_20 fix**: Torch/Caffe2 CUDA autodetect misparses GB10 capability 12.1 →
  injects bogus `-gencode arch=compute_20,code=sm_121` (nvcc fatal). Fixed by
  `set(TORCH_CUDA_ARCH_LIST "9.0;12.1")` before `find_package(Torch)` in the app CMakeLists.
- **Operator CUDA-13 fixes**: 6 app-dependency operators (fft, spectrogram,
  coherent_power_signal_detector, mask_replay_detector, cuda_dino_detector, dinov3_signal_detector)
  pinned `70;80;90` (invalid `compute_70` on CUDA 13) → `90;121`. `TORCH_CUDA_ARCH_LIST=9.0;12.1`
  exported at build time so all Torch targets emit valid gencode (Torch autodetect otherwise emits
  compute_20/compute_50). The **gpunetio** manager targets the DOCA 2.x API and does NOT compile
  against this base's DOCA 3.1 (`doca_flow_fwd.rss_queues` gone, `doca_gpu_dev_eth_rxq_get_buf`
  now returns void, etc.) → build **dpdk-only** via `-DANO_MGR=dpdk` (configs use `manager: dpdk`;
  the DPDK manager still does GPUDirect). All these flags are baked into `build_demo_container.sh`.
  If the gpunetio backend is ever needed, its DOCA API usage must be ported 2.x→3.1 (separate job).
- **App build DONE**: all 4 binaries built for sm_90/sm_121 (`usrp_wideband_signal_detection`,
  `run_offline_cuda_detector_eval`, `offline_spectrogram_visualizer`, `offline_cuda_dino_operator_replay`).
- **Phase 4 offline validation DONE — GB10 runtime confirmed** on `comprehensive_ordered.sigmf-data`
  (245.76 MHz, 286 frames). Both detectors ran with no CUDA errors and produced detections on
  286/286 frames (coherent_power dynamic-floor mean mask coverage ~0.42; cuda_dino ~0.34, running
  the exported TorchScript DINOv3 on the GB10 GPU). This validates run+detect, **not** detection
  *quality* — threshold/calibration tuning is a downstream step, best done with live X410 data
  (the committed calibration `.npy`s were fit on the x86 bench). Artifacts under
  `/tmp/usrp_spectrograms/offline_{coherent_dynamic,cuda_dino}`.
- **Realtime visualization ready**: live configs have `visualization.enable: true` (HoloViz
  spectrum-analyzer: PSD strip + max-hold, waterfall with detector-mask overlay, readouts, sliders).
  Container display is wired: `DISPLAY=:2`, `/tmp/.X11-unix` mounted, driver caps include
  `graphics,display`, Vulkan ICD at `/etc/vulkan/icd.d/nvidia_icd.json` + NVIDIA GL/Vulkan libs +
  GB10 visible. HoloViz bundles GLFW (no system libglfw needed). Headless dashboard preview
  verified via `offline_spectrogram_visualizer --screenshot`
  (`/tmp/usrp_spectrograms/offline_coherent_dynamic/dashboard_preview.png`). Live-window caveat:
  container root must be allowed on the host X server (`xhost +local:root`, done by after_reboot.sh);
  if it prints `Failed to initialize glfw`, recreate the container from the target desktop session.
  **PROVEN (2026-08-10)**: live HoloViz window opened on the `:2` desktop — Vulkan selected the
  GB10, created an xcb X11 surface + swapchain, HoloViz accepted the input spec, replayed 303
  frames and exited 0. Command: `offline_spectrogram_visualizer --config
  old_configs/config_offline_replay.yaml --offline-dir <out>/spectrograms --mask-dir
  <out>/mask_previews` (after `xhost +local:root`). NOTE: that tool needs a config WITH an
  `offline_replay:` block (has `tensor_name`); a live config crashes it with
  `holoviz - Failed to retrieve input ''`. The live app itself uses `ops::SpectrogramToHolovizOp`
  (main.cpp), a separate correctly-wired path, not the offline_replay tool.
- **Display-number hardening (2026-08-10)**: the X display number can change across reboots/logins
  (this box went `:2`→`:1` after the reboot), and `docker start` can't retrofit a container's baked
  DISPLAY, so `run_demo_container.sh` used to hard-fail and demand a recreate. Relaxed
  `require_current_display_forwarding` to only REQUIRE the `/tmp/.X11-unix` bind mount (which is a
  whole-dir mount and always exposes the current session's socket) and just WARN on a baked-DISPLAY
  mismatch. The run wrappers (`run_torchscript_performance_test.sh`, and everything delegating to it)
  already pass `-e DISPLAY=$DISPLAY` / `-e XAUTHORITY` at `docker exec`, so the app targets the live
  display. Net: a reboot that changes the display no longer needs a container recreate (with
  after_reboot.sh's `xhost +local:root`). One-time recreate was still done this session to align the
  baked value.

## Live ingest — GPUDirect RDMA blocker (2026-08-10)
Live app gets through DPDK EAL + mlx5 NIC probe (`0000:01:00.0`, MAC 4C:BB:47:2C:45:13) + GPU RX
buffer alloc (`CH1_Data_RX_GPU`, 528 MB), then FAILS registering that buffer with the NIC:
`mlx5_common: Fail to create MR ... unable to DMA map ... Could not DMA map EXT memory: EINVAL`.
- **Cause**: the DPDK manager (`operators/advanced_network/advanced_network/managers/dpdk/adv_network_dpdk_mgr.cpp:478`)
  maps GPU memory via `rte_dev_dma_map` → mlx5 `ibv_reg_mr` on the GPU VA, which needs a kernel
  **peer-memory provider (nvidia-peermem)**. peermem can't load here (inbox `ib_core` lacks the
  peer-memory client API). The manager does NOT implement the modern **dmabuf** path.
- **Also fixed en route**: a stale `/etc/fstab` entry mounted `/dev/hugepages` as 2M despite the 1G
  `default_hugepagesz` → corrected to `pagesize=1G,mode=1777` (backup `/etc/fstab.bak.holohub`),
  so DPDK now gets past hugepages.
- **Platform CAN do dmabuf GPUDirect with no host change**: container rdma-core (MLNX 2507) exposes
  `ibv_reg_dmabuf_mr`, kernel 6.11 `mlx5_ib` supports dmabuf, nvidia-open 580 exports dma-buf.
- **Options (all avoid CPU staging)**: (A) patch the DPDK manager to register GPU memory via dmabuf
  (`cuMemGetHandleForAddressRange` DMA_BUF_FD + `ibv_reg_dmabuf_mr`); (B) port the DOCA GPUNetIO
  manager to the DOCA 3.1 API and run `manager: gpunetio` (NVIDIA's supported GPU-networking path);
  (C) install DOCA-Host / MLNX_OFED kernel modules to get a peer-memory `ib_core` + nvidia-peermem
  (no app change; host-stack change; module-compat risk on the 6.11.0-1014-nvidia kernel).

## RESOLUTION: GB10 is an INTEGRATED (Tegra-class) GPU — classic GPUDirect RDMA does not apply (2026-08-10)
`cuDeviceGetAttribute` on "NVIDIA GB10" (driver reports it as "NVIDIA Tegra NVIDIA GB10"):
`INTEGRATED=1`, `VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1`, **`GPU_DIRECT_RDMA_SUPPORTED=0`**,
**`GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED=0`**, **`DMA_BUF_SUPPORTED=0`**. `cuMemAlloc` memory is not
dmabuf-exportable (`cuMemGetHandleForAddressRange` → EINVAL) and `cuMemCreate(gpuDirectRDMACapable=1)`
→ "invalid device ordinal".
- **The GB10 has no discrete VRAM/PCIe BAR** — CPU and GPU share coherent LPDDR5X. So NIC-DMA-into-
  *device* memory (the app's `kind: device` region, which fails with EINVAL) **cannot work by ANY
  path** here: dmabuf (unsupported), peermem (nothing to map), or DOCA GPUNetIO (same HW limit).
  This is architectural, not a software gap — so the earlier "dmabuf patch / GPUNetIO / peermem"
  options are all moot for device-memory GPUDirect on this SoC.
- **Correct GB10 model**: NIC DMAs into HOST memory (`kind: huge` / `HOST_PINNED`) — a normal MR,
  no peermem/dmabuf — and the GPU reads that SAME coherent memory. This is NOT the discrete-GPU
  "CPU staging" penalty (no PCIe copy). On the coherent SoC it's the native, near-zero-cost path
  (shared LPDDR5X ~273 GB/s vs a ~8 Gb/s single / ~32 Gb/s dual stream). Key implementation point:
  make the downstream operators access the RX buffer zero-copy (pinned/coherent) rather than an
  explicit host→device `cudaMemcpy`.
- **Only remaining: Phase 5 live bring-up** (needs hardware) — see the Reboot checklist above.

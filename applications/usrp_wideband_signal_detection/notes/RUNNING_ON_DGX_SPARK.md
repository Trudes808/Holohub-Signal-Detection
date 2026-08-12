# Running the USRP Wideband Signal Detection app on DGX Spark (AArch64 / GB10)

**Audience:** anyone bringing this app up on an NVIDIA DGX Spark (or validating the AArch64 port).
This is the usage guide; the engineering history and rationale live in
[`dgx_spark_port.md`](dgx_spark_port.md) and
[`gb10_unified_memory_ingest.md`](gb10_unified_memory_ingest.md).

The app was originally written for an x86_64 bench (discrete GPU, GPUDirect RDMA via
nvidia-peermem, NIC at PCIe `0000:a2:00.0`, CUDA 12.6). This port runs it on the DGX Spark:
**aarch64, GB10 (integrated Blackwell, sm_121, unified CPU/GPU memory), CUDA 13, ConnectX-7.**

---

## 1. What is different from the x86 bench (summary)

| Area | x86 bench | DGX Spark (this port) |
| --- | --- | --- |
| Container image | `usrp_freq_detection/Dockerfile` (CUDA 12.6 base) | **`Dockerfile.spark_cuda13`** (Holoscan v3.7-cuda13 base + DOCA 3.1 dev + MatX; nvcc knows sm_121) |
| PyTorch | cu126 x86 wheels | **cu130 aarch64 wheels** (`torch==2.10.0+cu130`) |
| CUDA arch | `70;80;90` | **`90;121`** (CUDA 13 dropped sm_70; 121 = GB10) — app + the 6 detector/fft/spectrogram operators |
| GPUDirect | NIC DMAs into GPU VRAM (`kind: device`, needs nvidia-peermem) | **GB10 has no GPUDirect RDMA / dma-buf** (integrated GPU, no VRAM). NIC DMAs into **host hugepages** (`kind: "huge"`); the GPU reads the same memory coherently, zero-copy. This is the native path on a unified-memory SoC, not a staging penalty. |
| Hugepages | runtime-reserved | **8× 1 GB via GRUB** (`default_hugepagesz=1G hugepagesz=1G hugepages=8`) |
| NIC / DPDK bind | `0000:a2:00.0` | **`0000:01:00.0`** (`enp1s0f0np0`) — the data link |
| advanced_network managers | dpdk + gpunetio | **dpdk only** (`-DANO_MGR=dpdk`; the gpunetio manager targets DOCA 2.x APIs and does not compile against DOCA 3.1) |

**X410 ↔ host cabling (two QSFP cables — control and data cannot share a port,**
**because DPDK takes the whole port from the kernel):**

| Link | X410 side | Host side | Owner |
| --- | --- | --- | --- |
| Control (UHD/RFNoC) | `sfp1` = `192.168.21.2` | `enp1s0f1np1` (`0000:01:00.1`) = `192.168.21.1` | kernel |
| Data (CHDR → DPDK) | `sfp0` = `192.168.10.2` | `enp1s0f0np0` (`0000:01:00.0`) = `192.168.10.1` | DPDK |
| Mgmt (optional) | `eth0` = `192.168.30.3` | `enP7s7` = `192.168.30.1` | kernel |

Both RF channels stream out **sfp0** as two UDP flows (dst ports 1234/1235, both `udp_src 49153`);
the sender only supports one `--adapter`. The X410 is `root@192.168.21.2` (no password) if you need
to inspect its interfaces.

---

## 2. One-time setup

```bash
# 0. Prereqs already done on this box (for reference / a fresh Spark):
#    - GRUB: default_hugepagesz=1G hugepagesz=1G hugepages=8   (then reboot)
#    - /etc/fstab: /dev/hugepages hugetlbfs pagesize=1G,mode=1777
#    - Docker nvidia runtime: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
#    - DINOv3: git clone https://github.com/facebookresearch/dinov3 ~/Documents/dinov3
#      + place the gated weights at ~/Documents/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth
#      (Meta license form: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/ — use wget, not a browser)

# 1. Build the CUDA-13 aarch64 image (NOT the stock x86 Dockerfile):
cd <repo>/applications/usrp_wideband_signal_detection
sudo docker build --network host -f Dockerfile.spark_cuda13 -t usrp_x410_signal_detection_demo:latest <repo-root>

# 2. Create + provision the container (installs cu130 torch, exports the DINO TorchScript,
#    builds the app for sm_90/sm_121 with -DANO_MGR=dpdk — all baked into the script):
sudo env SKIP_IMAGE_BUILD=1 CAPTURES_HOST_DIR=/tmp/usrp_captures ./bash_scripts/build_demo_container.sh
```

Note: run step 2 from a desktop session (`DISPLAY` set) so the container gets display forwarding.
If the X display number later changes, no recreate is needed — the run wrappers pass the live
`DISPLAY` at exec time (`run_demo_container.sh` only warns).

## 3. After every reboot

```bash
cd <repo>/applications/usrp_wideband_signal_detection
sudo ./bash_scripts/after_reboot.sh
```

This mounts/permissions the 1 GB hugepages, resets the data NIC for DPDK, **enforces
`192.168.10.1/24` on `enp1s0f0np0`** (the host netplan carries stale addressing for that port),
grants X access, starts the container, and clears stale DPDK state. The `nvidia-peermem` warning is
expected — GB10 does not use peermem.

## 4. Live runs (two terminals)

**Terminal A — the app** (from the app dir; visualization opens on your desktop):

Pass the config as the **first argument** — a `CONFIG_NAME=...` env prefix does NOT survive
`sudo` (the wrapper then silently falls back to the cuda_dino default):

```bash
# single channel, coherent detector (dynamic floor):
sudo ./bash_scripts/run_torchscript_performance_test.sh config_coherent_power_perf_dynamic_single_channel.yaml

# DUAL channel, coherent detector (dual-panel viz):
sudo ./bash_scripts/run_torchscript_performance_test.sh config_coherent_power_performance_emit_stride1_two_channel.yaml

# single channel, DINOv3 detector (inference-bound; processes a subset of the stream):
sudo ./bash_scripts/run_torchscript_performance_test.sh config_cuda_dino_performance_single_channel.yaml
```

**Terminal B — the radio (over-the-air collection).** This step is NOT a synthetic/demo signal
source: it commands the X410 to tune its RF frontends, **receive over the air through the
antennas**, and stream the received IQ into the app. It is a separate process by design — the
Holoscan app is a pure DPDK consumer and cannot own the UHD/RFNoC control session (control rides
the kernel QSFP). The bench defaults are baked into a wrapper:

```bash
# dual channel (RF ports 0+1 @ 2400/1000 MHz), streams until Ctrl-C:
./bash_scripts/start_radio_stream.sh

# single channel:
CHANNELS="0" FREQS="2400e6" DEST_PORTS="1234" ./bash_scripts/start_radio_stream.sh

# other centers / gain / timed run:
FREQS="915e6 1800e6" GAIN=40 DURATION=60 ./bash_scripts/start_radio_stream.sh
```

(Equivalent raw command: `applications/usrp_freq_detection/rx_to_remote_udp.py --args
"addr=192.168.21.2" --freq <f0> [f1] --rate 500e6 --gain 30 --channels 0 [1] --adapter sfp0
--dest-addr 192.168.10.1 --dest-port 1234 [1235] --dest-mac-addr 4c:bb:47:2c:45:13 --spp 1024`.)

Notes:
- **The HoloViz window opens as soon as the app starts but stays blank/frozen until the radio
  stream starts** — the renderer only draws when frames arrive, so with no data GNOME may report
  the window "not responding" and it can appear transparent (showing the desktop behind it). This
  is the expected idle state, not a failure: start the sender (Terminal B) and the window comes
  alive within a couple of seconds.
- `--dest-mac-addr` is `enp1s0f0np0`'s MAC. This X410 negotiates **491.52 Msps** when 500e6 is
  requested — expected; the receiver derives FFT geometry from the actual rate.
- Control rides `addr=192.168.21.2` (sfp1/kernel); data exits `--adapter sfp0`. Don't swap these.
- Restarting the app: make sure the old instance fully exited first (DPDK holds a lock during
  shutdown), or the new one dies with `Cannot create lock ... Is another primary process running?`.

## 5. Offline (no radio) and validation

```bash
# offline eval, same operators as live (either detector):
sudo python3 run_cuda_dino_offline_file.py <capture.sigmf-data> \
  --detector coherent_power --config config_coherent_power_perf_dynamic_single_channel.yaml \
  --output-root /tmp/usrp_spectrograms/offline_test
```

Reference capture used to validate this port:
`~/Documents/holoscan_waveform_generation/composition/composites/comprehensive_ordered.sigmf-data`
(245.76 Msps composite with ground-truth annotations). `/tmp` outputs do not survive reboots.

## 6. Performance expectations on GB10 (measured)

| Scenario | Ingest | chdr→fft latency | Frame coverage |
| --- | --- | --- | --- |
| 1 channel, coherent | 491.52 Msps ingest | ~200 ms (batch 256) | ~80–85% (NIC micro-drop bursts + converter; continuous display, no blanking) |
| 2 channels, coherent | 2× 491.52 Msps ingest | ~435 ms (batch 512, 8 workers, emit_stride 2) | ~75%/ch, symmetric (GPU-contention ceiling — see shedding note) |
| 1 channel, cuda_dino | full wire rate; DINO throttles processing via backpressure valve | DINO-bound (~fft→preview 320 ms+) | subset (ViT inference cost) |

Knobs: `chdr_converter.num_ffts_per_batch` (= `fft.num_bursts`) trades latency vs converter load;
`scheduler.worker_thread_number: 8` needed for dual-channel symmetry; `render_every_n_frames`
decimates only the display (detection runs every emitted frame at `emit_stride: 1`).

**Dual-channel shedding (expected, not a malfunction).** Dual full rate (2× 491.52 Msps = 48k
FFT/s) is ~2× the GB10 pipeline ceiling, so roughly half the frames are shed; the waterfalls stay
live and detection runs on every processed frame. *Where* the excess is shed varies with batch
size and run-to-run scheduling:
- `num_ffts_per_batch: 512` (the committed default) sheds **quietly at batch assembly** — RX pools
  stay healthy, logs mostly calm (occasional `panic reset` self-heals, sporadic NIC-drop
  messages).
- Smaller batches (e.g. 256) shed via **output-queue backpressure**: queued batches pin the entire
  RX mempool, the NIC starves, and the log fills with `Fell behind in processing on GPU!` +
  `Dropped N packets since last poll` + `might get dropped` spam. Avoid for dual-channel.
- There is **no half-rate escape hatch on this X410**: the CG_400 FPGA image is fixed at
  491.52 Msps (requests for lower rates are refused). Single-channel runs are within the ceiling
  and clean.

**What the dual-channel bottleneck actually is (profiled 2026-08-12):** GPU contention, not
networking. Evidence: RX cores moved to the isolated CPUs (5,7) changed nothing; the converter's
out-queue sits pegged at its max (downstream won't consume); and the same detector kernel that
costs ~3.7 ms/frame single-channel costs ~14.5 ms/frame dual (spectrogram preview adds ~11 ms) —
per-kernel wall time inflates ~4× when both channels' converter+FFT+preview+detector kernels
contend for the integrated GPU. Config levers already applied: `emit_stride: 2` (detect every 2nd
frame, +30% throughput, −25% latency), `render_every_n_frames: 3`, 8 scheduler workers, RX on the
isolated cores. Going to ~100% dual coverage would need code-level work (batch both channels into
single kernel launches, CUDA graphs to cut launch overhead, fuse/trim the preview path) — or a
discrete GPU.

**DPDK core-pinning trap:** the EAL takes the *lowest* core in its `-l` list as the main lcore,
and RX workers cannot run there. `master_core` must therefore be numerically LOWER than every
queue `cpu_core` (the config uses master 3, workers on isolated cores 5,7) — otherwise one RX
worker silently fails to start and that queue receives nothing.

## 7. Troubleshooting quick table

| Symptom | Cause / fix |
| --- | --- |
| `Fail to create MR` / `unable to DMA map ... EINVAL` at startup | An RX data region has `kind: "device"` — must be `kind: "huge"` on GB10 |
| `EAL: FATAL: Cannot get hugepage information` | `/dev/hugepages` mounted with the wrong pagesize — rerun `after_reboot.sh` (fstab pins 1G) |
| `Cannot create lock ... Is another primary process running?` | Previous app instance still shutting down — wait for it to exit, clear `/dev/hugepages/nwlrbbmqbh*` |
| Sender: `No devices found for addr: 192.168.21.2` | The app's DPDK grabbed the **control** port — the config must bind `0000:01:00.0` (data), never `01:00.1` |
| App runs but `packets=0` | X410 not streaming, or sender used the wrong `--adapter`/dest (data must exit **sfp0** to `192.168.10.1` / MAC `...:45:13`) |
| Window frozen / GNOME "not responding" / window looks transparent (mirrors the desktop) | No data flowing yet — the renderer only draws when frames arrive. Start the radio stream; if it's already running, check the app log's `RX worker summary` lines for `packets=0` and fix the stream (see the row above) |
| Dual-channel: `Fell behind in processing on GPU!` spam + `Dropped N packets since last poll` + `might get dropped` warnings | Expected at dual full rate — the GB10 ceiling shedding (see the shedding note in §6). Ensure `num_ffts_per_batch: 512` (smaller batches make it much worse). Not a malfunction: the viz stays live and detection runs on all processed frames |
| Spectrogram periodically **zeroes out and restarts** mid-stream (looks like the USRP restarting) | It isn't the radio — `degraded_shutdown_on_rx_queue_warning_threshold: 3` made the CHDR converter fully resync the channel after every 3rd NIC micro-drop warning, flushing all state. Set it to `0` (the current configs' default); the partial-flush self-heal still covers real stream stalls |
| Dual-channel: **both panels labeled with the same center frequency** (data clearly differs) | The radio's stream-params sidecar carries a single (channel-0) center, which the run wrapper used to export as a global override. The wrapper now ignores it for multi-channel configs (per-channel labels come from `chdr_converter.channel_center_frequencies_hz`). Cosmetic only — each channel's RF tuning was always correct |
| `Failed to initialize glfw` | X access: `xhost +local:root` (after_reboot does this), or recreate the container from a desktop session |
| `modprobe nvidia-peermem ... Invalid argument` | Expected on GB10 — ignore (unified memory path is used instead) |
| `nvcc fatal : Unsupported gpu architecture 'compute_20'` during a rebuild | Torch's CUDA autodetect misparses GB10 capability 12.1. Fixed by `set(TORCH_CUDA_ARCH_LIST "9.0;12.1")` before every `find_package(Torch)` (app + cuda_dino_detector + dinov3_signal_detector CMakeLists) and exported by both build wrappers — if it reappears, a new `find_package(Torch)` call site is missing the pin |

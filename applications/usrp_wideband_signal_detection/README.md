# USRP Wideband Signal Detection

GPU-accelerated wideband RF signal-detection pipeline for the USRP X410, built on Holoscan.
It ingests high-rate IQ over DPDK, computes a live FFT/spectrogram, runs a signal detector,
and renders a spectrum-analyzer-style visualization.

```
chdrConverterOp → fftOp → (spectrogramOp) → detectorOp → visualization
```

Two detectors are supported, selected per config via `pipeline.detector_type`:

- **`coherent_power`** — CUDA coherent-power detector with a calibrated or live-learned
  per-frequency noise floor. Lowest latency; the default for live wideband sweeps.
- **`cuda_dino`** — CUDA DINOv3 feature detector (TorchScript runtime) with coherence-gate
  fusion, structure mask, and a positional template.

Visualization is enabled by default on all of the current live configs.

> This app is part of an RF/AI fork of HoloHub. Install and verify the base platform using the
> repository-root [README](../../README.md) first; this document covers everything specific to
> the signal-detection pipeline.

---

## Repository layout

All commands below are run **from this directory** (`applications/usrp_wideband_signal_detection/`)
as the working root; wrappers and helpers live in subfolders and are called with their path.

| Path | Contents |
| --- | --- |
| `config_*.yaml` (top level) | The **current** configs — see the table below. Only these are synced into the container build tree by the rebuild wrapper. |
| `bash_scripts/` | All container/build/run shell wrappers (`build`, `rebuild`, `run_*`, `enter`, `after_reboot`, and the shared `container_env.sh` / `container_repo_guard.sh`). |
| `calibration/` | Calibration configs, calibrate scripts (`calibrate_*.sh` + their `.py`), and the emitted `.npy` calibration artifacts. |
| `old_configs/` | Superseded / experimental configs kept for reference. Not synced automatically; runnable via the wrappers by passing the `old_configs/<name>.yaml` path. |
| `notes/` | Historical development notes and design plans. |
| `main.cpp`, `spectrogram_visualization.cu`, `run_offline_cuda_detector_eval.cpp`, … | App and offline-eval sources. Detectors themselves live in `operators/{coherent_power_signal_detector,cuda_dino_detector}`. |
| `infocom_evals/signal_detection_experiments/` | Offline evaluation notebook + harness and experiment records. |

### Current configs

| Config | Detector | Use |
| --- | --- | --- |
| `config_coherent_power_perf_perfreq_single_channel.yaml` | coherent_power | Live, calibrated per-frequency floor (`per_freq_threshold_mode: calibrated`). |
| `config_coherent_power_perf_dynamic_single_channel.yaml` | coherent_power | Live, live-learned dynamic floor (`per_freq_threshold_mode: dynamic`; no calibration run needed). |
| `config_cuda_dino_performance_single_channel.yaml` | cuda_dino | Live CUDA DINO detector. |
| `config_coherent_power_performance_single_channel_replay.yaml` | coherent_power | Cable-loopback replay of a captured SigMF file. |
| `config_coherent_power_capture_chdr_single_channel.yaml` | coherent_power | Capture a frozen CHDR/IQ snapshot for offline replay. |

Calibration configs live in `calibration/`; superseded configs live in `old_configs/`.

---

## Container build

The supported workflow is container-based (do **not** run a host-local CMake build for the live
app). The container name/image are centralized in `bash_scripts/container_env.sh` so every
wrapper targets the same container; override per-run by exporting `CONTAINER_NAME` / `IMAGE_NAME`.

**First-time setup** (creates the container, mounts this checkout, stages the DINOv3 runtime,
builds the app). The image is shared; pass `SKIP_IMAGE_BUILD=1` to reuse an existing one:

```bash
git clone https://github.com/facebookresearch/dinov3   # once, for the DINO TorchScript runtime
cd applications/usrp_wideband_signal_detection
sudo env SKIP_IMAGE_BUILD=1 ./bash_scripts/build_demo_container.sh
```

**After any code or config change** — rebuild inside the container:

```bash
cd applications/usrp_wideband_signal_detection
sudo ./bash_scripts/rebuild_demo_container_app.sh          # FORCE_REBUILD=1 to force
```

The rebuild wrapper syncs the top-level `config*.yaml` files into the build tree, verifies the
container is mounted from *this* checkout (it refuses to run against another clone), and builds
the app plus the offline-eval binaries. Open a shell with `sudo ./bash_scripts/enter_demo_container.sh`.

---

## Running live with a USRP X410

Visualization is on by default. Use two terminals.

**1. Start the radio stream** from `applications/usrp_freq_detection/` (host Python terminal;
single channel shown — add a second `--channels`/`--freq`/`--dest-port` for dual-channel). The
IPs/MAC/ports below are **bench-specific** — match them to your NIC and the app's receive flow
(see the alignment note under the run command):

```bash
python3.12 rx_to_remote_udp.py \
	--args "mgmt_addr=192.168.20.3,addr=192.168.100.3" \
	--freq 2400e6 --rate 500e6 --gain 30 --channels 0 \
	--dest-addr 192.168.10.51 --dest-port 1234 \
	--dest-mac-addr E0:9D:73:E0:5B:6B --spp 1024
```

No `--adapter` is needed: the current single-channel configs bind the NIC at PCIe
**`0000:a2:00.0`** (`advanced_network.interfaces[].address`), which is the port the X410's
**default** egress reaches — so the sender's default egress lands on the bound NIC. (`--adapter`
forces a specific X410 SFP; its identifier is UHD-version-specific — `sfp0` is rejected by
UHD 4.6 — and forcing the *other* SFP would require pointing the config at `0000:a2:00.1`
instead. If the app reports `Received packets: 0` / `Missed packets: 0` at shutdown, the sender
egressed to a port the bound NIC isn't cabled to.) On this bench the X410 negotiates
**245.76 Msps** even when `--rate 500e6` is requested; the receiver derives FFT geometry from the
actual CHDR sample rate, so that is expected.

**2. Run the pipeline** (from this directory). Launch the app *before* starting the transmitter
for the cleanest startup. The generic runner rebuilds if needed, syncs the config, and runs it:

```bash
# Coherent power, calibrated per-frequency floor:
CONFIG_NAME=config_coherent_power_perf_perfreq_single_channel.yaml sudo ./bash_scripts/run_torchscript_performance_test.sh

# Coherent power, dynamic (self-calibrating) floor:
CONFIG_NAME=config_coherent_power_perf_dynamic_single_channel.yaml sudo ./bash_scripts/run_torchscript_performance_test.sh

# CUDA DINO detector (this is the runner's default):
sudo ./bash_scripts/run_torchscript_performance_test.sh
```

Convenience wrappers exist: `sudo ./bash_scripts/run_coherent_power_performance.sh` runs the
calibrated coherent config directly. Any config (including `old_configs/<name>.yaml`) can be run
via `CONFIG_NAME=<path> sudo ./bash_scripts/run_torchscript_performance_test.sh`.

**Sender ↔ receiver alignment.** The app's `advanced_network` block binds a DPDK flow that
matches on UDP ports, so the sender's destination must line up with it (values from the current
single-channel configs):

- sender `--dest-port 1234` ↔ config flow `match.udp_dst: 1234`
- X410 channel-0 source port `49153` ↔ config flow `match.udp_src: 49153` (automatic for ch0)
- the DPDK NIC is bound at PCIe `0000:a2:00.0` (`interfaces[].address`) — the port the X410's
  default egress reaches; the sender's `--dest-mac-addr` / `--dest-addr` target that NIC
- sender `--dest-addr` is that NIC's host IP

For a second channel, use `--dest-port 1235` (→ `udp_dst: 1235`, X410 src `49154`). See
**Stream alignment** below for the FFT-geometry details.

---

## Running offline on captured files

Offline evaluation runs the **same** detector operators as the live app, driven by the batch
binary `run_offline_cuda_detector_eval` over a SigMF file — no radio required.

Single file, either detector:

```bash
cd applications/usrp_wideband_signal_detection
python3 run_cuda_dino_offline_file.py <capture.sigmf-data> \
	--detector cuda_dino \
	--config config_cuda_dino_performance_single_channel.yaml \
	--output-root /tmp/usrp_spectrograms/offline_cuda_dino/<run>
# swap --detector coherent_power --config config_coherent_power_perf_perfreq_single_channel.yaml
```

`CONTAINER_NAME` is honored by the driver; export it (or source `bash_scripts/container_env.sh`)
so it targets your container. For interactive sweeps and mask comparisons, use the notebook
[`infocom_evals/signal_detection_experiments/debugging_signal_detection_eval.ipynb`](infocom_evals/signal_detection_experiments/debugging_signal_detection_eval.ipynb),
which routes both detectors through the same batch binary. Detector masks and spectrogram
previews are written under the `--output-root` directory.

---

## Running via cable-loopback replay

To exercise the full live DPDK ingest path deterministically, replay a captured SigMF file over
a cable loopback into the X410 RX instead of receiving over the air:

1. Wire the loopback (TX/signal source → X410 RX) per your bench setup.
2. Replay the captured SigMF onto that path with your signal source at the same `--rate 500e6`.
3. Run the app live with the replay config:

```bash
cd applications/usrp_wideband_signal_detection
CONFIG_NAME=config_coherent_power_performance_single_channel_replay.yaml sudo ./bash_scripts/run_torchscript_performance_test.sh
```

To first **capture** a frozen snapshot for later replay, run live with
`config_coherent_power_capture_chdr_single_channel.yaml`; snapshots land under
`/tmp/usrp_spectrograms` (host).

---

## Calibration

Calibration is offline (reads captured SigMF from `generated_inputs/`, runs the detector with a
stats-dump config, and fits the calibration artifacts). Run the calibrate scripts **from this
directory**; they `cd` to the app root internally and source `bash_scripts/container_env.sh`, so
they target your container. Rebuild the container first so the stats-dump code path is compiled.

```bash
cd applications/usrp_wideband_signal_detection

# Coherent-power per-frequency floor → calibration/coherent_power_per_freq_floor.npy
sudo ./calibration/calibrate_coherent_power_config.sh

# CUDA DINO coherence per-frequency floor + gate threshold
sudo ./calibration/calibrate_dino_coherence_config.sh

# DINO positional (RoPE) noise template
sudo ./calibration/calibration_script.sh
```

The calibrated `.npy` artifacts and the emitted calibrated configs are written under
`calibration/`; the live `perfreq` config already points at
`calibration/coherent_power_per_freq_floor.npy`. See the header of each calibrate script for its
policy knobs. The `dynamic` coherent config needs no calibration run.

### Coherent-power per-frequency floor modes

The coherent detector's per-frequency fill (OR-ed into the fast mask; fires where `corrected_db`
exceeds a per-row floor by `per_freq_threshold_offset_db`) sources that floor via
`per_freq_threshold_mode` in the `coherent_power_signal_detector` config block:

- `"calibrated"` — load the static per-row floor `.npy` from
  `calibrate_coherent_power_config.sh` (config: `..._perf_perfreq_...`).
- `"dynamic"` — learn the floor live; each bin starts high (`dynamic_floor_init_db`) and only
  descends toward the quietest power it sees, so an always-on signal's bin stays high and is
  ignored by design. Re-seeds on app reset / center-frequency change (config: `..._perf_dynamic_...`).
- `"static"` — disable the per-frequency fill; only the global `fast_power_floor_db` /
  `fast_score_threshold` path runs.

Dynamic-mode knobs (config block): `dynamic_floor_init_db` (40.0), `dynamic_floor_std_k` (2.0),
`dynamic_floor_window_slots` (8), `dynamic_floor_slot_frames` (16, effective window =
slots×slot_frames frames), `dynamic_floor_warmup_frames` (0). `per_freq_threshold_offset_db` sets
the firing margin above the floor in every mode.

---

## Platform notes (GPUDirect)

This host has two verified GPUDirect constraints:

- `sudo ./operators/advanced_network/python/tune_system.py --check bar1-size` reports GPU BAR1
  size `256 MiB` (the advanced-network guide recommends `1 GiB`+ for GPUDirect-heavy workloads).
- `sudo ./operators/advanced_network/python/tune_system.py --check topo` reports a non-ideal
  GPU-to-NIC path (`SYS`, not `PIX`/`PXB` in `nvidia-smi topo -m`).

Practical guidance: treat `26624` GPU-RX buffers per channel as the current best known
two-channel operating point; single-channel configs use a larger single pool. Larger values can
fail during DPDK startup (`mlx5_common: Fail to create MR` / `Could not DMA map EXT memory`) —
this is GPUDirect DMA registration, not framebuffer exhaustion. Re-check BAR1 size and PCIe
topology if the machine is reconfigured.

---

## Stream alignment

The receive side derives its runtime FFT sizing from `chdr_converter.channel_sample_rates_hz`
when set consistently, otherwise from the local FFT config — so sender and receiver must match.
`master_clock_rate=500e6` in the USRP device args does **not** propagate into Holoscan metadata.

Keep these aligned for the current X410 flow:

- sender `rx_to_remote_udp.py --rate 500e6`
- receiver `fft.span: 500000000`, `fft.transform_points: 20480`
- effective reference bin width ≈ `24414 Hz`, so `fft.resolution: 24414` in the configs

Set `fft.override_fft_bin_size` to request an exact target bin width (still quantized to the CHDR
packet width, currently `num_complex_samples_per_packet: 1024`). If these values do not match,
saved spectrograms, detector metadata, and offline replay can all appear calibrated to the wrong
span.

---

## After reboot

If the host was rebooted since the last successful run, restore host-side prerequisites first:

```bash
cd applications/usrp_wideband_signal_detection
sudo ./bash_scripts/after_reboot.sh
```

That reloads `nvidia-peermem`, reserves/mounts hugepages, fixes `/dev/hugepages` permissions,
resets the dedicated Mellanox ports (`ens4f0np0`/`ens4f1np1`; skip with `RESET_MLX_PORTS=0`),
starts the container if needed, and clears stale DPDK runtime state. It defaults to `3` hugepages
if none are persisted — override with `HUGEPAGES_COUNT=<n>`, and add `PERSIST_BOOT_CONFIG=1` to
persist. Verify with:

```bash
lsmod | grep nvidia_peermem
grep -E 'HugePages_Total|HugePages_Free|Hugepagesize' /proc/meminfo
mount | grep hugetlbfs
```

If visualization later fails with `Failed to initialize glfw`, the container was created without
usable display forwarding for the current session — recreate it from the desktop session you want
to render through: `sudo env SKIP_IMAGE_BUILD=1 ./bash_scripts/build_demo_container.sh`.

---

## Visualization and offline spectrogram replay

The live app opens a HoloViz spectrum-analyzer window from `spectrogramOp` when
`visualization.enable: true` (default on the current configs): top PSD strip, max-hold trace,
analyzer spectrogram panel, side readouts, and color-limit sliders.

`offline_spectrogram_visualizer` replays saved `.pgm` spectrogram frames without a radio:

```bash
# from the container build directory
./offline_spectrogram_visualizer --offline-dir /tmp/usrp_spectrograms --mask-dir /tmp/usrp_dino_masks --fps 8
# headless screenshot instead of a window:
./offline_spectrogram_visualizer --offline-dir /workspace/spectrograms --screenshot offline_preview.png
```

The viewer opens a native HoloViz window on the machine running it (local desktop, remote
desktop, or a display-forwarded container); `--screenshot` is the no-desktop fallback and writes
the composed dashboard frame without HoloViz/GLFW/Vulkan.

Default host-side debug outputs: spectrograms in `/tmp/usrp_spectrograms`, DINO masks in
`/tmp/usrp_dino_masks`.

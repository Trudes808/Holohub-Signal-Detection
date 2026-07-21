# USRP Wideband Signal Detection

GPU-accelerated wideband RF signal-detection pipeline for the USRP X410, built on Holoscan.
It ingests high-rate IQ over DPDK, computes a live FFT/spectrogram, runs a signal detector,
and renders a spectrum-analyzer-style visualization.

```
chdrConverterOp → fftOp → (spectrogramOp) → detectorOp → visualization
```

Three detectors are supported, selected per config via `pipeline.detector_type`:

- **`coherent_power`** — CUDA coherent-power detector with a calibrated or live-learned
  per-frequency noise floor. Lowest latency; the default for live wideband sweeps.
- **`cuda_dino`** — CUDA DINOv3 feature detector (TorchScript runtime), **zero-shot** backbone
  with coherence-gate fusion, structure mask, and a positional template.
- **`cuda_dino_finetuned`** — CUDA **fine-tuned** DINOv3 segmenter (backbone + trained seg head).
  Emits a mask directly (`sigmoid(model) ≥ threshold`), no fusion stack. Trained at a fixed
  spectrogram geometry, so it taps raw IQ and runs its own dedicated FFT to reproduce that
  geometry live. See [Fine-tuned DINO detector](#fine-tuned-dino-detector-cuda_dino_finetuned).

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
| `calibration/` | Calibration configs, calibrate scripts (`calibrate_*.sh` + their `.py`), the region-extraction helpers (`extract_*_regions.py`), and the emitted `.npy` calibration artifacts. |
| `old_configs/` | Superseded / experimental configs kept for reference. Not synced automatically; runnable via the wrappers by passing the `old_configs/<name>.yaml` path. |
| `notes/` | Historical development notes and design plans. |
| `debug_scripts/` | Ad-hoc debug / analysis / plotting helpers. Not part of the build or the documented workflows; nothing in the app depends on them. |
| `run_cuda_dino_offline_file.py` | Offline-eval driver (documented below). `export_dinov3_torchscript.py` is the one-time DINO TorchScript export used during container setup. |
| `main.cpp`, `spectrogram_visualization.cu`, `run_offline_cuda_detector_eval.cpp`, `*.hpp`, … | App and offline-eval build sources. The detectors themselves live in `operators/{coherent_power_signal_detector,cuda_dino_detector}`. |
| `infocom_evals/signal_detection_experiments/` | Offline evaluation notebook + harness and experiment records. |

### Current configs

| Config | Detector | Use |
| --- | --- | --- |
| `config_coherent_power_perf_perfreq_single_channel.yaml` | coherent_power | Live, calibrated per-frequency floor (`per_freq_threshold_mode: calibrated`). |
| `config_coherent_power_perf_dynamic_single_channel.yaml` | coherent_power | Live, live-learned dynamic floor (`per_freq_threshold_mode: dynamic`; no calibration run needed). |
| `config_cuda_dino_performance_single_channel.yaml` | cuda_dino | Live CUDA DINO detector (zero-shot). |
| `config_cuda_dino_finetuned_performance_single_channel.yaml` | cuda_dino_finetuned | Live **fine-tuned** DINO segmenter. Geometry-matched dedicated FFT; see [Fine-tuned DINO detector](#fine-tuned-dino-detector-cuda_dino_finetuned). |
| `config_coherent_power_performance_single_channel_replay.yaml` | coherent_power | Cable-loopback replay of a captured SigMF file. |
| `config_coherent_power_capture_chdr_single_channel.yaml` | coherent_power | Capture a frozen CHDR/IQ snapshot for offline replay. |
| `config_coherent_power_performance_emit_stride1_two_channel.yaml` | coherent_power | Live **dual-channel** (two 500 Msps channels); binds NIC `0000:a2:00.1` with two flows (ch0→`udp_dst 1234`, ch1→`1235`). |
| `config_signal_snipper_single_channel.yaml` | coherent_power | Live **signal snipper**: cuts each detected signal out of the stream and writes it (or hands it to a downstream classifier). Dynamic floor, visualization off, `emit_stride: 1`. See [Signal snipper](#signal-snipper-cutting-signals-out-of-the-stream). |

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

# CUDA DINO detector, zero-shot (this is the runner's default):
sudo ./bash_scripts/run_torchscript_performance_test.sh

# CUDA fine-tuned DINO segmenter (requires the exported .ts + .meta.json in the container,
# see "Fine-tuned DINO detector" below):
CONFIG_NAME=config_cuda_dino_finetuned_performance_single_channel.yaml sudo ./bash_scripts/run_torchscript_performance_test.sh
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

## Fine-tuned DINO detector (`cuda_dino_finetuned`)

The `cuda_dino_finetuned` detector runs a **fine-tuned DINOv3 segmenter** (backbone + a trained
segmentation head) that outputs a signal/noise mask directly — no coherence/structure/fusion stack.
Offline it beats both the zero-shot `cuda_dino` and `coherent_power` detectors, especially at low
SNR (see `dino_fine_tuning/reports/`).

### Why this detector needs special handling

The network input is **architecturally fixed** at `tile_rows × nfft` (patch 16 → a fixed token
grid), and — more importantly — the fine-tune is tied to the **physical meaning of each pixel**:

| axis | trained value (shipped model) | set by |
| --- | --- | --- |
| frequency | **240 kHz/bin** | `sample_rate / nfft` = 245.76e6 / 1024 |
| time | **4.17 µs/row** | `nfft / sample_rate` = 1024 / 245.76e6 |

The live app's wide analysis FFT (~20480-pt) has a *different* per-pixel physics, so the detector
does **not** consume it. Instead it **taps the raw IQ** (like the signal snipper) and runs its own
**dedicated `nfft`-point FFT**, reproducing the trained geometry exactly. Because the USRP X410
delivers 245.76 MSps (see [Stream alignment](#stream-alignment)) — the same rate the model was
trained on — this is an exact match at the documented receive setup.

This means every checkpoint carries a **geometry contract**. The exporter writes it as a
`<model>.meta.json` sidecar next to the `.ts`:

```json
{ "nfft": 1024, "tile_rows": 256, "sample_rate_hz": 245760000.0,
  "bin_hz": 240000.0, "row_seconds": 4.167e-06,
  "db_vmin": -46.934, "db_vmax": 19.557, "threshold": 0.85 }
```

The operator config **must match that contract** (`nfft`, `tile_rows`, `db_vmin`, `db_vmax`,
`threshold`). "Use our model" vs. "retrain for a new setup" differ *only* in this contract.

### Quick run (shipped model)

1. **Export the checkpoint to TorchScript** (once). Use the repo's `.venv` (has torch + CUDA). The
   repo lives under `/home/sat3737` (mode `750`), so a non-owner shell (e.g. `lab-admin`) must run it
   under `sudo` and name the venv Python by **absolute path** (sudo ignores `PATH`). `--dinov3-repo`
   puts the DINOv3 package on `sys.path` (it is not installed in the venv):

   ```bash
   sudo /home/sat3737/holohub-dev/.venv/bin/python \
     /home/sat3737/holohub-dev/applications/usrp_wideband_signal_detection/export_dinov3_finetuned_torchscript.py \
     --ckpt   /home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/checkpoints/M2_ft/best.pt \
     --tile-rows 256 --nfft 1024 --sample-rate-hz 245760000 \
     --eval-meta /home/bqn82/Holohub-Signal-Detection/dino_fine_tuning/eval_out/M2_ft/eval_meta.json \
     --dinov3-repo /home/bqn82/dinov3 \
     --output /home/sat3737/holohub-dev/dino_fine_tuning/weights/finetuned_dino_m2.ts
   ```

   This writes `finetuned_dino_m2.ts` **and** `finetuned_dino_m2.meta.json`, and self-checks that the
   traced mask matches the eager model (IoU ≈ 1). (If your shell owns the repo, drop `sudo` and just
   use `.venv/bin/python`.)

2. **Place the `.ts` where the config points** (container path
   `/workspace/holohub/dino_fine_tuning/weights/finetuned_dino_m2.ts`), then rebuild + run:

   ```bash
   sudo ./bash_scripts/rebuild_demo_container_app.sh
   CONFIG_NAME=config_cuda_dino_finetuned_performance_single_channel.yaml \
     sudo ./bash_scripts/run_torchscript_performance_test.sh
   ```

The config field that selects the model is `finetuned_dino_detector.model_script_path`; the decision
threshold is `finetuned_dino_detector.threshold` (M2_ft = 0.85, M1_ft = 0.45).

### Fine-tuning process

The fine-tuning pipeline lives in `dino_fine_tuning/` (see its `README.md` and
`reports/pipeline.md`). It trains binary signal/noise segmentation on SigMF captures, on a single
spectrogram grid so training, GT, and inference all share one geometry.

**For our receive setup (245.76 MSps, 256×1024).** The shipped M1/M2 checkpoints were produced by
`dino_fine_tuning/scripts/run_full.sh` at `nfft=1024, frame_rows=256`. To reproduce and export:

```bash
cd dino_fine_tuning
bash scripts/run_full.sh                    # dataset -> models -> eval (resumable)
# then export as in "Quick run" above.
```

**For a different sample rate / spectrogram settings.** Decide with this tree:

1. **Same physics reachable?** If your receive rate lets `nfft` land near **240 kHz/bin** and
   **4.17 µs/row** simultaneously (i.e. the rate is close to 245.76 MSps, or a near-integer
   multiple/divisor), the **shipped model still applies** — just set `nfft` (and `sample_rate_hz` at
   export) so the contract matches, and set `finetuned_dino_detector.nfft` in the config. **No
   retrain.**
2. **Otherwise, retrain** at the deployment's native geometry:
   - Edit `dino_fine_tuning/configs/dataset.yaml`: set `nfft` and `frame_rows` (both multiples of 16)
     to your target grid; leave `db_vmin/db_vmax: null` so the calibration pass re-estimates the dB
     clip for the new geometry.
   - Rebuild the dataset + retrain: `bash dino_fine_tuning/scripts/run_full.sh`.
   - Export with the matching geometry:
     `python export_dinov3_finetuned_torchscript.py --ckpt <new best.pt> --tile-rows <frame_rows>
     --nfft <nfft> --sample-rate-hz <your_rate> --output <name>.ts`.
   - Point the config at the new `.ts` and copy `tile_rows`/`nfft`/`db_vmin`/`db_vmax`/`threshold`
     from the emitted `.meta.json`.

### Adjusting the config

Everything the operator needs is in the `finetuned_dino_detector:` block of
`config_cuda_dino_finetuned_performance_single_channel.yaml`:

| field | meaning |
| --- | --- |
| `tile_rows`, `nfft` | model input shape; **must** equal the checkpoint's `.meta.json` |
| `db_vmin`, `db_vmax` | global dB→[0,1] clip from the training dataset calibration |
| `threshold` | `sigmoid(logits) ≥ threshold` (M2_ft = 0.85, M1_ft = 0.45) |
| `model_script_path` | container path to the exported `.ts` |
| `torch_dtype` | `fp32` \| `fp16` |
| `emit_stride` | process every Nth IQ frame |

> **Target geometry change in progress:** this deployment is moving to a **512 time × 1024 freq**
> input (keeps 240 kHz/bin, doubles time context to ~2.13 ms/frame). That needs a **retrain** at
> `frame_rows: 512` (the 256×1024 checkpoints will not fit). Once trained, flip the config to
> `tile_rows: 512` and point `model_script_path` at the 512×1024 `.ts`.

---

## Running offline on captured files

Offline evaluation runs the **same** detector operators as the live app, driven by the batch
binary `run_offline_cuda_detector_eval` over a SigMF file — no radio required.

Single file, either detector:

```bash
cd applications/usrp_wideband_signal_detection
sudo python3 run_cuda_dino_offline_file.py <capture.sigmf-data> \
	--detector cuda_dino \
	--config config_cuda_dino_performance_single_channel.yaml \
	--output-root /tmp/usrp_spectrograms/offline_cuda_dino/<run>
# swap --detector coherent_power --config config_coherent_power_perf_perfreq_single_channel.yaml
```

`sudo` is needed (the driver runs the container via `docker` and stages inputs under
`/tmp/usrp_spectrograms`). The driver auto-resolves the container name from
`bash_scripts/container_env.sh`, so it targets your container even though `sudo` strips the
environment (override with `CONTAINER_NAME=... sudo ...` if needed). For interactive sweeps and
mask comparisons, use the notebook
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

## Signal snipper: cutting signals out of the stream

The **signal snipper** turns the detector mask into actual extracted signals. It clusters the mask
into per-signal boxes, cuts the corresponding IQ out of the wideband stream, and emits a batch of
self-describing signal cutouts that are either written to disk (SigMF) or consumed by a downstream
classifier. Two operators implement this (`signal_snipper` → `sigmf_file_sink`), added to the graph
when `pipeline.enable_signal_snipper: true`.

Run it (live; also works over cable-loopback replay or the offline path):

```bash
cd applications/usrp_wideband_signal_detection
sudo ./bash_scripts/run_torchscript_performance_test.sh config_signal_snipper_single_channel.yaml
```

Offline against a capture (the snipper + sink run in the offline eval graph too):

```bash
python3 run_cuda_dino_offline_file.py <file.sigmf-data> \
  --detector coherent_power --config config_signal_snipper_single_channel.yaml \
  --output-root /tmp/usrp_spectrograms/snip_test
```

Snipped files land under the `sigmf_file_sink.output_dir` (default `/workspace/spectrograms/snippets`
→ host `/tmp/usrp_spectrograms/snippets`).

### Modes (`signal_snipper:` config block)

- **`mode: time_only`** — keep the time regions that contain any signal (full bandwidth, full rate);
  each emitted snippet is a full-band IQ slice annotated with every signal's freq edges in that
  interval. Tosses the signal-free time.
- **`mode: frequency`** — additionally isolate each signal: digital down-convert to baseband,
  low-pass to the detected bandwidth, and decimate to the minimum rate that preserves it plus
  `oversample_percent`. Each signal becomes its own baseband IQ stream at its own (lower) rate.
- Clustering knobs: `min_box_pixels` (speckle filter), `merge_gap_rows` / `merge_gap_cols` (coalesce
  fragments of one signal). `emit_stride` on the detector controls how often snipping runs.

### Output data format

The sink writes **SigMF** (`cf32_le`, interleaved I/Q float32). `sigmf_file_sink.mode`:

- **`per_signal`** — one `.sigmf-data` + `.sigmf-meta` per snippet.
- **`pack`** (default) — accumulate `pack_frames` frames and write **one file per pack**. If every
  snippet in the pack shares one (rate, center) — e.g. `time_only` — it is a standard concatenated
  SigMF recording. Otherwise (frequency mode: every signal a different rate) it is a **variable-rate
  container**: all snippets concatenated into one `.sigmf-data`, with the global `core:sample_rate`
  as a reference clock and `wfgt:container: true` / `wfgt:layout: concatenated_variable_rate`. This
  is what makes `emit_stride: 1` sustainable (one file per pack, not one per signal).

Every annotation is self-describing, so a downstream ingestor can split a pack/container:

| Field | Meaning |
| --- | --- |
| `core:sample_start`, `core:sample_count` | the snippet's slice **within this file** |
| `wfgt:snippet_sample_rate` | the snippet's own (decimated) rate — **authoritative** in a container |
| `wfgt:center_frequency` | the snippet's RF center |
| `core:freq_lower_edge`, `core:freq_upper_edge` | detected band edges (RF Hz) |
| `wfgt:frame_number`, `wfgt:orig_sample_start`, `wfgt:orig_sample_count`, `wfgt:orig_sample_end` | provenance in the original full-rate stream (which chunks belong together / span frames) |

To read a container: iterate `annotations`, slice `[core:sample_start, +core:sample_count)` from the
`.sigmf-data`, and interpret each slice at its `wfgt:snippet_sample_rate` (fall back to the standard
single global `core:sample_rate` when `wfgt:container` is absent).

If disk can't keep the snippet byte rate the sink logs `OVERFLOW` and drops whole batches (reporting
signals / IQ samples / original-rate samples lost) rather than stalling the pipeline; raise
`emit_stride`, raise `max_queued_batches`, or point `output_dir` at faster storage (e.g. `/dev/shm`).

### Hooking up a classifier instead of the file sink

The snipper emits a `holoscan::ops::SnippetBatchMessage` (defined in `signal_snip_types.hpp`) on its
`snippets_out` port. Each `SignalSnippet` carries its IQ **on the GPU** (`device_iq`, a pooled
`shared_ptr<cuda::std::complex<float>>`) plus `n_iq`, `sample_rate_hz`, `center_freq_hz`, the
original-stream provenance, and its `annotations` — so a classifier consumes it **directly from GPU
memory, zero-copy**. Only the file sink ever copies to host.

To classify instead of (or alongside) writing files, add your operator and wire it to `snippets_out`
in `main.cpp` exactly like the sink:

```cpp
// in compose(), where sigmfFileSinkOps are created/wired:
auto classifier = make_operator<ops::MySignalClassifier>("signalClassifierOpCh0", from_config("signal_classifier"));
add_operator(classifier);
add_flow(signalSnipperOps[ch], classifier, {{"snippets_out", "in"}});   // fans out; can coexist with the sink
```

Your operator's input port is typed `spec.input<holoscan::ops::SnippetBatchMessage>("in")`; in
`compute`, iterate `batch.snippets` and run inference on each `snippet.device_iq` (already on the
device) using `snippet.n_iq` / `sample_rate_hz` / `center_freq_hz`. The port fans out, so a classifier
and the file sink can both consume the same batch (the pooled buffer refcounts and recycles when the
last consumer is done).

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

## Config reference

Each config is a Holoscan YAML with the blocks below. Only the knobs you'd actually tune are
listed; unlisted keys are structural. Detector blocks are read only for the active
`pipeline.detector_type`.

**`pipeline`** — `enable_spectrogram` / `enable_detector` (toggle stages); `detector_type`
(`coherent_power` | `cuda_dino`); `log_from_spectrogram` (route the throughput logger off the
spectrogram instead of FFT).

**`advanced_network.cfg`** (DPDK ingest) — `interfaces[].address` (PCIe address of the NIC to
bind, e.g. `0000:a2:00.0`); `interfaces[].rx.flows[].match.{udp_src,udp_dst}` (per-channel packet
filter — ch0 `udp_dst 1234`, ch1 `1235`); `queues` / `memory_regions` (one RX queue + GPU/CPU
buffer pools per channel).

**`chdr_converter`** — `num_channels` (active RF channels; must equal the detector's
`num_channels`); `channel_sample_rates_hz` / `channel_center_frequencies_hz` (per-channel rate/
freq — must match the sender); `num_complex_samples_per_packet`, `num_packets_per_fft`,
`num_ffts_per_batch`, `num_simul_batches` (ingest batching).

**`fft`** — `span` (analyzed bandwidth in Hz — match the sender rate); `transform_points` (FFT
size in bins); `resolution` (bin width ≈ `span / transform_points`); `reference_span_hz` /
`reference_fft_size` (reference pair the runtime uses to scale the live FFT); see **Stream
alignment**.

**`visualization.renderer`** (display only — **no effect on detection**) — `output_width`/
`output_height` + `holoviz.width`/`height` (window size); `db_floor` / `db_ceil` (spectrogram
dB-normalization range — lower ceiling = more contrast); `blue_limit` / `red_limit` (color-scale
clamps); `center_frequency_hz` (frequency-axis readout — keep matched to the sender `--freq`);
`refresh_hz` (UI redraw rate); `demo_title` / `demo_subtitle` (on-screen titles); `rows_per_frame`
/ `render_every_n_frames` (waterfall scroll cadence); `num_channels` (channels shown).

**`coherent_power_signal_detector`** —
- `fast_power_floor_db` / `fast_power_span_db` / `fast_score_threshold` — fast-path score
  normalization window and detection threshold (the primary sensitivity knobs).
- `fast_strong_rescue_enable` / `_excess_db` / `_min_time_bins` — OR strong, frequency-narrow
  signals back in after morphology.
- `per_freq_threshold_mode` (`calibrated` | `dynamic` | `static`) + `per_freq_threshold_offset_db`
  + `per_freq_threshold_path` — per-frequency noise-floor fill; `dynamic_floor_*` tune the
  live-learned mode. See **Coherent-power per-frequency floor modes** above.
- `live_emit_freq_persistence_window` / `_min_hits` — temporal persistence gating on the mask.
- `frontend_reference_q` / `_smooth_sigma` / `_max_boost_db` / `_signal_cap_db` — spectral
  front-end correction.
- `filter_detection_mask`, `emit_stride`, `ignore_sideband_hz` — mask post-filter, emit decimation,
  edge guard.

**`cuda_dino_detector`** —
- `coherence_band_threshold` (+ `_threshold_quantile`, `_open_time_px`, `_min_area_px`) — coherence-band gate.
- `dino_structure_open_len` / `dino_structure_threshold_quantile` — DINO structure mask.
- `raw_dino_positional_template_path` / `_strength` / `raw_dino_positional_deweight` — positional (RoPE) template suppression.
- `dino_coherence_gate_floor` / `_span_db` — coherence gate; `hybrid_fusion_mode`
  (`coherence_primary`), `coherence_primary_legacy_score` (`max`), `dino_contribution_strength` —
  fusion combine.
- `chunk_bandwidth_hz` / `chunk_overlap_hz`, `input_height`/`input_width`, `patch_size`,
  `max_tokens_per_inference`, `torch_dtype` — chunk tiling and model I/O.
- `frontend_correction_*` — spectral front-end correction (same idea as the coherent one).

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

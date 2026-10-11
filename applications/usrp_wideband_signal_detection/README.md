# USRP Wideband Signal Detection (Argus)

GPU-native wideband RF signal detection on NVIDIA Holoscan. A USRP X410 streams up to
2 × 491.52 MSps of IQ over 100 GbE; the pipeline ingests it with DPDK, computes a live wide
FFT / spectrogram, runs a detector that segments the regions of interest, cuts those regions out
of the raw IQ as SigMF, and renders a spectrum-analyzer dashboard.

```
X410 ──100GbE──▶ advanced_network (DPDK) ──▶ chdr_converter ──▶ fft ──▶ spectrogram ──▶ detector ──▶ mask
                                                   │                                        │
                                                   └──── raw IQ ────▶ signal_snipper ◀───────┘
                                                                          │
                                                   SigMF snippets ◀── sigmf_file_sink      visualization (HoloViz dashboard)
                                                   (optional external classifier daemon reads the snippets)
```

Three detectors are selectable per config (`pipeline.detector_type`):

| `detector_type` | What it is | When to use |
| --- | --- | --- |
| `coherent_power` | CUDA coherent-power detector with a live-learned (`dynamic`) or calibrated per-frequency noise floor. | Lowest latency, highest precision; default live detector. |
| `cuda_dino` | Zero-shot DINOv3 ViT-B/16 features + coherence-gate fusion (TorchScript). | Reference / comparison detector. |
| `cuda_dino_finetuned` | Fine-tuned DINOv3 segmenter, **M5-S (ViT-S)** deployed; per-frame at 491.52 MSps. | Best recall at low SNR; the "DINO" entry of the live demo. See [`dino_fine_tuning/`](../../dino_fine_tuning/README.md). |

**Validated bench:** NVIDIA DGX Spark (GB10, aarch64, CUDA 13) + USRP X410 (`CG_400` FPGA image,
491.52 MSps) + ConnectX-7. The original x86 bench (RTX 4000 Ada, CUDA 12.6, GPUDirect RDMA) still
builds; the deltas are listed in [`notes/RUNNING_ON_DGX_SPARK.md`](notes/RUNNING_ON_DGX_SPARK.md) §1.
All commands below run **from this directory**.

---

## 1. Install (one time)

**Host prerequisites**

- Docker with the NVIDIA runtime (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`).
- 1 GB hugepages: GRUB `default_hugepagesz=1G hugepagesz=1G hugepages=8`, reboot (the x86 bench used runtime 2 MB pages; `after_reboot.sh` handles both).
- UHD **master** (4.9.0 lacks needed features) with the X410 `CG_400` image, and the Python bindings for the radio sender (`applications/usrp_freq_detection/rx_to_remote_udp.py`). Two QSFP cables: X410 `sfp0` → DPDK data port, `sfp1` → kernel control port.
- `tcpreplay` only for the no-radio loopback demo.

**External model inputs** (not in git)

| Input | Where it must be | Used by |
| --- | --- | --- |
| [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) checkout + gated `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` under its `weights/` | `<repo>/../dinov3` (sibling of this repo; override `HOST_DINOV3_ROOT`) | container build (`cuda_dino`, exported to TorchScript in-container) |
| `finetuned_dino_m5s_491_bf16.ts` + `.meta.json` | `dino_fine_tuning/weights/` | every `cuda_dino_finetuned` config of the demo |
| `holoscan_waveform_generation` checkout with its `.venv-ml` (optional) | `~/Documents/holoscan_waveform_generation` (override `VENV_PY`, `PYCODEC_ROOT`) | the `v3`/`v5` classifier daemon and the loopback pcaps |

Older fine-tuned exports (`m4_491`, `m3_491`, `m2_dr`) are only needed by the legacy configs listed in §6.
The release_v2 signal-classifier weights used by the `v5` dashboard panels ship in the repo
(`infocom_evals/pycodec_e2e/amc/weights_v2/`); without a classifier the daemon falls back to a blind cascade.

**Build the image and create the container**

```bash
# DGX Spark (aarch64 / CUDA 13):
sudo docker build --network host -f Dockerfile.spark_cuda13 -t usrp_x410_signal_detection_demo:latest ../..
sudo env SKIP_IMAGE_BUILD=1 ./bash_scripts/build_demo_container.sh

# x86 bench (CUDA 12.6 image from applications/usrp_freq_detection/Dockerfile):
sudo ./bash_scripts/build_demo_container.sh
```

Run this from a desktop session (`DISPLAY` set) so the container gets display forwarding. The
script creates a long-lived container that bind-mounts this checkout at `/workspace/holohub`,
stages the DINOv3 runtime, installs the matching PyTorch, exports the DINOv3 TorchScript, and
builds the app plus the offline binaries. Container/image names live in
`bash_scripts/container_env.sh`; override per run with `sudo env CONTAINER_NAME=... <script>`.

**After any code or config change**

```bash
sudo ./bash_scripts/rebuild_demo_container_app.sh        # FORCE_REBUILD=1 to force
```

**After every reboot**

```bash
sudo ./bash_scripts/after_reboot.sh                      # hugepages, NIC reset, host IPs, X access, start container
```

---

## 2. Run the live demo

Cabling check first: `sudo ./bash_scripts/check_radio_topology.sh` confirms which host port is wired
to which X410 interface and sets the canonical IPs (`run_live_demo.sh` runs it automatically).

**One command** (starts the app, waits for DPDK to arm, starts the over-the-air radio stream,
mirrors the app log, Ctrl-C stops everything):

```bash
sudo ./bash_scripts/run_live_demo.sh v5         # single channel @ 2.4 GHz  (the current demo)
sudo ./bash_scripts/run_live_demo.sh v5dual     # dual channel: 2.4 GHz + 1.0 GHz, 491.52 MSps each
```

What you get: the HoloViz dashboard (PSD + max-hold, waterfall, detection overlay, throughput and
GPU readouts) with a **DEMO CONTROLS** panel. The demo starts on **Coherent Power**; pick
**DINO v3 M5-S** in the detector dropdown to switch (the conductor relaunches the app, ~15 s).
The `v5` presets also start the classify-only AMC daemon, which classifies the snipped signals
(release_v2 classes `qpsk / bfm / he / nr5g / ble / noise`) and feeds the classifier panels.

| Preset | Config | Detector switch | Classifier stack |
| --- | --- | --- | --- |
| `v5` / `v5dual` | `config_live_v3_single_channel.yaml` / `config_live_v3_two_channel.yaml` | Coherent Power ↔ DINO v3 M5-S | release_v2 (6 classes) |
| `single` / `dual` | `config_coherent_power_perf_dynamic_single_channel.yaml` / `config_coherent_power_performance_emit_stride1_two_channel.yaml` | none (coherent only) | none, minimal footprint |
| `v3` `v3dual` `v4` `v4dual` `v3m5s` `v3m5sdual` | the v3 configs | as `v5` | legacy v1 classifier (kept for reference) |
| `<config.yaml>` | any top-level config | none | none |

Radio knobs are environment variables (use `sudo env VAR=... ./bash_scripts/run_live_demo.sh ...`;
a plain `VAR=...` prefix does not survive `sudo`): `GAIN` (10 on the busy 2.4 GHz band; 15 for the
quieter 1 GHz band), `FREQS`, `CHANNELS`, `DEST_PORTS`, `DURATION`. `USRP_ROWS_PER_FRAME` sets
the waterfall scroll cadence (display only).

**Two-terminal mode** (retune the radio without restarting the app):

```bash
# A: the app — pass the config as the FIRST ARGUMENT (an env prefix is lost under sudo)
sudo ./bash_scripts/run_torchscript_performance_test.sh config_coherent_power_perf_dynamic_single_channel.yaml
# B: the radio (over-the-air receive, streams until Ctrl-C)
CHANNELS="0" FREQS="2400e6" DEST_PORTS="1234" GAIN=10 ./bash_scripts/start_radio_stream.sh
```

The dashboard window opens immediately but stays blank until samples arrive; that is the idle
state, not a failure. The X410 negotiates **491.52 MSps** whatever `--rate` asks for; the pipeline
derives its FFT geometry from the true rate.

**No-radio loopback demo.** With a QSFP cable between the two ConnectX-7 ports, `tcpreplay`
replays a captured CHDR stream into the DPDK port and the full pipeline runs unchanged:

```bash
./bash_scripts/run_loopback_m5s_single.sh          # ch0 only, M5-S per frame
./bash_scripts/run_loopback_m5s_dual.sh            # ch0 + ch1 on one wire
sudo env DEMO_DISPLAY=:1 ./bash_scripts/loopback_channel_supervisor.sh dual   # + live Single↔Dual / detector toggles
./bash_scripts/stop_loopback_m5s_dual.sh
```

The replay pcaps are frame-aligned captures made with `bash_scripts/make_frame_aligned_capture.sh`
(default location `~/Documents/holoscan_waveform_generation/composition/composites`, override
`COMPOSITES`); they must carry the DPDK port's destination MAC. Details and the older 245.76 MSps
composite demos: [`notes/no_radio_loopback_demo.md`](notes/no_radio_loopback_demo.md).

---

## 3. Preprocess an offline dataset (segment the regions of interest)

Point a detector at a directory of SigMF captures and get a sibling directory of signal cutouts:

```bash
# 1. pick / edit the detector config (defaults: see the table in §6; any config with the detector block works)
# 2. make sure the container is built (§1)
# 3. run the selected detector over the directory
sudo python3 preprocess_dataset.py /data/argus_captures --detector coherent_power
sudo python3 preprocess_dataset.py /data/argus_captures --detector cuda_dino_finetuned   # DINO-FT M5-S
sudo python3 preprocess_dataset.py /data/argus_captures --detector cuda_dino --mode time_only --out /data/argus_time_slices
```

Every `*.sigmf-data` + `.sigmf-meta` pair in the directory (`--recursive` to descend) is replayed
through the **same compiled operators as the live app** (`run_offline_cuda_detector_eval`, driven by
`run_cuda_dino_offline_file.py`): detector → `signal_snipper` → `sigmf_file_sink`. The output
defaults to `<dir>_preprocessed`:

```
/data/argus_captures_preprocessed/
  manifest.csv                     one row per region across the whole dataset
  preprocess_summary.json          detector, config, mode, per-capture counts, bytes in/out, kept fraction
  <capture_stem>/
    <capture_stem>.source.sigmf-meta        copy of the input metadata (provenance)
    snippets/<stem>_ch0_f<frame>_s<n>.sigmf-{data,meta}   one SigMF recording per region
    manifest.csv
    frame_manifest.csv, offline_eval_summary.json          per-frame detector bookkeeping (frames processed, masks emitted)
    [mask_arrays/*.packed.npz]                             the detector masks, with --save-masks
```

Runs are resumable (a capture with a `manifest.csv` is skipped; `--force` redoes it) and each
capture's staged copy under `/tmp/usrp_spectrograms` is freed after use (`--keep-staged` to keep it).

**Modes.** `--mode frequency` (default) isolates each region: digital down-convert to baseband,
low-pass to the detected bandwidth, decimate to the minimum rate plus `oversample_percent`
(`--no-downsample` keeps the original rate). `--mode time_only` keeps full-band, full-rate time
slices that contain any detection, with every signal's band edges annotated.

**Why the output is not ragged.** Filter-and-decimate makes every region its own length *and* its
own sample rate. Rather than one array with uneven rows, the default (`--sink per_signal`) writes
each region as a standalone, standard SigMF recording: its `core:sample_rate` is the region's own
rate, `captures[0].core:frequency` is its RF centre, and the annotation records the band edges
plus provenance (`wfgt:orig_sample_start`, `wfgt:orig_sample_rate`, `wfgt:frame_number`,
`wfgt:decimation_factor`). The flat `manifest.csv` (capture, file, rate, centre, duration, band
edges, bandwidth, provenance) is what a loader iterates; any SigMF reader opens the files. For the
compact form the live app uses, `--sink pack` writes one **variable-rate container** per
`pack_frames` frames: all regions concatenated into one `.sigmf-data` with `wfgt:container: true`,
`wfgt:layout: concatenated_variable_rate`, each annotation's `core:sample_start`/`sample_count`
locating its slice and `wfgt:snippet_sample_rate` being its authoritative rate. Both layouts are
produced by the same sink; the container trades a custom reader for far fewer files.

**Knobs that change what gets cut** live in the base config's `signal_snipper` block:
`min_box_pixels` (speckle filter), `merge_gap_rows` / `merge_gap_cols` (coalesce fragments, in
detector-mask pixels), `oversample_percent`, `bandwidth_margin_hz`, and the detector's own
sensitivity (`coherent_power_signal_detector.fast_score_threshold`,
`finetuned_dino_detector.threshold`, …). Pass your edited copy with `--config`.

Scoring detectors against annotated ground truth (masks vs SigMF annotations, per class / SNR /
bandwidth) is a separate harness:
[`infocom_evals/signal_detection_experiments/run_signal_detection_README.md`](infocom_evals/signal_detection_experiments/run_signal_detection_README.md).
Single-file offline runs (masks + spectrogram tensors, no snipping):

```bash
sudo python3 run_cuda_dino_offline_file.py <capture.sigmf-data> --detector coherent_power \
     --config config_coherent_power_perf_dynamic_single_channel.yaml --output-root /tmp/usrp_spectrograms/offline_test
```

---

## 4. Calibration (optional)

The `dynamic` coherent configs and the fine-tuned DINO need no calibration. The calibrated
per-frequency floor (`config_coherent_power_perf_perfreq_single_channel.yaml`) and the zero-shot
DINO coherence floors are fitted offline from captured SigMF:

```bash
sudo ./calibration/calibrate_coherent_power_config.sh     # -> calibration/coherent_power_per_freq_floor.npy
sudo ./calibration/calibrate_dino_coherence_config.sh      # -> DINO coherence floor + gate threshold
sudo ./calibration/calibration_script.sh                   # -> DINO positional (RoPE) noise template
```

See the header of each script for its inputs and policy knobs, and
[`calibration/README.md`](calibration/README.md) for the positional template.

---

## 5. Repository layout

| Path | Contents |
| --- | --- |
| `config_*.yaml` | The **current** configs (table in §6). Only these are synced into the container build tree by the rebuild wrapper. |
| `bash_scripts/` | All build / run / loopback / reboot wrappers; `container_env.sh` holds the container identity, `container_repo_guard.sh` refuses to run against another checkout. |
| `preprocess_dataset.py`, `run_cuda_dino_offline_file.py`, `repack_offline_masks.py` | Offline drivers (§3). `export_dinov3_finetuned_torchscript.py` exports a fine-tuned checkpoint for the live operator; `export_dinov3_torchscript.py` exports the zero-shot backbone during container setup. |
| `calibration/` | Calibrate scripts + configs + emitted `.npy` artifacts. |
| `old_configs/` | Superseded configs, still runnable as `old_configs/<name>.yaml`. |
| `infocom_evals/` | Evaluation harnesses and records: detector scoring, snipper data-saving eval, BER eval, latency, the demo conductor + classifier daemon (`pycodec_e2e/`). |
| `notes/` | Design notes, bring-up guides and development history (start with `RUNNING_ON_DGX_SPARK.md`, `live_demo_dev_state.md`). |
| `debug_scripts/` | Ad-hoc analysis helpers; nothing depends on them. |
| `main.cpp`, `*.cu`, `*.hpp`, `run_offline_cuda_detector_eval.cpp` | App, visualization, snipper / sink / compression operators and the offline binary. The detectors live in `operators/{coherent_power_signal_detector,cuda_dino_detector,finetuned_dino_detector}`. |

---

## 6. Configs and weights

| Config | Detector | Role |
| --- | --- | --- |
| `config_live_v3_single_channel.yaml` / `config_live_v3_two_channel.yaml` | coherent_power | **Live demo** start configs (`v5`, `v5dual`): snipper on, dashboard with DEMO CONTROLS. |
| `config_live_v3_dino_ft_m5s.yaml` / `config_live_v3_dino_ft_m5s_two_channel.yaml` | cuda_dino_finetuned (M5-S) | Live demo DINO target (conductor switch) and the M5-S loopback runners. |
| `config_live_v3_dino.yaml` | cuda_dino | Live zero-shot DINO target of the conductor. |
| `config_live_v3_dino_ft_sb.yaml` | cuda_dino_finetuned (M4) | Legacy conductor entry (hidden in the dropdown). |
| `config_loopback_v3_{single_channel,dino,dino_ft,dino_ft_sb}.yaml` | all | 245.76 MSps composite loopback demo (`run_loopback_v3_demo.sh`); `dino_ft*` use the M3 export. |
| `config_coherent_power_perf_dynamic_single_channel.yaml` | coherent_power | Plain live run, live-learned floor (`single` preset). |
| `config_coherent_power_perf_perfreq_single_channel.yaml` | coherent_power | Plain live run, calibrated floor; offline-eval base config. |
| `config_coherent_power_performance_emit_stride1_two_channel.yaml` | coherent_power | Plain dual-channel live run (`dual` preset). |
| `config_cuda_dino_performance_single_channel.yaml` | cuda_dino | Plain live / offline-eval zero-shot DINO. |
| `config_cuda_dino_finetuned_performance_single_channel.yaml` | cuda_dino_finetuned (M5-S) | Plain live / offline-eval fine-tuned DINO (no snipper). |
| `config_signal_snipper_single_channel.yaml` | coherent_power | Headless snipper (default base of `preprocess_dataset.py --detector coherent_power`). |
| `config_mask_replay_snip_single_channel.yaml` | mask_replay | Snip from precomputed masks of any detector (`snip_eval`). |
| `config_snipper_viz_demo.yaml` / `config_dino_finetuned_viz_demo.yaml` | coherent_power / cuda_dino_finetuned (M2_dr) | 245.76 MSps replay demo with decode (`run_loopback_demo.sh`). |
| `config_coherent_power_capture_chdr_single_channel.yaml` | coherent_power | Capture a frozen CHDR/IQ snapshot for offline replay. |

Fine-tuned exports referenced by the configs (all gitignored, under `dino_fine_tuning/weights/`):
`finetuned_dino_m5s_491_bf16.ts` (**required** for the demo), `finetuned_dino_m4_491_bf16.ts`
(`_sb`), `finetuned_dino_m3_491_bf16.ts` (`loopback_v3_dino_ft*`), `finetuned_dino_m2_dr_bf16.ts`
(`dino_finetuned_viz_demo`). How they were trained and how to export a new one:
[`dino_fine_tuning/README.md`](../../dino_fine_tuning/README.md).

---

## 7. Config reference (the knobs you actually tune)

**`pipeline`** — `detector_type`; `enable_signal_snipper` (adds snipper + sink);
`enable_snippet_compression` (demo codec between snipper and sink).

**`advanced_network.cfg`** — `interfaces[].address` (PCIe address of the DPDK NIC: `0000:01:00.0`
on the Spark); `memory_regions[].kind` (`huge` on GB10 unified memory, `device` for GPUDirect on
x86); `rx.flows[].match.{udp_src,udp_dst}` (ch0 `49153`/`1234`, ch1 `49154`/`1235` must match the
sender's `--dest-port`); `master_core` must be numerically lower than every queue `cpu_core`.

**`chdr_converter`** — `num_channels`; `channel_sample_rates_hz` / `channel_center_frequencies_hz`
(the run wrappers override these from the radio sidecar via `USRP_SAMPLE_RATE_HZ` /
`USRP_CENTER_FREQ_HZ`); `num_ffts_per_batch` (latency vs converter load).

**`fft`** — `span`, `transform_points` (20480 at 491.52/500 MSps), `resolution`; the runtime
rescales from `reference_span_hz` / `reference_fft_size` to the true rate.

**`coherent_power_signal_detector`** — `fast_power_floor_db` / `fast_power_span_db` /
`fast_score_threshold` (sensitivity); `per_freq_threshold_mode` (`dynamic` | `calibrated` |
`static`) + `per_freq_threshold_offset_db`; `dynamic_floor_*`; `live_emit_freq_persistence_*`
(temporal gating); `emit_stride` (detect every N-th frame).

**`cuda_dino_detector`** — `coherence_band_threshold*`, `dino_structure_*`,
`raw_dino_positional_template_*`, `dino_coherence_gate_*`, `hybrid_fusion_mode`,
`chunk_bandwidth_hz` / `chunk_overlap_hz`, `torch_dtype`.

**`finetuned_dino_detector`** — geometry contract from the export's `.meta.json`: `tile_rows`,
`nfft`, `fft_window`, `db_vmin` / `db_vmax`, `threshold`, `model_script_path`; front end
`real_time_downsample`, `flatten_noise_floor`, `adaptive_normalization` (+ `adaptive_span_db`,
`adaptive_floor_frac`, `adaptive_robust_floor`); `emit_stride`; ingest guard
`invalid_frame_min_occupancy` / `invalid_frame_occupancy_k`.

**`signal_snipper`** / **`sigmf_file_sink`** — see §3. `ring_depth` must exceed
`2 × emit_stride + pipeline lag`; `max_queued_batches` / `write_budget_mb_s` bound the sink.

**`visualization.renderer`** (display only) — `db_floor` / `db_ceil`, `dynamic_color_*`,
`rows_per_frame`, `overlay_alpha`, `demo_title` / `demo_subtitle`, `detector_label`;
`holoviz.headless: true` for off-screen captures.

---

## 8. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `Received packets: 0` / dashboard stays blank | Radio not streaming, or data exits the wrong X410 port. `sudo ./bash_scripts/check_radio_topology.sh`; crossed SFP cables look like "control works, data address does not ping". |
| `Fail to create MR` / `unable to DMA map` at start | RX region `kind: "device"` on GB10; must be `kind: "huge"`. |
| `Cannot create lock ... another primary process running` | Previous app still exiting; wait, or clear `/dev/hugepages/nwlrbbmqbh*` (`after_reboot.sh` does). |
| `Failed to initialize glfw` | X access: `xhost +local:root`, or recreate the container from a desktop session. |
| `rx xport timed out getting a response from mgmt_portal` | X410 MPM wedged: `ssh root@192.168.21.2 systemctl restart usrp-hwd`, wait 10 s. |
| Full-width bright bar / max-hold pinned | GPU backlog recycling a buffer; raise the DINO `emit_stride`, keep `invalid_frame_min_occupancy: 0.40`. |
| `preprocess_dataset.py`: "offline binary missing" | `sudo ./bash_scripts/rebuild_demo_container_app.sh`. |
| Container "mounted from the wrong checkout" | The wrappers refuse to touch a container created from another clone; recreate it from this checkout. |

More: [`notes/RUNNING_ON_DGX_SPARK.md`](notes/RUNNING_ON_DGX_SPARK.md) §7 and
[`notes/live_demo_dev_state.md`](notes/live_demo_dev_state.md) §7.

---

## 9. Snippet message contract (hooking up your own consumer)

The snipper emits `holoscan::ops::SnippetBatchMessage` (`signal_snip_types.hpp`) on `snippets_out`.
Each `SignalSnippet` carries its IQ **on the GPU** (`device_iq`, pooled) plus `n_iq`,
`sample_rate_hz`, `center_freq_hz`, source-stream provenance and `annotations`; only the file sink
copies to host. To classify in-graph, add an operator with `spec.input<SnippetBatchMessage>("in")`
and wire it in `main.cpp` next to the sink:

```cpp
auto classifier = make_operator<ops::MySignalClassifier>("signalClassifierOpCh0", from_config("signal_classifier"));
add_operator(classifier);
add_flow(signalSnipperOps[ch], classifier, {{"snippets_out", "in"}});   // fans out; can coexist with the sink
```

The demo instead runs an external Python daemon (`infocom_evals/pycodec_e2e/rt_decode_daemon.py`)
on the written snippets, so classifier models can change without rebuilding the app.

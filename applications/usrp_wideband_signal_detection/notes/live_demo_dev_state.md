# Live Demo — Development State & Handoff (2026-09-20)

Master resume doc for the DINO-FT real-time signal-detection demo. Read this first after a context
compaction. Companion: `notes/dino_ft_finetune_plan.md` (the fine-tune roadmap + diagnostics). Branch:
**`live_demo`** (all work pushed to `origin/live_demo`). Platform: **DGX Spark (GB10/ARM64)**, USRP
**X410** over-the-air at 2.4 GHz, 491.52 MSps.

---

## 1. What the system is

A GPU-native Holoscan pipeline that detects + classifies RF signals in a wideband stream and renders a
live dashboard:

```
X410 radio → DPDK (enp1s0f0np0) → CHDR → wide FFT (20480) → DINO-FT detector (mask)
   → signal snipper (mask→boxes→SigMF snippets) → sink
   → spectrogram_visualization (Holoviz dashboard)
classify-only AMC daemon (external Python): reads snippets → classifies (VT-CNN2/ResNet1D/T-PRIME)
   → writes rt_metrics.json → viz reads it for the per-signal class overlay
```

Two detectors selectable at runtime (dashboard dropdown, via the conductor): **coherent_power** and
**cuda_dino_finetuned (M3_491)**.

## 2. The DINO-FT M3_491 model (the core deliverable)

- **What:** DINOv3 ViT-B/16 + seg head, fine-tuned (ft_lastN) on the exact 491.52 deployment front-end.
- **Weights (gitignored, on disk):** `dino_fine_tuning/weights/finetuned_dino_m3_491_bf16.ts` (traced,
  bf16). Checkpoint: `dino_fine_tuning/checkpoints/M3_491/best.pt`. Base DINOv3 + weights at
  `~/Documents/dinov3` (needs `PYTHONPATH=~/Documents/dinov3` + `src` for the eval/train scripts).
- **Front-end (training == inference, VERIFIED EXACT):** wide FFT 20480 → power→dB → gain_offset
  (13.01 dB @ 491.52) → **per-freq floor flatten** → **robust p20 low-percentile floor normalization**
  → resize freq 20480→1024 → tile into 256-row tiles. Training front-end
  `dino_fine_tuning/src/frontend.py` (`FrontEndCfg`) matches the C++ operator
  `operators/finetuned_dino_detector/finetuned_dino_detector.cu` defaults EXACTLY:
  flatten `reference_q 75 / smooth 0.005 / max_boost 12 / signal_cap 6`; robust/adaptive
  `low_pct 20 / high_pct 95 / min_range 8 / floor_below_calib 25`; `span 34 / floor_frac 0.12`;
  `db_vmin -47.6527 / db_vmax 20.4563`; nfft 1024. Threshold 0.6.
- **Why it exists:** on sparse live OTA the OLD model (M2_dr) missed weak signals. Robust normalization +
  domain-match fine-tune recover them. VALIDATED (see §5).

## 3. Configs (top-level, in the app dir)

- **Live DINO-FT:** `config_live_v3_dino_ft.yaml` (+ `_sb`, `_two_channel`). detector_type
  `cuda_dino_finetuned`, threshold 0.6, `adaptive_robust_floor: true`, `flatten_noise_floor: true`,
  `adaptive_normalization: true`, `real_time_downsample: true`, emit_stride 4.
- **Loopback DINO-FT:** `config_loopback_v3_dino_ft.yaml` (+ `_sb`). Same detector block.
- **Coherent v3:** `config_live_v3_single_channel.yaml` / `config_live_v3_two_channel.yaml`.
- **Loopback A/B (M3):** `config_loopback_eval_dino_ft_rt.yaml` — has `debug_mask_dump_dir` for dumping
  every emitted mask; detector block tracks the live config (updated M2_dr→M3 this effort).
- **Renderer knobs** (in the `renderer:` block): `overlay_enable` (default false = button-driven),
  `class_colors_enable` (default false), `overlay_alpha 0.7`, `decode_metrics_json`, `demo_control_json`.
- Snipper (`signal_snipper:` block): `merge_gap_cols 8` (=1.92 MHz on the 1024-col DINO mask; coherent
  uses 80 on its 10240-col mask — SAME physical gap, grid-scaled, NOT a hack), `merge_gap_rows 16`,
  `min_box_pixels 256`, `max_mask_occupancy` (flood guard).

## 4. How to run

- **Live OTA (single):** `sudo ./bash_scripts/run_live_demo.sh v3` (coherent default + dropdown) or a
  config directly. **`v4` = v3 + overlay + class colors ON at startup** (this effort). Dual: `v3dual` /
  **`v4dual`**. Radio env: `GAIN` (10 for busy 2.4 GHz), `FREQS`, `CHANNELS`, `DEST_PORTS`. v3/v4 set
  `V3_STACK=1` → start the classify-only daemon + conductor + snippet janitor.
- **v4/v4dual mechanism:** presets export `USRP_OVERLAY=1 USRP_CLASS_COLORS=1`; the viz operator honors
  these env flags at init; `demo_conductor.py` forwards them on each detector-switch relaunch so colors
  persist. No config duplication. (v4 currently defaults to coherent like v3 — OPEN: make it default to
  DINO-FT? one-line change to the preset's demo_control detector.)
- **Loopback:** `bash_scripts/run_loopback_v3_demo.sh` (tcpreplay over the QSFP loopback). Needs the
  frame-aligned pcap (below) for clean 1:1.
- **Offline eval (ground truth):** `run_cuda_dino_offline_file.py <sigmf-data> --detector
  cuda_dino_finetuned --config <cfg>`. Fine-tune held-out eval: `dino_fine_tuning/src/eval_heldout.py`.
- **Ground-truth A/B harness (this effort, reusable):** `dino_fine_tuning/ab_norm_validate.py`
  (normalization ablation) and `dino_fine_tuning/ab_span_floor_sweep.py` (span/floor grid). Run with
  `PYTHONPATH=~/Documents/dinov3 <.venv-ml python>`. Dataset `dino_fine_tuning/data/dataset_491/heldout_sigmf`.
- **ML venv:** `/home/genesys-dgx1/Documents/holoscan_waveform_generation/.venv-ml/bin/python`
  (torch 2.13+cu130). In-container torch 2.10.
- **Build after code/config change:** `sudo ./bash_scripts/rebuild_demo_container_app.sh`. Container
  `usrp_x410_sig_det_sat3737`.

## 5. Validation results (all on ground truth / real radio)

- **Normalization ablation** (`dino_ft_robust_norm/results/norm_ablation_ab.md`): `adaptive_normalization`
  is strongly load-bearing (OFF: dense 94→64%, sparse low-SNR 90→40%). `flatten_noise_floor` +
  `adaptive_robust_floor` are neutral on synthetic (their benefit is live-specific: X410 edge roll-off /
  dense occupancy) but train-consistent → keep. Baseline detection: dense 94% / sparse 90%.
- **span/floor sweep** (`.../span_floor_sweep.{md,png}`): flat surface (F1 0.959–0.964); baseline
  (34, 0.12) already optimal — do NOT retune.
- **Loopback online==offline:** RT reproduces offline BIT-EXACT on one clean pass (34/46 pixel-identical,
  mean IoU 0.998). Looped copies scramble ONLY because the raw capture is 46.875 frames (tcpreplay
  loop-seam) — fixed by the frame-aligned capture. Opus-reviewed.
- **Frame-aligned capture:** `bash_scripts/make_frame_aligned_capture.sh` → 46-whole-frame pcaps
  (`x410_ota_2g4_gain10_20260908_46f.spark.pcap` port 1234, `x410_ota_ch1_p1235_46f.spark.pcap` port
  1235) in `~/Documents/holoscan_waveform_generation/composition/composites/`. Validated: all loops
  0.998 IoU vs offline.
- **LIVE OTA (2026-09-20):** full pipeline on the real X410, M3 + classifier + class overlay; 16 live
  class markers (OFDM dominant = WiFi), CHDR partial_drops=0, GPU ~77%. Screenshots in
  `dino_ft_class_overlay/results/live_ota_*.png`.
- **Bright-bar root cause + fix (2026-09-20, OTA):** the intermittent full-width bright bar (also pins
  PSD max-hold) = a **uniform full-scale recycled-mbuf garbage frame** (full-count 10240/10240, distinct
  fp, NOT short/partial) formed when DINO GPU load stalls the RX (`Fell behind`). Load-driven: single
  `emit_stride=4` → 23 Fell-behind + 7 bars/120s; **`=12` → 0/0**; dual `=16` → 0/0. Also found the
  invalid-frame guard mis-calibrated (`min_occupancy=0.03` < real 4-15% busy-band → 10 false-suppressions
  + cold-start baseline deadlock). Fixes (live DINO configs + viz only; eval/loopback untouched):
  `emit_stride 4→12` (dino_ft, _sb), guard `min_occupancy 0.03→0.40`/`k 6→2` (all 3 live DINO), a viz
  **broadband-suppress guard** (`SpectrogramPreviewOp`, param `broadband_suppress_frac` default 0.85,
  skips uniform-full-scale frames), and a **"Reset Max Hold"** ImGui button. Record:
  `infocom_evals/signal_detection_experiments/dino_ft_bright_bar_rootcause/`. Memory
  `dino-ft-bright-bar-rootcause`. (The older short-packet bar was already fixed by CHDR zero-fill
  `14b7ae67`; re-verified.)

## 6. The class-color overlay (feature built this effort)

- **"Color Mask by Class" dashboard toggle** colors the detection mask per predicted class, PER SIGNAL
  REGION, separated in TIME and FREQUENCY. Palette (5 distinct, legible on the blue waterfall): PSK green,
  QAM magenta, FSK orange, **OFDM cyan** (not blue — blends), **NOISE red**; unclassified = neutral gray.
- **Mechanism:** the classify-only daemon (`rt_decode_daemon.py`) now emits per-signal frequency-tagged
  class markers with band edges `[f_lo_hz, f_hi_hz]` in `rt_metrics.json` `recent_decodes` (fixed: it
  previously only kept aggregate counts in classify-only mode). The viz colors each mask pixel by the
  decode whose band CONTAINS it, and **freezes the class into a parallel `history_class_id` ring at
  capture time** (so old waterfall rows keep their class; no whole-column recolor). All in
  `spectrogram_visualization.cu` (`class_color_table`, `class_id_color_lut`, `sample_class_at_max_ring`,
  `overlay_mask_ring`, and the freeze in `patch_history_mask_for_frame`).
- **Requires the classifier daemon.** Off by default (button / the v4 env flags). Artifact:
  https://claude.ai/artifact/RSdrKitPhtsFgSsyX5K3Ca. Repo:
  `infocom_evals/signal_detection_experiments/dino_ft_class_overlay/`.
- **Class set** is `PSK/QAM/FSK/OFDM/NOISE` (BPSK/QPSK fold into PSK). On live OTA the classifier is
  out-of-distribution (trained synthetic) — labels only meaningful on known/loopback waveforms; the
  coloring MECHANISM is what's validated live.

## 7. Hardware / environment gotchas (READ before radio bring-up)

- **X410 SFP cables can get swapped** on rewire. Symptom: control/SSH/`uhd_find_devices` work but device
  open fails at RFNoC GSM init ("recv error on socket: Connection refused") + data addr 192.168.10.2
  won't ping. Diagnose via MAC mapping (data 10.2 → sfp0 MAC `…f5`; if the host reaches `…f5` on the
  CONTROL NIC, cables are crossed). Fix = swap the two SFP cables. `check_radio_topology.sh` does NOT
  catch this. See memory `x410-swapped-sfp-cables`.
- **X410 boot** ~90–120 s; discovery answers before a full open is ready — wait for `uhd_usrp_probe
  --args addr=192.168.21.2` to print the Mboard tree. SSH: `root@192.168.21.2` (key-based, no password).
  X410 runs Alchemy 2024.00 (busybox ip); MPM = `usrp-hwd.service`; FPGA image `CG_400` @ 491.52 MSps.
  X410 IPs: sfp0=192.168.10.2 (data), sfp1=192.168.21.2 (control). Host: enp1s0f0np0=192.168.10.1 (data,
  DPDK), enp1s0f1np1=192.168.21.1 (control). A stray 192.168.21.1/24 tends to re-appear on the data NIC
  after rewiring — harmless once cables are right.
- **X410 `rx xport timed out getting a response from mgmt_portal`** at RX-streamer start (control RPC
  works, rates/freq/gain set, then `get_rx_stream` times out): MPM is in a bad state (often after a
  restart or a prior stream not cleanly stopped). Fix = `ssh root@192.168.21.2 systemctl restart
  usrp-hwd`, wait ~10 s for it to re-serve, then re-run. Also clears an X410 stuck streaming stale.
- **Loopback dst-MAC:** pcaps need dst-MAC `4c:bb:47:2c:45:13` or DPDK drops them (memory
  `loopback-dstmac-spark`).
- **`pkill -f` self-match** (exit 144): commands containing the literal process name match their own
  shell. Use bracketed patterns like `[r]x_to_remote_udp`.
- **Container** may need bring-up after a reboot (run_live_demo rebuilds/starts it).

## 8. Key commits (live_demo, this effort, newest first)

`c1e7d241` v4/v4dual presets · `1ce578dd` live OTA validation · `42438b37` span/floor sweep ·
`242f5363` normalization ablation · `77d18961` stale flatten-comment fix · `477f52a1` per-region
time+freq coloring · `430be135` refresh artifacts · `53c96560` 5 distinct colors · `035d5a60` daemon
per-signal markers + class_colors config param · `267fafff`/`a646d024`/`9f9dd494` class-color overlay ·
`e4f0e8b7` frame-aligned capture · `ad33dfbb` honest online-vs-offline metric · `01f08d3b` lime overlay
fix. Earlier: `19350a7c` M3_491 fine-tune, `af121eaf` robust normalization, `754f4ad2` ingest
invalid-frame guard.

## 9. Open items / possible next steps

- **v4 default detector:** currently coherent (mirrors v3); consider defaulting v4 to DINO-FT M3 so the
  class colors show best (one-line: set the v4 preset's demo_control detector to `cuda_dino_finetuned`).
- **Dual-channel OTA:** now run on two live RF streams (2026-09-20, 2.4 + 1.0 GHz): CHDR
  partial_drops=0, panic_resets=0, Fell-behind=0 at DINO `emit_stride=16` — clean.
- **Class overlay label accuracy on OTA:** classifier is out-of-distribution on real WiFi/BT — retrain
  on OTA-like classes if label accuracy (not just the coloring mechanism) matters for the demo.
- **Artifact:** could add the live-OTA shot to RSdrKitPhtsFgSsyX5K3Ca alongside the loopback ones.
- Radio currently booted + correctly cabled + released (ready to run).

## 10. Memory pointers (auto-loaded each session)

`dino-ft-finetune-plan`, `dino-ft-bright-bar-rootcause`, `class-color-mask-overlay`,
`loopback-46875-frame-seam`, `x410-swapped-sfp-cables`, `dino-ft-rt-overlay-not-detector`,
`dino-ft-live-artifact-is-ingest`, `loopback-dstmac-spark`, `spark-port-environment`.

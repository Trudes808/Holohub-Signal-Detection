# DINO-FT 491.52 domain-match fine-tune — plan + diagnostics (resume doc, 2026-09-14)

Resume point for the robust fix to real-time DINO-FT under-detection. Demo is ~1 week out.
All diagnostic numbers below are DONE — do not re-run them; start at "Roadmap".

## Objective
DINO-FT (M2_dr) under-detects real signals live at 491.52 MSps. Deliver a robust fix that works
across unknown field conditions (near/far transmitters, sparse or dense spectra).

## Locked decisions (user, 2026-09-14)
1. **Data: BOTH dense AND sparse composites** — live density/content is unknown; cover both.
2. **Normalization: make it robust AND train the model on it** — field distance varies (SNR/level
   varies), so normalization must adapt; the fine-tune must learn that same normalized input.

## Diagnostics established (with numbers — don't repeat)
- **Live sparse OTA (representative): the MODEL is the bottleneck.** Adaptive norm works live
  (per-frame floor -31.96 dB, noise->0.15, signals reach 1.0), but DINO masks the dominant signal and
  misses many clear ones incl. bright *small* ones. Coherent cross-check on the gain-10 OTA capture:
  DINO misses **2.27%** real signal (energetic cols) vs **0.29%** coherent-noise-FP (~8:1). Config
  tuning (drop guard + adaptive norm) is exhausted; recall needs the model.
- **Prior config fixes already shipped + committed** (branch live_demo, pushed): invalid-frame guard
  (kills drop-corruption blobs), per-frame adaptive normalization (recovers moderate signals; enabled
  in the 5 DINO-FT demo configs). See memory [[dino-ft-live-artifact-is-ingest]].
- **comprehensive_ordered SNR benchmark is CONFOUNDED — do not fine-tune to it as-is.** It's a *dense*
  composite (73% active, 110-MHz-wide GT boxes). Results are normalization-dominated, not detection:
  adaptive rec/detect% = clean 0/0, snr12 15/27, snr0 9/12; fixed-clip = clean 4.8/5.3, snr12 15/26,
  snr0 40/37. Wide boxes + dense scene make detection-rate gameable by broad firing.
- **Adaptive normalization is NOT robust at extremes** (root of the confound + a real field risk): it
  anchors to the flatten reference (q=75 per-freq floor), which is invalid when a frame is mostly-signal
  (floor = signal) or noiseless (no floor). Ties fixed-clip on sparse/moderate; loses badly at 0 dB and
  on clean. This is exactly the "field distance varies" robustness the user wants fixed.

## The robust change (two parts)
### A. Density/noise-robust normalization (implement, then train on it)
Current: `finetuned_dino_detector` flatten path, `adaptive_normalization` anchors clip_vmin to
`frontend_reference_device` (q=75 smoothed per-freq floor), span 34. Replace the FLOOR ESTIMATE with a
robust one that survives any occupancy/SNR:
- Estimate noise floor as a LOW percentile of the dB image (e.g. 10-20th pct of all pixels, or a robust
  per-freq low-order stat) instead of q=75 — survives high occupancy.
- FALLBACK to the fixed calibrated clip when the floor is unreliable: if (high_pct - low_pct) is tiny
  (mostly-signal OR noiseless) the scene has no usable dynamic range spread -> use the fixed db_vmin/vmax.
- Keep the fixed span (~34 dB) anchored to the robust floor, mapping floor->~0.1, signals->~1.0.
Validate across sparse/dense/clean/0dB that it no longer collapses (the failure above). Then TRAIN the
fine-tune on this exact normalization so the model matches deployment.

### B. Domain-match fine-tune of the segmenter for 491.52
Train the model on data produced by the *exact* deployment front-end (wide FFT -> resize -> flatten ->
robust-norm) at the 491.52 geometry, across dense+sparse scenes and varied per-signal SNR.

## Feasibility inventory (verified present on this Spark)
- Training pipeline: `dino_fine_tuning/src/` — `train.py` (`--config configs/train.yaml --dataset
  data/dataset --mode ft_lastN --name M2 --out checkpoints/M2`), `model.py` (DinoSegmenter = frozen
  DINOv3 ViT-B/16 + SegHead, `--mode ft_lastN` unfreezes last 4 blocks + norm), `dataset.py`
  (RFSegDataset), `build_dataset.py`, `rfdata.py::build_frame_mask` (GT rasterizer from SigMF annotations),
  DiceBCE loss (pos_weight 3). Input [B,1,256,1024] in [0,1]; channel-repeat + imagenet-norm INSIDE model.
- **Export EXISTS**: `applications/usrp_wideband_signal_detection/export_dinov3_finetuned_torchscript.py`
  — `--ckpt best.pt --tile-rows 256 --nfft 1024 --sample-rate-hz 491.52e6 --autocast bf16 --dinov3-repo
  /home/genesys-dgx1/Documents/dinov3`. Traces full segmenter under bf16 autocast; writes `.ts` + `.meta.json`.
  Deployment contract (from `operators/finetuned_dino_detector/finetuned_dino_torch_helpers.{hpp,cpp}`):
  input float32 [B,1,tile_rows,nfft] in [0,1]; output **logits** [B,1,tile_rows,nfft]; bf16 baked at export;
  the C++ calls `forward` (native) and `forward_downsampled` (wide->resize->tile->sigmoid>=thr->stitch).
- Base weights PRESENT: `/home/genesys-dgx1/Documents/dinov3/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth`.
  DINOv3 source repo: `/home/genesys-dgx1/Documents/dinov3` (add to sys.path; not pip-installed).
- Composite generators PRESENT: `/home/genesys-dgx1/Documents/holoscan_waveform_generation/composition/`
  (`compose.py` writes truth `.sigmf-meta` WITH annotations: core:freq_lower/upper_edge, sample_start/count,
  core:label, wfgt:kind/class/occupied_bw_hz/power_db/time_group — directly consumable by
  `rfdata.build_frame_mask` and `mask_eval_metrics.load_source_annotations`). Box GT converter:
  `composition/sigmf_annotations_to_detection_boxes.py`.
- GPU torch env: `/home/genesys-dgx1/Documents/holoscan_waveform_generation/.venv-ml/bin/python`
  (torch 2.13.0+cu130, CUDA OK on GB10). Imports dinov3 with the repo on sys.path.

## MISSING / must recreate
1. **M2_dr `best.pt`** — absent. Fine-tune restarts from the base backbone (full ~18-epoch ft_lastN),
   NOT a warm-start.
2. **Materialized dataset** (`data/dataset/`) — rebuild via `build_dataset.py`.
3. **491.52 composites** — `composition/geometry.py FS=245.76e6` is hardwired; edit for 491.52 and
   re-validate the frequency plan. Add a SPARSE generation mode (spaced signals + noise gaps) alongside
   the existing dense composites.
4. **Env fixups**: `pip install pyyaml` into `.venv-ml`; repoint `configs/train.yaml:2` (weights_path)
   and `configs/dataset.yaml` (captures_dir/out_dir) to `/home/genesys-dgx1/Documents/...`; export
   `--dinov3-repo /home/genesys-dgx1/Documents/dinov3`. `scripts/setup_env.sh` is old-bench (conda/cu124) — ignore.
5. (Optional) the `_dr` rate-randomization + hann-window recipe is NOT in this checkout; single-rate
   491.52 training is fine for a fixed-rate deployment.

## Roadmap (start here after compaction)
1. **[DONE 2026-09-14] Robust normalization** in `finetuned_dino_detector` — implemented + offline-validated.
   New knobs: `adaptive_robust_floor` (default true), `adaptive_low_pct` (20), `adaptive_high_pct` (95),
   `adaptive_min_range_db` (8), `adaptive_floor_below_calib_db` (25). Path: histogram the flattened dB
   image (2 new kernels `ft_db_histogram_kernel`/`ft_hist_percentiles_kernel`), take the p20 floor, anchor
   the fixed span to it; fall back to the fixed calibrated clip when (p95-p20) < min_range (dense/mostly-
   signal) or the floor is > floor_below_calib below the fixed vmin (noiseless). Validation:
   `infocom_evals/signal_detection_experiments/dino_ft_robust_norm/` (gen_configs.py + run_validation.sh +
   analyze.py + README + results/). FINDING: robust never collapses (no scene >50% sat or >90% dead; std
   0.14-0.35 everywhere), FIXES adaptive's clean/dense over-saturation (7% vs 45% sat). Residual: at
   snr0 robust keeps signals clearly visible (higher std than fixed) but the CURRENT fixed-clip-trained
   checkpoint doesn't fire on robust's brightness -> a model-DISTRIBUTION mismatch, exactly what step 2
   (train ON robust norm) fixes. SAFETY: the 5 shipped demo configs are pinned `adaptive_robust_floor:
   false` (legacy q-blend) until a checkpoint fine-tuned on robust norm exists; flip to true with it.
2. **[DONE] 491.52 data+GT pipeline** — gen_491.py (real library waveforms placed on a 491.52 canvas,
   dense+sparse, per-signal SNR) + build_dataset_rt.py (stream through frontend.py -> tiles+GT). Dataset
   `dino_fine_tuning/data/dataset_491/`: train 2265 / val 277 / test 201 tiles + heldout_sigmf/ (6 comps).
3. **[DONE] train.py --mode ft_lastN from base** (M3_491): 18 epochs, val IoU 0.748 -> **0.9555**, F1 0.977,
   P 0.975, R 0.980. `dino_fine_tuning/checkpoints/M3_491/best.pt`.
4. **[DONE] export** -> `dino_fine_tuning/weights/finetuned_dino_m3_491_bf16.ts` (+ .meta.json), thr 0.6,
   491.52 geometry, bf16 (eager-vs-traced IoU 1.0). Loads+runs in the container (torch 2.10). The 5 DINO-FT
   demo configs now point at it with `adaptive_robust_floor: true`, threshold 0.6.
5. **[DONE offline / live PENDING] Validate**: held-out per-SNR region detection **83-100% across SNR**
   (dense+sparse) @thr 0.6; container A/B vs M2_dr (real operator): dense region-det 51%->72%, **sparse
   43%->86%**, sparse recall 35%->86%. Results in `dino_ft_robust_norm/` (heldout_snr.png, ab_summary.json,
   compare_ab.py, config_m3_eval.yaml/config_m2_eval.yaml). STILL TO DO: a **live radio run** with
   config_live_v3_dino_ft.yaml (M3 + robust) to confirm on-air; push (held per user).

## DINO-FT config inventory (which model each uses, 2026-09-14)
- **M3_491 (491.52 downsample+flatten+robust, thr 0.6):** the LIVE + LOOPBACK configs the demo actually
  runs — config_live_v3_dino_ft{,_sb,_two_channel}.yaml, config_loopback_v3_dino_ft{,_sb}.yaml. The v3/
  v3dual dashboard's DINO-FT detector switch (demo_conductor CONFIG_BY_DETECTOR_LIVE / _LIVE_DUAL /
  _LOOPBACK) loads these -> M3.
- **M2_dr (245.76 native, fixed clip, thr 0.95) — INTENTIONALLY KEPT:** config_dino_finetuned_viz_demo.yaml
  (the no-radio 245.76 replay/viz demo, demo_conductor CONFIG_BY_DETECTOR replay map). M2_dr is the
  rate-matched model at 245.76 native (240 kHz/bin = its training geometry); M3 is a 491.52 model and
  would run off-distribution there. Do NOT "consistency-fix" this to M3 without also moving the replay to
  491.52 (user decision 2026-09-14).

## Step 2 build decisions (2026-09-14, user: full autonomous build)
- **Front-end replica VALIDATED**: `dino_fine_tuning/src/frontend.py` reproduces the deployed downsample
  front-end (wide FFT auto=20480@491.52 -> dB - gain13.01 -> flatten -> robust p20 norm -> bilinear
  resize freq->1024 -> tile 256). vs the CUDA operator on the real 491.52 capture: **corr 1.0000, MAE
  0.0002**. Robust norm is scale-invariant (int16 capture vs cf32 composites both anchor to the per-frame
  floor) -> training on cf32 composites transfers to the sc16 radio. Training is FAST/CHEAP on GB10
  (ft_lastN 30.3M trainable, ~0.07s/step batch2, 1.5GB) -> compute is not the constraint.
- **Data source**: real MATLAB library `~/Documents/holoscan_waveform_generation/generated_waveforms_24576/`
  (9 classes BPSK/QPSK/16QAM/OFDM/802_11ax/5G_Downlink/Bluetooth/Broadband_FM/Narrowband_FM, all
  outputSampleRateHz=245.76e6, designedOccupiedBandwidthHz in the .json). Placer resamples each 2x->491.52,
  freq-shifts anywhere in +/-~230 MHz, scales to a per-signal SNR (near/far), adds complex AWGN floor.
- **Two densities**: DENSE (many overlapping signals filling the band) + SPARSE (few signals, big
  time/freq gaps -> matches the failed live scene). Per-signal SNR sweep covers field distance.
- **Pipeline**: `gen_491.py` (in-memory composite + SigMF annotations) -> `build_dataset_rt.py` streams
  each composite frame through frontend.py -> uint8 tiles + GT masks (rasterized on the wide-time-row grid,
  samples_per_row=fft_size) into memmapped train/val/test. A few composites also written to SigMF for the
  held-out benchmark (offline eval).

## Step 2 pipeline files (all in dino_fine_tuning/, .venv-ml + PYTHONPATH=~/Documents/dinov3:src)
- `src/frontend.py` — deployment front-end replica (VALIDATED corr 1.0) + `rasterize_gt` (wide-time-row GT).
- `src/gen_491.py` — 491.52 composite generator (real library waveforms, dense/sparse, per-signal SNR).
- `src/build_dataset_rt.py` — streams composites through frontend -> frames_{split}.npy/masks_{split}.npy +
  frames.csv (RFSegDataset format) into data/dataset_491/; also writes heldout_sigmf/ (test composites).
- `configs/train_491.yaml` — train config (weights_path repointed to ~/Documents/dinov3/weights).
- TRAIN: `PYTHONPATH=~/Documents/dinov3:src .venv-ml/bin/python src/train.py --config configs/train_491.yaml
  --dataset data/dataset_491 --mode ft_lastN --name M3_491 --out checkpoints/M3_491`
- EXPORT: `applications/.../export_dinov3_finetuned_torchscript.py --ckpt checkpoints/M3_491/best.pt
  --train-yaml dino_fine_tuning/configs/train_491.yaml --dinov3-repo ~/Documents/dinov3
  --sample-rate-hz 491.52e6 --autocast bf16 --output <new>.ts` -> then flip the demo configs'
  `adaptive_robust_floor: true` + point model_script_path at the new .ts.
- DEPLOY NORM (must match training): adaptive_normalization true + adaptive_robust_floor TRUE, span 34,
  floor_frac 0.12, low_pct 20, high_pct 95, min_range 8, floor_below_calib 25 (= FrontEndCfg defaults).

## Artifact / results locations
- A/B + weak-signal + fixes report: `dino_ft_offline_vs_loopback/` (build_artifact.py, results/, results_guard/).
  Live report artifact URL: https://claude.ai/code/artifact/091622b6-a4e8-43d7-b9c0-56c200f2e267
  Weak-signal deep-dive artifact: https://claude.ai/code/artifact/d97d1c0a-87a8-416e-85ab-bdbd2062f8a3
- **Retrain report artifact (M3_491): https://claude.ai/code/artifact/48ace403-604c-460b-9e9e-f74c1ff1a61b**
- Robust-norm + retrain experiment: `dino_ft_robust_norm/` (README, results/, gen_configs/run_validation/
  analyze/gen_491/build_dataset_rt/eval_heldout/compare_ab/render_live_ab/build_report).
- SNR benchmark: `dino_ft_snr_benchmark/` (make_snr_variants.py, snr_eval.py, results/, config_noadapt.yaml).
  Variants + eval outputs under `/tmp/usrp_spectrograms/snr_bench/` and `/tmp/usrp_spectrograms/offline_eval/snr_dino*/` (EPHEMERAL /tmp).
- Live dump: `/tmp/usrp_spectrograms/live_eval/dino_ft/` (EPHEMERAL).
- Capture: `~/Documents/captures/x410_ota_2g4_gain10_20260908.sigmf-*` (491.52/2.4GHz/1s, no annotations).
- Committed work on branch `live_demo` (pushed): guard, adaptive norm (enabled in 5 demo configs),
  offline-vs-loopback + weak-signal experiments.

## Ops notes
- Live run (radio): `sudo env FREQS=2400e6 CHANNELS=0 DEST_PORTS=1234 GAIN=10 DISPLAY=:1
  ./bash_scripts/run_live_demo.sh config_live_v3_dino_ft.yaml` (WITH sudo; xhost +local: first for viz).
  Stop: `sudo pkill -INT -f run_live_demo.sh`. NEVER `pkill -f 'tail -f ...'` (self-match kills the shell, exit 144).
- Offline eval: `python3 run_cuda_dino_offline_file.py <sigmf-data> --detector cuda_dino_finetuned
  --config <cfg> --output-root <dir>` (saves mask_arrays + gt_masks; eval via `mask_eval_metrics`/`eval_detector_masks.py`).
- Loopback pcaps on THIS host need dst-MAC 4c:bb:47:2c:45:13 (see [[loopback-dstmac-spark]]).

## RT dashboard "misses the loud signal / odd placements" — ROOT CAUSE = viz overlay color (2026-09-14)
Thoroughly investigated (radio + loopback + 2x loopback + offline). The DINO-FT M3 masks are CORRECT:
detect the loud burst 96-100% of frames, cover it MORE than coherent (20% vs 1% px), match offline<->
loopback (occ 0.67/0.70%), keep up (inference 52ms < 85ms stride-4 budget, CHDR partial_drops=0, no
frame gaps), and the overlay is frame-synced (a temporary MASKSYNC probe showed curmatch=true, lag=0,
dims match on every frame, even under 2x oversaturation). So it is NOT a sync/drift/dropped-frame or
detector problem. ROOT CAUSE was purely rendering: `mask_overlay_color` blended to pale-yellow at high
mask values, matching bright/yellow high-power signal pixels -> the overlay washed out on the loudest
signals while staying visible on dark noise ("missing the loud signal, detections in odd places").
Reproduced offline by replaying the exact overlay_mask alpha blend (results/faithful_overlay.png,
overlay_color_options.png). FIX (commit 01f08d3b): constant high-contrast lime-green mask color +
low-value alpha floor 0.18->0.35 (both overlay + ring paths). Verify on the dashboard: the loud burst
should now be clearly green-masked. NOTE: could not reproduce any radio-specific frame drift; if the
radio still shows misregistration under heavier load, re-add the MASKSYNC probe and read curmatch/lag.

## Online (loopback RT) vs offline M3 — CORRECTED finding (2026-09-19)
User asked for the online masks to be EXACTLY the offline masks (moved to loopback wiring for this), and
for an Opus subagent to double-check. Verdict: on ONE CLEAN PASS the RT pipeline reproduces the offline
detector output BIT-FOR-BIT (direct 1:1, loop 1: 34/46 frames pixel-identical, mean IoU 0.998, median
1.000, 100% >= 0.95). So the RT path is correct.
- HONEST-METRIC CORRECTION: an earlier "best-IoU over all online frames" number (~1.0) was OVERSTATED
  (asymmetric recall metric). The pcap was looped 2.6x; loop 1 always supplies a clean copy of every
  offline frame, so a max() returns ~1.0 regardless of the looped copies. The correct metric is direct
  1:1 per loop segment (compare_online_offline_exact.py, results/online_vs_offline_perloop.png).
- ROOT CAUSE of the loop-2/3 mismatch (mean IoU ~0.02/0.03): the capture is 491,520,000 samples =
  exactly 46.875 detector frames (10,485,760 samples/frame) -> NOT a whole number. `tcpreplay --loop K`
  therefore shifts the detector frame boundaries after the first pass; loops 2..K start mid-frame and each
  frame is a shifted splice of two capture frames. Verified it is MIS-ALIGNMENT not corruption: loop2's
  input spectrogram is a normal distribution (mean 0.250 std 0.138, dead%<=0.02 = 9.7), corr(loop1 f5,
  loop2 f51) = 0.002, partial_drops = 0 (not saturation/dropped-packet fill). A test-rig artifact of
  looping a non-frame-aligned pcap, NOT an RT bug. The continuous live radio stream has no loop seam and
  is unaffected.
- For a fully clean 1:1 across all frames: replay with `--loop 1`, or trim the pcap to a whole number of
  frames (46 frames = 483,000,000 samples). Artifact (corrected caption): B5gZNwSSro36Y7uarzWTnr.

## Frame-aligned canonical test capture — BUILT + VALIDATED (2026-09-19)
Generator: `bash_scripts/make_frame_aligned_capture.sh` (FRAMES=46 default). Trims the raw OTA
capture (480000 pkts = 46.875 frames) to a whole number of frames so tcpreplay loops start on a
frame boundary. Outputs (in composites/, ~1.99 GB each, not committed):
  x410_ota_2g4_gain10_20260908_46f.spark.pcap   (ch0, dst port 1234, 471040 pkts)
  x410_ota_ch1_p1235_46f.spark.pcap             (ch1, dst port 1235, portmap of ch0)
Alignment property: 471040 % 10240 == 0 (46.0 frames); original 480000 % 10240 == 8960 (seam).

VALIDATION (RT, M3, stride1, --loop 3 @ PPS 160000, detector debug_mask_dump; 138 masks, drops=0):
direct 1:1 vs offline M3 masks, EVERY loop identical:
  loop 1: mean IoU 0.9982, 34/46 pixel-identical, 100% >= 0.95
  loop 2: mean IoU 0.9982, 34/46 pixel-identical, 100% >= 0.95   (was 0.020 with the 46.875f pcap)
  loop 3: mean IoU 0.9982, 34/46 pixel-identical, 100% >= 0.95   (was 0.032 with the 46.875f pcap)
Before/after mean over all frames: 0.398 -> 0.998. Figure: results/aligned_capture_before_after.png.
The ~12 non-identical frames/loop are IoU >= 0.979 (1-2 px bf16 boundary jitter, identical loop-to-loop).
Config: config_loopback_eval_dino_ft_rt.yaml updated M2_dr -> M3_491 so loopback masks are directly
comparable to the offline M3 masks (detector block now tracks config_live_v3_dino_ft.yaml; stride stays
4 for the full-rate A/B, use a stride-1 temp copy + slowed replay for the 1:1 mask validation).

### Opus adversarial re-verification (2026-09-19) — VERDICT: SUPPORTED
Independent recompute (own script, symmetric IoU, strict j%46, offset sweep): aligned loops all
0.99822 mean / 34/46 bit-exact / min 0.97922 / 100% >= 0.95; old-capture loops 2-3 collapse 0.020/0.032.
Strongest evidence: online-vs-online ACROSS loops = IoU 1.00000 exact (bit-exact deterministic RT), so the
0.998 vs OFFLINE is purely bf16 jitter between the two backend paths, not misregistration. Adversarial
checks all pass: masks distinct per frame (adjacent differ 47k-283k px), real signal (offline occ mean
0.67%, range 0.13-2.17%), offset-0 is the UNIQUE IoU maximum (not best-match inflation), both-empty->1.0
never triggered. Caveat: mask-level 1:1 validated for ch0 only (ch1 pcap frame-alignment confirmed, but no
ch1 loopback masks were dumped to compare). Precise framing: online-vs-offline 0.998; online-vs-online 1.0.

## Class-color mask overlay — NEW feature (2026-09-19)
Dashboard toggle "Color Mask by Class" (spectrogram_visualization.cu): colors the detection-mask
overlay by the classifier's predicted modulation class instead of the single lime. Commits 9f9dd494 +
a646d024 (Opus-reviewed: SHIP WITH FIXES, all applied). Viz-only, off by default, no-op unless the
classifier daemon publishes markers -> current demo behavior unchanged.
- Class->color LUT: PSK=green, QAM=purple, FSK=amber, OFDM=blue (single source of truth; add a future
  class = one entry). NOISE/undecoded fall through to lime. Inline legend under the toggle.
- Mechanism (Route A): compose_visualization_rgb fetches decode markers once via
  decode_metrics_snapshot() (rt_metrics.json recent_decodes = {f_hz, mod, crc_ok}); per channel it
  builds a per-column color array by matching each column's frequency to the nearest in-band marker
  within +/-4% of the display span, and passes it to overlay_mask_ring (new optional param).
- LIMITATION: markers are frequency-only (no time bounds), so a class tint spans the full column height
  and a +/-4%-span band, not the true signal ROI. For time-accurate per-signal coloring, Route B threads
  a real per-signal class id through snipper->viz message->mask ring (~6 files + an in-process classifier
  op) -- see the data-flow map in this session.
- TO DEMO/TEST: run a config with the classifier daemon enabled so rt_metrics.json gets recent_decodes
  (the DINO-FT demo configs do NOT enable it), then flip the toggle. Live color render NOT yet verified
  end-to-end (needs the classifier pipeline up); compiles + links + Opus-reviewed.

## Class-color overlay — VALIDATED LIVE + daemon fix (2026-09-19)
Made the "Color Mask by Class" overlay actually light up (commit 035d5a60):
- rt_decode_daemon.py --classify-only now emits a per-signal freq-tagged class marker (note_decode with
  the predicted label) so rt_metrics.json recent_decodes populates (was aggregate-only -> empty markers ->
  no colors, and empty panel triangles too).
- Added renderer.class_colors_enable config Parameter (both viz operator classes) to start colors on for a
  headless run (default false = button-driven; shipped configs unchanged).
Live loopback validation (aligned 491.52 pcap, classify-only AMC daemon VT-CNN2/ResNet1D/T-PRIME, gate
T-PRIME; screenshots via gnome-screenshot on DISPLAY=:1; driver capture_overlay.sh):
- DINO-FT M3 + classifier: masks class-colored, OFDM blue dominant at 2.4 GHz (WiFi-like), 0 lime;
  recent_decodes 16 (OFDM 5, PSK 1, NOISE 10); overlay px OFDM 1157 / FSK 47 / QAM 32 / PSK 12 / lime 0.
- DINO-FT M3 classifier OFF: overlay falls back to uniform lime (no markers).
- coherent_power + classifier: same coloring on noisier coherent masks -> detector-agnostic.
Real-time at full radio rate (tcpreplay 480k pps = 491.52 MSps), GPU ~82%, CHDR partial_drops=0 in all
three. Artifact RSdrKitPhtsFgSsyX5K3Ca; repo infocom_evals/signal_detection_experiments/dino_ft_class_overlay/.
LIMITATION unchanged: markers are frequency-only -> tint spans the full column height (+/-4% span band),
not the true signal ROI (Route B = per-signal class id through snipper->viz, follow-up).

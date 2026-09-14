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
1. **Robust normalization** in `finetuned_dino_detector` (part A above) + offline-validate it no longer
   collapses on clean/dense/0dB (reuse `dino_ft_snr_benchmark/`). Config knobs already exist:
   `adaptive_normalization/adaptive_span_db/adaptive_floor_frac` — add a robust floor + fallback.
2. **491.52 data+GT pipeline**: edit geometry FS; generate DENSE + SPARSE composites, per-signal SNR
   sweep (near/far), GT via annotations; run through the deployment front-end (+ robust norm) -> training
   tiles + a held-out clean benchmark (replaces the confounded dense-only one).
3. **build_dataset.py** on that data (repoint paths); **train.py --mode ft_lastN** from base in `.venv-ml`.
4. **export_dinov3_finetuned_torchscript.py** -> new `.ts` + `.meta.json`; point the demo configs'
   `model_script_path/tile_rows/nfft/db_vmin/db_vmax/threshold` at the new sidecar.
5. **Validate**: held-out benchmark (recall/precision/IoU per SNR, sparse+dense), the offline-vs-loopback
   A/B harness (`dino_ft_offline_vs_loopback/`), and a live run.

## Artifact / results locations
- A/B + weak-signal + fixes report: `dino_ft_offline_vs_loopback/` (build_artifact.py, results/, results_guard/).
  Live report artifact URL: https://claude.ai/code/artifact/091622b6-a4e8-43d7-b9c0-56c200f2e267
  Weak-signal deep-dive artifact: https://claude.ai/code/artifact/d97d1c0a-87a8-416e-85ab-bdbd2062f8a3
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

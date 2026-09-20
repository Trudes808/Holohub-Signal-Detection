# DINO-FT real-OTA detection quality — diagnosis + robust-fix plan (2026-09-20)

Why DINO-FT M3_491 detections "make no sense" on the live 2.4 GHz band, and a concrete plan to fix
it robustly by domain-adapting the model to real OTA at 2.4 GHz **and** 1 GHz.

## Method
Offline, deterministic replay of a real 1 s OTA capture (`~/Documents/captures/
x410_ota_2g4_gain10_20260908.sigmf-data`, 491.52 MSps, 2.4 GHz) through the **same detector operators
as live** (`run_cuda_dino_offline_file.py --detector cuda_dino_finetuned --config
config_live_v3_dino_ft.yaml`), dumping per-frame masks + input spectrograms. Same for the coherent
detector as a comparison/teacher candidate. Scripts: `analyze_dino_ota.py`, `compare_dino_coherent.py`.

## Diagnosis (see figures)
`dino_ota_analysis.png`, `dino_vs_coherent_ota.png` (46 frames):
- **DINO-FT does NOT hallucinate across empty bands.** Aggregate detection density tracks the input
  energy; 0/20480 freq columns show detection with no energy. Per-frame mask occupancy ~0.67% (max 2.2%).
- **The failure is frequency imprecision / over-merging.** DINO paints **coarse horizontal blobs across
  the whole active region (~2.40-2.45 GHz)**, merging distinct WiFi channels and the quiet inter-channel
  gaps into big bands (15.6% of detected pixels land on quiet gaps). Stacked over time = the "stripes."
- **Config levers can't fix it.** Threshold 0.6 -> 0.8 cut detections only 349 -> 329 snippets (5.7%):
  the blobs are HIGH-confidence, i.e. the model is confidently coarse. Not a threshold/sideband problem.
- **Root cause = domain gap.** M3_491 was trained on clean synthetic composites (well-separated signals,
  synthetic AWGN). On the dense, overlapping, real-OTA scene it degrades to patch-coarse blobs. This is
  a MODEL/data problem, fixable only by adapting the model to the real domain.
- **Coherent is the opposite** on the same scene: tight per-signal segments (good precision) BUT speckle
  scattered across the whole band (its own small-noise false positives). occ 0.52%.

## Robust fix: domain-adaptation fine-tune on real OTA (both bands)
Teacher-distillation + hard negatives. Reuses the `dino_fine_tuning/` pipeline.

**Data (real X410 OTA, at 2.4 GHz AND 1 GHz):**
- **Positives** — antenna-on captures (varied times/scenes), pseudo-labeled by the **coherent detector
  run as a high-precision teacher**: raise its threshold (`per_freq_threshold_offset_db` / and/or
  `fast_score_threshold`) and `signal_snipper.min_box_pixels` so its masks are TIGHT and SPECKLE-FREE
  (precision >> recall — the student learns whatever the teacher shows, so clean labels matter more
  than catching every weak signal).
- **Negatives** — antenna-off (or 50 Ohm load) **noise-only** captures at both bands -> all-zero masks.
  Guaranteed negatives that teach "this real-receiver-noise texture = no signal," directly attacking the
  over-detection/edge-firing (needs NO pseudo-labeling; highest-value-per-effort data).
- **Anti-forgetting** — mix in a fraction of the original SYNTHETIC labeled set so the model keeps its
  learned signal-type discrimination while gaining real-OTA precision.

**Pipeline:**
1. Capture SigMF at 2.4 + 1 GHz, antenna-on (positives) + antenna-off (negatives). (Radio operator step.)
2. Label: `run_cuda_dino_offline_file.py --detector coherent_power` (teacher config, raised threshold)
   -> `mask_arrays/` as pseudo-GT; noise-only -> zero masks.
3. Build dataset (adapt `dino_fine_tuning/src/build_dataset_rt.py`): (real spectrogram, pseudo-mask)
   pairs + noise negatives, through the SAME robust front-end (`frontend.py`).
4. Fine-tune M3_491 (`train.py`) on the mixed set; export `.ts` (`export_dinov3_finetuned_torchscript.py`).
5. Validate: held-out real OTA (vs coherent) + synthetic held-out (no forgetting) + eyeball precision;
   re-run this offline harness to confirm the blobs resolve into per-signal boxes.

## Immediate state
- The **v4 demo (coherent detector) is clean and unaffected** — usable now.
- DINO-FT needs the above fine-tune to be demo-quality on real OTA; no config-only shortcut exists.

## Artifacts
`dino_ota_analysis.png` (detection-vs-energy + overlay), `dino_vs_coherent_ota.png` (DINO vs coherent
same frame), analysis scripts. Offline dumps: `/tmp/usrp_spectrograms/dino_ota_char`, `.../coh_ota`,
`.../dino_ota_thr08` (masks + input tensors; not committed — regenerate via the scripts).

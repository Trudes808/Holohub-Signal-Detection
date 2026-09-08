# Detector mask sanity check — comprehensive_ordered (pre-live validation)

**Date:** 2026-09-08 · **Branch/commit:** `live_demo` @ `5a31e650` · **Bench:** DGX Spark GB10, container `usrp_x410_sig_det_sat3737`

Purpose: before trusting the detectors live, run all of them offline on the same
capture — `comprehensive_ordered.sigmf-data` (245.76 MSps, 6.11 s, MATLAB composite
with full SigMF ground truth; 4 056 GT regions on the DINO frame grid / 4 595 on the
coherent grid) — and check the masks against ground truth. Specifically requested:
validate the **real-time DINO-FT path exactly as configured live**
(`real_time_downsample: true`, `downsample_fft_size: 0` auto, `flatten_noise_floor:
true`, per `config_live_v3_dino_ft.yaml`) and confirm its masks match the validated
native offline path.

## How it was run

```
python3 run_cuda_dino_offline_file.py <capture> --detector {coherent_power|cuda_dino|cuda_dino_finetuned} [--config <cfg>]
python3 eval_detector_masks.py --run-dir <out> --detector <tag> --captures-dir <composites> --out-dir <eval>
python3 detector_mask_sanity/render_mask_sanity.py     # agreement CSV + 6-panel figures
```

Six runs (outputs under `/tmp/usrp_spectrograms/offline_eval/<tag>/comprehensive_ordered`):

| tag | config | notes |
|---|---|---|
| `coherent_power` (guarded) | `config_coherent_power_perf_dynamic_single_channel.yaml` | live defaults incl. `max_emit_occupancy=0.35` |
| `coherent_power_noguard` | same + `max_emit_occupancy: 0.0` | emit gate off (dense-composite variant) |
| `cuda_dino` | `config_cuda_dino_performance_single_channel.yaml` | zero-shot DINOv3 coherence |
| `cuda_dino_finetuned` | `config_cuda_dino_finetuned_performance_single_channel.yaml` | M2_dr **native** path (dedicated 1024-pt FFT = trained 240 kHz/bin at this rate) |
| `cuda_dino_finetuned_rt` | native cfg + the three **live v3** knobs | wide 10240-pt FFT (auto @245.76M) → 10:1 bilinear freq resize → model; flatten on; `emit_stride` kept at 1 for full mask coverage (stride changes cadence, not mask content) |
| `cuda_dino_finetuned_rt_noflatten` | RT cfg + `flatten_noise_floor: false` | A/B to attribute the NBFM miss |

## Headline results (vs SigMF ground truth)

Region = one GT annotation box; detected = mask covers ≥50% of the box.
Pixel metrics are per-frame means on the GT grid. Full tables:
`detector_summary_metrics.csv`, `detector_perclass_detection.csv`.

| detector | regions detected | mean coverage | pixel precision | pixel recall | pixel IoU | FP area |
|---|---|---|---|---|---|---|
| coherent_power (live guard ON) | 56.5 % | 0.57 | 0.37 | 0.44 | 0.17 | 0.06 |
| coherent_power (guard off) | **93.0 %** | 0.93 | 0.41 | 0.93 | 0.39 | 0.26 |
| cuda_dino (zero-shot) | **98.5 %** | 0.96 | 0.33 | 0.80 | 0.26 | 0.27 |
| DINO-FT native (offline path) | 81.3 % | 0.76 | **0.95** | 0.93 | **0.89** | 0.003 |
| **DINO-FT RT (live path)** | **93.5 %** | 0.84 | 0.88 | 0.92 | 0.82 | 0.007 |
| DINO-FT RT, flatten off | 93.5 % | 0.84 | 0.88 | 0.93 | 0.83 | 0.007 |

**All three detectors produce sensible masks on this capture.** Characters are as
expected: coherent power is high-recall with moderate over-extension into signal
shoulders; zero-shot cuda_dino catches nearly everything but is blobby (27 % of
empty area lit, incl. patches in silent gaps — see `panels_frame280.png` top-right);
DINO-FT is by far the sharpest (FP area 0.3–0.7 %, pixel IoU 0.82–0.89).

## RT (live-path) DINO-FT vs native — do the masks match?

Per-frame mask agreement on a common 512×1024 grid over all 286 frames
(`dinoft_rt_vs_native_agreement.csv`): **mean IoU 0.80, median 0.87, p5 0.45**.
On dense frames the two paths are near-identical (frame 110: IoU 0.91,
`panels_frame110.png` bottom-right — thin blue fringes only). The low-IoU tail is
entirely **sparse µs-burst frames** (capture tail, ~1 % occupancy) where RT catches
the same real bursts but smears them to its coarser 41.7 µs time rows
(`panels_frame280.png`: RT-only red sits on real 802.11ax/ZC lines, not on empty
space). RT's "extra" pixels are real signals widened — not phantoms.

Two systematic, explainable differences (per-class table in
`detector_perclass_detection.csv`):

1. **Short bursts (ZC sync, 12 µs ≈ 3 native FFT rows): RT 96 % vs native 30 %.**
   Native's 16-row model patches under-segment sub-patch bursts; RT's coarse rows
   smear the burst across a whole 41.7 µs row and the model fires. RT is *better*
   here (region-level), at the cost of time-extent over-estimation.
2. **Always-on narrowband carriers (Narrowband_FM ~200 kHz): RT 16 % vs native 84 %.**
   The flatten A/B (last table row) proves `flatten_noise_floor` is NOT the cause
   (16 % with flatten off too). The mechanism is the **10:1 bilinear frequency
   resize**: a carrier occupying 1–2 of the 10 240 24-kHz bins falls between the
   ~2 bilinear taps per output pixel and is skipped/diluted. Native's dedicated
   1024-pt FFT integrates the full 240 kHz per bin, so the tone always lands.
   *Candidate fix (not applied): anti-aliased freq reduction before the model —
   max-pool (preserves tone peaks; slightly lifts the perceived floor) or area
   averaging (dilutes a 1-bin tone ~10×, likely insufficient) in
   `forward_downsampled`. Needs a fresh threshold check against M2_dr training
   stats before going live.*

Also confirmed: **`flatten_noise_floor: true` is neutral on flat-floor data**
(all metrics within noise of the flatten-off run) — it only removes the X410
band-edge rolloff live, so keeping it on in `config_live_v3_dino_ft.yaml` is safe.

## Live-guard interaction (coherent power)

The `max_emit_occupancy = 0.35` emit gate added for live garbage-flood containment
**blanks legitimate dense frames on this composite** (simultaneous wide signals
light 43–77 % of band-time): region detection drops 93 % → 56.5 %. That is the
guard working as designed on the wrong data. Keep it for live OTA; disable it
(`max_emit_occupancy: 0.0`) for offline dense-composite evals, as done here.
If a live demo ever needs a legitimately dense environment (e.g. replaying this
composite over cable), raise/disable the gate in the live config too.

## Panel figures

`panels_frame{020,056,060,110,160,210,260,280}.png` — 6 panels each: raw
spectrogram + GT boxes; coherent_power (noguard), cuda_dino, DINO-FT native,
DINO-FT RT masks; RT-vs-native disagreement map (yellow both / blue native-only /
red RT-only). Regenerate with `render_mask_sanity.py` (frame list at top).

## Verdict for the live demo

- **DINO-FT (RT live path) is working properly**: masks track ground truth with
  0.88 precision / 0.92 recall, match the validated native path where it matters,
  and its residual deltas are understood (µs-burst smearing — benign; sub-pixel
  narrowband carriers — known gap with a candidate fix).
- **cuda_dino works** as the zero-shot fallback: it will find essentially
  everything (98.5 %) but paints loose blobs and some empty-gap area — expect
  noisier snips/ROIs than DINO-FT.
- **coherent_power works** (93 % regions, guard off); live keeps the 0.35 emit
  gate because OTA traffic is sparse and the gate is what contains data-path
  garbage floods.

Artifacts: masks/GT/previews in `/tmp/usrp_spectrograms/offline_eval/…` (≈70 GB
incl. spectrogram tensors — safe to delete; regenerate via the commands above).
Eval fact tables in the job scratch were summarized into the two CSVs here.
